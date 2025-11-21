import torch
from tqdm import tqdm
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.models.flashvsr_dit import FlashVSRModel
from teleai_pipe.models.wan_video_vae import WanVideoVAE
from teleai_pipe.models.TCDecoder import TAEHV
from teleai_pipe.schedulers.flow_match import FlowMatchScheduler
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from teleai_pipe.models.color_corrector import TorchColorCorrectorWavelet
from PIL import Image
import numpy as np
from loguru import logger
from teleai_pipe.utils.profiler import ProfilingContext4Debug


def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device="cuda"):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(
        device=device, dtype=torch.float32
    )  # HWC
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
    return t.to(dtype)


def compute_scaled_and_target_dims(
    w0: int, h0: int, scale: float = 4.0, multiple: int = 128
):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop_tensor(
    img_tensor: torch.Tensor, original_size: tuple, scale: float, tW: int, tH: int
) -> torch.Tensor:
    """
    使用PyTorch插值进行上采样和中心裁剪

    Args:
        img_tensor: 形状为(C, H, W)的tensor，值范围[-1, 1]
        original_size: 原始图像尺寸(w, h)
        scale: 缩放比例
        tW: 目标宽度
        tH: 目标高度

    Returns:
        处理后的tensor，形状为(C, tH, tW)
    """
    w0, h0 = original_size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    # 上采样到缩放尺寸
    # 先将tensor从[-1, 1]转换到[0, 1]以便插值
    img_01 = (img_tensor + 1) / 2.0
    # 使用bicubic插值上采样
    upsampled = torch.nn.functional.interpolate(
        img_01.unsqueeze(0), size=(sH, sW), mode="bicubic", align_corners=False
    ).squeeze(0)

    # 中心裁剪
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    cropped = upsampled[:, t : t + tH, l : l + tW]

    # 转换回[-1, 1]范围
    return cropped * 2.0 - 1.0


def prepare_input_tensor(video, scale: float = 4, dtype=torch.bfloat16, device="cuda"):
    first = video[0]
    w0, h0 = first.size
    total = len(video)
    if total <= 0:
        raise RuntimeError("input video is empty")

    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    print(f"[Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")
    pad_total = ((((((total - 21) + 4) // 8) * 8 + 21) + 4) // 8) * 8 + 1
    print(
        f"[Original Resolution: {w0}x{h0} | Original Frames: {total} to pad frames {pad_total}"
    )
    idx = list(range(total)) + [total - 1] * (pad_total - total)
    print(f"id len {len(idx)}")
    F = largest_8n1_leq(len(idx))
    if F == 0:
        raise RuntimeError(f"Got {len(idx)}.")
    idx = idx[:F]
    print(f"[Target Frames (8n-3): {F-4}")

    frames = []
    with ProfilingContext4Debug("upscale then crop"):
        try:
            for i in idx:
                img = video[i].convert("RGB")
                # 先将PIL图像转换为tensor
                img_tensor = pil_to_tensor_neg1_1(
                    img, dtype=torch.float32, device="cpu"
                )
                # 使用PyTorch插值版本进行处理
                processed_tensor = upscale_then_center_crop_tensor(
                    img_tensor, (w0, h0), scale=scale, tW=tW, tH=tH
                )
                # 转换到目标dtype和设备
                frames.append(processed_tensor.to(dtype=dtype, device=device))
        finally:
            try:
                video.close()
            except Exception:
                pass

    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    print(f"vid shape, {vid.shape}")
    return vid, tH, tW, F


class FlashVSRPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: FlashVSRModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.prompt_emb_posi = None
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

    def fetch_models(self, model_manager: ModelManager):
        self.dit = model_manager.fetch_model("flashvsr_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = FlashVSRPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        return self.dit

    def prepare_extra_input(self, latents=None):
        return {}

    def decode_video(
        self,
        latents,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        cond_video=None,
    ):
        if cond_video is not None:
            frames = (
                self.vae.decode(
                    latents.transpose(1, 2),
                    parallel=False,
                    show_progress_bar=False,
                    cond=cond_video,
                )
                .transpose(1, 2)
                .mul_(2)
                .sub_(1)
            )
        else:
            frames = self.vae.decode(
                latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return frames

    @torch.no_grad()
    def __call__(
        self,
        seed=None,
        tiled=True,
        tile_size=(64, 128),
        tile_stride=(32, 64),
        LQ_video=None,
        is_full_block=False,
        if_buffer=False,
        sparse_ratio=2.0,
        kv_ratio=3.0,
        local_range=9,
        color_fix=True,
        scale=4,
        time_tile_decode: bool = False,
    ):
        logger.info(f"sr input video size is {len(LQ_video)}x{LQ_video[0].size}")
        with ProfilingContext4Debug("prepare input"):
            real_num_frames = len(LQ_video)
            LQ_video, height, width, F = prepare_input_tensor(
                LQ_video, scale=scale, dtype=self.torch_dtype, device="cpu"
            )
            num_frames = F
            topk_ratio = sparse_ratio * 768 * 1280 / (height * width)
            self.load_models_to_device(["dit", "vae"])
            # 尺寸修正
            height, width = self.check_resize_height_width(height, width)
            if num_frames % 4 != 1:
                num_frames = (num_frames + 2) // 4 * 4 + 1
                print(
                    f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
                )

            logger.info(f"start generating {num_frames}x{width}x{height} sr video")
            # Tiler 参数
            tiler_kwargs = {
                "tiled": tiled,
                "tile_size": tile_size,
                "tile_stride": tile_stride,
            }

            # 初始化噪声
            if if_buffer:
                noise = self.generate_noise(
                    (1, 16, (num_frames - 1) // 4, height // 8, width // 8),
                    seed=seed,
                    dtype=self.torch_dtype,
                )
            else:
                noise = self.generate_noise(
                    (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                    seed=seed,
                    dtype=self.torch_dtype,
                )
            latents = noise
            process_total_num = (num_frames - 1) // 8 - 2  # 8
            is_stream = True

            # 清理可能存在的 LQ_proj_in cache
            if hasattr(self.dit, "LQ_proj_in"):
                self.dit.LQ_proj_in.clear_cache()

            latents_total = []
            frames_total = []
            if hasattr(self.vae, "clean_mem"):
                self.vae.clean_mem()
            LQ_pre_idx = 0
            LQ_cur_idx = 0
        with torch.no_grad():
            for cur_process_idx in tqdm(range(process_total_num)):
                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.dit.blocks)
                    pre_cache_v = [None] * len(self.dit.blocks)
                    LQ_latents = None
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        cur = (
                            self.denoising_model().LQ_proj_in.stream_forward(
                                LQ_video[
                                    :,
                                    :,
                                    max(0, inner_idx * 4 - 3) : (inner_idx + 1) * 4 - 3,
                                    :,
                                    :,
                                ].to(self.device)
                            )
                            if LQ_video is not None
                            else None
                        )
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat(
                                    [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                )
                    LQ_cur_idx = (inner_loop_num - 1) * 4 - 3
                    cur_latents = latents[:, :, :6, :, :].to(self.device)
                else:
                    LQ_latents = None
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        video_clip = LQ_video[
                            :,
                            :,
                            cur_process_idx * 8 + 17 + inner_idx * 4 : cur_process_idx
                            * 8
                            + 21
                            + inner_idx * 4,
                            :,
                            :,
                        ].to(self.device)
                        cur = (
                            self.denoising_model().LQ_proj_in.stream_forward(video_clip)
                            if LQ_video is not None
                            else None
                        )
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat(
                                    [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                )
                    LQ_cur_idx = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4
                    cur_latents = latents[
                        :, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2, :, :
                    ].to(self.device)

                # 推理（无 motion_controller / vace）
                noise_pred_posi, pre_cache_k, pre_cache_v = self.dit(
                    x=cur_latents,
                    LQ_latents=LQ_latents,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    local_range=local_range,
                )

                # 更新 latent
                cur_latents = cur_latents - noise_pred_posi
                if time_tile_decode:
                    cur_LQ_frames = LQ_video[:, :, LQ_pre_idx:LQ_cur_idx].to(
                        self.device
                    )
                    cur_frames = self.decode_video(
                        cur_latents, cond_video=cur_LQ_frames, **tiler_kwargs
                    )
                    if color_fix:
                        cur_frames = self.ColorCorrector(
                            cur_frames.to(device=self.device),
                            cur_LQ_frames,
                            clip_range=(-1, 1),
                            chunk_size=None,
                            method="adain",
                        )
                    frames_total.append(cur_frames.cpu())
                else:
                    latents_total.append(cur_latents.cpu())
                LQ_pre_idx = LQ_cur_idx

            with ProfilingContext4Debug("post process"):
                if latents_total:
                    latents = torch.cat(latents_total, dim=2)
                    if isinstance(self.vae, TAEHV):
                        cond_video = LQ_video[:, :, :LQ_cur_idx, :, :].to(self.device)
                        frames = self.decode_video(
                            latents=latents.to(self.device),
                            cond_video=cond_video,
                            **tiler_kwargs,
                        )
                    else:
                        frames = self.decode_video(
                            latents=latents.to(self.device), **tiler_kwargs
                        )
                    # 颜色校正（wavelet）
                    try:
                        if color_fix:
                            frames = self.ColorCorrector(
                                frames.to(device=LQ_video.device),
                                LQ_video[:, :, : frames.shape[2], :, :],
                                clip_range=(-1, 1),
                                chunk_size=16,
                                method="adain",
                            )
                    except Exception:
                        pass
                else:
                    frames = torch.cat(frames_total, dim=2)
        self.load_models_to_device([])
        video = self.tensor2video(frames[0])[:real_num_frames]
        return video
