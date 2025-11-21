import os
import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.flashvsr import FlashVSRPipeline
from teleai_pipe.utils.video import save_video, VideoData

PPL_CONFIG = dict(
    name="flashvsr_1.3b",
    dit_path="/nvfile-heatstorage/model_zoo/huggingface/FlashVSR/flashvsr_dit_streaming_dmd.safetensors",
    vae_path="/nvfile-heatstorage/model_zoo/huggingface/FlashVSR/TCDecoder.ckpt",
    # vae_path="/nvfile-heatstorage/model_zoo/huggingface/FlashVSR/Wan2.1_VAE.pth",
    sparse_ratio=2,
    seed=0,
    tiled=True,
)


def get_pipeline():
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([PPL_CONFIG["dit_path"]])
    mm.load_models([PPL_CONFIG["vae_path"]])
    pipe = FlashVSRPipeline.from_model_manager(mm, device="cuda")
    pipe.enable_cpu_offload()
    return pipe


def run(pipeline, input_video, scale, seed=PPL_CONFIG["seed"]):
    video = pipeline(
        seed=seed,
        LQ_video=input_video,
        is_full_block=False,
        if_buffer=False,
        sparse_ratio=PPL_CONFIG["sparse_ratio"],
        kv_ratio=3.0,
        local_range=11,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
        color_fix=True,
        scale=scale,
        tiled=PPL_CONFIG["tiled"],
        time_tile_decode=True if len(input_video) > 81 else False,
    )
    return video


def main():
    import time

    pipeline = get_pipeline()
    ROOT = "/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/work_dirs/FlashVSR/examples/WanVSR/inputs"
    inputs = [
        # f"{ROOT}/example0.mp4",
        # f"{ROOT}/example1.mp4",
        # f"{ROOT}/example2.mp4",
        # f"{ROOT}/example3.mp4",
        # "/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/teleai_hr_video_v1_2_wan22_14b_text_to_hr_video_prompt_enhance_distill_lora_h100_wan22_8step.mp4"
        # "/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/work_dirs/benchmark_result/t2v/wan_14b_text_to_video_official/std100_250630_zh_extend/2025-08-05_10-50-58/000010.mp4"
        # "/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/examples/data/dag.mp4"
        "/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/teleai_long_video_v1_teleai_image_to_video_v1_h100.mp4"
    ]
    for p in inputs:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        name = os.path.basename(p.rstrip("/"))
        input_video = VideoData(p)
        video = run(pipeline, input_video, scale=4)
        s = time.time()
        video = run(pipeline, input_video, scale=4)
        print(f"cost {time.time() - s } s")
        save_video(
            video,
            f"FlashVSR_{name.split('.')[0]}.mp4",
            fps=input_video.fps(),
            quality=6,
        )
    print("Done.")


if __name__ == "__main__":
    main()
