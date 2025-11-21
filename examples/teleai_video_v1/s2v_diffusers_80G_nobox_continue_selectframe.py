from teleai_pipe.pipelines.wan_video import WanVideoPipeline as VideoPipeline
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.utils.utils import load_state_dict
import open_clip

import torch
from diffusers.utils import export_to_video
from typing import List, Dict
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
import os
import glob
import pickle
from teleai_pipe.utils.vision import (
    GeneratePoseControlImages,
    GenerateBoxControlImages,
    GeneratePluckerControlImages,
    create_4grid_video,
)
import json
from tqdm import tqdm
import random
import imageio
import numpy as np
import torch.nn as nn
from torchvision.transforms import CenterCrop, InterpolationMode, Normalize, Resize
from loguru import logger


class AesNode:
    def __init__(
        self, path1: str, path2: str, batch_size: int = 6, device="cpu"
    ) -> None:
        """
        初始化 AesNode 类。

        :param path1: aes 模型的第一节路径
        :param path2: aes 模型的第二节路径
        :param batch_size: 批处理大小，默认为 6
        """
        self.path1: str = path1
        self.path2: str = path2
        self.batch_size: int = batch_size
        self.device = device
        self.init()

    def init(self) -> None:
        """
        初始化模型和预处理操作。
        """
        logger.info(f"init  aes model on device {self.device} from {self.path1}")
        self.clip_model, self.clip_preprocess, self.aesthetic_model = (
            self.get_aesthetic_model(self.path1, self.path2, device=self.device)
        )
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.norm = Normalize(self.mean, self.std)
        self.resize = Resize(
            size=224, interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        self.centercrop = CenterCrop((224, 224))

    def process(self, batch_imgs) -> dict:
        """
        处理输入数据，计算美学分数。

        :param data: 输入数据，包含 'video' 键
        :return: 处理后的数据，包含 'aes' 键
        """

        assert len(batch_imgs) == self.batch_size
        imgs = batch_imgs

        imgs = imgs / 255
        imgs = self.resize(imgs)
        imgs = self.centercrop(imgs)
        imgs = self.norm(imgs)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            aesthetic_score = self.aesthetic_model(image_features)
        return aesthetic_score

    @staticmethod
    def get_aesthetic_model(
        aes1_model_path: str,
        aes2_model_path: str,
        clip_model_name: str = "vit_l_14",
        device: str = "cpu",
    ) -> tuple[torch.nn.Module, any, torch.nn.Module]:
        """
        加载美学模型。

        :param aes1_model_path: 第一个模型的路径
        :param aes2_model_path: 第二个模型的路径
        :param clip_model_name: CLIP 模型的名称，默认为 'vit_l_14'
        :param device: 设备名称，默认为 'cpu'
        :return: 加载好的 CLIP 模型、CLIP 预处理操作和美学模型
        """
        if clip_model_name == "vit_l_14":
            aesthetic_model = nn.Linear(768, 1)
        elif clip_model_name == "vit_b_32":
            aesthetic_model = nn.Linear(512, 1)
        else:
            raise ValueError(f"Unsupported clip_model_name: {clip_model_name}")

        path_to_model = aes2_model_path
        s = torch.load(path_to_model)
        aesthetic_model.load_state_dict(s)
        aesthetic_model.eval()

        if clip_model_name == "vit_l_14":
            pretrained_path = aes1_model_path
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=pretrained_path, device=device
            )
        else:
            raise AssertionError(f"Unsupported clip_model_name: {clip_model_name}")

        aesthetic_model.to(device)
        return clip_model, clip_preprocess, aesthetic_model


def get_prefer_start_frame(video):
    # 为了保证效率，从后面24帧选择一帧
    idxs = [25, 29, 33, 37, 41, 44]
    selected_frames = torch.stack(
        [torch.from_numpy(np.array(video[idx])) for idx in idxs]
    ).permute(0, 3, 1, 2)
    selected_frames.to(aes_model.device)
    scores = aes_model.process(selected_frames)
    return idxs[scores.argmax().item()], video[idxs[scores.argmax().item()]]


def find_head_tail(pose_imgs):
    """
    找到含pose的最长连续序列
    """

    max_start, max_end = -1, -1
    current_start = -1
    max_length = 0

    for i, pose_img in enumerate(pose_imgs):
        if np.array(pose_img).sum() != 0:
            if current_start == -1:
                current_start = i
            if i - current_start + 1 > max_length:
                max_length = i - current_start + 1
                max_start, max_end = current_start, i
        else:
            current_start = -1

    return max_start, max_end


model_root = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.1-I2V-14B-720P"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float16,  # Image Encoder is loaded with float16
)

dit_path = [
    os.path.join(model_root, f)
    for f in os.listdir(model_root)
    if f.endswith(".safetensors")
]
model_manager.load_models(
    [
        dit_path,
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        f"{model_root}/Wan2.1_VAE.pth",
    ],
    # torch_dtype=torch.float8_e4m3fn,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
# lora_model_path = "/nvfile-heatstorage/yfq/code/vast/vast/work_dirs/wanvideo/wanvideo_i2v_pose_202504011_lora/models/checkpoint_epoch_2_step_6400/transformer/transformer_lora.pt"
# lora_model_path = "/nvfile-heatstorage/yfq/code/vast/vast/work_dirs/wanvideo/wanvideo_i2v_pose_202504015_lora/models/checkpoint_epoch_2_step_2900/transformer/transformer_lora.pt"
lora_model_path = "/nvfile-heatstorage/yfq/code/vast/vast/work_dirs/wanvideo/wanvideo_i2v_poseplucker_20250508_lora/models/checkpoint_epoch_5_step_3700/transformer/transformer_lora.pt"

lora_state_dict = load_state_dict(lora_model_path)
model_manager.load_lora("", lora_state_dict, lora_alpha=1.0)

pipe = VideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)
pipe.dit.patch_embedding = nn.Conv3d(
    in_channels=52,
    out_channels=5120,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2),
    dtype=torch.bfloat16,
)
pipe.dit.load_state_dict(lora_state_dict, strict=False)
pipe.enable_compile_dit()
pipe.enable_cpu_offload()
aes_model = AesNode(
    path1="/nvfile-heatstorage/Text2Video/new_ppl_models/ViT-L-14.pt",
    path2="/nvfile-heatstorage/Text2Video/new_ppl_models/sa_0_4_vit_l_14_linear.pth",
    device="cpu",
)

width = 1280
height = 720
num_frames = 45
continue_num_frames = num_frames * 4  # 4段
num_inference_steps = 50
output_dir = "work_dirs/output_guangdian2_continue/"
os.makedirs(output_dir, exist_ok=True)


def inference_from_json_dataset(idxs, data):
    for idx in tqdm(idxs):
        clip = data["clips"][idx]
        if (
            clip["filter_state"]["optical_flow"] < 2
            or clip["filter_state"]["aesthetic"] < 4.5
        ):
            continue
        if not os.path.exists(clip["raw_video_path"]):
            continue

        prompt = clip["caption"]["dense_caption"][0]
        print(f"prompt: {prompt}")
        data_dict = {}
        try:
            with open(clip["mesh"], "rb") as file:
                mesh = pickle.load(file)
                data_dict["joints2d_nonparam"] = mesh["joints2d_nonparam"]
                data_dict["joint_uncertainties"] = mesh["joint_uncertainties"]

        except Exception as e:
            print(f"Failed to load mesh: {e}")

        try:
            with open(clip["bbox"], "r") as file:
                content = file.read()
                bbox = json.loads(content)
                data_dict["bbox"] = bbox
        except Exception as e:
            print(f"Failed to load bbox: {e}")

        try:
            data_dict["camera_poses"] = np.load(clip["camera_poses"])
        except Exception as e:
            print(f"Failed to load camera_poses: {e}")

        data_dict["video_height"] = clip["height"]
        data_dict["video_width"] = clip["width"]
        data_dict["sample_indexes"] = [
            32 + i * 2
            for i in range(continue_num_frames)
            if 32 + i * 2 < len(data_dict["joints2d_nonparam"])
        ]
        # data_dict["sample_indexes"] = list(range(num_frames))

        pose_generator = GeneratePoseControlImages()
        box_generator = GenerateBoxControlImages()
        plucker_generator = GeneratePluckerControlImages()
        pose_images = pose_generator(data_dict)
        box_images = box_generator(data_dict)
        plucker_ray_images, plucker_mom_images = plucker_generator(data_dict)

        if np.array(pose_images[0]).sum() == 0:
            continue
        reader = imageio.get_reader(clip["raw_video_path"])
        ref_frames = [reader.get_data(i) for i in data_dict["sample_indexes"]]
        ref_images = [Image.fromarray(frame) for frame in ref_frames]
        pose_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in pose_images
        ]
        box_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in box_images
        ]
        plucker_ray_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in plucker_ray_images
        ]
        plucker_mom_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in plucker_mom_images
        ]
        ref_images = [
            F.resize(img, (height, width), InterpolationMode.BILINEAR)
            for img in ref_images
        ]
        input_images = [
            Image.new("RGB", (width, height), (0, 0, 0)) for _ in ref_images
        ]

        # 接口,用于检查pose和box图像是否有效
        if not pose_generator.if_valid:
            print("no valid pose images")
        if not box_generator.if_valid:
            print("no valid box images")
        if not plucker_generator.if_valid:
            print("no valid plucker images")

        # modify
        # pose_images = filter_tail_poses(pose_images)
        start_idx, end_idx = find_head_tail(pose_images)  # 寻找连续不为空的序列

        final_videos = []
        while start_idx < end_idx and start_idx < continue_num_frames:
            if len(final_videos) == 0:
                input_image = ref_images[start_idx]
            else:
                video_idx, input_image = get_prefer_start_frame(video)
                start_idx = start_idx + video_idx
                final_videos = final_videos[: video_idx - num_frames]
            if len(pose_images[start_idx : start_idx + num_frames]) < num_frames:
                break

            input_images[start_idx] = input_image

            video = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                cfg_scale=5.0,
                seed=0,
                tiled=True,
                tile_size=(34, 34),
                tile_stride=(18, 16),
                input_image=input_image,
                cn_images=(
                    pose_images[start_idx : start_idx + num_frames],
                    plucker_ray_images[start_idx : start_idx + num_frames],
                    plucker_mom_images[start_idx : start_idx + num_frames],
                ),
                model_type="Wan2.1-I2V-14B-720P",
                tea_cache_l1_thresh=0.15,
                attn_impl="cudnn",
            )
            final_videos.extend(video)
            # break
            # import pdb; pdb.set_trace()

        export_to_video(final_videos, output_dir + f"{idx}_cfg.mp4", fps=15)
        # 将pose,box,ref和生成的视频合并为一个视频进行展示
        combined_frames = create_4grid_video(
            [pose_images, box_images, input_images, final_videos]
        )
        imageio.mimsave(
            output_dir + f"{idx}_combined.mp4",
            [np.array(frame) for frame in combined_frames],
            fps=16,
        )


# json_path = "/nvfile-heatstorage/Text2Video/jyt/filter_json/guangdian/guangdian_4_B007成品_1080P.json"
json_path = "/nvfile-heatstorage/Text2Video/XJY/selected_videos_2.json"
with open(json_path, "r") as file:
    data = json.load(file)

inference_from_json_dataset(range(0, len(data["clips"])), data)
