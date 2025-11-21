import os
import cv2
import math
import torch
import pickle
import json
import imageio
import numpy as np
import sys
import argparse


from PIL import Image
from decord import VideoReader
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import AutoTokenizer
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from teleai_pipe.pipelines.teleai_i2v_omini_drive import WanFunInpaintMoPipeline
from teleai_pipe.models.teleai_i2v_omini_drive_init import (
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
    SMPL_VQVAE,
    VectorQuantizer,
    Encoder,
    Decoder,
    get_1d_rotary_pos_embed,
    get_3d_motion_spatial_embed,
    WanTransformer3DModelMoCn,
)
from teleai_pipe.pipelines.teleai_i2v_omini_drive.wan_lora import WanLoraWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Omini image to video: drive anything from motion and text cues."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/data/online_selected_0827_hz.pkl",
        help="The input pkl file that contains motion list.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory for output"
    )
    parser.add_argument(
        "--ref_image_path", type=str, default="", help="Reference image."
    )
    parser.add_argument(
        "--infer_num", type=int, default=3, help="Inference batch number."
    )
    parser.add_argument("--stride", type=int, default=1, help="Inference frame stride.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        default="",
        help="prompt text.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="720p",
        choices=["480p", "720p"],
        help="The buckget size of the generated video. The aspect ratio of the output video will follow that of the input image.",
    )
    args = parser.parse_args()
    return args


args = parse_args()


def filter_kwargs(cls, kwargs):
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def concat_and_save(video_list, save_path, rescale_params, method=1, fps=25):
    processed_videos = []
    for video, rescale in zip(video_list, rescale_params):
        if video is None:
            continue
        video = video.squeeze(0)  # (T, C, H, W)
        if rescale:
            video = (video + 1.0) / 2.0  # [-1,1] -> [0,1]
        video = video.clamp(0, 1)
        processed_videos.append(video)

    if method == 1:
        cat_video = torch.cat(processed_videos, dim=2)
    elif method == 2:
        cat_video = torch.cat(processed_videos, dim=3)
    cat_video = (cat_video * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    writer = imageio.get_writer(save_path, fps=fps)
    for frame in cat_video:
        frame = cv2.resize(
            frame,
            (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        writer.append_data(frame)
    writer.close()


def draw_smpl_body(
    canvas, points, stickwidth=2, r=2, box=None, scores=None, score_thre=None
):
    colors = [
        [255, 0, 0],  # 0
        [0, 255, 0],  # 1
        [0, 0, 255],  # 2
        [255, 0, 255],  # 3
        [255, 255, 0],  # 4
        [85, 255, 0],  # 5
        [0, 75, 255],  # 6
        [0, 255, 85],  # 7
        [0, 255, 170],  # 8
        [170, 0, 255],  # 9
        [85, 0, 255],  # 10
        [0, 85, 255],  # 11
        [0, 255, 255],  # 12
        [85, 0, 255],  # 13
        [170, 0, 255],  # 14
        [255, 0, 255],  # 15
        [255, 0, 170],  # 16
        [255, 0, 85],  # 17
    ]
    connetions = [
        [15, 12],
        [12, 16],
        [16, 18],
        [18, 20],
        [20, 22],
        [12, 17],
        [17, 19],
        [19, 21],
        [21, 23],
        [12, 9],
        [9, 6],
        [6, 3],
        [3, 0],
        [0, 1],
        [1, 4],
        [4, 7],
        [7, 10],
        [0, 2],
        [2, 5],
        [5, 8],
        [8, 11],
    ]
    connection_colors = [
        [255, 0, 0],  # 0
        [0, 255, 0],  # 1
        [0, 0, 255],  # 2
        [255, 255, 0],  # 3
        [255, 0, 255],  # 4
        [0, 255, 0],  # 5
        [0, 85, 255],  # 6
        [255, 175, 0],  # 7
        [0, 0, 255],  ## 8
        [255, 85, 0],  # 9
        [0, 255, 85],  # 10
        [255, 0, 255],  # 11
        [255, 0, 0],  # 12
        [0, 175, 255],  # 13
        [255, 255, 0],  # 14
        [0, 0, 255],  # 15
        [0, 255, 0],  # 16
    ]

    # draw point
    for i in range(len(points)):
        x, y = points[i][0:2]
        x, y = int(x), int(y)
        if i == 13 or i == 14:
            continue
        overlay = canvas.copy()
        if i == 23:
            cv2.circle(overlay, (x, y), 5, colors[i % 17], thickness=-1)
        else:
            cv2.circle(overlay, (x, y), r, colors[i % 17], thickness=-1)

        canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)

    # draw line
    for i in range(len(connetions)):
        point1_idx, point2_idx = connetions[i][0:2]
        point1 = points[point1_idx]
        point2 = points[point2_idx]
        Y = [point2[0], point1[0]]
        X = [point2[1], point1[1]]
        mX = int(np.mean(X))
        mY = int(np.mean(Y))
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (mY, mX), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, polygon, connection_colors[i % 17])
        canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)

    return canvas


def intrinsic_matrix_from_field_of_view(imshape, fov_degrees: float = 25):  # 35
    imshape = np.array(imshape)
    fov_radians = fov_degrees * np.array(np.pi / 180)
    larger_side = np.max(imshape)
    focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
    # intrinsic_matrix 3*3
    return np.array(
        [
            [focal_length, 0, imshape[1] / 2],
            [0, focal_length, imshape[0] / 2],
            [0, 0, 1],
        ]
    )


def p3d_to_p2d(point_3d, height, width):  # point3d n*1024*3
    camera_matrix = intrinsic_matrix_from_field_of_view((height, width))
    camera_matrix = np.expand_dims(camera_matrix, axis=0)
    camera_matrix = np.expand_dims(camera_matrix, axis=0)  # 1*1*3*3
    point_3d = np.expand_dims(point_3d, axis=-1)  # n*1024*3*1
    point_2d = (camera_matrix @ point_3d).squeeze(-1)
    point_2d[:, :, :2] = point_2d[:, :, :2] / point_2d[:, :, 2:3]
    return point_2d[:, :, :]  # n*1024*2


def get_control_conditions(poses, h, w, use_2d):
    if use_2d:
        control_images = []
        for idx, pose in enumerate(poses):
            canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
            try:
                joints3d = p3d_to_p2d(pose, h, w)
                canvas = draw_smpl_body(
                    canvas,
                    joints3d[0],
                    stickwidth=int(h / 350),
                )
                resized_canvas = cv2.resize(canvas, (w, h))
                # Image.fromarray(resized_canvas).save(f'tmp/{idx}_pose.jpg')
                control_images.append(resized_canvas)
            except Exception as e:
                print("wrong:", e)
                control_images.append(Image.fromarray(canvas))
        control_pixel_values = np.array(control_images)
        control_pixel_values = (
            torch.from_numpy(control_pixel_values)
            .permute(0, 3, 1, 2)
            .contiguous()
            .unsqueeze(0)
            / 255.0
        )
        control_pixel_values = video_transforms(control_pixel_values)
    else:
        control_pixel_values = None

    return control_pixel_values


def prepare_motion_embeddings(
    num_frames, num_joints, joints_mean, joints_std, theta=10000, device="cuda"
):
    time_embed = get_1d_rotary_pos_embed(44, num_frames, theta, use_real=True)
    time_embed_cos = (
        time_embed[0][:, None, :]
        .expand(-1, num_joints, -1)
        .reshape(num_frames * num_joints, -1)
    )
    time_embed_sin = (
        time_embed[1][:, None, :]
        .expand(-1, num_joints, -1)
        .reshape(num_frames * num_joints, -1)
    )
    spatial_motion_embed = get_3d_motion_spatial_embed(
        [28, 28, 28], num_joints, joints_mean, joints_std, theta
    )
    spatial_embed_cos = (
        spatial_motion_embed[0][None, :, :]
        .expand(num_frames, -1, -1)
        .reshape(num_frames * num_joints, -1)
    )
    spatial_embed_sin = (
        spatial_motion_embed[1][None, :, :]
        .expand(num_frames, -1, -1)
        .reshape(num_frames * num_joints, -1)
    )
    motion_embed_cos = torch.cat([time_embed_cos, spatial_embed_cos], dim=-1).to(
        device=device
    )
    motion_embed_sin = torch.cat([time_embed_sin, spatial_embed_sin], dim=-1).to(
        device=device
    )
    return motion_embed_cos, motion_embed_sin


def get_nearest_resize_shape(h, w):
    aspect_ratio = h / w
    ASPECT_RATIO = ASPECT_RATIO_480p
    ratios = np.array([float(k) for k in ASPECT_RATIO.keys()])
    closest_key = str(ratios[np.argmin(np.abs(ratios - aspect_ratio))])
    return ASPECT_RATIO[closest_key]


ASPECT_RATIO_512 = {
    "0.25": [256.0, 1024.0],
    "0.26": [256.0, 992.0],
    "0.27": [256.0, 960.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "2.89": [832.0, 288.0],
    "3.0": [864.0, 288.0],
    "3.11": [896.0, 288.0],
    "3.62": [928.0, 256.0],
    "3.75": [960.0, 256.0],
    "3.88": [992.0, 256.0],
    "4.0": [1024.0, 256.0],
}

scale_factor = 1.25
ASPECT_RATIO_480p = {
    ratio: [width * scale_factor // 16 * 16, height * scale_factor // 16 * 16]
    for ratio, (width, height) in ASPECT_RATIO_512.items()
}
scale_factor = 1.875
ASPECT_RATIO_720p = {
    ratio: [width * scale_factor // 16 * 16, height * scale_factor // 16 * 16]
    for ratio, (width, height) in ASPECT_RATIO_512.items()
}
# 支持 480p/ 720p
if args.size == "480p":
    scale_factor = 1.0
else:
    scale_factor = 1.875 / 1.25


device = "cuda"
weight_dtype = torch.bfloat16

data_path = args.data_path
pretrained_model_name_or_path = (
    "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/Wan2.1-Fun-V1.1-14B-InP"
)

vqvae_model_path = "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/vqvae_model/model.safetensors"  # for 43k

output_dir = args.output_dir

transformer_model_path = (
    "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/43K-model/transformer"
)

lora_path = "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/lora_model/Wan2.1_T2V_14B_FusionX_LoRA.safetensors"
# lora_path = ""
config_path = (
    "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/config/wan2.1/wan_civitai.yaml"
)
ref_image_path = ""
ref_image_path = args.ref_image_path


ref_img_list = [
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/taiji.jpg",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/person3.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/owl.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/数字人女.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/机器人图片/机器人图片/seq_1753178434210_c57b9d0b_1753178438_output.jpg",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/小男孩图片/小男孩图片/seq_1753234580864_80d067b4_1753234599_output.jpg",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/卡通形象/pikachu.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/卡通形象/猫3.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/tiger.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/dog.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/haimianbaobao.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/goku.png",
    "/nvfile-heatstorage/AIGC_H100/human_guozz2/code/dyb/wan_0826/ref_images/daizong.png",
]

video_transforms = transforms.Normalize(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
)
os.makedirs(output_dir, exist_ok=True)
video_length = 81
seed = 42  # 42
infer_num = args.infer_num

overlap = 9
guidance_scale = 6.0
fps = 16
stride = args.stride
# stride = 6
use_first_clip = False

# Load config
config = OmegaConf.load(config_path)

# Get transformer
transformer3d = WanTransformer3DModelMoCn.from_pretrained(
    transformer_model_path, use_2d=False
).to(device, weight_dtype)

# Get text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(
        pretrained_model_name_or_path,
        config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
    ),
    additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
    torch_dtype=weight_dtype,
).to(device, weight_dtype)

# Get clip image encoder
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(
        pretrained_model_name_or_path,
        config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
    ),
)

# Get vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(
        pretrained_model_name_or_path, config["vae_kwargs"].get("vae_subpath", "vae")
    ),
    additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
).to(device, weight_dtype)

# Get scheduler
scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(
        FlowMatchEulerDiscreteScheduler,
        OmegaConf.to_container(config["scheduler_kwargs"]),
    )
)

# Get tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(
        pretrained_model_name_or_path,
        config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
    ),
)

# Get motion tokenizer
# encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=5120, downsample_time=[2, 2], downsample_joint=[1, 1])
# vq = VectorQuantizer(nb_code=8192, code_dim=5120, is_train=False)
# decoder = Decoder(in_channels=5120, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
# vqvae = SMPL_VQVAE(encoder, decoder, vq)
# print(vqvae.load_state_dict(load_file(vqvae_model_path), strict=True))
encoder = Encoder(
    in_channels=3,
    mid_channels=[128, 512],
    out_channels=3072,
    downsample_time=[2, 2],
    downsample_joint=[1, 1],
)
vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
decoder = Decoder(
    in_channels=3072,
    mid_channels=[512, 128],
    out_channels=3,
    upsample_rate=2.0,
    frame_upsample_rate=[2.0, 2.0],
    joint_upsample_rate=[1.0, 1.0],
)
vqvae = SMPL_VQVAE(encoder, decoder, vq)
# print(vqvae.load_state_dict(torch.load(vqvae_model_path, map_location='cpu')['module'], strict=True))
print(vqvae.load_state_dict(load_file(vqvae_model_path), strict=True))


pipeline = WanFunInpaintMoPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    transformer=transformer3d,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
)
pipeline = pipeline.to(device)
vqvae = vqvae.to(pipeline.device, weight_dtype)
if lora_path != "":
    lora_wrapper = WanLoraWrapper(pipeline.transformer)
    for lora_path, lora_scale in zip([lora_path], [1]):
        lora_name = lora_wrapper.load_lora(lora_path)
        lora_wrapper.apply_lora(lora_name, lora_scale)

if data_path.split(".")[-1] == "json":
    with open(data_path, "r") as f:
        data = json.load(f)
else:
    with open(data_path, "rb") as f:
        data = pickle.load(f)

# global_mean = np.load('/gemini/platform/public/aigc/human_guozz2/global_mean.npy')
# global_std = np.load('/gemini/platform/public/aigc/human_guozz2/global_std.npy')
global_mean = np.load(
    "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/statistics/global_mean_724.npy"
)  # for 43k
global_std = np.load(
    "/nvfile-heatstorage/human_guozz2/code/dyb/wan_0826/statistics/global_std_724.npy"
)
# global_mean = np.load("/gemini/platform/public/aigc/human_guozz2/code/dyb/wan/statistics/global_mean_804.npy")
# global_std = np.load("/gemini/platform/public/aigc/human_guozz2/code/dyb/wan/statistics/global_std_804.npy")

selected_indexes = [0, 4, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 29, 30]
# selected_indexes = list(range(0, len(data)))
selected_indexes = [0, 5, 6, 7, 8, 9, 91, 94, 105, 106, 208, 211, 212, 218, 219] + list(
    range(220, len(data))
)
selected_indexes = list(range(51, len(data)))
# selected_indexes = [7]
# selected_indexes = list(range(0, len(data)))
# selected_indexes = list(range(4, len(data)))
selected_indexes = list(range(224, len(data), 200))
selected_indexes = list(range(0, len(data)))
# selected_indexes = [68, 98]


generator = torch.Generator(device=device).manual_seed(seed)
for index, data_dict in enumerate(data):
    # if index == 0 and ref_image_path != "":
    #     output_dir = f"{output_dir}/{ref_image_path.split('/')[-1].replace('.png', '')}"
    #     os.makedirs(output_dir, exist_ok=True)

    # temp for infer long dance 07:
    # if index==1:
    #     infer_num=3
    data_dict["pose"] = np.load(data_dict["16fps_npy"])

    # ref_image_path = np.random.choice(ref_img_list)

    if index not in selected_indexes:
        continue

    expect_length = (video_length - overlap) * (infer_num - 1) + video_length

    if data_dict["video_length"] < expect_length:
        _pose = data_dict["pose"]
        _n = _pose.shape[0]
        _last_p = _pose[-1:]
        _repeat_t = expect_length - _n
        _padding = np.repeat(_last_p, _repeat_t, axis=0)
        real_pose = np.concatenate([_pose, _padding], axis=0)
    else:
        real_pose = data_dict["pose"]

    # read video
    input_video_path = data_dict["video_path"]
    # input_video_path = "/gemini/platform/public/aigc/human_guozz2/code/dyb/comotion/data/taiji.mp4"
    try:
        vr = VideoReader(input_video_path)
    except Exception as e:
        print("Wrong: ", e)
        root_dir = "/nvfile-heatstorage/AIGC_H100/lbc/fps_convert_videos/motion_3.1_videos_16fps/"
        input_video_path = os.path.basename(input_video_path).replace(".pkl", ".mp4")
        input_video_path = os.path.join(root_dir, input_video_path)
        # import pdb

        # pdb.set_trace()
        vr = VideoReader(input_video_path)
        # continue

    print(f"video path: {input_video_path}")
    # stride = vr.get_avg_fps() // fps
    fps = vr.get_avg_fps()

    # save_path
    if ref_image_path == "":
        save_path = os.path.join(
            output_dir,
            f"{index}_{guidance_scale}_{seed}_{os.path.basename(input_video_path)}",
        )
    else:
        ref_name = os.path.splitext(os.path.basename(ref_image_path))[0]
        save_path = os.path.join(
            output_dir,
            f"{index}_{guidance_scale}_{seed}_{ref_name}_{os.path.basename(input_video_path)}",
        )

    with torch.autocast("cuda", dtype=weight_dtype):
        try:
            start_idx = data_dict["pose_start_index"]
        except Exception as e:
            print("wrong:", e)
            start_idx = data_dict["video_start_index"]

        # start_idx = 1200
        if ref_image_path != "":
            prompt = args.prompt

        else:
            try:
                prompt = data_dict["short_prompt"]
            except Exception as e:
                print("wrong:", e)
                prompt = data_dict["long_prompt"]

        # length = data_dict['video_length']
        # length = (length - 1) // 8 * 8 + 1
        # indices = np.linspace(data_dict['video_start_index'], data_dict['video_start_index'] + length - 1, length, dtype=int)s
        previous_sample = None
        total_sample = None
        total_pixel_values = None
        total_recon_control_pixel_values = None
        total_ref_images = None
        for clip_index in range(infer_num):
            indices = np.linspace(
                start_idx,
                start_idx + stride * (video_length - 1),
                video_length,
                dtype=int,
            )
            try:
                frames = vr.get_batch(indices).asnumpy()
            except Exception as e:
                print(e)
                last_frame_idx = len(vr) - 1
                safe_indices = []
                for idx in indices:
                    if idx < 0 or idx >= len(vr):
                        safe_indices.append(last_frame_idx)
                    else:
                        safe_indices.append(idx)
                frames = vr.get_batch(safe_indices).asnumpy()

            if ref_image_path == "":
                h, w = frames[0].shape[:2]
                new_h, new_w = get_nearest_resize_shape(h, w)
                new_h, new_w = int(new_h), int(new_w)
            else:
                ref_image = Image.open(ref_image_path).convert("RGB")
                ref_w, ref_h = ref_image.size  # PIL uses (width, height)
                new_h, new_w = get_nearest_resize_shape(ref_h, ref_w)
                new_h, new_w = int(new_h), int(new_w)
            frames_resized = np.stack(
                [cv2.resize(frame, (new_w, new_h)) for frame in frames]
            )
            pixel_values = (
                torch.from_numpy(frames_resized)
                .permute(0, 3, 1, 2)
                .contiguous()
                .unsqueeze(0)
                / 255.0
            )
            pixel_values = video_transforms(pixel_values)

            # ref pixel value and mask pixel value
            mask = torch.zeros((1, video_length, 1, new_h, new_w), dtype=torch.uint8)
            mask[:, 1:, :, :, :] = 1
            if ref_image_path == "":
                ref_image = pixel_values[:, 0:1, :, :, :]
                mask_pixel_values = pixel_values * (1 - mask)
            else:
                ref_image = Image.open(ref_image_path).convert("RGB")
                ref_image = ref_image.resize((new_w, new_h), Image.BICUBIC)
                ref_image = (
                    torch.from_numpy(np.array(ref_image))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                ref_image = ref_image.unsqueeze(1)
                ref_image = video_transforms(ref_image)
                mask_pixel_values = ref_image.repeat(1, video_length, 1, 1, 1) * (
                    1 - mask
                )
            clip_pixel_values = (
                ref_image[0][0].permute(1, 2, 0).contiguous() * 0.5 + 0.5
            ) * 255
            clip_image = Image.fromarray(np.uint8(clip_pixel_values))

            # poses
            poses = real_pose[indices - data_dict["video_start_index"]]  # N, 24, 3
            control_pixel_values = get_control_conditions(
                poses, new_h, new_w, use_2d=True
            )
            norm_poses = torch.tensor((poses - global_mean) / global_std)
            ref_pose = norm_poses[0]
            norm_poses = norm_poses - ref_pose
            norm_poses = norm_poses.unsqueeze(0)
            # pose_min = np.min(poses, axis=(0, 1))
            # pose_max = np.max(poses, axis=(0, 1))
            # norm_poses = torch.tensor(2 * ((poses - pose_min) / (pose_max - pose_min)) - 1).unsqueeze(0)
            motion_tokens, vq_loss = vqvae(
                norm_poses.to(device, weight_dtype), return_vq=True
            )
            motion_rotary_emb = prepare_motion_embeddings(
                (video_length - 1) // 4 + 1, 24, global_mean, global_std, device=device
            )
            recon_motion = (
                vqvae(norm_poses.to(device, weight_dtype))[0][0]
                .to(dtype=torch.float32)
                .detach()
                .cpu()
            )
            recon_motion = recon_motion + ref_pose
            # reconstruction_loss = nn.L1Loss()(recon_motion, norm_poses)
            recon_control_pixel_values = get_control_conditions(
                recon_motion * global_std + global_mean, new_h, new_w, use_2d=True
            )
            # recon_control_pixel_values = get_control_conditions((recon_motion + 1) / 2 * (pose_max - pose_min) + pose_min, new_h, new_w, use_2d=True)
            # print(f"vq loss: {vq_loss},   recon loss: {reconstruction_loss.item()}")
            # if reconstruction_loss > 0.01 or vq_loss > 0.01:
            #     import pdb; pdb.set_trace()

            print(f"{index} --- {clip_index} --- {indices[0]} --- {prompt}")
            sample = pipeline(
                prompt=prompt,
                num_frames=video_length,
                num_inference_steps=10 if lora_path != "" else 50,
                negative_prompt="bad hands, extra limbs, fused fingers, blurry, low quality",
                height=new_h,
                width=new_w,
                guidance_scale=guidance_scale,
                generator=generator,
                video=pixel_values,
                mask=mask,
                mask_video=mask_pixel_values,
                clip_image=first_clip_image
                if use_first_clip and clip_index != 0
                else clip_image,
                ref_image=ref_image,
                control_video=None,
                motion_tokens=motion_tokens,
                motion_rotary_emb=motion_rotary_emb,
                use_2d=False,
                previous_sample=previous_sample,
                lora_path=lora_path,
                use_first_clip_image=use_first_clip,
            ).videos

            if clip_index == 0:
                if use_first_clip:
                    first_clip_image = clip_image
                total_sample = sample[:, :, :-overlap]
                total_pixel_values = pixel_values[:, :-overlap]
                total_recon_control_pixel_values = recon_control_pixel_values[
                    :, :-overlap
                ]
                if infer_num == 1:
                    total_sample = sample
                    total_pixel_values = pixel_values
                    total_recon_control_pixel_values = recon_control_pixel_values
                    total_ref_images = ref_image.repeat(1, video_length, 1, 1, 1)
                else:
                    total_ref_images = ref_image.repeat(
                        1, video_length - overlap, 1, 1, 1
                    )
            elif clip_index != infer_num - 1:
                total_sample = torch.cat([total_sample, sample[:, :, :-overlap]], dim=2)
                total_pixel_values = torch.cat(
                    [total_pixel_values, pixel_values[:, :-overlap]], dim=1
                )
                total_recon_control_pixel_values = torch.cat(
                    [
                        total_recon_control_pixel_values,
                        recon_control_pixel_values[:, :-overlap],
                    ],
                    dim=1,
                )
                total_ref_images = torch.cat(
                    [
                        total_ref_images,
                        previous_sample[:, :, -1:]
                        .transpose(1, 2)
                        .repeat(1, video_length - overlap, 1, 1, 1)
                        * 2
                        - 1,
                    ],
                    dim=1,
                )
            else:
                total_sample = torch.cat([total_sample, sample], dim=2)
                total_pixel_values = torch.cat(
                    [total_pixel_values, pixel_values], dim=1
                )
                total_recon_control_pixel_values = torch.cat(
                    [total_recon_control_pixel_values, recon_control_pixel_values],
                    dim=1,
                )
                total_ref_images = torch.cat(
                    [
                        total_ref_images,
                        previous_sample[:, :, -1:]
                        .transpose(1, 2)
                        .repeat(1, video_length, 1, 1, 1)
                        * 2
                        - 1,
                    ],
                    dim=1,
                )

            previous_sample = sample[:, :, -overlap:]
            start_idx = indices[-overlap]

        print(f"save to {save_path}")
        concat_and_save(
            [total_sample.transpose(1, 2)],
            save_path,
            rescale_params=[False],
            method=2,
            fps=fps,
        )
