import torch
from PIL import Image
# from diffsynth import ModelManager, FluxImagePipeline
# from diffsynth.pipelines.flux_image_ip import FluxImagePipeline as FluxImageIPPipeline
# from diffsynth.models.sd3_text_encoder import SD3TextEncoder1

from teleai_pipe.pipelines.flux_image_ip import FluxImagePipeline as FluxImageIPPipeline
from teleai_pipe.models.sd3_text_encoder import SD3TextEncoder1
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.utils.timer import Timer

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

sd3_text_encoder_configs = [
    (
        None,
        "94eefa3dac9cec93cb1ebaf1747d7b78",
        ["sd3_text_encoder_1"],
        [SD3TextEncoder1],
        "diffusers",
    )
]

model_manager = ModelManager(
    file_path_list=[
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/image_encoder.safetensors",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/text_encoder/model.safetensors",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/text_encoder_2",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/flux1-dev.safetensors",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/ae.safetensors",
    ],
    torch_dtype=torch.float8_e4m3fn,
    # torch_dtype=torch.bfloat16,
    device="cuda",
    index_detector=0,
    append_configs=sd3_text_encoder_configs,
)
pipe = FluxImageIPPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)
pipe.enable_vram_management(num_persistent_param_in_dit=None)

pipe = torch.compile(pipe)

ref_dressings = [Image.open("/nvfile-heatstorage/zwshi/resources/ref/ref3.png")]
prompt = "A photo of a beautiful girl"
height = 512
width = 512
seed = 0
attn_impl = "cudnn"

while True:
    with Timer():
        image = pipe(
            prompt=prompt,
            embedded_guidance=3.5,
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=25,
            tea_cache_l1_thresh=0.5,  # 0.5
            ref_images=ref_dressings,
            attn_impl=attn_impl,
        )
    image.save("image_ipt2i.png")
