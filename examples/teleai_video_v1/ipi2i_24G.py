import torch
from PIL import Image, ImageDraw
# from diffsynth import ModelManager

from teleai_pipe.pipelines.flux_image_fill_ip import FluxImageFillPipeline
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
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/image_encoder_fill.safetensors",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/text_encoder/model.safetensors",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/text_encoder_2",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/flux1-dev-fill-ip.safetensors",
        "/nvfile-heatstorage/model_zoo/televideo/teleai_keyframe_v1/FLUX-IP/FLUX.1-dev/ae.safetensors",
    ],
    torch_dtype=torch.float8_e4m3fn,  # 16G, 12s
    # torch_dtype=torch.bfloat16, # 30G, 22s
    device="cuda",
    index_detector=0,
    append_configs=sd3_text_encoder_configs,
)
pipe = FluxImageFillPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)
pipe.enable_vram_management(num_persistent_param_in_dit=None)

pipe = torch.compile(pipe)

ref_edit = Image.open("/nvfile-heatstorage/zwshi/resources/ref/ref5.png")
w, h = ref_edit.size
ref_bboxes = [
    [100, 100, w - 100, h - 100],
]
# ref_bboxes = [[0, 0, w, h]]
ref_dressings = [Image.open("/nvfile-heatstorage/zwshi/resources/ref/ref3.png")]
# ref_dressings = [None]
prompt = "A photo of a woman"
attn_impl = "cudnn"


def run(
    pipe, prompt, ref_edit, ref_dressings, ref_bboxes, prompts_for_bbox=None, seed=0
):
    if prompts_for_bbox is None:
        prompts_for_bbox = [prompt for _ in ref_bboxes]

    input_image = ref_edit
    for _r, _bbox, p in zip(ref_dressings, ref_bboxes, prompts_for_bbox):
        i = input_image
        mask_image = Image.new("L", (i.width, i.height), color=0)
        draw = ImageDraw.Draw(mask_image)
        x1, y1, x2, y2 = map(int, _bbox)
        draw.rectangle([x1, y1, x2, y2], fill=255)

        image = pipe(
            prompt=p,
            input_image=input_image,
            mask_image=mask_image,
            embedded_guidance=30,
            height=512,
            width=512,
            seed=seed,
            num_inference_steps=25,
            tea_cache_l1_thresh=0.5,  # 0.15
            ref_images=[_r] if _r is not None else None,
            attn_impl=attn_impl,
        )
        input_image = image
    return image


while True:
    with Timer():
        image = run(pipe, prompt, ref_edit, ref_dressings, ref_bboxes, seed=0)
    image.save("image_ipi2i.png")
