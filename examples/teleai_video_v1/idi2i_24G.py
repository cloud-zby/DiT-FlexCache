# import debugpy
# import os
# print("waiting to connect")
# debugpy.listen(("0.0.0.0", os.environ.get("DEBUG_PORT", 5881)))
# debugpy.wait_for_client()
# print("connected with client")


import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw

from teleai_pipe.pipelines.sdxl_image_inpiant_id import SDXLInpaintIDImagePipeline
from teleai_pipe.models.face_feature import FaceFeatureProcessor
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.utils.timer import Timer   

#torch.cuda.set_per_process_memory_fraction(0.3, 0)

current_path = os.getcwd()
sys.path.append(f"{current_path}/teleai_pipe/models/")


real_path="/nvfile-heatstorage/zwshi/models/hub/models--zwshi--008g-merge/snapshots/aeab11b103b9211559139b4039011f634722a758/"
sdxl_path="/nvfile-heatstorage/zwshi/models/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/"
lora_path = "/nvfile-heatstorage/zwshi/models/private/021.d/output"
lora_file: str = f"{lora_path}/ckpt-step00028000.safetensors"
condition_path: str = f"{lora_path}/ckpt-step00028000.safetensors_mlp.pt"
face_model_path: str = '/nvfile-heatstorage/zwshi/models/insightface/'

# ##### load lora into unet
# from sbp.nn.app.diffusers.sd_lora_loader import LoraLoader
# from diffusers import StableDiffusionXLPipeline
# pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.bfloat16, variant='fp16')
# pipe = pipe.to('cuda')
# lora_loader = LoraLoader()
# lora_loader.load_lora_for_pipelines([pipe], [{lora_file: 0.8}])
# torch.save(pipe.unet.state_dict(), f"{lora_path}/lora_loaded.pt")

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager = ModelManager(
    file_path_list=[
        f"{real_path}/text_encoder/model.safetensors",
        f"{real_path}/text_encoder_2/model.safetensors",
        f"{sdxl_path}/vae/diffusion_pytorch_model.safetensors",
        #f"{real_path}/unet/diffusion_pytorch_model.safetensors", # original unet
        f"{lora_path}/lora_loaded.pt" # unet merged with lora
    ],   
    #torch_dtype=torch.float8_e4m3fn,
    torch_dtype=torch.bfloat16,
    device="cuda", 
)
pipe = SDXLInpaintIDImagePipeline.from_model_manager(model_manager)

#pipe = torch.compile(pipe)

face_app = FaceFeatureProcessor(face_model_path)
conditioner = torch.load(condition_path, map_location="cuda", weights_only=False)
pipe.face_app = face_app
pipe.conditioner = conditioner

NEGATIVE_COMMON_SHORT = 'ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, cartoon, anime, render, painting, 3d, watermark, signature, letters, logo, word, nsfw'
MODIFIER_REALISTIC_SDXL = 'realistic, photo, best quality, highest resolution, 8K, ultra detailed, cinematic'
prompt = "the face of a person" + MODIFIER_REALISTIC_SDXL
negative_prompt = NEGATIVE_COMMON_SHORT


while True:
    image = Image.open('/nvfile-heatstorage/zwshi/resources/ref/ref5.png').convert('RGB')
    #image = Image.open("/nvfile-heatstorage/zwshi/projects/sbp/misc/241113_train_ddpm/experiments/001.b/output/eval-000000/1-0_ref.jpg").convert('RGB')
    ref_faces = ["/nvfile-heatstorage/zwshi/projects/sbp/misc/241113_train_ddpm/experiments/001.b/output/eval-000000/1-1_ref.jpg"]

    target_faces = face_app.get_topk_faces(np.array(image)[..., ::-1], 3)
    if target_faces is None:
        raise ValueError
    indices = face_app.match_faces(target_faces, ref_faces)
    for i, target_face in zip(indices, target_faces):
        
        cropped_image, mask, bbox, hw = face_app.generate_mask_from_detection(image, target_face)
        x1, y1, x2, y2 = bbox
        with Timer():
            patch_fixed = pipe(
                input_image=cropped_image, 
                mask_image=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=0.5,
                cfg_scale=3.5, #3.5 better result, 1.0: quicker exec
                width=hw[1], 
                height=hw[0],
                num_inference_steps=10,
                ref_faces=ref_faces
            )
        image.paste(patch_fixed.resize((x2 - x1, y2 - y1)), (x1, y1, x2, y2))
    image.save("512.jpg")

