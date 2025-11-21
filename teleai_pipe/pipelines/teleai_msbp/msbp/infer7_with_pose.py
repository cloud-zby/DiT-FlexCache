import json
import os
import cv2
import fire
import torch
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger
from sklearn.preprocessing import normalize
from torchvision import transforms
import safetensors
from insightface.app import FaceAnalysis

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers import FluxTransformer2DModel, FluxPipeline
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline

from diffusers import FluxControlNetModel, FluxControlNetPipeline

from sbp.nn.torch.models.projection import ImageEncoder
from sbp.nn.torch.models.transformer_flux_custom import FluxTransformer2DModel as FluxTransformerModelWithIP
from sbp.nn.app.diffusers.sd_lora_loader import LoraLoader
from sbp.nn.app.ultralytics.segment import YoloMaskPredictor
from sbp.nn.torch.models.recognition_model import RecognitionModel
from sbp.nn.torch.prompt_utils.sdxl_prompt import get_weighted_text_embeddings
from sbp.nn.model_paths import MODEL_PATHS
from sbp.texts.constants.t2i_styles import NEGATIVE_COMMON_SHORT, MODIFIER_REALISTIC_SDXL
from sbp.vision.common import relax_bbox, relax_bbox_square

from sbp.uno.captioner.qwenvl_caption import caption as Caption
from sbp.dreamo.pipeline import TeleSbpPipe


IMAGE_TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((384, 384), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class FaceFeatureProcessor:

    def __init__(self, model_dir='~/.insightface') -> None:
        self.face_analysis = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'], root=model_dir)
        self.face_analysis.prepare(0, det_size=(640, 640), det_thresh=0.5)

    @staticmethod
    def _area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def process_ref_image(self, image_face):
        faces = self.face_analysis.get(image_face)
        faces = sorted(faces, key=lambda x: -self._area(x['bbox']))
        if len(faces) > 0:
            face = faces[0]
            face_feature = face['embedding']
            return face_feature
        else:
            print("warning: no face detected")
            return None

    def process_ref_image_list(self, image_face_list):
        face_features = []
        for image_face in image_face_list:
            image_face = cv2.imread(image_face)
            ff = self.process_ref_image(image_face)
            if ff is not None:
                face_features.append(ff)
        if len(face_features) == 0:
            return None
        else:
            face_features = normalize(np.stack(face_features, axis=0))
            return torch.tensor(face_features).unsqueeze(0).cuda() * 20
        
    def detect_image(self, image):
        return self.face_analysis.get(image)
    
    def get_topk_faces(self, image, topk=3):
        faces = self.face_analysis.get(image)
        faces = sorted(faces, key=lambda x: -self._area(x['bbox']))[:topk]
        if len(faces) > 0:
            return faces
        else:
            return None
        
    def match_faces(self, faces, ref_faces):
        """match ref_faces with faces which is topk faces returned, give the final index of order of ref_faces"""
        target_face_features = [face['embedding'] for face in faces]
        ref_face_features = []
        for face in ref_faces:
            image_face = cv2.imread(face)
            ff = self.process_ref_image(image_face)
            if ff is not None:
                ref_face_features.append(ff)
        if len(target_face_features) == 0 or len(ref_face_features) == 0:
            return []
        target_face_features = normalize(np.stack(target_face_features, axis=0))
        ref_face_features = normalize(np.stack(ref_face_features, axis=0))
        sim_matrix = target_face_features @ ref_face_features.T
        matched_indices = sim_matrix.argmax(axis=1)
        return matched_indices

class CharacterGenModel:

    def __init__(
        self, 
        sdxl_path: str,
        vae_path: str, 
        lora_path: str, 
        person_detection_path: str,
        image_encoder_path: str,
        face_model_path: str
    ) -> None:
        # - sdxl base model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.bfloat16, variant='fp16' if 'base' in sdxl_path else None)
        self.pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe = self.pipe.to('cuda')
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.pipe.vae,
            unet=self.pipe.unet, 
            scheduler=self.pipe.scheduler,
            text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, # type: ignore
        )

        # - lora for sdxl base model
        self.lora_loader = LoraLoader()
        self.lora_loader.load_lora_for_pipelines([self.pipe], [{lora_path: 0.8}])
        self.pipe.scheduler = EulerAncestralDiscreteScheduler(beta_start=6e-4, beta_end=9e-3)
        self.inpaint_pipe.scheduler = EulerAncestralDiscreteScheduler(beta_start=8e-4, beta_end=10.5e-3, beta_schedule='scaled_linear')

        # - projection from recognition feature to text-embedding
        self.conditioner_path = f'{lora_path}_mlp.pt'
        self.conditioner = torch.load(self.conditioner_path).to('cuda').eval()

        # - recognition model for face and body
        # self.body_app = BodyFeatureProcessor(person_detection_path, image_encoder_path)
        self.face_app = FaceFeatureProcessor(face_model_path)

    def get_text_embedding_with_ref(self, prompt, negative_prompt, ref_dressing=[], ref_faces=[]):
        extra_conditions = {}
        # 1. get recognition for face and body
        # body_features = self.body_app.process_ref_image_list(ref_dressing)
        face_features = self.face_app.process_ref_image_list(ref_faces)
        # if body_features is not None:
        #     extra_conditions['body'] = body_features
        #     extra_conditions['character'] = body_features
        if face_features is not None:
            extra_conditions['face'] = face_features

        logger.info(f"{prompt} condition shape: {[(k, v.shape) for k, v in extra_conditions.items()]})")
        
        # 2. get text embedding
        prompt = prompt + MODIFIER_REALISTIC_SDXL
        text_embedding, _, uncond_embedding, _ = get_weighted_text_embeddings(self.pipe, prompt, negative_prompt)
        text_embedding2, pool2, uncond_embedding2, unpool2 = get_weighted_text_embeddings(self.pipe, prompt, negative_prompt, is_sdxl_text_encoder2=True)

        # 3. project feature and merge with text embedding
        if len(extra_conditions) != 0:
            text_embedding, text_embedding2, _ = self.conditioner([text_embedding, text_embedding2, pool2], extra_conditions)
        text_embedding = torch.cat([text_embedding, text_embedding2], axis=-1) # type: ignore
        uncond_embedding = torch.cat([uncond_embedding, uncond_embedding2], axis=-1) # type: ignore
        return text_embedding, uncond_embedding, pool2, unpool2, extra_conditions
    
    def refine_face(self, image, ref_faces, guidance_scale=5, num_inference_steps=30, crop_size=1024, strength=0.3, padding=0.4):
        try:
            target_faces = self.face_app.get_topk_faces(np.array(image)[..., ::-1], len(ref_faces))
            if target_faces is None:
                return image, None
        except Exception as e:
            return image, None
        indices = self.face_app.match_faces(target_faces, ref_faces)
        ref_bbox = [None for _ in indices]
        for i, target_face in zip(indices, target_faces):
            H, W = image.height, image.width
            bbox = relax_bbox_square(target_face['bbox'], H, W)
            ref_bbox[i] = bbox
            bbox = relax_bbox(bbox, H, W, 0.02, 0.02, 0.02, return_integer=True)
            x1, y1, x2, y2 = relax_bbox(bbox, H, W, padding, padding, padding, return_integer=True)
            cropped_image = image.crop((x1, y1, x2, y2))
            mask = Image.new('L', (W, H), 0)
            ImageDraw.Draw(mask).rectangle(bbox, fill=255)
            mask = mask.crop((x1, y1, x2, y2))
            h, w = mask.height, mask.width
            r = crop_size / min(h, w)
            h = int((h * r) // 64 * 64)
            w = int((w * r) // 64 * 64)
            text_embedding, uncond_embedding, pool2, unpool2, _ = self.get_text_embedding_with_ref(
                'the face of a person, ' + MODIFIER_REALISTIC_SDXL,
                NEGATIVE_COMMON_SHORT,
                ref_faces=[ref_faces[i]]
            )
            patch_fixed =  self.inpaint_pipe(
                image=cropped_image,
                mask_image=mask,
               strength=strength,
                prompt_embeds=text_embedding, 
                negative_prompt_embeds=uncond_embedding, 
                pooled_prompt_embeds=pool2, 
                negative_pooled_prompt_embeds=unpool2, 
                guidance_scale=guidance_scale, 
                num_inference_steps=num_inference_steps, 
                width=w, 
                height=h
            ).images[0]
            image.paste(patch_fixed.resize((x2 - x1, y2 - y1)), (x1, y1, x2, y2))
        return image, ref_bbox


class FluxCharacterModel:

    device = 'cuda:0'

    def __init__(
        self, 
        flux_path: str,
        ip_adapter_path: str, 
        flux_fill_path: str,
        msbp_model_path_1: str,
        msbp_model_path_2: str,
        ben2_model_path: str,
        ip_adapter_fill_path: str,
        qwenvl_model_path: str,
        conditioner_base_model: str = 'eva02_large_patch14_448.mim_in22k_ft_in1k',
        conditioner_base_model_path: str = "/nvfile-heatstorage/AIGC_H100/cv_banc/fengzr/hf_model/teleai_pipe_msbp/eva02_large_patch14_448.mim_in22k/model.safetensors",
        controlnet_model_path: str = None,
        device=None
    ) -> None:

        self.device = device or self.device
        self.pipe = TeleSbpPipe(model_root=flux_path, ben2_model_path=ben2_model_path, use_turbo=False, msbp_model_path_1=msbp_model_path_1, msbp_model_path_2=msbp_model_path_2)

        # self.pipe, self.proj = self.load_model(flux_path, ip_adapter_path, conditioner_base_model, num_condition_layer=12, condition_use_pyramid=True)

        self.proj = ImageEncoder(output_dim=4096, layer_num=12, seq_len=2, 
            device=self.device, base_model=conditioner_base_model, conditioner_base_model_path=conditioner_base_model_path,
            use_pyramid=True).to(self.device, dtype=torch.bfloat16)
        state_dict = safetensors.torch.load_file(ip_adapter_path)
        state_dict = {'.'.join(k.split('.')[1:]): state_dict[k] for k in state_dict.keys()}
        self.proj = self.proj.load_state_dict(state_dict, strict=False)

        # self.pipe = TeleSbpPipe(model_root=flux_path, ben2_model_path=ben2_model_path, use_turbo=False, msbp_model_path_1=msbp_model_path_1, msbp_model_path_2=msbp_model_path_2)
        self.caption = Caption(model_path=qwenvl_model_path, device_map="cuda:1")
        if controlnet_model_path is not None:
            controlnet = FluxControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.bfloat16).to(device=self.device)
            self.control_pipe = FluxControlNetPipeline(
                scheduler = self.pipe.telesbp_pipeline.scheduler,
                vae = self.pipe.telesbp_pipeline.vae,
                text_encoder = self.pipe.telesbp_pipeline.text_encoder,
                tokenizer = self.pipe.telesbp_pipeline.tokenizer,
                text_encoder_2 = self.pipe.telesbp_pipeline.text_encoder_2,
                tokenizer_2 = self.pipe.telesbp_pipeline.tokenizer_2,
                transformer = self.pipe.telesbp_pipeline.transformer,
                controlnet = controlnet
            )
        if flux_fill_path is not None and flux_fill_path != "none":
            self.pipe_fill, self.proj_fill = self.load_model(
                flux_fill_path, ip_adapter_fill_path, conditioner_base_model, is_fill=True, 
                text_encoder = None,
                text_encoder_2 = None,
                vae = self.pipe.telesbp_pipeline.vae,
                conditioner_base_model_path=conditioner_base_model_path
            )
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.276])
        ])

    def load_model(self, model_path, ip_adapter_path, conditioner_base_model, conditioner_base_model_path="", num_condition_layer=9, condition_use_pyramid=False, is_fill=False, **pipe_kwargs):
        print(f"loading {model_path}, {ip_adapter_path}...")
        copy = FluxTransformer2DModel.from_pretrained(model_path, subfolder='transformer', torch_dtype=torch.bfloat16, disable_mmap=True)
        model = FluxTransformerModelWithIP.from_config(os.path.join(model_path, 'transformer', 'config.json'), torch_dtype=torch.bfloat16) # type: ignore
        model.load_state_dict(copy.state_dict(), strict=False)
        del copy
        image_proj = ImageEncoder(output_dim=4096, layer_num=num_condition_layer, seq_len=2, 
            device=self.device, base_model=conditioner_base_model, conditioner_base_model_path=conditioner_base_model_path, use_pyramid=condition_use_pyramid).to(self.device, dtype=torch.bfloat16)
        Pipeline = FluxFillPipeline if is_fill else FluxPipeline
        pipe = Pipeline.from_pretrained(model_path, transformer=model, torch_dtype=torch.bfloat16, disable_mmap=True, **pipe_kwargs).to(device=self.device, dtype=torch.bfloat16)
        state_dict = safetensors.torch.load_file(ip_adapter_path)
        state_dict = {'.'.join(k.split('.')[1:]): state_dict[k] for k in state_dict.keys()}
        diff1 = model.load_state_dict(state_dict, strict=False)
        diff2 = image_proj.load_state_dict(state_dict, strict=False)
        return pipe, image_proj
    
    def gen_image(self, prompt, ref_images, **kwargs):
        with torch.no_grad():
            ref_conds = []
            if len(ref_images) > 0 and kwargs['pose_keypoints'] is None:
                ref_imgs = [Image.open(ref_image) for ref_image in ref_images]
                ref_img_prompts = []
                for ref_img in ref_imgs:
                    ref_img_prompt = self.caption.generate_caption(ref_img, caption_prompt=self.caption.caption_prompt).strip('.')
                    ref_img_prompts.append(ref_img_prompt)
                person_num = len(ref_img_prompts)
                if person_num == 1:
                    combine_prompt = self.caption.combine_one_person_prompt_template.format(description=ref_img_prompts[0], action=prompt)
                    prompt = self.caption.generate_caption([], caption_prompt=combine_prompt)
                elif person_num == 2:
                    combine_prompt = self.caption.combine_two_person_prompt_template.format(description0=ref_img_prompts[0], description1=ref_img_prompts[1], action=prompt)
                    prompt = self.caption.generate_caption([], caption_prompt=combine_prompt)
                elif person_num == 3:
                    combine_prompt = self.caption.combine_three_person_prompt_template.format(description0=ref_img_prompts[0], description1=ref_img_prompts[1], 
                                                                                                description2=ref_img_prompts[2], action=prompt)
                    prompt = self.caption.generate_caption([], caption_prompt=combine_prompt)
                else:
                    prompt = ", ".join(ref_img_prompts + [prompt])

                for idx, ref_img in enumerate(ref_imgs):
                    ref_cond = self.pipe.preprocess_image(idx, ref_img)
                    ref_conds.append(ref_cond)
            logger.info(f"prompt input: {prompt}")
            h, w = kwargs.get('height', 432), kwargs.get('width', 768)
            if 'pose_keypoints' in kwargs and kwargs['pose_keypoints'] is not None:
                prompt_embeds, pooled_prompt_embeds, txt_ids = self.pipe.telesbp_pipeline.encode_prompt(prompt, prompt, max_sequence_length=256)
                if len(ref_images) > 0:
                    print("use", ref_images)
                    image_refs = map(torch.stack, [
                        [self.transform(Image.open(i).convert('RGB')) for i in ref_images]
                    ])    
                    image_refs = [i.to(dtype=torch.bfloat16, device=self.device) for i in image_refs]
                    visual_prompt_embeds = self.proj(image_refs)
                    prompt_embeds_with_ref = torch.cat([prompt_embeds, visual_prompt_embeds], dim=1)
                else:
                    prompt_embeds_with_ref = prompt_embeds
                pose_keypoints = kwargs['pose_keypoints']
                if len(pose_keypoints[0]) == 134:
                    from sbp.vision.protocal import ImageResult, DWPoseResult
                    from sbp.nn.app.dwpose import draw_keypoints_image_result
                    pose_keypoints = np.array(pose_keypoints)
                    pose_keypoints[:, 18:24] = -1
                    pose_keypoints[:, 91:] = -1

                    def get_bbox(keypoints):
                        kps = np.array(keypoints)
                        kps = kps[kps[:, 0] > 0]
                        return min(kps[:, 0]), min(kps[:, 1]), max(kps[:, 0]), max(kps[:, 1])

                    def to_norm_kps(keypoints):
                        keypoints = [
                            ((x / w, y / h, 1) if x > 0 and y > 0 else (-1, -1, -1)) if _i < 91 and _i not in list(range(18, 24)) else (-1, -1, -1)
                            for _i, (x, y) in enumerate(keypoints)
                        ]
                        return keypoints

                    image_result = ImageResult(detection_results=[
                            DWPoseResult('person', confidence=1, bbox=get_bbox(keypoints), keypoints=to_norm_kps(keypoints)) 
                            for keypoints in pose_keypoints
                        ], height=h, width=w)
                    image_result.detection_results = image_result.get_detection('person', topk=1)

                    ref_pose_image = draw_keypoints_image_result(image_result, draw_hand=True, draw_face=True)[1][0]
                    ref_pose_image = ref_pose_image.resize((w, h))
                else:
                    from sbp.vision.utils.convert_kps import draw_dwpose_from_smpl
                    ref_pose_image = draw_dwpose_from_smpl(kwargs['pose_keypoints'], (w, h))

                gen_kw = {
                    'control_image': ref_pose_image,
                    'controlnet_conditioning_scale': 0.5,
                    'control_mode': 4,
                    'guidance_scale': 3.5
                }
                pipe = self.control_pipe
                logger.info("using control pipeline ...")
                images = pipe(
                    prompt_embeds=prompt_embeds_with_ref,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    height=h,
                    width=w,
                    num_images_per_prompt=kwargs.get('num_images_per_prompt', 1),
                    num_inference_steps=kwargs.get('num_inference_steps', 25),
                    generator=torch.Generator().manual_seed(kwargs.get('seed', 0)) if 'seed' in kwargs else None,
                    **gen_kw).images
            else:
                self.pipe.telesbp_pipeline.switch_lora(len(ref_images))
                images = self.pipe.telesbp_pipeline(
                prompt=prompt,
                width=w,
                height=h,
                num_inference_steps=25,
                num_images_per_prompt=kwargs.get('num_images_per_prompt', 1),
                guidance_scale=3.5,
                ref_conds=ref_conds,
                generator=torch.Generator(device="cpu").manual_seed(kwargs.get('seed', 0)) if 'seed' in kwargs else None,
                true_cfg_scale=1,
                true_cfg_start_step=0,
                true_cfg_end_step=0 ,
                negative_prompt="",
                neg_guidance_scale=3.5,
                first_step_guidance_scale=3.5,
                ).images
            
            return images

    def fill_image(self, prompt, edit_image, ref_images, bbox, keypoints=None, **kwargs):
        ori_width, ori_height = edit_image.size
        target_width = 16 * (ori_width // 16)
        target_height = 16 * (ori_height // 16)
        target_bbox = [
            bbox[0] * target_width // ori_width,
            bbox[1] * target_height // ori_height,
            bbox[2] * target_width // ori_width,
            bbox[3] * target_height // ori_height,
        ]
        mask = Image.new('L', (target_width, target_height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(target_bbox, fill=255)
        with torch.no_grad():
            image_refs = map(torch.stack, [
                [self.transform(Image.open(i).convert('RGB')) for i in ref_images]
            ])    
            image_refs = [i.to(dtype=torch.bfloat16, device=self.device) for i in image_refs]
            prompt_embeds, pooled_prompt_embeds, txt_ids = self.pipe.telesbp_pipeline.encode_prompt(prompt, prompt)
            visual_prompt_embeds = self.proj_fill(image_refs)
            prompt_embeds_with_ref = torch.cat([prompt_embeds, visual_prompt_embeds], dim=1)
            image = self.pipe_fill(
                image=edit_image,
                mask_image=mask,
                width=target_width,
                height=target_height,
                prompt_embeds=prompt_embeds_with_ref,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=kwargs.get('num_inference_steps', 25),
                guidance_scale=kwargs.get('guidance_scale', 30),
            ).images[0]

            image = image.resize((ori_width, ori_height), Image.LANCZOS)

            return image


@fire.Fire
def main(
    lora_path: str,
    testsets: str,
    output_dir: str,
    sdxl_path: str,
    vae_path:str,
    person_detection_path: str,
    image_encoder_path: str,
    conditioner_base_model_path: str,
    flux_path: str = None,
    flux_ip_model_path: str = None,
    msbp_model_path_1: str = None,
    msbp_model_path_2: str = None,
    ben2_model_path: str = None,
    flux_fill_model_path: str = None,
    flux_ip_fill_model_path: str = None,
    qwenvl_model_path: str = None,
    flux_controlnet_model_path: str = None,
    face_model_path: str = '~/.insightface/',
    num_images_per_prompt: int = 1,
    height: int = 768,
    width: int = 512,
):
    cgmodel = CharacterGenModel(sdxl_path, vae_path, lora_path, person_detection_path, image_encoder_path, face_model_path)
    flux_model = FluxCharacterModel(flux_path, 
                                    flux_ip_model_path, 
                                    flux_fill_model_path,
                                    msbp_model_path_1,
                                    msbp_model_path_2,
                                    ben2_model_path,
                                    flux_ip_fill_model_path,
                                    qwenvl_model_path,
                                    conditioner_base_model_path=conditioner_base_model_path,
                                    controlnet_model_path=flux_controlnet_model_path)
    os.makedirs(output_dir, exist_ok=True)
    negative_prompt = NEGATIVE_COMMON_SHORT

    def gen_single(
        prompt, 
        ref_faces = [],
        ref_dressing = [],
        ref_edit = [],
        pose_keypoints = None,
        height = height,
        width = width,
        bboxes = [],
        prompt_for_bboxes = [],
        seed = None
    ):  
        kwargs = {'seed': seed} if seed is not None else {}
        if len(ref_edit) == 0:
            generated_images = [
                flux_model.gen_image(prompt, ref_dressing, num_images_per_prompt=1, height=height, width=width, pose_keypoints=pose_keypoints, **kwargs)[0] 
                for i in range(num_images_per_prompt)
            ]
        else:
            generated_images = [Image.open(i).convert('RGB') for i in ref_edit]
        if len(ref_edit) > 0 and len(ref_dressing) > 0:
            new_generated_images = []
            im = generated_images[0]
            for bbox, prompt_for_bbox, ref_i in zip(
                bboxes,
                prompt_for_bboxes,
                ref_dressing
            ):
                im = flux_model.fill_image(
                    prompt_for_bbox, 
                    edit_image = im, 
                    ref_images = [ref_i], 
                    bbox = bbox,
                    **kwargs
                )
                new_generated_images.append(im)
            generated_images = new_generated_images

        generated_images, ref_bboxes = zip(*[cgmodel.refine_face(
            im, 
            ref_faces=ref_faces,
            num_inference_steps=12,
            guidance_scale=3.5,
            crop_size=768
            ) for im in generated_images])
        return generated_images, ref_bboxes

    for i, testcase in enumerate((json.loads(line.strip()) for line in open(testsets))):
        # print(testcase['prompt'], negative_prompt)
        # generated_images = gen_single(testcase['prompt'], testcase.get('ref_faces', []), testcase.get('ref_dressing', []), testcase.get('ref_edit', []))
        logger.info(f"testcase input: {testcase}")
        generated_images, ref_bboxes = gen_single(**testcase)

        for j, image in enumerate(generated_images):
            image.save(f'{output_dir}/{i}_{j}.png')
    
