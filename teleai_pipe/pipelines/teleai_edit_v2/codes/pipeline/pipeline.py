from os.path import join, basename, dirname
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, AutoencoderKL, FluxPipeline
from diffusers.loaders import FluxLoraLoaderMixin
import torch
from optimum.quanto import freeze, qfloat8, quantize
import random

from .controlnet_flux import FluxControlNetModel
from .transformer_flux import FluxTransformer2DModel as FluxControlTransformer2DModel
from .pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from .pipeline_flux_fill import AccelerateTeleFillPipeline

from utils.encryption import decryption_model


class TeleEditPipeline():
    def __init__(self, args):
        self.args = args.teleeditpipeline
        self.device = args.device
        self.current_dir = dirname(dirname(__file__))
        self.fill_path = join(self.current_dir, self.args.fill_path)
        self.gguf_path = join(self.current_dir, self.args.gguf_path)
        self.control_path = join(self.current_dir, self.args.control_path)
        self.dev_path = join(self.current_dir, self.args.dev_path)
        self.lora_path = join(self.current_dir, self.args.lora_path)
        self.transformer_device = {i:j for i,j in zip(args.device.transfromer.GPU, args.device.transfromer.MEM)}

        self.init_remove_pipe()
        self.init_add_replace_pipe()
        
        

    def init_add_replace_pipe(self):
        transfomer_config_path = join(self.fill_path, 'transformer/config.json')
        transformer = FluxTransformer2DModel.from_single_file(
            self.gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=transfomer_config_path,
            torch_dtype=torch.bfloat16,
        )
        text_encoder = FluxPipeline.from_pretrained(
            self.dev_path,
            transformer=None,
            vae=None,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        quantize(text_encoder.text_encoder_2, weights=qfloat8)
        freeze(text_encoder.text_encoder_2)
        
        self.fill_pipe = AccelerateTeleFillPipeline.from_pretrained(
            self.fill_path,
            text_encoder=text_encoder.text_encoder,
            text_encoder_2=text_encoder.text_encoder_2,
            tokenizer=text_encoder.tokenizer,
            tokenizer_2=text_encoder.tokenizer_2,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(self.device.fill_config)


    def init_remove_pipe(self):
        controlnet = FluxControlNetModel.from_pretrained(
            self.control_path, torch_dtype=torch.bfloat16
            ).to(self.device.control_config)
        text_encoder = FluxPipeline.from_pretrained(
            self.dev_path,
            transformer=None,
            vae=None,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            ).to(self.device.control_config)
        quantize(text_encoder.text_encoder_2, weights=qfloat8)
        freeze(text_encoder.text_encoder_2)
        transformer = FluxControlTransformer2DModel.from_pretrained(
            self.dev_path,
            subfolder="transformer",
            device_map="auto",
            max_memory=self.transformer_device,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            )
        
        if self.args.encryption:
            lora_path = join(self.lora_path, "pytorch_lora_weights.dll.safetensors")
            de_state_dict = decryption_model(lora_path)
            state_dict = FluxLoraLoaderMixin.lora_state_dict(de_state_dict)
        else:
            lora_path = join(self.lora_path, "pytorch_lora_weights.safetensors")
            state_dict = FluxLoraLoaderMixin.lora_state_dict(lora_path)
        FluxLoraLoaderMixin.load_lora_into_transformer(state_dict, network_alphas=None, transformer=transformer, adapter_name='remove')
        
        vae = AutoencoderKL.from_pretrained(
            self.dev_path,
            subfolder="vae", 
            torch_dtype=torch.bfloat16
            ).to(self.device.control_config)
        self.control_pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            self.dev_path,
            controlnet=controlnet,
            transformer=transformer,
            text_encoder=text_encoder.text_encoder,
            text_encoder_2=text_encoder.text_encoder_2,
            tokenizer=text_encoder.tokenizer,
            tokenizer_2=text_encoder.tokenizer_2,
            vae=vae,
            torch_dtype=torch.bfloat16
        )


    def operate_add_replace(self, prompt, image, mask, generator):
        image_out = self.fill_pipe(
            prompt=prompt,
            image=image,
            width=image.width,
            height=image.height,
            mask_image=mask, 
            guidance_scale=30, 
            num_inference_steps=20,
            generator=generator
            ).images[0]
        torch.cuda.empty_cache()
        return image_out


    def operate_remove(self, prompt, image, mask, generator):
        self.control_pipe.set_adapters(['remove'], adapter_weights=[1.0])
        image_out = self.control_pipe(
            prompt=prompt,
            control_image=image,
            control_mask=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt="",
            true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
        ).images[0]
        torch.cuda.empty_cache()
        return image_out


    def run(self, prompt, image, mask, operation, seed=0):
        if seed== -1:                                   # -1表示随机种子
            seed = random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator('cpu').manual_seed(seed)
        if operation == 'add' or operation == 'replace':
            image = self.operate_add_replace(prompt, image, mask, generator)
        elif operation == 'remove':
            prompt = 'There is nothing here.'
            image = self.operate_remove(prompt, image, mask, generator)

        return image

    