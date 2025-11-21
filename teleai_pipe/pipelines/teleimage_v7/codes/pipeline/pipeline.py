import torch
from diffusers import DiffusionPipeline
from utils.encryption import decryption_model
import time

class TeleImagePipeline():
    def __init__(self, args, device, lora_model_paths=None):

        self.args = args
        model_name = args.model_path
        torch_dtype = torch.bfloat16

        # 加载模型到指定GPU
        print("--------加载图像生成模型---------")
        print("模型路径: ", model_name, "到", device)
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.pipe = pipe.to(device)
        self.device = device
        print("--------------------------\n\n\n")

        # 加载LoRA权重
        if lora_model_paths is not None:
            lora_names = []
            lora_adapter_weights = []
            print("--------加载lora---------")
            for idx, (lora_model_path, adapter_weight) in enumerate(lora_model_paths.items()):
                lora_name = f"lora_{str(idx)}"
                lora_names.append(lora_name)
                lora_adapter_weights.append(float(adapter_weight))
                if 'dll.' not in lora_model_path:
                    self.pipe.load_lora_weights(lora_model_path, adapter_name=lora_name)
                else:
                    # 解密模型
                    de_state_dict, metadata = decryption_model(lora_model_path)
                    state_dict = self.pipe.lora_state_dict(de_state_dict, adapter_name=lora_name)
                    self.pipe.load_lora_into_transformer(
                        state_dict,
                        transformer=self.pipe.transformer,
                        adapter_name=lora_name,
                        metadata=metadata,
                        _pipeline=self.pipe,
                        low_cpu_mem_usage=False,
                    )
                #print(f"成功加载lora {lora_model_path}, 权重为 {adapter_weight}")
            # 设置权重
            self.pipe.set_adapters(lora_names, adapter_weights=lora_adapter_weights)

            #print("--------------------------\n\n\n")

        # self.aspect_ratios = {
        #     "1:1": (1328, 1328),
        #     "16:9": (1664, 928),
        #     "9:16": (928, 1664),
        #     "4:3": (1472, 1104),
        #     "3:4": (1104, 1472),
        #     "3:2": (1584, 1056),
        #     "2:3": (1056, 1584),
        #     "480p": (720, 480),
        #     "720p": (1280, 720),
        #     "1080p": (1920, 1080),
        #     "480b": (480, 720),
        #     "720b": (720, 1280),
        #     "1080b": (1080, 1920),
        # }
        self.aspect_ratios = {
            "highres": {
                "16:9": (1920, 1080),
                "9:16": (1080, 1920),
                "3:2": (1760, 1184),
                "2:3": (1184, 1760),
                "1:1": (1440, 1440)
            },
            "medres": {
                "16:9": (1376, 768),
                "9:16": (768, 1376),
                "3:2": (1248, 832),
                "2:3": (832, 1248),
                "1:1": (1024, 1024)
            },
            "lowres": {
                "16:9": (1920, 1080),
                "9:16": (1080, 1920),
                "3:2": (1760, 1184),
                "2:3": (1184, 1760),
                "1:1": (1440, 1440)
            },
            
        }
        self.lower_aspect_ratios = {
            "16:9": (800, 448),
            "9:16": (448, 800),
            "3:2": (720, 480),
            "2:3": (480, 720),
            "1:1": (592, 592)
        }
    def run(self, prompt, aspect_ratios, seed, batch_size, num_inference_steps, true_cfg_scale, quality):
        # seed = seed
        width, height = self.aspect_ratios[quality][aspect_ratios]


        start_time = time.perf_counter()

        #print("--------生成图像---------\n")
        #print("num_inference_steps: ", num_inference_steps)
        #print("true_cfg_scale: ", true_cfg_scale)
        #print("seed: ", seed)
        #print("batch_size: ", batch_size)
        #print("width: ", width)
        #print("height: ", height)
        #print("prompt: ", prompt)

        
        image = self.pipe(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_images_per_prompt=batch_size
        ).images

        
        if quality == 'lowres':
            output_images = []
            for i, output_image in enumerate(image):
                output_image = output_image.resize(self.lower_aspect_ratios[aspect_ratios])
                output_images.append(output_image)
            image = output_images

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"生成{batch_size} 张图像， 耗时: {execution_time:.3f} 秒")
        print("--------------------------\n\n\n")

        return image
