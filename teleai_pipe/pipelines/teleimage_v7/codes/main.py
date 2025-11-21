import os
from os.path import join, basename, dirname
from utils.util import dict2namespace, load_yaml

from pipeline.pipeline import TeleImagePipeline
from PIL import Image
import json



class TeleImage:
    def __init__(self, yaml_file):
        self.current_dir    = dirname(__file__)
        args                = load_yaml(join(self.current_dir, 'configs', yaml_file))            # 配置文件
        self.args           = dict2namespace({**args})
        if self.args.preprocess.enable_preprocess == True:
            if self.args.preprocess.use_cilent == True:
                from preprocess.preprocess_cilent import TeleReasoningCilent
                self.preprocessor = TeleReasoningCilent(self.args.preprocess, device=None)
            else:
                from preprocess.preprocess import TeleReasoning
                self.preprocessor = TeleReasoning(self.args.preprocess)

        lora_model_paths = vars(self.args.tele_image_pipeline.lora_model_paths)

        self.pipeline = TeleImagePipeline(self.args.tele_image_pipeline, device=self.args.device.edit_config, lora_model_paths=lora_model_paths)
        

    def run(
        self, 
        prompt, 
        aspect_ratios=None, 
        seed=None, 
        batch_size=1, 
        num_inference_steps=None, 
        true_cfg_scale=None, 
        style=None,
        quality="highres"
        ):
        
        if self.args.preprocess.enable_preprocess == True:
            prompt_processed = self.preprocessor.run(prompt, style)
        else:
            prompt_processed = prompt
            
        output_images = self.pipeline.run(
            prompt_processed, 
            aspect_ratios = aspect_ratios if aspect_ratios is not None else self.args.tele_image_pipeline.aspect_ratios,
            seed = seed if seed is not None else self.tele_image_pipeline.args.seed, 
            batch_size = batch_size, 
            num_inference_steps = num_inference_steps if num_inference_steps is not None else self.args.tele_image_pipeline.num_inference_steps, 
            true_cfg_scale = true_cfg_scale if true_cfg_scale is not None else self.args.true_cfg_scale,
            quality = quality
            )
        return prompt_processed, output_images
        


if __name__ == '__main__':
    
    # === 输入 ===
    prompt = "一个可爱的小猫"
    use_cilent = True
    

    # === 输出目录 ===
    output_dir = '../results/test'
    os.makedirs(output_dir, exist_ok=True)


    # === 超参数 （若置为None，若yaml中有则按yaml中默认）===
    batch_size = 1
    aspect_ratios = "1:1"  # 支持的分辨率： "1:1",  "9:16", "16:9", "3:2", "2:3" 
    seed = 42
    num_inference_steps = 8
    true_cfg_scale = 1.0
    style = None # style置为None，不做任何多余操作
    quality = "lowres" # 支持的分辨率： "highres", "medres", "lowres"
    

    # === 初始化 ===
    if use_cilent:
        yaml_file = 'config_cilent.yaml'
    else:
        yaml_file = 'config.yaml'
    tele_image = TeleImage(yaml_file)



    # === 运行 ===
    new_prompt, output_images = tele_image.run(
        prompt, 
        aspect_ratios=aspect_ratios, 
        seed=seed, 
        batch_size=batch_size, 
        num_inference_steps=num_inference_steps, 
        true_cfg_scale=true_cfg_scale,
        style=style,
        quality = quality
    )

    # === 保存图像 ===
    os.makedirs(output_dir, exist_ok=True)
    for i, output_image in enumerate(output_images):
        output_path = join(output_dir, f'{str(i)}.jpg')
        output_image.save(
            output_path, lossless=True
        )
        print(f"Save image_{str(i)} to {output_path}")