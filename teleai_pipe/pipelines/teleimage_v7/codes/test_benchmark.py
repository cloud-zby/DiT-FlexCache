import os
from os.path import join, basename, dirname
from utils.util import dict2namespace, load_yaml
from preprocess.preprocess import TeleReasoning
from pipeline.pipeline import TeleImagePipeline
from PIL import Image
import json

def read_list_from_txt(path):
    data = open(path).readlines()
    data = [i.strip() for i in data]
    return data


class TeleImage:
    def __init__(self):
        self.current_dir    = dirname(__file__)
        args                = load_yaml(join(self.current_dir, 'configs', 'config.yaml'))            # 配置文件
        self.args           = dict2namespace({**args})
        # self.preprocessor = TeleReasoning(self.args.preprocess, device=self.args.device.trans_config)

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
        style=None
        ):
        prompt_processed = prompt
        # prompt_processed = self.preprocessor.run(prompt, style)
        output_images = self.pipeline.run(
            prompt_processed, 
            aspect_ratios = aspect_ratios if aspect_ratios is not None else self.args.tele_image_pipeline.aspect_ratios,
            seed = seed if seed is not None else self.tele_image_pipeline.args.seed, 
            batch_size = batch_size, 
            num_inference_steps = num_inference_steps if num_inference_steps is not None else self.args.tele_image_pipeline.num_inference_steps, 
            true_cfg_scale = true_cfg_scale if true_cfg_scale is not None else self.args.true_cfg_scale
            )
        return prompt_processed, output_images
        


if __name__ == '__main__':
    
    # === 输入 ===
    bench_paths = {
        # "EvalMuse-40K": "/gemini-1/space/cv/zangxh/projects/Tool-codes/Benchmarks-Prompts/EvalMuse-40K",
        "T2I-CompBench" : "/nvfile-heatstorage/AIGC_H100/cv_banc/zangxh/projects/Tool-codes/Benchmarks-Prompts/T2I-CompBench/T2I-CompBench_dataset",
        "HRS-Bench"     : "/nvfile-heatstorage/AIGC_H100/cv_banc/zangxh/projects/Tool-codes/Benchmarks-Prompts/HRS-Bench",
    } 
    

    # === 输出目录 ===
    output_dir = '../results/test'
    os.makedirs(output_dir, exist_ok=True)


    # === 超参数 （若置为None，若yaml中有则按yaml中默认）===
    batch_size = 1
    aspect_ratios = "1:1"  # 支持的分辨率： "1:1",  "9:16", "16:9", "4:3", "3:4", "3:2", "2:3" 
    seed = 42
    num_inference_steps = 8
    true_cfg_scale = 1.0
    style = None # style置为None，不做任何多余操作

    

    # === 初始化 ===
    tele_image = TeleImage()

    test_date = '20240813'
    for bench in ["HRS-Bench", "T2I-CompBench"]:
        if bench == 'HRS-Bench':
            # for metric in ['spatial']:
            for metric in ['colors', 'spatial', 'size']:
                prompts = read_list_from_txt(os.path.join(bench_paths[bench], f'{metric}_composition_prompts.txt'))
                idx_infer = 0
                rst_path = os.path.join('../results', test_date)
                save_path = os.path.join(rst_path, bench, f'{metric}_{idx_infer}', 'samples')
                print(f'============================================\n{save_path}\n============================================')
                os.makedirs(save_path, exist_ok=True)
                for idx, prompt in enumerate(prompts):
                    output_path = join(save_path, '{:05d}_{}.jpg'.format(idx, prompts[:20]))
                    if os.path.exists(output_path):
                        print(f"Skip image_{str(i)} to {output_path}")
                        continue
                    # === 运行 ===
                    new_prompt, output_images = tele_image.run(
                        prompt, 
                        aspect_ratios=aspect_ratios, 
                        seed=seed, 
                        batch_size=batch_size, 
                        num_inference_steps=num_inference_steps, 
                        true_cfg_scale=true_cfg_scale,
                        style=style,
                    )

                    # === 保存图像 ===
                    for i, output_image in enumerate(output_images):
                        output_path = join(save_path, '{:05d}_{}.jpg'.format(idx, new_prompt[:20]))
                        output_image.save(
                            output_path, lossless=True
                        )
                        print(f"Save image_{str(i)} to {output_path}")

                    print(new_prompt)

        # if bench == 'T2I-CompBench':
        #     for metric in ['color', 'shape', 'texture', 'spatial', 'non-spatial', 'complex']:
        #         prompts = read_list_from_txt(os.path.join(bench_paths[bench], f'{metric}_val.txt'))
                # rst_path = os.path.join('../results', test_date)
                # idx_infer = 0
        #         save_path = os.path.join(rst_path, bench, f'{metric}_{idx_infer}', 'samples')
        #         print(f'============================================\n{save_path}\n============================================')
        #         os.makedirs(save_path, exist_ok=True)
        #         for idx, prompt in enumerate(prompts):
        #             # === 运行 ===
        #             new_prompt, output_images = tele_image.run(
        #                 prompt, 
        #                 aspect_ratios=aspect_ratios, 
        #                 seed=seed, 
        #                 batch_size=batch_size, 
        #                 num_inference_steps=num_inference_steps, 
        #                 true_cfg_scale=true_cfg_scale,
        #                 style=style,
        #             )

        #             # === 保存图像 ===
        #             for i, output_image in enumerate(output_images):
        #                 output_path = join(save_path, '{:05d}_{}.jpg'.format(idx, new_prompt[:20]))
        #                 output_image.save(
        #                     output_path, lossless=True
        #                 )
        #                 print(f"Save image_{str(i)} to {output_path}")

        #             print(new_prompt)