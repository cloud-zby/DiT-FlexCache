import io
import os
gpu_idx = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
import torch
import random

import gradio as gr
import numpy as np

from PIL import Image, ImageCms, ImageFilter
import torch
from main import TeleEdit


torch.set_grad_enabled(False)
os.environ['GRADIO_TEMP_DIR'] = './tmp'


def load_model():
    global teleedit
    # teleedit = TeleEdit()    

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def predict(
    input_image,
    prompt,
    ddim_steps,
    seed,
    scale,
):

    image = input_image["background"].convert("RGB")
    mask = input_image["layers"][0]
    mask = np.asarray(mask)[:, :, -1]
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    mask = Image.fromarray(mask * 255)
    mask.save('mask.jpg')


    if seed == -1:
        seed = random.randint(1, 2147483647)
        set_seed(random.randint(1, 2147483647))
    else:
        set_seed(seed)
    
    gr.Info(str(f"Set seed = {seed}"))

    _, result = teleedit.run(image, mask, prompt=prompt, operation_id=1, seed=seed)

    mask_np = np.array(input_image["mask"].convert("RGB"))
    Info = ""
    # ===============红色Mask覆盖到原始图像================================
    red = np.array(input_image["image"]).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(input_image["image"])
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0) + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )
    # ===============背景回贴==========================
    image_np = np.array(result)
    src_img_np = np.array(input_image["image"])

    blur = ImageFilter.GaussianBlur(8)
    mask_blur = input_image["mask"].convert("RGB").filter(blur)
    mask_np1 = np.array(mask_blur) / 255.0
    res2_np = image_np * mask_np1 + src_img_np * (1 - mask_np1)
    res2 = Image.fromarray(res2_np.astype(np.uint8))

    dict_res = [input_image["image"], result_m, result, res2]

    gr.Info(Info)
    dict_out = [result]

    return dict_out, dict_res


def infer(
    input_image,
    ddim_steps,
    seed,
    scale,
    removal_prompt,
):
    return predict(input_image, 
                   removal_prompt, 
                   ddim_steps, 
                   seed,
                   scale)


with gr.Blocks(css="style.css") as demo:

    load_model()
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### TeleEdit v2")
            input_image = gr.ImageEditor(label="Source",
                                                    type="pil",
                                                    brush=gr.Brush(colors=["#000000"], default_size=30,
                                                                   color_mode="fixed"),
                                                    layers=False,
                                                    interactive=True,
                                                    )

            task = "teleEdit"
            # _ = gr.Text(label="Tips", value='This page is for testing purposes only.', interactive=False)

            run_button = gr.Button("Generate")
            with gr.Accordion("Advanced options", open=True):

                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=28, step=1, interactive=True)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=45.0, value=1.0, step=0.1, interactive=True)
                seed = gr.Slider(
                    label="Seed",
                    value=1234,
                    minimum=-1,
                    maximum=2147483647,
                    step=1,
                    randomize=False,
                    interactive=True
                )

                prompt = gr.Textbox(label="提示词", placeholder="输入描述图像的文本...")

        with gr.Column():
            gr.Markdown("### generated image")
            inpaint_result = gr.Gallery(label="Generated images", show_label=False, columns=2, height='auto')
            gr.Markdown("### 初始图象, masked_image, 模型直出, 回贴")
            gallery = gr.Gallery(label="Generated masks", show_label=False, columns=2, height='auto')

    run_button.click(
        fn=infer,
        inputs=[
            input_image,
            ddim_steps,
            seed,
            scale,
            prompt,
        ],
        outputs=[inpaint_result, gallery],
    )


if __name__ == '__main__':
    demo.queue()
    demo.launch(share=False, 
                server_name="0.0.0.0", 
                server_port=7862, 
                # debug=True,
                )