import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers.utils import make_image_grid
from os.path import join, dirname
from utils.file_operating import image_to_base64


class TeleEditPostProcess():
    def __init__(self, args):
        self.args = args.postprocess
        self.current_dir = dirname(dirname(__file__))
        self.font_path = join(self.current_dir, self.args.font_path)

    def red_mask_add_text(self, image, mask, text=None):
        image = image.convert('RGB')
        mask = mask.resize((image.width,image.height))
        mask_np = np.array(mask.convert("RGB"))
        red = np.array(image).astype("float") * 1
        red[:, :, 0] = 180.0  # 红色通道
        red[:, :, 2] = 0      # 蓝色通道
        red[:, :, 1] = 0      # 绿色通道
        result_m = np.array(image)
        result_m = Image.fromarray((
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0) \
             + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )

        # 如果需要添加文本
        if text:
            draw = ImageDraw.Draw(result_m)

            font = ImageFont.truetype(self.font_path, 24)
            
            # 计算mask的非零区域中心
            mask_np_L = np.array(mask.convert("L"))
            y_indices, x_indices = np.where(mask_np_L > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                
                # 计算文本大小
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 文本位置：mask中心上方
                x = center_x - text_width // 2
                y = center_y - text_height - 10  # 在mask上方10像素
                
                # 确保文本不会超出图像边界
                x = max(0, min(x, image.width - text_width))
                y = max(0, min(y, image.height - text_height))
                
                # 绘制文本（白色带黑色描边效果）
                for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                    draw.text((x+dx, y+dy), text, font=font, fill="black")
                draw.text((x, y), text, font=font, fill="white")

        return result_m

    
    def resize_image(self, input, mask, image_out):
        input = input.convert('RGB')
        max_side = max(input.width, input.height)

        if max_side > self.args.max_length:
            ratio = self.args.max_length / max_side
            new_width, new_height = ratio * input.width, ratio * input.height
        else:
            new_width, new_height = input.width, input.height

        W = int(new_width - new_width % 16)
        H = int(new_height - new_height % 16)
        input_resized = input.resize((W,H))
        mask_resized = mask.resize((W,H))
        image_out_resize = image_out.resize((W,H))
        
        return input_resized, mask_resized, image_out_resize
    
    
    def concatenate_image(self, org_image, mask, image_out, prompt):
        org_image_resize, mask_out_resize, image_out_resize = self.resize_image(org_image, mask, image_out) 
        image_out = image_to_base64(image_out_resize)
        if self.args.demo:
            red_mask_img = self.red_mask_add_text(org_image_resize, mask_out_resize, prompt)
            image_grid = make_image_grid(
                [
                    red_mask_img,
                    image_out_resize,
                ],
                rows=1, cols=2, resize=None
            )
            image_grid = image_to_base64(image_grid)
        else:
            image_grid = None
        return image_grid, image_out
