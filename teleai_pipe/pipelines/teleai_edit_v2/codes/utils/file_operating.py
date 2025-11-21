from tqdm import tqdm
from glob import glob
import os
import tarfile
import cv2
from io import BytesIO
import numpy as np
from multiprocessing import Pool
from functools import partial
import base64
from PIL import Image


def pic_compress(pic_path, output_dir, target_size=199, quality=90, step=5, pic_type='.jpg'):
    # 读取图片bytes
    with open(pic_path, 'rb') as f:
        pic_byte = f.read()

    img_np = np.frombuffer(pic_byte, np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)

    current_size = len(pic_byte) / 1024
    # print("图片压缩前的大小为(KB):", current_size)
    while current_size > target_size:
        pic_byte = cv2.imencode(pic_type, img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
        if quality - step < 0:
            break
        quality -= step
        current_size = len(pic_byte) / 1024

    # 保存图片
    out_path = os.path.join(output_dir, os.path.basename(pic_path).replace('.png', '.jpg'))
    with open(out_path, 'wb') as f:
        f.write(BytesIO(pic_byte).getvalue())

    # print("图片压缩后的大小为(KB):", len(pic_byte) / 1024)
    return True

def pic_compress_mp(input_dir, output_dir):
    
    imgs = sorted(glob(os.path.join(input_dir, '*.png'), recursive=True))
    
    check_image_func = partial(pic_compress, output_dir=output_dir, target_size=90)
    with Pool(processes=32) as pool:
        list(tqdm(pool.imap(check_image_func, imgs), total=len(imgs))) 


# 分批压缩文件夹中的内容
def compress_folder_to_gzip(folder_to_compress, batch_size=5000):
    
    # 压缩一组文件/文件夹为一个 tar.gz 文件
    def compress_batch_to_gzip(batch, output_tar_gz):
        with tarfile.open(output_tar_gz, "w:gz") as tar:
            for item in batch:
                tar.add(item, arcname=os.path.basename(item))
        print(f"已成功压缩 {len(batch)} 个文件/文件夹到 {output_tar_gz}")
        
    
    output_dir = os.path.dirname(folder_to_compress)
    
    # 获取文件夹中的所有文件和子文件夹
    items = [os.path.join(folder_to_compress, item) for item in os.listdir(folder_to_compress)]
    
    # 将文件/文件夹分批处理
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # 逐批压缩
    for i, batch in enumerate(batches):
        output_tar_gz = os.path.join(output_dir, f"{os.path.basename(folder_to_compress)}_part{i + 1}.tar.gz")
        compress_batch_to_gzip(batch, output_tar_gz)

    print(f"{folder_to_compress} 中的所有内容已完成分批压缩")


def image_to_base64(image):
    # 将图片保存到字节流中
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 可以选择其他格式，如 JPEG
    buffer.seek(0)  # 重置文件指针
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_image(input_image):
    image_data = base64.b64decode(input_image)
    image = Image.open(BytesIO(image_data))
    # image = image.convert("RGB")
    return image

        
