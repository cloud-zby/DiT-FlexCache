import io
import os
# gpu_idx = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

from PIL import Image
from os.path import join, basename, dirname

from utils.util import dict2namespace, load_yaml
from utils.file_reading import read_list_from_txt
from utils.file_operating import base64_to_image, image_to_base64
from glob import glob
from tqdm import tqdm
import os
import time
from datetime import datetime
import numpy as np

from preprocess.preprocess import TeleReasoning
from pipeline.pipeline import TeleEditPipeline
from postprocess.postprocess import TeleEditPostProcess


class TeleEdit:
    def __init__(self, translate_only = False, debug = False):
        self.current_dir    = dirname(__file__)
        args                = load_yaml(join(self.current_dir, 'configs', 'config.yaml'))            # 配置文件
        self.args           = dict2namespace({**args})
        self.debug = debug
        
        self.preprocessor = TeleReasoning(self.args)
        self.pipeline = TeleEditPipeline(self.args)
        self.postprocessor = TeleEditPostProcess(self.args)


    def run(self, image, mask, prompt='狗', seed=-1):
        image, mask = base64_to_image(image), base64_to_image(mask)
        operation, prompt_en, image_pre, mask_pre = self.preprocessor.run(prompt, image, mask)
        image_out = self.pipeline.run(prompt_en, image_pre, mask_pre, operation, seed=seed)
        image_grid, image_out_resize = self.postprocessor.concatenate_image(image, mask_pre, image_out, prompt)
        return image_grid, image_out_resize        


if __name__ == '__main__':

    # # *************************** #
    # #          调用测试            #
    # # **************************  # 
    
    # teleedit = TeleEdit()
    # CUR_DIR = dirname(dirname(__file__))
    # TeleEditBench = join(CUR_DIR, 'benchmarks')

    # TEST_DATE = datetime.now().strftime("%Y%m%d%H%M%S")
    # TEST_DATE = '20250507102714'
    
    # all_test_imgs = glob(f'{TeleEditBench}/**/*input.png', recursive=True) + \
    #     glob(f'{TeleEditBench}/**/*input.jpg', recursive=True)
    
    # all_test_imgs = sorted(all_test_imgs)

    # add_save_dir = join(CUR_DIR, 'results', TEST_DATE, 'add')
    # remove_save_dir = join(CUR_DIR, 'results', TEST_DATE, 'remove')
    # replace_save_dir = join(CUR_DIR, 'results', TEST_DATE, 'replace')

    # os.makedirs(add_save_dir, exist_ok=True)
    # os.makedirs(remove_save_dir, exist_ok=True)
    # os.makedirs(replace_save_dir, exist_ok=True)

    # cnt, total_time = 0, 0
    # for test_img in tqdm(all_test_imgs):
    #     if 'add' in test_img:
    #         save_path_grid = join(add_save_dir, basename(test_img).replace('input', 'output_grid'))
    #         save_path = join(add_save_dir, basename(test_img).replace('input', 'output'))
    #     elif 'remove' in test_img:
    #         save_path_grid = join(remove_save_dir, basename(test_img).replace('input', 'output_grid'))
    #         save_path = join(remove_save_dir, basename(test_img).replace('input', 'output'))
    #     elif 'replace' in test_img:
    #         save_path_grid = join(replace_save_dir, basename(test_img).replace('input', 'output_grid'))
    #         save_path = join(replace_save_dir, basename(test_img).replace('input', 'output'))

    #     if os.path.exists(save_path_grid):
    #         continue

    #     if os.path.exists(test_img.replace('input', 'mask')):
    #         mask_path = test_img.replace('input', 'mask')
    #     elif os.path.exists(test_img.replace('input', 'mask1')):
    #         mask_path = test_img.replace('input', 'mask1')
    #     object_path = join(dirname(test_img), 'object.txt')
    #     if os.path.exists(object_path):
    #         object_name = read_list_from_txt(object_path)[0]
    #     else:
    #         object_name = None

    #     image = image_to_base64(Image.open(test_img))
    #     mask = image_to_base64(Image.open(mask_path))

    #     start_time = time.time()
    #     image_grid, image_out_resize = teleedit.run(image, mask, prompt=object_name, seed=0)
    #     cnt += 1
    #     if cnt > 2:
    #         total_time += time.time() - start_time

    #     image_grid, image_out_resize = base64_to_image(image_grid), base64_to_image(image_out_resize)
    #     image_grid.save(save_path_grid)
    #     # image_out_resize.save(save_path)
    
    # print(f'average time is {total_time/(cnt-1):.3f}s')

    # # *************************** #
    # #          指令测试            #
    # # **************************  # 
    
    teleedit = TeleEdit()
    CUR_DIR = dirname(dirname(__file__))
    # TeleEditBench = join(CUR_DIR, 'benchmarks')
    TeleEditBench = '/nvfile-heatstorage/cv_banc/zangxh/projects/Tool-codes/Benchmarks-Datas/TeleEdit'

    TEST_DATE = datetime.now().strftime("%Y%m%d%H%M%S") + '_object'
    # TEST_DATE = '20250513003041_object'
    
    all_test_imgs = glob(f'{TeleEditBench}/**/*input.png', recursive=True) + \
        glob(f'{TeleEditBench}/**/*input.jpg', recursive=True)
    
    all_test_imgs = sorted(all_test_imgs)

    add_save_dir = join(CUR_DIR, 'results', TEST_DATE, 'add')
    remove_save_dir = join(CUR_DIR, 'results', TEST_DATE, 'remove')
    replace_save_dir = join(CUR_DIR, 'results', TEST_DATE, 'replace')

    os.makedirs(add_save_dir, exist_ok=True)
    os.makedirs(remove_save_dir, exist_ok=True)
    os.makedirs(replace_save_dir, exist_ok=True)

    cnt, total_time = 0, 0
    for test_img in tqdm(all_test_imgs):
        if 'add' in test_img:
            save_path_grid = join(add_save_dir, basename(test_img).replace('input', 'output_grid'))
            save_path = join(add_save_dir, basename(test_img).replace('input', 'output'))
        elif 'remove' in test_img:
            save_path_grid = join(remove_save_dir, basename(test_img).replace('input', 'output_grid'))
            save_path = join(remove_save_dir, basename(test_img).replace('input', 'output'))
        elif 'replace' in test_img:
            save_path_grid = join(replace_save_dir, basename(test_img).replace('input', 'output_grid'))
            save_path = join(replace_save_dir, basename(test_img).replace('input', 'output'))

        if os.path.exists(save_path_grid):
            continue

        if os.path.exists(test_img.replace('input', 'mask')):
            mask_path = test_img.replace('input', 'mask')
        elif os.path.exists(test_img.replace('input', 'mask1')):
            mask_path = test_img.replace('input', 'mask1')

        object_path = join(dirname(test_img), 'object.txt')
        if os.path.exists(object_path):         # 如果存在object.txt
            object_name = read_list_from_txt(object_path)[0]
            if object_name == None:
                continue
            if 'remove' in test_img:
                object_name = None
        else:
            object_name = None
            if 'remove' not in test_img:
                continue

        image = image_to_base64(Image.open(test_img))
        mask = image_to_base64(Image.open(mask_path))

        start_time = time.time()
        image_grid, image_out_resize = teleedit.run(image, mask, prompt=object_name, seed=0)
        cnt += 1
        if cnt > 2:
            total_time += time.time() - start_time

        image_grid, image_out_resize = base64_to_image(image_grid), base64_to_image(image_out_resize)
        image_grid.save(save_path_grid)
        # image_out_resize.save(save_path)
    
    print(f'average time is {total_time/(cnt-1):.3f}s')
    exit()
