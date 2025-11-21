import argparse
import csv
import yaml
import random
import numpy as np
import torch
import json
import os
import pickle
import jsonlines

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value

        setattr(namespace, key, new_value)
    return namespace

def load_csv(csv_file):       # 读取 csv 文件到内存中缓存起来
    csv_dicts = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        cache = dict(reader)
    first_key = next(iter(cache))
    if '\ufeff' in first_key:       # 去除可能出现的首字符异常
        csv_dicts = {first_key[1:]: cache.pop(first_key)}
    csv_dicts.update(cache)
    return csv_dicts

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_dicts = yaml.safe_load(f)
    return yaml_dicts

def load_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        txt_content = f.readlines()
    return [i.strip() for i in txt_content]

def write_json(prompt_history, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompt_history, f, ensure_ascii=False)

def read_jsonlines(path):
    data = []
    if os.path.exists(path):
        with jsonlines.open(path) as f:
            for i in f:
                data.append(i)
    return data

def read_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    return data

def write_log(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def write_pkl(data, path):
    pickle.dump(data, open(path, 'wb'))

def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def creat_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def sample_yingshe_dict(yingshe, num_samples):

    # 每个列表采样num_samples个
    def random_sample_from_list(data_list, num_samples=num_samples):
        return random.sample(data_list, min(num_samples, len(data_list)))

    prompts = []                                        # 由于yingshe采用字典形式，下述并不能固定类别
    prompts += random_sample_from_list(list(yingshe)[2616:2795])        # 电信词汇      0:179
    prompts += random_sample_from_list(list(yingshe)[2342:2459])        # 特殊翻译      179:297
    prompts += random_sample_from_list(list(yingshe)[4795:5076])        # 古诗旧        297:578
    prompts += random_sample_from_list(list(yingshe)[1386:1482])        # 中国成语      578:674
    prompts += random_sample_from_list(list(yingshe)[2155:2342])        # 中国歇后语    674:861
    prompts += random_sample_from_list(list(yingshe)[1482:1630])        # 中国人        861:1009
    prompts += random_sample_from_list(list(yingshe)[1630:2155])        # 中国地点      1009:1534
    prompts += random_sample_from_list(list(yingshe)[:1386])            # 中国菜        1534:2920
    prompts += random_sample_from_list(list(yingshe)[2459:2616])        # 外国菜        2920:3077
    prompts += random_sample_from_list(list(yingshe)[2795:4795])        # 古诗2000      3077:5077
    prompts += random_sample_from_list(list(yingshe)[5076:])            # 古诗360       5077:5437

    return prompts


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)