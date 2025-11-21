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