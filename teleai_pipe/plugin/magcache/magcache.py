import json
import os
from loguru import logger
import numpy as np
import torch.nn.functional as F


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    src_array = np.array(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


class MagCacheCalibrator:
    def __init__(self, num_steps, output_path: str):
        print("init magcache calibrateor")
        self.num_steps = num_steps
        self.output_path = output_path
        self.reset()

    def reset(self):
        self.cnt = 0
        self.accumulated_ratio = [1.0]
        self.accumulated_err = [0.0]
        self.accumulated_steps = [0]
        self.norm_ratio = []
        self.norm_std = []
        self.cos_dis = []
        self.residual_cache = None

    def check(self):
        self.cnt += 1
        return False

    def store(self, x, ori_x):
        residual_x = x - ori_x
        if self.cnt >= 2:
            norm_ratio = (
                (residual_x.norm(dim=-1) / self.residual_cache.norm(dim=-1)).mean()
            ).item()
            norm_std = (
                (residual_x.norm(dim=-1) / self.residual_cache.norm(dim=-1))
                .std()
                .item()
            )
            cos_dis = (
                (
                    1
                    - F.cosine_similarity(
                        residual_x, self.residual_cache, dim=-1, eps=1e-8
                    )
                )
                .mean()
                .item()
            )
            self.norm_ratio.append(round(norm_ratio, 5))
            self.norm_std.append(round(norm_std, 5))
            self.cos_dis.append(round(cos_dis, 5))
            print(
                f"time: {self.cnt}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}"
            )
        self.residual_cache = residual_x
        if self.cnt >= self.num_steps:
            result = dict(
                norm_ratio=self.norm_ratio, norm_std=self.norm_std, cos_dis=self.cos_dis
            )
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            logger.info(f"save magcache calibrate data to {self.output_path}")
            with open(self.output_path, "w") as f:
                json.dump(result, f, indent=4)
            self.reset()

    def update(self, x):
        pass


class MagCache:
    def __init__(
        self,
        num_inference_steps,
        thresh,
        K,
        mag_ratios,
        retention_ratio,
        init_step: int = 0,
    ):
        self.num_inference_steps = num_inference_steps
        self.init_step = init_step
        self.step = init_step
        self.thresh = thresh
        self.start_step = int(num_inference_steps * retention_ratio)
        if len(mag_ratios) != num_inference_steps:
            mag_ratios = nearest_interp(mag_ratios, target_length=num_inference_steps)
        self.mag_ratios = mag_ratios
        self.K = K
        self.reset()

    def reset(self):
        self.accumulated_err = 0.0
        self.accumulated_steps = 0
        self.accumulated_ratio = 1.0
        self.residual_cache = None

    def check(self):
        if (
            self.step <= self.start_step
            or self.step == self.num_inference_steps - 1
            or self.step == self.init_step
        ):
            should_calc = True
        else:
            cur_mag_ratio = self.mag_ratios[self.step]
            self.accumulated_ratio = (
                self.accumulated_ratio * cur_mag_ratio
            )  # magnitude ratio between current step and the cached step
            self.accumulated_steps += 1  # skip steps plus 1
            cur_skip_err = np.abs(
                1 - self.accumulated_ratio
            )  # skip error of current steps
            self.accumulated_err += cur_skip_err  # accumulated error of multiple steps
            if self.accumulated_err < self.thresh and self.accumulated_steps <= self.K:
                should_calc = False
            else:
                self.reset()
                should_calc = True
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = self.init_step
            self.reset()
        return not should_calc

    def store(self, x, ori_x):
        self.residual_cache = x - ori_x

    def update(self, x):
        return x + self.residual_cache
