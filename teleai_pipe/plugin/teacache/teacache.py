import numpy as np
from teleai_pipe.distributed.parallel import get_sp_world_size
from teleai_pipe.distributed.utils import sync_and_check_flag
import torch
import os
import json
from loguru import logger
import torch.nn.functional as F
from typing import List


class TeaCacheCalibrator:
    def __init__(
        self,
        num_steps,
        output_path: str,
        model_type: str = "Wan2.1-I2V-14B-720P",
        prompt_id: int = 0,
    ):
        print("init teacache calibrateor")
        self.num_steps = num_steps
        self.output_path = output_path
        self.model_type = model_type
        self.prompt_id = prompt_id
        self.reset()

    def reset(self):
        self.cnt = 0
        self.mod_input_diffs: List[float] = []
        self.res_output_diffs: List[float] = []
        self.input_diffs: List[float] = []
        self.time_emb_diffs: List[float] = []
        self.previous_mod_input = None
        self.previous_residual = None
        self.previous_hidden_states = None
        self.previous_input = None
        self.previous_time_emb = None

    def check(self, img, modulated_inp, use_usp=False):
        self.cnt += 1
        self.previous_hidden_states = img.clone()
        return False

    def update(self, hidden_states):
        pass

    def _load_existing_data(self):
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    existing_data = json.load(f)
                return existing_data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Failed to load existing data from {self.output_path}: {e}"
                )
                return None
        return None

    def _update_existing_data(self, existing_data, new_item):
        if existing_data is None:
            return {"name": self.model_type, "items": [new_item]}

        if "items" not in existing_data:
            existing_data["items"] = []

        found = False
        for i, item in enumerate(existing_data["items"]):
            if item.get("prompt_id") == self.prompt_id:
                existing_data["items"][i] = new_item
                found = True
                break

        if not found:
            existing_data["items"].append(new_item)

        existing_data["name"] = self.model_type

        return existing_data

    def store(self, current_input, hidden_states, current_mod_input, current_time_emb):
        current_residual = hidden_states - self.previous_hidden_states
        if self.cnt < 2:
            self.previous_residual = current_residual.clone()
            self.previous_mod_input = current_mod_input.clone()
            self.previous_time_emb = current_time_emb.clone()
            self.previous_input = current_input
        else:
            self.previous_hidden_states = None
            l1_diff = torch.norm(current_input - self.previous_input, p=1)
            norm_pre = torch.norm(self.previous_input, p=1)
            self.input_diffs.append((l1_diff / (norm_pre + 1e-8)).item())
            self.previous_input = current_input

            l1_diff = torch.norm(current_residual - self.previous_residual, p=1)
            norm_pre = torch.norm(self.previous_residual, p=1)
            self.res_output_diffs.append((l1_diff / (norm_pre + 1e-8)).item())
            self.previous_residual = current_residual.clone()

            l1_diff = torch.norm(current_mod_input - self.previous_mod_input, p=1)
            norm_pre = torch.norm(self.previous_mod_input, p=1)
            self.mod_input_diffs.append((l1_diff / (norm_pre + 1e-8)).item())
            self.previous_mod_input = current_mod_input.clone()

            l1_diff = torch.norm(current_time_emb - self.previous_time_emb, p=1)
            norm_pre = torch.norm(self.previous_time_emb, p=1)
            self.time_emb_diffs.append((l1_diff / (norm_pre + 1e-8)).item())
            self.previous_time_emb = current_time_emb.clone()

        if self.cnt >= self.num_steps:
            current_calibrate_data = {
                "prompt_id": self.prompt_id,
                "calibrate_data": {
                    "input_diffs": self.input_diffs.copy(),
                    "res_output_diffs": self.res_output_diffs.copy(),
                    "mod_input_diffs": self.mod_input_diffs.copy(),
                    "time_emb_diffs": self.time_emb_diffs.copy(),
                },
            }

            existing_data = self._load_existing_data()

            result = self._update_existing_data(existing_data, current_calibrate_data)

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            logger.info(
                f"save teacache calibrate data to {self.output_path} (prompt_id: {self.prompt_id})"
            )
            with open(self.output_path, "w") as f:
                json.dump(result, f, indent=4)

            self.reset()


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, coefficients, start_step=0):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        self.coefficients = coefficients
        self.rescale_func = np.poly1d(self.coefficients)
        self.start_step = start_step

    def set_coefficients(self, coeff):
        self.coefficients = coeff
        self.rescale_func = np.poly1d(self.coefficients)

    def check(self, img, modulated_inp, use_usp=False):
        if self.step <= self.start_step or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            self.accumulated_rel_l1_distance += self.rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .float()
                .numpy()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if use_usp:
            should_not_calc = sync_and_check_flag(not should_calc, get_sp_world_size())
            should_calc = not should_not_calc
        if should_calc:
            self.previous_hidden_states = img.clone()
        return not should_calc

    def store(self, current_input, hidden_states, current_mod_input, current_time_emb):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states
