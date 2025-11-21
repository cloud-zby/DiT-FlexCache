from teleai_pipe.models.rift_hdv3 import IFNet
import torch
from teleai_pipe.common.model_manager import ModelManager
from teleai_pipe.pipelines.base_pipeline import BasePipeline
from typing import List
from PIL import Image
import numpy as np


class RIFEVFIPipeline(BasePipeline):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.model: IFNet = None
        self.model_names = ["vfi_model"]

    def fetch_models(self, model_manager: ModelManager):
        self.model = model_manager.fetch_model("vfi_model")

    @classmethod
    def from_model_manager(
        cls,
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        compile=False,
    ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = cls(
            device=device,
            torch_dtype=torch_dtype,
        )
        pipe.fetch_models(model_manager)
        if compile:
            cls.model = torch.compile(cls.model)
        return pipe

    @torch.no_grad()
    def __call__(self, input_video: List[Image.Image], base_fps: int, target_fps: int):
        src_tensor_list = [
            torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0
            for image in input_video
        ]
        src_tensor = torch.concat(src_tensor_list, dim=0)
        with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
            result_tensor = self.model.interpolate_frames(
                src_tensor, base_fps, target_fps, device=self.device
            )
        frames = (
            ((result_tensor.float()) * 255).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        result_video = [Image.fromarray(frame) for frame in frames]
        return result_video

    def __del__(self):
        del self.model
