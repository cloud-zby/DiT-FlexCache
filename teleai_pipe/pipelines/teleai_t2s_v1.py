from typing import Optional
import warnings

import torch
from clip.model import build_model as build_clip_model

from teleai_pipe.pipelines.base_pipeline import BasePipeline
from teleai_pipe.utils.utils import init_weights_on_device, load_state_dict
from teleai_pipe.models.teleai_t2s_model import Text2MotionTransformer


class TeleAIT2SPipeline(BasePipeline):
    def __init__(self, model: Text2MotionTransformer) -> None:
        super().__init__()
        self.model = model

    def __call__(self, prompt, control, num_frames, infer_opt):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        controls = [control] if not isinstance(control, list) else control
        all_num_frames = [num_frames] if isinstance(num_frames, int) else num_frames
        return self.model.generate(all_num_frames, controls, prompts, infer_opt)

    def normally_generate(self, prompt, num_frames, infer_opt):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        all_num_frames = [num_frames] if isinstance(num_frames, int) else num_frames
        return self.model.normally_generate(all_num_frames, prompts, infer_opt)

    def edit(self, control, motion):
        controls = [control] if not isinstance(control, list) else control
        motions = [motion] if not isinstance(motion, list) else motion
        return self.model.eidt(controls, motions)

    def concat(self, motion1, motion2):
        pass

    @classmethod
    def from_pretrained(
        cls,
        vae_path: str,
        tf_path: str,
        *,
        device: torch.device,
        torch_dtype: torch.dtype,
        vae_mean_path: Optional[str] = None,
        vae_std_path: Optional[str] = None,
        tf_opt: Optional[dict] = None,
        vae_opt: Optional[dict] = None,
    ):
        vae_ckpt = torch.load(vae_path, weights_only=True)
        vae_state_dict = vae_ckpt["state_dict"]
        for k in list(vae_state_dict.keys()):
            vae_state_dict[f"vqvae.{k}"] = vae_state_dict.pop(k)

        if vae_ckpt.get("hyper_parameters") is not None:
            vae_opt = vae_ckpt["hyper_parameters"]["vae_opt"]
            vae_state_dict["vqvae.mean"] = vae_ckpt["hyper_parameters"]["mean"]
            vae_state_dict["vqvae.std"] = vae_ckpt["hyper_parameters"]["std"]
        elif (
            vae_mean_path is not None
            and vae_std_path is not None
            and vae_opt is not None
        ):
            vae_state_dict["vqvae.mean"] = torch.load(vae_mean_path, weights_only=True)
            vae_state_dict["vqvae.std"] = torch.load(vae_std_path, weights_only=True)
        else:
            raise ValueError(
                "VAE mean and std must be provided if the model was trained with them or \nVAE config must be provided."
            )

        tf_ckpt = torch.load(tf_path, weights_only=True)
        tf_state_dict = tf_ckpt["state_dict"]
        if tf_ckpt.get("hyper_parameters") is not None:
            tf_opt = tf_ckpt["hyper_parameters"]["transformer_opt"]
        elif tf_opt is None:
            raise ValueError(
                "TeleAI T2S Masked Transformer model options must be provided."
            )
        tf_opt["clip_version"] = (
            ""  # In inference stage, we need to lazy load clip model
        )
        with init_weights_on_device():
            model = Text2MotionTransformer(tf_opt, vae_opt)
        if hasattr(model, "eval"):
            model = model.eval()

        # Load weights
        state_dict = {}
        state_dict.update(vae_state_dict)
        state_dict.update(tf_state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False, assign=True
        )

        if len(missing_keys) > 0:
            warnings.warn(f"Missing keys: {missing_keys}")

        # Load clip model
        clip_state_dict = {
            key.replace("transformer.clip_model.", ""): tf_state_dict[key]
            for key in unexpected_keys
            if key.startswith("transformer.clip_model.")
        }
        model.transformer.clip_model = build_clip_model(clip_state_dict)
        model = model.to(dtype=torch_dtype, device=device)

        # frezee all parameters.
        for param in model.parameters():
            param.requires_grad = False

        model = torch.compile(model)

        return cls(model)
