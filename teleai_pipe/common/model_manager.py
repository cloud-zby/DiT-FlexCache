import os
import torch
import json
import importlib
from typing import List, Tuple


from teleai_pipe.models.lora import get_lora_loaders
from .model_config import model_loader_configs
from teleai_pipe.utils.utils import (
    load_state_dict,
    init_weights_on_device,
    hash_state_dict_keys,
    split_state_dict_with_prefix,
)
from teleai_pipe.quantize.quantized_linear import Linear_fp8_per_chn

def load_model_from_single_file(
    state_dict, model_names, model_classes, model_resource, torch_dtype, device
):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(
                f"        This model is initialized with extra kwargs: {extra_kwargs}"
            )
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}
        torch_dtype = (
            torch.float32
            if extra_kwargs.get("upcast_to_float32", False)
            else torch_dtype
        )
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()
        if 'wan_video_quantized_dit' in model_names:
            for block in model.blocks:
                block.self_attn.q = Linear_fp8_per_chn(block.self_attn.q, torch.float8_e4m3fn)
                block.self_attn.k = Linear_fp8_per_chn(block.self_attn.k, torch.float8_e4m3fn)
                block.self_attn.v = Linear_fp8_per_chn(block.self_attn.v, torch.float8_e4m3fn)
                block.self_attn.o = Linear_fp8_per_chn(block.self_attn.o, torch.float8_e4m3fn)

                block.cross_attn.q = Linear_fp8_per_chn(block.cross_attn.q, torch.float8_e4m3fn)
                block.cross_attn.k = Linear_fp8_per_chn(block.cross_attn.k, torch.float8_e4m3fn)
                block.cross_attn.v = Linear_fp8_per_chn(block.cross_attn.v, torch.float8_e4m3fn)
                block.cross_attn.o = Linear_fp8_per_chn(block.cross_attn.o, torch.float8_e4m3fn)

                block.ffn[0] = Linear_fp8_per_chn(block.ffn[0], torch.float8_e4m3fn)
                block.ffn[2] = Linear_fp8_per_chn(block.ffn[2], torch.float8_e4m3fn)

        model.load_state_dict(model_state_dict, assign=True)
        if 'wan_video_quantized_dit' in model_names:
            for block in model.blocks:
                block.self_attn.q._reshape_params()
                block.self_attn.k._reshape_params()
                block.self_attn.v._reshape_params()
                block.self_attn.o._reshape_params()

                block.cross_attn.q._reshape_params()
                block.cross_attn.k._reshape_params()
                block.cross_attn.v._reshape_params()
                block.cross_attn.o._reshape_params()

                block.ffn[0]._reshape_params()
                block.ffn[2]._reshape_params()
        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


def load_model_from_huggingface_folder(
    file_path, model_names, model_classes, torch_dtype, device
):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        if torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            model = model_class.from_pretrained(
                file_path, torch_dtype=torch_dtype
            ).eval()
        else:
            model = model_class.from_pretrained(file_path).eval().to(dtype=torch_dtype)
        if torch_dtype == torch.float16 and hasattr(model, "half"):
            model = model.half()
        try:
            model = model.to(device=device)
        except Exception:
            pass
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


def load_single_patch_model_from_single_file(
    state_dict, model_name, model_class, base_model, extra_kwargs, torch_dtype, device
):
    print(
        f"    model_name: {model_name} model_class: {model_class.__name__} extra_kwargs: {extra_kwargs}"
    )
    base_state_dict = base_model.state_dict()
    base_model.to("cpu")
    del base_model
    model = model_class(**extra_kwargs)
    model.load_state_dict(base_state_dict, strict=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(dtype=torch_dtype, device=device)
    return model


def load_patch_model_from_single_file(
    state_dict,
    model_names,
    model_classes,
    extra_kwargs,
    model_manager,
    torch_dtype,
    device,
):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        while True:
            for model_id in range(len(model_manager.model)):
                base_model_name = model_manager.model_name[model_id]
                if base_model_name == model_name:
                    base_model_path = model_manager.model_path[model_id]
                    base_model = model_manager.model[model_id]
                    print(
                        f"    Adding patch model to {base_model_name} ({base_model_path})"
                    )
                    patched_model = load_single_patch_model_from_single_file(
                        state_dict,
                        model_name,
                        model_class,
                        base_model,
                        extra_kwargs,
                        torch_dtype,
                        device,
                    )
                    loaded_model_names.append(base_model_name)
                    loaded_models.append(patched_model)
                    model_manager.model.pop(model_id)
                    model_manager.model_path.pop(model_id)
                    model_manager.model_name.pop(model_id)
                    break
            else:
                break
    return loaded_model_names, loaded_models


class ModelDetectorTemplate:
    def __init__(self):
        pass

    def match(self, file_path="", state_dict={}):
        return False

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        **kwargs,
    ):
        return [], []


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(
        self,
        keys_hash,
        keys_hash_with_shape,
        model_names,
        model_classes,
        model_resource,
    ):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (
            model_names,
            model_classes,
            model_resource,
        )
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (
                model_names,
                model_classes,
                model_resource,
            )

    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True
        return False

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        **kwargs,
    ):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[
                keys_hash_with_shape
            ]
            loaded_model_names, loaded_models = load_model_from_single_file(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
            )
            return loaded_model_names, loaded_models

        # Load models without strict matching
        # (the shape of parameters may be inconsistent, and the state_dict_converter will modify the model architecture)
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            loaded_model_names, loaded_models = load_model_from_single_file(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
            )
            return loaded_model_names, loaded_models

        return loaded_model_names, loaded_models


class ModelDetectorFromSplitedSingleFile(ModelDetectorFromSingleFile):
    def __init__(self, model_loader_configs=[]):
        super().__init__(model_loader_configs)

    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                return True
        return False

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        **kwargs,
    ):
        # Split the state_dict and load from each component
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        valid_state_dict = {}
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                valid_state_dict.update(sub_state_dict)
        if super().match(file_path, valid_state_dict):
            loaded_model_names, loaded_models = super().load(
                file_path, valid_state_dict, device, torch_dtype
            )
        else:
            loaded_model_names, loaded_models = [], []
            for sub_state_dict in splited_state_dict:
                if super().match(file_path, sub_state_dict):
                    loaded_model_names_, loaded_models_ = super().load(
                        file_path, valid_state_dict, device, torch_dtype
                    )
                    loaded_model_names += loaded_model_names_
                    loaded_models += loaded_models_
        return loaded_model_names, loaded_models


class ModelDetectorFromHuggingfaceFolder:
    def __init__(self, model_loader_configs=[]):
        self.architecture_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(
        self, architecture, huggingface_lib, model_name, redirected_architecture
    ):
        self.architecture_dict[architecture] = (
            huggingface_lib,
            model_name,
            redirected_architecture,
        )

    def match(self, file_path="", state_dict={}):
        if not isinstance(file_path, str) or os.path.isfile(file_path):
            return False
        file_list = os.listdir(file_path)
        if "config.json" not in file_list:
            return False
        with open(os.path.join(file_path, "config.json"), "r") as f:
            config = json.load(f)
        if "architectures" not in config and "_class_name" not in config:
            return False
        return True

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        **kwargs,
    ):
        with open(os.path.join(file_path, "config.json"), "r") as f:
            config = json.load(f)
        loaded_model_names, loaded_models = [], []
        architectures = (
            config["architectures"]
            if "architectures" in config
            else [config["_class_name"]]
        )
        for architecture in architectures:
            huggingface_lib, model_name, redirected_architecture = (
                self.architecture_dict[architecture]
            )
            if redirected_architecture is not None:
                architecture = redirected_architecture
            model_class = importlib.import_module(huggingface_lib).__getattribute__(
                architecture
            )
            loaded_model_names_, loaded_models_ = load_model_from_huggingface_folder(
                file_path, [model_name], [model_class], torch_dtype, device
            )
            loaded_model_names += loaded_model_names_
            loaded_models += loaded_models_
        return loaded_model_names, loaded_models


class ModelDetectorFromPatchedSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(
        self, keys_hash_with_shape, model_name, model_class, extra_kwargs
    ):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (
            model_name,
            model_class,
            extra_kwargs,
        )

    def match(self, file_path="", state_dict={}):
        if not isinstance(file_path, str) or os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        return False

    def load(
        self,
        file_path="",
        state_dict={},
        device="cuda",
        torch_dtype=torch.float16,
        model_manager=None,
        **kwargs,
    ):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        loaded_model_names, loaded_models = [], []
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, extra_kwargs = self.keys_hash_with_shape_dict[
                keys_hash_with_shape
            ]
            loaded_model_names_, loaded_models_ = load_patch_model_from_single_file(
                state_dict,
                model_names,
                model_classes,
                extra_kwargs,
                model_manager,
                torch_dtype,
                device,
            )
            loaded_model_names += loaded_model_names_
            loaded_models += loaded_models_
        return loaded_model_names, loaded_models


class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        index_detector: int = -1,
        append_configs: List[Tuple] = [],
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.dit_quantized = False
        self.model = []
        self.model_path = []
        self.model_name = []
        self.model_detector = [
            ModelDetectorFromSingleFile(model_loader_configs),
            ModelDetectorFromSplitedSingleFile(model_loader_configs),
        ]
        self.update_model_detector(index_detector, append_configs)
        self.state_dict_new_module = {}
        self.state_dict_new = {}
        self.state_dict_self_attn = {} ################
        self.state_dict_new_module_high = {}
        self.state_dict_new_module_low = {}

    def update_model_detector(self, index_detector, append_configs):
        if index_detector == -1:
            return
        map_detector = {
            0: ModelDetectorFromSingleFile,
            1: ModelDetectorFromSplitedSingleFile,
            2: ModelDetectorFromHuggingfaceFolder,
            3: ModelDetectorFromPatchedSingleFile,
        }
        map_configs = {
            0: model_loader_configs,
            1: model_loader_configs,
        }
        configs = map_configs[index_detector] + append_configs
        self.model_detector[index_detector] = map_detector[index_detector](configs)

    def load_model_from_single_file(
        self,
        file_path="",
        state_dict={},
        model_names=[],
        model_classes=[],
        model_resource=None,
    ):
        print(f"Loading models from file: {file_path}")
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        model_names, models = load_model_from_single_file(
            state_dict,
            model_names,
            model_classes,
            model_resource,
            self.torch_dtype,
            self.device,
        )
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {model_names}.")

    def load_model_from_huggingface_folder(
        self, file_path="", model_names=[], model_classes=[]
    ):
        print(f"Loading models from folder: {file_path}")
        model_names, models = load_model_from_huggingface_folder(
            file_path, model_names, model_classes, self.torch_dtype, self.device
        )
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {model_names}.")

    def load_patch_model_from_single_file(
        self,
        file_path="",
        state_dict={},
        model_names=[],
        model_classes=[],
        extra_kwargs={},
    ):
        print(f"Loading patch models from file: {file_path}")
        model_names, models = load_patch_model_from_single_file(
            state_dict,
            model_names,
            model_classes,
            extra_kwargs,
            self,
            self.torch_dtype,
            self.device,
        )
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following patched models are loaded: {model_names}.")

    def load_lora(self, file_path="", state_dict={}, lora_alpha=1.0):
        if isinstance(file_path, list):
            for file_path_ in file_path:
                self.load_lora(file_path_, state_dict=state_dict, lora_alpha=lora_alpha)
        else:
            print(f"Loading LoRA models from file: {file_path}")
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            for model_name, model, model_path in zip(
                self.model_name, self.model, self.model_path
            ):
                for lora in get_lora_loaders():
                    match_results = lora.match(model, state_dict)
                    if match_results is not None:
                        print(f"    Adding LoRA to {model_name} ({model_path}).")
                        lora_prefix, model_resource = match_results
                        lora.load(
                            model,
                            state_dict,
                            lora_prefix,
                            alpha=lora_alpha,
                            model_resource=model_resource,
                        )
                        break

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype
        torch_dtype = None if self.dit_quantized else torch_dtype
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path, torch_dtype=torch_dtype))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path, torch_dtype=torch_dtype)
        else:
            state_dict = None
        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                model_names, models = model_detector.load(
                    file_path,
                    state_dict,
                    device=device,
                    torch_dtype=torch_dtype,
                    allowed_model_names=model_names,
                    model_manager=self,
                )
                for model_name, model in zip(model_names, models):
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(model_name)
                print(f"    The following models are loaded: {model_names}.")
                break
        # TODO: Unreachable??
        else:
            raise RuntimeError("    We cannot detect the model type. No models are loaded.")

    def load_models(
        self, file_path_list, model_names=None, device=None, torch_dtype=None
    ):
        for file_path in file_path_list:
            self.load_model(
                file_path, model_names, device=device, torch_dtype=torch_dtype
            )
    def load_new_module(self, file_path="", state_dict={}, lora_alpha=1.0, is_final = True, is_high = True):
        if isinstance(file_path, list):
            len_file_path = len(file_path)
            ii=1
            for file_path_ in file_path:
                is_final = False
                if ii==len_file_path:
                    is_final = True
                self.load_lora_v3(file_path_, state_dict=state_dict, lora_alpha=lora_alpha, is_final = is_final)
                ii += 1
        else:
            print(f"============> Loading wanx22 LoRA models from file: {file_path}")
            is_loaded = False
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            wx_success = False
            # extract weights of "dwpose_embedding" for added modules
            for key in state_dict.keys():
                if "pose_embedding_high" in key and is_high:
                    self.state_dict_new_module_high[key] = state_dict[key]
                if "pose_embedding_high" in key and not is_high:
                    self.state_dict_new_module_low[key] = state_dict[key]
                    
                if "blocks" in key or "text_embedding" in key or "time_embedding" in key or "time_projection" in key:
                    # print("key1:",key)
                    if is_high:
                        self.state_dict_new_module_high[key] = state_dict[key]
                    else:
                        self.state_dict_new_module_low[key] = state_dict[key]
                        
            for model_name, model, model_path in zip(self.model_name, self.model, self.model_path):
                for lora in get_lora_loaders():
                    
                    match_results = lora.match(model, state_dict)
                    
                    if match_results is not None:
                        print(f"    Adding LoRA to {model_name} ({model_path}).")
                        lora_prefix, model_resource = match_results
                        lora.load(model, state_dict, lora_prefix, alpha=lora_alpha, model_resource=model_resource)
                        is_loaded = True
                        wx_success = True
                        break
            if not wx_success:

                for key in state_dict.keys():
                    if 'lora' in key:
                        key_new = key.split("pipe.dit.")[1]
                        self.state_dict_new[key_new] = state_dict[key]
                        # self.state_dict_new_save[key] = state_dict[key]
                    
                if is_final:
                    # from ipdb import set_trace; set_trace()
                    # torch.save(self.state_dict_new_save, 'UniAnimate-Wan2.1-14B-Lora.ckpt')
                    for model_name, model, model_path in zip(self.model_name, self.model, self.model_path):
                        for lora in get_lora_loaders():
                            
                            match_results = lora.match(model, self.state_dict_new)
                            if match_results is not None:
                                print(f"    Adding LoRA to {model_name} ({model_path}).")
                                lora_prefix, model_resource = match_results
                                lora.load(model, self.state_dict_new, lora_prefix, alpha=lora_alpha, model_resource=model_resource)
                                is_loaded = True
                                wx_success = True
                                break
            
            if not is_loaded:
                print(f"    Cannot load LoRA: {file_path}")

    def fetch_model(
        self, model_name, file_path=None, require_model_path=False, index=None
    ):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(
            self.model, self.model_path, self.model_name
        ):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            # import ipdb
            # ipdb.set_trace()
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        else:
            if index is None:
                model = fetched_models[0]
                path = fetched_model_paths[0]
                print(
                    f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[0]}."
                )
            elif isinstance(index, int):
                model = fetched_models[:index]
                path = fetched_model_paths[:index]
                print(
                    f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[:index]}."
                )
            else:
                model = fetched_models
                path = fetched_model_paths
                print(
                    f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths}."
                )
        if require_model_path:
            return model, path
        else:
            return model

    def to(self, device):
        for model in self.model:
            model.to(device)

    def load_s2v_3d(self, file_path="", state_dict={}, lora_alpha=1.0, is_final=True):
        if isinstance(file_path, list):
            len_file_path = len(file_path)
            ii = 1
            for file_path_ in file_path:
                is_final = False
                if ii == len_file_path:
                    is_final = True
                self.load_s2v_3d(
                    file_path_,
                    state_dict=state_dict,
                    lora_alpha=lora_alpha,
                    is_final=is_final,
                )
                ii += 1
        else:
            print(f"Loading LoRA models from file: {file_path}")
            is_loaded = False
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            wx_success = False
            # extract weights of "dwpose_embedding" for added modules
            for key in state_dict.keys():
                if (
                    "dwpose_embedding" in key
                    or "randomref_embedding_pose" in key
                    or "inpaint_embedding" in key
                    or "learn_in_embedding" in key
                ):
                    self.state_dict_new_module[key] = state_dict[key]

                if "pipe.dit.blocks" in key:
                    self.state_dict_self_attn[key] = state_dict[key]

            for model_name, model, model_path in zip(
                self.model_name, self.model, self.model_path
            ):
                for lora in get_lora_loaders():
                    match_results = lora.match(model, state_dict)

                    if match_results is not None:
                        print(f"    Adding LoRA to {model_name} ({model_path}).")
                        lora_prefix, model_resource = match_results
                        lora.load(
                            model,
                            state_dict,
                            lora_prefix,
                            alpha=lora_alpha,
                            model_resource=model_resource,
                        )
                        is_loaded = True
                        wx_success = True
                        break
            if not wx_success:
                for key in state_dict.keys():
                    if "lora" in key:
                        key_new = key.split("pipe.dit.")[1]
                        self.state_dict_new[key_new] = state_dict[key]

                if is_final:
                    for model_name, model, model_path in zip(
                        self.model_name, self.model, self.model_path
                    ):
                        for lora in get_lora_loaders():
                            match_results = lora.match(model, self.state_dict_new)
                            if match_results is not None:
                                print(
                                    f"    Adding LoRA to {model_name} ({model_path})."
                                )
                                lora_prefix, model_resource = match_results
                                lora.load(
                                    model,
                                    self.state_dict_new,
                                    lora_prefix,
                                    alpha=lora_alpha,
                                    model_resource=model_resource,
                                )
                                is_loaded = True
                                wx_success = True

                                break

            if not is_loaded:
                print(f"    Cannot load LoRA: {file_path}")

    def load_s2v_3d(self, file_path="", state_dict={}, lora_alpha=1.0, is_final=True):
        if isinstance(file_path, list):
            len_file_path = len(file_path)
            ii = 1
            for file_path_ in file_path:
                is_final = False
                if ii == len_file_path:
                    is_final = True
                self.load_s2v_3d(
                    file_path_,
                    state_dict=state_dict,
                    lora_alpha=lora_alpha,
                    is_final=is_final,
                )
                ii += 1
        else:
            print(f"Loading LoRA models from file: {file_path}")
            is_loaded = False
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            wx_success = False
            # extract weights of "dwpose_embedding" for added modules
            for key in state_dict.keys():
                if (
                    "dwpose_embedding" in key
                    or "randomref_embedding_pose" in key
                    or "inpaint_embedding" in key
                    or "learn_in_embedding" in key
                ):
                    self.state_dict_new_module[key] = state_dict[key]

                if "pipe.dit.blocks" in key:
                    self.state_dict_self_attn[key] = state_dict[key]

            for model_name, model, model_path in zip(
                self.model_name, self.model, self.model_path
            ):
                for lora in get_lora_loaders():
                    match_results = lora.match(model, state_dict)

                    if match_results is not None:
                        print(f"    Adding LoRA to {model_name} ({model_path}).")
                        lora_prefix, model_resource = match_results
                        lora.load(
                            model,
                            state_dict,
                            lora_prefix,
                            alpha=lora_alpha,
                            model_resource=model_resource,
                        )
                        is_loaded = True
                        wx_success = True
                        break
            if not wx_success:
                for key in state_dict.keys():
                    if "lora" in key:
                        key_new = key.split("pipe.dit.")[1]
                        self.state_dict_new[key_new] = state_dict[key]

                if is_final:
                    for model_name, model, model_path in zip(
                        self.model_name, self.model, self.model_path
                    ):
                        for lora in get_lora_loaders():
                            match_results = lora.match(model, self.state_dict_new)
                            if match_results is not None:
                                print(
                                    f"    Adding LoRA to {model_name} ({model_path})."
                                )
                                lora_prefix, model_resource = match_results
                                lora.load(
                                    model,
                                    self.state_dict_new,
                                    lora_prefix,
                                    alpha=lora_alpha,
                                    model_resource=model_resource,
                                )
                                is_loaded = True
                                wx_success = True

                                break

            if not is_loaded:
                print(f"    Cannot load LoRA: {file_path}")


    def load_lora_v3(self, file_path="", state_dict={}, lora_alpha=1.0, is_final=True):
        import torch
        from safetensors import safe_open
        from loguru import logger
        from tqdm import tqdm

        def _load_lora_file(path, dtype):
            with safe_open(path, framework="pt") as f:
                return {key: f.get_tensor(key).to(dtype) for key in f.keys()}

        def _get_param_by_name(model, param_name):
            parts = param_name.split('.')
            current = model
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            return current

        @torch.no_grad()
        def _apply_lora_weights(model, lora_weights, alpha=1.0, device='cpu'):
            prefix = "diffusion_model."
            lora_pairs = {}
            for key in lora_weights.keys():
                if key.endswith("lora_down.weight") and key.startswith(prefix):
                    base_name = key[len(prefix):].replace("lora_down.weight", "weight")
                    b_key = key.replace("lora_down.weight", "lora_up.weight")
                    if b_key in lora_weights:
                        lora_pairs[base_name] = (key, b_key)
                elif key.endswith("diff_b") and key.startswith(prefix):
                    base_name = key[len(prefix):].replace("diff_b", "bias")
                    lora_pairs[base_name] = key
                elif key.endswith("diff") and key.startswith(prefix):
                    base_name = key[len(prefix):].replace("diff", "weight")
                    lora_pairs[base_name] = key

            applied_count = 0
            for name in tqdm(lora_pairs.keys(), desc="Applying LoRA"):
                try:
                    param = _get_param_by_name(model, name)
                    #print("lora ok!")
                except AttributeError:
                    logger.warning(f"Param {name} not found in model, skipping.")
                    continue
                dtype = torch.float32 if device == 'cpu' else param.dtype
                if isinstance(lora_pairs[name], tuple):
                    name_lora_A, name_lora_B = lora_pairs[name]
                    lora_A = lora_weights[name_lora_A].to(device, dtype)
                    lora_B = lora_weights[name_lora_B].to(device, dtype)
                    delta = torch.matmul(lora_B, lora_A) * alpha
                    delta = delta.to(param.device, param.dtype)
                    # 改这里
                    # print("delta0:",delta)
                    param.data.add_(delta)
                else:
                    name_lora = lora_pairs[name]
                    delta = lora_weights[name_lora].to(device, dtype) * alpha
                    delta = delta.to(param.device, param.dtype)
                    # 改这里
                    # print("delta1:",delta)
                    param.data.add_(delta)
                applied_count += 1

            logger.info(f"Applied {applied_count} LoRA weight adjustments")
            if applied_count == 0:
                logger.warning("No LoRA weights applied — check key naming.")
        # --- 支持多个路径 ---
        if isinstance(file_path, list):
            len_file_path = len(file_path)
            for idx, fp in enumerate(file_path):
                self.load_lora_v3(
                    fp, state_dict=state_dict, lora_alpha=lora_alpha, is_final=(idx == len_file_path - 1)
                )
            return

        # --- 打印路径 ---
        logger.info(f"[LoRA Loader] Loading LoRA model from: {file_path}")

        # --- 加载 state_dict ---
        if len(state_dict) == 0:
            state_dict = _load_lora_file(file_path, torch.bfloat16)

        logger.info(f"  Loaded state_dict with {len(state_dict)} keys.")
        logger.info(f"  Example keys: {list(state_dict.keys())[:5]}")

        # --- 直接应用 LoRA ---
        for model_name, model, model_path in zip(self.model_name, self.model, self.model_path):
            if model_name == "wan_video_dit":
                logger.info(f"  Applying LoRA to {model_name} ({model_path}) with alpha={lora_alpha}")
                _apply_lora_weights(model, state_dict, alpha=lora_alpha, device="cuda")
