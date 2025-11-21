import torch
import torch.nn as nn
from typing import Optional


def to_pin_memory(module):
    for name, param in module._parameters.items():
        if param is not None and param.device.type == "cpu":
            module._parameters[name] = param.pin_memory()
    for name, buf in module._buffers.items():
        if buf is not None and buf.device.type == "cpu":
            module._buffers[name] = buf.pin_memory()
    for child in module._modules.values():
        if child is not None:
            to_pin_memory(child)


class ComputePhase:
    def __init__(
        self, phase_name: str, parent_block: nn.Module, required_modules: list
    ):
        self.phase_name = phase_name
        self.parent_block = parent_block
        self.required_modules = required_modules

    def to_cuda(self, non_blocking: bool = False):
        for module_name in self.required_modules:
            if hasattr(self.parent_block, module_name):
                module = getattr(self.parent_block, module_name)
                if hasattr(module, "to_cuda"):
                    module.to_cuda(non_blocking=non_blocking)
                else:
                    module.to("cuda", non_blocking=non_blocking)
        return self

    def to_cpu(self, non_blocking: bool = False):
        for module_name in self.required_modules:
            if hasattr(self.parent_block, module_name):
                module = getattr(self.parent_block, module_name)
                if hasattr(module, "to_cpu"):
                    module.to_cpu(non_blocking=non_blocking)
                else:
                    module.to("cpu", non_blocking=non_blocking)
        return self

    def to_cuda_async(self):
        return self.to_cuda(non_blocking=True)

    def to_cpu_async(self):
        return self.to_cpu(non_blocking=True)

    def calculate_size(self):
        total_size = 0
        for module_name in self.required_modules:
            if hasattr(self.parent_block, module_name):
                module = getattr(self.parent_block, module_name)

                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        total_size += param.numel() * param.element_size()
                    for buffer in module.buffers():
                        total_size += buffer.numel() * buffer.element_size()

                elif isinstance(module, nn.Parameter):
                    total_size += module.numel() * module.element_size()

                elif isinstance(module, torch.Tensor):
                    total_size += module.numel() * module.element_size()
        return total_size

    def clear(self):
        self.to_cpu()
        torch.cuda.empty_cache()


def _to_cuda_impl(self, non_blocking: bool = False):
    for name, param in self.named_parameters():
        if param is not None:
            param.data = param.data.to("cuda", non_blocking=non_blocking)

    for name, buffer in self.named_buffers():
        if buffer is not None:
            buffer.data = buffer.data.to("cuda", non_blocking=non_blocking)

    for name, module in self.named_children():
        if module is not None and hasattr(module, "to_cuda"):
            module.to_cuda(non_blocking=non_blocking)
        elif module is not None:
            module.to("cuda", non_blocking=non_blocking)

    return self


def _to_cpu_impl(self, non_blocking: bool = False):
    for name, param in self.named_parameters():
        if param is not None:
            param.data = param.data.to("cpu", non_blocking=non_blocking)

    for name, buffer in self.named_buffers():
        if buffer is not None:
            buffer.data = buffer.data.to("cpu", non_blocking=non_blocking)

    for name, module in self.named_children():
        if module is not None and hasattr(module, "to_cpu"):
            module.to_cpu(non_blocking=non_blocking)
        elif module is not None:
            module.to("cpu", non_blocking=non_blocking)

    return self


def _to_cuda_async_impl(self):
    return self.to_cuda(non_blocking=True)


def _to_cpu_async_impl(self):
    return self.to_cpu(non_blocking=True)


def _calculate_size_impl(self):
    total_size = 0
    if hasattr(self, "parameters"):
        for param in self.parameters():
            total_size += param.numel() * param.element_size()
    if hasattr(self, "buffers"):
        for buffer in self.buffers():
            total_size += buffer.numel() * buffer.element_size()
    return total_size


def _clear_impl(self):
    self.cpu()
    torch.cuda.empty_cache()


def _load_from_disk_impl(self):
    pass


def add_weight_management_methods(module: nn.Module):
    if hasattr(module, "_wm_added"):
        return module

    module.to_cuda = _to_cuda_impl.__get__(module, module.__class__)
    module.to_cpu = _to_cpu_impl.__get__(module, module.__class__)
    module.to_cuda_async = _to_cuda_async_impl.__get__(module, module.__class__)
    module.to_cpu_async = _to_cpu_async_impl.__get__(module, module.__class__)
    module.calculate_size = _calculate_size_impl.__get__(module, module.__class__)
    module.clear = _clear_impl.__get__(module, module.__class__)
    module.load_from_disk = _load_from_disk_impl.__get__(module, module.__class__)

    module._wm_added = True

    return module


def add_phase_management_to_block(
    block: nn.Module, phase_config: Optional[list] = None
):
    # 设置默认的 DiT 阶段配置
    if phase_config is None:
        phase_config = [
            ("modulation", ["modulation"]),
            ("self_attn", ["norm1", "self_attn"]),
            ("cross_attn", ["gate", "cross_attn", "norm3"]),
            ("ffn", ["norm2", "ffn", "gate"]),
        ]

    add_weight_management_methods(block)

    if not hasattr(block, "compute_phases"):
        block.compute_phases = [
            ComputePhase(phase_name, block, required_modules)
            for phase_name, required_modules in phase_config
        ]

    return block


def create_phase_config(phase_definitions):
    if isinstance(phase_definitions, dict):
        return [
            (phase_name, required_modules)
            for phase_name, required_modules in phase_definitions.items()
        ]
    else:
        return phase_definitions


# def prepare_model_for_offload(model: nn.Module, phase_config: Optional[list] = None):

#     if hasattr(model, 'blocks'):
#         for block in model.blocks:
#             add_phase_management_to_block(block, phase_config)

#     add_weight_management_methods(model)

#     return model
