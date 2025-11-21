import torch
import torch.nn as nn

# import torch.nn.functional as F
try:
    from vllm import _custom_ops as ops
except ImportError:
    print("failed to load vllm, please install vllm==0.7.3")


class Linear_fp8_per_chn(nn.Module):
    def __init__(self, original_linear: nn.Linear, data_type: torch.dtype):
        super().__init__()
        self.weight = original_linear.weight
        self.weight = nn.Parameter(self.weight.to(data_type))
        self.bias = original_linear.bias
        out_features = original_linear.out_features
        device = original_linear.weight.device
        self.weight_scale = nn.Parameter(
            torch.FloatTensor(out_features, 1).zero_().to(device)
        )

        self.is_reshaped = False

    @torch.inference_mode()
    def _reshape_params(self):
        if not self.is_reshaped:
            self.weight.transpose_(1, 0)
            self.weight = nn.Parameter(self.weight.to(torch.float8_e4m3fn))
            self.weight_scale = nn.Parameter(self.weight_scale.to(torch.float32))
            self.bias = nn.Parameter(self.bias.to(torch.float16))
            self.is_reshaped = True

    def act_quant_fp8_perchannel_sym_vllm(self, x):
        input_tensor_quant, input_tensor_scale = ops.scaled_fp8_quant(
            x, None, scale_ub=None, use_per_token_if_dynamic=True
        )
        return input_tensor_quant, input_tensor_scale

    def forward(self, x):
        assert x.dim() == 3
        # x.squeeze_(0)
        batch_num = x.shape[0]
        x = x.reshape(-1, x.shape[2])
        shape = (x.shape[0], self.weight.shape[1])
        # dtype = x.dtype
        device = x.device
        output_tensor = torch.empty(
            shape, dtype=torch.float16, device=device, requires_grad=False
        )

        input_tensor_quant, input_tensor_scale = self.act_quant_fp8_perchannel_sym_vllm(
            x
        )

        torch.ops._C.cutlass_scaled_mm(
            output_tensor,
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            self.bias,
        )
        output_tensor = output_tensor.reshape(batch_num, -1, output_tensor.shape[1])
        return output_tensor.to(torch.bfloat16)
