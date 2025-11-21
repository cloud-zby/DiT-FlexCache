import torch
import torch.nn.functional as F

from torch import nn
from vector_quantize_pytorch import FSQ
from torch import Tensor
from typing import List, Tuple


class nonlinearity(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(
        self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=0.2
    ):
        super(ResConv1DBlock, self).__init__()

        padding = dilation
        self.norm = norm

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(
            n_state,
            n_in,
            1,
            1,
            0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            ResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate * (depth + 1),
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# [B, F, IW] -> [B, F/(2**down_t)), OW]
class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width, depth, dilation_growth_rate, activation=activation, norm=norm
                ),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.model(x).permute(0, 2, 1)


# [B, F/(2**down_t)), OW] -> [B, F, IW]
class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.model(x).permute(0, 2, 1)


# [B, F, IW] -> # [B, F, IW]
class Conditioner(nn.Module):
    def __init__(
        self, input_emb_width, output_emb_width, output_emb_num, c_width, c_depth
    ):
        super().__init__()
        self.output_emb_num = output_emb_num
        self.output_emb_width = output_emb_num

        blocks = []
        blocks.append(nn.Linear(input_emb_width, c_width, bias=True))
        blocks.append(nn.ReLU())
        for _ in range(c_depth):
            blocks.append(nn.Linear(c_width, c_width, bias=True))
            blocks.append(nn.ReLU())
        blocks.append(
            nn.Linear(c_width, 2 * output_emb_num * output_emb_width, bias=True)
        )
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class VQCVAE(nn.Module):
    def __init__(
        self,
        num_joints,
        window_size,
        fsq_levels,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        dilation_growth_rate,
        activation,
        norm,
        c_width,
        c_depth,
        **kwargs
    ) -> None:
        super(VQCVAE, self).__init__()

        self.num_joints = num_joints
        self.output_emb_num = window_size // (2**down_t)
        self.output_emb_width = output_emb_width

        # build modules
        self.quantizer = FSQ(levels=fsq_levels)
        self.encoder = Encoder(
            input_emb_width=input_emb_width,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            input_emb_width=input_emb_width,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.conditioner = Conditioner(
            input_emb_width=input_emb_width,
            output_emb_width=output_emb_width,
            output_emb_num=self.output_emb_num,
            c_width=c_width,
            c_depth=c_depth,
        )

        self.register_buffer("mean", torch.zeros((24, 2)))
        self.register_buffer("std", torch.zeros((24, 2)))

        self.window_size = window_size
        self.down_rate = 2**down_t
        self.code_window_size = window_size // self.down_rate
        self.num_joints = num_joints

    # 根据motion长度获取编码长度
    # 特殊行为1：由于decoder需要固定长度的编码（即编码窗口的长度），因此返回的编码长度是该固定长度的倍数。
    # 特殊行为2：motion决定的编码长度不等于固定长度的倍数时，会将编码长度增加到固定长度的倍数。
    @torch.no_grad()
    def get_code_length(self, m_length: Tensor) -> Tensor:
        # 获取motion决定的编码长度
        code_length = m_length // self.down_rate
        pad_position = (m_length % self.down_rate) != 0
        code_length[pad_position] += 1

        # 将motion决定的编码长度编码长度增加到固定长度的倍数
        num_rears = code_length % self.code_window_size
        pad_position = num_rears != 0
        code_length[pad_position] = code_length[pad_position] + (
            self.code_window_size - num_rears[pad_position]
        )
        return code_length

    # 将一个list的motion变成一个batch的motion
    # 特殊行为1：为了组成batch，会将所有motion填充到list中最长motion的长度，填充模式为填充0
    @torch.no_grad()
    def get_batch(self, motion_list: List[Tensor]) -> Tuple[Tensor, List[int]]:
        # 获得最长motion的长度
        target_length = max([motion.shape[0] for motion in motion_list])

        # 组成batch
        m_length = []
        batch_motion = []
        for i, motion in enumerate(motion_list):
            m_length.append(motion.shape[0])
            padded_motion = F.pad(
                motion, (0, 0, 0, target_length - m_length[i]), mode="constant", value=0
            )
            batch_motion.append(padded_motion)
        batch_motion = torch.stack(batch_motion, dim=0)
        return batch_motion, m_length

    # 将原始格式的motion list变成可输入到encoder的motion batch
    # 原始格式的motion的shape为[长度, 关键点数， 坐标维数]，encoder需要的shape为[长度，关键点数*坐标维数]
    # 特殊行为1：为了组成batch，会将所有motion填充到list中最长motion的长度，填充模式为填充0
    @torch.no_grad()
    def input_process(self, motion_list: List[Tensor]) -> Tuple[Tensor, List]:
        n_motion_list = [(motion - self.mean) / self.std for motion in motion_list]
        n_motion_list = [
            motion.reshape(-1, 2 * self.num_joints) for motion in n_motion_list
        ]
        batch_motion, m_length = self.get_batch(n_motion_list)
        batch_motion = batch_motion.reshape(
            batch_motion.shape[0], batch_motion.shape[1], -1
        )
        batch_motion = batch_motion.to(self.mean.device)
        return batch_motion, m_length

    @torch.no_grad()
    # 将decoder返回的motion batch重新变成motion list
    # 特殊行为1：由于motion batch中可能有填充，因此将基于原始motion长度将填充部分去掉
    def output_process(self, batch_motion: Tensor, m_length) -> Tensor:
        motion_list = []
        for i in range(batch_motion.shape[0]):
            motion = batch_motion[i][: m_length[i]]
            motion = motion.reshape(-1, self.num_joints, 2)
            motion = motion * self.std + self.mean
            motion_list.append(motion)
        return motion_list

    # 将控制信号list变成可输入到conditioner的控制信号batch
    # 原始格式的控制信号的shape为[关键点数， 坐标维数]，conditioner需要的shape为[关键点数*坐标维数]
    @torch.no_grad()
    def control_input_process(self, control_list: List[Tensor]) -> Tensor:
        n_control_list = [(control - self.mean) / self.std for control in control_list]
        n_control_list = [
            control.reshape(2 * self.num_joints) for control in n_control_list
        ]
        batch_control = torch.stack(n_control_list, dim=0)
        batch_control = batch_control.to(self.mean.device)
        return batch_control

    @torch.no_grad()
    # 将一个batch的motion拆为一个batch的窗口
    # 特殊行为1：如果motion的末尾凑不出一个窗口，则会将其填充到一个窗口，填充模式为重复尾帧
    # 特殊行为2：代码中涉及的motion长度以原始长度为准，因为motion batch中的motion本身就可能包含填充
    # 特殊行为3：将额外返回一个list，记录了哪些窗口属于同一个motion，以及motion的原始长度
    def pack_windows(self, batch_motion: Tensor, m_length):
        pointer = 0
        batch_info = []
        batch_windows = []
        for i in range(batch_motion.shape[0]):
            # 将一个motion拆为多个窗口
            num_windows = m_length[i] // self.window_size
            num_rears = m_length[i] % self.window_size
            windows = [
                batch_motion[i, j * self.window_size : (j + 1) * self.window_size, :]
                for j in range(num_windows)
            ]
            if num_rears != 0:
                rear = F.pad(
                    batch_motion[i, m_length[i] - num_rears : m_length[i], :].T,
                    (0, self.window_size - num_rears),
                    mode="replicate",
                ).T
                windows.append(rear)

            # 记录这个motion的信息（窗口在窗口batch中的起始与结束索引以及motion的原始长度）
            info = (pointer, pointer + len(windows), m_length[i])
            pointer += len(windows)

            # 将motion的窗口添加到batch中
            batch_windows += windows
            batch_info.append(info)
        batch_windows = torch.stack(batch_windows, dim=0)
        return batch_windows, batch_info

    @torch.no_grad()
    # 将一个batch的窗口重组为一个batch的motion
    # 特殊行为1：为了组成batch，会将所有motion填充到最长motion的长度，填充模式为填充0
    # 特殊行为2：在填充之前，会先根据motion原始长度将窗口中的填充去除
    def unpack_windows(self, batch_windows: Tensor, batch_info):
        # 获得最长motion的长度
        target_length = max([info[2] for info in batch_info])

        # 重组motion batch
        batch_motion = []
        for start_idx, end_idx, length in batch_info:
            motion = batch_windows[start_idx:end_idx]
            motion = motion.reshape(self.window_size * (end_idx - start_idx), -1)
            motion = F.pad(
                motion[:length, :],
                (0, 0, 0, target_length - length),
                mode="constant",
                value=0,
            )
            batch_motion.append(motion)
        return torch.stack(batch_motion, dim=0)

    @torch.no_grad()
    # 将一个batch的编码序列拆为一个batch的编码窗口
    # 特殊行为1：考虑到编码序列batch中可能包含填充，因此在这里引用原始motion长度来推出编码序列的真实长度
    # 特殊行为2：将额外返回一个list，记录了哪些窗口属于同一个编码序列，以及对应motion的原始长度
    def pack_code_windows(self, batch_code: Tensor, m_length):
        # 获取编码序列的真实长度
        c_length = self.get_code_length(m_length)

        # 将编码序列拆为编码窗口
        pointer = 0
        batch_info = []
        batch_code_windows = []
        for i in range(batch_code.shape[0]):
            num_windows = c_length[i] // self.code_window_size
            code_windows = [
                batch_code[
                    i, j * self.code_window_size : (j + 1) * self.code_window_size
                ]
                for j in range(num_windows)
            ]
            info = (pointer, pointer + len(code_windows), m_length[i])
            pointer += len(code_windows)
            batch_code_windows += code_windows
            batch_info.append(info)
        batch_code_windows = torch.stack(batch_code_windows, dim=0)
        return batch_code_windows, batch_info

    @torch.no_grad()
    # 将一个batch的编码窗口重组为一个batch的编码序列
    # 特殊行为1：为了组成batch，会将所有编码序列填充到最长编码序列的长度，填充模式为填充-1
    def unpack_code_windows(self, batch_code_windows: Tensor, batch_info):
        batch_code = []
        code_length = []
        for start_idx, end_idx, _ in batch_info:
            code_windows = batch_code_windows[start_idx:end_idx]
            length = self.code_window_size * (end_idx - start_idx)
            code = code_windows.reshape(length)
            code_length.append(length)
            batch_code.append(code)
        max_code_length = max(code_length)
        batch_code = [
            F.pad(
                batch_code[i],
                (0, max_code_length - code_length[i]),
                mode="constant",
                value=-1,
            )
            for i in range(len(batch_code))
        ]
        return torch.stack(batch_code, dim=0), code_length

    @torch.no_grad()
    # 将一个batch的motion编码为一个batch的编码窗口
    def batch_encode(self, batch_motion: Tensor, m_length):
        batch_windows, batch_info = self.pack_windows(batch_motion, m_length)
        continuous = self.encoder(batch_windows.to(torch.float32))
        _, batch_code_windows = self.quantizer(continuous)
        return batch_code_windows, batch_info

    @torch.no_grad()
    # 将一个batch的编码窗口解码为一个batch的motion
    # 特殊行为1：为了组成batch，会将所有motion填充到最长motion的长度，填充模式为填充0
    def batch_decode(self, batch_code_windows: Tensor, batch_info):
        batch_latent_windows = self.quantizer.indices_to_codes(batch_code_windows)
        batch_recon_windows = self.decoder(batch_latent_windows)
        batch_recon = self.unpack_windows(batch_recon_windows, batch_info)
        return batch_recon

    @torch.no_grad()
    # 将一个motion的编码窗口链式解码为一个motion
    # 特殊行为1：可选约束首帧
    # 特殊行为2：由于编码窗口可能包含填充，因此motion中可能包含填充
    def chain_decode(self, code_windows: Tensor, control=None):
        num_windows = code_windows.shape[0]
        recon = []
        for i in range(num_windows):
            if control is not None:
                condition, _ = torch.chunk(
                    self.conditioner(control.unsqueeze(0)), 2, dim=1
                )
                condition = condition.reshape(
                    self.output_emb_num, self.output_emb_width
                )
                latent_window = self.quantizer.indices_to_codes(
                    code_windows[i].unsqueeze(0)
                )
                latent_window += condition
            else:
                latent_window = self.quantizer.indices_to_codes(
                    code_windows[i].unsqueeze(0)
                )
            recon_window = self.decoder(latent_window).squeeze(0)
            recon.append(recon_window)
            control = recon_window[-1, :]
        return torch.cat(recon, dim=0)

    @torch.no_grad()
    # 将一个batch的窗口链式解码为一个batch的motion
    # 特殊行为1：可选约束首帧
    # 特殊行为2：为了组成batch，会将所有motion填充到最长motion的长度，填充模式为填充0
    def batch_chain_decode(
        self, batch_code_windows: Tensor, batch_control: Tensor, batch_info
    ):
        batch_recon = []
        target_length = max([info[2] for info in batch_info])
        for i, info in enumerate(batch_info):
            start_idx, end_idx, length = info
            code_windows = batch_code_windows[start_idx:end_idx]
            recon = self.chain_decode(code_windows, batch_control[i])
            recon = F.pad(
                recon[:length, :],
                (0, 0, 0, target_length - length),
                mode="constant",
                value=0,
            )
            batch_recon.append(recon)
        return torch.stack(batch_recon, dim=0)

    @torch.no_grad()
    # 编码并重建一个batch的motion
    def batch_reconstruct(self, batch_motion: Tensor, m_length) -> Tensor:
        batch_code_windows, batch_info = self.batch_encode(batch_motion, m_length)
        batch_recon = self.batch_decode(batch_code_windows, batch_info)
        return batch_recon

    @torch.no_grad()
    # 编码并链式重建一个batch的motion
    def batch_chain_reconstruct(self, batch_motion: Tensor, m_length) -> Tensor:
        target_length = batch_motion.shape[1]
        batch_code_windows, batch_info = self.batch_encode(batch_motion, m_length)
        batch_recon = []
        for start_idx, end_idx, length in batch_info:
            code_windows = batch_code_windows[start_idx:end_idx]
            recon = self.chain_decode(code_windows, None)
            recon = F.pad(
                recon[:length, :],
                (0, 0, 0, target_length - length),
                mode="constant",
                value=0,
            )
            batch_recon.append(recon)
        return torch.stack(batch_recon, dim=0)
