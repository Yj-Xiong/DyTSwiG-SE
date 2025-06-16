# Reference: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py

import torch
import torch.nn as nn

from models.mamba_ssm.modules import Mamba


# from mamba_ssm.models.mixer_seq_simple import _init_weights
# from mamba_ssm.ops.triton.layernorm import RMSNorm

# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
# def create_block(
#     d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
#     ):
#     d_state = cfg['model_cfg']['d_state'] # 16
#     d_conv = cfg['model_cfg']['d_conv'] # 4
#     expand = cfg['model_cfg']['expand'] # 4
#     norm_epsilon = cfg['model_cfg']['norm_epsilon'] # 0.00001
#
#     mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
#     norm_cls = partial(
#         nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
#     )
#     block = Block(
#             d_model,
#             mixer_cls,
#             norm_cls=norm_cls,
#             fused_add_norm=fused_add_norm,
#             residual_in_fp32=residual_in_fp32,
#             )
#     block.layer_idx = layer_idx
#     return block
class DyTanh(nn.Module):
    def __init__(self, num_features, alpha=0.5):
        super().__init__()
        self.init_alpha = alpha
        self.alpha = nn.Parameter(torch.full((1, 1, 1), self.init_alpha))  # 可学习缩放因子
        self.beta = nn.Parameter(torch.zeros(1,  1, num_features))  # 可学习偏移因子
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))  # 可学习缩放因子

    def forward(self, x):
        # 动态调整输入分布
        x = self.gamma * torch.tanh(self.alpha * x)   + self.beta
        return x


class Mamba_Block(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, layer_idx=0, rms_norm=True, fused_add_norm=False,
                 residual_in_fp32=False):
        super(Mamba_Block, self).__init__()
        self.Mamba = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )
        # self.norm = DyTanh(d_model)
        # self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        # x = self.norm(x)
        x = self.Mamba(x)
        return x
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma，初始化为全 1
        self.weight = nn.Parameter(torch.ones(1,1,dim))  # shape: (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, T, d)
        Returns:
            normalized tensor of same shape
        """
        # 计算均方根 (RMS) 值（沿特征维度 d）
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # 归一化并应用缩放
        return x / rms * self.weight

class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_block = Mamba_Block(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=in_channels,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )
        self.backward_block = Mamba_Block(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=in_channels,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )
        self.norm = RMSNorm(in_channels)
        self.norm1 = RMSNorm(in_channels)
        # self.norm = nn.LayerNorm(in_channels)
        # self.norm1 = nn.LayerNorm(in_channels)
        # self.norm = DyTanh(in_channels)
        # self.norm1 = DyTanh(in_channels)
        # self.apply(
        #     partial(
        #         _init_weights,
        #         n_layer=n_layer,
        #     )
        # )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = x_forward, x_backward

        # Forward
        # for layer in self.forward_blocks:
        #     x_forward, resi_forward = layer(x_forward, resi_forward)
        x_forward = self.forward_block(x_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward
        y_forward = self.norm(y_forward)
        # Backward
        # for layer in self.backward_blocks:
        #     x_backward, resi_backward = layer(x_backward, resi_backward)
        x_backward = self.backward_block(x_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(
            x_backward, [1])
        y_backward = self.norm1(y_backward)

        return torch.cat([y_forward, y_backward], -1)


class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.

    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """

    def __init__(self, cfg, inchannels):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = inchannels

        # Initialize Mamba blocks
        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)

        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        """
        Forward pass of the TFMamba block.

        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).

        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


