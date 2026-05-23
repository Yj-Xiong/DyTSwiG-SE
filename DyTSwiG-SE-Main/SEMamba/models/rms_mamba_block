# Reference: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange
from SEMamba.modules.mamba_simple import Mamba


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
def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size() #[B,C,H,W]
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group, #[B,4,C/4,H,W]
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()  #[B,C/4,4,H,W]
    # reshape into orignal
    x = x.view(batch_size, channels, height, width) #[B,C,H,W]
    return x
class DyTanh(nn.Module):
    def __init__(self, num_features, alpha=1.):
        super().__init__()
        self.init_alpha = alpha
        self.alpha = nn.Parameter(torch.full((1, 1, 1), self.init_alpha))  # 可学习缩放因子
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))  # 可学习偏移因子
        self.gamma = nn.Parameter(torch.ones(1, 1, 1))  # 可学习缩放因子
        self.beta1 = nn.Parameter(torch.zeros(1, 1, num_features))  # 可学习偏移因子

    def forward(self, x):
        # 动态调整输入分布
        x = self.gamma * torch.tanh(self.alpha * x + self.beta1) + self.beta
        return x
class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False,device='cuda:0'):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.

        """
        super().__init__()
        self.device = device
        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.weight = nn.Parameter(torch.Tensor(dim).to(torch.float32)).to(self.device)
        nn.init.ones_(self.weight)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32)).to(self.device)
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        b, c ,t,f = input.shape
        input = input.view(b, t*f, c)
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.weight
        if self.bias:
            output = output + self.beta
        output = output.view(b, c, t, f)
        return output
class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))  #α. (in_features, 1) For each feature of the data, having a separate slope parameter
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x) #First scale the input using a learnable slope paramet
class RMS_SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True,
                 fre: int = 201,
                 ):
        super().__init__()

        self.gn = RMSGroupNorm(num_groups=4, dim=oup_channels,eps=1e-5,device='cuda:1')
        self.gate_treshold = gate_treshold
        self.lsigmoid = LearnableSigmoid_2d(fre, beta=2.0) # v1+fe 1.23


    def forward(self, x):
        gn_x = self.gn(x)   # x * self.weight + self.bias
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        x1 = gn_x * w_gamma
        reweigts = self.lsigmoid(x1.permute(0,1,3,2)).permute(0,1,3,2)  #W weight
        f_reweigts = torch.flip(reweigts, [1])  # F operation
        x_1 = reweigts * x  #X1
        x_2 = f_reweigts * x  #X2
        y2 = shuffle_channels(x_2, 4) #S operation
        y = x_1 + y2

        return y

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma，初始化为全 1
        self.weight = nn.Parameter(torch.ones(1, 1, dim))  # shape: (dim,)

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


class Mamba_Block(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, layer_idx=0, rms_norm=True, fused_add_norm=False,
                 residual_in_fp32=False):
        super(Mamba_Block, self).__init__()
        self.Mamba = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=64,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )

        self.norm = RMSNorm(d_model, eps=1e-6)

    def forward(self, x):

        x = self.Mamba(x)
        x = self.norm(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_block = Mamba_Block(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=64,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )
        self.backward_block = Mamba_Block(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=64,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )


    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = x_forward, x_backward

        # Forward
        # for layer in self.forward_blocks:
        #     x_forward, resi_forward = layer(x_forward, resi_forward)
        x_forward = self.forward_block(x_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward
       # y_forward = self.norm1(y_forward)
        # Backward
        # for layer in self.backward_blocks:
        #     x_backward, resi_backward = layer(x_backward, resi_backward)
        x_backward = self.backward_block(x_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(
            x_backward, [1])

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

    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']

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
