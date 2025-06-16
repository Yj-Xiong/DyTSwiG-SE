# ------------------------------------------------------------------------
# Copyright (c) 2022 Murufeng. All Rights Reserved.
# ------------------------------------------------------------------------
'''
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import random
from .kaconv.kaconv.fastkanconv import FastKANConv1DLayer as KANConv1d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
# from models.modules.mamba_simple import Mamba

# class DyTanh(nn.Module):
#     def __init__(self, num_features, eps=1e-6):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.full((1, num_features, 1), 1))  # 可学习缩放因子
#         self.beta = nn.Parameter(torch.zeros(1, num_features,  1))  # 可学习偏移因子
#
#     def forward(self, x):
#         # 动态调整输入分布
#         x = torch.tanh(self.alpha * x + self.beta)
#         return x
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
class CNN_RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma，初始化为全 1
        self.weight = nn.Parameter(torch.ones(1,dim,1))  # shape: (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, T, d)
        Returns:
            normalized tensor of same shape
        """
        # 计算均方根 (RMS) 值（沿特征维度 d）
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-2, keepdim=True) + self.eps)
        # 归一化并应用缩放
        return x / rms * self.weight
class DyTanh(nn.Module):
    def __init__(self, num_features, alpha=1.):
        super().__init__()
        self.init_alpha = alpha
        self.alpha = nn.Parameter(torch.full((1, 1, 1), self.init_alpha))  # 可学习缩放因子
        self.alpha1 = nn.Parameter(torch.full((1, 1, 1), self.init_alpha))  # 可学习缩放因子
        self.beta = nn.Parameter(torch.zeros(1, num_features,  1))  # 可学习偏移因子
        self.beta1 = nn.Parameter(torch.zeros(1, num_features,  1))  # 可学习偏移因子
        # self.gamma = nn.Parameter(torch.ones(1, num_features,  1))  # 可学习缩放因子
    def forward(self, x):
        # 动态调整输入分布
        x = self.alpha1 *torch.tanh(self.alpha * x + self.beta) + self.beta1
        # x = self.gamma * torch.tanh(self.alpha * x) + self.beta
        return x
class FFN_DyTanh(nn.Module):
    def __init__(self, num_features, alpha=1.):
        super().__init__()
        self.init_alpha = alpha
        self.alpha = nn.Parameter(torch.full((1, 1, 1), self.init_alpha))  # 可学习缩放因子
        self.alpha1 = nn.Parameter(torch.full((1, 1, 1), self.init_alpha))  # 可学习缩放因子
        self.beta = nn.Parameter(torch.zeros(1,  1, num_features))  # 可学习偏移因子
        self.beta1 = nn.Parameter(torch.zeros(1, 1, num_features))  # 可学习偏移因子
        # self.gamma = nn.Parameter(torch.ones(1,  1, num_features))  # 可学习缩放因子
    def forward(self, x):
        # 动态调整输入分布
        x = self.alpha1 *torch.tanh(self.alpha * x + self.beta) + self.beta1
        # x = self.gamma * torch.tanh(self.alpha * x) + self.beta
        return x
class AffinePReLU(nn.Module):
    def __init__(self, num_channels=64, num_features=201):
        """
        初始化 Affine PReLU 层。

        参数:
            num_channels (int): 输入的通道数量。
            num_features (int): 每个通道的特征维度。
        """
        super(AffinePReLU, self).__init__()

        # 定义可学习的参数
        self.gamma = nn.Parameter(torch.ones(num_channels, num_features))  # 乘法参数 γ
        self.beta = nn.Parameter(torch.zeros(num_channels, num_features))  # 偏置参数 β
        self.alpha = nn.Parameter(torch.tensor(0.25))  # PReLU 的可学习参数 α

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (Batch, Channels, Features)

        返回:
            torch.Tensor: 经过 Affine PReLU 处理的输出张量。
        """
        # 计算 PReLU
        prelu_x = F.relu(x) + self.alpha * (x - F.relu(x))

        # 计算 Affine PReLU
        output = self.gamma * x + self.beta + prelu_x

        return output
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class SwiGLU(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # 初始化为1.0，接近标准Swish

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return (x1 * torch.sigmoid(self.beta * x1)) * x2
class ChunkBiMambaFFN(nn.Module):
    def __init__(self, d_model,ffn_expand,chunks, gpu='cuda:0',dropout=0):
        super().__init__()

        self.mid_dim = ffn_expand*d_model
        # self.MLP = MLP(d_model,ffn_expand*d_model)
        self.MLP2 = MLP(d_model, ffn_expand * d_model)
        # self.max_dim = 2*self.mid_dim
        self.Glob_FMamba = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.Glob_BMamba = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.Glob_Fnorm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5, device=gpu)
        self.Glob_Bnorm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5, device=gpu)
        self.FMamba= nn.ModuleList([Mamba(d_model=d_model//chunks,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,) for _ in range(chunks)])
        self.BMamba = nn.ModuleList([Mamba(d_model=d_model//chunks,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,) for _ in range(chunks)])
        self.linear_1 = nn.ConvTranspose1d(self.mid_dim, d_model, 1, stride=1)
        self.rearrange1 = Rearrange('b n c -> b c n')
        self.rearrange2 = Rearrange('b c n -> b n c')

        self.linear_2 = nn.ConvTranspose1d(self.mid_dim, d_model, 1, stride=1)


        self.F_norm = nn.ModuleList([RMSGroupNorm(num_groups=4, dim=d_model//chunks, eps=1e-5,device=gpu) for _ in range(chunks)])
        self.B_norm = nn.ModuleList([RMSGroupNorm(num_groups=4, dim=d_model//chunks, eps=1e-5,device=gpu) for _ in range(chunks)])

        self.rmsg_norm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5,device=gpu)
        self.post_norm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5, device=gpu)
    def forward(self, x):
        # self.Mamba.flatten_parameters()
        # x = self.MLP(x)
        x_in = x.clone()
        x_b = x.flip(1)
        x_forward = self.Glob_FMamba(x)
        x_forward = self.Glob_Fnorm(x + x_forward)#*self.fchannel_scale
        x_backforward = self.Glob_BMamba(x_b)
        x_backforward =self.Glob_Bnorm( x_b + x_backforward)#*self.bchannel_scale
        x_bp = torch.cat([x_forward, x_backforward], dim=-1)

        x = self.linear_1(self.rearrange1(x_bp)) #256

        x = self.rearrange2(x) + x_in
        x = self.rmsg_norm(x)
        x1,x2,x3,x4 = x.chunk(4, dim=2)
        x_in = x.clone()
        x_b1,x_b2,x_b3,x_b4 = x.flip(1).chunk(4, dim=2)
        x_forward1 = self.FMamba[0](x1)
        x_forward1 = self.F_norm[0](x1 + x_forward1)
        x_forward2 = self.FMamba[1](x2)
        x_forward2 = self.F_norm[1](x2 + x_forward2)
        x_forward3 = self.FMamba[2](x3)
        x_forward3 = self.F_norm[2](x3 + x_forward3)
        x_forward4 = self.FMamba[3](x4)
        x_forward4 = self.F_norm[3](x4 + x_forward4)
        x_backforward1 = self.BMamba[0](x_b1)
        x_backforward1 =self.B_norm[0]( x_b1 + x_backforward1)
        x_backforward2 = self.BMamba[1](x_b2)
        x_backforward2 =self.B_norm[1]( x_b2 + x_backforward2)
        x_backforward3 = self.BMamba[2](x_b3)
        x_backforward3 =self.B_norm[2]( x_b3 + x_backforward3)
        x_backforward4 = self.BMamba[3](x_b4)
        x_backforward4 =self.B_norm[3]( x_b4 + x_backforward4)

        x_bp = torch.cat([x_forward1, x_forward2, x_forward3, x_forward4, x_backforward1,x_backforward2, x_backforward3, x_backforward4], dim=2)

        x = self.linear_2(self.rearrange1(x_bp))  # 256
        x = self.rearrange2(x) +  x_in
        x = self.MLP2(x)
        x = self.post_norm(x)
        # x = self.rearrange2(x)
        # x = self.post_norm(x)

        return x
class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False,device='cuda:1'):
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

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32)).to(self.device)
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32)).to(self.device)
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        B, C, T = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        B, C, T = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
class CRA(nn.Module):
    def __init__(self,  in_channels, bias=True):
        super(CRA, self).__init__()
        self.pool =  nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, n = x.shape
        inp = x.clone()
        x = x.permute(0, 2, 1)
        # Ec = x.pow(2).mean(dim=-1,keepdim=True) # b, n, 1
        Ec = self.pool(x)
        x = self.fc(x) * Ec
        x = self.sigmoid(x)

        # x = x.permute(0, 2, 1) @ Ec  # b, c, 1

        return x.permute(0, 2, 1) + inp


class LayerNorm1d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class DynamicConv1d(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=2,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias

        # 1D卷积核参数 (num_groups, dim, kernel_size)
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size), requires_grad=True)

        # 1D自适应池化
        self.pool = nn.AdaptiveAvgPool1d(output_size=kernel_size)

        # === 修改点1：替换ConvModule为普通Sequential ===
        self.proj = nn.Sequential(
            # 第一层：1D卷积 + BN + GELU
            nn.Conv1d(dim, dim // reduction_ratio, kernel_size=1),
            # 第二层：1D卷积（无BN/激活）
            nn.Conv1d(dim // reduction_ratio, dim * num_groups, kernel_size=1)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, N = x.shape

        # 动态权重计算
        pooled = self.pool(x)  # (B, C, K)
        scale = self.proj(pooled).reshape(B, self.num_groups, C, self.K)  # (B, G, C, K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)  # (B, G, C, K) * (1, G, C, K)
        weight = torch.sum(weight, dim=1)  # (B, C, K)
        weight = weight.reshape(-1, 1, self.K)  # (B*C, 1, K)

        # 动态偏置计算
        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=-1, keepdim=True))  # (B, C, 1) -> (B, G*C, 1)
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)  # (B, G, C)
            bias = scale * self.bias.unsqueeze(0)  # (B, G, C) * (1, G, C)
            bias = torch.sum(bias, dim=1).flatten(0)  # (B*C,)
        else:
            bias = None

        # 1D卷积操作
        x = F.conv1d(
            x.reshape(1, -1, N),  # (1, B*C, N)
            weight=weight,  # (B*C, 1, K)
            padding=self.K // 2,
            groups=B * C,
            bias=bias
        )
        return x.reshape(B, C, N)  # 恢复原始形状


class MultiScaleDynamicConv(nn.Module):
    def __init__(self, dw_channel, kernel_sizes=[1, 1, 1], num_groups=[2, 4, 8], bias=True):
        super().__init__()
        self.dyconvs = nn.ModuleList([
            DynamicConv1d(dw_channel // 2, kernel_size=k, num_groups=g, bias=bias)
            for k, g in zip(kernel_sizes, num_groups)
        ])

        # 直接定义可学习的权重参数 (1, K, 1, 1) -> 广播到 (B, K, 1, 1)
        self.weights = nn.Parameter(torch.ones(1, len(kernel_sizes), 1, 1))

    def forward(self, x):
        # 多尺度动态卷积特征提取
        features = [conv(x) for conv in self.dyconvs]  # List[(B, C, N)]

        # 动态权重（Softmax归一化）
        weights = torch.softmax(self.weights, dim=1)  # (1, K, 1, 1) -> 广播到 (B, K, 1, 1)

        # 加权融合
        out = torch.stack(features, dim=1)  # (B, len(K), C, N)
        return (out * weights).sum(dim=1)  # (B, C, N)
class DyCWA(nn.Module):
    def __init__(self, in_channels, DW_Expand=2, drop_out_rate=0., kernel_size=3):
        super().__init__()

        dw_channel = in_channels * DW_Expand


        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.weight = nn.Parameter(torch.empty(2, in_channels, kernel_size), requires_grad=True)
        # Simplified Channel Attention
        # self.type = type
        self.dyca = nn.Sequential(nn.AdaptiveAvgPool1d(1),
            DynamicConv1d(dw_channel//2, kernel_size=1, num_groups=2, bias=True)
        )
        # self.dyca = MultiScaleDynamicConv(
        #     dw_channel=dw_channel,
        #     kernel_sizes=[1, 1, 1],  # 多尺度卷积核
        #     num_groups=[2, 4, 8]      # 分组设置
        # )

        # SimpleGate
        self.sg = SimpleGate()

        self.norm1 = DyTanh(in_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        # self.LKEFN = FeedForward(in_channels, FFN_Expand, kernel_size=kernel_size, bias=False)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        x = self.pos_embed(x) + x

        inp2 = x


        x = self.norm1(inp2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.dyca(x)

        # x = self.conv3(x)


        x = self.dropout1(x)

        x = inp2 + x * self.beta

        return x

class LKFCA_Block(nn.Module):
    def __init__(self, in_channels, DW_Expand=2, FFN_Expand=1, drop_out_rate=0., type='sca', kernel_size=31):
        super().__init__()

        dw_channel = in_channels * DW_Expand


        # 注意力


        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)



        # Simplified Channel Attention

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()


        self.norm1 = LayerNorm1d(in_channels)
        self.norm2 = LayerNorm1d(in_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.GCGFN = GCGFN(in_channels)


    def forward(self, x):


        inp2 = x


        x = self.norm1(inp2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)


        x = self.dropout1(x)

        inp3 = inp2 + x * self.beta


        x = self.GCGFN(inp3)

        x = inp3 + x * self.gamma


        return x

    # def __init__(self, in_channels, DW_Expand=2, FFN_Expand=1, drop_out_rate=0., type='sca', kernel_size=31):
    #     super().__init__()
    #
    #     dw_channel = in_channels * DW_Expand
    #     # ConvModule
    #     self.MPL1 = SimpleSwiGLU_FFN(in_channels)
    #
    #     self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
    #                            bias=True)
    #     self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
    #                            groups=dw_channel,
    #                            bias=True)
    #     self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
    #                            groups=1, bias=True)
    #
    #
    #
    #     # Simplified Channel Attention
    #     # self.type = type
    #     self.sca = nn.Sequential(
    #         nn.AdaptiveAvgPool1d(1),
    #         nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
    #                   groups=1, bias=True),
    #     )
    #
    #     # SimpleGate
    #     self.sg = SimpleGate()
    #     # self.sg = SwishGate()
    #     self.norm1 = LayerNorm1d(in_channels)
    #     # self.norm2 = LayerNorm1d(in_channels)
    #     # self.norm1 = DyTanh(in_channels)
    #
    #     self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
    #     # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
    #
    #     # self.lamda = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
    #     self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
    #     self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
    #     # self.scale = nn.Parameter(torch.zeros((1, 1, in_channels)), requires_grad=True)
    #     # self.scale2 = nn.Parameter(torch.zeros((1, 1, in_channels)), requires_grad=True)
    #     # self.LKEFN = FeedForward(in_channels, FFN_Expand, kernel_size=kernel_size, bias=False)
    #     self.GCGFN = GCGFN(in_channels)
    #     self.MPL2 = SimpleSwiGLU_FFN(in_channels)
    #     self.post_norm = nn.LayerNorm(in_channels)
    #     # self.post_norm = DyTanh(in_channels)
    #     self.Rearrange1 = Rearrange('b n c -> b c n')
    #     self.Rearrange2 = Rearrange('b c n -> b n c')
    # def forward(self, x):
    #     x = self.Rearrange1(x)
    #     x = 0.5*self.MPL1(x) + x
    #     x = self.Rearrange2(x)
    #     inp2 = x
    #
    #     x = self.norm1(inp2)
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.sg(x)
    #
    #     x = x * self.sca(x)
    #     x = self.conv3(x)
    #
    #
    #     x = self.dropout1(x)
    #
    #     inp3 = inp2 + x * self.beta
    #
    #
    #     x = self.GCGFN(inp3)
    #     x = self.Rearrange1(x)
    #     x = 0.5*self.MPL2(x) + x
    #     x = self.post_norm(x)
    #     x = self.Rearrange2(x)
    #     return x
class DyTanh_LKFCA(nn.Module):
        def __init__(self, in_channels, DW_Expand=2, FFN_Expand=1, drop_out_rate=0., type='sca', kernel_size=31):
        super().__init__()

        dw_channel = in_channels * DW_Expand
        # ConvModule
        self.MPL1 = DyTanh_FFN(in_channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)



        # Simplified Channel Attention

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            # KANConv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
            #           groups=1, bias=True)
        )
                # SimpleGate
        self.sg = SimpleGate()
        self.norm1 = DyTanh(in_channels)

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.GCGFN = DyTanh_GCGFN(in_channels)
        self.MPL2 = DyTanh_FFN(in_channels)

        # self.post_norm = nn.LayerNorm(in_channels)
        self.post_norm = FFN_DyTanh(in_channels)
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')
    def forward(self, x):
        x = self.Rearrange1(x)
        x = 0.5*self.MPL1(x) + x
        x = self.Rearrange2(x)
        inp2 = x

        x = self.norm1(inp2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)

        inp3 = inp2 + x * self.beta


        x = self.GCGFN(inp3)
        x = self.Rearrange1(x)
        x = 0.5*self.MPL2(x) + x
        x = self.post_norm(x)
        x = self.Rearrange2(x)
        return x
class RMSNorm_LKFCA(nn.Module):

    def __init__(self, in_channels, DW_Expand=2, FFN_Expand=1, drop_out_rate=0., type='sca', kernel_size=31):
        super().__init__()

        dw_channel = in_channels * DW_Expand
        # ConvModule
        self.MPL1 = DyTanh_FFN(in_channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)



        # Simplified Channel Attention

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()
        # self.sg = SwishGate()
        self.norm1 = CNN_RMSNorm(in_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.GCGFN = RMSNorm_GCGFN(in_channels)
        self.MPL2 = DyTanh_FFN(in_channels)

        self.post_norm = RMSNorm(in_channels)
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')
    def forward(self, x):
        x = self.Rearrange1(x)
        x = 0.5*self.MPL1(x) + x
        x = self.Rearrange2(x)
        inp2 = x

        x = self.norm1(inp2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)


        x = self.dropout1(x)

        inp3 = inp2 + x * self.beta


        x = self.GCGFN(inp3)
        x = self.Rearrange1(x)
        x = 0.5*self.MPL2(x) + x
        x = self.post_norm(x)
        x = self.Rearrange2(x)
        return x


class SwiGLU_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, beta=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)

        # 扩展层 (可省略，直接用w1/w2替代)
        self.expand = nn.Linear(in_dim, in_dim * 4)

        # SwiGLU 部分
        hidden_dim = hidden_dim or in_dim * 2  # 通常设为扩展后的1/2
        self.w1 = nn.Linear(in_dim * 4, hidden_dim)  # 门控分支
        self.w2 = nn.Linear(in_dim * 4, hidden_dim)  # 值分支
        self.beta = nn.Parameter(torch.tensor(beta))

        # 收缩层
        self.contract = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.expand(x)  # [B, ..., in_dim*4]

        # SwiGLU
        x1, x2 = self.w1(x), self.w2(x)  # 拆分为两个分支
        x = x1 * torch.sigmoid(self.beta * x1) * x2  # Swish(x1) * x2

        x = self.contract(x)  # [B, ..., in_dim]
        return x
class SimpleSwiGLU_FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, beta=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)

        # 扩展层 (可省略，直接用w1/w2替代)
        self.expand = nn.Linear(in_dim, in_dim * 4)

        # SwiGLU 部分
        hidden_dim = hidden_dim or in_dim * 2  # 通常设为扩展后的1/2
        self.beta = nn.Parameter(torch.tensor(beta))

        # 收缩层
        self.contract = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.expand(x)  # [B, ..., in_dim*4]

        # SwiGLU
        x1, x2 = torch.chunk(x, 2, dim=-1)  # 直接将扩展后的张量拆分为两个分支
        x = x1 * torch.sigmoid(self.beta * x1) * x2  # Swish(x1) * x2

        x = self.contract(x)  # [B, ..., in_dim]
        return x
class DyTanh_FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, beta=1.0, drop=0.):
        super().__init__()

        self.norm = FFN_DyTanh(in_dim)
        # 扩展层 (可省略，直接用w1/w2替代)
        self.expand = nn.Linear(in_dim, in_dim * 4)

        # SwiGLU 部分
        hidden_dim = hidden_dim or int(in_dim * 2)  # 通常设为扩展后的1/2
        self.beta = nn.Parameter(torch.tensor(beta))
        # 收缩层
        self.contract = nn.Linear(hidden_dim, in_dim)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.expand(x)  # [B, ..., in_dim*4]

        # SwiGLU
        x1, x2 = torch.chunk(x, 2, dim=-1)  # 直接将扩展后的张量拆分为两个分支
        x = x1 * torch.sigmoid(self.beta * x1) * x2  # Swish(x1) * x2

        x = self.contract(x)  # [B, ..., in_dim]
        return x
    # def __init__(self, in_dim, hidden_dim=None, beta=1.0):
    #     super().__init__()
    #     # self.norm = nn.LayerNorm(in_dim)
    #
    #     self.norm = RMSNorm(in_dim)
    #     # 扩展层 (可省略，直接用w1/w2替代)
    #     self.expand = nn.Linear(in_dim, in_dim * 4)
    #
    #     # SwiGLU 部分
    #     hidden_dim = hidden_dim or in_dim * 2  # 通常设为扩展后的1/2
    #     self.beta = nn.Parameter(torch.tensor(beta))
    #
    #     # 收缩层
    #     self.contract = nn.Linear(hidden_dim, in_dim)
    #
    # def forward(self, x):
    #     x = self.norm(x)
    #     x = self.expand(x)  # [B, ..., in_dim*4]
    #
    #     # SwiGLU
    #     x1, x2 = torch.chunk(x, 2, dim=-1)  # 直接将扩展后的张量拆分为两个分支
    #     x = x1 * torch.sigmoid(self.beta * x1) * x2  # Swish(x1) * x2
    #
    #     x = self.contract(x)  # [B, ..., in_dim]
    #     return x
class GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm1d(n_feats)
        # self.norm = DyTanh(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.X3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))


    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x
class DyTanh_GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        # self.norm = LayerNorm1d(n_feats)
        self.norm = DyTanh(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.X3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))


    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x
class RMSNorm_GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        # self.norm = LayerNorm1d(n_feats)
        self.norm = CNN_RMSNorm(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.X3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))


    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x
class Light_GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats
        self.n_feats = n_feats
        self.i_feats = i_feats
        self.norm = DyTanh(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)
        # 动态权重初始化
        self.kernel_weights = nn.Parameter(torch.ones(4))

        # 更均衡的核尺寸：3, 7, 15, 31
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a1, a2, a3, a4 = torch.chunk(x, 4, dim=1)

        # 加权多尺度融合
        w = F.softmax(self.kernel_weights, dim=0)
        out = torch.cat([
            w[0] * self.LKA3(a1),
            w[1] * self.LKA5(a2),
            w[2] * self.LKA7(a3),
            w[3] * self.LKA9(a4)
        ], dim=1)

        return self.proj_last(out) * self.scale + shortcut

class KAN_GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm1d(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            KANConv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            KANConv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            KANConv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            KANConv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))
        # self.LKA9 = nn.Sequential(
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),
        #
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))
        #
        # self.LKA7 = nn.Sequential(
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),
        #
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))
        #
        # self.LKA5 = nn.Sequential(
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),
        #
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))
        #
        # self.LKA3 = nn.Sequential(
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),
        #
        #     nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))
        self.X3 = KANConv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = KANConv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = KANConv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = KANConv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))


    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        # x = torch.cat([self.LKA3(a_1) * a_1, self.LKA5(a_2) * a_2, self.LKA7(a_3) * a_3,
        #                self.LKA9(a_4) * a_4], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x

class GPFCA_ori(nn.Module):

    def __init__(self, in_channels, num_blocks=2):
        super().__init__()

        self.naf_blocks = nn.ModuleList([LKFCA_Block(in_channels) for _ in range(num_blocks)])

        # self.norm1 = LayerNorm1d(in_channels)
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')

    def forward(self, x):

        # x = self.norm1(x)

        x = self.Rearrange1(x)


        for block in self.naf_blocks:
            x = block(x)

        x = self.Rearrange2(x)

        return x
class GPFCA(nn.Module):

    def __init__(self, in_channels, num_blocks=1):
        super().__init__()

        self.naf_blocks = nn.ModuleList([DyTanh_LKFCA(in_channels) for _ in range(num_blocks)])

        # self.norm1 = LayerNorm1d(in_channels)
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')

    def forward(self, x):

        # x = self.norm1(x)

        x = self.Rearrange1(x)


        for block in self.naf_blocks:
            x = block(x)

        x = self.Rearrange2(x)

        return x
class BiasNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(BiasNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # 可学习的逐通道偏置 (b)
        self.bias = nn.Parameter(torch.zeros(num_features))

        # 可学习的标量参数 (γ)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 确保输入是 2D 或更高维度 (batch_size, num_features, ...)
        if x.dim() < 2:
            raise ValueError("Input must have at least 2 dimensions.")

        # 计算 x - b
        x_centered = x - self.bias.view(1, -1, *([1] * (x.dim() - 2)))

        # 计算 RMS[x - b] (在通道维度上计算均方根值)
        rms = torch.sqrt(torch.mean(x_centered ** 2, dim=1, keepdim=True) + self.eps)

        # 归一化并应用缩放
        x_norm = x / rms
        x_norm = x_norm * torch.exp(self.gamma)

        return x_norm
class Attn_Residual(nn.Module):
    def __init__(self, fn, scale_factor=1.0):
        super(Attn_Residual, self).__init__()
        self.fn = fn
        self.scale_factor = scale_factor

    def forward(self, x):
        return x + self.fn(x,4,4) * self.scale_factor
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)
