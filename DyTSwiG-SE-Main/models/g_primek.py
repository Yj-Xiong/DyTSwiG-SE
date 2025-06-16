import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GPFCA import GPFCA_ori as GPFCA
from models.dysample import DySample
import math
from .kaconv.kaconv.fastkanconv import FastKANConvLayer as ConvKAN
from utils import get_padding_2d, LearnableSigmoid_2d
from pesq import pesq
from joblib import Parallel, delayed
from torchvision.ops.deform_conv import DeformConv2d

class AffinePReLU(nn.Module):
    def __init__(self, num_channels, num_features):
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
class SwiGLU(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # 可设为可学习参数

    def forward(self, x):
        x1, x2 = x.chunk(1, dim=1)
        return (x1 * torch.sigmoid(self.beta * x1)) * x2
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
class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1, 1)
    ):
        super().__init__()

        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,
                                                      stride= 1,padding= 1,bias= False,groups=in_ch),
                                            nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                      kernel_size=1,
                                                      stride=1, padding=0, bias=False)
                                            )
        self.dcn=DeformConv2d(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=in_ch
                    )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)

        x = self.pwconv(x)
        x = self.act(x)
        return x
class TFCA(nn.Module):# F_in = [B,T*F,C]
    def __init__(self, c_in):
        super(TFCA, self).__init__()
        self.TFca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1,1),
            nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        

class Deform_Embedding(nn.Module):

    def __init__(self,
                 in_chans=64,
                 embed_dim=64,
                 patch_size=3,
                 stride=1,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1, 1)):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                act_layer=act_layer,
                offset_clamp=offset_clamp
            )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class DS_DDB(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DS_DDB, self).__init__()
        self.h = h
        self.depth = depth
        # self.Deform_Embedding = Deform_Embedding(in_chans=h.dense_channel, embed_dim=h.dense_channel)
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel*(i+1), kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, dilation=(dil, 1)), groups=h.dense_channel*(i+1), bias=True),
                nn.Conv2d(in_channels=h.dense_channel*(i+1), out_channels=h.dense_channel, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):

        skip = x
        for i in range(self.depth):

            # if i == 0:
            #     x = self.Deform_Embedding(x)
            #     x = self.dense_block[i](x)
            # else:
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DS_DDB(h, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        # x = self.dysample(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        # self.dysample = DySample(h.dense_channel)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        # x = self.dysample(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x



class TS_tfBLOCK(nn.Module):
    def __init__(self, h):
        super(TS_tfBLOCK, self).__init__()
        self.h = h
        self.time = GPFCA(h.dense_channel)
        self.freq = GPFCA(h.dense_channel)
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        self.TFca = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvKAN(in_channels=h.dense_channel, out_channels=h.dense_channel, groups=16, padding=1),
            # nn.Conv2d(in_channels=h.dense_channel, out_channels=h.dense_channel, kernel_size=1, padding=0, stride=1,
            #           groups=1, bias=True),
        )
        self.LayerNorm = nn.LayerNorm(h.dense_channel)
        self.LayerNorm2 = nn.LayerNorm(h.dense_channel)
        self.TFca2 = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),
            ConvKAN(in_channels=h.dense_channel, out_channels=h.dense_channel, groups=16, padding=1),
            # nn.Conv2d(in_channels=h.dense_channel, out_channels=h.dense_channel, kernel_size=1, padding=0, stride=1,
            #           groups=1, bias=True),
        )
    def forward(self, x):
        b, c, t, f = x.size()
        # print(x.shape)
        x_tf = self.LayerNorm(x.permute(0, 2, 3, 1).contiguous()).reshape(b, c, t, f).contiguous()
        x_tf = self.TFca(x_tf)
        # x_tf = x_tf.permute(0, 2, 3, 1).contiguous()
        x_t = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = (x).permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time(x) + x_t * self.beta
        x = x.reshape(b, c, t, f).contiguous() * x_tf
        x_tf2 = x.reshape(b, f, t, c).contiguous()
        x_tf2 = self.TFca2(self.LayerNorm2(x_tf2).reshape(b, c, f, t).contiguous())
        # # # x_tf2 = x_tf2.permute(0, 3, 2, 1).contiguous().view(b,1,1, c)
        # # x_2 = x.view(b, f, t, c)
        # # x_f = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = (x).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq(x) + x * self.gamma
        x = x.reshape(b, c, t, f).contiguous() * x_tf2
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
class TS_BLOCK(nn.Module):
    def __init__(self, h):
        super(TS_BLOCK, self).__init__()
        self.h = h
        self.time = GPFCA(h.dense_channel)
        self.freq = GPFCA(h.dense_channel)
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        
        x = self.time(x) + x * self.beta
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)

        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

class AdaptiveDecoder(nn.Module):
    def __init__(self, h, num_features, f_types,device='cuda:0'):  # out_channel=[M,P,C]
        super(AdaptiveDecoder, self).__init__()
        self.h = h

        self.dense_block = DS_DDB(h, depth=4)
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        # self.initial_conv = nn.Sequential(
        #     nn.ConvTranspose2d(ori_channel, ori_channel, (1, 3), (1, 2)),
        #     nn.InstanceNorm2d(ori_channel, affine=True),
        #     nn.PReLU(ori_channel)
        # )
        self.mask_conv = nn.Conv2d(h.dense_channel, 1, (1, 1))
        # self.mask_act = nn.Sequential(BiasNorm(1),nn.Softmax(dim=1))
        self.mask_act = LearnableSigmoid_2d(num_features)
        # self.LSigmoid2d = LearnableSigmoid_2d(num_features)
        self.weights = [nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device),
                        nn.Parameter(torch.tensor(1.0), requires_grad=True).to(device)]
        # self.weights = [w.to(device) for w in self.weights]
        if f_types == "C":
            self.complex_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.complex_conv = nn.Sequential(self.complex_spspconv,
            nn.Conv2d(h.dense_channel, 2, (1, 2)))
            # self.BiasNorm = BiasNorm(2)
        elif f_types == "P":
            self.phase_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.phase_conv = nn.modules.ModuleList([nn.Conv2d(h.dense_channel, 1, (1, 1)) for i in range(2)])
            # self.BiasNorm = [BiasNorm(1).cuda(), BiasNorm(1).cuda()]
        elif f_types == "CP":
            self.complex_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.complex_conv = nn.Sequential(self.complex_spconv,
                                              nn.Conv2d(h.dense_channel, 2, (1, 1)))
            self.phase_spconv = nn.Sequential(
                nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.phase_conv = nn.modules.ModuleList([nn.Conv2d(h.dense_channel, 1, (1, 1)) for i in range(2)])
            self.weights.append(nn.Parameter(torch.tensor(1.0), requires_grad=True).cuda())

    def forward(self, x, f_types):

        if f_types == "C":
            device = x.device
            # self.weights = [w.to(device) for w in self.weights]
            x = self.dense_block(x)
            x = self.initial_conv(x)
            x_mask = self.mask_act(self.mask_conv(x ).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2,
                                                                                                                1).unsqueeze(
                1)
            x_complex = self.complex_conv(x )
            return x_mask, x_complex
        if f_types == "P":
            device = x.device
            # print(x.shape)
            self.weights = [w.to(device) for w in self.weights]

            x = self.dense_block(x)
            x_p = self.phase_spconv(x* self.weights[1])
            x = self.initial_conv(x* self.weights[0])
            # print(x.shape)
            # x = self.initial_conv(x)
            x_mask = self.mask_act(self.mask_conv(x).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2, 1).unsqueeze(1)
            # x_p = x* self.weights[1]
            x_r = (self.phase_conv[0](x_p))
            x_i = (self.phase_conv[1](x_p))
            x_pha = torch.atan2(x_i, x_r)
            return x_mask, x_pha
        if f_types == "CP":
            device = x.device
            self.weights = [w.to(device) for w in self.weights]
            x = self.dense_block(x)
            x_complex = self.complex_conv(x*self.weights[2])
            x_p = self.phase_spconv(x*self.weights[1])
            x = self.initial_conv(x*self.weights[0])
            x_mask = self.mask_act(self.mask_conv(x ).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2, 1).unsqueeze(1)

            x_r = self.phase_conv[0](x_p)
            x_i = self.phase_conv[1](x_p)
            x_pha = torch.atan2(x_i, x_r)
            return x_mask, x_complex, x_pha

class DualMaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(DualMaskDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.kan_conv = ConvKAN(in_channels=h.dense_channel,out_channels=h.dense_channel,groups=16,padding=1)
        self.SP_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
             nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            # nn.InstanceNorm2d(out_channel, affine=True),
            # nn.PReLU(out_channel),
            # nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.mag_conv = nn.Conv2d(out_channel, 1, (1, 1))
        self.mask_mag = nn.PReLU(h.n_fft // 2 + 1, init=-0.25)
        self.mask_pha = nn.Sequential(nn.Conv2d(out_channel, 1, (1, 1)),
                                       LearnableSigmoid_2d(h.n_fft // 2 + 1, beta=h.beta))

    def forward(self, x):
        x = self.dense_block(x)
        # x = self.dysample(x)
        x = self.kan_conv(x)
        x = self.SP_conv(x)

        m_mag = self.mask_mag(self.mag_conv(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        m_pha = self.mask_pha(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1 , 3, 2).contiguous()
        return m_mag, m_pha
# class LKFCA_Net(nn.Module):
#     def __init__(self, h, num_tsblock=4):
#         super(LKFCA_Net, self).__init__()
#         self.h = h
#         self.num_tsblock = num_tsblock
#         self.dense_encoder = DenseEncoder(h, in_channel=2)
#         self.LKFCAnet = nn.ModuleList([])
#         for i in range(num_tsblock):
#             self.LKFCAnet.append(TS_BLOCK(h))
#         # self.dense_decoder = DualMaskDecoder(h, out_channel=2)
#         self.dense_decoder = AdaptiveDecoder(h,h.n_fft//2+1, f_types="P")
#         # self.mask_decoder = MaskDecoder(h, out_channel=1)
#         # self.phase_decoder = PhaseDecoder(h, out_channel=1)
#     def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
#         noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
#         x = self.dense_encoder(x)
#
#         for i in range(self.num_tsblock):
#             x = self.LKFCAnet[i](x)
#         mask, phase = self.dense_decoder(x, f_types="P")
#         denoised_mag = (noisy_mag * mask).permute(0, 3, 2, 1).squeeze(-1)
#         denoised_pha = phase.permute(0, 3, 2, 1).squeeze(-1)
#         denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
#                                     denoised_mag * torch.sin(denoised_pha)), dim=-1)
#
#         return denoised_mag, denoised_pha, denoised_com
#     def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
#         noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
#         x = self.dense_encoder(x)
#
#         for i in range(self.num_tsblock):
#             x = self.LKFCAnet[i](x)
#         mask_mag, mask_phase = self.dense_decoder(x, f_types="P")
#         denoised_mag = (noisy_mag * mask_mag).permute(0, 3, 2, 1).squeeze(-1)
#         denoised_pha = (noisy_pha * mask_phase).permute(0, 3, 2, 1).squeeze(-1)
#         denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
#                                     denoised_mag * torch.sin(denoised_pha)), dim=-1)
#
#         return denoised_mag, denoised_pha, denoised_com
# class LKFCA_Net(nn.Module):
#     def __init__(self, h, num_tsblock=4):
#         super(LKFCA_Net, self).__init__()
#         self.h = h
#         self.num_tsblock = num_tsblock
#         self.dense_encoder = DenseEncoder(h, in_channel=4)
#         self.LKFCAnet = nn.ModuleList([])
#         for i in range(num_tsblock):
#             self.LKFCAnet.append(TS_BLOCK(h))
#         self.dense_decoder = AdaptiveDecoder(h, h.n_fft // 2 + 1, f_types="CP")
#         # self.mask_decoder = MaskDecoder(h, out_channel=1)
#         # self.phase_decoder = PhaseDecoder(h, out_channel=1)
# 
#     def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
#         noisy_com = torch.stack((noisy_mag * torch.cos(noisy_pha),
#                                  noisy_mag * torch.sin(noisy_pha)), dim=-1).permute(0, 3, 2, 1)
#         noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
#         noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
# 
#         x = torch.cat((noisy_mag, noisy_pha,noisy_com), dim=1)  # [B, 2, T, F]
#         x = self.dense_encoder(x)
# 
#         for i in range(self.num_tsblock):
#             x = self.LKFCAnet[i](x)
#         mask, com, phase = self.dense_decoder(x, f_types="CP")
#         denoised_mag = (noisy_mag * mask).permute(0, 3, 2, 1).squeeze(-1)
#         denoised_pha = phase.permute(0, 3, 2, 1).squeeze(-1)
#         denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
#                                     denoised_mag * torch.sin(denoised_pha)), dim=-1) + com.permute(0, 3, 2, 1)
# 
#         return denoised_mag, denoised_pha, denoised_com
class LKFCA_Net(nn.Module):
    def __init__(self, h, num_tsblock=4):
        super(LKFCA_Net, self).__init__()
        self.h = h
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])
        for i in range(num_tsblock):
            self.LKFCAnet.append(TS_BLOCK(h))
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tsblock):
            x = self.LKFCAnet[i](x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com


def phase_losses(phase_r, phase_g, h):

    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score
