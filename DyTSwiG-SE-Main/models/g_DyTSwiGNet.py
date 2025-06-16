import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GPFCA import GPFCA
import math
from utils import get_padding_2d, LearnableSigmoid_2d
from pesq import pesq
from joblib import Parallel, delayed

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




class DyTanh(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_features,  1))  # 可学习缩放因子
        self.beta = nn.Parameter(torch.zeros(1, num_features,  1))  # 可学习偏移因子
        self.eps = eps  # 数值稳定性

    def forward(self, x):
        # 动态调整输入分布
        x = torch.tanh(self.alpha * x + self.beta)
        return x

class DS_DDB(nn.Module):
    def __init__(self, h, freq=100,kernel_size=(3, 3), depth=4):
        super(DS_DDB, self).__init__()
        self.h = h
        self.depth = depth
        # self.Deform_Embedding = Deform_Embedding(in_chans=h.dense_channel, embed_dim=h.dense_channel)
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel * (i + 1), h.dense_channel * (i + 1), kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, dilation=(dil, 1)), groups=h.dense_channel * (i + 1),
                          bias=True),
                nn.Conv2d(in_channels=h.dense_channel * (i + 1), out_channels=h.dense_channel, kernel_size=1, padding=0,
                          stride=1, groups=1,
                          bias=True),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
               # Harmonic_attn(h.dense_channel,freq=freq)
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


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel, n_head=4, att_hid_chan=4, n_freq=1, ori_freq=201):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.dense_block = DS_DDB(h, freq=201,depth=4)  # [b, h.dense_channel, ndim_time, h.n_fft//2+1]
        self.dense_conv_2 = nn.Sequential(
             nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
             nn.InstanceNorm2d(h.dense_channel, affine=True),
             nn.PReLU(h.dense_channel),
         )

    def forward(self, x):
        x = self.dense_conv_1(x)   # [b, 64, T, F]
        x = self.dense_block(x)  # [b, 64, T, F]

        x = self.dense_conv_2(x)  # [b, 64, T, F//2]

        return x


class DualMaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(DualMaskDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        # self.dysample = DySample(h.dense_channel)
        self.SP_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
             nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            # nn.InstanceNorm2d(out_channel, affine=True),
            # nn.PReLU(out_channel),
            # nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.mask_mag = nn.Sequential(nn.Conv2d(out_channel, 1, (1, 1)),
                                       nn.PReLU(h.n_fft // 2 + 1, init=-0.25))
        self.mask_pha = nn.Sequential(nn.Conv2d(out_channel, 1, (1, 1)),
                                       LearnableSigmoid_2d(h.n_fft // 2 + 1, beta=h.beta))

    def forward(self, x):
        x = self.dense_block(x)
        # x = self.dysample(x)
        x = self.SP_conv(x)
        m_mag = self.mask_mag(x)
        m_pha = self.mask_pha(x)
        return m_mag, m_pha
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

       # x = self.HA(x)
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
)
        x = self.dense_block(x)

        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
 

class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(CausalConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.left_pad = kernel_size[1] - 1
        # padding = (kernel_size[0] // 2, 0)
        padding = (kernel_size[0] // 2, self.left_pad)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        # x = F.pad(x, [self.left_pad, 0])
        return self.conv(x)[..., :T]
class DyTanh2d(nn.Module):
    def __init__(self, num_features, alpha=1.):
        super().__init__()
        self.init_alpha = alpha
        self.alpha = nn.Parameter(torch.full((1, 1, 1,1), self.init_alpha))  # 可学习缩放因子
        self.alpha1 = nn.Parameter(torch.full((1, 1, 1,1), self.init_alpha))  # 可学习缩放因子
        self.beta = nn.Parameter(torch.zeros(1, num_features,  1,1))  # 可学习偏移因子
        self.beta1 = nn.Parameter(torch.zeros(1, num_features,  1,1))  # 可学习偏移因子
        # self.gamma = nn.Parameter(torch.ones(1, num_features,  1,1))  # 可学习缩放因子
    def forward(self, x):
        # 动态调整输入分布
        x = self.alpha1 *torch.tanh(self.alpha * x + self.beta) + self.beta1
        # x = self.gamma * torch.tanh(self.alpha * x) + self.beta
        return x

class Harmonic_attn(nn.Module):
    def __init__(self, h, inchan,freq=100,channel=64,head=2,ks=1,stride=1):
        super(Harmonic_attn, self).__init__()
        self.h = h
        # self.conv_res = nn.Sequential(nn.Conv2d(inchan, channel, ks, padding=0,stride=stride),
        #     nn.InstanceNorm2d(channel),
        #     nn.PReLU(),)
        max_channel = channel * head
        self.norm = DyTanh2d(channel)
        self.norm1 = DyTanh2d(channel)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=max_channel,  kernel_size=(1,3), padding=(0,1), stride=1,
                               groups=1,
                               bias=True))

        self.k_conv = nn.Conv2d(in_channels=channel, out_channels=max_channel, kernel_size=(1,3), padding=(0,1), stride=1,
                      groups=1, bias=True)
        self.kq_conv = nn.Conv2d(in_channels=max_channel, out_channels=max_channel,  kernel_size=(1,3), padding=(0,1), stride=1,
                      groups=1, bias=True)
        self.conv_out = nn.Sequential(nn.Conv2d(in_channels=max_channel, out_channels=channel,  kernel_size=(1,3), padding=(0,1), stride=1,
                      groups=1, bias=True))
        #self.HA_weight = nn.Parameter(torch.ones(1,channel,  1, 1),requires_grad=True)
        self.conver_matrix = comb_pitch_conversion_matrix(R=1,F=freq,sr=16000)
    def forward(self, x):
        b, c, t, f = x.size()
        residual = x.clone()
        # x = self.conv1(x)
        # # x = self.gate(x)
        # x = x + residual
        # residual = x.clone()
        x = self.norm(x)
        v = self.conv1(x)
        # k_in = x/torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))
        k_in =x**2
        k_out = self.k_conv(self.norm1(k_in))
        h_mat = self.conver_matrix.unsqueeze(1).to(x.device)
        kq = torch.softmax(torch.matmul(k_out,torch.transpose(h_mat,2,3)),dim=-1)
        h = kq @ h_mat
        h = self.kq_conv(h)
        x = h*v
        x = self.conv_out(x)
        x = x + residual
        return x


def comb_pitch_conversion_matrix(R, F, sr, batch_size=1):
    """
    Generate the Comb-Pitch Conversion Matrix Q.

    Parameters:
        R (int): Resolution parameter.
        F (int): Frequency-related parameter (spectral resolution).
        sr (int): Sampling rate.
        batch_size (int): Batch size for processing multiple instances simultaneously.

    Returns:
        Q (torch.Tensor): The Comb-Pitch Conversion Matrix of shape [batch_size, N_c, F].
    """
    # Step 1: Initialize matrix Q with batch dimension
    N_c = int(420 / R) - int(60 / R)   # Number of pitch candidates
    Q = torch.zeros([batch_size, N_c, F], dtype=torch.float32)

    # Step 2: Generate pitch candidates in the range [60 Hz, 420 Hz]
    pitch_candidates = torch.linspace(60, 420, N_c)

    # Step 3: Outer loop over pitch candidates
    for j, fc in enumerate(pitch_candidates):
        # Step 4: Inner loop over harmonic periods p
        max_p = int(sr / (2 * R * fc))
        for p in range(1, max_p + 1):
            # Step 5: Compute harmonic location
            loc = (R * fc * p * F / (sr / 2)).floor().long()  # Compute harmonic location
            # Ensure loc is broadcastable across batch_size
            loc = loc.unsqueeze(0).repeat(batch_size, 1)  # Replicate for batch_size

            # Step 6: Compute harmonic weight
            peak = 1 / np.sqrt(p)

            # Step 7: Update matrix Q based on harmonic location and weight
            Q[:, j, loc] += peak  # Add weight to the corresponding location

            # Step 8: Handle interpolation between harmonics
            # If there are gaps between harmonics, use cosine curve interpolation
            if p > 1:
                last_loc = (R * fc * (p - 1) * F / (sr / 2)).floor().long()
                gap = loc - last_loc
                # Fill the gap with cosine interpolation
                if torch.any(gap > 1):
                    num_inter = int(gap)
                    F_cos = torch.cos(torch.linspace(0, 2 * np.pi, num_inter + 1))
                    F = torch.linspace(1 / np.sqrt(p - 1), 1 / np.sqrt(p), num_inter + 1)
                    interpolation_weights = F_cos * F
                    for k in range(num_inter):
                        Q[:, j, last_loc + k] += interpolation_weights[k]

    return Q


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
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time(x) + x * self.beta
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class LKFCA_Net(nn.Module):
    def __init__(self, h, num_tsblock=4):
        super(LKFCA_Net, self).__init__()
        self.h = h
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])

        for i in range(4):
            self.LKFCAnet.append(TS_BLOCK(h))
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(4):
            x = self.LKFCAnet[i](x)
        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = (self.phase_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com


def phase_losses(phase_r, phase_g, h):
    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq),
                                                                                     diagonal=2) - torch.eye(
        dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time),
                                                                                      diagonal=2) - torch.eye(
        dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

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
