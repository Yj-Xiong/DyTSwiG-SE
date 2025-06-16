import torch
import torch.nn as nn
from einops import rearrange

from .lsigmoid import LearnableSigmoid2D


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding size for a convolutional layer.

    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.

    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    """
    Calculate the padding size for a 2D convolutional layer.

    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).

    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))


class DS_DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DS_DenseBlock, self).__init__()
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

    def forward(self, x):
        """
        Forward pass for the DenseBlock module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after processing through the dense block.
        """
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """

    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg['model_cfg']['input_channel']
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.dense_block = DS_DenseBlock(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

    def forward(self, x):
        """
        Forward pass for the DenseEncoder module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Encoded tensor.
        """
        x = self.dense_conv_1(x)  # [batch, hid_feature, time, freq]
        x = self.dense_block(x)  # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x


class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """

    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DS_DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']

        self.mask_conv = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(self.hid_feature, self.hid_feature, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1),
                      groups=self.hid_feature, bias=False),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        """
        Forward pass for the MagDecoder module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Decoded tensor with magnitude information.
        """
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """

    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DS_DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']

        self.phase_conv = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(self.hid_feature, self.hid_feature, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1),
                      groups=self.hid_feature, bias=False),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        """
        Forward pass for the PhaseDecoder module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Decoded tensor with phase information.
        """
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
