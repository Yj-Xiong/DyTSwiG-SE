import torch,math
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from utils import *

def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError('Both Speech Files must be same length.')

    overall_snr = 10 * np.log10(np.sum(np.square(clean_speech)) / np.sum(np.square(clean_speech - processed_speech)))

    # Global Variables
    winlength = round(30 * sample_rate / 1000)    # window length in samples
    skiprate = math.floor(winlength / 4)     # window skip in samples
    MIN_SNR = -10    # minimum SNR in dB
    MAX_SNR = 35     # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(clean_length / skiprate - (winlength / skiprate))   # number of frames , 32000/120-4
    start = 0      # starting sample
    window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)))

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = np.sum(np.square(clean_frame))
        noise_energy = np.sum(np.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * math.log10(signal_energy / (noise_energy + EPS) + EPS)
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return overall_snr, segmental_snr
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score
def ssnr_loss(clean, noisy, sr=16000):
    try:
        snr_dist, segsnr_dist = snr(clean, noisy, sr)
        # snr_mean = snr_dist
        ssnr_score = np.mean(segsnr_dist)
    except:
        # error can happen due to silent period
        pesq_score = -1
    return ssnr_score

def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=15)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)
def batch_ssnr(clean, noisy):
    ssnr_score = Parallel(n_jobs=15)(delayed(ssnr_loss)(c, n) for c, n in zip(clean, noisy))
    ssnr_score = np.array(ssnr_score)
    # if -1 in ssnr_score:
    #     return None
    # ssnr_score =
    return torch.FloatTensor(ssnr_score)

class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super(MetricDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*2, affine=True),
            nn.PReLU(dim*2),
            nn.utils.spectral_norm(nn.Conv2d(dim*2, dim*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*4, affine=True),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Conv2d(dim*4, dim*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(dim*8, affine=True),
            nn.PReLU(dim*8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim*8, dim*4)),
            nn.Dropout(0.3),
            nn.PReLU(dim*4),
            nn.utils.spectral_norm(nn.Linear(dim*4, 1)),
            LearnableSigmoid_1d(1)
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)