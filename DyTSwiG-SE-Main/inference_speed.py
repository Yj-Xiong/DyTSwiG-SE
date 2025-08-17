from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
# from models.MN_Net import MN_Net as Model
from models.g_3090 import DBD_LKFCA_Net as Model
# from MPmodels.model import MPNet as Model
import soundfile as sf
# from SEMamba.models.generator import SEMamba as Model
# from MambaSEUNet.models.generator import MambaSEUNet as Model
import time
import torch
import argparse
import json
import os
from attrdict import AttrDict
from thop import profile

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(a):
    model = Model(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    with open(a.input_test_file, 'r', encoding='utf-8') as fi:
        test_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for i, index in enumerate(test_indexes):
            print(index)
            noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index+'.wav'), h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index+'.wav')

            sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--data_len', default=2)
    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    length = a.data_len
    sr = 16000
    # Measure inference speed and FLOPS
    model = Model(h).to(device)
    input_size = sr * length
    input_data = torch.randn(input_size)
    input_data = torch.FloatTensor(input_data).to(device)
    norm_factor = torch.sqrt(len(input_data) / torch.sum(input_data ** 2.0)).to(device)
    noisy_wav = (input_data * norm_factor).unsqueeze(0)
    noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

    print('Warming up the model...')
    for i in range(10):
        with torch.no_grad():
            model(noisy_amp, noisy_pha)

    print('Measuring inference speed...')
    total_time = 0
    num_iters = 100
    for i in range(num_iters):
        start_time = time.time()
        with torch.no_grad():
            model(noisy_amp, noisy_pha)


        end_time = time.time()
        total_time += end_time - start_time

    avg_time = total_time / num_iters
    print('Avg. Inference Time: {:.3f} seconds'.format(avg_time))

    # Measure FLOPS
    flops, params = profile(model, inputs=(noisy_amp, noisy_pha))
    print('Number of FLOPs: {:.3f} GFLOPs'.format(flops / 1e9))
    num_params = sum(p.numel() for p in model.parameters())
    max_memory = 0
    num_iters = 100
    for i in range(num_iters):
        torch.cuda.reset_max_memory_allocated()
        with torch.no_grad():
            model(noisy_amp, noisy_pha)
        max_memory = max(max_memory, torch.cuda.max_memory_allocated() / 1024**2)  # 转换为MB
    print('Data Length: {:.3f} seconds'.format(input_size/sr))
    print('Avg. Inference Time: {:.3f} seconds'.format(avg_time))
    print('Number of FLOPs: {:.3f} GFLOPs'.format(flops / 1e9))
    print('Max Memory Usage: {:.2f} MB'.format(max_memory))


    print(f"Manual calculation of parameters: {num_params}")
if __name__ == '__main__':
    main()



