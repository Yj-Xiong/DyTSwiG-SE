from __future__ import absolute_import, division, print_function, unicode_literals
import whisper
import jiwer
import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from SEMamba.models.generator import SEMamba as Model
# from MambaSEUNet.models.generatorU import SEUNet as Model
# from MPmodels.model import MPNet as Model
# from MambaSEUNet.models.pcs400 import cal_pcs
# from models.g_3090 import DBD_LKFCA_Net as Model
import soundfile as sf
import os
import argparse
import librosa
import numpy as np
from compute_metrics import compute_all_metrics as  compute_metrics
from rich.progress import track
h = None
device = None
from funasr import AutoModel
# from funasr.utils.misc import postprocess_utils
import Levenshtein


def calculate_cer(hyp, ref):
    """
    计算 CER（字错误率）
    :param hyp: 预测文本
    :param ref: 参考文本
    :return: CER（百分比）
    """
    if not ref:  # 如果参考文本为空，返回 100%（如果预测文本非空）
        return 100.0 if hyp else 0.0

    edit_distance = Levenshtein.distance(hyp, ref)
    cer = (edit_distance / len(ref)) * 100
    return cer


def get_dataset_filelist(a):
    with open(a.input_test_file, 'r', encoding='utf-8') as fi:
        indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return indexes

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
            if a.post_processing_PCS == True:
                audio_g = cal_pcs(audio_g.squeeze().cpu().numpy())
                output_file = os.path.join(a.output_dir, index + '.wav')

                sf.write(output_file, audio_g, h.sampling_rate, 'PCM_16')
                torch.cuda.empty_cache()

            else:
                output_file = os.path.join(a.output_dir, index+'.wav')

                sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')
                torch.cuda.empty_cache()
def cal_wer_whisper(whisper,clean_file, enhanced_file):
    # 加载Whisper模型
    wer_model = whisper

    # 对原始音频文件进行转录
    result_clean = wer_model.transcribe(clean_file)
    reference = result_clean["text"]

    # 对增强后的音频文件进行转录
    result_enhanced = wer_model.transcribe(enhanced_file)
    hypothesis = result_enhanced["text"]

    # 计算次错率
    wer = jiwer.wer(reference, hypothesis)
    return wer
def cal_cer(model,clean_file, enhanced_file):
    # (1) 加载模型


    # (2) 语音识别（假设 audio.wav 是输入音频）
    res = model.generate(input=enhanced_file)  # 返回 ASR 结果
    hyp_text = res[0]["text"]  # 预测文本

    # (3) 计算 CER
    ref = model.generate(input=clean_file)  # 返回 ASR 结果
    ref_text = ref[0]["text"]  # 预测文本
    cer = calculate_cer(hyp_text, ref_text)


    return cer
def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_clean_wavs_dir', default='/home/dataset/TIMIT+CHiME3/clean/')
    #parser.add_argument('--input_noisy_wavs_dir', default='/home/dataset/TIMIT+CHiME3/noisy/')
    #parser.add_argument('--input_test_file', default='/home/xyj/Experiments/TIMIT+CHiME3_test.txt')
    parser.add_argument('--input_clean_wavs_dir', default='/home/dataset/THCHS-30/data_thchs30/test')
    parser.add_argument('--input_noisy_wavs_dir', default='/home/xyj/datasets/chinese/test_noisy')
    parser.add_argument('--input_test_file', default='/home/xyj/datasets/chinese/chinese_test.txt')
    parser.add_argument('--output_dir', default='/home/xyj/Experiments/CMG-v1/Enh_audio-primeknet-zh-43/')
    parser.add_argument('--checkpoint_file', default='/home/xyj/Experiments/g_best_PESQ3.549408779850284_epoch84')
    parser.add_argument('--post_processing_PCS', default=False)
    a = parser.parse_args()

    config_file = '/home/xyj/Experiments/DyTSwiG-SE-main/config.json'
    with open(config_file) as f:
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

    #inference(a)
    # wer_model = whisper.load_model("turbo", device="cuda")

    cer_model = AutoModel(model="paraformer-zh",device="cuda")
    indexes = get_dataset_filelist(a)
    num = len(indexes)
    print(num)
    metrics_total = np.zeros(8)
    wer_total = np.zeros(1)
    # 用于保存每个指标的单次结果，计算标准差
    metrics_list = []
    wer_list = []
    for index in track(indexes):
        clean_wav = os.path.join(a.input_clean_wavs_dir, index + '.wav')
        noisy_wav = os.path.join(a.output_dir, index + '.wav')
        clean, sr = librosa.load(clean_wav, h.sampling_rate)
        noisy, sr = librosa.load(noisy_wav, h.sampling_rate)
        # wer = cal_wer_whisper(wer_model, clean_wav, noisy_wav)
        wer = cal_cer(cer_model, clean_wav, noisy_wav)
        # metrics = compute_metrics(clean, noisy, sr, 0)
        # metrics = np.array(metrics)
        wer = np.array(wer)
        wer_total += wer
        wer_list.append(wer)  # 添加WER到列表
        # metrics_total += metrics
        # metrics_list.append(metrics)  # 添加指标到列表
    wer_avg = wer_total / num
    wer_std = np.std(wer_list, axis=0)  # 计算标准差
    # metrics_avg = metrics_total / num
    # metrics_std = np.std(metrics_list, axis=0)  # 计算标准差
    print(# 'pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2],
          # 'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'sisnr: ', metrics_avg[5], 'sisdr: ', metrics_avg[6],'stoi: ', metrics_avg[7],
          'wer: ', wer_avg[0]
          )

    file_path = os.path.join(a.output_dir, 'cer-output.txt')
    with open(file_path, 'w') as f:
        print(
          #   'pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2],
          # 'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'sisnr: ', metrics_avg[5], 'sisdr: ', metrics_avg[6],'stoi: ', metrics_avg[7], 
          'wer: ', wer_avg[0],
        # 'pesq_std:', float(metrics_std[0]),
        # 'csig_std:', float(metrics_std[1]),
        # 'cbak_std:', float(metrics_std[2]),
        # 'covl_std:', float(metrics_std[3]),
        # 'ssnr_std:', float(metrics_std[4]),
        # 'sisnr_std:', float(metrics_std[5]),
        # 'sisdr_std:', float(metrics_std[6]),
        # 'stoi_std:', float(metrics_std[7]),
       'wer_std:', float(wer_std),
         file=f)
if __name__ == '__main__':
    main()

