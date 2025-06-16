import whisper
import jiwer
import argparse
import os
import json,sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
from natsort import natsorted
from tqdm import tqdm
from rich.progress import track
def cal_wer_whisper(clean_file, enhanced_file):
    # 加载Whisper模型
    model = whisper.load_model("turbo",device="cuda:1")

    # 对原始音频文件进行转录
    result_clean = model.transcribe(clean_file)
    reference = result_clean["text"]

    # 对增强后的音频文件进行转录
    result_enhanced = model.transcribe(enhanced_file)
    hypothesis = result_enhanced["text"]

    # 计算次错率
    wer = jiwer.wer(reference, hypothesis)
    return wer

def evaluation( noisy_dir, clean_dir,  saved_dir):


    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)

    # 仅保留以 .wav 结尾的文件
    audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(1)
    for audio in tqdm(audio_list, desc="Evaluation Processing"):
        est_path = os.path.join(saved_dir, audio)
        est_audio, sr = sf.read(est_path)
        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)
        wer = cal_wer_whisper(clean_path, est_path)
        metrics = np.array(wer)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('wer: ', metrics_avg[0],)
    # 保存结果
    result_file_path = os.path.join(saved_dir, 'metrics_avg.json')
    results = {
        'wer': float(metrics_avg[0]),
    }

    # 如果文件存在，加载已有数据并追加新结果
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r') as json_file:
            all_results = json.load(json_file)
    else:
        all_results = []

    # 追加当前实验的结果
    all_results.append(results)

    # 保存更新后的结果
    with open(result_file_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

parser = argparse.ArgumentParser()

parser.add_argument("--test_dir", type=str, default='/home/dataset/Voicebank/noisy-vctk-16k/',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='/home/xyj/Experience/PrimeK-Net-main/generated_files/g_best/', help="where enhanced tracks to be saved")
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    noisy_dir = os.path.join(args.test_dir, 'noisy_testset_wav_16k')
    clean_dir = os.path.join(args.test_dir, 'clean_testset_wav_16k')
    # noisy_dir = args.test_dir
    # noisy_dir = '/home/xyj/datasets/chinese/test_noisy'
    #         noisy_dir = os.path.join(args.test_dir, noisy_dir)


    # # #得到增强算法1的评价指标
    # sisnr_evaluation(noisy_dir=noisy_dir, clean_dir='/home/dataset/Voicebank/noisy-vctk-16k/clean_testset_wav_16k/',
    #                             saved_dir=args.save_dir)
    evaluation( noisy_dir=noisy_dir, clean_dir = clean_dir,
               saved_dir=args.save_dir)