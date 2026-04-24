import os
import random
import torch
import torch.utils.data
import librosa
from datasets.jsonl_dataset import JsonlAudioDataset
import torch.nn.functional as F
import numpy as np
def _min_length(audio1, audio2, mode='constant', max_length=160000):
    """支持1D音频的安全长度对齐函数，增加最大长度限制"""
    # 转换为Tensor并确保是1D
    audio1 = torch.as_tensor(audio1).flatten()
    audio2 = torch.as_tensor(audio2).flatten()
    
    # 应用最大长度限制
    len1, len2 = min(len(audio1), max_length), min(len(audio2), max_length)
    target_len = max(len1, len2)
    
    # 统一转为2D格式 [1, L]
    audio1 = audio1[:len1].unsqueeze(0)
    audio2 = audio2[:len2].unsqueeze(0)
    
    def _safe_pad(audio, target_len):
        current_len = audio.shape[-1]
        if current_len >= target_len:
            return audio
        
        pad_size = target_len - current_len
        if mode == 'reflect':
            # 分段反射填充（避免原始长度不足）
            repeats = (pad_size // current_len) + 1
            reflected = torch.cat([torch.flip(audio, [-1])] * repeats, dim=-1)
            return torch.cat([audio, reflected[..., :pad_size]], dim=-1)
        else:
            return F.pad(audio, (0, pad_size), mode=mode)
    
    try:
        if len1 < target_len:
            audio1 = _safe_pad(audio1, target_len)
        if len2 < target_len:
            audio2 = _safe_pad(audio2, target_len)
        return audio1.squeeze(0), audio2.squeeze(0)
    except Exception as e:
        # 异常时返回零填充（确保训练继续）
        min_len = min(len1, len2)
        return audio1[..., :min_len].squeeze(0), audio2[..., :min_len].squeeze(0)
def _adjust_length(audio, length):
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start:start+length]
    else:
        return torch.nn.functional.pad(audio, (0, length - len(audio)), mode='reflect')
def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, addeps=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    eps = 1e-10
    if addeps == False:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)


    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return training_indexes, validation_indexes


class Val_Dataset(torch.utils.data.Dataset):
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dir, segment_size, n_fft, hop_size, win_size,
                 sampling_rate, compress_factor, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(os.path.join(self.clean_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
            noisy_audio, _ = librosa.load(os.path.join(self.noisy_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        #target_len = min(len(clean_audio), len(noisy_audio))  # 策略1：取最小长度
    # target_len = len(noisy)                # 策略2：固定对齐到noisy长度
    # target_len = self.segment_size         # 策略3：固定目标长度
    
    # 智能裁剪/填充

    
        #clean_audio = _adjust_length(clean_audio, target_len)
        #noisy_audio = _adjust_length(noisy_audio, target_len)

        clean_audio, noisy_audio = _min_length(clean_audio, noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)
        #clean_audio, noisy_audio = align_clean_to_noisy(clean_audio, noisy_audio, mode="pad")
        #print(clean_audio.shape, noisy_audio.shape,)
        assert clean_audio.size(1) == noisy_audio.size(1)
        #
        # if self.split:
        #     if clean_audio.size(1) >= self.segment_size:
        #         max_audio_start = clean_audio.size(1) - self.segment_size
        #         rand_num = random.random()
        #
        #         if rand_num < 0.35:
        #             audio_start = 0
        #         elif rand_num < 0.7:
        #             audio_start = max_audio_start
        #         else:
        #             audio_start = random.randint(0, max_audio_start)
        #
        #         clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
        #         noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
        #     else:
        #         clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)),
        #                                               'constant')
        #         noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)),
        #                                               'constant')
        #
        # clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, self.n_fft, self.hop_size, self.win_size,
        #                                                self.compress_factor)  # [1, n_fft/2+1, frames]
        # noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size,
        #                                                self.compress_factor)  # [1, n_fft/2+1, frames]

        return (
        clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.audio_indexes)


class Datasetv1(torch.utils.data.Dataset):
    def __init__(self, training_indexes, dataloader, segment_size, n_fft, hop_size, win_size, 
                sampling_rate, compress_factor, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)

        self.dataloader = dataloader
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        batch = next(iter(self.dataloader))  # 获取一个 batch
        clean_audio, noisy_audio = batch[0][index], batch[1][index]  # 从 batch 中提取单条数据
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        
        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                rand_num = random.random()

                if rand_num < 0.01:
                    audio_start = 0
                elif rand_num < 0.02:
                    audio_start = max_audio_start
                else:
                    audio_start = random.randint(0, max_audio_start)

                clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)),
                                                      'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)),
                                                      'constant')
        #clean_audio,noisy_audio = clean_audio.squeeze(0),noisy_audio.squeeze(0)
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor) #[1, n_fft/2+1, frames]
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor) #[1, n_fft/2+1, frames]

        return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), noisy_mag.squeeze(), noisy_pha.squeeze())

    def __len__(self):
        return len(self.audio_indexes)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dir, segment_size, n_fft, hop_size, win_size, 
                sampling_rate, compress_factor, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        # 添加一个列表来记录跳过文件的索引
        self.skipped_indices = []

    def __getitem__(self, index):
        # 如果这个索引已经被标记为跳过，就自动跳到下一个索引
        while index in self.skipped_indices:
            index = (index + 1) % len(self.audio_indexes)
        
        filename = self.audio_indexes[index]
        
        try:
            if self._cache_ref_count == 0:
                try:
                    clean_audio, _ = librosa.load(os.path.join(self.clean_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
                    noisy_audio, _ = librosa.load(os.path.join(self.noisy_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
                except Exception as e:
                    print(f"Error loading file {filename}.wav: {str(e)}. Skipping...")
                    self.skipped_indices.append(index)
                    # 递归调用下一个索引
                    return self.__getitem__((index + 1) % len(self.audio_indexes))
                
                self.cached_clean_wav = clean_audio
                self.cached_noisy_wav = noisy_audio
                self._cache_ref_count = self.n_cache_reuse
            else:
                clean_audio = self.cached_clean_wav
                noisy_audio = self.cached_noisy_wav
                self._cache_ref_count -= 1
            
            clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
            
            # 检查音频长度是否为0
            if len(clean_audio) == 0 or len(noisy_audio) == 0:
                print(f"Empty audio file {filename}.wav. Skipping...")
                self.skipped_indices.append(index)
                return self.__getitem__((index + 1) % len(self.audio_indexes))
            
            norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
            clean_audio = (clean_audio * norm_factor).unsqueeze(0)
            noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

            assert clean_audio.size(1) == noisy_audio.size(1)

            if self.split:
                if clean_audio.size(1) >= self.segment_size:
                    max_audio_start = clean_audio.size(1) - self.segment_size
                    rand_num = random.random()

                    if rand_num < 0.01:
                        audio_start = 0
                    elif rand_num < 0.02:
                        audio_start = max_audio_start
                    else:
                        audio_start = random.randint(0, max_audio_start)

                    clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                    noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
                else:
                    clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)),
                                                          'constant')
                    noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)),
                                                          'constant')

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

            return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), noisy_mag.squeeze(), noisy_pha.squeeze())
        
        except Exception as e:
            print(f"Unexpected error processing file {filename}.wav: {str(e)}. Skipping...")
            self.skipped_indices.append(index)
            return self.__getitem__((index + 1) % len(self.audio_indexes))

    def __len__(self):
        return len(self.audio_indexes)

class SelectDataset(torch.utils.data.Dataset):
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dirs, segment_size, n_fft, hop_size, win_size, 
                sampling_rate, compress_factor, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dirs = noisy_wavs_dirs
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(os.path.join(self.clean_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
        
            # 随机选择一个噪声目录
            
        
        # Create a shuffled copy of noisy directories to try
            noisy_dirs_to_try = self.noisy_wavs_dirs.copy()
            random.shuffle(noisy_dirs_to_try)
        
            noisy_audio = None
            for selected_noisy_dir in noisy_dirs_to_try:
                try:
                # If selected directory is the generated directory, try its subdirectories
                    if selected_noisy_dir == '/root/SRdataset/train_v1/generated/':
                        subdirs = [d for d in os.listdir(selected_noisy_dir) 
                                 if os.path.isdir(os.path.join(selected_noisy_dir, d))]
                        if subdirs:
                            random.shuffle(subdirs)
                            for subdir in subdirs:
                                noisy_path = os.path.join(selected_noisy_dir, subdir, filename + '.wav')
                                if os.path.exists(noisy_path):
                                    noisy_audio, _ = librosa.load(noisy_path, sr=self.sampling_rate)
                                    break
                    else:
                        noisy_path = os.path.join(selected_noisy_dir, filename + '.wav')
                        if os.path.exists(noisy_path):
                            noisy_audio, _ = librosa.load(noisy_path, sr=self.sampling_rate)
                
                    if noisy_audio is not None:
                        break
                except Exception as e:
                    print(f"Error loading noisy audio from {selected_noisy_dir}: {e}")
                    continue
        
            if noisy_audio is None:
            # If no noisy version found in any directory, return another random sample
                print(f"No noisy version found for {filename} in any directory, selecting another sample")
                return self[random.randint(0, len(self)-1)]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        
        #clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        target_len = min(len(clean_audio), len(noisy_audio))  # 策略1：取最小长度
    # target_len = len(noisy)                # 策略2：固定对齐到noisy长度
    # target_len = self.segment_size         # 策略3：固定目标长度
    
    # 智能裁剪/填充
        
    
        #clean_audio = _adjust_length(clean_audio, target_len)
        #noisy_audio = _adjust_length(noisy_audio, target_len)
        clean_audio, noisy_audio = _min_length(clean_audio, noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)
        #clean_audio, noisy_audio = align_clean_to_noisy(clean_audio, noisy_audio, mode="pad")
        #print(clean_audio.shape, noisy_audio.shape,)
        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                rand_num = random.random()

                if rand_num < 0.01:
                    audio_start = 0
                elif rand_num < 0.02:
                    audio_start = max_audio_start
                else:
                    audio_start = random.randint(0, max_audio_start)

                clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)),
                                                      'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)),
                                                      'constant')

        clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor) #[1, n_fft/2+1, frames]
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor) #[1, n_fft/2+1, frames]

        return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), noisy_mag.squeeze(), noisy_pha.squeeze())

    def __len__(self):
        return len(self.audio_indexes)
