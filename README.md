# DyTSwiG-Mamba: Layer Normalization-free CNN-Mamba Speech Enhancement Network with Dual-branch Phase Prediction (Under TASLP Review)

**English** | [中文](README_CN.md)

Official PyTorch implementation of the paper "DyTSwiG-Mamba: Layer Normalization-free CNN-Mamba Speech Enhancement Network with Dual-branch Phase Prediction" by Xiong et al. Audio samples are from VoiceBank+DEMAND dataset and THCHS+DNS dataset (mixed with THCHS-30 dataset and DNS-Challenge dataset). The source code is located in the directory "DyTSwiG-SE-Main". The wav files are resampled to 16kHz in our experiments.
### Yujie Xiong, Zhihua Huang and Bixin Wu

**Abstract:** 
Speech enhancement (SE) models based on multiple stacked Two-Stage (TS) blocks achieve impressive performance. However, they face a fundamental compromise between performance and efficiency within each TS block, while suffering from a limited data adaptability due to the pervasive use of layer normalization. Besides, widely adopted phase decoders tend to ignore the noisy phase condition, leading to a suboptimal noise robustness. In response, we propose two layer-normalization-free monaural SE networks: DyTSwiG-Net and DyTSwiG-Mamba. First, we introduce the SwiGLUformer as a more efficient replacement for the TS block, along with an Input-Biased Dynamic Tanh (IB-DyT) activation for layer normalization-free architectures. For DyTSwiG-Mamba, we further design a Dual-Branch Phase Decoder (DBPD) to jointly estimate the phase mapping and masking via adding the noisy phase spectrogram, and integrate a Global Bidirectional Mamba (G-BiMamba) module to enhance feature aggregation across the network.
Both models are thoroughly evaluated on four datasets in English and Mandarin. Ablation studies and visual analysis verify the effectiveness of the SwiGLUformer and DBPD. Experimental results show that DyTSwiG-Net achieves faster inference than existing methods while maintaining competitive performance. Notably, DyTSwiG-Mamba outperforms the SOTA model on the public dataset while saving around 25% of overall costs. In addition, the proposed IB-DyT can be seamlessly integrated into diverse architectures, yielding considerable performance gain in cross-dataset evaluation. Some methods exhibit varying effectiveness for Mandarin speech and low-SNR scenarios, while our models show superior performance and noise robustness. 


## Pre-requisites
1. Python >= 3.9.
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](https://github.com/Yj-Xiong/DyTSwiG-SE/blob/main/DyTSwiG-SE-Main/requirements.txt).
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942). 
5. Move the clean and noisy wavs to `VoiceBank+DEMAND/wavs_clean` and `VoiceBank+DEMAND/wavs_noisy` or any path you want, and change the path in [train.py](https://github.com/Yj-Xiong/DyTSwiG-SE/blob/main/DyTSwiG-SE-Main/train.py) [parser.add_argument('--input_clean_wavs_dir', default=] and [parser.add_argument('--input_noisy_wavs_dir', default=], respectively.

## Training
For a single GPU in recommended environment settings, DyTSwiG-Net needs at least 14GB GPU memery, whereas DyTSwiG-Mamba needs at least 16GB GPU memery. Edit imports of models (generators) in [train.py](https://github.com/Yj-Xiong/DyTSwiG-SE/blob/main/DyTSwiG-SE-Main/train.py) script and run:
```bash
cd DyTSwiG-SE-Main
CUDA_VISIBLE_DEVICES={GPU_ids} python train.py \
    --config "config.json" 
```

## Training with Other Dataset
Ensure the new clean and noisy files are moved to `OtherDataset/wavs_clean` and `OtherDataset/wavs_noisy`.
Edit path in [make_file_list.py](https://github.com/Yj-Xiong/DyTSwiG-SE/blob/main/DyTSwiG-SE-Main/tools/make_file_list.py) and run:
``` bash
cd DyTSwiG-SE-Main/tools
python make_file_list.py
```
Then replace the [test.txt](https://github.com/Yj-Xiong/DyTSwiG-SE/blob/main/DyTSwiG-SE-Main/AudioFiles/test.txt) and [training.txt](https://github.com/Yj-Xiong/DyTSwiG-SE/blob/main/DyTSwiG-SE-Main/AudioFiles/training.txt) with generated files in folder "./OtherDataset" and put your train and test sets in the same folder(clean or noisy).

## Inference and Evaluation
### Inference and Compute All Metrics
Change the path in '--checkpoint_file' option with the ckpt file. You can use the pretrained best checkpoint file we provide in `ckpt/g_best`.
```bash
cd DyTSwiG-SE-Main
python inference_and_cal_metric.py
```
Generated wav files are saved in `/home/xyj/Experiments/g_best` by default.<br>
You can change the path by editing `--output_dir` option.
### Compute Character Error Rate (CER) Only
Modify the folder paths in [cal_cer.py](https://github.com/Yj-Xiong/DyTSwiG-SE/blob/main/DyTSwiG-SE-Main/cal_cer.py) script and follow the commands:
```bash
cd DyTSwiG-SE-Main
python cal_cer.py
```
## Model Architecture
The architecture of proposed DyTSwiG-Net
![model_DyTSwiG-Net](Figures/Models/DyTSwiG-Net.png)
The architecture of proposed DyTSwiG-Mamba
![model_DyTSwiG-Mamba](Figures/Models/DyTSwiG-Mamba.png)
Module illustration of proposed SwiGLUformer.
![module_SwiGLUformer](Figures/Models/SwiGLUformer.png)
## Efficiency Comparison
The efficiency comparison with other state-of-the-art metheds. Our proposed models are with a pentagram mark.
![comparison](Figures/Comparison/Efficiency.png)

## Visualization
For DBPD module, the spectral visualization uses p232_020.wav and p232_020.wav from VB+DEMAND dataset, respectively.
![visualization_DBPD_1](/Figures/Visualization/DBPD/p232_020.png)
![visualization_DBPD_2](/Figures/Visualization/DBPD/p257_020.png)
For Mandarin enhancement results with different denoising models, the spectral visualization uses D21_866.wav from THCHS-30 dataset.
![visualization_zh-models](/Figures/Visualization/Enhanced-zh/D21_866.png)

## Audio Demos
Visit our [Demo Page](https://Yj-Xiong.github.io/DyTSwiG-SE) for English and Mandarin speech enhancement comparisons.

## Acknowledgements
We referred to [PrimeK-Net](https://github.com/huaidanquede/PrimeK-Net/).
