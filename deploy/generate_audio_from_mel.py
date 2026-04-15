"""
从梅尔频谱生成音频
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
import soundfile as sf
from utils.model import get_vocoder, vocoder_infer


def generate_audio(mel_path, output_path=None, config_path="config/AISHELL3/preprocess.yaml", model_config_path="config/AISHELL3/model.yaml"):
    """
    从梅尔频谱生成音频
    
    Args:
        mel_path: 梅尔频谱文件路径 (.npy)
        output_path: 输出音频路径，默认为同名 .wav
        config_path: 预处理配置文件
        model_config_path: 模型配置文件
    """
    print(f"加载梅尔频谱: {mel_path}")
    mel = np.load(mel_path)
    print(f"  形状: {mel.shape}")
    print(f"  范围: [{mel.min():.4f}, {mel.max():.4f}]")
    
    # 找到有效的 mel 长度（基于能量阈值）
    mel_energy = np.abs(mel).sum(axis=1)
    
    # 使用自适应阈值：中位数 + 2倍标准差
    energy_median = np.median(mel_energy)
    energy_std = np.std(mel_energy)
    threshold = max(energy_median + 2 * energy_std, 30)  # 最小阈值30
    
    valid_frames = np.where(mel_energy > threshold)[0]
    if len(valid_frames) > 0:
        start_frame = valid_frames[0]
        end_frame = valid_frames[-1] + 1
        valid_mel_len = end_frame - start_frame
        print(f"  能量阈值: {threshold:.2f}")
        print(f"  有效范围: [{start_frame}:{end_frame}]")
        print(f"  有效长度: {valid_mel_len} 帧")
    else:
        print(f"  警告: 未找到有效帧，使用全部")
        start_frame = 0
        valid_mel_len = mel.shape[0]
    
    # 截取有效部分
    mel_valid = mel[start_frame:end_frame]
    
    # 加载配置
    print(f"\n加载配置...")
    with open(config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 加载声码器
    print("加载声码器...")
    device = torch.device("cpu")
    vocoder = get_vocoder(model_config, device)
    
    # 转换 mel 为 tensor [batch, time, mel] -> [batch, mel, time]
    mel_tensor = torch.from_numpy(mel_valid).unsqueeze(0).float()
    mel_tensor = mel_tensor.transpose(1, 2)  # [1, 80, time]
    print(f"  Mel tensor shape: {mel_tensor.shape}")
    
    # 生成音频
    print("生成音频...")
    with torch.no_grad():
        wavs = vocoder_infer(
            mel_tensor,
            vocoder,
            model_config,
            preprocess_config,
        )
    
    wav = wavs[0]
    sample_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    
    # 保存音频
    if output_path is None:
        output_path = mel_path.replace("_mel.npy", ".wav")
    
    sf.write(output_path, wav, sample_rate)
    print(f"\n音频已保存: {output_path}")
    print(f"  时长: {len(wav) / sample_rate:.2f} 秒")
    print(f"  采样率: {sample_rate} Hz")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从梅尔频谱生成音频")
    parser.add_argument("mel", type=str, help="梅尔频谱文件路径 (.npy)")
    parser.add_argument("--output", type=str, default=None, help="输出音频路径")
    parser.add_argument("--config", type=str, default="config/AISHELL3/preprocess.yaml", help="预处理配置")
    parser.add_argument("--model-config", type=str, default="config/AISHELL3/model.yaml", help="模型配置")
    
    args = parser.parse_args()
    
    generate_audio(
        mel_path=args.mel,
        output_path=args.output,
        config_path=args.config,
        model_config_path=args.model_config,
    )
