"""
FastSpeech2 TTS 本地 API
最简单的使用方式：
    from tts_api import tts
    tts("你好世界")
"""

import os
import sys
import torch
import yaml
import numpy as np
import soundfile as sf
from pathlib import Path

# 配置路径
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
CKPT_PATH = "output/ckpt/AISHELL3/600000.pth.tar"
ONNX_PATH = "onnx/fastspeech2_tensorrt.onnx"

# 全局模型缓存（只加载一次）
_pytorch_model = None
_vocoder = None
_onnx_session = None
_preprocess_config = None
_model_config = None
_device = None


def _init_pytorch():
    """初始化 PyTorch 模型（只执行一次）"""
    global _pytorch_model, _vocoder, _preprocess_config, _model_config, _device
    
    if _pytorch_model is not None:
        return
    
    print("[TTS] 加载 PyTorch 模型...")
    
    from model.fastspeech2 import FastSpeech2
    from utils.model import get_vocoder
    
    # 加载配置
    with open(PREPROCESS_CONFIG, "r", encoding='utf-8') as f:
        _preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(MODEL_CONFIG, "r", encoding='utf-8') as f:
        _model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 设备
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    _pytorch_model = FastSpeech2(_preprocess_config, _model_config).to(_device)
    ckpt = torch.load(CKPT_PATH, map_location=_device)
    _pytorch_model.load_state_dict(ckpt["model"])
    _pytorch_model.eval()
    
    # 加载声码器
    _vocoder = get_vocoder(_model_config, _device)
    
    print("[TTS] PyTorch 模型加载完成")


def _init_onnx():
    """初始化 ONNX 模型（只执行一次）"""
    global _onnx_session, _vocoder, _preprocess_config, _model_config, _device
    
    if _onnx_session is not None:
        return
    
    print("[TTS] 加载 ONNX 模型...")
    
    import onnxruntime as ort
    from utils.model import get_vocoder
    
    # 加载配置
    with open(PREPROCESS_CONFIG, "r", encoding='utf-8') as f:
        _preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(MODEL_CONFIG, "r", encoding='utf-8') as f:
        _model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 设备
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载声码器
    _vocoder = get_vocoder(_model_config, _device)
    
    # 加载 ONNX
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    _onnx_session = ort.InferenceSession(ONNX_PATH, sess_options, providers=providers)
    
    print("[TTS] ONNX 模型加载完成")


def _text_to_sequence(text):
    """文本转音素序列（中文）"""
    from text import text_to_sequence
    from pypinyin import pinyin, Style
    
    # 确保配置已加载
    global _preprocess_config
    if _preprocess_config is None:
        with open(PREPROCESS_CONFIG, "r", encoding='utf-8') as f:
            _preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 中文处理
    text = text.strip()
    if not text:
        return []
    
    # 添加中文标点
    if text[-1] not in "。！？，、；：""''（）《》【】":
        text += "。"
    
    # 读取词典
    lexicon_path = _preprocess_config["path"]["lexicon_path"]
    lexicon = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lexicon[parts[0]] = parts[1:]
    
    # 拼音转换
    phones = []
    pinyins = [
        p[0]
        for p in pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
    
    phones = "{" + " ".join(phones) + "}"
    print(f"[TTS] 文本: {text}")
    print(f"[TTS] 音素: {phones}")
    
    # 转换为音素序列
    sequence = text_to_sequence(phones, [])
    return sequence


def tts_pth(text, speaker_id=0, output_path=None):
    """
    PyTorch 模型推理
    
    Args:
        text: 要合成的中文文本
        speaker_id: 说话人ID（默认0）
        output_path: 输出音频路径（默认自动生成）
    
    Returns:
        音频文件路径
    """
    _init_pytorch()
    
    from utils.model import vocoder_infer
    
    # 文本预处理
    text_sequence = _text_to_sequence(text)
    if len(text_sequence) == 0:
        raise ValueError("文本为空或无法处理")
    
    # 准备输入
    texts = torch.LongTensor([text_sequence]).to(_device)
    src_lens = torch.LongTensor([len(text_sequence)]).to(_device)
    speakers = torch.LongTensor([speaker_id]).to(_device)
    max_src_len = len(text_sequence)  # 整数，不是 tensor
    
    # 推理
    with torch.no_grad():
        output = _pytorch_model(
            speakers=speakers,
            texts=texts,
            src_lens=src_lens,
            max_src_len=max_src_len,
            mel_lens=None, max_mel_len=None,
            p_targets=None, e_targets=None, d_targets=None
        )
    
    mel_postnet = output[1]
    mel_len = output[9][0]
    
    # mel 维度: [batch, time, mel] -> [batch, mel, time]
    mel_for_vocoder = mel_postnet[:, :mel_len, :].transpose(1, 2)
    
    # 声码器生成音频
    wavs = vocoder_infer(
        mel_for_vocoder,
        _vocoder,
        _model_config,
        _preprocess_config
    )
    
    wav = wavs[0]
    sr = _preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    
    # 保存音频
    if output_path is None:
        output_dir = "tts_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{text[:20]}.wav")
    
    sf.write(output_path, wav, sr)
    
    print(f"[TTS] 音频已保存: {output_path}, 时长: {len(wav)/sr:.2f}s")
    
    return output_path


def tts_onnx(text, speaker_id=0, output_path=None):
    """
    ONNX 模型推理
    
    Args:
        text: 要合成的中文文本
        speaker_id: 说话人ID（默认0，当前未使用）
        output_path: 输出音频路径（默认自动生成）
    
    Returns:
        音频文件路径
    """
    _init_onnx()
    
    from utils.model import vocoder_infer
    
    # 文本预处理
    text_sequence = _text_to_sequence(text)
    if len(text_sequence) == 0:
        raise ValueError("文本为空或无法处理")
    
    # 准备输入
    texts = np.expand_dims(text_sequence, axis=0).astype(np.int64)
    src_lens = np.array([len(text_sequence)], dtype=np.int64)
    
    # 推理
    outputs = _onnx_session.run(None, {
        'texts': texts,
        'src_lens': src_lens,
    })
    
    mel_output = outputs[0]
    mel_len = int(outputs[1][0])
    
    # 截取有效部分 [time, mel] -> [mel, time]
    mel_valid = mel_output[0, :mel_len, :]
    mel_tensor = torch.from_numpy(mel_valid).unsqueeze(0).float()  # [1, time, mel]
    mel_tensor = mel_tensor.transpose(1, 2)  # [1, mel, time]
    
    # 声码器生成音频
    wavs = vocoder_infer(
        mel_tensor,
        _vocoder,
        _model_config,
        _preprocess_config
    )
    
    wav = wavs[0]
    sr = _preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    
    # 保存音频
    if output_path is None:
        output_dir = "tts_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{text[:20]}.wav")
    
    sf.write(output_path, wav, sr)
    
    print(f"[TTS] 音频已保存: {output_path}, 时长: {len(wav)/sr:.2f}s")
    
    return output_path


# 默认使用 PyTorch 模型
tts = tts_pth


if __name__ == "__main__":
    # 测试
    print("测试 PyTorch 推理...")
    tts_pth("你好世界")
    
    print("\n测试 ONNX 推理...")
    tts_onnx("今天天气不错")
