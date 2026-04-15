#!/usr/bin/env python3
"""
TTS 服务 - 常驻内存，直接调用

使用方法：
    1. 启动服务（只执行一次）：
       python3 tts_service.py
    
    2. 在另一个 Python 程序中调用：
       from tts_service import speak
       speak("你好世界")
"""
import os
import sys
import yaml
import io
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import wave
import subprocess

from synthesize_tensorrt_fixed import (
    TensorRTInference, preprocess_mandarin, synthesize_segment, device
)
from utils.model import get_vocoder

# ==================== 配置 ====================
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50

# ==================== 全局变量（服务状态）====================
_service = None
_lock = threading.Lock()


class TTSService:
    """TTS服务类 - 只初始化一次"""
    
    def __init__(self):
        self.trt = None
        self.vocoder = None
        self.config = None
        self.sample_rate = 22050
        self.initialized = False
        
    def init(self):
        """初始化模型"""
        if self.initialized:
            return
            
        print("=" * 50)
        print("正在加载 TTS 模型...")
        
        # 加载配置
        with open(PREPROCESS_CONFIG, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sample_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        
        # 加载声码器和TensorRT
        print("  加载声码器...")
        self.vocoder = get_vocoder(model_config, device)
        
        print("  加载 TensorRT 引擎...")
        self.trt = TensorRTInference(ENGINE_PATH, self.config)
        
        self.initialized = True
        print("✅ TTS 模型加载完成！")
        print("=" * 50)
        
    def synthesize(self, text, speaker_id=0):
        """合成音频，返回WAV字节数据"""
        with _lock:  # 线程安全
            if not self.initialized:
                self.init()
            
            # 文本预处理
            sequence = preprocess_mandarin(text, self.config)
            if len(sequence) == 0:
                print("警告：音素序列为空")
                return None
            
            # 分段合成
            if len(sequence) > MAX_SEQ_LEN:
                wavs = []
                for i in range(0, len(sequence), MAX_SEQ_LEN):
                    segment = sequence[i:i+MAX_SEQ_LEN]
                    original_len = len(segment)
                    wav = synthesize_segment(
                        self.trt, segment, speaker_id, MAX_SEQ_LEN, 
                        self.vocoder, original_len
                    )
                    wavs.append(wav)
                wav = np.concatenate(wavs)
            else:
                original_len = len(sequence)
                wav = synthesize_segment(
                    self.trt, sequence, speaker_id, MAX_SEQ_LEN,
                    self.vocoder, original_len
                )
            
            # 转WAV格式
            wav = wav.astype(np.float32)
            max_val = np.max(np.abs(wav))
            if max_val > 1.0:
                wav = wav / max_val
            wav_int16 = (wav * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(wav_int16.tobytes())
            
            return buf.getvalue()


def _get_service():
    """获取服务实例（单例模式）"""
    global _service
    if _service is None:
        _service = TTSService()
        _service.init()
    return _service


# ==================== 对外接口 ====================

def speak(text, speaker_id=0):
    """
    合成并播放语音
    
    参数:
        text: 要合成的文本
        speaker_id: 说话人ID，默认0
    
    返回:
        True/False 表示是否成功
    
    示例:
        speak("你好世界")
        speak("今天天气不错", speaker_id=1)
    """
    try:
        service = _get_service()
        audio_data = service.synthesize(text, speaker_id)
        
        if audio_data is None:
            return False
        
        # 保存临时文件并播放
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            tmp_file = f.name
        
        # 播放音频
        subprocess.run(["aplay", tmp_file], capture_output=True)
        
        # 删除临时文件
        os.remove(tmp_file)
        
        return True
        
    except Exception as e:
        print(f"TTS错误: {e}")
        return False


def synthesize_to_file(text, output_file, speaker_id=0):
    """
    合成语音并保存到文件
    
    参数:
        text: 要合成的文本
        output_file: 输出文件路径，如 "output.wav"
        speaker_id: 说话人ID，默认0
    
    返回:
        True/False 表示是否成功
    
    示例:
        synthesize_to_file("你好", "hello.wav")
    """
    try:
        service = _get_service()
        audio_data = service.synthesize(text, speaker_id)
        
        if audio_data is None:
            return False
        
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        return True
        
    except Exception as e:
        print(f"TTS错误: {e}")
        return False


# ==================== 测试 ====================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TTS 服务测试")
    print("=" * 50)
    
    # 测试1：直接播放
    print("\n测试1: 直接播放")
    speak("你好，我是TTS服务")
    
    # 测试2：保存到文件
    print("\n测试2: 保存到文件")
    if synthesize_to_file("这是测试音频", "test.wav"):
        print("✅ 已保存到 test.wav")
        print("   播放: aplay test.wav")
    
    # 测试3：连续调用（测试速度）
    print("\n测试3: 连续调用")
    import time
    texts = ["第一句话", "第二句话", "第三句话"]
    for text in texts:
        start = time.time()
        speak(text)
        print(f"  '{text}': {(time.time()-start)*1000:.0f}ms")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
