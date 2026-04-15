#!/usr/bin/env python3
"""
TTS 服务 - 调试用版本（保留音频文件）
"""
import os
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

PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50

_service = None
_lock = threading.Lock()


class TTSService:
    def __init__(self):
        self.trt = None
        self.vocoder = None
        self.config = None
        self.sample_rate = 22050
        self.initialized = False

    def init(self):
        if self.initialized:
            return
        print("正在加载 TTS 模型...")
        with open(PREPROCESS_CONFIG, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sample_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.vocoder = get_vocoder(model_config, device)
        self.trt = TensorRTInference(ENGINE_PATH, self.config)
        self.initialized = True
        print("模型加载完成！")

    def synthesize(self, text, speaker_id=0):
        with _lock:
            if not self.initialized:
                self.init()
            
            print(f"  文本预处理: '{text}'")
            sequence = preprocess_mandarin(text, self.config)
            print(f"  音素数: {len(sequence)}")
            
            if len(sequence) == 0:
                print("  错误：音素序列为空")
                return None

            if len(sequence) > MAX_SEQ_LEN:
                print(f"  分段合成: {(len(sequence) + MAX_SEQ_LEN - 1) // MAX_SEQ_LEN} 段")
                wavs = []
                for i in range(0, len(sequence), MAX_SEQ_LEN):
                    segment = sequence[i:i+MAX_SEQ_LEN]
                    original_len = len(segment)
                    wav = synthesize_segment(self.trt, segment, speaker_id, MAX_SEQ_LEN, self.vocoder, original_len)
                    wavs.append(wav)
                wav = np.concatenate(wavs)
            else:
                original_len = len(sequence)
                wav = synthesize_segment(self.trt, sequence, speaker_id, MAX_SEQ_LEN, self.vocoder, original_len)
            
            print(f"  音频长度: {len(wav)} 样本")
            
            wav = wav.astype(np.float32)
            max_val = np.max(np.abs(wav))
            print(f"  最大振幅: {max_val:.4f}")
            if max_val > 1.0:
                wav = wav / max_val
            wav_int16 = (wav * 32767).astype(np.int16)
            
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(wav_int16.tobytes())
            
            audio_bytes = buf.getvalue()
            print(f"  WAV大小: {len(audio_bytes)} bytes")
            return audio_bytes


def _get_service():
    global _service
    if _service is None:
        _service = TTSService()
        _service.init()
    return _service


def speak(text, speaker_id=0, output_file=None):
    """
    合成并播放语音
    
    参数:
        text: 要合成的文本
        speaker_id: 说话人ID
        output_file: 如果指定，保存到该文件（不删除）
    """
    try:
        print(f"\n📝 TTS请求: '{text}'")
        service = _get_service()
        audio_data = service.synthesize(text, speaker_id)

        if audio_data is None:
            print("  ❌ 合成失败")
            return False

        # 保存文件
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"  💾 已保存: {output_file}")
            tmp_file = output_file
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp_file = f.name
            print(f"  💾 临时文件: {tmp_file}")

        # 播放
        print(f"  🔊 播放音频...")
        result = subprocess.run(["aplay", tmp_file], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ 播放失败: {result.stderr}")
            return False
        
        print(f"  ✅ 播放完成")
        
        # 如果不是指定文件，询问是否删除
        if not output_file:
            print(f"  文件保留在: {tmp_file}")
            print(f"  播放命令: aplay {tmp_file}")

        return True

    except Exception as e:
        print(f"  ❌ TTS错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("TTS 调试测试")
    print("=" * 60)

    # 测试1：基本播放（保留文件）
    print("\n测试1: 基本播放")
    speak("你好世界", output_file="test1.wav")

    # 测试2：连续调用
    print("\n测试2: 连续调用")
    texts = ["第一句话", "第二句话"]
    for text in texts:
        start = time.time()
        speak(text)
        elapsed = (time.time() - start) * 1000
        print(f"   总耗时: {elapsed:.0f}ms")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("音频文件: test1.wav, /tmp/tmp*.wav")
    print("=" * 60)
