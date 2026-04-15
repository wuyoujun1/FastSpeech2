#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import io
import threading
import subprocess
import tempfile

import torch
import numpy as np
import wave

from synthesize_tensorrt_fixed import (
    TensorRTInference, preprocess_mandarin, synthesize_segment, device
)
from utils.model import get_vocoder

PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50


class TTS:
    """TTS接口类 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.trt = None
        self.vocoder = None
        self.config = None
        self.sample_rate = 22050
        self._model_lock = threading.Lock()
        
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        print("[TTS] 正在加载模型...")
        with open(PREPROCESS_CONFIG, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sample_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.vocoder = get_vocoder(model_config, device)
        self.trt = TensorRTInference(ENGINE_PATH, self.config)
        print("[TTS] 模型加载完成！")
    
    def synthesize(self, text, speaker_id=0, volume=1.0):
        """
        合成语音
        
        Args:
            text: 要合成的文本
            speaker_id: 说话人ID
            volume: 音量增益，1.0为原始音量，>1.0增大音量，<1.0减小音量
        """
        with self._model_lock:
            try:
                sequence = preprocess_mandarin(text, self.config)
                if len(sequence) == 0:
                    return None
                
                if len(sequence) > MAX_SEQ_LEN:
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
                
                wav = wav.astype(np.float32)
                max_val = np.max(np.abs(wav))
                if max_val > 1.0:
                    wav = wav / max_val
                
                # 应用音量增益
                wav = wav * volume
                # 限制在有效范围内，防止削波
                wav = np.clip(wav, -1.0, 1.0)
                
                wav_int16 = (wav * 32767).astype(np.int16)
                
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(wav_int16.tobytes())
                
                return buf.getvalue()
            except Exception as e:
                print(f"[TTS] 错误: {e}")
                return None
    
    def speak(self, text, speaker_id=0, device="plughw:0,0"):
        audio_data = self.synthesize(text, speaker_id)
        if audio_data is None:
            return False
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp_file = f.name
            
            # 播放（指定设备）
            cmd = ["aplay", "-D", device, tmp_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[TTS] 设备错误，尝试默认设备: {result.stderr}")
                # 尝试默认设备
                subprocess.run(["aplay", tmp_file], capture_output=True)
            
            # 清理临时文件
            def cleanup():
                import time
                time.sleep(5)
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            threading.Thread(target=cleanup, daemon=True).start()
            
            return True
        except Exception as e:
            print(f"[TTS] 播放错误: {e}")
            return False
    
    def save(self, text, output_file, speaker_id=0):
        audio_data = self.synthesize(text, speaker_id)
        if audio_data is None:
            return False
        
        try:
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            return True
        except Exception as e:
            print(f"[TTS] 保存错误: {e}")
            return False


_tts_instance = None

def _get_tts():
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTS()
    return _tts_instance

def speak(text, speaker_id=0):
    return _get_tts().speak(text, speaker_id)

def save(text, output_file, speaker_id=0):
    return _get_tts().save(text, output_file, speaker_id)
