#!/usr/bin/env python3
"""
TTS 调试版本 - 查看详细错误
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
import uvicorn
import yaml
import io
import wave
import traceback
import numpy as np
import torch

from synthesize_tensorrt_fixed import TensorRTInference, preprocess_mandarin, synthesize_segment, device
from utils.model import get_vocoder

PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50

state = {"trt": None, "vocoder": None, "config": None, "sr": 22050}

def init():
    print("="*60)
    with open(PREPROCESS_CONFIG, "r") as f:
        state["config"] = yaml.load(f, Loader=yaml.FullLoader)
    with open(MODEL_CONFIG, "r") as f:
        mc = yaml.load(f, Loader=yaml.FullLoader)
    state["sr"] = state["config"]["preprocessing"]["audio"]["sampling_rate"]
    print(f"采样率: {state['sr']} Hz")
    state["vocoder"] = get_vocoder(mc, device)
    state["trt"] = TensorRTInference(ENGINE_PATH, state["config"])
    print("✅ 初始化完成")
    print("="*60)

app = FastAPI()

@app.on_event("startup")
def startup():
    init()

@app.get("/")
def root():
    return {"msg": "TTS Debug API"}

@app.get("/tts")
def tts_debug(text: str = "你好", speaker_id: int = 0):
    """调试版本 - 返回详细信息"""
    try:
        print(f"\n{'='*60}")
        print(f"收到请求: text='{text}', speaker_id={speaker_id}")
        
        # 1. 文本预处理
        print("\n[1] 文本预处理...")
        seq = preprocess_mandarin(text, state["config"])
        print(f"    音素数: {len(seq)}")
        print(f"    音素: {seq[:20]}...")
        
        if len(seq) == 0:
            return PlainTextResponse("错误: 音素序列为空", status_code=400)
        
        # 2. TTS 推理
        print("\n[2] TTS 推理...")
        if len(seq) > MAX_SEQ_LEN:
            print(f"    分段处理: {(len(seq) + MAX_SEQ_LEN - 1) // MAX_SEQ_LEN} 段")
            wavs = []
            for i in range(0, len(seq), MAX_SEQ_LEN):
                seg = seq[i:i+MAX_SEQ_LEN]
                print(f"    处理段 {i//MAX_SEQ_LEN + 1}...")
                wav = synthesize_segment(state["trt"], seg, speaker_id, MAX_SEQ_LEN, state["vocoder"])
                print(f"    段音频形状: {wav.shape}, 范围: [{wav.min():.3f}, {wav.max():.3f}]")
                wavs.append(wav)
            wav = np.concatenate(wavs)
        else:
            wav = synthesize_segment(state["trt"], seq, speaker_id, MAX_SEQ_LEN, state["vocoder"])
        
        print(f"\n[3] 音频信息:")
        print(f"    形状: {wav.shape}")
        print(f"    数据类型: {wav.dtype}")
        print(f"    范围: [{wav.min():.6f}, {wav.max():.6f}]")
        print(f"    均值: {wav.mean():.6f}")
        print(f"    标准差: {wav.std():.6f}")
        
        # 3. 转换为 WAV
        print("\n[4] 转换为 WAV...")
        
        # 确保是 float32
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
            print(f"    转换类型为 float32")
        
        # 归一化
        max_val = np.max(np.abs(wav))
        if max_val > 1.0:
            wav = wav / max_val
            print(f"    归一化到 [-1, 1], 原最大值: {max_val}")
        
        # 转换为 16-bit
        wav_int16 = (wav * 32767).astype(np.int16)
        print(f"    转换为 int16, 范围: [{wav_int16.min()}, {wav_int16.max()}]")
        
        # 创建 WAV
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(state["sr"])
            wf.writeframes(wav_int16.tobytes())
        
        wav_bytes = buf.getvalue()
        print(f"    WAV 大小: {len(wav_bytes)} bytes")
        print(f"{'='*60}\n")
        
        # 返回 WAV 文件
        from fastapi.responses import Response
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-Audio-Info": f"samples={len(wav)},sr={state['sr']}"
            }
        )
        
    except Exception as e:
        error_msg = f"错误: {str(e)}\n\n详细堆栈:\n{traceback.format_exc()}"
        print(error_msg)
        return PlainTextResponse(error_msg, status_code=500)

if __name__ == "__main__":
    print("\n🔧 TTS Debug Server")
    print("   端口: 8000\n")
    uvicorn.run("tts_debug:app", host="0.0.0.0", port=8000)
