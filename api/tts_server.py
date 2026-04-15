#!/usr/bin/env python3
"""
TensorRT TTS FastAPI Server - 修复版
确保输出标准 WAV 格式音频
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import yaml
import io
import base64
import time
import wave
from typing import Optional
import numpy as np
import soundfile as sf
import torch

from synthesize_tensorrt_fixed import TensorRTInference, preprocess_mandarin, synthesize_segment, device
from utils.model import get_vocoder

# ==================== 配置 ====================
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50

# ==================== 全局状态 ====================
state = {"trt": None, "vocoder": None, "config": None, "sr": 22050}

# ==================== 数据模型 ====================
class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0

# ==================== 初始化 ====================
def init():
    print("="*60)
    print("🚀 初始化 TensorRT TTS")
    with open(PREPROCESS_CONFIG, "r") as f:
        state["config"] = yaml.load(f, Loader=yaml.FullLoader)
    with open(MODEL_CONFIG, "r") as f:
        mc = yaml.load(f, Loader=yaml.FullLoader)
    state["sr"] = state["config"]["preprocessing"]["audio"]["sampling_rate"]
    print(f"   采样率: {state['sr']} Hz")
    state["vocoder"] = get_vocoder(mc, device)
    state["trt"] = TensorRTInference(ENGINE_PATH, state["config"])
    print("✅ 模型加载完成")
    print("="*60)

# ==================== TTS核心 ====================
def synth(text, sid=0):
    """执行TTS推理"""
    print(f"\n📝 文本: {text}")
    seq = preprocess_mandarin(text, state["config"])
    print(f"   音素数: {len(seq)}")
    
    if len(seq) == 0:
        raise ValueError("音素序列为空")
    
    t0 = time.time()
    
    if len(seq) > MAX_SEQ_LEN:
        wavs = []
        num = (len(seq) + MAX_SEQ_LEN - 1) // MAX_SEQ_LEN
        print(f"   分段: {num} 段")
        for i in range(0, len(seq), MAX_SEQ_LEN):
            seg = seq[i:i+MAX_SEQ_LEN]
            wav = synthesize_segment(state["trt"], seg, sid, MAX_SEQ_LEN, state["vocoder"])
            wavs.append(wav)
        wav = np.concatenate(wavs)
    else:
        wav = synthesize_segment(state["trt"], seq, sid, MAX_SEQ_LEN, state["vocoder"])
    
    infer_time = (time.time() - t0) * 1000
    duration = len(wav) / state["sr"]
    print(f"✅ 完成: {duration:.2f}s | {infer_time:.1f}ms")
    
    return wav, infer_time, duration

def wav_to_bytes(wav_data, sample_rate):
    """
    将 numpy 音频数组转换为标准 WAV 格式的字节
    确保格式: PCM 16-bit, 单声道
    """
    # 确保音频数据是 float32 且在 [-1, 1] 范围内
    if wav_data.dtype != np.float32:
        wav_data = wav_data.astype(np.float32)
    
    # 归一化到 [-1, 1]
    max_val = np.max(np.abs(wav_data))
    if max_val > 1.0:
        wav_data = wav_data / max_val
    
    # 转换为 16-bit PCM
    wav_int16 = (wav_data * 32767).astype(np.int16)
    
    # 创建 WAV 文件字节
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(wav_int16.tobytes())
    
    buf.seek(0)
    return buf.read()

# ==================== FastAPI ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init()
    yield
    print("\n🛑 服务关闭")

app = FastAPI(title="TensorRT TTS", lifespan=lifespan)

@app.get("/")
def root():
    return {
        "service": "TensorRT TTS API",
        "version": "2.1.0",
        "endpoints": {
            "GET /health": "健康检查",
            "POST /tts": "TTS合成 (返回base64)",
            "GET /tts": "TTS合成 (返回音频流)"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": state["trt"] is not None,
        "sample_rate": state["sr"]
    }

@app.post("/tts")
def post_tts(r: TTSRequest):
    """POST方式：返回JSON包含base64音频"""
    try:
        wav, it, dur = synth(r.text, r.speaker_id)
        
        # 使用标准 WAV 格式
        wav_bytes = wav_to_bytes(wav, state["sr"])
        b64 = base64.b64encode(wav_bytes).decode()
        
        return {
            "success": True,
            "message": "合成成功",
            "audio_base64": b64,
            "inference_time_ms": round(it, 2),
            "duration_sec": round(dur, 2),
            "sample_rate": state["sr"]
        }
    except Exception as e:
        import traceback
        print(f"❌ 错误: {e}")
        traceback.print_exc()
        return {"success": False, "message": str(e)}

@app.get("/tts")
def get_tts(text: str, speaker_id: int = 0):
    """GET方式：直接返回WAV音频流"""
    try:
        wav, it, dur = synth(text, speaker_id)
        
        # 使用标准 WAV 格式
        wav_bytes = wav_to_bytes(wav, state["sr"])
        
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "X-Inference-Time": f"{it:.2f}",
                "X-Duration": f"{dur:.2f}",
                "X-Sample-Rate": str(state["sr"])
            }
        )
    except Exception as e:
        import traceback
        print(f"❌ 错误: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# ==================== 启动 ====================
if __name__ == "__main__":
    print("\n🔧 TensorRT TTS Server v2.1")
    print(f"   端口: 8000")
    print(f"   引擎: {ENGINE_PATH}")
    print(f"   设备: {device}\n")
    
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8000, reload=False)
