#!/usr/bin/env python3
"""
FastSpeech2 TTS API 服务 (简化版)
支持 PyTorch 和 ONNX 两种推理方式
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


app = FastAPI(title="FastSpeech2 TTS API", version="1.0.0")


# ==================== API 模型 ====================
class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0


class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_path: str = None
    duration: float = None


# ==================== API 路由 ====================
@app.get("/")
def root():
    """根路径 - 服务状态"""
    return {
        "service": "FastSpeech2 TTS API",
        "version": "1.0.0",
        "endpoints": {
            "pth": "/tts/pth - PyTorch 推理",
            "onnx": "/tts/onnx - ONNX 推理",
            "health": "/health - 健康检查"
        }
    }


@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/tts/pth", response_model=TTSResponse)
def tts_pytorch(request: TTSRequest):
    """
    PyTorch 模型推理
    
    - **text**: 要合成的文本（中文）
    - **speaker_id**: 说话人ID（默认0）
    """
    try:
        print(f"[API] PyTorch 推理: '{request.text}'")
        
        # 调用命令行脚本
        cmd = [
            "python", "inference/synthesize.py",
            "--text", request.text,
            "--restore_step", "600000",
            "--mode", "single",
            "-p", "config/AISHELL3/preprocess.yaml",
            "-m", "config/AISHELL3/model.yaml",
            "-t", "config/AISHELL3/train.yaml",
            "--speaker_id", str(request.speaker_id)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print(f"[API] 错误: {result.stderr}")
            raise HTTPException(status_code=500, detail=result.stderr)
        
        # 查找生成的音频文件
        output_file = f"output/result/AISHELL3/{request.text}.wav"
        
        if os.path.exists(output_file):
            import soundfile as sf
            wav, sr = sf.read(output_file)
            duration = len(wav) / sr
            
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_path=output_file,
                duration=duration
            )
        else:
            raise HTTPException(status_code=500, detail="音频文件未生成")
        
    except Exception as e:
        print(f"[API] 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/onnx", response_model=TTSResponse)
def tts_onnx(request: TTSRequest):
    """
    ONNX 模型推理
    
    - **text**: 要合成的文本（中文）
    - **speaker_id**: 说话人ID（默认0）
    """
    try:
        print(f"[API] ONNX 推理: '{request.text}'")
        
        # 步骤1: ONNX 推理生成 mel
        cmd1 = [
            "python", "deploy/inference_tensorrt_onnx.py",
            "--text", request.text
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True, encoding='utf-8')
        
        if result1.returncode != 0:
            print(f"[API] ONNX 推理错误: {result1.stderr}")
            raise HTTPException(status_code=500, detail=result1.stderr)
        
        # 步骤2: 生成音频
        mel_file = f"tensorrt_output/{request.text}_mel.npy"
        
        if not os.path.exists(mel_file):
            raise HTTPException(status_code=500, detail="mel 文件未生成")
        
        cmd2 = [
            "python", "deploy/generate_audio_from_mel.py",
            mel_file
        ]
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8')
        
        if result2.returncode != 0:
            print(f"[API] 音频生成错误: {result2.stderr}")
            raise HTTPException(status_code=500, detail=result2.stderr)
        
        # 查找生成的音频文件
        output_file = f"tensorrt_output/{request.text}.wav"
        
        if os.path.exists(output_file):
            import soundfile as sf
            wav, sr = sf.read(output_file)
            duration = len(wav) / sr
            
            return TTSResponse(
                success=True,
                message="合成成功",
                audio_path=output_file,
                duration=duration
            )
        else:
            raise HTTPException(status_code=500, detail="音频文件未生成")
        
    except Exception as e:
        print(f"[API] 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 启动 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("FastSpeech2 TTS API 服务")
    print("=" * 60)
    print("端点:")
    print("  - http://127.0.0.1:8000/           服务状态")
    print("  - http://127.0.0.1:8000/health     健康检查")
    print("  - POST /tts/pth                    PyTorch 推理")
    print("  - POST /tts/onnx                   ONNX 推理")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
