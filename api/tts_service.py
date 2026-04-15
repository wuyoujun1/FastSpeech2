#!/usr/bin/env python3
"""
FastSpeech2 TTS API 服务
支持 PyTorch 和 ONNX 两种推理方式
"""

import os
import sys
import io
import base64
import tempfile
import threading
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# 导入 TTS 相关模块
from text import text_to_sequence
from pypinyin import pinyin, Style
import re


app = FastAPI(title="FastSpeech2 TTS API", version="1.0.0")


# ==================== 配置 ====================
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
CKPT_PATH = "output/ckpt/AISHELL3/600000.pth.tar"
ONNX_PATH = "onnx/fastspeech2_tensorrt.onnx"
SPEAKER_ID = 0


# ==================== 文本预处理 ====================
def read_lexicon(lex_path):
    """读取词典文件"""
    lexicon = {}
    with open(lex_path, encoding='utf-8') as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_mandarin(text, preprocess_config):
    """预处理中文文本"""
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    
    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
    
    phones = "{" + " ".join(phones) + "}"
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return sequence


# ==================== TTS 服务类 ====================
class TTSService:
    """TTS 服务管理类"""
    
    def __init__(self):
        self.pytorch_model = None
        self.vocoder = None
        self.onnx_session = None
        self.config = None
        self.preprocess_config = None
        self.model_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        
    def load_pytorch_model(self):
        """加载 PyTorch 模型"""
        if self.pytorch_model is not None:
            return
            
        print("[TTS] 加载 PyTorch 模型...")
        
        # 加载配置
        with open(PREPROCESS_CONFIG, "r", encoding='utf-8') as f:
            self.preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r", encoding='utf-8') as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 加载模型
        from model.fastspeech2 import FastSpeech2
        from utils.model import get_vocoder
        
        self.pytorch_model = FastSpeech2(self.preprocess_config, self.model_config).to(self.device)
        
        ckpt = torch.load(CKPT_PATH, map_location=self.device)
        self.pytorch_model.load_state_dict(ckpt["model"])
        self.pytorch_model.eval()
        
        # 加载声码器
        self.vocoder = get_vocoder(self.model_config, self.device)
        
        print("[TTS] PyTorch 模型加载完成")
    
    def load_onnx_model(self):
        """加载 ONNX 模型"""
        if self.onnx_session is not None:
            return
            
        print("[TTS] 加载 ONNX 模型...")
        
        import onnxruntime as ort
        
        # 加载配置
        with open(PREPROCESS_CONFIG, "r", encoding='utf-8') as f:
            self.preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r", encoding='utf-8') as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 加载声码器
        from utils.model import get_vocoder
        self.vocoder = get_vocoder(self.model_config, self.device)
        
        # 加载 ONNX
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(ONNX_PATH, sess_options, providers=providers)
        
        print("[TTS] ONNX 模型加载完成")
    
    def synthesize_pytorch(self, text):
        """PyTorch 推理"""
        with self._lock:
            if self.pytorch_model is None:
                self.load_pytorch_model()
        
        # 预处理文本
        text_sequence = preprocess_mandarin(text, self.preprocess_config)
        
        # 准备输入
        texts = torch.LongTensor([text_sequence]).to(self.device)
        src_lens = torch.LongTensor([len(text_sequence)]).to(self.device)
        speakers = torch.LongTensor([SPEAKER_ID]).to(self.device)
        max_src_len = torch.LongTensor([len(text_sequence)]).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.pytorch_model(
                speakers, texts, src_lens, max_src_len,
                mel_lens=None, max_mel_len=None,
                p_targets=None, e_targets=None, d_targets=None
            )
        
        mel_postnet = output[1]
        mel_len = output[9][0]
        
        # 声码器生成音频
        from utils.model import vocoder_infer
        wavs = vocoder_infer(
            mel_postnet[:, :mel_len, :],
            self.vocoder,
            self.model_config,
            self.preprocess_config
        )
        
        return wavs[0], mel_postnet[0, :mel_len, :].cpu().numpy()
    
    def synthesize_onnx(self, text):
        """ONNX 推理"""
        with self._lock:
            if self.onnx_session is None:
                self.load_onnx_model()
        
        # 预处理文本
        text_sequence = preprocess_mandarin(text, self.preprocess_config)
        
        # 准备输入
        texts = np.expand_dims(text_sequence, axis=0).astype(np.int64)
        src_lens = np.array([len(text_sequence)], dtype=np.int64)
        
        # 推理
        outputs = self.onnx_session.run(None, {
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
        from utils.model import vocoder_infer
        wavs = vocoder_infer(
            mel_tensor,
            self.vocoder,
            self.model_config,
            self.preprocess_config
        )
        
        return wavs[0], mel_valid


# 全局 TTS 服务实例
tts_service = TTSService()


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
    return {
        "status": "healthy",
        "pytorch_loaded": tts_service.pytorch_model is not None,
        "onnx_loaded": tts_service.onnx_session is not None
    }


@app.post("/tts/pth", response_model=TTSResponse)
def tts_pytorch(request: TTSRequest):
    """
    PyTorch 模型推理
    
    - **text**: 要合成的文本（中文）
    - **speaker_id**: 说话人ID（默认0）
    """
    try:
        print(f"[API] PyTorch 推理: '{request.text}'")
        
        wav, mel = tts_service.synthesize_pytorch(request.text)
        
        # 保存音频
        output_dir = "api_output"
        os.makedirs(output_dir, exist_ok=True)
        import hashlib
        text_hash = hashlib.md5(request.text.encode()).hexdigest()[:8]
        wav_path = os.path.join(output_dir, f"pth_{text_hash}.wav")
        
        sf.write(
            wav_path,
            wav,
            tts_service.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        )
        
        duration = len(wav) / tts_service.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        
        print(f"[API] 音频已保存: {wav_path}, 时长: {duration:.2f}s")
        
        return TTSResponse(
            success=True,
            message="合成成功",
            audio_path=wav_path,
            duration=duration
        )
        
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
        
        wav, mel = tts_service.synthesize_onnx(request.text)
        
        # 保存音频
        output_dir = "api_output"
        os.makedirs(output_dir, exist_ok=True)
        import hashlib
        text_hash = hashlib.md5(request.text.encode()).hexdigest()[:8]
        wav_path = os.path.join(output_dir, f"onnx_{text_hash}.wav")
        
        sf.write(
            wav_path,
            wav,
            tts_service.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        )
        
        duration = len(wav) / tts_service.preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        
        print(f"[API] 音频已保存: {wav_path}, 时长: {duration:.2f}s")
        
        return TTSResponse(
            success=True,
            message="合成成功",
            audio_path=wav_path,
            duration=duration
        )
        
    except Exception as e:
        print(f"[API] 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """获取音频文件"""
    wav_path = os.path.join("api_output", filename)
    if os.path.exists(wav_path):
        return FileResponse(wav_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="文件不存在")


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
