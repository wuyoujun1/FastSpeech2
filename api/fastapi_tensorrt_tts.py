"""
FastAPI 封装 TensorRT TTS 服务
用于 Jetson 设备部署
"""

import os
import sys
import re
import time
import yaml
import io
import base64
import numpy as np
import torch
from typing import Optional
from contextlib import asynccontextmanager

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入 FastAPI
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# 导入 TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError as e:
    print(f"警告: TensorRT 或 PyCUDA 未安装: {e}")
    TRT_AVAILABLE = False

# 导入项目模块
from pypinyin import pinyin, Style
from text import text_to_sequence
from utils.model import get_vocoder
import soundfile as sf


# ============ 配置 ============
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
OUTPUT_DIR = "temp_output"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============ 请求/响应模型 ============
class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0


class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_base64: Optional[str] = None
    inference_time_ms: Optional[float] = None
    audio_duration_sec: Optional[float] = None


# ============ TensorRT 推理类 ============
class TensorRTInference:
    """TensorRT 推理类"""
    
    def __init__(self, engine_path, preprocess_config):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        print(f"加载 TensorRT 引擎: {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        print(f"输入: {self.input_names}")
        print(f"输出: {self.output_names}")
        
        # 分配 GPU 内存
        self.bindings = []
        self.inputs = {}
        self.outputs = {}
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # 分配内存
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device_mem = cuda.mem_alloc(size)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs[name] = {"host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype}
            else:
                self.outputs[name] = {"host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype}
        
        # 创建 CUDA 流
        self.stream = cuda.Stream()
        
        # 音频配置
        self.preprocess_config = preprocess_config
        
    def infer(self, inputs_dict):
        """执行推理"""
        # 复制输入数据到 GPU
        for name, data in inputs_dict.items():
            if name in self.inputs:
                self.inputs[name]["host"][:data.size] = data.flatten()
                cuda.memcpy_htod_async(self.inputs[name]["device"], self.inputs[name]["host"], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 复制输出数据到 CPU
        outputs = {}
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.outputs[name]["host"], self.outputs[name]["device"], self.stream)
            outputs[name] = self.outputs[name]["host"].reshape(self.outputs[name]["shape"])
        
        self.stream.synchronize()
        
        return outputs


# ============ 全局变量 ============
trt_infer = None
vocoder = None
preprocess_config = None
model_config = None


def read_lexicon(lex_path):
    """读取词典"""
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_mandarin(text, preprocess_config):
    """中文文本预处理"""
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
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize_segment(sequence, speaker_id, max_seq_len):
    """合成单个段落"""
    global vocoder
    
    # 准备输入数据（填充到固定长度 50）
    if len(sequence) > max_seq_len:
        sequence = sequence[:max_seq_len]
    elif len(sequence) < max_seq_len:
        sequence = np.pad(sequence, (0, max_seq_len - len(sequence)), mode='constant')
    
    # 创建输入张量
    speakers = np.array([speaker_id], dtype=np.int64)
    texts = sequence.reshape(1, -1).astype(np.int64)
    src_lens = np.array([max_seq_len], dtype=np.int64)
    max_src_len = np.array(max_seq_len, dtype=np.int64)
    p_control = np.array(1.0, dtype=np.float32)
    e_control = np.array(1.0, dtype=np.float32)
    d_control = np.array(1.0, dtype=np.float32)
    
    inputs = {
        "speakers": speakers,
        "texts": texts,
        "src_lens": src_lens,
        "max_src_len": max_src_len,
        "p_control": p_control,
        "e_control": e_control,
        "d_control": d_control,
    }
    
    # 执行推理
    outputs = trt_infer.infer(inputs)
    
    # 获取 mel 输出
    mel_output = outputs["mel_output"]
    
    # 调整形状: (batch, time, n_mels) -> (batch, n_mels, time)
    if len(mel_output.shape) == 3 and mel_output.shape[2] == 80:
        mel_output = mel_output.transpose(0, 2, 1)
    
    # 使用声码器生成音频
    mel_tensor = torch.from_numpy(mel_output).float().to(device)
    
    with torch.no_grad():
        wav = vocoder(mel_tensor).squeeze()
    
    return wav.cpu().numpy()


def synthesize_with_tensorrt(text, speaker_id=0):
    """使用 TensorRT 进行语音合成"""
    global trt_infer, preprocess_config
    
    # 文本预处理
    print(f"\n处理文本: {text}")
    sequence = preprocess_mandarin(text, preprocess_config)
    print(f"音素序列长度: {len(sequence)}")
    
    if len(sequence) == 0:
        raise ValueError("音素序列为空!")
    
    # 固定序列长度
    max_seq_len = 50
    
    # 如果序列太长，分段处理
    if len(sequence) > max_seq_len:
        num_segments = (len(sequence) + max_seq_len - 1) // max_seq_len
        print(f"文本过长，分段处理: {num_segments} 段...")
        
        all_wavs = []
        start_time = time.time()
        
        for i in range(0, len(sequence), max_seq_len):
            segment = sequence[i:i+max_seq_len]
            segment_num = i // max_seq_len + 1
            print(f"处理第 {segment_num}/{num_segments} 段...")
            wav = synthesize_segment(segment, speaker_id, max_seq_len)
            all_wavs.append(wav)
        
        wav = np.concatenate(all_wavs)
        inference_time = (time.time() - start_time) * 1000
    else:
        start_time = time.time()
        wav = synthesize_segment(sequence, speaker_id, max_seq_len)
        inference_time = (time.time() - start_time) * 1000
    
    print(f"推理时间: {inference_time:.2f} ms")
    
    return wav, inference_time


# ============ FastAPI 应用 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global trt_infer, vocoder, preprocess_config, model_config
    
    print("=" * 60)
    print("🚀 启动 TensorRT TTS FastAPI 服务")
    print("=" * 60)
    
    # 加载配置
    print("加载配置...")
    with open(PREPROCESS_CONFIG, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 加载声码器
    print("加载声码器...")
    vocoder = get_vocoder(model_config, device)
    
    # 初始化 TensorRT
    print("初始化 TensorRT...")
    trt_infer = TensorRTInference(ENGINE_PATH, preprocess_config)
    
    print("✅ 服务启动完成！")
    print(f"   引擎: {ENGINE_PATH}")
    print(f"   设备: {device}")
    print("=" * 60)
    
    yield
    
    print("\n🛑 关闭服务...")


app = FastAPI(
    title="TensorRT TTS API",
    description="基于 TensorRT 的 FastSpeech2 文本转语音 API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    """根路径"""
    return {
        "message": "TensorRT TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/tts": "POST - 文本转语音",
            "/tts/file": "GET - 文本转语音并返回文件"
        }
    }


@app.post("/tts", response_model=TTSResponse)
def tts_endpoint(request: TTSRequest):
    """
    文本转语音 API
    
    - **text**: 要转换的文本
    - **speaker_id**: 说话人 ID (默认 0)
    
    返回 base64 编码的音频数据
    """
    try:
        print(f"\n📥 收到请求: text='{request.text}', speaker_id={request.speaker_id}")
        
        # 执行 TTS
        wav, inference_time = synthesize_with_tensorrt(request.text, request.speaker_id)
        
        # 转换为 base64
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, wav, preprocess_config["preprocessing"]["audio"]["sampling_rate"], format='WAV')
        wav_bytes.seek(0)
        audio_base64 = base64.b64encode(wav_bytes.read()).decode('utf-8')
        
        audio_duration = len(wav) / preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        
        print(f"✅ 合成成功: {audio_duration:.2f}s, {inference_time:.2f}ms")
        
        return TTSResponse(
            success=True,
            message="合成成功",
            audio_base64=audio_base64,
            inference_time_ms=inference_time,
            audio_duration_sec=audio_duration
        )
        
    except Exception as e:
        print(f"❌ 合成失败: {e}")
        return TTSResponse(
            success=False,
            message=f"合成失败: {str(e)}"
        )


@app.get("/tts/file")
def tts_file_endpoint(
    text: str = Query(..., description="要转换的文本"),
    speaker_id: int = Query(0, description="说话人 ID")
):
    """
    文本转语音并返回音频文件
    
    - **text**: 要转换的文本
    - **speaker_id**: 说话人 ID (默认 0)
    
    返回 WAV 音频文件
    """
    try:
        print(f"\n📥 收到请求: text='{text}', speaker_id={speaker_id}")
        
        # 执行 TTS
        wav, inference_time = synthesize_with_tensorrt(text, speaker_id)
        
        # 保存临时文件
        output_path = os.path.join(OUTPUT_DIR, f"tts_{int(time.time())}.wav")
        sf.write(output_path, wav, preprocess_config["preprocessing"]["audio"]["sampling_rate"])
        
        audio_duration = len(wav) / preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        print(f"✅ 合成成功: {audio_duration:.2f}s, {inference_time:.2f}ms")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="output.wav"
        )
        
    except Exception as e:
        print(f"❌ 合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"合成失败: {str(e)}")


@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "ok", "tensorrt_available": TRT_AVAILABLE}


# ============ 启动入口 ============
if __name__ == "__main__":
    # 在 Jetson 上运行
    uvicorn.run(
        "fastapi_tensorrt_tts:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
