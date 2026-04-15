#!/usr/bin/env python3
"""
TTS Unix Socket 服务 - 常驻内存，等待调用
使用方式：
  1. 启动：python3 tts_server_unix.py
  2. 调用：python3 tts_client_unix.py "要合成的文本"
"""
import os
import sys
import json
import yaml
import time
import socket
import threading
import io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import wave

from synthesize_tensorrt_fixed import (
    TensorRTInference, preprocess_mandarin, synthesize_segment, device
)
from utils.model import get_vocoder

# 配置
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50
SOCKET_PATH = "/tmp/tts_socket.sock"


class TTSServer:
    def __init__(self):
        self.trt = None
        self.vocoder = None
        self.config = None
        self.sample_rate = 22050
        self.initialized = False
        self.lock = threading.Lock()
        
    def init(self):
        """初始化模型（只执行一次）"""
        print("=" * 60)
        print("🔧 初始化 TTS 服务...")
        print("=" * 60)
        
        print("[1/3] 加载配置...")
        with open(PREPROCESS_CONFIG, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sample_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        
        print("[2/3] 加载声码器...")
        self.vocoder = get_vocoder(model_config, device)
        
        print("[3/3] 初始化 TensorRT...")
        self.trt = TensorRTInference(ENGINE_PATH, self.config)
        
        self.initialized = True
        print("✅ 初始化完成！")
        print(f"   Socket: {SOCKET_PATH}")
        print(f"   设备: {device}")
        print("=" * 60)
        print()
        
    def synthesize(self, text, speaker_id=0):
        """合成音频，返回 WAV 字节"""
        with self.lock:  # 线程安全
            try:
                sequence = preprocess_mandarin(text, self.config)
                if len(sequence) == 0:
                    return None, "音素序列为空"
                
                # 分段合成
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
                
                # 转 WAV
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
                
                return buf.getvalue(), None
                
            except Exception as e:
                return None, str(e)
    
    def handle_client(self, conn, addr):
        """处理客户端请求"""
        try:
            # 接收数据
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                # 检查是否接收完（简单协议：以\n结尾）
                if b"\n" in data:
                    break
            
            # 解析请求
            request = json.loads(data.decode().strip())
            text = request.get("text", "")
            speaker_id = request.get("speaker_id", 0)
            
            print(f"📝 合成: {text[:40]}...")
            
            # 合成
            start_time = time.time()
            audio_data, error = self.synthesize(text, speaker_id)
            duration = time.time() - start_time
            
            if audio_data:
                # 发送成功响应
                header = json.dumps({
                    "status": "success",
                    "duration_sec": len(audio_data) / (2 * self.sample_rate),  # 16bit = 2 bytes
                    "inference_time_ms": duration * 1000
                }).encode()
                conn.sendall(header + b"\n" + audio_data)
                print(f"✅ 完成 ({duration*1000:.0f}ms)")
            else:
                # 发送错误响应
                response = json.dumps({
                    "status": "error",
                    "message": error
                }).encode()
                conn.sendall(response + b"\n")
                print(f"❌ 错误: {error}")
                
        except Exception as e:
            print(f"处理客户端错误: {e}")
        finally:
            conn.close()
    
    def start(self):
        """启动服务"""
        self.init()
        
        # 删除旧 socket
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        
        # 创建 socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(5)
        
        # 设置权限（允许所有用户访问）
        os.chmod(SOCKET_PATH, 0o666)
        
        print("🚀 TTS 服务已启动")
        print(f"   Socket: {SOCKET_PATH}")
        print("   等待连接... (按 Ctrl+C 停止)")
        print("-" * 60)
        
        try:
            while True:
                conn, addr = server.accept()
                # 每个连接一个线程
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print("\n👋 停止服务...")
        finally:
            server.close()
            if os.path.exists(SOCKET_PATH):
                os.remove(SOCKET_PATH)


if __name__ == "__main__":
    server = TTSServer()
    server.start()
