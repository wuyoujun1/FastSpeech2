#!/usr/bin/env python3
"""
TTS 后台服务 - 文件监听模式
监听指定目录的文本文件，合成后输出音频文件

使用方式：
1. 启动服务：python3 tts_service_file.py
2. 大模型写入文本文件到 input/ 目录
3. 服务自动合成，输出到 output/ 目录
"""
import os
import sys
import time
import json
import yaml
import torch
import numpy as np
import wave
import threading
import queue
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from synthesize_tensorrt_fixed import (
    TensorRTInference, preprocess_mandarin, synthesize_segment, device
)
from utils.model import get_vocoder

# 配置
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50
INPUT_DIR = "tts_input"
OUTPUT_DIR = "tts_output"


class TTSService:
    def __init__(self):
        self.trt = None
        self.vocoder = None
        self.config = None
        self.sample_rate = 22050
        self.initialized = False
        self.task_queue = queue.Queue()
        self.running = False
        
    def init(self):
        """初始化模型"""
        print("=" * 60)
        print("🔧 初始化 TTS 服务...")
        print("=" * 60)
        
        # 创建输入输出目录
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 加载配置
        print("[1/4] 加载配置...")
        with open(PREPROCESS_CONFIG, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sample_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        
        # 加载声码器
        print("[2/4] 加载声码器...")
        self.vocoder = get_vocoder(model_config, device)
        
        # 初始化 TensorRT
        print("[3/4] 初始化 TensorRT...")
        self.trt = TensorRTInference(ENGINE_PATH, self.config)
        
        self.initialized = True
        print("[4/4] ✅ 初始化完成！")
        print(f"   输入目录: {INPUT_DIR}/")
        print(f"   输出目录: {OUTPUT_DIR}/")
        print(f"   设备: {device}")
        print("=" * 60)
        print()
        
    def synthesize(self, text, speaker_id=0):
        """合成音频，返回音频数据"""
        if not self.initialized:
            return None
            
        try:
            sequence = preprocess_mandarin(text, self.config)
            if len(sequence) == 0:
                return None
            
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
            
            # 转换为 WAV 格式
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
            
        except Exception as e:
            print(f"合成错误: {e}")
            return None
    
    def process_task(self, task_file):
        """处理单个任务"""
        try:
            # 读取任务
            with open(task_file, 'r', encoding='utf-8') as f:
                task = json.load(f)
            
            text = task.get('text', '')
            speaker_id = task.get('speaker_id', 0)
            task_id = task.get('task_id', 'unknown')
            
            print(f"\n📝 处理任务 [{task_id}]: {text[:30]}...")
            
            # 合成
            start_time = time.time()
            audio_data = self.synthesize(text, speaker_id)
            
            if audio_data:
                # 保存音频
                output_file = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                # 保存完成标记
                done_file = os.path.join(OUTPUT_DIR, f"{task_id}.done")
                with open(done_file, 'w') as f:
                    f.write(json.dumps({
                        'task_id': task_id,
                        'status': 'success',
                        'output_file': output_file,
                        'duration': time.time() - start_time
                    }))
                
                print(f"✅ 完成: {output_file} ({time.time()-start_time:.2f}s)")
            else:
                # 保存错误标记
                error_file = os.path.join(OUTPUT_DIR, f"{task_id}.error")
                with open(error_file, 'w') as f:
                    f.write(json.dumps({'task_id': task_id, 'status': 'error'}))
                print(f"❌ 失败: {task_id}")
            
            # 删除已处理的任务文件
            os.remove(task_file)
            
        except Exception as e:
            print(f"处理任务错误: {e}")
    
    def worker_thread(self):
        """工作线程"""
        while self.running:
            try:
                # 获取任务
                task_file = self.task_queue.get(timeout=1)
                self.process_task(task_file)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"工作线程错误: {e}")
    
    def monitor_thread(self):
        """监控线程 - 监听输入目录"""
        processed_files = set()
        
        while self.running:
            try:
                # 扫描输入目录
                for filename in os.listdir(INPUT_DIR):
                    if filename.endswith('.json') and filename not in processed_files:
                        filepath = os.path.join(INPUT_DIR, filename)
                        # 检查文件是否写入完成（等待1秒）
                        time.sleep(0.5)
                        self.task_queue.put(filepath)
                        processed_files.add(filename)
                        print(f"📥 新任务: {filename}")
                
                # 清理已处理文件记录（防止内存无限增长）
                if len(processed_files) > 1000:
                    processed_files.clear()
                
                time.sleep(0.1)  # 100ms 轮询间隔
                
            except Exception as e:
                print(f"监控线程错误: {e}")
                time.sleep(1)
    
    def start(self):
        """启动服务"""
        self.init()
        self.running = True
        
        # 启动工作线程
        worker = threading.Thread(target=self.worker_thread)
        worker.daemon = True
        worker.start()
        
        print("🚀 TTS 服务已启动")
        print(f"   监控目录: {INPUT_DIR}/")
        print(f"   输出目录: {OUTPUT_DIR}/")
        print("   等待任务... (按 Ctrl+C 停止)")
        print("-" * 60)
        
        # 启动监控
        try:
            self.monitor_thread()
        except KeyboardInterrupt:
            print("\n👋 停止服务...")
            self.running = False
            worker.join(timeout=5)


def create_task(text, speaker_id=0, task_id=None):
    """辅助函数：创建任务文件（供大模型调用）"""
    if task_id is None:
        task_id = f"task_{int(time.time()*1000)}"
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    task = {
        'task_id': task_id,
        'text': text,
        'speaker_id': speaker_id,
        'timestamp': time.time()
    }
    
    task_file = os.path.join(INPUT_DIR, f"{task_id}.json")
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(task, f, ensure_ascii=False)
    
    return task_id


def wait_for_result(task_id, timeout=30):
    """辅助函数：等待任务完成（供大模型调用）"""
    output_file = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    done_file = os.path.join(OUTPUT_DIR, f"{task_id}.done")
    error_file = os.path.join(OUTPUT_DIR, f"{task_id}.error")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(done_file):
            with open(done_file, 'r') as f:
                return json.load(f)
        if os.path.exists(error_file):
            return {'status': 'error'}
        time.sleep(0.1)
    
    return {'status': 'timeout'}


if __name__ == "__main__":
    import io
    
    # 启动服务
    service = TTSService()
    service.start()
