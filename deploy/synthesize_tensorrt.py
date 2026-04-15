"""
使用 TensorRT 引擎进行 TTS 推理
包含：文本预处理 -> TensorRT 推理 -> 声码器 -> 音频输出
"""
import os
import sys
import re
import time
import yaml
import numpy as np
import torch
from pypinyin import pinyin, Style

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from text import text_to_sequence
from utils.model import get_vocoder


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


def synthesize_with_tensorrt(text, speaker_id, engine_path, preprocess_config, model_config, vocoder):
    """使用 TensorRT 进行语音合成"""
    
    # 初始化 TensorRT
    trt_infer = TensorRTInference(engine_path, preprocess_config)
    
    # 文本预处理（使用中文预处理）
    print(f"\n处理文本: {text}")
    sequence = preprocess_mandarin(text, preprocess_config)
    print(f"音素序列长度: {len(sequence)}")
    
    if len(sequence) == 0:
        print("错误: 音素序列为空!")
        return None, 0
    
    # 准备输入数据（固定形状为 50）
    max_seq_len = 50
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
    print("\n执行 TensorRT 推理...")
    start_time = time.time()
    outputs = trt_infer.infer(inputs)
    inference_time = (time.time() - start_time) * 1000
    print(f"推理时间: {inference_time:.2f} ms")
    
    # 获取 mel 输出
    mel_output = outputs["mel_output"]
    print(f"Mel 输出形状 (原始): {mel_output.shape}")
    
    # 调整形状: (batch, time, n_mels) -> (batch, n_mels, time)
    if len(mel_output.shape) == 3 and mel_output.shape[2] == 80:
        mel_output = mel_output.transpose(0, 2, 1)  # (1, 395, 80) -> (1, 80, 395)
    print(f"Mel 输出形状 (调整后): {mel_output.shape}")
    
    # 使用声码器生成音频
    print("\n使用声码器生成音频...")
    mel_tensor = torch.from_numpy(mel_output).float().to(device)
    
    with torch.no_grad():
        wav = vocoder(mel_tensor).squeeze()
    
    wav = wav.cpu().numpy()
    
    return wav, inference_time


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="输入文本")
    parser.add_argument("--speaker_id", type=int, default=0, help="说话人 ID")
    parser.add_argument("--engine", type=str, default="fastspeech2.engine", help="TensorRT 引擎路径")
    parser.add_argument("--output", type=str, default="output_tensorrt.wav", help="输出音频路径")
    parser.add_argument("--preprocess_config", type=str, default="config/AISHELL3/preprocess.yaml")
    parser.add_argument("--model_config", type=str, default="config/AISHELL3/model.yaml")
    args = parser.parse_args()
    
    # 读取配置
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    
    # 加载声码器
    print("加载声码器...")
    vocoder = get_vocoder(model_config, device)
    
    # 执行 TensorRT 推理
    wav, inference_time = synthesize_with_tensorrt(
        args.text,
        args.speaker_id,
        args.engine,
        preprocess_config,
        model_config,
        vocoder
    )
    
    if wav is not None:
        # 保存音频
        import soundfile as sf
        sf.write(args.output, wav, preprocess_config["preprocessing"]["audio"]["sampling_rate"])
        print(f"\n音频已保存: {args.output}")
        print(f"音频长度: {len(wav) / preprocess_config['preprocessing']['audio']['sampling_rate']:.2f} 秒")
        print(f"TensorRT 推理时间: {inference_time:.2f} ms")
        
        # 播放音频
        try:
            os.system(f"aplay -D plughw:0,0 {args.output}")
        except:
            pass


if __name__ == "__main__":
    main()
