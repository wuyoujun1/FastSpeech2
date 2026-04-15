"""
ONNX Runtime 推理脚本（支持 TensorRT）
用于验证导出的 ONNX 模型
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import onnxruntime as ort
from pypinyin import pinyin, Style
import re
from text import text_to_sequence
import time


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
    
    return np.array(sequence)


def inference_onnx(
    text="你好世界",
    onnx_path="onnx/fastspeech2_traced.onnx",
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    output_dir="onnx_output",
    use_tensorrt=False,
):
    """
    ONNX Runtime 推理
    
    Args:
        text: 输入文本
        onnx_path: ONNX模型路径
        preprocess_config_path: 预处理配置路径
        output_dir: 输出目录
        use_tensorrt: 是否使用 TensorRT 后端
    """
    print("=" * 60)
    print("ONNX Runtime 推理")
    print("=" * 60)
    
    # 加载配置
    print(f"\n加载配置: {preprocess_config_path}")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 设置 ONNX Runtime 会话选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 选择执行提供者
    if use_tensorrt:
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        print("使用 TensorRT 后端")
    else:
        providers = ['CPUExecutionProvider']
        print("使用 CPU 后端")
    
    # 加载 ONNX 模型
    print(f"加载 ONNX 模型: {onnx_path}")
    try:
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 打印模型信息
    print("\n模型输入:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    
    print("\n模型输出:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")
    
    # 预处理文本
    print(f"\n预处理文本: {text}")
    text_sequence = preprocess_mandarin(text, preprocess_config)
    print(f"文本序列: {text_sequence}")
    print(f"序列长度: {len(text_sequence)}")
    
    # 准备输入数据
    texts = np.expand_dims(text_sequence, axis=0).astype(np.int64)
    src_lens = np.array([len(text_sequence)], dtype=np.int64)
    
    print(f"\n输入数据:")
    print(f"  texts shape: {texts.shape}")
    print(f"  src_lens shape: {src_lens.shape}")
    
    # 执行推理
    print("\n执行 ONNX Runtime 推理...")
    input_dict = {
        'texts': texts,
        'src_lens': src_lens,
    }
    
    # 预热
    print("预热...")
    for _ in range(3):
        _ = session.run(None, input_dict)
    
    # 正式推理
    print("正式推理...")
    start_time = time.time()
    try:
        outputs = session.run(None, input_dict)
        inference_time = time.time() - start_time
        
        mel_output = outputs[0]
        print(f"\n推理成功!")
        print(f"  推理时间: {inference_time:.4f} 秒")
        print(f"  mel_output shape: {mel_output.shape}")
        print(f"  mel_output dtype: {mel_output.dtype}")
        print(f"  mel_output range: [{mel_output.min():.4f}, {mel_output.max():.4f}]")
        
        # 保存输出
        os.makedirs(output_dir, exist_ok=True)
        mel_save_path = os.path.join(output_dir, f"{text}_mel.npy")
        np.save(mel_save_path, mel_output[0])
        print(f"\n梅尔频谱图已保存: {mel_save_path}")
        
        # 使用声码器生成音频
        print("\n使用声码器生成音频...")
        try:
            import torch
            from utils.model import get_vocoder
            from utils.tools import vocoder_infer
            
            # 加载配置
            with open("config/AISHELL3/model.yaml", "r") as f:
                model_config = yaml.load(f, Loader=yaml.FullLoader)
            
            # 加载声码器
            device = torch.device("cpu")
            vocoder = get_vocoder(model_config, device)
            
            # 转换 mel 为 torch tensor
            mel_tensor = torch.from_numpy(mel_output).float()
            
            # 生成音频
            with torch.no_grad():
                wavs = vocoder_infer(
                    mel_tensor,
                    vocoder,
                    model_config,
                    preprocess_config,
                )
            
            # 保存音频
            import soundfile as sf
            wav = wavs[0]
            wav_path = os.path.join(output_dir, f"{text}.wav")
            sf.write(wav_path, wav, preprocess_config["preprocessing"]["audio"]["sampling_rate"])
            print(f"音频已保存: {wav_path}")
            
        except Exception as e:
            print(f"声码器推理失败: {e}")
            print("梅尔频谱图已保存，可以使用其他声码器转换")
        
        print("\n" + "=" * 60)
        print("推理完成!")
        print("=" * 60)
        
        return mel_output
        
    except Exception as e:
        print(f"\n推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Runtime 推理")
    parser.add_argument("--text", type=str, default="你好世界", help="测试文本")
    parser.add_argument("--onnx", type=str, default="onnx/fastspeech2_traced.onnx", help="ONNX模型路径")
    parser.add_argument("--config", type=str, default="config/AISHELL3/preprocess.yaml", help="配置路径")
    parser.add_argument("--output", type=str, default="onnx_output", help="输出目录")
    parser.add_argument("--tensorrt", action="store_true", help="使用 TensorRT 后端")
    
    args = parser.parse_args()
    
    inference_onnx(
        text=args.text,
        onnx_path=args.onnx,
        preprocess_config_path=args.config,
        output_dir=args.output,
        use_tensorrt=args.tensorrt,
    )
