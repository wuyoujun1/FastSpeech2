"""
ONNX Runtime推理脚本
可以在CPU上运行，比PyTorch更快

使用方法：
python test_onnxruntime_inference.py --text "测试文本"
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import onnxruntime as ort
from pypinyin import pinyin, Style
import re
from text import text_to_sequence


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


def onnxruntime_inference(
    text="测试ONNX Runtime",
    onnx_path="onnx/fastspeech2.onnx",
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    output_dir="onnxruntime_output",
    speaker_id=0,
):
    """
    ONNX Runtime推理
    
    Args:
        text: 输入文本
        onnx_path: ONNX模型路径
        preprocess_config_path: 预处理配置路径
        output_dir: 输出目录
        speaker_id: 说话人ID
    """
    print("=" * 60)
    print("ONNX Runtime推理测试")
    print("=" * 60)
    
    # 加载配置
    print(f"\n加载配置文件: {preprocess_config_path}")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 加载ONNX模型
    print(f"加载ONNX模型: {onnx_path}")
    
    # 设置优化选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 创建推理会话
    session = ort.InferenceSession(onnx_path, sess_options)
    
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
    print(f"文本序列长度: {len(text_sequence)}")
    
    # 准备输入数据
    speakers = np.array([speaker_id], dtype=np.int64)
    texts = np.expand_dims(text_sequence, axis=0).astype(np.int64)
    src_lens = np.array([len(text_sequence)], dtype=np.int64)
    max_src_len = np.array(len(text_sequence), dtype=np.int64)
    
    # 准备输入字典
    input_dict = {
        'speakers': speakers,
        'texts': texts,
        'src_lens': src_lens,
        'max_src_len': max_src_len,
    }
    
    # 添加控制参数（如果模型需要）
    for inp in session.get_inputs():
        if inp.name not in input_dict:
            if 'control' in inp.name:
                input_dict[inp.name] = np.array(1.0, dtype=np.float64)
            else:
                input_dict[inp.name] = np.array(1.0, dtype=np.float64)
    
    # 执行推理
    print("\n执行ONNX Runtime推理...")
    import time
    start_time = time.time()
    
    try:
        outputs = session.run(None, input_dict)
        inference_time = time.time() - start_time
        print(f"推理时间: {inference_time:.4f} 秒")
        
        # 获取输出
        mel_output = outputs[0]
        print(f"\n输出形状: {mel_output.shape}")
        
        # 保存输出
        os.makedirs(output_dir, exist_ok=True)
        mel_save_path = os.path.join(output_dir, f"{text}_mel.npy")
        np.save(mel_save_path, mel_output[0])
        print(f"梅尔频谱图已保存: {mel_save_path}")
        
        print("\n注意: ONNX模型输出的是梅尔频谱图")
        print("需要使用声码器转换为音频波形")
        
        print("=" * 60)
        print("推理完成!")
        print("=" * 60)
        
        return mel_output
        
    except Exception as e:
        print(f"推理失败: {e}")
        print("\n这可能是因为ONNX模型包含动态控制流")
        print("建议使用PyTorch模型进行推理")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Runtime推理测试")
    parser.add_argument("--text", type=str, default="测试ONNX Runtime", help="测试文本")
    parser.add_argument("--onnx", type=str, default="onnx/fastspeech2.onnx", help="ONNX模型路径")
    parser.add_argument("--config", type=str, default="config/AISHELL3/preprocess.yaml", help="配置路径")
    parser.add_argument("--output", type=str, default="onnxruntime_output", help="输出目录")
    
    args = parser.parse_args()
    
    onnxruntime_inference(
        text=args.text,
        onnx_path=args.onnx,
        preprocess_config_path=args.config,
        output_dir=args.output,
    )
