"""
TensorRT ONNX 模型推理脚本
支持 Windows 和 Jetson GPU
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


def inference_tensorrt_onnx(
    text="你好世界",
    onnx_path="onnx/fastspeech2_tensorrt.onnx",
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    output_dir="tensorrt_output",
    use_cuda=False,
):
    """
    TensorRT ONNX 模型推理
    
    Args:
        text: 输入文本
        onnx_path: ONNX模型路径
        preprocess_config_path: 预处理配置路径
        output_dir: 输出目录
        use_cuda: 是否使用 CUDAExecutionProvider (Windows)
    """
    print("=" * 70)
    print("TensorRT ONNX 模型推理")
    print("=" * 70)
    
    # 加载配置
    print(f"\n[1/5] 加载配置...")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    print("  ✓ 配置加载完成")
    
    # 设置 ONNX Runtime
    print(f"\n[2/5] 初始化 ONNX Runtime...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 选择执行提供者
    if use_cuda:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("  使用 CUDA 后端")
    else:
        providers = ['CPUExecutionProvider']
        print("  使用 CPU 后端")
    
    # 加载模型
    print(f"\n[3/5] 加载 ONNX 模型: {onnx_path}")
    try:
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return None
    
    # 模型信息
    print("\n[4/5] 模型信息:")
    print("  输入:")
    for inp in session.get_inputs():
        print(f"    {inp.name}: {inp.shape} ({inp.type})")
    print("  输出:")
    for out in session.get_outputs():
        print(f"    {out.name}: {out.shape} ({out.type})")
    
    # 预处理文本
    print(f"\n[5/5] 推理: '{text}'")
    text_sequence = preprocess_mandarin(text, preprocess_config)
    print(f"  文本序列长度: {len(text_sequence)}")
    
    # 准备输入
    texts = np.expand_dims(text_sequence, axis=0).astype(np.int64)
    src_lens = np.array([len(text_sequence)], dtype=np.int64)
    
    print(f"  输入形状: texts={texts.shape}, src_lens={src_lens.shape}")
    
    input_dict = {
        'texts': texts,
        'src_lens': src_lens,
    }
    
    # 预热
    print("\n  预热...")
    for _ in range(3):
        _ = session.run(None, input_dict)
    
    # 推理
    print("  正式推理...")
    times = []
    for i in range(10):
        start = time.time()
        outputs = session.run(None, input_dict)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    mel_output = outputs[0]
    mel_len = outputs[1]
    
    print(f"\n  ✓ 推理完成!")
    print(f"    平均推理时间: {avg_time:.2f} ms")
    print(f"    输出形状: {mel_output.shape}")
    print(f"    实际 mel 长度: {mel_len[0]}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 mel 频谱
    mel_save_path = os.path.join(output_dir, f"{text}_mel.npy")
    np.save(mel_save_path, mel_output[0])
    print(f"\n  梅尔频谱已保存: {mel_save_path}")
    
    # 使用声码器生成音频
    print("\n  使用声码器生成音频...")
    try:
        import torch
        from utils.model import get_vocoder, vocoder_infer
        import soundfile as sf
        
        # 加载模型配置
        with open("config/AISHELL3/model.yaml", "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 加载声码器
        device = torch.device("cpu")
        vocoder = get_vocoder(model_config, device)
        
        # 截取有效的 mel 部分
        valid_mel_len = int(mel_len[0])
        mel_tensor = torch.from_numpy(mel_output[:, :valid_mel_len, :]).float()
        
        # 生成音频
        with torch.no_grad():
            wavs = vocoder_infer(
                mel_tensor,
                vocoder,
                model_config,
                preprocess_config,
            )
        
        # 保存音频
        wav = wavs[0]
        wav_path = os.path.join(output_dir, f"{text}.wav")
        sf.write(
            wav_path, 
            wav, 
            preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        )
        print(f"  音频已保存: {wav_path}")
        print(f"  音频时长: {len(wav) / preprocess_config['preprocessing']['audio']['sampling_rate']:.2f} 秒")
        
    except Exception as e:
        print(f"  声码器推理失败: {e}")
        print("  梅尔频谱已保存，可以使用其他声码器转换")
    
    print("\n" + "=" * 70)
    print("推理完成!")
    print("=" * 70)
    
    return mel_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorRT ONNX 推理")
    parser.add_argument("--text", type=str, default="你好世界", help="测试文本")
    parser.add_argument("--onnx", type=str, default="onnx/fastspeech2_tensorrt.onnx", help="ONNX模型路径")
    parser.add_argument("--config", type=str, default="config/AISHELL3/preprocess.yaml", help="配置路径")
    parser.add_argument("--output", type=str, default="tensorrt_output", help="输出目录")
    parser.add_argument("--cuda", action="store_true", help="使用 CUDA 后端")
    
    args = parser.parse_args()
    
    inference_tensorrt_onnx(
        text=args.text,
        onnx_path=args.onnx,
        preprocess_config_path=args.config,
        output_dir=args.output,
        use_cuda=args.cuda,
    )
