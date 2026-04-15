import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
from model.fastspeech2 import FastSpeech2


def export_to_onnx(
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    model_config_path="config/AISHELL3/model.yaml",
    ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
    onnx_save_path="onnx/fastspeech2.onnx",
    opset_version=11,
):
    print("=" * 50)
    print("开始导出ONNX模型")
    print("=" * 50)
    
    # 创建保存目录
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)
    
    # 加载配置文件
    print(f"加载配置文件: {preprocess_config_path}")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"加载配置文件: {model_config_path}")
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
    print("创建模型...")
    model = FastSpeech2(preprocess_config, model_config)
    model.eval()
    
    # 加载权重
    print(f"加载权重文件: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("权重加载成功!")
    
    # 创建示例输入
    batch_size = 1
    max_seq_len = 100
    
    # 模拟输入数据
    speakers = torch.tensor([0], dtype=torch.long)  # 说话人ID
    texts = torch.randint(0, 100, (batch_size, max_seq_len), dtype=torch.long)  # 文本序列
    src_lens = torch.tensor([max_seq_len], dtype=torch.long)  # 文本长度
    max_src_len = max_seq_len
    
    # 控制参数
    p_control = 1.0  # 音调控制
    e_control = 1.0  # 能量控制
    d_control = 1.0  # 时长控制
    
    print(f"输入形状:")
    print(f"  speakers: {speakers.shape}")
    print(f"  texts: {texts.shape}")
    print(f"  src_lens: {src_lens.shape}")
    
    # 测试模型前向传播
    print("测试模型前向传播...")
    with torch.no_grad():
        try:
            outputs = model(
                speakers=speakers,
                texts=texts,
                src_lens=src_lens,
                max_src_len=max_src_len,
                p_control=p_control,
                e_control=e_control,
                d_control=d_control,
            )
            print(f"模型输出形状: {outputs[0].shape}")
        except Exception as e:
            print(f"模型前向传播错误: {e}")
            return
    
    # 导出ONNX
    print(f"导出ONNX模型到: {onnx_save_path}")
    try:
        torch.onnx.export(
            model,
            (speakers, texts, src_lens, max_src_len),
            onnx_save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['speakers', 'texts', 'src_lens', 'max_src_len'],
            output_names=['mel_output', 'mel_lens'],
            dynamic_axes={
                'texts': {0: 'batch_size', 1: 'seq_length'},
                'src_lens': {0: 'batch_size'},
                'mel_output': {0: 'batch_size', 1: 'mel_seq_length'},
                'mel_lens': {0: 'batch_size'}
            },
            verbose=False,
        )
        print("ONNX模型导出成功!")
        
        # 验证ONNX模型
        print("验证ONNX模型...")
        import onnx
        onnx_model = onnx.load(onnx_save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证通过!")
        
        # 打印模型信息
        print(f"\nONNX模型信息:")
        print(f"  文件大小: {os.path.getsize(onnx_save_path) / 1024 / 1024:.2f} MB")
        print(f"  Opset版本: {opset_version}")
        print(f"  输入节点: {['speakers', 'texts', 'src_lens', 'max_src_len']}")
        print(f"  输出节点: {['mel_output', 'mel_lens']}")
        
    except Exception as e:
        print(f"ONNX导出错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)
    print("导出完成!")
    print("=" * 50)


if __name__ == "__main__":
    export_to_onnx()
