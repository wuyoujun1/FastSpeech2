"""
使用 TorchScript Trace 导出 ONNX
这种方法可以处理动态循环问题
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
from model.fastspeech2 import FastSpeech2


class FastSpeech2Wrapper(torch.nn.Module):
    """
    包装器：简化输入输出，便于 ONNX 导出
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, texts, src_lens):
        """
        简化版前向传播
        Args:
            texts: [batch, seq_len] 文本序列
            src_lens: [batch] 文本长度
        Returns:
            mel_output: [batch, mel_len, 80] 梅尔频谱
        """
        batch_size = texts.shape[0]
        max_src_len = texts.shape[1]
        
        # 创建 speaker（固定为0）
        speakers = torch.zeros(batch_size, dtype=torch.long, device=texts.device)
        
        # 调用原始模型
        with torch.no_grad():
            output = self.model(
                speakers=speakers,
                texts=texts,
                src_lens=src_lens,
                max_src_len=max_src_len,
            )
        
        # 返回 mel_output
        return output[0]


def export_with_trace(
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    model_config_path="config/AISHELL3/model.yaml", 
    ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
    onnx_save_path="onnx/fastspeech2_traced.onnx",
):
    """
    使用 torch.jit.trace 导出 ONNX
    """
    print("=" * 60)
    print("使用 TorchScript Trace 导出 ONNX")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)
    
    # 加载配置
    print("\n加载配置...")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
    print("创建模型...")
    model = FastSpeech2(preprocess_config, model_config)
    model.eval()
    
    # 加载权重
    print(f"加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt["model"])
    
    # 包装模型
    wrapped_model = FastSpeech2Wrapper(model)
    wrapped_model.eval()
    
    # 创建示例输入（用于 trace）
    print("\n准备示例输入...")
    batch_size = 1
    seq_len = 50
    
    example_texts = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
    example_src_lens = torch.tensor([seq_len], dtype=torch.long)
    
    print(f"  texts: {example_texts.shape}")
    print(f"  src_lens: {example_src_lens.shape}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    with torch.no_grad():
        output = wrapped_model(example_texts, example_src_lens)
        print(f"  输出形状: {output.shape}")
    
    # 使用 torch.jit.trace
    print("\n使用 torch.jit.trace...")
    try:
        traced_model = torch.jit.trace(
            wrapped_model,
            (example_texts, example_src_lens),
            strict=False
        )
        print("Trace 成功!")
        
        # 测试 traced 模型
        print("\n测试 traced 模型...")
        with torch.no_grad():
            traced_output = traced_model(example_texts, example_src_lens)
            print(f"  traced 输出形状: {traced_output.shape}")
            
            # 比较输出
            diff = torch.abs(output - traced_output).max().item()
            print(f"  最大差异: {diff:.6f}")
            
            if diff < 1e-5:
                print("  traced 模型输出与原模型一致!")
            else:
                print(f"  警告: 输出差异较大 ({diff})")
        
    except Exception as e:
        print(f"Trace 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 导出 ONNX
    print(f"\n导出 ONNX: {onnx_save_path}")
    try:
        torch.onnx.export(
            traced_model,
            (example_texts, example_src_lens),
            onnx_save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['texts', 'src_lens'],
            output_names=['mel_output'],
            dynamic_axes={
                'texts': {0: 'batch_size', 1: 'seq_length'},
                'src_lens': {0: 'batch_size'},
                'mel_output': {0: 'batch_size', 1: 'mel_length'}
            },
        )
        print("ONNX 导出成功!")
        
        # 验证
        print("\n验证 ONNX 模型...")
        import onnx
        onnx_model = onnx.load(onnx_save_path)
        onnx.checker.check_model(onnx_model)
        print("验证通过!")
        
        print(f"\n模型信息:")
        print(f"  文件大小: {os.path.getsize(onnx_save_path) / 1024 / 1024:.2f} MB")
        
        # 简化模型（可选，用于 TensorRT）
        print("\n尝试简化模型...")
        try:
            import onnxsim
            simplified_path = onnx_save_path.replace('.onnx', '_sim.onnx')
            onnx_model_sim, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model_sim, simplified_path)
                print(f"简化模型已保存: {simplified_path}")
                print(f"  原模型大小: {os.path.getsize(onnx_save_path) / 1024 / 1024:.2f} MB")
                print(f"  简化后大小: {os.path.getsize(simplified_path) / 1024 / 1024:.2f} MB")
        except ImportError:
            print("  未安装 onnx-simplifier，跳过简化")
            print("  建议: pip install onnx-simplifier")
        
    except Exception as e:
        print(f"导出失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    
    return onnx_save_path


if __name__ == "__main__":
    export_with_trace()
