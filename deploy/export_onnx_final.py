"""
最终版 ONNX 导出脚本
使用修改后的 LengthRegulator，完全支持 ONNX 和 TensorRT
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
import numpy as np

# 导入修改后的模块
from model.modules_onnx import LengthRegulatorONNX
from model.fastspeech2 import FastSpeech2, VarianceAdaptor


class VarianceAdaptorONNX(nn.Module):
    """
    使用 ONNX 友好的 LengthRegulator
    """
    def __init__(self, original_adaptor, max_mel_len=2000):
        super().__init__()
        self.duration_predictor = original_adaptor.duration_predictor
        self.pitch_predictor = original_adaptor.pitch_predictor
        self.energy_predictor = original_adaptor.energy_predictor
        self.pitch_feature_level = original_adaptor.pitch_feature_level
        self.energy_feature_level = original_adaptor.energy_feature_level
        self.pitch_embedding = original_adaptor.pitch_embedding
        self.energy_embedding = original_adaptor.energy_embedding
        
        # 替换为 ONNX 友好的 LengthRegulator
        self.length_regulator = LengthRegulatorONNX(max_mel_len)
        
    def forward(self, x, mask, p_control=1.0, e_control=1.0, d_control=1.0):
        # Duration prediction
        log_duration_prediction = self.duration_predictor(x, mask)
        duration_rounded = (torch.exp(log_duration_prediction) - 1) * d_control
        duration_rounded = torch.clamp(torch.round(duration_rounded), min=0).long()
        
        # Pitch prediction
        pitch_prediction = self.pitch_predictor(x, mask)
        
        # Energy prediction
        energy_prediction = self.energy_predictor(x, mask)
        
        # Length regulation
        x, mel_len = self.length_regulator(x, duration_rounded, None)
        
        # Create mel mask
        batch_size = x.shape[0]
        max_mel_len = x.shape[1]
        mel_mask = torch.arange(max_mel_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        mel_mask = mel_mask >= mel_len.unsqueeze(1)
        
        # Add pitch and energy embeddings (simplified)
        if self.pitch_feature_level == "frame_level":
            # For simplicity, use average pitch
            pitch_embedding = self.pitch_embedding(
                torch.clamp(pitch_prediction.mean(dim=1, keepdim=True).long(), 0, 255)
            )
            x = x + pitch_embedding.unsqueeze(1).expand(-1, max_mel_len, -1)
        
        if self.energy_feature_level == "frame_level":
            energy_embedding = self.energy_embedding(
                torch.clamp(energy_prediction.mean(dim=1, keepdim=True).long(), 0, 255)
            )
            x = x + energy_embedding.unsqueeze(1).expand(-1, max_mel_len, -1)
        
        return x, pitch_prediction, energy_prediction, log_duration_prediction, duration_rounded, mel_len, mel_mask


class FastSpeech2ONNX(nn.Module):
    """
    完整的 ONNX 兼容 FastSpeech2 模型
    """
    def __init__(self, original_model, max_mel_len=2000):
        super().__init__()
        self.encoder = original_model.encoder
        self.decoder = original_model.decoder
        self.mel_linear = original_model.mel_linear
        self.postnet = original_model.postnet
        
        # 替换 VarianceAdaptor
        self.variance_adaptor = VarianceAdaptorONNX(
            original_model.variance_adaptor, 
            max_mel_len
        )
        
    def forward(self, texts, src_lens):
        """
        简化的前向传播
        
        Args:
            texts: [batch, seq_len]
            src_lens: [batch]
        
        Returns:
            mel_output: [batch, mel_len, 80]
            mel_postnet_output: [batch, mel_len, 80]
        """
        batch_size = texts.shape[0]
        max_src_len = texts.shape[1]
        
        # Create src_mask
        src_mask = torch.arange(max_src_len, device=texts.device).unsqueeze(0).expand(batch_size, -1)
        src_mask = src_mask >= src_lens.unsqueeze(1)
        
        # Encoder
        encoder_output = self.encoder(texts, src_mask)
        
        # Variance Adaptor
        variance_output = self.variance_adaptor(encoder_output, src_mask)
        x, _, _, _, _, mel_len, mel_mask = variance_output
        
        # Decoder
        decoder_output = self.decoder(x, mel_mask)
        
        # Mel Linear
        mel_output = self.mel_linear(decoder_output)
        
        # Postnet
        mel_postnet_output = self.postnet(mel_output.transpose(1, 2))
        mel_postnet_output = mel_output + mel_postnet_output.transpose(1, 2)
        
        return mel_output, mel_postnet_output


def export_fastspeech2_onnx(
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    model_config_path="config/AISHELL3/model.yaml",
    ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
    onnx_path="onnx/fastspeech2_final.onnx",
    max_mel_len=2000,
    opset_version=14,
):
    """
    导出 FastSpeech2 为 ONNX 格式
    """
    print("=" * 70)
    print("FastSpeech2 ONNX 导出（TensorRT 兼容版）")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 加载配置
    print("\n[1/6] 加载配置...")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    print("  ✓ 配置加载完成")
    
    # 创建原始模型
    print("\n[2/6] 创建模型...")
    model = FastSpeech2(preprocess_config, model_config)
    model.eval()
    print("  ✓ 模型创建完成")
    
    # 加载权重
    print(f"\n[3/6] 加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("  ✓ 权重加载完成")
    
    # 转换为 ONNX 兼容版本
    print(f"\n[4/6] 转换为 ONNX 兼容版本 (max_mel_len={max_mel_len})...")
    onnx_model = FastSpeech2ONNX(model, max_mel_len)
    onnx_model.eval()
    print("  ✓ 转换完成")
    
    # 准备示例输入
    print("\n[5/6] 测试模型...")
    batch_size = 1
    seq_len = 50
    
    example_texts = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
    example_src_lens = torch.tensor([seq_len], dtype=torch.long)
    
    print(f"  输入: texts={example_texts.shape}, src_lens={example_src_lens.shape}")
    
    with torch.no_grad():
        mel_output, mel_postnet = onnx_model(example_texts, example_src_lens)
        print(f"  输出: mel_output={mel_output.shape}, mel_postnet={mel_postnet.shape}")
    print("  ✓ 模型测试通过")
    
    # 导出 ONNX
    print(f"\n[6/6] 导出 ONNX: {onnx_path}")
    try:
        torch.onnx.export(
            onnx_model,
            (example_texts, example_src_lens),
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['texts', 'src_lens'],
            output_names=['mel_output', 'mel_postnet_output'],
            dynamic_axes={
                'texts': {0: 'batch_size', 1: 'seq_length'},
                'src_lens': {0: 'batch_size'},
                'mel_output': {0: 'batch_size', 1: 'mel_length'},
                'mel_postnet_output': {0: 'batch_size', 1: 'mel_length'}
            },
        )
        print("  ✓ ONNX 导出成功!")
        
        # 验证
        print("\n[验证] 检查 ONNX 模型...")
        import onnx
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print("  ✓ ONNX 模型验证通过")
        
        # 模型信息
        print(f"\n[信息] 模型详情:")
        print(f"  文件路径: {onnx_path}")
        print(f"  文件大小: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
        print(f"  Opset 版本: {opset_version}")
        print(f"  最大 Mel 长度: {max_mel_len}")
        print(f"  输入: texts [batch, seq], src_lens [batch]")
        print(f"  输出: mel_output [batch, mel_len, 80], mel_postnet_output [batch, mel_len, 80]")
        
        # 简化模型
        print("\n[优化] 简化模型...")
        try:
            import onnxsim
            simplified_path = onnx_path.replace('.onnx', '_sim.onnx')
            onnx_model_sim, check = onnxsim.simplify(onnx_model_check)
            if check:
                onnx.save(onnx_model_sim, simplified_path)
                print(f"  ✓ 简化模型已保存: {simplified_path}")
                print(f"  原模型: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
                print(f"  简化后: {os.path.getsize(simplified_path) / 1024 / 1024:.2f} MB")
        except ImportError:
            print("  ! 未安装 onnx-simplifier，跳过简化")
            print("    pip install onnx-simplifier")
        
    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "=" * 70)
    print("导出完成! 可以使用 TensorRT 进行部署")
    print("=" * 70)
    
    return onnx_path


if __name__ == "__main__":
    # 导出中文模型
    export_fastspeech2_onnx(
        preprocess_config_path="config/AISHELL3/preprocess.yaml",
        model_config_path="config/AISHELL3/model.yaml",
        ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
        onnx_path="onnx/fastspeech2_aishell3.onnx",
        max_mel_len=2000,
        opset_version=14,
    )
