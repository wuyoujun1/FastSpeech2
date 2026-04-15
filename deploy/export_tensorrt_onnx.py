"""
TensorRT 友好的 ONNX 导出脚本
专为 Jetson GPU 优化

特点：
1. 使用固定最大长度，避免动态维度
2. 简化 LengthRegulator，使用矩阵操作
3. 支持 FP16 推理
4. 兼容 TensorRT 8.x
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
import numpy as np
from model.fastspeech2 import FastSpeech2
from model.modules import LengthRegulator


class LengthRegulatorTRT(nn.Module):
    """
    TensorRT 友好的 LengthRegulator
    使用固定最大长度，避免动态循环
    """
    def __init__(self, max_mel_len=2000):
        super().__init__()
        self.max_mel_len = max_mel_len
        
    def forward(self, x, duration, max_len=None):
        """
        Args:
            x: [batch, seq_len, hidden]
            duration: [batch, seq_len]
            max_len: 未使用，保持接口兼容
        Returns:
            output: [batch, max_mel_len, hidden]
            mel_len: [batch]
        """
        batch_size, seq_len, hidden = x.shape
        device = x.device
        
        # 计算累计长度
        cumsum_duration = torch.cumsum(duration, dim=1)  # [batch, seq_len]
        total_len = cumsum_duration[:, -1]  # [batch]
        
        # 创建位置索引
        positions = torch.arange(self.max_mel_len, device=device).float()
        positions = positions.view(1, 1, -1).expand(batch_size, seq_len, -1)
        # positions: [batch, seq_len, max_mel_len]
        
        # 计算每个音素的起始和结束位置
        # start: [batch, seq_len, 1], end: [batch, seq_len, 1]
        cumsum_duration_shifted = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            cumsum_duration[:, :-1]
        ], dim=1)
        
        start = cumsum_duration_shifted.unsqueeze(2)  # [batch, seq_len, 1]
        end = cumsum_duration.unsqueeze(2)  # [batch, seq_len, 1]
        
        # 创建 mask: start <= position < end
        mask = (positions >= start) & (positions < end)  # [batch, seq_len, max_mel_len]
        mask = mask.float()
        
        # 使用 mask 进行加权求和
        # x: [batch, seq_len, hidden] -> [batch, seq_len, 1, hidden]
        # mask: [batch, seq_len, max_mel_len, 1]
        x_expanded = x.unsqueeze(2)  # [batch, seq_len, 1, hidden]
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, max_mel_len, 1]
        
        # 加权求和
        output = (x_expanded * mask_expanded).sum(dim=1)  # [batch, max_mel_len, hidden]
        
        # 实际的 mel 长度
        mel_len = torch.clamp(total_len, max=self.max_mel_len).long()
        
        return output, mel_len


class FastSpeech2TRT(nn.Module):
    """
    TensorRT 优化的 FastSpeech2
    固定 batch_size=1，简化推理
    """
    def __init__(self, original_model, max_mel_len=2000):
        super().__init__()
        self.encoder = original_model.encoder
        self.decoder = original_model.decoder
        self.mel_linear = original_model.mel_linear
        self.postnet = original_model.postnet
        
        # Variance predictor
        self.duration_predictor = original_model.variance_adaptor.duration_predictor
        self.pitch_predictor = original_model.variance_adaptor.pitch_predictor
        self.energy_predictor = original_model.variance_adaptor.energy_predictor
        
        # Embeddings
        self.pitch_embedding = original_model.variance_adaptor.pitch_embedding
        self.energy_embedding = original_model.variance_adaptor.energy_embedding
        
        # Length regulator
        self.length_regulator = LengthRegulatorTRT(max_mel_len)
        
        self.max_mel_len = max_mel_len
        
    def forward(self, texts, src_lens):
        """
        简化的前向传播
        
        Args:
            texts: [1, seq_len] - 文本序列（固定 batch=1）
            src_lens: [1] - 文本长度
            
        Returns:
            mel_output: [1, max_mel_len, 80] - 梅尔频谱
            mel_len: [1] - 实际 mel 长度
        """
        batch_size = 1
        seq_len = texts.shape[1]
        
        # Create mask
        src_mask = torch.arange(seq_len, device=texts.device).unsqueeze(0) >= src_lens.unsqueeze(1)
        
        # Encoder
        encoder_output = self.encoder(texts, src_mask)
        
        # Duration prediction
        log_duration = self.duration_predictor(encoder_output, src_mask)
        duration = torch.exp(log_duration) - 1
        duration = torch.clamp(torch.round(duration), min=0).long()
        
        # Pitch prediction
        pitch = self.pitch_predictor(encoder_output, src_mask)
        
        # Energy prediction
        energy = self.energy_predictor(encoder_output, src_mask)
        
        # Length regulation
        x, mel_len = self.length_regulator(encoder_output, duration, None)
        
        # Add pitch and energy embeddings
        # 使用平均 pitch/energy 进行简单处理
        pitch_mean = pitch.mean(dim=1, keepdim=True).long()
        pitch_mean = torch.clamp(pitch_mean, 0, 255)
        pitch_emb = self.pitch_embedding(pitch_mean)  # [1, 1, 256]
        x = x + pitch_emb.expand(-1, self.max_mel_len, -1)
        
        energy_mean = energy.mean(dim=1, keepdim=True).long()
        energy_mean = torch.clamp(energy_mean, 0, 255)
        energy_emb = self.energy_embedding(energy_mean)  # [1, 1, 256]
        x = x + energy_emb.expand(-1, self.max_mel_len, -1)
        
        # Create mel mask
        mel_mask = torch.arange(self.max_mel_len, device=x.device).unsqueeze(0) >= mel_len.unsqueeze(1)
        
        # Decoder
        decoder_output = self.decoder(x, mel_mask)
        
        # Handle tuple output from decoder
        if isinstance(decoder_output, tuple):
            decoder_output = decoder_output[0]
        
        # Mel linear
        mel_output = self.mel_linear(decoder_output)
        
        return mel_output, mel_len


def export_for_tensorrt(
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    model_config_path="config/AISHELL3/model.yaml",
    ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
    onnx_path="onnx/fastspeech2_tensorrt.onnx",
    max_mel_len=2000,
    seq_len=100,
):
    """
    导出 TensorRT 友好的 ONNX 模型
    """
    print("=" * 70)
    print("TensorRT 友好的 ONNX 导出")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 加载配置
    print("\n[1/5] 加载配置...")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    print("  ✓ 配置加载完成")
    
    # 创建模型
    print("\n[2/5] 创建模型...")
    model = FastSpeech2(preprocess_config, model_config)
    model.eval()
    print("  ✓ 模型创建完成")
    
    # 加载权重
    print(f"\n[3/5] 加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("  ✓ 权重加载完成")
    
    # 转换为 TensorRT 版本
    print(f"\n[4/5] 转换为 TensorRT 版本 (max_mel_len={max_mel_len})...")
    trt_model = FastSpeech2TRT(model, max_mel_len)
    trt_model.eval()
    
    # 测试
    print("\n[5/5] 测试模型...")
    example_texts = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
    example_src_lens = torch.tensor([seq_len], dtype=torch.long)
    
    with torch.no_grad():
        mel_output, mel_len = trt_model(example_texts, example_src_lens)
        print(f"  输入: texts={example_texts.shape}, src_lens={example_src_lens.shape}")
        print(f"  输出: mel_output={mel_output.shape}, mel_len={mel_len}")
    print("  ✓ 模型测试通过")
    
    # 导出 ONNX
    print(f"\n[导出] 导出 ONNX: {onnx_path}")
    try:
        torch.onnx.export(
            trt_model,
            (example_texts, example_src_lens),
            onnx_path,
            export_params=True,
            opset_version=13,  # TensorRT 8.4+ 支持 opset 13
            do_constant_folding=True,
            input_names=['texts', 'src_lens'],
            output_names=['mel_output', 'mel_len'],
            # 固定 batch=1，只让 seq_len 动态
            dynamic_axes={
                'texts': {1: 'seq_len'},
            },
        )
        print("  ✓ ONNX 导出成功!")
        
        # 验证
        print("\n[验证] 检查 ONNX 模型...")
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX 验证通过")
        
        # 简化模型
        print("\n[优化] 简化模型...")
        try:
            import onnxsim
            sim_path = onnx_path.replace('.onnx', '_sim.onnx')
            model_sim, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_sim, sim_path)
                print(f"  ✓ 简化模型: {sim_path}")
        except ImportError:
            print("  ! 未安装 onnx-simplifier")
        
        # 信息
        print(f"\n[信息] 模型详情:")
        print(f"  文件: {onnx_path}")
        print(f"  大小: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
        print(f"  输入: texts [1, seq_len], src_lens [1]")
        print(f"  输出: mel_output [1, {max_mel_len}, 80], mel_len [1]")
        
    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "=" * 70)
    print("导出完成!")
    print("\n下一步：转换为 TensorRT")
    print(f"  trtexec --onnx={onnx_path} --saveEngine=fastspeech2.trt --fp16")
    print("=" * 70)
    
    return onnx_path


if __name__ == "__main__":
    export_for_tensorrt(
        preprocess_config_path="config/AISHELL3/preprocess.yaml",
        model_config_path="config/AISHELL3/model.yaml",
        ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
        onnx_path="onnx/fastspeech2_tensorrt.onnx",
        max_mel_len=2000,
        seq_len=100,
    )
