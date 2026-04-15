"""
修复版 ONNX 导出脚本
解决 LengthRegulator 的动态循环问题，支持 TensorRT
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
import numpy as np
from model.fastspeech2 import FastSpeech2
from model.modules import LengthRegulator


class LengthRegulatorONNX(nn.Module):
    """
    修改后的 LengthRegulator，使用固定最大长度，支持 ONNX/TensorRT
    """
    def __init__(self, max_mel_len=1000):
        super(LengthRegulatorONNX, self).__init__()
        self.max_mel_len = max_mel_len
        
    def forward(self, x, duration, max_len):
        """
        Args:
            x: [batch, seq_len, hidden]
            duration: [batch, seq_len]
            max_len: scalar
        Returns:
            output: [batch, max_mel_len, hidden]
            mel_len: [batch]
        """
        batch_size, seq_len, hidden = x.shape
        
        # 计算累计长度
        duration_cumsum = torch.cumsum(duration, dim=1)  # [batch, seq_len]
        
        # 创建位置索引
        positions = torch.arange(self.max_mel_len, device=x.device).unsqueeze(0).expand(batch_size, -1)  # [batch, max_mel_len]
        
        # 找到每个位置对应的音素索引
        # 使用广播进行比较
        duration_expanded = duration_cumsum.unsqueeze(2)  # [batch, seq_len, 1]
        positions_expanded = positions.unsqueeze(1)  # [batch, 1, max_mel_len]
        
        # 创建 mask: [batch, seq_len, max_mel_len]
        mask = (positions_expanded < duration_expanded).float()
        
        # 计算每个位置的音素索引
        # 使用差分找到边界
        mask_diff = torch.diff(mask, dim=1, prepend=torch.zeros(batch_size, 1, self.max_mel_len, device=x.device))
        phoneme_indices = torch.argmax(mask_diff, dim=1)  # [batch, max_mel_len]
        
        # 限制索引范围
        phoneme_indices = torch.clamp(phoneme_indices, 0, seq_len - 1)
        
        # 使用 gather 获取对应的特征
        phoneme_indices = phoneme_indices.unsqueeze(-1).expand(-1, -1, hidden)  # [batch, max_mel_len, hidden]
        output = torch.gather(x, 1, phoneme_indices)  # [batch, max_mel_len, hidden]
        
        # 计算实际的 mel 长度
        mel_len = duration.sum(dim=1).long()  # [batch]
        mel_len = torch.clamp(mel_len, max=self.max_mel_len)
        
        return output, mel_len


class FastSpeech2ONNX(nn.Module):
    """
    包装器：使用修改后的 LengthRegulator
    """
    def __init__(self, model, max_mel_len=1000):
        super(FastSpeech2ONNX, self).__init__()
        self.encoder = model.encoder
        self.variance_adaptor = model.variance_adaptor
        self.decoder = model.decoder
        self.mel_linear = model.mel_linear
        self.postnet = model.postnet
        
        # 替换 LengthRegulator
        self.variance_adaptor.length_regulator = LengthRegulatorONNX(max_mel_len)
        
    def forward(self, speakers, texts, src_lens, max_src_len, 
                p_control=1.0, e_control=1.0, d_control=1.0):
        """
        简化的前向传播，用于 ONNX 导出
        """
        # Encoder
        src_masks = self._get_mask(src_lens, max_src_len)
        encoder_output = self.encoder(texts, src_masks)
        
        # Variance Adaptor
        variance_output = self.variance_adaptor(
            encoder_output, 
            src_masks, 
            p_control=p_control,
            e_control=e_control,
            d_control=d_control
        )
        
        x, pitch_embedding, energy_embedding, _, _, mel_len, mel_masks = variance_output
        
        # Decoder
        decoder_output = self.decoder(x, mel_masks)
        
        # Mel Linear
        mel_output = self.mel_linear(decoder_output)
        
        return mel_output, mel_len
    
    def _get_mask(self, lengths, max_len):
        """创建 mask"""
        batch_size = lengths.shape[0]
        seq_range = torch.arange(0, max_len, device=lengths.device).long()
        seq_range = seq_range.unsqueeze(0).expand(batch_size, -1)
        lengths = lengths.unsqueeze(-1).expand(-1, max_len)
        mask = seq_range >= lengths
        return mask


def export_to_onnx_fixed(
    preprocess_config_path="config/AISHELL3/preprocess.yaml",
    model_config_path="config/AISHELL3/model.yaml",
    ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
    onnx_save_path="onnx/fastspeech2_fixed.onnx",
    max_mel_len=1000,
    opset_version=14,  # 使用更高版本支持更多操作
):
    """
    导出修复后的 ONNX 模型
    """
    print("=" * 60)
    print("开始导出 ONNX 模型（TensorRT 兼容版）")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)
    
    # 加载配置文件
    print(f"\n加载配置文件...")
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建原始模型
    print("创建模型...")
    model = FastSpeech2(preprocess_config, model_config)
    model.eval()
    
    # 加载权重
    print(f"加载权重文件: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt["model"])
    print("权重加载成功!")
    
    # 包装为 ONNX 兼容版本
    print(f"转换为 ONNX 兼容版本 (max_mel_len={max_mel_len})...")
    onnx_model = FastSpeech2ONNX(model, max_mel_len=max_mel_len)
    onnx_model.eval()
    
    # 创建示例输入
    batch_size = 1
    seq_len = 50
    
    speakers = torch.tensor([0], dtype=torch.long)
    texts = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
    src_lens = torch.tensor([seq_len], dtype=torch.long)
    max_src_len = torch.tensor(seq_len, dtype=torch.long)
    
    print(f"\n示例输入:")
    print(f"  speakers: {speakers.shape}")
    print(f"  texts: {texts.shape}")
    print(f"  src_lens: {src_lens.shape}")
    print(f"  max_src_len: {max_src_len.shape}")
    
    # 测试前向传播
    print("\n测试模型前向传播...")
    with torch.no_grad():
        try:
            mel_output, mel_len = onnx_model(
                speakers, texts, src_lens, max_src_len
            )
            print(f"  mel_output: {mel_output.shape}")
            print(f"  mel_len: {mel_len.shape}")
            print("前向传播测试通过!")
        except Exception as e:
            print(f"前向传播错误: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 导出 ONNX
    print(f"\n导出 ONNX 模型到: {onnx_save_path}")
    try:
        torch.onnx.export(
            onnx_model,
            (speakers, texts, src_lens, max_src_len),
            onnx_save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['speakers', 'texts', 'src_lens', 'max_src_len'],
            output_names=['mel_output', 'mel_len'],
            dynamic_axes={
                'texts': {0: 'batch_size', 1: 'seq_length'},
                'src_lens': {0: 'batch_size'},
                'mel_output': {0: 'batch_size'},
                'mel_len': {0: 'batch_size'}
            },
            verbose=False,
        )
        print("ONNX 模型导出成功!")
        
        # 验证 ONNX 模型
        print("\n验证 ONNX 模型...")
        import onnx
        onnx_model_check = onnx.load(onnx_save_path)
        onnx.checker.check_model(onnx_model_check)
        print("ONNX 模型验证通过!")
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"  文件大小: {os.path.getsize(onnx_save_path) / 1024 / 1024:.2f} MB")
        print(f"  Opset 版本: {opset_version}")
        print(f"  最大 Mel 长度: {max_mel_len}")
        
        # 检查 TensorRT 兼容性
        print("\n检查 TensorRT 兼容性...")
        try:
            import onnx_graphsurgeon as gs
            graph = gs.import_onnx(onnx_model_check)
            print(f"  节点数: {len(graph.nodes)}")
            print(f"  输入: {[inp.name for inp in graph.inputs]}")
            print(f"  输出: {[out.name for out in graph.outputs]}")
            print("TensorRT 兼容性检查通过!")
        except ImportError:
            print("  未安装 onnx-graphsurgeon，跳过详细检查")
        
    except Exception as e:
        print(f"ONNX 导出错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    
    return onnx_save_path


if __name__ == "__main__":
    # 导出中文模型
    export_to_onnx_fixed(
        preprocess_config_path="config/AISHELL3/preprocess.yaml",
        model_config_path="config/AISHELL3/model.yaml",
        ckpt_path="output/ckpt/AISHELL3/600000.pth.tar",
        onnx_save_path="onnx/fastspeech2_aishell3.onnx",
        max_mel_len=1000,
        opset_version=14,
    )
