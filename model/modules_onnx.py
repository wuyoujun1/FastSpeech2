"""
ONNX 兼容的 LengthRegulator
使用矩阵操作替代动态循环
"""

import torch
import torch.nn as nn


class LengthRegulatorONNX(nn.Module):
    """
    ONNX 友好的 Length Regulator
    使用固定最大长度，避免动态循环
    """
    def __init__(self, max_mel_len=2000):
        super(LengthRegulatorONNX, self).__init__()
        self.max_mel_len = max_mel_len
        
    def forward(self, x, duration, max_len):
        """
        Args:
            x: [batch, seq_len, hidden]
            duration: [batch, seq_len] - 每个音素的持续时间
            max_len: scalar (未使用，为了兼容性保留)
        Returns:
            output: [batch, max_mel_len, hidden]
            mel_len: [batch] - 实际的 mel 长度
        """
        batch_size, seq_len, hidden = x.shape
        device = x.device
        
        # 计算累计长度（每个音素的结束位置）
        cumsum_duration = torch.cumsum(duration, dim=1)  # [batch, seq_len]
        
        # 创建 mel 位置索引 [0, 1, 2, ..., max_mel_len-1]
        mel_positions = torch.arange(self.max_mel_len, device=device).view(1, 1, -1).expand(batch_size, -1, -1)
        # mel_positions: [batch, 1, max_mel_len]
        
        # 扩展 cumsum 用于比较 [batch, seq_len, 1]
        cumsum_expanded = cumsum_duration.unsqueeze(2)  # [batch, seq_len, 1]
        
        # 创建 mask: mel_positions < cumsum_duration
        # [batch, 1, max_mel_len] < [batch, seq_len, 1] -> [batch, seq_len, max_mel_len]
        mask = mel_positions < cumsum_expanded  # [batch, seq_len, max_mel_len]
        
        # 计算前缀和，找到每个 mel 帧对应的音素索引
        # 如果 mask 从 False 变为 True，说明是这个音素的开始
        mask_float = mask.float()
        
        # 使用 cumsum 找到每个位置对应的音素索引
        # 对于每个 mel 位置，找到第一个 cumsum > position 的音素
        phoneme_idx = torch.argmax(mask_float, dim=1)  # [batch, max_mel_len]
        
        # 处理超出范围的情况（所有 mask 都是 False）
        all_false = (mask_float.sum(dim=1) == 0)  # [batch, max_mel_len]
        phoneme_idx = torch.where(all_false, torch.tensor(seq_len - 1, device=device), phoneme_idx)
        
        # 限制索引范围
        phoneme_idx = torch.clamp(phoneme_idx, 0, seq_len - 1)  # [batch, max_mel_len]
        
        # 使用 gather 获取对应的特征
        # phoneme_idx: [batch, max_mel_len] -> [batch, max_mel_len, 1] -> [batch, max_mel_len, hidden]
        phoneme_idx_expanded = phoneme_idx.unsqueeze(-1).expand(-1, -1, hidden)
        
        # gather: [batch, seq_len, hidden] 按照 phoneme_idx 在 dim=1 上 gather
        output = torch.gather(x, 1, phoneme_idx_expanded)  # [batch, max_mel_len, hidden]
        
        # 计算实际的 mel 长度
        mel_len = duration.sum(dim=1).long()  # [batch]
        mel_len = torch.clamp(mel_len, max=self.max_mel_len)
        
        return output, mel_len


# 兼容性：保留原类名
LengthRegulator = LengthRegulatorONNX
