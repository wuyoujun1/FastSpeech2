# ==================== 统一版：三模式门控Transformer（可复现版） ====================
# 四种运行标识：no_gate, classical, hybrid, classicallast
# 注意：classicallast 使用 classical 的门控，但专门用于寻找最差模型

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pennylane as qml
import os
import re
import json
import random
import warnings
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

warnings.filterwarnings('ignore')

# ==================== 配置类 ====================
class QGenConfig:
    def __init__(self, mode="classical"):
        # 注意：classicallast 使用 classical 的门控实现
        assert mode in ["no_gate", "classical", "hybrid", "classicallast"]
        self.mode = mode
        
        # 实际门控模式（classicallast 映射到 classical）
        self.actual_gate_mode = "classical" if mode == "classicallast" else mode
        
        # 模型架构
        self.vocab_size      = 5000
        self.sequence_length = 64
        self.embedding_dim   = 128
        self.num_heads       = 4
        self.num_layers      = 2
        self.head_dim        = 32
        self.hidden_dim      = 256
        self.dropout         = 0.5
        
        # 量子配置（仅hybrid模式使用）
        self.n_qubits        = 7
        self.n_qlayers       = 2
        self.q_device        = "default.qubit"
        
        # 训练配置
        self.batch_size      = 32
        self.epochs          = 50
        self.early_stop_patience = 7
        self.learning_rate   = 1e-3
        self.q_learning_rate = 2e-3
        self.grad_clip_norm  = 0.5
        
        # 生成配置
        self.generate_length = 61
        self.temperature     = 1.0
        self.top_k           = 50
        self.top_p           = 0.9
        self.repetition_penalty = 1.2
        self.pad_token_id    = 0
        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 量子电路（仅hybrid模式） ====================
class FastQuantumCircuit(nn.Module):
    def __init__(self, n_qubits, n_qlayers, q_device="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.dev = qml.device(q_device, wires=n_qubits)
        
        def _circuit(inputs, weights):
            theta = torch.pi * (inputs[:, :self.n_qubits] + 1) / 2
            phi = torch.pi * (inputs[:, self.n_qubits:] + 1) / 2
            
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(theta[:, i], wires=i)
                qml.RZ(phi[:, i], wires=i)
            
            for layer in range(self.n_qlayers):
                for i in range(self.n_qubits):
                    qml.RZ(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                if layer % 2 == 0:
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                else:
                    offset = self.n_qubits // 2
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + offset) % self.n_qubits])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")
        self.weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        self.q_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
        
        for param in self.q_linear.parameters():
            nn.init.uniform_(param, -np.pi, np.pi)
    
    def forward(self, x):
        if x.dim() == 3:
            B, L, _ = x.shape
            x_flat = x.view(-1, 2 * self.n_qubits)
            out = self.q_linear(x_flat)
            return out.view(B, L, self.n_qubits)
        else:
            return self.q_linear(x)

# ==================== 三种门控机制 ====================

class NoGate(nn.Module):
    """模式1: 无门控 - 恒为1"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.embedding_dim
    
    def forward(self, projected):
        return torch.ones_like(projected)

class ClassicalGate(nn.Module):
    """模式2: 经典门控 - 使用全局特征"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.embedding_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
    
    def forward(self, projected):
        B, L, _ = projected.shape
        global_feat = projected.mean(dim=1, keepdim=True).expand(B, L, -1)
        combined = torch.cat([projected, global_feat], dim=-1)
        return self.gate_mlp(combined)

class HybridGate(nn.Module):
    """模式3: 量子门控 - 使用量子电路"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.embedding_dim
        self.n_qubits = config.n_qubits
        self.gate_down = nn.Linear(self.d_model * 2, 2 * self.n_qubits)
        self.qc = FastQuantumCircuit(self.n_qubits, config.n_qlayers, config.q_device)
        self.gate_up = nn.Linear(self.n_qubits, self.d_model)
    
    def forward(self, projected):
        B, L, _ = projected.shape
        global_feat = projected.mean(dim=1, keepdim=True).expand(B, L, -1)
        combined = torch.cat([projected, global_feat], dim=-1)
        
        gate_in = torch.tanh(self.gate_down(combined))
        with torch.cuda.amp.autocast(enabled=False):
            q_out = self.qc(gate_in)
        return torch.sigmoid(self.gate_up(q_out))

# ==================== 注意力层 ====================
class PostAttentionGating(nn.Module):
    def __init__(self, config: QGenConfig):
        super().__init__()
        self.d_model = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.total_dim = self.num_heads * self.head_dim
        # 使用 actual_gate_mode 来决定门控类型
        self.mode = config.actual_gate_mode
        
        self.qkv = nn.Linear(self.d_model, 3 * self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.d_model)
        
        # 根据 actual_gate_mode 选择门控（classicallast 也会用 classical）
        if self.mode == "no_gate":
            self.gate_module = NoGate(config)
        elif self.mode == "classical":
            self.gate_module = ClassicalGate(config)
        elif self.mode == "hybrid":
            self.gate_module = HybridGate(config)
        
        self.dropout = nn.Dropout(config.dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, is_causal=True):
        B, L, _ = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn_out = torch.matmul(attn, v)
        
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.total_dim)
        projected = self.out_proj(attn_out)
        
        # 应用门控
        gate = self.gate_module(projected)
        
        gated_out = gate * projected
        output = x + self.dropout(gated_out)
        
        return output, gate

# ==================== Transformer层和模型 ====================
class GatedTransformerLayer(nn.Module):
    def __init__(self, config: QGenConfig):
        super().__init__()
        self.attn = PostAttentionGating(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, x, is_causal=True):
        attn_out, gate = self.attn(self.norm1(x), is_causal)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, gate

class GatedTransformer(nn.Module):
    def __init__(self, config: QGenConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embed = nn.Embedding(config.sequence_length, config.embedding_dim)
        
        self.layers = nn.ModuleList([
            GatedTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, input_ids, is_causal=True, return_gate=False):
        B, L = input_ids.shape
        
        x = self.embedding(input_ids)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)
        x = self.dropout(x)
        
        all_gates = []
        for layer in self.layers:
            x, gate = layer(x, is_causal)
            all_gates.append(gate)
        
        x = self.norm(x)
        logits = self.head(x)
        
        if return_gate:
            return logits, all_gates
        return logits
    
    @torch.no_grad()
    def generate(self, prompt, word_to_idx, idx_to_word, max_length=61,
                 temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0):
        device = next(self.parameters()).device
        words = prompt.lower().split()
        input_ids = [word_to_idx.get(w, word_to_idx.get("<unk>", 0)) for w in words]
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        generated = words.copy()
        generated_token_ids = input_ids[0].tolist()

        for _ in range(max_length):
            context_size = self.config.sequence_length - 1
            if input_ids.size(1) > context_size:
                curr_input = input_ids[:, -context_size:]
            else:
                curr_input = input_ids
                
            logits = self.forward(curr_input, is_causal=True)[:, -1, :] / temperature
            
            if repetition_penalty > 1.0:
                for token_id in set(generated_token_ids):
                    logits[:, token_id] /= repetition_penalty
                    
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('inf'))
                
            probs = F.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            generated_token_ids.append(next_token_idx)
            next_word = idx_to_word.get(next_token_idx, "<unk>")
            generated.append(next_word)
            if next_word == "<eos>":
                break
            new_token = torch.tensor([[next_token_idx]], dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, new_token], dim=1)
            
        return " ".join(generated)

# ==================== 工具函数 ====================
CHECKPOINT_DIR = "checkpoints_unified"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def set_seed(seed):
    """设置所有随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_save_path(mode, seed):
    """模型保存路径"""
    return f"gated_transformer_{mode}_seed{seed}_unified.pth"

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def load_data(file_path="data/alice_1.txt", min_freq=1, sequence_length=64, train_ratio=0.8, seed=42):
    """加载数据，使用指定seed确保划分一致"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    word_counts = Counter(words)
    valid_words = [w for w, cnt in word_counts.items() if cnt >= min_freq]
    word_to_idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for word in valid_words:
        word_to_idx[word] = len(word_to_idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    
    indexed_words = [word_to_idx.get(w, 1) for w in words]
    
    sequences = []
    for i in range(len(indexed_words) - sequence_length):
        x = indexed_words[i:i+sequence_length]
        y = indexed_words[i+1:i+sequence_length+1]
        sequences.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
    
    # 关键：使用传入的seed进行shuffle，确保可复现
    random.seed(seed)
    random.shuffle(sequences)
    split_idx = int(len(sequences) * train_ratio)
    train_sequences = sequences[:split_idx]
    test_sequences = sequences[split_idx:]
    
    print(f"词汇表大小: {vocab_size}")
    print(f"训练集: {len(train_sequences)} | 测试集: {len(test_sequences)}")
    
    return train_sequences, test_sequences, word_to_idx, idx_to_word, vocab_size

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]

# ==================== 训练和验证 ====================
def train_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    total_loss = 0.0
    total_gate_mean = 0.0
    total_gate_std = 0.0
    num_batches = 0
    
    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(config.device)
        target_ids = target_ids.to(config.device)
        
        # 使用 actual_gate_mode 判断是否使用门控
        if config.actual_gate_mode in ["classical", "hybrid"]:
            logits, gates = model(input_ids, is_causal=True, return_gate=True)
            all_gates = torch.cat([g.view(-1) for g in gates])
            gate_mean = all_gates.mean().item()
            gate_std = all_gates.std().item()
            total_gate_mean += gate_mean
            total_gate_std += gate_std
        else:
            logits = model(input_ids, is_causal=True)
            gate_mean = 0.0
            gate_std = 0.0
        
        loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        if config.actual_gate_mode == "hybrid":
            quantum_params = [p for n, p in model.named_parameters() 
                            if "qc" in n and p.requires_grad and p.grad is not None]
            if quantum_params:
                torch.nn.utils.clip_grad_norm_(quantum_params, max_norm=config.grad_clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * input_ids.size(0)
        num_batches += 1
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_gate_mean = total_gate_mean / num_batches if num_batches > 0 else 0
    avg_gate_std = total_gate_std / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_gate_mean, avg_gate_std

def validate_epoch(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(config.device)
            target_ids = target_ids.to(config.device)
            logits = model(input_ids, is_causal=True)
            loss = criterion(logits.reshape(-1, config.vocab_size), target_ids.reshape(-1))
            total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classic = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "qc" not in n)
    quantum = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "qc" in n)
    return total, classic, quantum

def run_training(mode="classical", seed=42, force_restart=True):
    """
    训练指定模式的模型
    
    Args:
        mode: "no_gate", "classical", "hybrid", "classicallast"
        seed: 随机种子
        force_restart: 是否强制从头训练（默认True确保可复现）
    """
    print(f"\n{'='*80}")
    print(f"🚀 训练模式: {mode.upper()} | Seed: {seed}")
    print(f"{'='*80}")

    # 设置随机种子
    set_seed(seed)

    config = QGenConfig(mode=mode)
    
    # 打印配置信息
    mode_desc = {
        "no_gate": "无门控",
        "classical": "经典全局门控",
        "hybrid": "量子全局门控",
        "classicallast": "经典全局门控（用于寻找最差模型）"
    }
    print(f"门控类型: {mode_desc[mode]}")
    print(f"实际门控实现: {config.actual_gate_mode}")
    print(f"配置: 序列长度={config.sequence_length}, 嵌入维度={config.embedding_dim}, 层数={config.num_layers}")

    # 加载数据
    print("\n📥 加载数据...")
    train_sequences, test_sequences, word_to_idx, idx_to_word, vocab_size = load_data(
        file_path="data/alice_1.txt", 
        min_freq=1, 
        sequence_length=config.sequence_length,
        train_ratio=0.8,
        seed=seed
    )
    config.vocab_size = vocab_size

    train_loader = DataLoader(TextDataset(train_sequences), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TextDataset(test_sequences), batch_size=config.batch_size, shuffle=False)

    # 创建模型
    model = GatedTransformer(config).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 统计参数量
    total_params, classic_params, quantum_params = count_parameters(model)
    print(f"\n📊 参数量统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  经典参数: {classic_params:,}")
    if quantum_params > 0:
        print(f"  量子参数: {quantum_params:,} ({quantum_params/total_params:.2%})")

    save_path = get_model_save_path(mode, seed)
    early_stopper = EarlyStopper(patience=config.early_stop_patience, min_delta=0.005)

    gate_mean, gate_std = 0.0, 0.0
    best_val_loss = float("inf")
    
    try:
        for epoch in range(config.epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{config.epochs} | 早停计数: {early_stopper.counter}/{early_stopper.patience}")
            print(f"{'='*80}")

            train_loss, gate_mean, gate_std = train_epoch(
                model, train_loader, criterion, optimizer, config
            )
            test_loss = validate_epoch(model, test_loader, criterion, config)
            
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            gate_info = f" | gate_mean={gate_mean:.3f}, std={gate_std:.3f}" if config.actual_gate_mode != "no_gate" else ""
            print(f"训练损失: {train_loss:.4f}{gate_info}")
            print(f"测试损失: {test_loss:.4f}")
            print(f"学习率: {current_lr:.6f}")

            # 门控监控
            if config.actual_gate_mode != "no_gate" and epoch % 5 == 0:
                if gate_mean < 0.1:
                    print(f"⚠️ 门控均值过低 ({gate_mean:.3f})，可能梯度消失")
                elif gate_mean > 0.9:
                    print(f"⚠️ 门控均值过高 ({gate_mean:.3f})，可能过度激活")

            # 早停检查
            if early_stopper.step(test_loss):
                print("🛑 早停触发！")
                break

            # 保存最佳模型
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                print("✅ 保存最佳模型...")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "word_to_idx": word_to_idx,
                    "idx_to_word": idx_to_word,
                    "config": config.__dict__,
                    "best_val_loss": best_val_loss,
                    "total_params": total_params,
                    "seed": seed
                }, save_path)

    except KeyboardInterrupt:
        print("\n⚠️ 训练中断！")
        return None

    print(f"\n{'='*80}")
    print(f"🎉 训练完成！")
    print(f"总参数量: {total_params:,}")
    print(f"最佳测试损失: {best_val_loss:.4f}")
    if config.actual_gate_mode != "no_gate":
        print(f"最终门控均值: {gate_mean:.3f}, 标准差: {gate_std:.3f}")
    print(f"模型已保存: {save_path}")
    print(f"{'='*80}")
    
    return save_path

# ==================== 评估函数 ====================
SEQ_LENGTH = 64
BATCH_SIZE = 32

def load_data_for_eval(file_path="data/alice_1.txt", min_freq=1, seed=42):
    """评估时加载数据，必须与训练时使用相同的seed"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    word_counts = Counter(words)
    valid_words = [w for w, cnt in word_counts.items() if cnt >= min_freq]
    word_to_idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for word in valid_words:
        word_to_idx[word] = len(word_to_idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    indexed = [word_to_idx.get(w, 1) for w in words]
    sequences = []
    for i in range(len(indexed) - SEQ_LENGTH):
        x = indexed[i:i+SEQ_LENGTH]
        y = indexed[i+1:i+SEQ_LENGTH+1]
        sequences.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
    
    # 关键：使用与训练相同的seed
    random.seed(seed)
    random.shuffle(sequences)
    split_idx = int(len(sequences) * 0.8)
    
    return sequences[:split_idx], sequences[split_idx:], word_to_idx, idx_to_word

def compute_perplexity(model, sequences, config, device):
    """计算困惑度"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch = sequences[i:i+BATCH_SIZE]
            if not batch:
                continue
            x = torch.stack([s[0] for s in batch]).to(device)
            y = torch.stack([s[1] for s in batch]).to(device)
            
            logits = model(x, is_causal=True)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            
            total_loss += loss.item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss), avg_loss

def evaluate_model_full(model, train_seq, test_seq, word_to_idx, idx_to_word, config, device):
    """
    完整评估模型，返回所有指标
    """
    model.eval()
    
    # 1. 困惑度
    print("\n📊 计算困惑度...")
    train_ppl, _ = compute_perplexity(model, train_seq, config, device)
    test_ppl, _ = compute_perplexity(model, test_seq, config, device)
    
    # 2. BLEU分数
    print(f"📊 计算BLEU分数（全量测试集: {len(test_seq)} 样本）...")
    num_samples = len(test_seq)
    indices = list(range(num_samples))
    
    all_refs, all_hyps = [], []
    sentence_bleu1_scores = []
    sentence_bleu2_scores = []
    sentence_bleu4_scores = []
    smooth = SmoothingFunction().method1
    
    for i, idx in enumerate(indices):
        if (i + 1) % max(1, num_samples // 10) == 0 or i == 0:
            print(f"  进度: {i+1}/{num_samples}")
        
        x, y = test_seq[idx]
        prompt_words = [idx_to_word.get(w.item(), "<unk>") for w in x[:4]]
        prompt_text = ' '.join(prompt_words)
        ref_words = [idx_to_word.get(w.item(), "<unk>") for w in x[4:]] + [idx_to_word.get(y[-1].item(), "<unk>")]
        
        generated = model.generate(
            prompt=prompt_text,
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            max_length=61,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2
        )
        gen_words = generated.split()
        
        all_refs.append([ref_words])
        all_hyps.append(gen_words)
        
        bleu1 = sentence_bleu([ref_words], gen_words, weights=(1,0,0,0), smoothing_function=smooth)
        bleu2 = sentence_bleu([ref_words], gen_words, weights=(0.5,0.5,0,0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_words], gen_words, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
        
        sentence_bleu1_scores.append(bleu1)
        sentence_bleu2_scores.append(bleu2)
        sentence_bleu4_scores.append(bleu4)
    
    # 计算统计指标
    corpus_bleu4 = corpus_bleu(all_refs, all_hyps)
    sentence_bleu1_mean = np.mean(sentence_bleu1_scores)
    sentence_bleu2_mean = np.mean(sentence_bleu2_scores)
    sentence_bleu4_mean = np.mean(sentence_bleu4_scores)
    
    # SacreBLEU
    try:
        from sacrebleu import corpus_bleu as sacre_corpus_bleu
        ref_strs = [' '.join(r[0]) for r in all_refs]
        hyp_strs = [' '.join(h) for h in all_hyps]
        sacrebleu = sacre_corpus_bleu(hyp_strs, [ref_strs]).score
    except:
        sacrebleu = 0.0
    
    # 门控分析
    gate_mean = 0.0
    if config.actual_gate_mode in ["classical", "hybrid"]:
        print("\n📊 分析门控...")
        sample_indices = random.sample(range(len(test_seq)), min(100, len(test_seq)))
        all_gates = []
        
        model.eval()
        with torch.no_grad():
            for idx in sample_indices:
                x, _ = test_seq[idx]
                x = x.unsqueeze(0).to(device)
                _, gates = model(x, is_causal=True, return_gate=True)
                all_gates.append(gates[0].cpu())
        
        if all_gates:
            all_gates_tensor = torch.cat(all_gates, dim=0)
            gate_mean = all_gates_tensor.mean().item()
    
    return {
        'corpus_bleu4': corpus_bleu4,
        'sacrebleu': sacrebleu,
        'sentence_bleu1_mean': sentence_bleu1_mean,
        'sentence_bleu2_mean': sentence_bleu2_mean,
        'sentence_bleu4_mean': sentence_bleu4_mean,
        'gate_mean': gate_mean,
        'test_ppl': test_ppl,
        'train_ppl': train_ppl
    }

def compute_composite_score(metrics, weights=None):
    """
    计算综合评分（越低表示模型越差）
    默认权重与第二份代码一致
    """
    if weights is None:
        weights = {
            'corpus_bleu4': 0.25,
            'sacrebleu': 0.25,
            'sentence_bleu1_mean': 0.15,
            'sentence_bleu2_mean': 0.15,
            'sentence_bleu4_mean': 0.20
        }
    
    # SacreBLEU归一化到0-1
    normalized_sacrebleu = metrics['sacrebleu'] / 100.0
    
    score = (
        weights['corpus_bleu4'] * metrics['corpus_bleu4'] +
        weights['sacrebleu'] * normalized_sacrebleu +
        weights['sentence_bleu1_mean'] * metrics['sentence_bleu1_mean'] +
        weights['sentence_bleu2_mean'] * metrics['sentence_bleu2_mean'] +
        weights['sentence_bleu4_mean'] * metrics['sentence_bleu4_mean']
    )
    
    return score

def run_single_evaluation(mode, seed, device):
    """
    评估单个seed的模型（不自动训练，只评估已存在的模型）
    
    Args:
        mode: 模型模式
        seed: 随机种子
        device: 计算设备
    
    Returns:
        dict: 包含所有评估指标的结果字典
    """
    print(f"\n{'='*80}")
    print(f"🔍 评估模式: {mode.upper()} | Seed: {seed}")
    print(f"{'='*80}")
    
    model_path = get_model_save_path(mode, seed)
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return None
    
    # 加载模型
    print(f"📦 加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = QGenConfig(mode=checkpoint["config"]["mode"])
    for k, v in checkpoint["config"].items():
        setattr(config, k, v)
    config.device = device
    
    model = GatedTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载成功: {total_params:,} 参数")
    
    # 加载数据（使用相同seed）
    print(f"\n📥 加载数据（seed={seed}）...")
    train_seq, test_seq, word_to_idx, idx_to_word = load_data_for_eval(seed=seed)
    print(f"训练集: {len(train_seq)} | 测试集: {len(test_seq)} | 词表: {len(word_to_idx)}")
    
    # 完整评估
    metrics = evaluate_model_full(model, train_seq, test_seq, word_to_idx, idx_to_word, config, device)
    
    # 计算综合评分
    composite_score = compute_composite_score(metrics)
    
    result = {
        'seed': seed,
        'mode': mode,
        'composite_score': composite_score,
        **metrics,
        'model_path': model_path
    }
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"📋 评估结果 Seed {seed}")
    print(f"{'='*80}")
    print(f"综合评分:        {composite_score:.6f} (越低越差)")
    print(f"Corpus BLEU-4:   {metrics['corpus_bleu4']:.4f}")
    print(f"SacreBLEU:       {metrics['sacrebleu']:.2f}")
    print(f"Sentence BLEU-1: {metrics['sentence_bleu1_mean']:.4f}")
    print(f"Sentence BLEU-2: {metrics['sentence_bleu2_mean']:.4f}")
    print(f"Sentence BLEU-4: {metrics['sentence_bleu4_mean']:.4f}")
    print(f"门控均值:        {metrics['gate_mean']:.4f}")
    print(f"Test PPL:        {metrics['test_ppl']:.4f}")
    print(f"Train PPL:       {metrics['train_ppl']:.4f}")
    print(f"{'='*80}")
    
    return result

def find_worst_models(mode="classicallast", num_seeds=100, start_seed=0, weights=None):
    """
    循环评估多个seed，找出综合评分最差的5个模型
    只有 classicallast 模式会调用此函数
    
    Args:
        mode: 必须是 "classicallast"
        num_seeds: 评估的seed数量
        start_seed: 起始seed
        weights: 自定义指标权重
    
    Returns:
        list: 最差的5个模型的结果列表
    """
    if mode != "classicallast":
        print(f"⚠️ 警告：{mode} 模式不需要找最差模型，只有 classicallast 模式支持")
        return []
    
    print(f"\n{'='*80}")
    print(f"🎯 寻找 CLASSICAL 模式中综合评分最差的5个模型")
    print(f"（使用 classicallast 标识运行）")
    print(f"评价指标: Corpus BLEU-4 + SacreBLEU + Sentence BLEU-1/2/4")
    print(f"Seed范围: {start_seed} 到 {start_seed + num_seeds - 1}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    all_results = []
    
    for i in range(num_seeds):
        seed = start_seed + i
        print(f"\n{'='*80}")
        print(f"[{i+1}/{num_seeds}] Seed {seed}")
        
        # 训练（强制从头训练确保可复现）
        run_training(mode, seed=seed, force_restart=True)
        
        # 评估
        result = run_single_evaluation(mode, seed, device)
        
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("❌ 没有成功评估任何模型")
        return []
    
    # 按综合评分升序排序（最差的在前）
    all_results_sorted = sorted(all_results, key=lambda x: x['composite_score'])
    
    # 取最差的5个
    worst_5 = all_results_sorted[:5]
    
    # 详细报告
    print(f"\n{'='*80}")
    print(f"📊 评估完成！总模型数: {len(all_results)}")
    print(f"{'='*80}")
    print(f"\n🔴 综合评分最差的5个模型（从低到高）:")
    print(f"{'='*80}")
    
    for rank, result in enumerate(worst_5, 1):
        print(f"\n【第{rank}差】Seed {result['seed']}")
        print(f"  综合评分:        {result['composite_score']:.6f} ⭐")
        print(f"  Corpus BLEU-4:   {result['corpus_bleu4']:.6f}")
        print(f"  SacreBLEU:       {result['sacrebleu']:.2f}")
        print(f"  Sentence BLEU-1: {result['sentence_bleu1_mean']:.4f}")
        print(f"  Sentence BLEU-2: {result['sentence_bleu2_mean']:.4f}")
        print(f"  Sentence BLEU-4: {result['sentence_bleu4_mean']:.4f}")
        print(f"  门控均值:        {result['gate_mean']:.4f}")
        print(f"  Test PPL:        {result['test_ppl']:.4f}")
        print(f"  模型路径:        {result['model_path']}")
    
    # 保存结果
    output_data = {
        'mode': mode,
        'actual_gate': 'classical',
        'weights': weights if weights else {
            'corpus_bleu4': 0.25,
            'sacrebleu': 0.25,
            'sentence_bleu1_mean': 0.15,
            'sentence_bleu2_mean': 0.15,
            'sentence_bleu4_mean': 0.20
        },
        'total_evaluated': len(all_results),
        'worst_5': worst_5,
        'all_results': all_results
    }
    
    output_file = f"worst_5_classical_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n{'='*80}")
    print(f"✅ 完整结果已保存: {output_file}")
    print(f"{'='*80}")
    
    # 统计信息
    all_scores = [r['composite_score'] for r in all_results]
    all_corpus = [r['corpus_bleu4'] for r in all_results]
    all_sacre = [r['sacrebleu'] for r in all_results]
    
    print(f"\n📈 综合评分统计:")
    print(f"  最小值: {min(all_scores):.6f}")
    print(f"  最大值: {max(all_scores):.6f}")
    print(f"  平均值: {np.mean(all_scores):.6f}")
    print(f"  标准差: {np.std(all_scores):.6f}")
    
    print(f"\n📈 Corpus BLEU-4 统计:")
    print(f"  最小值: {min(all_corpus):.6f}")
    print(f"  最大值: {max(all_corpus):.6f}")
    print(f"  平均值: {np.mean(all_corpus):.6f}")
    
    print(f"\n📈 SacreBLEU 统计:")
    print(f"  最小值: {min(all_sacre):.2f}")
    print(f"  最大值: {max(all_sacre):.2f}")
    print(f"  平均值: {np.mean(all_sacre):.2f}")
    
    return worst_5

# ==================== 统一配置区域 ====================

# 运行模式选择：
# "train" - 只训练
# "eval"  - 只评估（模型必须已存在）
# "both"  - 训练+评估（默认）
TRAINING_MODE = "both"

# 模型模式: "no_gate", "classical", "hybrid", "classicallast"
# 注意：classicallast 使用 classical 的门控，但会循环找最差5个模型
MODEL_MODE = "hybrid"

# 随机种子设置（用于 single seed 模式）
SEED = 10                     

# classicallast 模式专用：寻找最差模型的参数
NUM_SEEDS = 86               # 评估的seed数量
START_SEED = 0                # 起始seed

# 指标权重（可自定义）
METRIC_WEIGHTS = {
    'corpus_bleu4': 0.25,
    'sacrebleu': 0.25,
    'sentence_bleu1_mean': 0.15,
    'sentence_bleu2_mean': 0.15,
    'sentence_bleu4_mean': 0.20
}

# 强制从头训练（确保可复现）
FORCE_RESTART = False

# ===================================================

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 判断是否使用 classicallast 模式（找最差模型）
    if MODEL_MODE == "classicallast":
        # classicallast 模式：循环多个seed，找最差5个
        worst_5 = find_worst_models(MODEL_MODE, num_seeds=NUM_SEEDS, start_seed=START_SEED, weights=METRIC_WEIGHTS)
        
        if worst_5:
            print(f"\n{'='*80}")
            print(f"🎯 最终报告: CLASSICAL 模式综合评分最差的5个种子")
            print(f"{'='*80}")
            for i, r in enumerate(worst_5, 1):
                print(f"{i}. Seed {r['seed']}: 综合评分={r['composite_score']:.6f}, "
                      f"Corpus BLEU-4={r['corpus_bleu4']:.4f}, "
                      f"SacreBLEU={r['sacrebleu']:.2f}, "
                      f"Sentence BLEU-4={r['sentence_bleu4_mean']:.4f}")
    else:
        # 其他三种模式：单种子训练/评估
        if TRAINING_MODE in ["train", "both"]:
            # 训练
            run_training(MODEL_MODE, seed=SEED, force_restart=FORCE_RESTART)
        
        if TRAINING_MODE in ["eval", "both"]:
            # 评估
            result = run_single_evaluation(MODEL_MODE, SEED, device)
            if result:
                # 保存结果
                output_file = f"evaluation_{MODEL_MODE}_seed{SEED}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n✅ 结果已保存: {output_file}")
            else:
                print(f"\n❌ 评估失败，请检查模型是否存在")