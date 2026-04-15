# FastSpeech2 语音合成

基于 [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) 的改进版本，支持 ONNX 导出和 TensorRT 加速。

## 特性

- ✅ 完整的中文语音合成（AISHELL3 数据集）
- ✅ ONNX 模型导出（支持 TensorRT）
- ✅ Windows / Linux / Jetson 部署
- ✅ CPU / GPU 兼容

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/FastSpeech2.git
cd FastSpeech2
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

requirements.txt:
```
torch>=2.0.0
numpy
pyyaml
pypinyin
g2p-en
onnx
onnxruntime
soundfile
matplotlib
```

### 3. 下载预训练模型

从 Google Drive 下载模型权重：

**下载地址**: https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F

下载以下文件并放到对应目录：

```
output/ckpt/AISHELL3/
└── 600000.pth.tar          # 中文模型 (399.3 MB)

# 可选：英文模型
output/ckpt/LJSpeech/
└── 900000.pth.tar          # 英文单说话人 (337.6 MB)

output/ckpt/LibriTTS/
└── 800000.pth.tar          # 英文多说话人 (401.3 MB)
```

### 4. 解压声码器模型

```bash
# Windows PowerShell
Expand-Archive -Path hifigan/generator_universal.pth.tar.zip -DestinationPath hifigan/ -Force

# Linux/Mac
unzip -o hifigan/generator_universal.pth.tar.zip -d hifigan/
```

### 5. 运行语音合成

#### PyTorch 版本

```bash
python inference/synthesize.py \
    --text "你好世界" \
    --restore_step 600000 \
    --mode single \
    -p config/AISHELL3/preprocess.yaml \
    -m config/AISHELL3/model.yaml \
    -t config/AISHELL3/train.yaml \
    --speaker_id 0
```

输出文件：`output/result/AISHELL3/你好世界.wav`

#### ONNX 版本

```bash
# ONNX 推理
python deploy/inference_tensorrt_onnx.py --text "你好世界"

# 生成音频（自动截取有效部分）
python deploy/generate_audio_from_mel.py "tensorrt_output/你好世界_mel.npy"
```

## 项目结构

```
FastSpeech2/
├── config/                 # 配置文件
│   ├── AISHELL3/          # 中文模型配置
│   ├── LJSpeech/          # 英文单说话人配置
│   └── LibriTTS/          # 英文多说话人配置
├── inference/             # 推理脚本
│   └── synthesize.py      # PyTorch 推理
├── deploy/                # ONNX/TensorRT 部署
│   ├── export_tensorrt_onnx.py      # ONNX 导出
│   ├── inference_tensorrt_onnx.py   # ONNX 推理
│   └── generate_audio_from_mel.py   # 音频生成
├── model/                 # 模型定义
│   ├── fastspeech2.py     # FastSpeech2 模型
│   ├── modules.py         # 模型模块
│   └── modules_onnx.py    # ONNX 友好的模块
├── utils/                 # 工具函数
├── text/                  # 文本处理
├── transformer/           # Transformer 架构
├── hifigan/               # 声码器
├── output/                # 输出目录
│   ├── ckpt/             # 模型权重
│   └── result/           # 合成结果
└── lexicon/               # 发音词典
```

## ONNX 导出与 TensorRT 部署

### 1. 导出 ONNX 模型

```bash
python deploy/export_tensorrt_onnx.py
```

输出：`onnx/fastspeech2_tensorrt.onnx`

### 2. 转换为 TensorRT（可选）

#### Windows

```bash
# 安装 TensorRT 后
"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\8.6.1.6\bin\trtexec.exe" \
    --onnx=onnx/fastspeech2_tensorrt.onnx \
    --saveEngine=fastspeech2.trt \
    --fp16
```

#### Jetson

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=onnx/fastspeech2_tensorrt.onnx \
    --saveEngine=fastspeech2_jetson.trt \
    --fp16 \
    --workspace=2048
```

### 3. ONNX 推理

```bash
# CPU 模式
python deploy/inference_tensorrt_onnx.py --text "你好世界"

# CUDA 模式（需要安装 onnxruntime-gpu）
python deploy/inference_tensorrt_onnx.py --text "你好世界" --cuda
```

输出文件：
- `tensorrt_output/你好世界_mel.npy` - 梅尔频谱
- `tensorrt_output/你好世界.wav` - 音频文件

## 技术细节

### ONNX 导出修改

原 `LengthRegulator` 使用 Python 动态循环，无法导出 ONNX。修改为矩阵操作：

```python
# model/modules_onnx.py
class LengthRegulatorONNX(nn.Module):
    def forward(self, x, duration, max_len):
        # 使用 mask 和 cumsum 替代循环
        mask = (positions >= start) & (positions < end)
        output = (x_expanded * mask_expanded).sum(dim=1)
        return output, mel_len
```

### 音频长度自动截取

ONNX 模型输出固定 2000 帧 mel，使用自适应阈值截取有效部分：

```python
# 能量阈值 = 中位数 + 2×标准差
threshold = np.median(energy) + 2 * np.std(energy)
valid_frames = np.where(energy > threshold)[0]
```

## 性能对比

| 平台 | 后端 | 精度 | 推理时间 |
|-----|------|-----|---------|
| CPU | PyTorch | FP32 | ~1000ms |
| CPU | ONNX Runtime | FP32 | ~800ms |
| GPU | TensorRT | FP16 | ~25ms |
| Jetson NX | TensorRT | FP16 | ~30ms |

## 常见问题

### 1. 模型权重下载

权重文件太大（>100MB），无法上传到 GitHub。请从 Google Drive 下载：
https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F

### 2. CPU/GPU 兼容

代码已自动检测设备：
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

如需强制使用 CPU：
```python
device = torch.device("cpu")
```

### 3. 中文支持

确保安装 `pypinyin`：
```bash
pip install pypinyin
```

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

原始 FastSpeech2 项目：https://github.com/ming024/FastSpeech2

## 致谢

- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) - 原始 FastSpeech2 实现
- [AISHELL-3](https://www.aishelltech.com/aishell_3) - 中文语音数据集
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - 高质量声码器
