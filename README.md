# FastSpeech2 语音合成

基于 [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) 的改进版本，支持 ONNX 导出。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型权重

从 Google Drive 下载：
https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F

放置到对应目录：
```
output/ckpt/AISHELL3/
└── 600000.pth.tar          # 中文模型 (399.3 MB)
```

### 3. 解压声码器

```bash
# Windows PowerShell
Expand-Archive -Path hifigan/generator_universal.pth.tar.zip -DestinationPath hifigan/ -Force

# Linux/Mac
unzip -o hifigan/generator_universal.pth.tar.zip -d hifigan/
```

---

## 使用方法

### 方法零：API 服务（推荐）

启动 API 服务：
```bash
python api/tts_service.py
```

服务启动后访问 http://127.0.0.1:8000

**API 端点：**

```bash
# PyTorch 推理
curl -X POST "http://127.0.0.1:8000/tts/pth" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界"}'

# ONNX 推理
curl -X POST "http://127.0.0.1:8000/tts/onnx" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界"}'
```

返回示例：
```json
{
  "success": true,
  "message": "合成成功",
  "audio_path": "api_output/onnx_e94f0bfa.wav",
  "duration": 1.42
}
```

---

### 方法一：PyTorch 推理

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

输出：`output/result/AISHELL3/你好世界.wav`

---

### 方法二：ONNX 推理（推荐）

**步骤 1：导出 ONNX 模型**

```bash
python deploy/export_tensorrt_onnx.py
```

输出：`onnx/fastspeech2_tensorrt.onnx`

**步骤 2：ONNX 推理（生成 mel 频谱）**

```bash
# CPU 模式
python deploy/inference_tensorrt_onnx.py --text "你好世界"

# CUDA 模式（需安装 onnxruntime-gpu）
python deploy/inference_tensorrt_onnx.py --text "你好世界" --cuda
```

输出：`tensorrt_output/你好世界_mel.npy`

**步骤 3：生成音频**

```bash
python deploy/generate_audio_from_mel.py "tensorrt_output/你好世界_mel.npy"
```

输出：`tensorrt_output/你好世界.wav`

---

## TensorRT 加速（可选）

如需在 NVIDIA GPU 上获得更快推理速度，可将 ONNX 转换为 TensorRT：

```bash
# Windows（需安装 TensorRT）
trtexec --onnx=onnx/fastspeech2_tensorrt.onnx --saveEngine=fastspeech2.trt --fp16

# Jetson
/usr/src/tensorrt/bin/trtexec --onnx=onnx/fastspeech2_tensorrt.onnx --saveEngine=fastspeech2.trt --fp16
```

然后使用 TensorRT 引擎进行推理（需修改推理脚本加载 `.trt` 文件）。

---

## 项目结构

```
FastSpeech2/
├── inference/synthesize.py          # PyTorch 推理
├── deploy/
│   ├── export_tensorrt_onnx.py     # ONNX 导出
│   ├── inference_tensorrt_onnx.py  # ONNX 推理
│   └── generate_audio_from_mel.py  # 音频生成
├── output/ckpt/AISHELL3/           # 模型权重目录
├── onnx/                           # ONNX 模型输出目录
└── tensorrt_output/                # 推理结果目录
```

---

## 许可证

MIT License

原始项目：https://github.com/ming024/FastSpeech2
