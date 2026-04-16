# FastSpeech2 语音合成（中文）

基于 [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) 的改进版本，支持 ONNX 导出和 TensorRT 加速。

---

## 🚀 小白快速上手（3分钟）

### 第1步：安装依赖

```bash
pip install -r requirements.txt
```

### 第2步：下载模型

1. 打开 Google Drive 链接：https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F
2. 下载 `600000.pth.tar`（中文模型，约400MB）
3. 放到文件夹：`output/ckpt/AISHELL3/600000.pth.tar`

### 第3步：解压声码器

**Windows:**
```powershell
Expand-Archive -Path hifigan/generator_universal.pth.tar.zip -DestinationPath hifigan/ -Force
```

**Mac/Linux:**
```bash
unzip -o hifigan/generator_universal.pth.tar.zip -d hifigan/
```

### 第4步：运行！

**最简单的方式 - Python代码调用：**

```python
from tts_api import tts

# 说一句话
tts("你好世界")

# 搞定！音频保存在 tts_output/你好世界.wav
```

**或者命令行方式：**

```bash
python inference/synthesize.py --text "你好世界" --restore_step 600000 --mode single -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml --speaker_id 0
```

音频保存在 `output/result/AISHELL3/你好世界.wav`

---

## 📁 项目结构

```
FastSpeech2-master/
├── tts_api.py              # ⭐ 最简单的API，一行代码调用
├── inference/synthesize.py # 命令行推理脚本
├── config/AISHELL3/        # 中文配置文件
├── model/                  # 模型定义
├── deploy/                 # ONNX/TensorRT相关
│   ├── export_tensorrt_onnx.py      # 导出ONNX模型
│   ├── inference_tensorrt_onnx.py   # ONNX推理
│   └── generate_audio_from_mel.py   # 从mel生成音频
└── output/ckpt/AISHELL3/   # 模型权重存放位置
```

---

## 🔧 进阶用法

### 方式一：Python API（推荐）

```python
from tts_api import tts, tts_onnx

# PyTorch模型（默认，第一次加载较慢）
audio_path = tts("今天天气真不错")

# 指定输出路径
audio_path = tts("你好", output_path="hello.wav")

# ONNX模型（更快，需要先导出）
audio_path = tts_onnx("你好世界")
```

### 方式二：ONNX推理（更快）

**导出ONNX模型（只需一次）：**
```bash
python deploy/export_tensorrt_onnx.py
```

**使用ONNX推理：**
```bash
# 生成mel频谱
python deploy/inference_tensorrt_onnx.py --text "你好世界"

# 生成音频
python deploy/generate_audio_from_mel.py "tensorrt_output/你好世界_mel.npy"
```

### 方式三：TensorRT加速（NVIDIA GPU最快）

```bash
# 1. 导出ONNX（如果还没做）
python deploy/export_tensorrt_onnx.py

# 2. 转换为TensorRT引擎
# Windows（需安装TensorRT）
trtexec --onnx=onnx/fastspeech2_tensorrt.onnx --saveEngine=fastspeech2.trt --fp16

# 3. 使用TensorRT推理（参考 deploy/TENSORRT_GUIDE.md）
```

---

## ❓ 常见问题

**Q: 第一次运行很慢？**  
A: 正常！第一次需要加载模型（约400MB），之后就会很快。

**Q: 可以生成多长的语音？**  
A: 建议一次不超过100个字，太长可以分段生成。

**Q: 支持英文吗？**  
A: 当前配置是中文模型，主要支持中文。如需英文合成，请下载英文模型（如 LJSpeech 预训练模型）并修改配置文件。

**Q: 怎么换其他声音？**  
A: 修改 `speaker_id` 参数（0-218，AISHELL3数据集有219个说话人）：
```python
tts("你好", speaker_id=10)  # 换第11个说话人
```

---

## 📦 模型下载

| 模型 | 大小 | 说明 |
|------|------|------|
| 600000.pth.tar | 399 MB | 中文语音合成模型（必选） |
| generator_universal.pth.tar | 54 MB | 声码器（必选，已包含在项目中） |

下载链接：https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F

---

## 📝 详细文档

- [ONNX/TensorRT部署指南](deploy/TENSORRT_GUIDE.md)

---

**快速开始只需要看上面的"小白快速上手"部分！** 🎉
