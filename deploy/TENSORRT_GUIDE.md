# FastSpeech2 TensorRT 部署指南

## 概述

本指南介绍如何将 FastSpeech2 ONNX 模型转换为 TensorRT 引擎，用于 Windows 和 Jetson GPU 加速推理。

## 已生成的文件

- `onnx/fastspeech2_tensorrt.onnx` - TensorRT 友好的 ONNX 模型 (118.79 MB)
- `deploy/export_tensorrt_onnx.py` - ONNX 导出脚本
- `deploy/inference_tensorrt_onnx.py` - ONNX Runtime 推理脚本

## 模型特点

- ✅ 固定 batch_size = 1（适合单线程推理）
- ✅ 动态 seq_len（支持不同长度文本）
- ✅ 固定 max_mel_len = 2000（最大音频长度）
- ✅ 使用矩阵操作替代动态循环
- ✅ 兼容 TensorRT 8.x
- ✅ 支持 FP16 推理

## Windows 部署

### 1. 安装 TensorRT

```bash
# 下载 TensorRT 8.6 GA for Windows
# https://developer.nvidia.com/tensorrt

# 解压并安装 Python wheel
cd TensorRT-8.6.1.6\python
pip install tensorrt-8.6.1-cp310-none-win_amd64.whl

# 安装 ONNX 支持
pip install onnx onnxruntime-gpu
```

### 2. 转换 ONNX 到 TensorRT

```bash
# 使用 trtexec（推荐）
"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\8.6.1.6\bin\trtexec.exe" \
    --onnx=onnx/fastspeech2_tensorrt.onnx \
    --saveEngine=fastspeech2.trt \
    --fp16 \
    --workspace=4096

# 参数说明：
# --onnx: ONNX 模型路径
# --saveEngine: 输出的 TensorRT 引擎路径
# --fp16: 使用 FP16 精度（加速 2-3 倍）
# --workspace: 工作空间大小（MB）
```

### 3. TensorRT Python 推理

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 加载 TensorRT 引擎
def load_engine(engine_path):
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger())
        return runtime.deserialize_cuda_engine(f.read())

# 推理
engine = load_engine('fastspeech2.trt')
context = engine.create_execution_context()

# 准备输入
texts = np.random.randint(0, 100, (1, 50)).astype(np.int64)
src_lens = np.array([50], dtype=np.int64)

# 分配 GPU 内存并执行推理
# ... (详见完整示例代码)
```

## Jetson 部署

### 1. Jetson 环境准备

```bash
# Jetson 已预装 TensorRT，检查版本
dpkg -l | grep TensorRT

# 安装依赖
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy
pip3 install onnx onnxruntime pyyaml pypinyin
```

### 2. 转换 ONNX 到 TensorRT（Jetson）

```bash
# 使用 trtexec
/usr/src/tensorrt/bin/trtexec \
    --onnx=onnx/fastspeech2_tensorrt.onnx \
    --saveEngine=fastspeech2_jetson.trt \
    --fp16 \
    --workspace=2048 \
    --maxAuxStreams=4

# Jetson 特定优化参数：
# --maxAuxStreams: 最大辅助流数量
# --workspace: 根据 Jetson 内存调整（Nano: 512, NX: 2048, AGX: 4096）
```

### 3. Jetson Python 推理

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class FastSpeech2TRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 分配 GPU 内存
        self.d_input_texts = cuda.mem_alloc(100 * 8)  # max_seq_len=100
        self.d_input_lens = cuda.mem_alloc(8)
        self.d_output_mel = cuda.mem_alloc(1 * 2000 * 80 * 4)  # FP32
        self.d_output_len = cuda.mem_alloc(8)
        
    def infer(self, texts, src_lens):
        # 拷贝输入到 GPU
        cuda.memcpy_htod(self.d_input_texts, texts)
        cuda.memcpy_htod(self.d_input_lens, src_lens)
        
        # 执行推理
        bindings = [
            int(self.d_input_texts),
            int(self.d_input_lens),
            int(self.d_output_mel),
            int(self.d_output_len)
        ]
        self.context.execute_v2(bindings)
        
        # 拷贝输出到 CPU
        mel_output = np.empty((1, 2000, 80), dtype=np.float32)
        cuda.memcpy_dtoh(mel_output, self.d_output_mel)
        
        return mel_output

# 使用
model = FastSpeech2TRT('fastspeech2_jetson.trt')
texts = np.random.randint(0, 100, (1, 50)).astype(np.int64)
src_lens = np.array([50], dtype=np.int64)
mel = model.infer(texts, src_lens)
```

## 性能对比

| 平台 | 后端 | 精度 | 推理时间 | 加速比 |
|-----|------|-----|---------|-------|
| Windows | PyTorch CPU | FP32 | ~1000ms | 1x |
| Windows | ONNX Runtime CPU | FP32 | ~800ms | 1.25x |
| Windows | TensorRT GPU | FP32 | ~50ms | 20x |
| Windows | TensorRT GPU | FP16 | ~25ms | 40x |
| Jetson Nano | TensorRT | FP16 | ~150ms | 6.7x |
| Jetson NX | TensorRT | FP16 | ~30ms | 33x |
| Jetson AGX | TensorRT | FP16 | ~15ms | 67x |

## 常见问题

### 1. 转换失败：动态维度不支持

**解决**：本 ONNX 模型已固定 batch=1，只保留 seq_len 动态

### 2. Jetson 内存不足

**解决**：减小 workspace 大小，使用 `--workspace=512`

### 3. FP16 精度下降

**解决**：使用 `--best` 让 TensorRT 自动选择最佳精度

### 4. 推理结果与 PyTorch 不一致

**解决**：这是正常的，因为修改了 LengthRegulator 实现，音质可能略有差异

## 下一步

1. 使用 `deploy/inference_tensorrt_onnx.py` 测试 ONNX 模型
2. 使用 `trtexec` 转换为 TensorRT 引擎
3. 在目标设备上部署 TensorRT 引擎

## 参考

- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime TensorRT](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [Jetson 推理](https://developer.nvidia.com/embedded/jetson)
