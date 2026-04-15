#!/bin/bash
# Jetson设备上ONNX转TensorRT快速部署脚本
# 使用方法: bash jetson_deploy.sh

set -e

echo "========================================"
echo "Jetson TensorRT 部署脚本"
echo "========================================"

# 检查环境
echo ""
echo "1. 检查环境..."
echo "JetPack版本:"
cat /etc/nv_tegra_release 2>/dev/null || echo "无法获取JetPack版本"

echo ""
echo "TensorRT版本:"
python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "TensorRT未安装"

echo ""
echo "CUDA版本:"
nvcc --version 2>/dev/null || echo "CUDA未安装"

# 创建目录
echo ""
echo "2. 创建项目目录..."
mkdir -p ~/FastSpeech2-master/onnx
mkdir -p ~/FastSpeech2-master/config/AISHELL3
mkdir -p ~/FastSpeech2-master/preprocessed_data/AISHELL3
mkdir -p ~/FastSpeech2-master/tensorrt_output

# 安装依赖
echo ""
echo "3. 安装Python依赖..."
python3 -m pip install --user numpy onnx onnxruntime pycuda pypinyin pyyaml soundfile 2>/dev/null || {
    echo "警告: 部分依赖安装失败，请手动安装"
}

# 检查ONNX文件
echo ""
echo "4. 检查ONNX模型..."
if [ -f "onnx/fastspeech2.onnx" ]; then
    echo "ONNX模型存在: onnx/fastspeech2.onnx"
    ls -lh onnx/fastspeech2.onnx
else
    echo "错误: ONNX模型不存在!"
    echo "请先从本地电脑传输ONNX模型到Jetson"
    echo "命令: scp onnx/fastspeech2.onnx user@jetson-ip:~/FastSpeech2-master/onnx/"
    exit 1
fi

# 转换模型
echo ""
echo "5. 转换ONNX为TensorRT引擎..."
echo "这可能需要几分钟，请耐心等待..."

python3 onnx_to_tensorrt.py \
    --onnx onnx/fastspeech2.onnx \
    --output fastspeech2_fp16.engine \
    --fp16 \
    --workspace 2

# 检查转换结果
echo ""
echo "6. 检查转换结果..."
if [ -f "fastspeech2_fp16.engine" ]; then
    echo "TensorRT引擎生成成功!"
    ls -lh fastspeech2_fp16.engine
else
    echo "错误: TensorRT引擎生成失败!"
    exit 1
fi

# 测试推理
echo ""
echo "7. 测试TensorRT推理..."
python3 test_tensorrt_inference.py \
    --engine fastspeech2_fp16.engine \
    --text "测试TensorRT推理" || {
    echo "警告: 推理测试失败，请检查配置文件"
}

echo ""
echo "========================================"
echo "部署完成!"
echo "========================================"
echo ""
echo "使用方法:"
echo "  python3 test_tensorrt_inference.py --engine fastspeech2_fp16.engine --text '你的文本'"
echo ""
