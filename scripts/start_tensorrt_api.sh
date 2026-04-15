#!/bin/bash
# TensorRT TTS FastAPI 服务启动脚本 (Jetson)

echo "=========================================="
echo "  TensorRT TTS FastAPI 服务"
echo "=========================================="

# 检查是否在 Jetson 上
if [ -f /etc/nv_tegra_release ]; then
    echo "✅ 检测到 Jetson 设备"
    cat /etc/nv_tegra_release
else
    echo "⚠️  未检测到 Jetson 设备"
fi

# 检查 TensorRT
python3 -c "import tensorrt; print(f'TensorRT 版本: {tensorrt.__version__}')" 2>/dev/null || {
    echo "❌ TensorRT 未安装"
    exit 1
}

# 检查引擎文件
ENGINE_FILE="fastspeech2_py.engine"
if [ ! -f "$ENGINE_FILE" ]; then
    echo "❌ 引擎文件不存在: $ENGINE_FILE"
    echo "请先运行转换脚本生成引擎文件"
    exit 1
fi

echo "✅ 引擎文件: $ENGINE_FILE"

# 启动服务
echo ""
echo "🚀 启动 FastAPI 服务..."
echo "   访问: http://$(hostname -I | awk '{print $1}'):8000"
echo "   文档: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "=========================================="
echo ""

python3 fastapi_tensorrt_tts.py
