#!/bin/bash
# FastSpeech2 项目最小化脚本 - 保留 PyTorch 和 TensorRT 两种推理方式

echo "=========================================="
echo "  FastSpeech2 项目最小化"
echo "  保留: PyTorch + TensorRT 双推理"
echo "=========================================="

# 创建备份目录
mkdir -p backup

echo ""
echo "📦 备份重要文件..."

# 备份主程序
cp synthesize.py backup/ 2>/dev/null
cp synthesize_tensorrt_fixed.py backup/ 2>/dev/null

# 备份配置文件
mkdir -p backup/config/AISHELL3
cp config/AISHELL3/preprocess.yaml backup/config/AISHELL3/
cp config/AISHELL3/model.yaml backup/config/AISHELL3/
cp config/AISHELL3/train.yaml backup/config/AISHELL3/

# 备份 Python 模块目录
for dir in text utils model hifigan audio transformer lexicon; do
    if [ -d "$dir" ]; then
        cp -r $dir backup/
        echo "  ✓ $dir/"
    fi
done

# 备份 dataset.py（PyTorch推理需要）
cp dataset.py backup/ 2>/dev/null

# 备份模型权重（如果有）
if [ -d "output/ckpt/AISHELL3" ]; then
    mkdir -p backup/output/ckpt/AISHELL3
    cp output/ckpt/AISHELL3/*.pth.tar backup/output/ckpt/AISHELL3/ 2>/dev/null
    echo "  ✓ 模型权重"
fi

# 备份引擎文件
if [ -f "fastspeech2_py.engine" ]; then
    cp fastspeech2_py.engine backup/
    echo "  ✓ TensorRT引擎"
fi

echo "✅ 备份完成"
echo ""

# 删除不需要的文件
echo "🗑️  删除不需要的文件..."

# 删除 API 相关文件
rm -f fastapi_*.py tts_*.py 2>/dev/null

# 删除导出/转换脚本
rm -f export_*.py convert_*.py build_*.py onnx_to_tensorrt.py 2>/dev/null

# 删除测试脚本
rm -f test_*.py 2>/dev/null

# 删除训练/评估脚本（保留 synthesize.py）
rm -f train.py evaluate.py preprocess.py prepare_align.py 2>/dev/null

# 删除其他合成脚本（保留 synthesize.py 和 synthesize_tensorrt_fixed.py）
rm -f synthesize_trtexec.py synthesize_tensorrt_long.py 2>/dev/null

# 删除应用脚本
rm -f app.py 2>/dev/null

# 删除预处理模块
rm -rf preprocessor/ 2>/dev/null

# 删除演示文件
rm -rf demo/ 2>/dev/null

# 删除 ONNX 目录
rm -rf onnx/ 2>/dev/null

# 删除输出目录（保留模型权重）
rm -rf output/result/ temp_output/ pytorch_output/ 2>/dev/null

# 删除缓存
rm -rf __pycache__ */__pycache__ */*/__pycache__ 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

# 删除文档和脚本
rm -f *.md *.sh *.txt 2>/dev/null
rm -f generate_charts.py 2>/dev/null

# 删除音频文件
rm -f *.wav 2>/dev/null

echo "✅ 清理完成"
echo ""

# 从备份恢复必需文件
echo "📂 恢复必需文件..."

cp backup/synthesize.py . 2>/dev/null
cp backup/synthesize_tensorrt_fixed.py . 2>/dev/null
cp -r backup/config . 2>/dev/null
cp -r backup/text . 2>/dev/null
cp -r backup/utils . 2>/dev/null
cp -r backup/model . 2>/dev/null
cp -r backup/hifigan . 2>/dev/null
cp -r backup/audio . 2>/dev/null
cp -r backup/transformer . 2>/dev/null
cp -r backup/lexicon . 2>/dev/null
cp backup/dataset.py . 2>/dev/null

# 恢复模型权重
if [ -d "backup/output/ckpt/AISHELL3" ]; then
    mkdir -p output/ckpt/AISHELL3
    cp backup/output/ckpt/AISHELL3/*.pth.tar output/ckpt/AISHELL3/ 2>/dev/null
fi

# 恢复引擎文件
cp backup/fastspeech2_py.engine . 2>/dev/null

echo "✅ 恢复完成"
echo ""

# 显示最终结构
echo "📊 最终项目结构:"
echo "=========================================="
echo "主程序:"
ls -1 *.py 2>/dev/null | grep -E "(synthesize|minimize)"
echo ""
echo "配置文件:"
find config -type f 2>/dev/null
echo ""
echo "Python模块:"
for dir in text utils model hifigan audio transformer lexicon; do
    if [ -d "$dir" ]; then
        echo "  $dir/ ($(ls $dir/*.py 2>/dev/null | wc -l) 个文件)"
    fi
done
echo ""
echo "模型文件:"
ls -lh output/ckpt/AISHELL3/*.pth.tar 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh fastspeech2_py.engine 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo "=========================================="

# 统计文件数
echo ""
FILE_COUNT=$(find . -type f 2>/dev/null | grep -v backup | wc -l)
echo "📈 剩余文件数: $FILE_COUNT"
echo ""
echo "✨ 最小化完成！"
echo ""
echo "📝 使用方式:"
echo "  1. PyTorch推理:"
echo "     export LD_LIBRARY_PATH=/opt/conda/miniforge3/envs/det/lib:\$LD_LIBRARY_PATH"
echo "     python3 synthesize.py --text '你好' --speaker_id 0 --restore_step 600000 ..."
echo ""
echo "  2. TensorRT推理:"
echo "     python3 synthesize_tensorrt_fixed.py --text '你好' --speaker_id 0 ..."
echo ""
echo "  备份保存在: backup/"
