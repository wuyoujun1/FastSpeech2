#!/bin/bash
# FastSpeech2 项目最小化脚本
# 只保留运行 synthesize_tensorrt_fixed.py 所需的文件

echo "=========================================="
echo "  FastSpeech2 项目最小化"
echo "=========================================="

# 创建备份目录
mkdir -p backup

echo ""
echo "📦 备份重要文件..."

# 备份主程序
cp synthesize_tensorrt_fixed.py backup/ 2>/dev/null

# 备份配置文件
mkdir -p backup/config/AISHELL3
cp config/AISHELL3/preprocess.yaml backup/config/AISHELL3/
cp config/AISHELL3/model.yaml backup/config/AISHELL3/

# 备份 Python 模块
mkdir -p backup/text backup/utils backup/model backup/hifigan backup/audio backup/transformer backup/lexicon

cp text/__init__.py text/symbols.py text/cleaners.py text/cmudict.py text/pinyin.py text/numbers.py backup/text/
cp utils/__init__.py utils/model.py utils/tools.py backup/utils/
cp model/__init__.py model/fastspeech2.py model/modules.py model/loss.py backup/model/
cp hifigan/__init__.py hifigan/models.py backup/hifigan/
cp audio/__init__.py audio/audio_processing.py audio/stft.py audio/tools.py backup/audio/
cp transformer/__init__.py transformer/Constants.py transformer/Modules.py transformer/SubLayers.py transformer/Layers.py transformer/Models.py backup/transformer/

# 备份词典
cp lexicon/pinyin-lexicon-r.txt backup/lexicon/

# 备份声码器权重（如果有）
if [ -f "hifigan/generator_LJSpeech.pth.tar" ]; then
    cp hifigan/generator_LJSpeech.pth.tar backup/hifigan/
fi
if [ -f "hifigan/generator_universal.pth.tar" ]; then
    cp hifigan/generator_universal.pth.tar backup/hifigan/
fi

# 备份引擎文件
if [ -f "fastspeech2_py.engine" ]; then
    cp fastspeech2_py.engine backup/
fi

echo "✅ 备份完成"
echo ""

# 删除不需要的文件
echo "🗑️  删除不需要的文件..."

# 删除 API 相关文件
rm -f fastapi_tensorrt_tts.py tts_api.py tts_server.py tts_debug.py tts_simple.py

# 删除导出/转换脚本
rm -f export_*.py convert_*.py build_*.py onnx_to_tensorrt.py

# 删除测试脚本
rm -f test_*.py

# 删除训练/评估脚本
rm -f train.py evaluate.py preprocess.py prepare_align.py dataset.py

# 删除其他合成脚本
rm -f synthesize.py synthesize_tensorrt.py synthesize_trtexec.py synthesize_tensorrt_long.py

# 删除应用脚本
rm -f app.py

# 删除预处理模块
rm -rf preprocessor/

# 删除演示文件
rm -rf demo/

# 删除 ONNX 目录
rm -rf onnx/

# 删除输出目录
rm -rf output/ pytorch_output/ temp_output/

# 删除缓存
rm -rf __pycache__ */__pycache__ */*/__pycache__
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# 删除文档和脚本
rm -f *.md *.sh *.txt 2>/dev/null
rm -f generate_charts.py

# 删除音频文件
rm -f *.wav

echo "✅ 清理完成"
echo ""

# 从备份恢复必需文件
echo "📂 恢复必需文件..."

cp backup/synthesize_tensorrt_fixed.py .
cp -r backup/config .
cp -r backup/text .
cp -r backup/utils .
cp -r backup/model .
cp -r backup/hifigan .
cp -r backup/audio .
cp -r backup/transformer .
cp -r backup/lexicon .
if [ -f "backup/fastspeech2_py.engine" ]; then
    cp backup/fastspeech2_py.engine .
fi

echo "✅ 恢复完成"
echo ""

# 显示最终结构
echo "📊 最终项目结构:"
echo "=========================================="
find . -type f -name "*.py" -o -name "*.yaml" -o -name "*.txt" -o -name "*.engine" -o -name "*.pth.tar" 2>/dev/null | grep -v backup | sort
echo "=========================================="

# 统计文件数
echo ""
FILE_COUNT=$(find . -type f 2>/dev/null | grep -v backup | wc -l)
echo "📈 剩余文件数: $FILE_COUNT"
echo ""
echo "✨ 最小化完成！"
echo "   备份保存在: backup/"
echo "   如需恢复原始项目，请从备份复制文件"
