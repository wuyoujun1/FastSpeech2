#!/bin/bash
# 创建 TTS 守护进程脚本

cat > tts_daemon.py << 'PYEOF'
#!/usr/bin/env python3
"""
TTS 守护进程 - 本地常驻服务
模型只加载一次，通过命令行交互使用
"""
import os
import sys
import yaml
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import wave

from synthesize_tensorrt_fixed import (
    TensorRTInference,
    preprocess_mandarin,
    synthesize_segment,
    device
)
from utils.model import get_vocoder

# 配置
PREPROCESS_CONFIG = "config/AISHELL3/preprocess.yaml"
MODEL_CONFIG = "config/AISHELL3/model.yaml"
ENGINE_PATH = "fastspeech2_py.engine"
MAX_SEQ_LEN = 50

class TTSDaemon:
    def __init__(self):
        self.trt = None
        self.vocoder = None
        self.config = None
        self.sample_rate = 22050
        self.initialized = False

    def init(self):
        """初始化模型（只执行一次）"""
        print("=" * 60)
        print("初始化 TTS 守护进程...")
        print("=" * 60)

        # 加载配置
        print("[1/4] 加载配置...")
        with open(PREPROCESS_CONFIG, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(MODEL_CONFIG, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sample_rate = self.config["preprocessing"]["audio"]["sampling_rate"]

        # 加载声码器
        print("[2/4] 加载声码器...")
        self.vocoder = get_vocoder(model_config, device)

        # 初始化 TensorRT
        print("[3/4] 初始化 TensorRT...")
        self.trt = TensorRTInference(ENGINE_PATH, self.config)

        self.initialized = True
        print("[4/4] 初始化完成！")
        print(f"   引擎: {ENGINE_PATH}")
        print(f"   设备: {device}")
        print(f"   采样率: {self.sample_rate} Hz")
        print("=" * 60)
        print()

    def synthesize(self, text, speaker_id=0, output_file="output.wav"):
        """合成音频"""
        if not self.initialized:
            return False, "服务未初始化"

        try:
            start_time = time.time()

            # 文本预处理
            sequence = preprocess_mandarin(text, self.config)
            if len(sequence) == 0:
                return False, "音素序列为空"

            # 分段合成
            if len(sequence) > MAX_SEQ_LEN:
                wavs = []
                for i in range(0, len(sequence), MAX_SEQ_LEN):
                    segment = sequence[i:i+MAX_SEQ_LEN]
                    original_len = len(segment)
                    wav = synthesize_segment(self.trt, segment, speaker_id, MAX_SEQ_LEN, self.vocoder, original_len)
                    wavs.append(wav)
                wav = np.concatenate(wavs)
            else:
                original_len = len(sequence)
                wav = synthesize_segment(self.trt, sequence, speaker_id, MAX_SEQ_LEN, self.vocoder, original_len)

            # 保存音频
            wav = wav.astype(np.float32)
            max_val = np.max(np.abs(wav))
            if max_val > 1.0:
                wav = wav / max_val
            wav_int16 = (wav * 32767).astype(np.int16)

            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(wav_int16.tobytes())

            inference_time = (time.time() - start_time) * 1000
            duration = len(wav) / self.sample_rate

            return True, {
                "output_file": output_file,
                "inference_time_ms": inference_time,
                "audio_duration_sec": duration,
                "rtf": inference_time / 1000 / duration if duration > 0 else 0
            }

        except Exception as e:
            return False, str(e)

    def run_interactive(self):
        """交互式运行"""
        print("TTS 守护进程已启动")
        print("   输入格式: <文本> [说话人ID] [输出文件名]")
        print("   示例: 你好世界 0 output.wav")
        print("   输入 'quit' 或 'exit' 退出")
        print("-" * 60)

        while True:
            try:
                # 读取输入
                user_input = input("\nTTS> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break

                # 解析输入
                parts = user_input.split()
                text = parts[0]
                speaker_id = int(parts[1]) if len(parts) > 1 else 0
                output_file = parts[2] if len(parts) > 2 else "output.wav"

                # 合成
                print(f"合成: '{text}' (说话人: {speaker_id})")
                success, result = self.synthesize(text, speaker_id, output_file)

                if success:
                    print(f"完成!")
                    print(f"   文件: {result['output_file']}")
                    print(f"   耗时: {result['inference_time_ms']:.1f} ms")
                    print(f"   时长: {result['audio_duration_sec']:.2f} s")
                    print(f"   RTF: {result['rtf']:.3f}")
                    print(f"   播放: aplay {result['output_file']}")
                else:
                    print(f"错误: {result}")

            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"错误: {e}")

    def run_once(self, text, speaker_id=0, output_file="output.wav"):
        """单次运行模式"""
        success, result = self.synthesize(text, speaker_id, output_file)
        if success:
            print(f"已保存: {result['output_file']}")
            return result
        else:
            print(f"错误: {result}")
            return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TTS 守护进程")
    parser.add_argument("--text", type=str, help="要合成的文本（如果不提供则进入交互模式）")
    parser.add_argument("--speaker_id", type=int, default=0, help="说话人ID")
    parser.add_argument("--output", type=str, default="output.wav", help="输出文件")
    args = parser.parse_args()

    # 创建守护进程
    daemon = TTSDaemon()
    daemon.init()

    if args.text:
        # 单次模式
        daemon.run_once(args.text, args.speaker_id, args.output)
    else:
        # 交互模式
        daemon.run_interactive()


if __name__ == "__main__":
    main()
PYEOF

chmod +x tts_daemon.py
echo "✅ tts_daemon.py 创建完成"
