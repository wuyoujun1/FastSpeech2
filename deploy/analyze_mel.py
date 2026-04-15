"""
分析梅尔频谱，找出有效语音部分
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_mel(mel_path, threshold=0.01):
    """
    分析梅尔频谱，找出有效语音部分
    
    Args:
        mel_path: 梅尔频谱文件路径
        threshold: 能量阈值，低于此值认为是静音
    """
    print(f"分析梅尔频谱: {mel_path}")
    mel = np.load(mel_path)
    print(f"  总形状: {mel.shape}")
    print(f"  总帧数: {mel.shape[0]}")
    
    # 计算每帧的能量
    frame_energy = np.abs(mel).sum(axis=1)
    print(f"\n  能量统计:")
    print(f"    最小值: {frame_energy.min():.4f}")
    print(f"    最大值: {frame_energy.max():.4f}")
    print(f"    平均值: {frame_energy.mean():.4f}")
    print(f"    中位数: {np.median(frame_energy):.4f}")
    
    # 找到有效帧（能量高于阈值）
    valid_frames = np.where(frame_energy > threshold)[0]
    
    if len(valid_frames) == 0:
        print(f"\n  ⚠️ 没有找到有效帧！")
        return None
    
    start_frame = valid_frames[0]
    end_frame = valid_frames[-1] + 1
    valid_length = end_frame - start_frame
    
    print(f"\n  有效语音部分:")
    print(f"    起始帧: {start_frame}")
    print(f"    结束帧: {end_frame}")
    print(f"    有效帧数: {valid_length}")
    print(f"    占总帧数比例: {valid_length / len(frame_energy) * 100:.1f}%")
    
    # 计算时长（假设采样率 22050Hz，hop_length=256）
    hop_length = 256
    sample_rate = 22050
    duration = valid_length * hop_length / sample_rate
    print(f"    估算时长: {duration:.2f} 秒")
    
    # 绘制能量图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(frame_energy)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值={threshold}')
    plt.axvline(x=start_frame, color='g', linestyle='--', label=f'起始帧={start_frame}')
    plt.axvline(x=end_frame, color='g', linestyle='--', label=f'结束帧={end_frame}')
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.title('Frame Energy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.imshow(mel.T, aspect='auto', origin='lower', interpolation='nearest')
    plt.axvline(x=start_frame, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=end_frame, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Mel Channel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plot_path = mel_path.replace('.npy', '_analysis.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  分析图已保存: {plot_path}")
    
    return {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'valid_length': valid_length,
        'duration': duration,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析梅尔频谱")
    parser.add_argument("mel", type=str, help="梅尔频谱文件路径 (.npy)")
    parser.add_argument("--threshold", type=float, default=0.01, help="能量阈值")
    
    args = parser.parse_args()
    
    result = analyze_mel(args.mel, args.threshold)
    
    if result:
        print(f"\n建议截取: [{result['start_frame']}:{result['end_frame']}]")
