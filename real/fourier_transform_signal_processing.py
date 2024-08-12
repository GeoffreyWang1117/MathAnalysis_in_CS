import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# 生成一个复合信号，包含两个频率成分
sampling_rate = 1000
t = np.linspace(0, 1, sampling_rate)
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# 计算傅里叶变换
fft_signal = fft(signal)

# 频率轴
frequencies = np.fft.fftfreq(len(fft_signal), 1/sampling_rate)

# 绘制信号和频谱
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_signal)[:len(frequencies)//2])
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
