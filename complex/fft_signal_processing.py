import numpy as np
import matplotlib.pyplot as plt

# 生成包含两个频率成分的信号
sampling_rate = 1000
t = np.linspace(0, 1, sampling_rate)
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# 执行FFT
fft_signal = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(fft_signal), 1/sampling_rate)

# 绘制频谱
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_signal)[:len(frequencies)//2])
plt.title("频率谱")
plt.xlabel("频率 (Hz)")
plt.ylabel("幅值")
plt.show()
