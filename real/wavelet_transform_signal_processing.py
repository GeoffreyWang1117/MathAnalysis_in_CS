import pywt
import matplotlib.pyplot as plt
import numpy as np

# 生成一个信号
t = np.linspace(0, 1, 400)
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# 小波变换
coeffs = pywt.wavedec(signal, 'db1', level=4)

# 绘制小波系数
plt.figure(figsize=(10, 8))
for i, coef in enumerate(coeffs):
    plt.subplot(len(coeffs), 1, i + 1)
    plt.plot(coef)
    plt.title(f"Wavelet Coefficients Level {i + 1}")
plt.tight_layout()
plt.show()
