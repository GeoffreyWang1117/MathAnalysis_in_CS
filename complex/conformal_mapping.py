import numpy as np
import matplotlib.pyplot as plt

def conformal_map(z):
    return z**2

# 在复平面上创建网格
re = np.linspace(-2, 2, 400)
im = np.linspace(-2, 2, 400)
re, im = np.meshgrid(re, im)
z = re + 1j * im

# 应用保角映射
w = conformal_map(z)

# 绘制原始网格
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.contour(re, im, np.abs(z), colors='black', linestyles='solid')
plt.title("原始网格")

# 绘制映射后的网格
plt.subplot(1, 2, 2)
plt.contour(w.real, w.imag, np.abs(w), colors='black', linestyles='solid')
plt.title("映射后的网格")

plt.show()
