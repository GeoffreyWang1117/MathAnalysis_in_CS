import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 生成稀疏数据点
x = np.linspace(0, 10, 10)
y = np.sin(x)

# 线性插值
f_linear = interp1d(x, y)

# 拟合更细的数据点
x_new = np.linspace(0, 10, 100)
y_linear = f_linear(x_new)

# 绘制原始数据和插值结果
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_new, y_linear, '-', label='Linear Interpolation')
plt.title("Linear Interpolation Example")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
