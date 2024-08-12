import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-3, 3, 100)
y = erf(x)

# 绘制误差函数
plt.plot(x, y, label="erf(x)")
plt.title("Error Function")
plt.xlabel("x")
plt.ylabel("erf(x)")
plt.grid(True)
plt.legend()
plt.show()
