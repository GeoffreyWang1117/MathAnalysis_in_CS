import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# 图像尺寸（像素）
width, height = 800, 800

# 绘图窗口
re_min, re_max = -2.0, 1.0
im_min, im_max = -1.5, 1.5

# 准备图像数组
image = np.zeros((height, width))

for x in range(width):
    for y in range(height):
        # 将像素坐标转换为复数
        c = complex(re_min + (x / width) * (re_max - re_min),
                    im_min + (y / height) * (im_max - im_min))
        # 计算颜色值
        image[y, x] = mandelbrot(c, 256)

# 显示曼德布罗集
plt.imshow(image, extent=[re_min, re_max, im_min, im_max], cmap='hot')
plt.colorbar()
plt.title("曼德布罗集")
plt.show()
