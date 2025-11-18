"""
工具函数集合
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
import seaborn as sns


# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_function(func: Callable, x_range: Tuple[float, float],
                  title: str = "函数图像", xlabel: str = "x",
                  ylabel: str = "y", num_points: int = 1000):
    """绘制一维函数"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2)
    plt.title(title, fontsize=14, fontproperties='SimHei')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_2d_function(func: Callable, x_range: Tuple[float, float],
                     y_range: Tuple[float, float], title: str = "3D函数图像",
                     num_points: int = 50):
    """绘制二维函数的3D图像"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title(title, fontsize=14, fontproperties='SimHei')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    fig.colorbar(surf)
    plt.tight_layout()
    return fig


def plot_vector_field(func_x: Callable, func_y: Callable,
                      x_range: Tuple[float, float],
                      y_range: Tuple[float, float],
                      title: str = "向量场", num_points: int = 20):
    """绘制向量场"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    U = func_x(X, Y)
    V = func_y(X, Y)

    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V, alpha=0.8)
    plt.title(title, fontsize=14, fontproperties='SimHei')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def numerical_derivative(func: Callable, x: float, h: float = 1e-5) -> float:
    """数值求导（中心差分）"""
    return (func(x + h) - func(x - h)) / (2 * h)


def numerical_integral(func: Callable, a: float, b: float,
                       n: int = 1000) -> float:
    """数值积分（梯形法则）"""
    x = np.linspace(a, b, n)
    y = func(x)
    return np.trapz(y, x)


def generate_random_matrix(n: int, m: int = None,
                          symmetric: bool = False,
                          positive_definite: bool = False) -> np.ndarray:
    """生成随机矩阵"""
    if m is None:
        m = n

    if positive_definite:
        # 生成正定矩阵
        A = np.random.randn(n, n)
        return A.T @ A + np.eye(n)
    elif symmetric:
        # 生成对称矩阵
        A = np.random.randn(n, n)
        return (A + A.T) / 2
    else:
        return np.random.randn(n, m)


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_result(name: str, value):
    """打印结果"""
    print(f"{name}: {value}")


def format_matrix(matrix: np.ndarray) -> str:
    """格式化打印矩阵"""
    return np.array2string(matrix, precision=4, suppress_small=True)


# 常用数学常数
PI = np.pi
E = np.e
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


# 常用函数
def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU函数"""
    return np.maximum(0, x)


def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def gaussian(x, mu=0, sigma=1):
    """高斯函数"""
    return (1 / (sigma * np.sqrt(2 * PI))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
