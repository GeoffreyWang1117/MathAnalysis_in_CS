"""
练习 3: 积分计算
================

学习目标：
- 掌握数值积分方法
- 理解定积分的几何意义
- 应用积分解决实际问题

任务：
实现各种数值积分算法
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def riemann_sum(f, a, b, n=1000, method='midpoint'):
    """
    黎曼和近似定积分

    参数:
        f: 被积函数
        a, b: 积分区间 [a, b]
        n: 分割数
        method: 'left', 'right', 'midpoint'

    返回:
        积分近似值
    """
    # TODO: 实现黎曼和
    dx = (b - a) / n
    total = 0.0

    if method == 'left':
        for i in range(n):
            x = a + i * dx
            total += f(x) * dx
    elif method == 'right':
        for i in range(1, n + 1):
            x = a + i * dx
            total += f(x) * dx
    elif method == 'midpoint':
        for i in range(n):
            x = a + (i + 0.5) * dx
            total += f(x) * dx

    return total


def trapezoidal_rule(f, a, b, n=1000):
    """
    梯形法则积分

    公式: ∫f(x)dx ≈ (h/2)[f(x0) + 2f(x1) + 2f(x2) + ... + 2f(x_{n-1}) + f(xn)]
    其中 h = (b-a)/n
    """
    # TODO: 实现梯形法则
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # 梯形法则公式
    result = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return result


def simpson_rule(f, a, b, n=1000):
    """
    辛普森法则（抛物线法）

    要求 n 为偶数
    公式: ∫f(x)dx ≈ (h/3)[f(x0) + 4f(x1) + 2f(x2) + 4f(x3) + ... + f(xn)]
    """
    # TODO: 实现辛普森法则
    if n % 2 == 1:
        n += 1  # 确保 n 为偶数

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # 辛普森法则
    result = (h / 3) * (y[0] + y[-1] +
                        4 * np.sum(y[1:-1:2]) +
                        2 * np.sum(y[2:-1:2]))
    return result


def monte_carlo_integration(f, a, b, n=100000):
    """
    蒙特卡洛积分法

    原理：通过随机采样估计积分
    ∫f(x)dx ≈ (b-a) * mean(f(random_points))
    """
    # TODO: 实现蒙特卡洛积分
    random_points = np.random.uniform(a, b, n)
    values = f(random_points)
    result = (b - a) * np.mean(values)
    return result


def double_integral(f, x_range, y_range, nx=100, ny=100):
    """
    二重积分计算
    ∬ f(x,y) dxdy

    参数:
        f: 二元函数 f(x, y)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        nx, ny: x和y方向的分割数
    """
    # TODO: 实现二重积分
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # 使用黎曼和
    result = np.sum(Z) * dx * dy
    return result


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 测试函数: ∫(0 to π) sin(x)dx = 2
    f1 = np.sin
    a, b = 0, np.pi
    exact = 2.0

    print("\n测试 1: 黎曼和计算 ∫sin(x)dx from 0 to π")
    result1 = riemann_sum(f1, a, b, n=1000, method='midpoint')
    results.append(v.assert_close(result1, exact, rtol=1e-3, name="黎曼和"))

    print("\n测试 2: 梯形法则计算 ∫sin(x)dx from 0 to π")
    result2 = trapezoidal_rule(f1, a, b, n=1000)
    results.append(v.assert_close(result2, exact, rtol=1e-5, name="梯形法则"))

    print("\n测试 3: 辛普森法则计算 ∫sin(x)dx from 0 to π")
    result3 = simpson_rule(f1, a, b, n=1000)
    results.append(v.assert_close(result3, exact, rtol=1e-8, name="辛普森法则"))

    print("\n测试 4: 蒙特卡洛积分")
    result4 = monte_carlo_integration(f1, a, b, n=100000)
    results.append(v.assert_close(result4, exact, rtol=0.01, name="蒙特卡洛"))

    print("\n测试 5: 二重积分 ∬(x+y)dxdy, x∈[0,1], y∈[0,1]")
    f2 = lambda x, y: x + y
    result5 = double_integral(f2, (0, 1), (0, 1), nx=100, ny=100)
    exact5 = 1.0  # ∫∫(x+y)dxdy = 1
    results.append(v.assert_close(result5, exact5, rtol=1e-2, name="二重积分"))

    return all(results)


if __name__ == "__main__":
    test()
