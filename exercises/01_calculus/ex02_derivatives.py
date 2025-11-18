"""
练习 2: 导数与梯度
==================

学习目标：
- 掌握数值微分方法
- 理解梯度下降的基础
- 应用导数求函数极值

任务：
实现数值求导并应用于优化问题
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def numerical_derivative(f, x, h=1e-5):
    """
    使用中心差分法计算导数
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    参数:
        f: 函数
        x: 求导点
        h: 步长
    """
    # TODO: 实现中心差分法
    derivative = (f(x + h) - f(x - h)) / (2 * h)  # 请实现
    return derivative


def gradient_2d(f, x, y, h=1e-5):
    """
    计算二元函数的梯度 ∇f = (∂f/∂x, ∂f/∂y)

    参数:
        f: 二元函数 f(x, y)
        x, y: 计算点
        h: 步长
    """
    # TODO: 实现梯度计算
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)  # 对 x 的偏导数
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)  # 对 y 的偏导数
    return np.array([df_dx, df_dy])


def find_minimum_gradient_descent(f, x0, learning_rate=0.1, iterations=100):
    """
    使用梯度下降法寻找函数最小值

    参数:
        f: 目标函数
        x0: 初始点
        learning_rate: 学习率
        iterations: 迭代次数

    返回:
        最小值点的 x 坐标
    """
    # TODO: 实现梯度下降算法
    x = x0
    for i in range(iterations):
        grad = numerical_derivative(f, x)
        x = x - learning_rate * grad  # 梯度下降更新规则
    return x


def newton_method(f, x0, iterations=10):
    """
    牛顿法求根: x_{n+1} = x_n - f(x_n)/f'(x_n)

    任务：找到 f(x) = x^2 - 2 的根（即求 √2）
    """
    # TODO: 实现牛顿法
    x = x0
    for i in range(iterations):
        fx = f(x)
        fpx = numerical_derivative(f, x)
        if abs(fpx) < 1e-10:
            break
        x = x - fx / fpx  # 牛顿法更新公式
    return x


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: 计算 f(x) = x^2 在 x=3 处的导数")
    f1 = lambda x: x**2
    deriv = numerical_derivative(f1, 3.0)
    expected = 6.0  # f'(x) = 2x, at x=3: 2*3 = 6
    results.append(v.assert_close(deriv, expected, name="x^2 的导数"))

    print("\n测试 2: 计算 f(x,y) = x^2 + y^2 在 (1,2) 处的梯度")
    f2 = lambda x, y: x**2 + y**2
    grad = gradient_2d(f2, 1.0, 2.0)
    expected_grad = np.array([2.0, 4.0])  # (2x, 2y) at (1,2)
    results.append(v.assert_close(grad, expected_grad, name="梯度"))

    print("\n测试 3: 使用梯度下降找到 f(x) = (x-2)^2 的最小值")
    f3 = lambda x: (x - 2)**2
    min_x = find_minimum_gradient_descent(f3, x0=0.0, learning_rate=0.1, iterations=100)
    results.append(v.assert_close(min_x, 2.0, rtol=1e-2, name="最小值点"))

    print("\n测试 4: 使用牛顿法计算 √2")
    f4 = lambda x: x**2 - 2
    sqrt_2 = newton_method(f4, x0=1.0)
    results.append(v.assert_close(sqrt_2, np.sqrt(2), rtol=1e-6, name="√2"))

    return all(results)


if __name__ == "__main__":
    test()
