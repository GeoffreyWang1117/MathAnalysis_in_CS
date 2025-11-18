"""
练习 4: 级数与泰勒展开
======================

学习目标：
- 理解无穷级数的收敛性
- 掌握泰勒级数展开
- 应用级数近似函数

任务：
实现级数计算和函数近似
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def geometric_series(r, n=100):
    """
    几何级数求和: S = 1 + r + r^2 + r^3 + ... + r^n

    当 |r| < 1 时，无穷级数收敛到 1/(1-r)
    """
    # TODO: 实现几何级数求和
    if abs(r) >= 1:
        raise ValueError("几何级数发散，|r| 必须 < 1")

    result = sum(r**i for i in range(n + 1))
    return result


def harmonic_series(n=1000):
    """
    调和级数: H_n = 1 + 1/2 + 1/3 + ... + 1/n

    注意：调和级数发散，但增长缓慢
    H_n ≈ ln(n) + γ (欧拉常数 γ ≈ 0.5772)
    """
    # TODO: 实现调和级数
    result = sum(1/i for i in range(1, n + 1))
    return result


def taylor_exp(x, n=10):
    """
    e^x 的泰勒展开:
    e^x = 1 + x + x^2/2! + x^3/3! + ... + x^n/n!

    参数:
        x: 计算点
        n: 展开项数
    """
    # TODO: 实现 e^x 的泰勒展开
    result = 0.0
    for i in range(n + 1):
        result += x**i / np.math.factorial(i)
    return result


def taylor_sin(x, n=10):
    """
    sin(x) 的泰勒展开:
    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

    参数:
        x: 计算点（弧度）
        n: 展开项数
    """
    # TODO: 实现 sin(x) 的泰勒展开
    result = 0.0
    for i in range(n):
        sign = (-1) ** i
        term = sign * x**(2*i + 1) / np.math.factorial(2*i + 1)
        result += term
    return result


def taylor_cos(x, n=10):
    """
    cos(x) 的泰勒展开:
    cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    """
    # TODO: 实现 cos(x) 的泰勒展开
    result = 0.0
    for i in range(n):
        sign = (-1) ** i
        term = sign * x**(2*i) / np.math.factorial(2*i)
        result += term
    return result


def leibniz_pi(n=100000):
    """
    莱布尼茨公式计算 π:
    π/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...

    返回 π 的近似值
    """
    # TODO: 实现莱布尼茨公式
    result = 0.0
    for i in range(n):
        sign = (-1) ** i
        result += sign / (2 * i + 1)
    return result * 4


def fourier_square_wave(x, n=10):
    """
    方波的傅里叶级数展开:
    f(x) = (4/π) * [sin(x) + sin(3x)/3 + sin(5x)/5 + ...]

    参数:
        x: 计算点
        n: 展开项数（奇数谐波数量）
    """
    # TODO: 实现方波的傅里叶展开
    result = 0.0
    for i in range(1, n + 1):
        k = 2 * i - 1  # 奇数项: 1, 3, 5, 7, ...
        result += np.sin(k * x) / k
    return (4 / np.pi) * result


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: 几何级数 (r=0.5)")
    result1 = geometric_series(0.5, n=20)
    exact1 = 2.0  # 1/(1-0.5) = 2
    results.append(v.assert_close(result1, exact1, rtol=1e-5, name="几何级数"))

    print("\n测试 2: e^1 的泰勒展开")
    result2 = taylor_exp(1.0, n=15)
    results.append(v.assert_close(result2, np.e, rtol=1e-8, name="e^x 泰勒展开"))

    print("\n测试 3: sin(π/6) 的泰勒展开")
    result3 = taylor_sin(np.pi/6, n=10)
    exact3 = 0.5  # sin(30°) = 0.5
    results.append(v.assert_close(result3, exact3, rtol=1e-8, name="sin(x) 泰勒展开"))

    print("\n测试 4: cos(π/3) 的泰勒展开")
    result4 = taylor_cos(np.pi/3, n=10)
    exact4 = 0.5  # cos(60°) = 0.5
    results.append(v.assert_close(result4, exact4, rtol=1e-8, name="cos(x) 泰勒展开"))

    print("\n测试 5: 莱布尼茨公式计算 π")
    result5 = leibniz_pi(n=100000)
    results.append(v.assert_close(result5, np.pi, rtol=1e-4, name="π 的莱布尼茨公式"))

    print("\n测试 6: 调和级数 H_1000")
    result6 = harmonic_series(1000)
    # H_1000 ≈ ln(1000) + 0.5772
    expected6 = np.log(1000) + 0.5772
    results.append(v.assert_close(result6, expected6, rtol=0.01, name="调和级数"))

    return all(results)


if __name__ == "__main__":
    test()
