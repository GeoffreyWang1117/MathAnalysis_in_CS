"""
练习 1: 极限计算
=================

学习目标：
- 理解极限的数值计算
- 掌握 NumPy 进行数值逼近
- 理解连续性的概念

任务：
计算以下极限的数值近似
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def limit_sin_x_over_x(epsilon=1e-10):
    """
    计算 lim(x→0) sin(x)/x

    提示：这是一个重要的极限，结果应该接近 1
    使用小的 x 值来逼近极限
    """
    # TODO: 实现极限计算
    # 提示: 使用 x = epsilon 来逼近 x→0
    x = epsilon
    result = np.sin(x) / x  # 请修改这一行
    return result


def limit_exp_definition(n=1000000):
    """
    计算 lim(n→∞) (1 + 1/n)^n

    提示：这个极限定义了自然常数 e ≈ 2.71828
    """
    # TODO: 实现 e 的极限定义
    result = (1 + 1/n) ** n  # 请修改这一行计算极限
    return result


def limit_squeeze_theorem(x=0.0001):
    """
    使用夹逼定理计算 lim(x→0) x^2 * sin(1/x)

    提示：虽然 sin(1/x) 在 x→0 时振荡，但 x^2 趋于 0
    因为 -1 ≤ sin(1/x) ≤ 1
    所以 -x^2 ≤ x^2*sin(1/x) ≤ x^2
    """
    # TODO: 实现此极限
    if abs(x) < 1e-10:
        return 0.0
    result = x**2 * np.sin(1/x)  # 请验证这个实现
    return result


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: lim(x→0) sin(x)/x = 1")
    result1 = limit_sin_x_over_x()
    results.append(v.assert_close(result1, 1.0, rtol=1e-3, name="sin(x)/x 的极限"))

    print("\n测试 2: lim(n→∞) (1 + 1/n)^n = e")
    result2 = limit_exp_definition()
    results.append(v.assert_close(result2, np.e, rtol=1e-4, name="e 的极限定义"))

    print("\n测试 3: lim(x→0) x^2 * sin(1/x) = 0")
    result3 = limit_squeeze_theorem(0.0001)
    results.append(v.assert_close(result3, 0.0, atol=1e-6, name="夹逼定理极限"))

    return all(results)


if __name__ == "__main__":
    test()
