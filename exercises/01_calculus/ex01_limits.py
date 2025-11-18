"""
练习 1: 极限计算
=================

本练习包含3道题目，由浅入深：
- 初级：基本极限的数值计算
- 中级：重要极限和极限定理
- 高级：极限的精确性和收敛速度

学习目标：
- 理解极限的数值计算
- 掌握 NumPy 进行数值逼近
- 理解连续性和收敛性

参考知识点：
- 极限的ε-δ定义
- 重要极限（sin(x)/x, e的定义）
- 夹逼定理
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：基本极限的数值计算
# ============================================================================

def limit_sin_x_over_x(epsilon=1e-10):
    """
    初级 - 计算 lim(x→0) sin(x)/x

    这是微积分中最重要的极限之一，结果应该接近 1
    使用小的 x 值来逼近极限

    参数:
        epsilon: 接近0的小值

    返回:
        极限的数值近似
    """
    # TODO: 实现极限计算
    # 提示: 使用 x = epsilon 来逼近 x→0
    x = epsilon
    result = np.sin(x) / x
    return result


def limit_polynomial_ratio(x=1000):
    """
    初级 - 计算 lim(n→∞) (n² + 3n + 1) / (2n² - n)

    对于多项式比值，看最高次项的系数比
    期望结果: 1/2

    参数:
        x: 较大的数值来逼近无穷

    返回:
        极限值
    """
    # TODO: 实现多项式极限
    n = x
    numerator = n**2 + 3*n + 1
    denominator = 2*n**2 - n
    result = numerator / denominator
    return result


# ============================================================================
# 中级题目：重要极限和极限定理
# ============================================================================

def limit_exp_definition(n=1000000):
    """
    中级 - 计算 lim(n→∞) (1 + 1/n)^n

    这个极限定义了自然常数 e ≈ 2.71828
    是数学中最重要的常数之一

    参数:
        n: 较大的整数

    返回:
        e 的近似值
    """
    # TODO: 实现 e 的极限定义
    result = (1 + 1/n) ** n
    return result


def limit_squeeze_theorem(x=0.0001):
    """
    中级 - 使用夹逼定理计算 lim(x→0) x² * sin(1/x)

    虽然 sin(1/x) 在 x→0 时振荡，但 x² 趋于 0
    因为 -1 ≤ sin(1/x) ≤ 1
    所以 -x² ≤ x²*sin(1/x) ≤ x²

    由夹逼定理，极限为 0

    参数:
        x: 接近0的值

    返回:
        极限值
    """
    # TODO: 实现此极限
    if abs(x) < 1e-10:
        return 0.0
    result = x**2 * np.sin(1/x)
    return result


def limit_composite_function(h=1e-6):
    """
    中级 - 计算复合函数极限 lim(x→0) (e^x - 1) / x

    这个极限等于 1，可用洛必达法则验证
    或用泰勒展开: e^x = 1 + x + x²/2! + ...

    参数:
        h: 接近0的值

    返回:
        极限值
    """
    # TODO: 实现复合函数极限
    if abs(h) < 1e-10:
        return 1.0
    result = (np.exp(h) - 1) / h
    return result


# ============================================================================
# 高级题目：极限的精确性和收敛速度
# ============================================================================

def richardson_extrapolation_limit(f, h=0.1, order=2):
    """
    高级 - Richardson外推法提高极限计算精度

    通过组合不同步长的近似来消除误差
    对于极限 lim(h→0) f(h)，使用:
    L ≈ (2^order * f(h/2) - f(h)) / (2^order - 1)

    参数:
        f: 函数，f(h) 趋向某个极限
        h: 初始步长
        order: 外推阶数

    返回:
        提高精度后的极限值
    """
    # TODO: 实现Richardson外推
    f_h = f(h)
    f_h2 = f(h/2)

    extrapolated = (2**order * f_h2 - f_h) / (2**order - 1)

    return extrapolated


def convergence_rate_analysis(sequence_func, n_values):
    """
    高级 - 分析序列的收敛速度

    计算序列 {a_n} 的收敛率
    如果 lim(n→∞) |a_{n+1} - L| / |a_n - L|^p = C
    则称序列以p阶收敛

    参数:
        sequence_func: 生成序列的函数 a_n = sequence_func(n)
        n_values: 用于分析的n值列表

    返回:
        估计的收敛阶 p
    """
    # TODO: 实现收敛速度分析
    # 简化版：计算连续项的比值
    values = [sequence_func(n) for n in n_values]

    # 假设极限已知或可以从最后几项估计
    L = values[-1]  # 粗略估计极限

    # 计算误差序列
    errors = [abs(val - L) for val in values[:-1]]

    # 估计收敛阶（简化）
    if len(errors) >= 2 and errors[-2] > 1e-10:
        ratio = errors[-1] / errors[-2]
        # 如果是线性收敛，p≈1
        if 0 < ratio < 1:
            p = -np.log(ratio) / np.log(2)  # 粗略估计
            return min(p, 2.0)  # 上限为2阶

    return 1.0  # 默认线性收敛


def epsilon_delta_verification(f, x0, L, epsilon, delta_func):
    """
    高级 - 验证 ε-δ 定义

    lim(x→x0) f(x) = L 当且仅当
    对所有 ε > 0, 存在 δ > 0 使得
    0 < |x - x0| < δ ⟹ |f(x) - L| < ε

    参数:
        f: 函数
        x0: 极限点
        L: 极限值
        epsilon: ε 值
        delta_func: 返回对应 δ 的函数

    返回:
        是否满足 ε-δ 条件
    """
    # TODO: 实现 ε-δ 验证
    delta = delta_func(epsilon)

    # 在 (x0 - delta, x0 + delta) 范围内采样
    test_points = np.linspace(x0 - delta, x0 + delta, 100)

    for x in test_points:
        if 0 < abs(x - x0) < delta:
            if abs(f(x) - L) >= epsilon:
                return False

    return True


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：基本极限的数值计算")
    print("="*60)

    print("\n测试 1.1: lim(x→0) sin(x)/x = 1")
    result1 = limit_sin_x_over_x()
    results.append(v.assert_close(result1, 1.0, rtol=1e-3, name="sin(x)/x 的极限"))

    print("\n测试 1.2: 多项式比值极限")
    result2 = limit_polynomial_ratio(1000)
    results.append(v.assert_close(result2, 0.5, rtol=1e-3, name="多项式极限"))

    print("\n" + "="*60)
    print("中级题目：重要极限和极限定理")
    print("="*60)

    print("\n测试 2.1: lim(n→∞) (1 + 1/n)^n = e")
    result3 = limit_exp_definition()
    results.append(v.assert_close(result3, np.e, rtol=1e-4, name="e 的极限定义"))

    print("\n测试 2.2: lim(x→0) x² * sin(1/x) = 0 (夹逼定理)")
    result4 = limit_squeeze_theorem(0.0001)
    results.append(v.assert_close(result4, 0.0, atol=1e-6, name="夹逼定理极限"))

    print("\n测试 2.3: lim(x→0) (e^x - 1) / x = 1")
    result5 = limit_composite_function(1e-6)
    results.append(v.assert_close(result5, 1.0, rtol=1e-3, name="复合函数极限"))

    print("\n" + "="*60)
    print("高级题目：极限的精确性和收敛速度")
    print("="*60)

    print("\n测试 3.1: Richardson外推")
    # 测试 lim(h→0) sin(h)/h = 1
    f = lambda h: np.sin(h)/h if h > 0 else 1.0
    extrapolated = richardson_extrapolation_limit(f, h=0.1, order=2)
    results.append(v.assert_close(extrapolated, 1.0, rtol=1e-4, name="Richardson外推"))

    print("\n测试 3.2: 收敛速度分析")
    # 序列: a_n = 1/n → 0 (线性收敛)
    seq = lambda n: 1.0 / n
    rate = convergence_rate_analysis(seq, list(range(10, 101, 10)))
    results.append(0.5 <= rate <= 2.0)  # 合理的收敛阶
    print(f"  估计收敛阶: {rate:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.3: ε-δ 验证")
    # 验证 lim(x→2) (x²) = 4
    f = lambda x: x**2
    delta_func = lambda eps: eps / 5  # 对于 f(x)=x² 在 x=2 附近
    verified = epsilon_delta_verification(f, x0=2, L=4, epsilon=0.1, delta_func=delta_func)
    results.append(verified)
    print(f"  ε-δ 定义验证: {'✓ 通过' if verified else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
