"""
练习 1: 测度论基础
==================

学习目标：
- 理解测度的概念
- 掌握Lebesgue测度
- 实现可测函数的数值计算

任务：
实现测度论的基本概念
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def lebesgue_measure_interval(a, b):
    """
    计算区间的Lebesgue测度

    对于区间 [a, b]，测度 μ([a, b]) = b - a

    参数:
        a, b: 区间端点
    """
    # TODO: 实现区间的Lebesgue测度
    return b - a


def lebesgue_measure_union_disjoint(intervals):
    """
    计算不相交区间并集的测度

    如果区间不相交，则 μ(A ∪ B) = μ(A) + μ(B)

    参数:
        intervals: 区间列表 [(a1, b1), (a2, b2), ...]
    """
    # TODO: 实现不相交区间并集的测度
    total_measure = 0
    for a, b in intervals:
        total_measure += (b - a)
    return total_measure


def outer_measure_approximation(points, epsilon=0.1):
    """
    外测度近似

    用开区间覆盖点集，计算外测度

    参数:
        points: 点集（一维数组）
        epsilon: 覆盖半径
    """
    # TODO: 实现外测度近似
    # 每个点用 (point - epsilon, point + epsilon) 覆盖
    points = np.array(points)

    # 合并重叠区间（简化版）
    if len(points) == 0:
        return 0

    total_length = len(points) * 2 * epsilon
    return total_length


def cantor_set_measure(n_iterations=5):
    """
    康托集的测度计算

    康托集是通过不断移除中间三分之一构造的
    每次迭代后，剩余长度为前一次的 2/3

    参数:
        n_iterations: 迭代次数

    返回:
        康托集的近似测度
    """
    # TODO: 实现康托集测度计算
    # 初始长度为 1
    length = 1.0

    for i in range(n_iterations):
        # 每次移除中间 1/3，剩余 2/3
        length *= (2.0 / 3.0)

    return length


def characteristic_function(x, interval):
    """
    示性函数（特征函数）

    χ_A(x) = 1 if x ∈ A, else 0

    参数:
        x: 点或点数组
        interval: (a, b) 区间
    """
    # TODO: 实现示性函数
    a, b = interval
    x = np.asarray(x)
    return np.where((x >= a) & (x <= b), 1.0, 0.0)


def simple_function_integral(x_values, intervals, coefficients):
    """
    简单函数的积分

    简单函数: f = Σ c_i * χ_{A_i}
    积分: ∫f dμ = Σ c_i * μ(A_i)

    参数:
        x_values: 采样点
        intervals: 区间列表
        coefficients: 系数列表
    """
    # TODO: 实现简单函数积分
    integral = 0
    for coef, interval in zip(coefficients, intervals):
        measure = lebesgue_measure_interval(interval[0], interval[1])
        integral += coef * measure
    return integral


def lebesgue_integral_approximation(f, a, b, n=10000):
    """
    Lebesgue积分的数值近似

    通过简单函数逼近来计算 Lebesgue 积分

    参数:
        f: 函数
        a, b: 积分区间
        n: 分割数
    """
    # TODO: 实现Lebesgue积分近似
    # 使用黎曼和作为Lebesgue积分的近似
    x = np.linspace(a, b, n)
    y = f(x)
    dx = (b - a) / n
    return np.sum(y) * dx


def almost_everywhere_equality(f, g, x_sample, tolerance=1e-6):
    """
    检验两个函数是否几乎处处相等

    如果 {x : f(x) ≠ g(x)} 的测度为 0，则 f = g a.e.

    参数:
        f, g: 两个函数
        x_sample: 采样点
        tolerance: 容差
    """
    # TODO: 实现几乎处处相等检验
    x_sample = np.array(x_sample)
    diff = np.abs(f(x_sample) - g(x_sample))

    # 计算不相等点的比例
    not_equal_ratio = np.mean(diff > tolerance)

    # 如果不相等点的比例很小，则认为几乎处处相等
    return not_equal_ratio < 0.01


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: 区间的Lebesgue测度")
    measure = lebesgue_measure_interval(0, 5)
    results.append(v.assert_close(measure, 5.0, name="区间[0,5]的测度"))

    print("\n测试 2: 不相交区间并集的测度")
    intervals = [(0, 1), (2, 3), (5, 7)]
    total = lebesgue_measure_union_disjoint(intervals)
    expected = 1 + 1 + 2  # = 4
    results.append(v.assert_close(total, expected, name="并集测度"))

    print("\n测试 3: 康托集测度")
    cantor_measure = cantor_set_measure(n_iterations=20)
    # 理论上，康托集的测度为 0
    results.append(cantor_measure < 0.001)
    print(f"  康托集测度（20次迭代）: {cantor_measure:.8f}")
    print(f"  {'✓ 通过（趋向0）' if cantor_measure < 0.001 else '✗ 失败'}")

    print("\n测试 4: 示性函数")
    x = np.array([0, 0.5, 1, 1.5, 2])
    chi = characteristic_function(x, (0, 1))
    expected_chi = np.array([1, 1, 1, 0, 0])
    results.append(v.assert_close(chi, expected_chi, name="示性函数"))

    print("\n测试 5: 简单函数积分")
    intervals = [(0, 1), (1, 2)]
    coefficients = [2, 3]
    integral = simple_function_integral(None, intervals, coefficients)
    expected_integral = 2 * 1 + 3 * 1  # = 5
    results.append(v.assert_close(integral, expected_integral, name="简单函数积分"))

    print("\n测试 6: Lebesgue积分近似")
    f = lambda x: x**2
    integral = lebesgue_integral_approximation(f, 0, 1, n=10000)
    expected = 1/3  # ∫₀¹ x² dx = 1/3
    results.append(v.assert_close(integral, expected, rtol=1e-3, name="Lebesgue积分"))

    print("\n测试 7: 几乎处处相等")
    f1 = lambda x: x**2
    f2 = lambda x: x**2  # 完全相同
    x_sample = np.linspace(0, 1, 1000)
    is_ae_equal = almost_everywhere_equality(f1, f2, x_sample)
    results.append(is_ae_equal)
    print(f"  f1 = f2 a.e.: {'✓ 是' if is_ae_equal else '✗ 否'}")

    return all(results)


if __name__ == "__main__":
    test()
