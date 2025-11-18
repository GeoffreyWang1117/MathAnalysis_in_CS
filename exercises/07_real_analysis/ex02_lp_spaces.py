"""
练习 2: Lp空间与函数范数
========================

学习目标：
- 理解Lp范数
- 掌握函数空间的性质
- 应用Holder不等式和Minkowski不等式

任务：
实现Lp空间的计算和验证
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def lp_norm(f, p, interval=(0, 1), n=10000):
    """
    计算函数的Lp范数

    ||f||_p = (∫|f|^p dx)^(1/p)

    参数:
        f: 函数
        p: 范数阶数（p >= 1）
        interval: 积分区间
        n: 采样点数
    """
    # TODO: 实现Lp范数
    a, b = interval
    x = np.linspace(a, b, n)
    y = np.abs(f(x))

    if p == np.inf:
        # L∞范数：本质上确界
        return np.max(y)
    else:
        # Lp范数
        dx = (b - a) / n
        integral = np.sum(y**p) * dx
        return integral ** (1/p)


def l_infinity_norm(f, interval=(0, 1), n=10000):
    """
    L∞范数（本质上确界范数）

    ||f||_∞ = ess sup |f(x)|

    参数:
        f: 函数
        interval: 积分区间
        n: 采样点数
    """
    # TODO: 实现L∞范数
    a, b = interval
    x = np.linspace(a, b, n)
    y = np.abs(f(x))
    return np.max(y)


def holder_inequality_verify(f, g, p, q, interval=(0, 1)):
    """
    验证Holder不等式

    ||fg||_1 ≤ ||f||_p * ||g||_q
    其中 1/p + 1/q = 1

    参数:
        f, g: 函数
        p, q: 共轭指数（1/p + 1/q = 1）
        interval: 积分区间

    返回:
        (lhs, rhs, holds): 左边、右边、是否成立
    """
    # TODO: 实现Holder不等式验证
    # 验证 1/p + 1/q = 1
    if abs(1/p + 1/q - 1) > 1e-6:
        raise ValueError("p和q必须是共轭指数：1/p + 1/q = 1")

    # 左边：||fg||_1
    fg = lambda x: f(x) * g(x)
    lhs = lp_norm(fg, 1, interval)

    # 右边：||f||_p * ||g||_q
    norm_f = lp_norm(f, p, interval)
    norm_g = lp_norm(g, q, interval)
    rhs = norm_f * norm_g

    holds = lhs <= rhs + 1e-6

    return lhs, rhs, holds


def minkowski_inequality_verify(f, g, p, interval=(0, 1)):
    """
    验证Minkowski不等式（三角不等式）

    ||f + g||_p ≤ ||f||_p + ||g||_p

    参数:
        f, g: 函数
        p: 范数阶数
        interval: 积分区间
    """
    # TODO: 实现Minkowski不等式验证
    # 左边：||f + g||_p
    f_plus_g = lambda x: f(x) + g(x)
    lhs = lp_norm(f_plus_g, p, interval)

    # 右边：||f||_p + ||g||_p
    rhs = lp_norm(f, p, interval) + lp_norm(g, p, interval)

    holds = lhs <= rhs + 1e-6

    return lhs, rhs, holds


def cauchy_schwarz_inequality(f, g, interval=(0, 1)):
    """
    Cauchy-Schwarz不等式（Holder不等式的特例，p=q=2）

    |∫fg dx| ≤ ||f||_2 * ||g||_2

    参数:
        f, g: 函数
        interval: 积分区间
    """
    # TODO: 实现Cauchy-Schwarz不等式
    a, b = interval
    x = np.linspace(a, b, 10000)
    dx = (b - a) / 10000

    # 左边：|∫fg dx|
    lhs = abs(np.sum(f(x) * g(x)) * dx)

    # 右边：||f||_2 * ||g||_2
    norm_f = lp_norm(f, 2, interval)
    norm_g = lp_norm(g, 2, interval)
    rhs = norm_f * norm_g

    return lhs, rhs, lhs <= rhs + 1e-6


def inner_product_l2(f, g, interval=(0, 1), n=10000):
    """
    L²空间的内积

    ⟨f, g⟩ = ∫f(x)g(x) dx

    参数:
        f, g: 函数
        interval: 积分区间
        n: 采样点数
    """
    # TODO: 实现L²内积
    a, b = interval
    x = np.linspace(a, b, n)
    dx = (b - a) / n
    return np.sum(f(x) * g(x)) * dx


def orthogonal_check(f, g, interval=(0, 1), tolerance=1e-6):
    """
    检查两个函数是否正交

    f ⊥ g 当且仅当 ⟨f, g⟩ = 0

    参数:
        f, g: 函数
        interval: 积分区间
        tolerance: 容差
    """
    # TODO: 实现正交性检验
    inner_prod = inner_product_l2(f, g, interval)
    return abs(inner_prod) < tolerance


def projection_onto_subspace(f, basis_functions, interval=(0, 1)):
    """
    将函数投影到子空间

    Proj_V(f) = Σ ⟨f, φ_i⟩ φ_i

    其中 {φ_i} 是正交归一基

    参数:
        f: 待投影函数
        basis_functions: 正交归一基函数列表
        interval: 区间

    返回:
        投影系数
    """
    # TODO: 实现投影
    coefficients = []
    for phi in basis_functions:
        coef = inner_product_l2(f, phi, interval)
        coefficients.append(coef)
    return np.array(coefficients)


def bessel_inequality(f, basis_functions, interval=(0, 1)):
    """
    Bessel不等式

    Σ |⟨f, φ_i⟩|² ≤ ||f||²

    参数:
        f: 函数
        basis_functions: 正交归一基
        interval: 区间
    """
    # TODO: 实现Bessel不等式验证
    # 左边：Σ |⟨f, φ_i⟩|²
    lhs = 0
    for phi in basis_functions:
        inner_prod = inner_product_l2(f, phi, interval)
        lhs += inner_prod**2

    # 右边：||f||²
    rhs = lp_norm(f, 2, interval)**2

    return lhs, rhs, lhs <= rhs + 1e-6


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: L²范数")
    f = lambda x: x
    norm = lp_norm(f, 2, (0, 1))
    # ||x||₂ = (∫₀¹ x² dx)^(1/2) = (1/3)^(1/2)
    expected = np.sqrt(1/3)
    results.append(v.assert_close(norm, expected, rtol=1e-3, name="L²范数"))

    print("\n测试 2: L∞范数")
    f = lambda x: np.sin(x)
    norm_inf = l_infinity_norm(f, (0, np.pi))
    expected_inf = 1.0  # max(sin(x)) on [0, π] = 1
    results.append(v.assert_close(norm_inf, expected_inf, rtol=1e-3, name="L∞范数"))

    print("\n测试 3: Holder不等式")
    f = lambda x: x
    g = lambda x: x**2
    p, q = 2, 2  # 1/2 + 1/2 = 1
    lhs, rhs, holds = holder_inequality_verify(f, g, p, q, (0, 1))
    results.append(holds)
    print(f"  ||fg||₁ = {lhs:.6f}, ||f||₂||g||₂ = {rhs:.6f}")
    print(f"  Holder不等式: {'✓ 成立' if holds else '✗ 不成立'}")

    print("\n测试 4: Minkowski不等式")
    f = lambda x: x
    g = lambda x: 1
    lhs, rhs, holds = minkowski_inequality_verify(f, g, 2, (0, 1))
    results.append(holds)
    print(f"  ||f+g||₂ = {lhs:.6f}, ||f||₂ + ||g||₂ = {rhs:.6f}")
    print(f"  Minkowski不等式: {'✓ 成立' if holds else '✗ 不成立'}")

    print("\n测试 5: Cauchy-Schwarz不等式")
    f = lambda x: x
    g = lambda x: x**2
    lhs, rhs, holds = cauchy_schwarz_inequality(f, g, (0, 1))
    results.append(holds)
    print(f"  |⟨f,g⟩| = {lhs:.6f}, ||f||₂||g||₂ = {rhs:.6f}")
    print(f"  Cauchy-Schwarz: {'✓ 成立' if holds else '✗ 不成立'}")

    print("\n测试 6: L²内积")
    f = lambda x: np.sin(np.pi * x)
    g = lambda x: np.cos(np.pi * x)
    inner_prod = inner_product_l2(f, g, (0, 1))
    # sin(πx) 和 cos(πx) 在 [0, 1] 上正交
    results.append(abs(inner_prod) < 0.01)
    print(f"  ⟨sin(πx), cos(πx)⟩ = {inner_prod:.6f}")
    print(f"  正交性: {'✓ 正交' if abs(inner_prod) < 0.01 else '✗ 不正交'}")

    print("\n测试 7: 正交性检验")
    f1 = lambda x: np.sqrt(2) * np.sin(np.pi * x)
    f2 = lambda x: np.sqrt(2) * np.sin(2 * np.pi * x)
    is_orthogonal = orthogonal_check(f1, f2, (0, 1))
    results.append(is_orthogonal)
    print(f"  f1 ⊥ f2: {'✓ 是' if is_orthogonal else '✗ 否'}")

    print("\n测试 8: Bessel不等式")
    f = lambda x: x**2
    # 使用前两个正交归一的正弦函数作为基
    basis = [
        lambda x: np.sqrt(2) * np.sin(np.pi * x),
        lambda x: np.sqrt(2) * np.sin(2 * np.pi * x)
    ]
    lhs, rhs, holds = bessel_inequality(f, basis, (0, 1))
    results.append(holds)
    print(f"  Σ|⟨f,φᵢ⟩|² = {lhs:.6f}, ||f||² = {rhs:.6f}")
    print(f"  Bessel不等式: {'✓ 成立' if holds else '✗ 不成立'}")

    return all(results)


if __name__ == "__main__":
    test()
