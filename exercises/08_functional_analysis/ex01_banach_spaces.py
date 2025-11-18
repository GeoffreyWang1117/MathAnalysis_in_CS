"""
练习 1: Banach空间
==================

本练习包含3道题目，由浅入深：
- 初级：赋范空间的基本概念
- 中级：完备性和压缩映射定理
- 高级：Hahn-Banach定理和对偶空间

参考教材：
- Rudin - Functional Analysis
- Conway - A Course in Functional Analysis
- Brezis - Functional Analysis, Sobolev Spaces and Partial Differential Equations
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：赋范空间的基本概念
# ============================================================================

def verify_norm_axioms(X_sample, norm_func, zero):
    """
    初级 - 验证范数公理

    范数必须满足：
    1. ||x|| ≥ 0, 且 ||x|| = 0 ⟺ x = 0
    2. ||αx|| = |α| ||x||
    3. ||x + y|| ≤ ||x|| + ||y|| (三角不等式)

    参数:
        X_sample: 空间元素样本
        norm_func: 候选范数函数
        zero: 零元素

    返回:
        是否满足范数公理
    """
    # TODO: 实现范数公理验证
    # 公理1: 正定性
    for x in X_sample:
        if norm_func(x) < 0:
            return False
        if np.allclose(x, zero) and norm_func(x) > 1e-10:
            return False

    # 公理2: 齐次性（抽样验证）
    if len(X_sample) > 0:
        x = X_sample[0]
        alpha = 2.5
        if not np.isclose(norm_func(alpha * x), abs(alpha) * norm_func(x), rtol=1e-5):
            return False

    # 公理3: 三角不等式（抽样验证）
    if len(X_sample) >= 2:
        x, y = X_sample[0], X_sample[1]
        if norm_func(x + y) > norm_func(x) + norm_func(y) + 1e-5:
            return False

    return True


def equivalent_norms(X_sample, norm1, norm2):
    """
    初级 - 检查两个范数是否等价

    范数等价：存在常数c, C使得 c||x||₁ ≤ ||x||₂ ≤ C||x||₁

    在有限维空间中，所有范数都等价

    参数:
        X_sample: 样本点
        norm1, norm2: 两个范数函数

    返回:
        (是否等价, 估计的常数)
    """
    # TODO: 实现范数等价性检验
    if len(X_sample) == 0:
        return False, (0, 0)

    ratios = []
    for x in X_sample:
        n1 = norm1(x)
        n2 = norm2(x)
        if n1 > 1e-10:  # 避免除零
            ratios.append(n2 / n1)

    if len(ratios) == 0:
        return False, (0, 0)

    c = min(ratios)
    C = max(ratios)

    # 检查是否有有限的界
    is_equivalent = (c > 0 and C < float('inf'))

    return is_equivalent, (c, C)


def unit_ball_elements(norm_func, n_samples=100, dim=2):
    """
    初级 - 生成单位球中的点

    B = {x ∈ X : ||x|| ≤ 1}

    参数:
        norm_func: 范数函数
        n_samples: 采样数
        dim: 维度
    """
    # TODO: 实现单位球采样
    unit_ball = []
    for _ in range(n_samples):
        x = np.random.randn(dim)
        # 归一化到单位球内
        if norm_func(x) > 0:
            x = x / (norm_func(x) + 0.1)  # 稍微缩小确保在球内
            unit_ball.append(x)

    return unit_ball


# ============================================================================
# 中级题目：完备性和压缩映射定理
# ============================================================================

def is_cauchy_sequence(sequence, norm_func, epsilon=1e-6):
    """
    中级 - 判断是否是Cauchy序列

    Cauchy序列：对所有 ε > 0, 存在 N 使得 n, m > N 时 ||x_n - x_m|| < ε

    参数:
        sequence: 点列
        norm_func: 范数
        epsilon: 容差
    """
    # TODO: 实现Cauchy序列判定
    n = len(sequence)
    if n < 2:
        return True

    # 检查序列尾部的距离
    tail_length = min(10, n // 2)
    for i in range(n - tail_length, n):
        for j in range(i + 1, n):
            if norm_func(sequence[i] - sequence[j]) > epsilon:
                return False

    return True


def contraction_mapping_fixed_point(T, x0, norm_func, max_iter=1000, tol=1e-6):
    """
    中级 - 压缩映射定理寻找不动点

    如果 ||T(x) - T(y)|| ≤ λ||x - y||, λ < 1
    则T有唯一不动点，且可通过迭代 x_{n+1} = T(x_n) 找到

    参数:
        T: 压缩映射
        x0: 初始点
        norm_func: 范数
        max_iter: 最大迭代次数
        tol: 容差

    返回:
        不动点
    """
    # TODO: 实现不动点迭代
    x = x0.copy() if isinstance(x0, np.ndarray) else x0

    for i in range(max_iter):
        x_new = T(x)

        if norm_func(x_new - x) < tol:
            return x_new

        x = x_new

    return x


def verify_contraction(T, X_sample, norm_func):
    """
    中级 - 验证映射是否是压缩映射

    参数:
        T: 映射
        X_sample: 样本点
        norm_func: 范数

    返回:
        (是否是压缩, 估计的Lipschitz常数)
    """
    # TODO: 实现压缩性验证
    if len(X_sample) < 2:
        return False, 1.0

    max_ratio = 0
    for x in X_sample:
        for y in X_sample:
            if np.allclose(x, y):
                continue

            dist_xy = norm_func(x - y)
            dist_Tx_Ty = norm_func(T(x) - T(y))

            if dist_xy > 1e-10:
                ratio = dist_Tx_Ty / dist_xy
                max_ratio = max(max_ratio, ratio)

    is_contraction = max_ratio < 1.0

    return is_contraction, max_ratio


# ============================================================================
# 高级题目：线性泛函和对偶空间
# ============================================================================

def is_linear_functional(f, X_sample):
    """
    高级 - 验证是否是线性泛函

    线性泛函：f(αx + βy) = αf(x) + βf(y)

    参数:
        f: 候选泛函
        X_sample: 样本点
    """
    # TODO: 实现线性泛函验证
    if len(X_sample) < 2:
        return True

    # 抽样验证线性性
    x, y = X_sample[0], X_sample[1]
    alpha, beta = 2.0, 3.0

    lhs = f(alpha * x + beta * y)
    rhs = alpha * f(x) + beta * f(y)

    return np.isclose(lhs, rhs, rtol=1e-5)


def operator_norm(T, norm_X, norm_Y, X_sample):
    """
    高级 - 计算算子范数

    ||T|| = sup{||T(x)||_Y / ||x||_X : x ≠ 0}

    参数:
        T: 线性算子
        norm_X: 定义域范数
        norm_Y: 值域范数
        X_sample: 样本点

    返回:
        算子范数的估计
    """
    # TODO: 实现算子范数
    max_ratio = 0

    for x in X_sample:
        norm_x = norm_X(x)
        if norm_x > 1e-10:
            norm_Tx = norm_Y(T(x))
            ratio = norm_Tx / norm_x
            max_ratio = max(max_ratio, ratio)

    return max_ratio


def riesz_representation_l2(f, basis_functions, interval=(0, 1)):
    """
    高级 - Riesz表示定理在L²上的应用

    L²上的每个连续线性泛函可表示为
    f(x) = ⟨x, y⟩ 对某个 y ∈ L²

    参数:
        f: 线性泛函
        basis_functions: L²的正交基
        interval: 区间

    返回:
        表示元素y的系数
    """
    # TODO: 实现Riesz表示
    # 在正交基下，y的系数就是 f(φ_i)
    coefficients = []

    for phi in basis_functions:
        coef = f(phi)
        coefficients.append(coef)

    return np.array(coefficients)


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：赋范空间的基本概念")
    print("="*60)

    print("\n测试 1.1: 范数公理验证")
    X_sample = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    norm_l2 = lambda x: np.linalg.norm(x, 2)
    zero = np.array([0, 0])
    result1 = verify_norm_axioms(X_sample, norm_l2, zero)
    results.append(result1)
    print(f"  L²范数满足公理: {'✓ 通过' if result1 else '✗ 失败'}")

    print("\n测试 1.2: 范数等价性")
    norm_l1 = lambda x: np.linalg.norm(x, 1)
    is_equiv, (c, C) = equivalent_norms(X_sample, norm_l1, norm_l2)
    results.append(is_equiv)
    print(f"  L¹和L²范数等价: {'✓ 是' if is_equiv else '✗ 否'}")
    print(f"  常数: c={c:.4f}, C={C:.4f}")

    print("\n" + "="*60)
    print("中级题目：完备性和压缩映射定理")
    print("="*60)

    print("\n测试 2.1: Cauchy序列")
    # 收敛序列: 1, 1/2, 1/3, ...
    sequence = [np.array([1/n]) for n in range(1, 21)]
    is_cauchy = is_cauchy_sequence(sequence, lambda x: np.abs(x[0]))
    results.append(is_cauchy)
    print(f"  序列是Cauchy: {'✓ 是' if is_cauchy else '✗ 否'}")

    print("\n测试 2.2: 压缩映射不动点")
    # T(x) = x/2 + 1, 不动点是 x = 2
    T = lambda x: x / 2 + 1
    x0 = np.array([0.0])
    fixed_point = contraction_mapping_fixed_point(T, x0, lambda x: np.abs(x[0]))
    results.append(v.assert_close(fixed_point, np.array([2.0]), rtol=1e-3, name="不动点"))

    print("\n测试 2.3: 压缩性验证")
    X_test = [np.array([x]) for x in [0, 1, 2, 3]]
    is_contr, lipschitz = verify_contraction(T, X_test, lambda x: np.abs(x[0]))
    results.append(is_contr)
    print(f"  是压缩映射: {'✓ 是' if is_contr else '✗ 否'}")
    print(f"  Lipschitz常数: {lipschitz:.4f}")

    print("\n" + "="*60)
    print("高级题目：线性泛函和对偶空间")
    print("="*60)

    print("\n测试 3.1: 线性泛函")
    # f(x) = x[0] + 2*x[1]
    f_linear = lambda x: x[0] + 2 * x[1]
    result3 = is_linear_functional(f_linear, X_sample)
    results.append(result3)
    print(f"  f(x)=x₁+2x₂是线性泛函: {'✓ 是' if result3 else '✗ 否'}")

    print("\n测试 3.2: 算子范数")
    # T(x) = 2x (伸缩算子)
    T_stretch = lambda x: 2 * x
    op_norm = operator_norm(T_stretch, norm_l2, norm_l2, X_sample)
    results.append(v.assert_close(op_norm, 2.0, rtol=0.1, name="算子范数"))

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
