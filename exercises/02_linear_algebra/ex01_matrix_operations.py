"""
练习 1: 矩阵运算基础
====================

本练习包含3道题目，由浅入深：
- 初级：矩阵的基本运算（乘法、转置、迹）
- 中级：矩阵性质检验（对称、正交、范数）
- 高级：矩阵分解和特殊矩阵构造

学习目标：
- 掌握矩阵的基本运算
- 理解矩阵乘法的几何意义
- 实现矩阵的各种性质检验
- 掌握矩阵的数值计算技巧

参考知识点：
- 矩阵乘法和线性变换
- 矩阵的秩和行列式
- 正交矩阵和QR分解
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：矩阵的基本运算
# ============================================================================

def matrix_multiply(A, B):
    """
    初级 - 矩阵乘法: C = AB

    要求: A的列数 = B的行数
    不要使用 np.dot 或 @，手动实现矩阵乘法

    参数:
        A: m×n 矩阵
        B: n×p 矩阵

    返回:
        C: m×p 矩阵
    """
    # TODO: 实现矩阵乘法
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "矩阵维度不匹配"

    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C


def transpose(A):
    """
    初级 - 矩阵转置: A^T

    不要使用 A.T，手动实现转置操作

    参数:
        A: m×n 矩阵

    返回:
        A^T: n×m 矩阵
    """
    # TODO: 实现矩阵转置
    m, n = A.shape
    A_T = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            A_T[j, i] = A[i, j]
    return A_T


def trace(A):
    """
    初级 - 矩阵的迹: tr(A) = Σ A[i,i]

    迹是对角线元素之和，是矩阵特征值之和

    参数:
        A: n×n 方阵

    返回:
        迹的值
    """
    # TODO: 实现矩阵的迹
    assert A.shape[0] == A.shape[1], "必须是方阵"
    result = sum(A[i, i] for i in range(A.shape[0]))
    return result


# ============================================================================
# 中级题目：矩阵性质检验
# ============================================================================

def is_symmetric(A, tol=1e-10):
    """
    中级 - 判断矩阵是否对称: A = A^T

    对称矩阵具有实特征值和正交特征向量

    参数:
        A: n×n 方阵
        tol: 数值容差

    返回:
        是否对称
    """
    # TODO: 实现对称性检验
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A, A.T, atol=tol)


def is_orthogonal(A, tol=1e-10):
    """
    中级 - 判断矩阵是否正交: A^T * A = I

    正交矩阵保持向量的长度和角度
    表示旋转或反射变换

    参数:
        A: n×n 方阵
        tol: 数值容差

    返回:
        是否正交
    """
    # TODO: 实现正交性检验
    if A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    product = A.T @ A
    identity = np.eye(n)
    return np.allclose(product, identity, atol=tol)


def frobenius_norm(A):
    """
    中级 - Frobenius范数: ||A||_F = sqrt(Σ Σ A[i,j]²)

    这是矩阵元素平方和的平方根
    不要使用 np.linalg.norm，手动实现

    参数:
        A: 任意矩阵

    返回:
        Frobenius范数
    """
    # TODO: 实现Frobenius范数
    result = np.sqrt(np.sum(A ** 2))
    return result


def condition_number(A):
    """
    中级 - 计算条件数: cond(A) = ||A|| * ||A^(-1)||

    条件数衡量矩阵的数值稳定性
    条件数大表示矩阵接近奇异

    参数:
        A: n×n 可逆方阵

    返回:
        条件数（使用2-范数）
    """
    # TODO: 实现条件数计算
    try:
        A_inv = np.linalg.inv(A)
        norm_A = np.linalg.norm(A, 2)
        norm_A_inv = np.linalg.norm(A_inv, 2)
        return norm_A * norm_A_inv
    except np.linalg.LinAlgError:
        return float('inf')


# ============================================================================
# 高级题目：矩阵分解和特殊矩阵
# ============================================================================

def outer_product(u, v):
    """
    高级 - 外积: u ⊗ v

    结果是一个矩阵 M，其中 M[i,j] = u[i] * v[j]
    外积是秩1矩阵

    参数:
        u: m维向量
        v: n维向量

    返回:
        m×n 矩阵
    """
    # TODO: 实现外积
    u = u.reshape(-1, 1)
    v = v.reshape(1, -1)
    return u @ v


def gram_schmidt(A):
    """
    高级 - Gram-Schmidt正交化

    将矩阵的列向量正交化
    Q的列向量构成正交基

    参数:
        A: m×n 矩阵，列向量线性无关

    返回:
        Q: m×n 矩阵，列向量两两正交且单位化
    """
    # TODO: 实现Gram-Schmidt正交化
    m, n = A.shape
    Q = np.zeros((m, n))

    for j in range(n):
        # 从A中取第j列
        v = A[:, j].copy()

        # 减去在已有正交向量上的投影
        for i in range(j):
            q = Q[:, i]
            v = v - np.dot(v, q) * q

        # 归一化
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-10:
            Q[:, j] = v / norm_v
        else:
            # 线性相关，无法正交化
            Q[:, j] = 0

    return Q


def construct_givens_rotation(i, j, theta, n):
    """
    高级 - 构造Givens旋转矩阵

    Givens旋转在(i,j)平面内旋转角度theta
    用于矩阵的QR分解和特征值计算

    参数:
        i, j: 旋转平面的两个坐标轴
        theta: 旋转角度（弧度）
        n: 矩阵维度

    返回:
        n×n Givens旋转矩阵
    """
    # TODO: 实现Givens旋转矩阵
    G = np.eye(n)

    c = np.cos(theta)
    s = np.sin(theta)

    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s

    return G


def power_iteration(A, num_iterations=100):
    """
    高级 - 幂迭代法求最大特征值

    通过反复乘以矩阵来找到主特征向量
    收敛到最大特征值对应的特征向量

    参数:
        A: n×n 方阵
        num_iterations: 迭代次数

    返回:
        (lambda_max, v): 最大特征值和对应特征向量
    """
    # TODO: 实现幂迭代法
    n = A.shape[0]
    # 随机初始向量
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iterations):
        # v = A * v
        v = A @ v
        # 归一化
        v = v / np.linalg.norm(v)

    # 计算特征值: λ = v^T A v / v^T v
    lambda_max = (v.T @ A @ v) / (v.T @ v)

    return lambda_max, v


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 准备测试矩阵
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    u = np.array([1, 2, 3])
    w = np.array([4, 5, 6])

    print("\n" + "="*60)
    print("初级题目：矩阵的基本运算")
    print("="*60)

    print("\n测试 1.1: 矩阵乘法")
    C = matrix_multiply(A, B)
    expected_C = np.array([[19, 22], [43, 50]])
    results.append(v.assert_array_equal(C, expected_C, name="矩阵乘法"))

    print("\n测试 1.2: 矩阵转置")
    A_T = transpose(A)
    expected_AT = np.array([[1, 3], [2, 4]])
    results.append(v.assert_array_equal(A_T, expected_AT, name="矩阵转置"))

    print("\n测试 1.3: 矩阵的迹")
    tr_A = trace(A)
    expected_tr = 5  # 1 + 4
    results.append(v.assert_close(tr_A, expected_tr, name="矩阵的迹"))

    print("\n" + "="*60)
    print("中级题目：矩阵性质检验")
    print("="*60)

    print("\n测试 2.1: 对称矩阵检验")
    S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    result4 = is_symmetric(S)
    results.append(result4)
    print(f"  对称矩阵检验: {'✓ 通过' if result4 else '✗ 失败'}")

    print("\n测试 2.2: 正交矩阵检验")
    # 旋转矩阵是正交的
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    result5 = is_orthogonal(R)
    results.append(result5)
    print(f"  正交矩阵检验: {'✓ 通过' if result5 else '✗ 失败'}")

    print("\n测试 2.3: Frobenius范数")
    norm_A = frobenius_norm(A)
    expected_norm = np.sqrt(30)  # sqrt(1+4+9+16)
    results.append(v.assert_close(norm_A, expected_norm, name="Frobenius范数"))

    print("\n测试 2.4: 条件数")
    M = np.array([[1, 0.5], [0.5, 1]])
    cond = condition_number(M)
    results.append(cond < 100)  # 良态矩阵
    print(f"  条件数: {cond:.4f}")
    print(f"  {'✓ 通过（良态）' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：矩阵分解和特殊矩阵")
    print("="*60)

    print("\n测试 3.1: 外积")
    outer = outer_product(u, w)
    expected_outer = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
    results.append(v.assert_array_equal(outer, expected_outer, name="外积"))

    print("\n测试 3.2: Gram-Schmidt正交化")
    A_gs = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    Q = gram_schmidt(A_gs)
    # 检查列向量是否正交
    orthogonal = np.allclose(Q.T @ Q, np.eye(2), atol=1e-10)
    results.append(orthogonal)
    print(f"  列向量正交性: {'✓ 通过' if orthogonal else '✗ 失败'}")

    print("\n测试 3.3: Givens旋转矩阵")
    G = construct_givens_rotation(0, 1, np.pi/2, 3)
    # Givens矩阵应该是正交的
    is_orth = is_orthogonal(G)
    results.append(is_orth)
    print(f"  Givens矩阵正交性: {'✓ 通过' if is_orth else '✗ 失败'}")

    print("\n测试 3.4: 幂迭代法")
    np.random.seed(42)
    A_power = np.array([[4, 1], [1, 3]])
    lambda_max, eigvec = power_iteration(A_power, num_iterations=100)
    # 真实最大特征值约为 4.618
    true_lambda = max(np.linalg.eigvals(A_power))
    results.append(v.assert_close(lambda_max, true_lambda, rtol=0.01, name="幂迭代最大特征值"))

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
