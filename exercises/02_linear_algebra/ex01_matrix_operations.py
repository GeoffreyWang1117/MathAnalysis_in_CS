"""
练习 1: 矩阵运算基础
====================

学习目标：
- 掌握矩阵的基本运算
- 理解矩阵乘法的几何意义
- 实现矩阵的各种性质检验

任务：
使用 NumPy 实现矩阵运算
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def matrix_multiply(A, B):
    """
    矩阵乘法: C = AB

    要求: A的列数 = B的行数
    不要使用 np.dot 或 @，手动实现矩阵乘法
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
    矩阵转置: A^T

    不要使用 A.T，手动实现
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
    矩阵的迹: tr(A) = Σ A[i,i]

    迹是对角线元素之和
    """
    # TODO: 实现矩阵的迹
    assert A.shape[0] == A.shape[1], "必须是方阵"
    result = sum(A[i, i] for i in range(A.shape[0]))
    return result


def is_symmetric(A, tol=1e-10):
    """
    判断矩阵是否对称: A = A^T
    """
    # TODO: 实现对称性检验
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A, A.T, atol=tol)


def is_orthogonal(A, tol=1e-10):
    """
    判断矩阵是否正交: A^T * A = I

    正交矩阵保持向量的长度和角度
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
    Frobenius范数: ||A||_F = sqrt(Σ Σ A[i,j]^2)

    不要使用 np.linalg.norm，手动实现
    """
    # TODO: 实现Frobenius范数
    result = np.sqrt(np.sum(A ** 2))
    return result


def outer_product(u, v):
    """
    外积: u ⊗ v

    结果是一个矩阵 M，其中 M[i,j] = u[i] * v[j]
    """
    # TODO: 实现外积
    u = u.reshape(-1, 1)
    v = v.reshape(1, -1)
    return u @ v


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

    print("\n测试 1: 矩阵乘法")
    C = matrix_multiply(A, B)
    expected_C = np.array([[19, 22], [43, 50]])
    results.append(v.assert_array_equal(C, expected_C, name="矩阵乘法"))

    print("\n测试 2: 矩阵转置")
    A_T = transpose(A)
    expected_AT = np.array([[1, 3], [2, 4]])
    results.append(v.assert_array_equal(A_T, expected_AT, name="矩阵转置"))

    print("\n测试 3: 矩阵的迹")
    tr_A = trace(A)
    expected_tr = 5  # 1 + 4
    results.append(v.assert_close(tr_A, expected_tr, name="矩阵的迹"))

    print("\n测试 4: 对称矩阵检验")
    S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    result4 = is_symmetric(S)
    results.append(result4)
    print(f"✓ 对称矩阵检验: {result4}")

    print("\n测试 5: 正交矩阵检验")
    # 旋转矩阵是正交的
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    result5 = is_orthogonal(R)
    results.append(result5)
    print(f"✓ 正交矩阵检验: {result5}")

    print("\n测试 6: Frobenius范数")
    norm_A = frobenius_norm(A)
    expected_norm = np.sqrt(30)  # sqrt(1+4+9+16)
    results.append(v.assert_close(norm_A, expected_norm, name="Frobenius范数"))

    print("\n测试 7: 外积")
    outer = outer_product(u, w)
    expected_outer = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
    results.append(v.assert_array_equal(outer, expected_outer, name="外积"))

    return all(results)


if __name__ == "__main__":
    test()
