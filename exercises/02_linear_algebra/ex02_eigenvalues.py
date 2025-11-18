"""
练习 2: 特征值与特征向量
========================

学习目标：
- 理解特征值和特征向量的概念
- 掌握幂法求主特征值
- 应用特征分解

任务：
实现特征值计算和应用
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def power_iteration(A, num_iterations=100):
    """
    幂迭代法求最大特征值和对应的特征向量

    算法:
    1. 随机初始化向量 v
    2. 迭代: v = A*v / ||A*v||
    3. 特征值 λ = v^T * A * v

    参数:
        A: n×n 矩阵
        num_iterations: 迭代次数

    返回:
        (eigenvalue, eigenvector)
    """
    # TODO: 实现幂迭代法
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iterations):
        # 矩阵向量乘法
        Av = A @ v
        # 归一化
        v = Av / np.linalg.norm(Av)

    # 计算特征值: λ = v^T A v (Rayleigh商)
    eigenvalue = v.T @ A @ v
    eigenvector = v

    return eigenvalue, eigenvector


def verify_eigenpair(A, eigenvalue, eigenvector, tol=1e-6):
    """
    验证特征值和特征向量: A*v = λ*v

    返回:
        是否满足特征方程
    """
    # TODO: 实现验证
    Av = A @ eigenvector
    lambda_v = eigenvalue * eigenvector
    return np.allclose(Av, lambda_v, atol=tol)


def matrix_diagonalization(A):
    """
    矩阵对角化: A = P * D * P^(-1)

    其中 D 是特征值对角矩阵，P 是特征向量矩阵

    使用 NumPy 的 np.linalg.eig 获取特征值和特征向量
    然后验证分解
    """
    # TODO: 实现矩阵对角化
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # D 是特征值构成的对角矩阵
    D = np.diag(eigenvalues)
    # P 是特征向量矩阵
    P = eigenvectors
    # P^(-1)
    P_inv = np.linalg.inv(P)

    # 验证: A = P * D * P^(-1)
    A_reconstructed = P @ D @ P_inv

    return D, P, A_reconstructed


def matrix_exponential(A, t=1.0):
    """
    矩阵指数: e^(At)

    使用特征分解计算:
    如果 A = P*D*P^(-1)，则 e^(At) = P * e^(Dt) * P^(-1)
    其中 e^(Dt) 是对角矩阵，对角元素是 e^(λ_i * t)

    这在求解ODE系统 dx/dt = Ax 时非常有用
    """
    # TODO: 实现矩阵指数
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # e^(Dt) 是对角矩阵
    exp_D = np.diag(np.exp(eigenvalues * t))

    P = eigenvectors
    P_inv = np.linalg.inv(P)

    # e^(At) = P * e^(Dt) * P^(-1)
    result = P @ exp_D @ P_inv

    return result


def spectral_radius(A):
    """
    谱半径: 所有特征值模的最大值

    ρ(A) = max|λ_i|

    谱半径决定了迭代法的收敛性
    """
    # TODO: 实现谱半径计算
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))


def condition_number(A):
    """
    条件数: κ(A) = ||A|| * ||A^(-1)||

    使用2-范数（最大奇异值）
    条件数衡量矩阵的数值稳定性
    """
    # TODO: 实现条件数
    # 提示：可以使用 np.linalg.cond
    return np.linalg.cond(A)


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 创建测试矩阵（对称矩阵，特征值为实数）
    A = np.array([[4, 1], [1, 3]], dtype=float)

    print("\n测试 1: 幂迭代法求最大特征值")
    eigenvalue, eigenvector = power_iteration(A, num_iterations=100)
    # 验证特征对
    is_valid = verify_eigenpair(A, eigenvalue, eigenvector)
    results.append(is_valid)
    print(f"  最大特征值: {eigenvalue:.6f}")
    print(f"  验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")

    print("\n测试 2: 矩阵对角化")
    D, P, A_reconstructed = matrix_diagonalization(A)
    reconstruction_valid = np.allclose(A, A_reconstructed, rtol=1e-10)
    results.append(reconstruction_valid)
    print(f"  特征值: {np.diag(D)}")
    print(f"  重构验证: {'✓ 通过' if reconstruction_valid else '✗ 失败'}")

    print("\n测试 3: 矩阵指数")
    exp_A = matrix_exponential(A, t=1.0)
    # 验证性质: (e^A)^T * e^A 应该是对称的 (对于对称矩阵A)
    exp_product = exp_A.T @ exp_A
    is_symmetric = np.allclose(exp_product, exp_product.T)
    results.append(is_symmetric)
    print(f"  e^A 计算完成")
    print(f"  对称性验证: {'✓ 通过' if is_symmetric else '✗ 失败'}")

    print("\n测试 4: 谱半径")
    rho = spectral_radius(A)
    max_eigenvalue = max(abs(np.linalg.eigvals(A)))
    results.append(v.assert_close(rho, max_eigenvalue, name="谱半径"))

    print("\n测试 5: 条件数")
    kappa = condition_number(A)
    # 对于良条件矩阵，条件数应该不会太大
    is_well_conditioned = kappa < 100
    results.append(is_well_conditioned)
    print(f"  条件数: {kappa:.2f}")
    print(f"  良条件性: {'✓ 通过' if is_well_conditioned else '✗ 失败'}")

    return all(results)


if __name__ == "__main__":
    test()
