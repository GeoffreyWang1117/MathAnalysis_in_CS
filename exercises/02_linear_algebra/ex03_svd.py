"""
练习 3: 奇异值分解（SVD）
==========================

学习目标：
- 理解SVD的数学意义
- 掌握SVD的应用（降维、图像压缩）
- 实现PCA主成分分析

任务：
应用SVD解决实际问题
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def compute_svd(A):
    """
    计算矩阵的奇异值分解: A = U * Σ * V^T

    使用 np.linalg.svd

    返回:
        U: 左奇异向量
        S: 奇异值（一维数组）
        VT: 右奇异向量的转置
    """
    # TODO: 计算SVD
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    return U, S, VT


def low_rank_approximation(A, k):
    """
    低秩近似: 用前 k 个奇异值重构矩阵

    A ≈ A_k = Σ(i=1 to k) σ_i * u_i * v_i^T

    这是矩阵的最佳秩-k近似（Frobenius范数意义下）

    参数:
        A: m×n 矩阵
        k: 保留的奇异值数量

    返回:
        A_k: 秩为k的近似矩阵
    """
    # TODO: 实现低秩近似
    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # 只保留前 k 个奇异值
    S_k = S.copy()
    S_k[k:] = 0

    # 重构矩阵
    A_k = U @ np.diag(S_k) @ VT

    return A_k


def matrix_rank_from_svd(A, tol=1e-10):
    """
    从SVD计算矩阵的秩

    秩 = 非零奇异值的数量（大于容差的奇异值）
    """
    # TODO: 计算矩阵的秩
    _, S, _ = np.linalg.svd(A)
    rank = np.sum(S > tol)
    return rank


def moore_penrose_pseudoinverse(A):
    """
    Moore-Penrose伪逆: A^+ = V * Σ^+ * U^T

    其中 Σ^+ 是 Σ 的伪逆（非零奇异值取倒数）

    伪逆用于求解最小二乘问题: min ||Ax - b||
    解为 x = A^+ * b
    """
    # TODO: 实现伪逆
    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # 计算 Σ^+（奇异值取倒数，零保持为零）
    S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])

    # A^+ = V * Σ^+ * U^T
    A_pinv = VT.T @ np.diag(S_inv) @ U.T

    return A_pinv


def pca_transform(X, n_components):
    """
    主成分分析（PCA）降维

    算法:
    1. 中心化数据: X_centered = X - mean(X)
    2. 计算协方差矩阵的SVD
    3. 投影到前 n_components 个主成分

    参数:
        X: (n_samples, n_features) 数据矩阵
        n_components: 保留的主成分数量

    返回:
        X_reduced: (n_samples, n_components) 降维后的数据
        components: (n_components, n_features) 主成分
        explained_variance_ratio: 每个主成分解释的方差比例
    """
    # TODO: 实现PCA
    # 1. 中心化
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 2. SVD
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

    # 3. 主成分 = 前 n_components 个右奇异向量
    components = VT[:n_components]

    # 4. 投影
    X_reduced = X_centered @ components.T

    # 5. 解释方差比例
    total_var = np.sum(S**2)
    explained_variance_ratio = (S[:n_components]**2) / total_var

    return X_reduced, components, explained_variance_ratio


def image_compression_demo():
    """
    图像压缩演示（使用随机图像模拟）

    返回压缩比和误差
    """
    # 创建一个简单的"图像"（随机矩阵）
    np.random.seed(42)
    image = np.random.rand(50, 50)

    # 使用 k=10 的低秩近似
    k = 10
    compressed = low_rank_approximation(image, k)

    # 计算压缩误差
    error = np.linalg.norm(image - compressed, 'fro')

    # 压缩比
    original_size = image.size
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    compression_ratio = original_size / compressed_size

    return compression_ratio, error


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 测试矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=float)

    print("\n测试 1: SVD分解")
    U, S, VT = compute_svd(A)
    A_reconstructed = U @ np.diag(S) @ VT
    results.append(v.assert_close(A_reconstructed, A, rtol=1e-10, name="SVD重构"))

    print("\n测试 2: 低秩近似")
    k = 2
    A_k = low_rank_approximation(A, k)
    rank_k = matrix_rank_from_svd(A_k, tol=1e-10)
    results.append(rank_k <= k)
    print(f"  近似矩阵的秩: {rank_k} (应该 ≤ {k})")
    print(f"  {'✓ 通过' if rank_k <= k else '✗ 失败'}")

    print("\n测试 3: 矩阵的秩")
    rank_A = matrix_rank_from_svd(A)
    expected_rank = 2  # 这个矩阵的秩是2
    results.append(v.assert_close(rank_A, expected_rank, name="矩阵的秩"))

    print("\n测试 4: Moore-Penrose伪逆")
    A_pinv = moore_penrose_pseudoinverse(A)
    # 验证性质: A * A^+ * A = A
    AAA = A @ A_pinv @ A
    results.append(v.assert_close(AAA, A, rtol=1e-10, name="伪逆性质"))

    print("\n测试 5: PCA降维")
    # 创建测试数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    n_components = 2
    X_reduced, components, var_ratio = pca_transform(X, n_components)

    # 验证降维后的形状
    expected_shape = (100, 2)
    results.append(v.assert_shape(X_reduced, expected_shape, name="PCA降维形状"))

    # 验证方差比例和为1（或接近1）
    print(f"  解释方差比例: {var_ratio}")
    print(f"  总解释方差: {np.sum(var_ratio):.4f}")

    print("\n测试 6: 图像压缩演示")
    compression_ratio, error = image_compression_demo()
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  重构误差: {error:.6f}")
    results.append(compression_ratio > 1.0)  # 应该有压缩效果

    return all(results)


if __name__ == "__main__":
    test()
