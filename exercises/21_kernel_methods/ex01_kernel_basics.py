"""
练习 1: 核方法
==============

本练习包含3道题目，由浅入深：
- 初级：核函数和核矩阵
- 中级：核技巧和RKHS
- 高级：核PCA和核方法应用

学习目标：
- 理解核函数和核技巧
- 掌握常用核函数的性质
- 理解再生核希尔伯特空间(RKHS)
- 实现核PCA和核方法

应用领域：
- 支持向量机 (SVM)
- 核主成分分析 (Kernel PCA)
- 高斯过程 (Gaussian Process)
- 核回归

参考教材：
- Schölkopf & Smola - Learning with Kernels
- Rasmussen & Williams - Gaussian Processes for Machine Learning
- Shawe-Taylor & Cristianini - Kernel Methods for Pattern Analysis
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：核函数和核矩阵
# ============================================================================

def linear_kernel(X, Y=None):
    """
    初级 - 线性核

    k(x, y) = x^T y

    这是最简单的核函数

    参数:
        X: n×d矩阵
        Y: m×d矩阵 (如果None,则Y=X)

    返回:
        n×m核矩阵
    """
    # TODO: 实现线性核
    if Y is None:
        Y = X

    K = X @ Y.T

    return K


def polynomial_kernel(X, Y=None, degree=3, coef0=1):
    """
    初级 - 多项式核

    k(x, y) = (x^T y + c)^d

    参数:
        X: n×d矩阵
        Y: m×d矩阵
        degree: 多项式度数d
        coef0: 常数项c

    返回:
        核矩阵
    """
    # TODO: 实现多项式核
    if Y is None:
        Y = X

    K = (X @ Y.T + coef0) ** degree

    return K


def rbf_kernel(X, Y=None, gamma=1.0):
    """
    初级 - 径向基函数核 (RBF/高斯核)

    k(x, y) = exp(-γ ||x - y||²)

    这是最常用的核函数之一

    参数:
        X: n×d矩阵
        Y: m×d矩阵
        gamma: 尺度参数γ

    返回:
        核矩阵
    """
    # TODO: 实现RBF核
    if Y is None:
        Y = X

    # 计算欧氏距离的平方
    # ||x - y||² = ||x||² + ||y||² - 2x^Ty
    XX = np.sum(X**2, axis=1).reshape(-1, 1)
    YY = np.sum(Y**2, axis=1).reshape(1, -1)
    XY = X @ Y.T

    sq_distances = XX + YY - 2 * XY

    K = np.exp(-gamma * sq_distances)

    return K


def check_kernel_positive_definite(K, tol=1e-10):
    """
    初级 - 检查核矩阵是否正定

    有效的核函数必须产生正半定核矩阵

    参数:
        K: 核矩阵
        tol: 数值容差

    返回:
        是否正定
    """
    # TODO: 实现正定性检查
    # 检查对称性
    if not np.allclose(K, K.T, atol=tol):
        return False

    # 检查特征值非负
    eigenvalues = np.linalg.eigvalsh(K)

    return np.all(eigenvalues >= -tol)


# ============================================================================
# 中级题目：核技巧和RKHS
# ============================================================================

def kernel_mean(K, weights=None):
    """
    中级 - 核均值

    在特征空间中计算均值:
    μ = (1/n) Σᵢ φ(xᵢ)

    使用核技巧: 不需要显式计算φ(x)

    参数:
        K: n×n核矩阵
        weights: 样本权重 (默认均匀)

    返回:
        核均值表示 (n维向量)
    """
    # TODO: 实现核均值
    n = K.shape[0]

    if weights is None:
        weights = np.ones(n) / n

    # 核均值在特征空间中的表示
    # 以核矩阵列的加权和表示
    kernel_mean_rep = K @ weights

    return kernel_mean_rep


def center_kernel_matrix(K):
    """
    中级 - 中心化核矩阵

    在特征空间中中心化数据:
    K_centered = (I - 11^T/n) K (I - 11^T/n)

    参数:
        K: n×n核矩阵

    返回:
        中心化后的核矩阵
    """
    # TODO: 实现核矩阵中心化
    n = K.shape[0]

    # 中心化矩阵
    one_n = np.ones((n, n)) / n

    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    return K_centered


def kernel_alignment(K1, K2):
    """
    中级 - 核对齐度

    衡量两个核的相似性:
    A(K1, K2) = <K1, K2>_F / (||K1||_F * ||K2||_F)

    其中<·,·>_F是Frobenius内积

    参数:
        K1, K2: 核矩阵

    返回:
        对齐度 ∈ [0, 1]
    """
    # TODO: 实现核对齐
    # Frobenius内积
    inner_product = np.sum(K1 * K2)

    # Frobenius范数
    norm_K1 = np.sqrt(np.sum(K1 ** 2))
    norm_K2 = np.sqrt(np.sum(K2 ** 2))

    if norm_K1 == 0 or norm_K2 == 0:
        return 0

    alignment = inner_product / (norm_K1 * norm_K2)

    return alignment


def kernel_ridge_regression(K, y, lambda_reg=0.1):
    """
    中级 - 核岭回归

    最小化: ||y - Kα||² + λ α^T K α

    解: α = (K + λI)^(-1) y

    参数:
        K: n×n核矩阵
        y: n维目标向量
        lambda_reg: 正则化参数λ

    返回:
        权重向量α
    """
    # TODO: 实现核岭回归
    n = K.shape[0]
    I = np.eye(n)

    alpha = np.linalg.solve(K + lambda_reg * I, y)

    return alpha


def kernel_trick_distance(x1, x2, kernel_func):
    """
    中级 - 使用核技巧计算特征空间距离

    ||φ(x1) - φ(x2)||² = k(x1,x1) + k(x2,x2) - 2k(x1,x2)

    参数:
        x1, x2: 数据点
        kernel_func: 核函数

    返回:
        特征空间距离
    """
    # TODO: 实现核技巧距离
    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)

    k11 = kernel_func(x1, x1)[0, 0]
    k22 = kernel_func(x2, x2)[0, 0]
    k12 = kernel_func(x1, x2)[0, 0]

    distance_sq = k11 + k22 - 2 * k12

    return np.sqrt(max(0, distance_sq))


# ============================================================================
# 高级题目：核PCA和核方法应用
# ============================================================================

def kernel_pca(X, n_components=2, kernel='rbf', gamma=1.0):
    """
    高级 - 核主成分分析 (Kernel PCA)

    在特征空间中进行PCA
    步骤:
    1. 计算核矩阵K
    2. 中心化核矩阵
    3. 特征分解
    4. 提取主成分

    参数:
        X: n×d数据矩阵
        n_components: 主成分数
        kernel: 核类型
        gamma: RBF核参数

    返回:
        (变换后的数据, 特征值, 特征向量)
    """
    # TODO: 实现核PCA
    n = X.shape[0]

    # 计算核矩阵
    if kernel == 'rbf':
        K = rbf_kernel(X, gamma=gamma)
    elif kernel == 'linear':
        K = linear_kernel(X)
    elif kernel == 'poly':
        K = polynomial_kernel(X)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # 中心化
    K_centered = center_kernel_matrix(K)

    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    # 按特征值降序排序
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 提取前n_components个
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # 归一化特征向量
    for i in range(n_components):
        if eigenvalues[i] > 0:
            eigenvectors[:, i] = eigenvectors[:, i] / np.sqrt(eigenvalues[i])

    # 投影到主成分
    X_transformed = eigenvectors

    return X_transformed, eigenvalues, eigenvectors


def kernel_k_means(X, n_clusters=2, kernel='rbf', gamma=1.0, max_iter=100):
    """
    高级 - 核k-means聚类

    在特征空间中进行k-means

    参数:
        X: n×d数据矩阵
        n_clusters: 聚类数
        kernel: 核类型
        gamma: 核参数
        max_iter: 最大迭代次数

    返回:
        聚类标签
    """
    # TODO: 实现核k-means
    n = X.shape[0]

    # 计算核矩阵
    if kernel == 'rbf':
        K = rbf_kernel(X, gamma=gamma)
    else:
        K = linear_kernel(X)

    # 随机初始化标签
    np.random.seed(42)
    labels = np.random.randint(0, n_clusters, n)

    for iteration in range(max_iter):
        old_labels = labels.copy()

        # 更新标签
        for i in range(n):
            min_dist = np.inf
            best_cluster = 0

            for c in range(n_clusters):
                # 计算到聚类中心的距离（在特征空间）
                cluster_mask = (labels == c)
                if np.sum(cluster_mask) == 0:
                    continue

                n_c = np.sum(cluster_mask)

                # ||φ(x) - μ_c||² = k(x,x) - 2/n_c Σ k(x,x_i) + 1/n_c² Σ Σ k(x_i,x_j)
                dist = K[i, i]
                dist -= 2 * np.sum(K[i, cluster_mask]) / n_c
                dist += np.sum(K[np.ix_(cluster_mask, cluster_mask)]) / (n_c ** 2)

                if dist < min_dist:
                    min_dist = dist
                    best_cluster = c

            labels[i] = best_cluster

        # 检查收敛
        if np.all(labels == old_labels):
            break

    return labels


def gaussian_process_regression(X_train, y_train, X_test, kernel_func, noise_var=0.1):
    """
    高级 - 高斯过程回归

    预测分布:
    p(f*|X*, X, y) = N(f* | K*^T (K + σ²I)^(-1) y, K** - K*^T (K + σ²I)^(-1) K*)

    参数:
        X_train: 训练数据
        y_train: 训练标签
        X_test: 测试数据
        kernel_func: 核函数
        noise_var: 噪声方差σ²

    返回:
        (预测均值, 预测方差)
    """
    # TODO: 实现高斯过程回归
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # 计算核矩阵
    K = kernel_func(X_train, X_train)
    K_s = kernel_func(X_train, X_test)
    K_ss = kernel_func(X_test, X_test)

    # 加入噪声
    K_noisy = K + noise_var * np.eye(n_train)

    # 求解
    L = np.linalg.cholesky(K_noisy)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # 预测均值
    mean = K_s.T @ alpha

    # 预测方差
    v = np.linalg.solve(L, K_s)
    variance = np.diag(K_ss - v.T @ v)

    return mean, variance


def maximum_mean_discrepancy(X, Y, kernel_func):
    """
    高级 - 最大平均差异 (MMD)

    衡量两个分布的差异:
    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

    参数:
        X: 第一个分布的样本
        Y: 第二个分布的样本
        kernel_func: 核函数

    返回:
        MMD²估计值
    """
    # TODO: 实现MMD
    n = X.shape[0]
    m = Y.shape[0]

    # 计算核矩阵
    K_XX = kernel_func(X, X)
    K_YY = kernel_func(Y, Y)
    K_XY = kernel_func(X, Y)

    # MMD²估计
    term1 = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1))
    term2 = (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1))
    term3 = np.sum(K_XY) / (n * m)

    mmd_sq = term1 + term2 - 2 * term3

    return mmd_sq


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    np.random.seed(42)
    X = np.random.randn(10, 3)
    Y = np.random.randn(5, 3)

    print("\n" + "="*60)
    print("初级题目：核函数和核矩阵")
    print("="*60)

    print("\n测试 1.1: 线性核")
    K_linear = linear_kernel(X)
    # 线性核应该等于X @ X.T
    expected = X @ X.T
    results.append(v.assert_array_equal(K_linear, expected, rtol=1e-10, name="线性核"))

    print("\n测试 1.2: 多项式核")
    K_poly = polynomial_kernel(X, degree=2, coef0=1)
    # 检查形状
    results.append(K_poly.shape == (10, 10))
    print(f"  多项式核形状: {K_poly.shape}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 1.3: RBF核")
    K_rbf = rbf_kernel(X, gamma=0.5)
    # RBF核对角线应该全为1
    results.append(v.assert_array_equal(np.diag(K_rbf), np.ones(10), rtol=1e-10, name="RBF核对角线"))

    print("\n测试 1.4: 核矩阵正定性")
    is_pd = check_kernel_positive_definite(K_rbf)
    results.append(is_pd)
    print(f"  RBF核正定性: {'✓ 通过' if is_pd else '✗ 失败'}")

    print("\n" + "="*60)
    print("中级题目：核技巧和RKHS")
    print("="*60)

    print("\n测试 2.1: 核矩阵中心化")
    K_centered = center_kernel_matrix(K_rbf)
    # 中心化后行和列和应该接近0
    row_sums = np.sum(K_centered, axis=1)
    results.append(v.assert_array_equal(row_sums, np.zeros(10), atol=1e-10, name="中心化行和"))

    print("\n测试 2.2: 核对齐")
    K1 = rbf_kernel(X, gamma=0.5)
    K2 = rbf_kernel(X, gamma=1.0)
    alignment = kernel_alignment(K1, K2)
    # 对齐度应该在[0,1]之间
    results.append(0 <= alignment <= 1)
    print(f"  核对齐度: {alignment:.4f}")
    print(f"  {'✓ 通过（在[0,1]内）' if results[-1] else '✗ 失败'}")

    print("\n测试 2.3: 核岭回归")
    y = np.random.randn(10)
    alpha = kernel_ridge_regression(K_rbf, y, lambda_reg=0.1)
    # 预测训练数据
    y_pred = K_rbf @ alpha
    # 训练误差应该较小
    train_error = np.mean((y - y_pred)**2)
    results.append(train_error < 1.0)
    print(f"  训练MSE: {train_error:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：核PCA和核方法应用")
    print("="*60)

    print("\n测试 3.1: 核PCA")
    X_pca, eigenvals, eigenvecs = kernel_pca(X, n_components=2, kernel='rbf', gamma=0.5)
    # 检查输出形状
    results.append(X_pca.shape == (10, 2))
    print(f"  降维后形状: {X_pca.shape}")
    print(f"  前2个特征值: {eigenvals[:2]}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.2: 核k-means")
    labels = kernel_k_means(X, n_clusters=2, kernel='rbf', gamma=0.5, max_iter=20)
    # 应该有2个聚类
    unique_labels = np.unique(labels)
    results.append(len(unique_labels) == 2)
    print(f"  聚类标签: {labels}")
    print(f"  聚类数: {len(unique_labels)}")
    print(f"  {'✓ 通过（2个聚类）' if results[-1] else '✗ 失败'}")

    print("\n测试 3.3: 高斯过程回归")
    X_train = np.array([[1], [3], [5]])
    y_train = np.array([1, 3, 5])
    X_test = np.array([[2], [4]])

    kernel_gp = lambda X, Y=None: rbf_kernel(X, Y, gamma=1.0)
    mean, var = gaussian_process_regression(X_train, y_train, X_test, kernel_gp)

    # 在训练点附近的预测应该接近真实值
    results.append(len(mean) == 2 and len(var) == 2)
    print(f"  预测均值: {mean}")
    print(f"  预测方差: {var}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.4: 最大平均差异")
    X_dist1 = np.random.randn(20, 2)
    X_dist2 = np.random.randn(20, 2) + 0.5  # 稍微不同的分布

    kernel_mmd = lambda X, Y=None: rbf_kernel(X, Y, gamma=1.0)
    mmd = maximum_mean_discrepancy(X_dist1, X_dist2, kernel_mmd)

    # MMD应该为正（分布不同）
    results.append(mmd > 0)
    print(f"  MMD²: {mmd:.6f}")
    print(f"  {'✓ 通过（MMD>0）' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
