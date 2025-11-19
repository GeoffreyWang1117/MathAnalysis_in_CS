"""
练习 2: 谱图理论
================

本练习包含3道题目，由浅入深：
- 初级：图拉普拉斯矩阵和谱性质
- 中级：谱聚类和图切割
- 高级：图卷积神经网络(GCN)的数学基础

学习目标：
- 理解图的谱分解
- 掌握谱聚类算法
- 理解Cheeger不等式和图切割
- 理解图卷积网络的数学原理

应用领域：
- 图神经网络 (GNN, GCN)
- 谱聚类 (Spectral Clustering)
- 社区发现 (Community Detection)
- 图信号处理

参考教材：
- Chung - Spectral Graph Theory
- Von Luxburg - A Tutorial on Spectral Clustering
- Kipf & Welling - Semi-Supervised Classification with GCNs
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：图拉普拉斯矩阵和谱性质
# ============================================================================

def graph_laplacian_unnormalized(adj_matrix):
    """
    初级 - 非归一化图拉普拉斯矩阵

    L = D - A
    其中 D 是度矩阵, A 是邻接矩阵

    性质:
    - L是半正定的
    - L的最小特征值是0
    - 0特征值的重数 = 连通分量数

    参数:
        adj_matrix: n×n邻接矩阵

    返回:
        拉普拉斯矩阵 L
    """
    # TODO: 实现非归一化拉普拉斯
    degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(degrees)
    L = D - adj_matrix
    return L


def graph_laplacian_normalized_symmetric(adj_matrix):
    """
    初级 - 对称归一化图拉普拉斯矩阵

    L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2)

    这是图卷积网络中常用的形式

    参数:
        adj_matrix: n×n邻接矩阵

    返回:
        对称归一化拉普拉斯矩阵
    """
    # TODO: 实现对称归一化拉普拉斯
    degrees = np.sum(adj_matrix, axis=1)

    # 避免除以0
    degrees[degrees == 0] = 1

    # D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    # L_sym = I - D^(-1/2) A D^(-1/2)
    L_sym = np.eye(len(degrees)) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    return L_sym


def graph_laplacian_normalized_random_walk(adj_matrix):
    """
    初级 - 随机游走归一化拉普拉斯

    L_rw = D^(-1) L = I - D^(-1) A

    其特征向量与对称拉普拉斯相关:
    L_rw和L_sym有相同的特征值

    参数:
        adj_matrix: n×n邻接矩阵

    返回:
        随机游走拉普拉斯矩阵
    """
    # TODO: 实现随机游走拉普拉斯
    degrees = np.sum(adj_matrix, axis=1)
    degrees[degrees == 0] = 1

    D_inv = np.diag(1.0 / degrees)

    L_rw = np.eye(len(degrees)) - D_inv @ adj_matrix

    return L_rw


def compute_spectrum(laplacian, k=None):
    """
    初级 - 计算图拉普拉斯的谱

    谱(spectrum): 特征值和特征向量的集合

    参数:
        laplacian: 拉普拉斯矩阵
        k: 要计算的最小k个特征值（None表示全部）

    返回:
        (eigenvalues, eigenvectors)元组
    """
    # TODO: 实现谱计算
    if k is None or k >= laplacian.shape[0]:
        # 计算所有特征值
        eigenvalues, eigenvectors = eigh(laplacian)
    else:
        # 只计算最小的k个
        eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')

    # 按特征值排序
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# ============================================================================
# 中级题目：谱聚类和图切割
# ============================================================================

def fiedler_vector(adj_matrix):
    """
    中级 - 计算Fiedler向量

    Fiedler向量: 第二小特征值对应的特征向量
    (第一小特征值=0对应全1向量)

    用途:
    - 图的二分
    - 连通性度量(代数连通度)

    参数:
        adj_matrix: n×n邻接矩阵

    返回:
        Fiedler向量
    """
    # TODO: 实现Fiedler向量计算
    L = graph_laplacian_unnormalized(adj_matrix)
    eigenvalues, eigenvectors = compute_spectrum(L, k=2)

    # 第二个特征向量（索引1）
    fiedler_vec = eigenvectors[:, 1]

    return fiedler_vec


def spectral_clustering(adj_matrix, k, normalized=True):
    """
    中级 - 谱聚类算法

    步骤:
    1. 计算拉普拉斯矩阵L
    2. 计算L的前k个最小特征向量
    3. 对特征向量矩阵的行进行k-means聚类

    参数:
        adj_matrix: n×n邻接矩阵
        k: 聚类数
        normalized: 是否使用归一化拉普拉斯

    返回:
        聚类标签数组
    """
    # TODO: 实现谱聚类
    # 选择拉普拉斯类型
    if normalized:
        L = graph_laplacian_normalized_symmetric(adj_matrix)
    else:
        L = graph_laplacian_unnormalized(adj_matrix)

    # 计算前k个特征向量
    eigenvalues, eigenvectors = compute_spectrum(L, k=k)

    # 提取前k个特征向量作为特征矩阵
    features = eigenvectors[:, :k]

    # 归一化特征向量（行归一化）
    row_norms = np.linalg.norm(features, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    features = features / row_norms

    # 简单k-means聚类
    labels = kmeans_simple(features, k)

    return labels


def kmeans_simple(X, k, max_iter=100):
    """
    中级 - 简单k-means实现

    参数:
        X: n×d特征矩阵
        k: 聚类数
        max_iter: 最大迭代次数

    返回:
        聚类标签
    """
    # TODO: 实现简单k-means
    n, d = X.shape

    # 随机初始化中心
    np.random.seed(42)
    centers = X[np.random.choice(n, k, replace=False)]

    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iter):
        # 分配标签（最近中心）
        distances = np.zeros((n, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)

        new_labels = np.argmin(distances, axis=1)

        # 检查收敛
        if np.all(labels == new_labels):
            break

        labels = new_labels

        # 更新中心
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)

    return labels


def graph_cut_size(adj_matrix, partition):
    """
    中级 - 计算图切割大小

    切割大小: 跨分区的边数
    cut(S, S̄) = Σ_{i∈S, j∈S̄} A[i,j]

    参数:
        adj_matrix: n×n邻接矩阵
        partition: 二分区标签(0或1)

    返回:
        切割大小
    """
    # TODO: 实现图切割计算
    cut_size = 0
    n = len(partition)

    for i in range(n):
        for j in range(n):
            if partition[i] != partition[j]:
                cut_size += adj_matrix[i, j]

    # 每条边被计算两次
    return cut_size / 2


def normalized_cut(adj_matrix, partition):
    """
    中级 - 归一化切割(Normalized Cut)

    Ncut(S, S̄) = cut(S, S̄)/vol(S) + cut(S, S̄)/vol(S̄)

    其中 vol(S) = Σ_{i∈S} deg(i)

    参数:
        adj_matrix: n×n邻接矩阵
        partition: 二分区标签

    返回:
        归一化切割值
    """
    # TODO: 实现归一化切割
    cut = graph_cut_size(adj_matrix, partition)

    # 计算体积
    degrees = np.sum(adj_matrix, axis=1)

    vol_S = np.sum(degrees[partition == 0])
    vol_Sbar = np.sum(degrees[partition == 1])

    if vol_S == 0 or vol_Sbar == 0:
        return np.inf

    ncut = cut / vol_S + cut / vol_Sbar

    return ncut


# ============================================================================
# 高级题目：图卷积神经网络的数学基础
# ============================================================================

def gcn_propagation_matrix(adj_matrix, add_self_loops=True):
    """
    高级 - GCN传播矩阵

    GCN的核心传播公式:
    H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

    其中:
    - Ã = A + I (添加自环)
    - D̃ 是Ã的度矩阵

    参数:
        adj_matrix: n×n邻接矩阵
        add_self_loops: 是否添加自环

    返回:
        传播矩阵 D̃^(-1/2) Ã D̃^(-1/2)
    """
    # TODO: 实现GCN传播矩阵
    if add_self_loops:
        # 添加自环
        A_tilde = adj_matrix + np.eye(adj_matrix.shape[0])
    else:
        A_tilde = adj_matrix

    # 计算度矩阵
    degrees = np.sum(A_tilde, axis=1)
    degrees[degrees == 0] = 1  # 避免除以0

    # D̃^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    # D̃^(-1/2) Ã D̃^(-1/2)
    prop_matrix = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return prop_matrix


def graph_convolution_layer(features, adj_matrix, weights, add_self_loops=True):
    """
    高级 - 图卷积层

    实现单层GCN:
    H' = ReLU(D̃^(-1/2) Ã D̃^(-1/2) H W)

    参数:
        features: n×d_in 节点特征矩阵
        adj_matrix: n×n 邻接矩阵
        weights: d_in×d_out 权重矩阵
        add_self_loops: 是否添加自环

    返回:
        n×d_out 输出特征矩阵
    """
    # TODO: 实现图卷积层
    # 获取传播矩阵
    prop_matrix = gcn_propagation_matrix(adj_matrix, add_self_loops)

    # 传播和变换
    output = prop_matrix @ features @ weights

    # 激活函数 (ReLU)
    output = np.maximum(0, output)

    return output


def cheeger_inequality(adj_matrix):
    """
    高级 - Cheeger不等式

    关联图的代数性质(第二小特征值λ₂)和
    组合性质(Cheeger常数h)

    λ₂ / 2 ≤ h ≤ √(2λ₂)

    其中:
    - λ₂: Fiedler值（拉普拉斯第二小特征值）
    - h: Cheeger常数（最小归一化切割）

    参数:
        adj_matrix: n×n邻接矩阵

    返回:
        (lambda_2, cheeger_lower_bound, cheeger_upper_bound)
    """
    # TODO: 实现Cheeger不等式验证
    L = graph_laplacian_normalized_symmetric(adj_matrix)
    eigenvalues, _ = compute_spectrum(L, k=2)

    lambda_2 = eigenvalues[1]

    # Cheeger不等式的界
    cheeger_lower = lambda_2 / 2
    cheeger_upper = np.sqrt(2 * lambda_2)

    return lambda_2, cheeger_lower, cheeger_upper


def graph_fourier_transform(signal, laplacian):
    """
    高级 - 图傅里叶变换

    图信号的傅里叶变换:
    ŝ = U^T s

    其中U是拉普拉斯特征向量矩阵

    逆变换:
    s = U ŝ

    参数:
        signal: n维图信号
        laplacian: n×n拉普拉斯矩阵

    返回:
        傅里叶系数
    """
    # TODO: 实现图傅里叶变换
    _, U = compute_spectrum(laplacian)

    # 傅里叶变换: ŝ = U^T s
    fourier_coeffs = U.T @ signal

    return fourier_coeffs


def graph_filter_spectral(signal, laplacian, filter_func):
    """
    高级 - 谱域图滤波

    图滤波在谱域的定义:
    g_θ(L) * s = U g_θ(Λ) U^T s

    其中:
    - L = U Λ U^T (拉普拉斯特征分解)
    - g_θ(Λ) 是对角矩阵，对角元素是g_θ(λᵢ)

    这是图卷积的一般形式

    参数:
        signal: n维图信号
        laplacian: n×n拉普拉斯矩阵
        filter_func: 滤波器函数 λ -> g(λ)

    返回:
        滤波后的信号
    """
    # TODO: 实现谱域图滤波
    eigenvalues, U = compute_spectrum(laplacian)

    # 应用滤波器到特征值
    filtered_eigenvalues = np.array([filter_func(lam) for lam in eigenvalues])

    # g_θ(L) = U g_θ(Λ) U^T
    # 滤波: y = U g_θ(Λ) U^T s
    fourier_coeffs = U.T @ signal
    filtered_coeffs = filtered_eigenvalues * fourier_coeffs
    filtered_signal = U @ filtered_coeffs

    return filtered_signal


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 创建测试图
    # 0 -- 1    3 -- 4
    #  \  /      \  /
    #   2         5
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])

    print("\n" + "="*60)
    print("初级题目：图拉普拉斯矩阵和谱性质")
    print("="*60)

    print("\n测试 1.1: 非归一化拉普拉斯")
    L = graph_laplacian_unnormalized(adj_matrix)
    # 检查行和为0
    row_sums = np.sum(L, axis=1)
    results.append(v.assert_array_equal(row_sums, np.zeros(6), atol=1e-10, name="拉普拉斯行和"))

    print("\n测试 1.2: 对称归一化拉普拉斯")
    L_sym = graph_laplacian_normalized_symmetric(adj_matrix)
    # 应该是对称的
    results.append(v.assert_array_equal(L_sym, L_sym.T, atol=1e-10, name="对称性"))

    print("\n测试 1.3: 拉普拉斯谱")
    eigenvalues, eigenvectors = compute_spectrum(L)
    # 第一个特征值应该接近0（图有2个连通分量）
    results.append(v.assert_close(eigenvalues[0], 0.0, atol=1e-10, name="最小特征值"))
    # 第二个特征值也应该接近0（2个连通分量）
    results.append(v.assert_close(eigenvalues[1], 0.0, atol=1e-10, name="第二小特征值"))

    print("\n" + "="*60)
    print("中级题目：谱聚类和图切割")
    print("="*60)

    print("\n测试 2.1: Fiedler向量")
    fiedler = fiedler_vector(adj_matrix)
    results.append(len(fiedler) == 6)
    print(f"  Fiedler向量: {fiedler}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 2.2: 谱聚类")
    labels = spectral_clustering(adj_matrix, k=2)
    # 应该将图分成两个聚类
    unique_labels = np.unique(labels)
    results.append(len(unique_labels) == 2)
    print(f"  聚类标签: {labels}")
    print(f"  {'✓ 通过（找到2个聚类）' if results[-1] else '✗ 失败'}")

    print("\n测试 2.3: 图切割")
    # 手动创建二分区
    partition = np.array([0, 0, 0, 1, 1, 1])
    cut = graph_cut_size(adj_matrix, partition)
    # 两个连通分量之间没有边，切割应该是0
    results.append(v.assert_close(cut, 0.0, name="图切割大小"))

    print("\n测试 2.4: 归一化切割")
    ncut = normalized_cut(adj_matrix, partition)
    results.append(ncut >= 0)
    print(f"  归一化切割: {ncut:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：图卷积神经网络的数学基础")
    print("="*60)

    print("\n测试 3.1: GCN传播矩阵")
    prop_matrix = gcn_propagation_matrix(adj_matrix)
    # 传播矩阵应该是对称的
    results.append(v.assert_array_equal(prop_matrix, prop_matrix.T, rtol=1e-10, name="GCN传播矩阵对称性"))

    print("\n测试 3.2: 图卷积层")
    features = np.random.randn(6, 4)
    weights = np.random.randn(4, 8)
    output = graph_convolution_layer(features, adj_matrix, weights)
    # 输出形状应该是 (6, 8)
    results.append(output.shape == (6, 8))
    print(f"  输出形状: {output.shape}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.3: Cheeger不等式")
    lambda_2, lower, upper = cheeger_inequality(adj_matrix)
    results.append(lower <= upper)
    print(f"  λ₂ = {lambda_2:.4f}")
    print(f"  Cheeger界: [{lower:.4f}, {upper:.4f}]")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.4: 图傅里叶变换")
    signal = np.random.randn(6)
    L = graph_laplacian_normalized_symmetric(adj_matrix)
    fourier_coeffs = graph_fourier_transform(signal, L)
    # 逆变换应该恢复原信号
    _, U = compute_spectrum(L)
    recovered = U @ fourier_coeffs
    results.append(v.assert_array_equal(signal, recovered, rtol=1e-10, name="图傅里叶变换逆变换"))

    print("\n测试 3.5: 谱域图滤波")
    # 低通滤波器
    filter_func = lambda lam: np.exp(-lam)
    filtered = graph_filter_spectral(signal, L, filter_func)
    results.append(len(filtered) == 6)
    print(f"  原信号能量: {np.linalg.norm(signal):.4f}")
    print(f"  滤波后能量: {np.linalg.norm(filtered):.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
