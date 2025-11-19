"""
练习 1: 张量运算
================

本练习包含3道题目，由浅入深：
- 初级：张量的基本操作和索引
- 中级：张量缩并和Einstein求和约定
- 高级：张量分解和Tucker分解

学习目标：
- 理解张量的定义和表示
- 掌握张量缩并(contraction)和Einstein求和
- 实现张量分解算法
- 理解深度学习框架中的张量操作

应用领域：
- 深度学习框架 (PyTorch, TensorFlow)
- 多线性代数
- 物理学张量分析
- 推荐系统 (张量分解)

参考教材：
- Kolda & Bader - Tensor Decompositions and Applications
- Cichocki et al. - Tensor Decompositions for Signal Processing
- Goodfellow et al. - Deep Learning
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：张量的基本操作和索引
# ============================================================================

def tensor_reshape(tensor, new_shape):
    """
    初级 - 张量重塑

    改变张量的形状，但保持元素总数不变
    类似于 NumPy 的 reshape

    参数:
        tensor: 输入张量
        new_shape: 新形状元组

    返回:
        重塑后的张量
    """
    # TODO: 实现张量重塑
    return np.reshape(tensor, new_shape)


def tensor_transpose(tensor, axes):
    """
    初级 - 张量转置

    重新排列张量的维度
    例如: (2, 3, 4) 按 axes=(2, 0, 1) -> (4, 2, 3)

    参数:
        tensor: 输入张量
        axes: 维度排列顺序

    返回:
        转置后的张量
    """
    # TODO: 实现张量转置
    return np.transpose(tensor, axes)


def tensor_outer_product(a, b):
    """
    初级 - 张量外积

    计算两个张量的外积(outer product)
    结果的形状是两个张量形状的拼接

    例如: (2, 3) ⊗ (4, 5) -> (2, 3, 4, 5)

    参数:
        a: 第一个张量
        b: 第二个张量

    返回:
        外积张量
    """
    # TODO: 实现张量外积
    # 使用 numpy 的外积
    result = np.multiply.outer(a, b)
    return result


def tensor_slice(tensor, indices):
    """
    初级 - 张量切片

    根据给定的索引切片张量
    支持花式索引和切片

    参数:
        tensor: 输入张量
        indices: 索引元组

    返回:
        切片后的张量
    """
    # TODO: 实现张量切片
    return tensor[indices]


# ============================================================================
# 中级题目：张量缩并和Einstein求和约定
# ============================================================================

def tensor_contraction(A, B, axes):
    """
    中级 - 张量缩并(Tensor Contraction)

    沿指定轴收缩两个张量
    这是矩阵乘法的高维推广

    例如: C[i,j,k,l] = Σ_m A[i,m,j] * B[m,k,l]

    参数:
        A: 第一个张量
        B: 第二个张量
        axes: 收缩的轴对 (A_axis, B_axis)

    返回:
        收缩后的张量
    """
    # TODO: 实现张量缩并
    # 使用 numpy 的 tensordot
    return np.tensordot(A, B, axes=axes)


def einstein_summation(subscripts, *operands):
    """
    中级 - Einstein求和约定

    实现Einstein求和约定
    例如:
    - 'ij,jk->ik': 矩阵乘法
    - 'ii->i': 对角元素
    - 'ij->ji': 转置
    - 'ij,ij->': 矩阵内积

    参数:
        subscripts: 下标字符串
        operands: 操作数张量

    返回:
        结果张量
    """
    # TODO: 实现Einstein求和
    # 使用 numpy 的 einsum
    return np.einsum(subscripts, *operands)


def mode_n_product(tensor, matrix, mode):
    """
    中级 - 模式n乘积(Mode-n Product)

    张量在第n个模式与矩阵相乘
    用于Tucker分解

    定义: (X ×_n U)[i₁,...,i_{n-1},j,i_{n+1},...,i_N]
          = Σ_i_n X[i₁,...,i_n,...,i_N] * U[j,i_n]

    参数:
        tensor: 输入张量
        matrix: 矩阵
        mode: 模式索引 (从0开始)

    返回:
        模式n乘积结果
    """
    # TODO: 实现模式n乘积
    # 将mode移到第一维
    tensor_moved = np.moveaxis(tensor, mode, 0)

    # 展开为矩阵
    shape = tensor_moved.shape
    tensor_matrix = tensor_moved.reshape(shape[0], -1)

    # 矩阵乘法
    result_matrix = matrix @ tensor_matrix

    # 恢复形状
    new_shape = (matrix.shape[0],) + shape[1:]
    result = result_matrix.reshape(new_shape)

    # 将mode移回原位置
    result = np.moveaxis(result, 0, mode)

    return result


def tensor_trace(tensor, axis1=0, axis2=1):
    """
    中级 - 张量迹

    计算张量在两个轴上的迹(对角和)

    参数:
        tensor: 输入张量
        axis1, axis2: 要计算迹的两个轴

    返回:
        迹（降低2维后的张量）
    """
    # TODO: 实现张量迹
    return np.trace(tensor, axis1=axis1, axis2=axis2)


# ============================================================================
# 高级题目：张量分解
# ============================================================================

def tensor_unfold(tensor, mode):
    """
    高级 - 张量展开(Tensor Unfolding/Matricization)

    将张量沿某个模式展开成矩阵
    用于Tucker分解和HOSVD

    例如: 3维张量 (I, J, K)
    - mode 0: (I, J*K)
    - mode 1: (J, I*K)
    - mode 2: (K, I*J)

    参数:
        tensor: 输入张量
        mode: 展开模式

    返回:
        展开后的矩阵
    """
    # TODO: 实现张量展开
    # 将mode移到第一维
    tensor_moved = np.moveaxis(tensor, mode, 0)

    # 展开为矩阵
    shape = tensor_moved.shape
    matrix = tensor_moved.reshape(shape[0], -1)

    return matrix


def tensor_fold(matrix, mode, original_shape):
    """
    高级 - 张量折叠(逆展开)

    将矩阵折叠回张量

    参数:
        matrix: 展开的矩阵
        mode: 展开时的模式
        original_shape: 原始张量形状

    返回:
        折叠后的张量
    """
    # TODO: 实现张量折叠
    # 计算折叠后的形状
    shape_list = list(original_shape)
    mode_size = shape_list.pop(mode)

    # 重塑为张量
    new_shape = [mode_size] + shape_list
    tensor_moved = matrix.reshape(new_shape)

    # 将mode移回原位置
    tensor = np.moveaxis(tensor_moved, 0, mode)

    return tensor


def hosvd(tensor, ranks=None):
    """
    高级 - 高阶奇异值分解(HOSVD/Tucker分解)

    将张量分解为核心张量和因子矩阵:
    X ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃

    这是SVD的高维推广

    参数:
        tensor: 输入张量
        ranks: 各模式的秩（None表示完整秩）

    返回:
        (core_tensor, factor_matrices) 元组
    """
    # TODO: 实现HOSVD
    ndims = tensor.ndim
    factor_matrices = []

    # 对每个模式做SVD
    for mode in range(ndims):
        # 展开张量
        unfolded = tensor_unfold(tensor, mode)

        # SVD
        U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)

        # 截断到指定秩
        if ranks is not None and ranks[mode] < U.shape[1]:
            U = U[:, :ranks[mode]]

        factor_matrices.append(U)

    # 计算核心张量
    core = tensor.copy()
    for mode, U in enumerate(factor_matrices):
        core = mode_n_product(core, U.T, mode)

    return core, factor_matrices


def cp_decomposition_als(tensor, rank, max_iter=100, tol=1e-6):
    """
    高级 - CP分解(CANDECOMP/PARAFAC)

    将张量分解为秩1张量的和:
    X ≈ Σᵣ λᵣ (a_r ⊗ b_r ⊗ c_r)

    使用交替最小二乘(ALS)算法

    参数:
        tensor: 输入3维张量
        rank: CP秩
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        (weights, factors) - 权重和因子矩阵列表
    """
    # TODO: 实现CP分解 (简化版)

    # 初始化因子矩阵
    I, J, K = tensor.shape
    A = np.random.rand(I, rank)
    B = np.random.rand(J, rank)
    C = np.random.rand(K, rank)

    # ALS迭代
    for iteration in range(max_iter):
        # 更新A
        X1 = tensor_unfold(tensor, 0)
        V = np.linalg.pinv(np.multiply.outer(B.T @ B, C.T @ C).reshape(rank, rank))
        A = X1 @ np.kron(C, B) @ V

        # 更新B
        X2 = tensor_unfold(tensor, 1)
        V = np.linalg.pinv(np.multiply.outer(A.T @ A, C.T @ C).reshape(rank, rank))
        B = X2 @ np.kron(C, A) @ V

        # 更新C
        X3 = tensor_unfold(tensor, 2)
        V = np.linalg.pinv(np.multiply.outer(A.T @ A, B.T @ B).reshape(rank, rank))
        C = X3 @ np.kron(B, A) @ V

        # 简化：这里应该检查收敛性
        # 实际应该计算重构误差

    # 归一化并提取权重
    weights = np.ones(rank)
    for r in range(rank):
        norm_a = np.linalg.norm(A[:, r])
        norm_b = np.linalg.norm(B[:, r])
        norm_c = np.linalg.norm(C[:, r])

        weights[r] = norm_a * norm_b * norm_c

        if norm_a > 0:
            A[:, r] /= norm_a
        if norm_b > 0:
            B[:, r] /= norm_b
        if norm_c > 0:
            C[:, r] /= norm_c

    return weights, [A, B, C]


def tensor_reconstruction_error(original, core, factors):
    """
    高级 - 张量重构误差

    计算Tucker分解的重构误差:
    ||X - X_reconstructed||_F / ||X||_F

    参数:
        original: 原始张量
        core: 核心张量
        factors: 因子矩阵列表

    返回:
        相对误差
    """
    # TODO: 实现重构误差计算
    # 重构张量
    reconstructed = core.copy()
    for mode, U in enumerate(factors):
        reconstructed = mode_n_product(reconstructed, U, mode)

    # 计算Frobenius范数
    error = np.linalg.norm(original - reconstructed)
    original_norm = np.linalg.norm(original)

    relative_error = error / original_norm if original_norm > 0 else 0

    return relative_error


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：张量的基本操作和索引")
    print("="*60)

    print("\n测试 1.1: 张量重塑")
    tensor = np.arange(24).reshape(2, 3, 4)
    reshaped = tensor_reshape(tensor, (6, 4))
    results.append(reshaped.shape == (6, 4))
    print(f"  原形状: {tensor.shape}, 新形状: {reshaped.shape}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 1.2: 张量转置")
    transposed = tensor_transpose(tensor, (2, 0, 1))
    results.append(transposed.shape == (4, 2, 3))
    print(f"  转置后形状: {transposed.shape}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 1.3: 张量外积")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    outer = tensor_outer_product(a, b)
    results.append(outer.shape == (2, 2, 2))
    print(f"  外积形状: {outer.shape}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("中级题目：张量缩并和Einstein求和约定")
    print("="*60)

    print("\n测试 2.1: 张量缩并")
    A = np.random.randn(2, 3, 4)
    B = np.random.randn(4, 5, 6)
    C = tensor_contraction(A, B, axes=([2], [0]))
    # 形状应该是 (2, 3, 5, 6)
    results.append(C.shape == (2, 3, 5, 6))
    print(f"  缩并后形状: {C.shape}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 2.2: Einstein求和 - 矩阵乘法")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = einstein_summation('ij,jk->ik', A, B)
    expected = A @ B
    results.append(v.assert_array_equal(C, expected, name="Einstein求和(矩阵乘法)"))

    print("\n测试 2.3: 模式n乘积")
    tensor = np.random.randn(3, 4, 5)
    matrix = np.random.randn(2, 4)
    result = mode_n_product(tensor, matrix, mode=1)
    # 形状应该是 (3, 2, 5)
    results.append(result.shape == (3, 2, 5))
    print(f"  模式1乘积后形状: {result.shape}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：张量分解")
    print("="*60)

    print("\n测试 3.1: 张量展开和折叠")
    tensor = np.random.randn(3, 4, 5)
    unfolded = tensor_unfold(tensor, mode=1)
    folded = tensor_fold(unfolded, mode=1, original_shape=(3, 4, 5))

    results.append(v.assert_array_equal(tensor, folded, name="张量展开-折叠恢复"))

    print("\n测试 3.2: HOSVD分解")
    np.random.seed(42)
    tensor = np.random.randn(4, 5, 6)
    core, factors = hosvd(tensor, ranks=[2, 3, 3])

    # 重构误差
    error = tensor_reconstruction_error(tensor, core, factors)
    results.append(error < 1.0)  # 应该有一定的近似
    print(f"  核心张量形状: {core.shape}")
    print(f"  因子矩阵形状: {[f.shape for f in factors]}")
    print(f"  重构误差: {error:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.3: CP分解")
    np.random.seed(42)
    # 创建低秩张量
    rank = 2
    A = np.random.randn(3, rank)
    B = np.random.randn(4, rank)
    C = np.random.randn(5, rank)

    # 构造秩为2的张量
    tensor = np.zeros((3, 4, 5))
    for r in range(rank):
        tensor += np.outer(A[:, r], np.outer(B[:, r], C[:, r]).ravel()).reshape(3, 4, 5)

    # CP分解
    weights, factors_cp = cp_decomposition_als(tensor, rank=rank, max_iter=50)

    results.append(len(factors_cp) == 3 and factors_cp[0].shape == (3, rank))
    print(f"  CP因子形状: {[f.shape for f in factors_cp]}")
    print(f"  权重: {weights}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
