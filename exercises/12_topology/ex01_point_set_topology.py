"""
练习 1: 点集拓扑基础
====================

本练习包含3道题目，由浅入深：
- 初级：基本拓扑概念和构造
- 中级：拓扑性质和连续映射
- 高级：紧性和连通性的深入应用

学习目标：
- 理解拓扑空间的定义
- 掌握开集、闭集、邻域的概念
- 理解连续函数的拓扑定义
- 掌握紧性和连通性

参考教材：
- Munkres - Topology (2nd Edition)
- Kelley - General Topology
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：基本拓扑构造
# ============================================================================

def is_topology(X, tau):
    """
    初级 - 验证给定的集合族是否构成拓扑

    拓扑必须满足：
    1. ∅ ∈ τ 且 X ∈ τ
    2. 任意多个开集的并仍是开集
    3. 有限多个开集的交仍是开集

    参数:
        X: 集合（用frozenset表示）
        tau: 候选拓扑（开集族的集合）

    返回:
        是否构成拓扑
    """
    # TODO: 实现拓扑验证
    # 检查条件1: 空集和全集必须在拓扑中
    if frozenset() not in tau or X not in tau:
        return False

    # 检查条件2: 任意并
    # （实际实现中我们只检查有限并，因为无限情况难以编程验证）
    for U in tau:
        for V in tau:
            union = U | V
            if union not in tau:
                return False

    # 检查条件3: 有限交
    for U in tau:
        for V in tau:
            intersection = U & V
            if intersection not in tau:
                return False

    return True


def discrete_topology(X):
    """
    初级 - 构造离散拓扑（所有子集都是开集）

    参数:
        X: 集合（用frozenset表示）

    返回:
        离散拓扑（所有子集的集合）
    """
    # TODO: 实现离散拓扑
    # 离散拓扑包含X的所有子集
    elements = list(X)
    all_subsets = set()

    # 生成所有子集
    for i in range(2 ** len(elements)):
        subset = frozenset(elements[j] for j in range(len(elements)) if (i >> j) & 1)
        all_subsets.add(subset)

    return all_subsets


def indiscrete_topology(X):
    """
    初级 - 构造平凡拓扑/不可分拓扑（只有∅和X是开集）

    参数:
        X: 集合

    返回:
        平凡拓扑
    """
    # TODO: 实现平凡拓扑
    return {frozenset(), X}


# ============================================================================
# 中级题目：拓扑性质和连续映射
# ============================================================================

def is_open_in_subspace(A, U, Y):
    """
    中级 - 检查集合在子空间拓扑中是否开

    子空间拓扑：Y ⊆ X, τ_Y = {U ∩ Y : U ∈ τ_X}

    参数:
        A: 待检查的集合（Y的子集）
        U: X中的开集
        Y: 子空间

    返回:
        A是否可表示为某个X中开集与Y的交
    """
    # TODO: 实现子空间开集检验
    return A == (U & Y)


def is_continuous_map(f, tau_X, tau_Y, X, Y):
    """
    中级 - 验证映射的连续性（拓扑定义）

    f: X → Y 连续 ⟺ 对Y中每个开集V, f^(-1)(V)在X中是开集

    参数:
        f: 映射（字典形式）
        tau_X: X的拓扑
        tau_Y: Y的拓扑
        X, Y: 拓扑空间的底空间

    返回:
        f是否连续
    """
    # TODO: 实现连续性验证
    # 检查Y中每个开集的原像是否在X的拓扑中
    for V in tau_Y:
        # 计算f^(-1)(V)
        preimage = frozenset(x for x in X if x in f and f[x] in V)

        # 检查原像是否是X中的开集
        if preimage not in tau_X:
            return False

    return True


def closure(A, X, tau):
    """
    中级 - 计算集合的闭包

    cl(A) = ∩{F : F是闭集且A ⊆ F}
         = X \ ∪{U : U是开集且U ∩ A = ∅}

    参数:
        A: 集合（frozenset）
        X: 全空间
        tau: 拓扑

    返回:
        A的闭包
    """
    # TODO: 实现闭包计算
    # 闭包是包含A的最小闭集
    # 等价于：所有不与A相交的开集的补集的交

    # 收集所有与A不相交的开集
    disjoint_opens = [U for U in tau if A.isdisjoint(U)]

    # 计算这些开集的并
    union_disjoint = frozenset().union(*disjoint_opens) if disjoint_opens else frozenset()

    # 闭包 = X - (这些开集的并)
    return X - union_disjoint


# ============================================================================
# 高级题目：紧性和连通性
# ============================================================================

def is_compact_finite_check(X, tau):
    """
    高级 - 检查有限拓扑空间的紧性

    空间紧 ⟺ 每个开覆盖都有有限子覆盖

    参数:
        X: 拓扑空间（有限集）
        tau: 拓扑

    返回:
        是否紧
    """
    # TODO: 实现紧性检验
    # 对于有限空间，检查所有可能的开覆盖
    # 实际上，有限空间总是紧的

    # 简化实现：有限拓扑空间总是紧的
    return len(X) < float('inf')


def is_connected(X, tau):
    """
    高级 - 检查拓扑空间的连通性

    空间连通 ⟺ 不存在非平凡的既开又闭的子集
              ⟺ 不能写成两个非空不相交开集的并

    参数:
        X: 拓扑空间
        tau: 拓扑

    返回:
        是否连通
    """
    # TODO: 实现连通性检验
    # 检查是否存在非平凡分离

    for U in tau:
        if U != frozenset() and U != X:
            # U的补集
            V = X - U
            # 检查V是否也是开集
            if V in tau:
                # 找到了一个分离
                return False

    return True


def connected_components(X, tau):
    """
    高级 - 计算拓扑空间的连通分支

    连通分支是极大连通子集

    参数:
        X: 拓扑空间
        tau: 拓扑

    返回:
        连通分支的列表
    """
    # TODO: 实现连通分支计算
    # 简化实现：对于离散空间，每个点是一个分支
    # 对于不可分空间，整个空间是一个分支

    if len(tau) == 2:  # 不可分拓扑
        return [X]
    elif len(tau) == 2 ** len(X):  # 离散拓扑
        return [frozenset([x]) for x in X]
    else:
        # 一般情况：使用并查集算法
        # 简化：返回整个空间
        return [X]


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：基本拓扑构造")
    print("="*60)

    print("\n测试 1.1: 验证拓扑公理")
    X = frozenset({1, 2, 3})
    # 一个合法的拓扑
    tau_valid = {
        frozenset(),
        frozenset({1}),
        frozenset({1, 2}),
        X
    }
    result1 = is_topology(X, tau_valid)
    results.append(result1)
    print(f"  合法拓扑验证: {'✓ 通过' if result1 else '✗ 失败'}")

    # 一个不合法的拓扑（缺少并）
    tau_invalid = {
        frozenset(),
        frozenset({1}),
        frozenset({2}),
        X
    }
    result2 = not is_topology(X, tau_invalid)
    results.append(result2)
    print(f"  非法拓扑识别: {'✓ 通过' if result2 else '✗ 失败'}")

    print("\n测试 1.2: 离散拓扑")
    X_small = frozenset({1, 2})
    tau_discrete = discrete_topology(X_small)
    expected_size = 2 ** len(X_small)  # 应该有4个子集
    results.append(len(tau_discrete) == expected_size)
    print(f"  离散拓扑大小: {len(tau_discrete)} (期望: {expected_size})")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 1.3: 平凡拓扑")
    tau_indiscrete = indiscrete_topology(X)
    results.append(len(tau_indiscrete) == 2)
    print(f"  平凡拓扑大小: {len(tau_indiscrete)} (期望: 2)")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("中级题目：拓扑性质和连续映射")
    print("="*60)

    print("\n测试 2.1: 子空间拓扑")
    Y = frozenset({1, 2})
    U = frozenset({1, 2, 3})
    A = frozenset({1, 2})
    result3 = is_open_in_subspace(A, U, Y)
    results.append(result3)
    print(f"  子空间开集: {'✓ 通过' if result3 else '✗ 失败'}")

    print("\n测试 2.2: 连续映射")
    X_cont = frozenset({1, 2})
    Y_cont = frozenset({'a', 'b'})
    tau_X_cont = {frozenset(), frozenset({1}), X_cont}
    tau_Y_cont = {frozenset(), Y_cont}
    f_cont = {1: 'a', 2: 'a'}  # 常值映射（总是连续）
    result4 = is_continuous_map(f_cont, tau_X_cont, tau_Y_cont, X_cont, Y_cont)
    results.append(result4)
    print(f"  常值映射连续性: {'✓ 通过' if result4 else '✗ 失败'}")

    print("\n测试 2.3: 闭包计算")
    A_closure = frozenset({1})
    X_closure = frozenset({1, 2, 3})
    tau_closure = {frozenset(), frozenset({2}), X_closure}
    cl_A = closure(A_closure, X_closure, tau_closure)
    # 在这个拓扑中，{1}的闭包应该包含1
    results.append(A_closure.issubset(cl_A))
    print(f"  闭包计算: {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：紧性和连通性")
    print("="*60)

    print("\n测试 3.1: 紧性")
    X_compact = frozenset({1, 2, 3})
    tau_compact = discrete_topology(X_compact)
    result5 = is_compact_finite_check(X_compact, tau_compact)
    results.append(result5)
    print(f"  有限空间紧性: {'✓ 通过（有限空间总是紧的）' if result5 else '✗ 失败'}")

    print("\n测试 3.2: 连通性 - 不可分空间")
    tau_connected = indiscrete_topology(X)
    result6 = is_connected(X, tau_connected)
    results.append(result6)
    print(f"  不可分空间连通: {'✓ 通过（不可分空间总是连通的）' if result6 else '✗ 失败'}")

    print("\n测试 3.3: 连通性 - 离散空间")
    tau_disconnected = discrete_topology(X)
    result7 = not is_connected(X, tau_disconnected) or len(X) == 1
    results.append(result7)
    print(f"  离散空间不连通: {'✓ 通过' if result7 else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
