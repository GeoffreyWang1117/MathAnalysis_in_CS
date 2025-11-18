"""
练习 1: 熵与信息论基础
======================

学习目标：
- 理解Shannon熵的概念
- 掌握互信息和条件熵
- 计算KL散度
- 应用信息论于机器学习

任务：
实现信息论的核心概念
"""

import numpy as np
from scipy.special import xlogy
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def shannon_entropy(p):
    """
    Shannon熵（信息熵）

    H(X) = -Σ p(x) log₂ p(x)

    参数:
        p: 概率分布（归一化的）

    返回:
        熵值（单位：bits）
    """
    # TODO: 实现Shannon熵
    p = np.array(p)
    # 移除零概率（0 * log(0) = 0）
    p = p[p > 0]
    # 使用log2计算熵（单位：bits）
    entropy = -np.sum(p * np.log2(p))
    return entropy


def joint_entropy(p_xy):
    """
    联合熵

    H(X, Y) = -Σ Σ p(x,y) log₂ p(x,y)

    参数:
        p_xy: 联合概率分布（二维数组）
    """
    # TODO: 实现联合熵
    p_xy = np.array(p_xy)
    p_xy = p_xy[p_xy > 0]
    return -np.sum(p_xy * np.log2(p_xy))


def conditional_entropy(p_xy):
    """
    条件熵

    H(Y|X) = H(X,Y) - H(X)

    参数:
        p_xy: 联合概率分布
    """
    # TODO: 实现条件熵
    p_xy = np.array(p_xy)

    # 计算边缘分布 p(x)
    p_x = np.sum(p_xy, axis=1)

    # H(X, Y)
    h_xy = joint_entropy(p_xy)

    # H(X)
    h_x = shannon_entropy(p_x)

    # H(Y|X) = H(X,Y) - H(X)
    return h_xy - h_x


def mutual_information(p_xy):
    """
    互信息

    I(X; Y) = H(X) + H(Y) - H(X, Y)
    或 I(X; Y) = H(X) - H(X|Y)

    衡量两个变量的相互依赖程度

    参数:
        p_xy: 联合概率分布
    """
    # TODO: 实现互信息
    p_xy = np.array(p_xy)

    # 边缘分布
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    # I(X; Y) = H(X) + H(Y) - H(X, Y)
    mi = shannon_entropy(p_x) + shannon_entropy(p_y) - joint_entropy(p_xy)

    return mi


def kl_divergence(p, q):
    """
    Kullback-Leibler散度（相对熵）

    D_KL(P||Q) = Σ p(x) log(p(x)/q(x))

    衡量两个概率分布的差异（非对称）

    参数:
        p: 真实分布
        q: 近似分布
    """
    # TODO: 实现KL散度
    p = np.array(p)
    q = np.array(q)

    # 只在 p > 0 的地方计算
    # 如果 p > 0 但 q = 0，KL散度为无穷
    mask = p > 0
    if np.any((p > 0) & (q == 0)):
        return np.inf

    # D_KL(P||Q) = Σ p log(p/q) = Σ p log p - Σ p log q
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return kl


def js_divergence(p, q):
    """
    Jensen-Shannon散度（对称版本的KL散度）

    D_JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
    其中 M = 0.5 * (P + Q)

    参数:
        p, q: 两个概率分布
    """
    # TODO: 实现JS散度
    p = np.array(p)
    q = np.array(q)

    # 混合分布
    m = 0.5 * (p + q)

    # JS散度
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return js


def cross_entropy(p, q):
    """
    交叉熵

    H(P, Q) = -Σ p(x) log q(x)

    常用于机器学习中的损失函数

    参数:
        p: 真实分布
        q: 预测分布
    """
    # TODO: 实现交叉熵
    p = np.array(p)
    q = np.array(q)

    # H(P, Q) = -Σ p log q
    ce = -np.sum(p * np.log(q + 1e-10))  # 加小量避免log(0)

    return ce


def perplexity(p):
    """
    困惑度

    Perplexity = 2^H(P)

    衡量概率模型的质量

    参数:
        p: 概率分布
    """
    # TODO: 实现困惑度
    h = shannon_entropy(p)
    return 2 ** h


def gini_impurity(p):
    """
    基尼不纯度（决策树中使用）

    Gini = 1 - Σ p(x)²

    参数:
        p: 类别概率分布
    """
    # TODO: 实现基尼不纯度
    p = np.array(p)
    return 1 - np.sum(p ** 2)


def information_gain(parent_entropy, child_entropies, child_weights):
    """
    信息增益（决策树分裂准则）

    IG = H(parent) - Σ w_i * H(child_i)

    参数:
        parent_entropy: 父节点熵
        child_entropies: 子节点熵列表
        child_weights: 子节点权重（样本比例）
    """
    # TODO: 实现信息增益
    weighted_child_entropy = np.sum(
        np.array(child_weights) * np.array(child_entropies)
    )
    return parent_entropy - weighted_child_entropy


def binary_entropy(p):
    """
    二元熵函数

    H(p) = -p log₂ p - (1-p) log₂(1-p)

    参数:
        p: 概率（0 <= p <= 1）
    """
    # TODO: 实现二元熵
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def channel_capacity_binary_symmetric(error_prob):
    """
    二元对称信道的信道容量

    C = 1 - H(p)

    其中 p 是错误概率

    参数:
        error_prob: 错误概率
    """
    # TODO: 实现信道容量
    return 1 - binary_entropy(error_prob)


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: Shannon熵")
    # 均匀分布的熵最大
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    h_uniform = shannon_entropy(p_uniform)
    expected = 2.0  # log₂(4) = 2 bits
    results.append(v.assert_close(h_uniform, expected, rtol=1e-6, name="均匀分布熵"))

    # 确定性分布的熵为0
    p_certain = np.array([1.0, 0.0, 0.0, 0.0])
    h_certain = shannon_entropy(p_certain)
    results.append(v.assert_close(h_certain, 0.0, atol=1e-10, name="确定性分布熵"))

    print("\n测试 2: 联合熵与条件熵")
    # 独立变量: H(X,Y) = H(X) + H(Y)
    p_x = np.array([0.5, 0.5])
    p_y = np.array([0.5, 0.5])
    p_xy_indep = np.outer(p_x, p_y)

    h_joint = joint_entropy(p_xy_indep)
    h_sum = shannon_entropy(p_x) + shannon_entropy(p_y)
    results.append(v.assert_close(h_joint, h_sum, rtol=1e-6, name="独立变量联合熵"))

    print("\n测试 3: 互信息")
    # 独立变量的互信息为0
    mi_indep = mutual_information(p_xy_indep)
    results.append(v.assert_close(mi_indep, 0.0, atol=1e-10, name="独立变量互信息"))

    # 完全相关的互信息 = H(X) = H(Y)
    p_xy_corr = np.array([[0.5, 0.0], [0.0, 0.5]])  # Y = X
    mi_corr = mutual_information(p_xy_corr)
    h_x_corr = shannon_entropy([0.5, 0.5])
    results.append(v.assert_close(mi_corr, h_x_corr, rtol=1e-6, name="完全相关互信息"))

    print("\n测试 4: KL散度")
    p = np.array([0.5, 0.5])
    q_same = np.array([0.5, 0.5])
    # D_KL(P||P) = 0
    kl_same = kl_divergence(p, q_same)
    results.append(v.assert_close(kl_same, 0.0, atol=1e-10, name="KL散度(P||P)"))

    # KL散度非对称
    q_diff = np.array([0.7, 0.3])
    kl_pq = kl_divergence(p, q_diff)
    kl_qp = kl_divergence(q_diff, p)
    is_asymmetric = abs(kl_pq - kl_qp) > 0.01
    results.append(is_asymmetric)
    print(f"  D_KL(P||Q) = {kl_pq:.4f}, D_KL(Q||P) = {kl_qp:.4f}")
    print(f"  KL散度非对称: {'✓ 是' if is_asymmetric else '✗ 否'}")

    print("\n测试 5: JS散度")
    # JS散度是对称的
    js_pq = js_divergence(p, q_diff)
    js_qp = js_divergence(q_diff, p)
    results.append(v.assert_close(js_pq, js_qp, rtol=1e-6, name="JS散度对称性"))

    print("\n测试 6: 交叉熵")
    # H(P, P) = H(P)
    ce_same = cross_entropy(p, p)
    h_p = shannon_entropy(p) / np.log(2) * np.log(np.e)  # 转换为自然对数
    results.append(v.assert_close(ce_same, h_p, rtol=1e-5, name="交叉熵H(P,P)"))

    # H(P, Q) = H(P) + D_KL(P||Q)
    ce_diff = cross_entropy(p, q_diff)
    expected_ce = h_p + kl_divergence(p, q_diff) / np.log(2) * np.log(np.e)
    results.append(v.assert_close(ce_diff, expected_ce, rtol=1e-5, name="交叉熵关系"))

    print("\n测试 7: 困惑度")
    # 均匀分布的困惑度 = 类别数
    perp = perplexity(p_uniform)
    results.append(v.assert_close(perp, 4.0, rtol=1e-6, name="困惑度"))

    print("\n测试 8: 基尼不纯度")
    # 均匀分布的基尼不纯度
    gini = gini_impurity(p_uniform)
    expected_gini = 1 - 4 * (0.25)**2  # = 0.75
    results.append(v.assert_close(gini, expected_gini, name="基尼不纯度"))

    # 纯节点的基尼不纯度 = 0
    gini_pure = gini_impurity([1.0, 0.0])
    results.append(v.assert_close(gini_pure, 0.0, atol=1e-10, name="纯节点基尼"))

    print("\n测试 9: 信息增益")
    parent_h = shannon_entropy([0.5, 0.5])
    child1_h = shannon_entropy([1.0, 0.0])  # 纯节点
    child2_h = shannon_entropy([0.0, 1.0])  # 纯节点
    ig = information_gain(parent_h, [child1_h, child2_h], [0.5, 0.5])
    # 完美分裂的信息增益 = 父节点熵
    results.append(v.assert_close(ig, parent_h, rtol=1e-6, name="信息增益"))

    print("\n测试 10: 信道容量")
    # 无噪声信道（p=0）的容量 = 1
    cap_perfect = channel_capacity_binary_symmetric(0.0)
    results.append(v.assert_close(cap_perfect, 1.0, name="无噪声信道容量"))

    # 最大噪声信道（p=0.5）的容量 = 0
    cap_noisy = channel_capacity_binary_symmetric(0.5)
    results.append(v.assert_close(cap_noisy, 0.0, atol=1e-10, name="最大噪声信道容量"))

    return all(results)


if __name__ == "__main__":
    test()
