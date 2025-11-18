"""
练习 1: 马尔可夫链
==================

学习目标：
- 理解马尔可夫链的基本性质
- 掌握转移矩阵
- 计算稳态分布
- 应用马尔可夫链建模

任务：
实现马尔可夫链的模拟和分析
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def simulate_markov_chain(P, initial_state, n_steps):
    """
    模拟马尔可夫链

    参数:
        P: 转移概率矩阵 (n_states × n_states)
           P[i,j] = P(X_{t+1} = j | X_t = i)
        initial_state: 初始状态
        n_steps: 模拟步数

    返回:
        状态序列
    """
    # TODO: 实现马尔可夫链模拟
    n_states = P.shape[0]
    states = [initial_state]

    current_state = initial_state
    for _ in range(n_steps):
        # 根据当前状态的转移概率选择下一个状态
        next_state = np.random.choice(n_states, p=P[current_state])
        states.append(next_state)
        current_state = next_state

    return np.array(states)


def transition_matrix_power(P, n):
    """
    计算转移矩阵的n次幂 P^n

    P^n[i,j] 表示从状态 i 经过 n 步到达状态 j 的概率

    参数:
        P: 转移矩阵
        n: 步数
    """
    # TODO: 实现矩阵幂运算
    result = np.linalg.matrix_power(P, n)
    return result


def find_stationary_distribution(P, method='eigenvalue'):
    """
    求稳态分布 π

    稳态分布满足: π^T P = π^T
    或等价地: P^T π = π

    参数:
        P: 转移矩阵
        method: 'eigenvalue' 或 'simulation'

    返回:
        稳态分布向量
    """
    # TODO: 实现稳态分布计算
    if method == 'eigenvalue':
        # 方法1：求解特征值问题 P^T π = π
        # 即求特征值为 1 的特征向量
        eigenvalues, eigenvectors = np.linalg.eig(P.T)

        # 找到特征值最接近 1 的特征向量
        idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
        stationary = np.real(eigenvectors[:, idx])

        # 归一化使其和为 1
        stationary = stationary / np.sum(stationary)

        return stationary

    elif method == 'simulation':
        # 方法2：通过大量迭代逼近
        # π = π P = π P² = ... = π P^n
        n = 1000
        P_n = transition_matrix_power(P, n)
        # 任意行都会收敛到稳态分布
        stationary = P_n[0, :]
        return stationary


def check_irreducibility(P):
    """
    检查马尔可夫链是否不可约

    不可约：从任意状态可以到达任意其他状态

    参数:
        P: 转移矩阵
    """
    # TODO: 实现不可约性检验
    n_states = P.shape[0]

    # 计算 P + P² + ... + P^n
    # 如果所有元素 > 0，则不可约
    P_sum = np.zeros_like(P)
    P_power = P.copy()

    for _ in range(n_states):
        P_sum += P_power
        P_power = P_power @ P

    # 检查是否所有元素都 > 0
    return np.all(P_sum > 0)


def check_aperiodicity(P):
    """
    检查马尔可夫链是否非周期

    非周期：gcd{n : P^n[i,i] > 0} = 1

    参数:
        P: 转移矩阵
    """
    # TODO: 实现非周期性检验
    # 简化检验：如果存在自环（P[i,i] > 0），则非周期
    has_self_loop = np.any(np.diag(P) > 0)
    return has_self_loop


def absorption_probability(P, transient_states, absorbing_states, initial_state):
    """
    计算吸收概率

    从瞬时态到达吸收态的概率

    参数:
        P: 转移矩阵
        transient_states: 瞬时状态列表
        absorbing_states: 吸收状态列表
        initial_state: 初始状态

    返回:
        吸收概率向量
    """
    # TODO: 实现吸收概率计算
    n_transient = len(transient_states)
    n_absorbing = len(absorbing_states)

    # 标准形式: P = [Q  R]
    #              [0  I]
    # Q: 瞬时态之间的转移
    # R: 瞬时态到吸收态的转移

    Q = P[np.ix_(transient_states, transient_states)]
    R = P[np.ix_(transient_states, absorbing_states)]

    # 基本矩阵: N = (I - Q)^(-1)
    I = np.eye(n_transient)
    N = np.linalg.inv(I - Q)

    # 吸收概率: B = N * R
    B = N @ R

    # 返回从初始状态的吸收概率
    idx = transient_states.index(initial_state)
    return B[idx, :]


def expected_hitting_time(P, start_state, target_state):
    """
    计算期望首达时间

    E[T_j | X_0 = i] = 从状态 i 首次到达状态 j 的期望步数

    参数:
        P: 转移矩阵
        start_state: 起始状态
        target_state: 目标状态

    返回:
        期望首达时间
    """
    # TODO: 实现期望首达时间计算
    # 使用模拟方法
    n_simulations = 10000
    hitting_times = []

    for _ in range(n_simulations):
        current = start_state
        steps = 0
        max_steps = 10000

        while current != target_state and steps < max_steps:
            next_state = np.random.choice(P.shape[0], p=P[current])
            current = next_state
            steps += 1

        if steps < max_steps:
            hitting_times.append(steps)

    if len(hitting_times) == 0:
        return np.inf

    return np.mean(hitting_times)


def page_rank(P, d=0.85, max_iter=100, tol=1e-6):
    """
    PageRank算法（马尔可夫链的应用）

    PageRank = d * P^T * PageRank + (1-d) * e/n

    参数:
        P: 转移矩阵（网页链接矩阵）
        d: 阻尼因子（通常为0.85）
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        PageRank向量
    """
    # TODO: 实现PageRank
    n = P.shape[0]
    rank = np.ones(n) / n  # 初始化为均匀分布

    for _ in range(max_iter):
        new_rank = d * P.T @ rank + (1 - d) / n
        if np.linalg.norm(new_rank - rank) < tol:
            break
        rank = new_rank

    return rank


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 简单的马尔可夫链：天气模型
    # 状态 0: 晴天, 状态 1: 雨天
    P_weather = np.array([[0.8, 0.2],   # 晴天->晴天 0.8, 晴天->雨天 0.2
                          [0.4, 0.6]])  # 雨天->晴天 0.4, 雨天->雨天 0.6

    print("\n测试 1: 马尔可夫链模拟")
    np.random.seed(42)
    states = simulate_markov_chain(P_weather, initial_state=0, n_steps=100)
    # 检查状态值是否合法
    valid_states = np.all((states >= 0) & (states <= 1))
    results.append(valid_states)
    print(f"  状态序列长度: {len(states)}")
    print(f"  {'✓ 通过' if valid_states else '✗ 失败'}")

    print("\n测试 2: 转移矩阵幂")
    P2 = transition_matrix_power(P_weather, 2)
    # 每行和应该为 1
    row_sums = np.sum(P2, axis=1)
    results.append(v.assert_close(row_sums, np.ones(2), name="转移矩阵行和"))

    print("\n测试 3: 稳态分布")
    stationary = find_stationary_distribution(P_weather, method='eigenvalue')
    # 验证 π^T P = π^T
    check = stationary @ P_weather
    results.append(v.assert_close(check, stationary, rtol=1e-5, name="稳态分布"))
    print(f"  稳态分布: {stationary}")

    print("\n测试 4: 不可约性检验")
    is_irreducible = check_irreducibility(P_weather)
    results.append(is_irreducible)
    print(f"  不可约: {'✓ 是' if is_irreducible else '✗ 否'}")

    print("\n测试 5: 非周期性检验")
    is_aperiodic = check_aperiodicity(P_weather)
    results.append(is_aperiodic)
    print(f"  非周期: {'✓ 是' if is_aperiodic else '✗ 否'}")

    print("\n测试 6: 吸收概率")
    # 随机游走吸收链：状态 0 和 2 是吸收态
    P_absorb = np.array([[1.0, 0.0, 0.0],
                         [0.5, 0.0, 0.5],
                         [0.0, 0.0, 1.0]])
    transient = [1]
    absorbing = [0, 2]
    prob = absorption_probability(P_absorb, transient, absorbing, 1)
    # 从状态1开始，各有0.5概率到达状态0和状态2
    expected_prob = np.array([0.5, 0.5])
    results.append(v.assert_close(prob, expected_prob, rtol=0.01, name="吸收概率"))

    print("\n测试 7: PageRank")
    # 简单的链接矩阵
    P_links = np.array([[0, 0.5, 0.5],
                        [1, 0, 0],
                        [0.5, 0.5, 0]])
    rank = page_rank(P_links, d=0.85)
    # 检查和为1
    sum_rank = np.sum(rank)
    results.append(v.assert_close(sum_rank, 1.0, name="PageRank归一化"))
    print(f"  PageRank: {rank}")

    return all(results)


if __name__ == "__main__":
    test()
