"""
练习 2: 布朗运动与随机过程
==========================

学习目标：
- 理解布朗运动的性质
- 模拟维纳过程
- 应用Ito引理
- 实现几何布朗运动（金融建模）

任务：
实现布朗运动的模拟和分析
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def standard_brownian_motion(T, n_steps, n_paths=1):
    """
    模拟标准布朗运动（维纳过程）

    性质:
    1. W(0) = 0
    2. W(t) - W(s) ~ N(0, t-s) for t > s
    3. 独立增量
    4. 连续路径

    参数:
        T: 终止时间
        n_steps: 时间步数
        n_paths: 路径数量

    返回:
        (t, W): 时间数组和布朗运动路径
    """
    # TODO: 实现标准布朗运动模拟
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # 生成增量: dW ~ N(0, √dt)
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

    # 累积求和得到布朗运动
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    return t, W


def brownian_bridge(T, n_steps, W_T=0):
    """
    布朗桥：固定端点的布朗运动

    B(t) = W(t) - (t/T)W(T)

    参数:
        T: 终止时间
        n_steps: 时间步数
        W_T: 终点值（默认为0）

    返回:
        (t, B): 时间和布朗桥
    """
    # TODO: 实现布朗桥
    t, W = standard_brownian_motion(T, n_steps, n_paths=1)
    W = W[0]

    # 调整使得 W(T) = W_T
    W = W - (t / T) * (W[-1] - W_T)

    return t, W


def geometric_brownian_motion(S0, mu, sigma, T, n_steps):
    """
    几何布朗运动（股票价格模型）

    dS = μS dt + σS dW
    解: S(t) = S₀ exp((μ - σ²/2)t + σW(t))

    参数:
        S0: 初始价格
        mu: 漂移率（期望收益率）
        sigma: 波动率
        T: 时间跨度
        n_steps: 步数

    返回:
        (t, S): 时间和价格路径
    """
    # TODO: 实现几何布朗运动
    t, W = standard_brownian_motion(T, n_steps, n_paths=1)
    W = W[0]

    # S(t) = S₀ exp((μ - σ²/2)t + σW(t))
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    return t, S


def ornstein_uhlenbeck_process(x0, theta, mu, sigma, T, n_steps):
    """
    Ornstein-Uhlenbeck过程（均值回复过程）

    dx = θ(μ - x)dt + σ dW

    用于利率建模等

    参数:
        x0: 初始值
        theta: 均值回复速度
        mu: 长期均值
        sigma: 波动率
        T: 时间跨度
        n_steps: 步数
    """
    # TODO: 实现OU过程
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps + 1)
    x[0] = x0

    # Euler-Maruyama方法
    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        x[i + 1] = x[i] + theta * (mu - x[i]) * dt + sigma * dW

    return t, x


def black_scholes_call_option(S0, K, T, r, sigma):
    """
    Black-Scholes欧式看涨期权定价

    C = S₀N(d₁) - Ke^(-rT)N(d₂)

    参数:
        S0: 当前股价
        K: 行权价
        T: 到期时间
        r: 无风险利率
        sigma: 波动率

    返回:
        期权价格
    """
    # TODO: 实现Black-Scholes公式
    from scipy.stats import norm

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    C = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return C


def monte_carlo_option_pricing(S0, K, T, r, sigma, n_paths=100000):
    """
    蒙特卡洛期权定价

    通过模拟大量路径计算期权价格

    参数:
        S0, K, T, r, sigma: 同Black-Scholes
        n_paths: 模拟路径数

    返回:
        期权价格
    """
    # TODO: 实现蒙特卡洛定价
    # 模拟终端股价
    Z = np.random.normal(0, 1, n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # 计算payoff
    payoff = np.maximum(S_T - K, 0)

    # 折现期望payoff
    option_price = np.exp(-r * T) * np.mean(payoff)

    return option_price


def quadratic_variation(W, t):
    """
    计算布朗运动的二次变差

    [W]_t = lim Σ(W(t_i) - W(t_{i-1}))²

    对于布朗运动，[W]_t = t

    参数:
        W: 布朗运动路径
        t: 时间数组

    返回:
        二次变差
    """
    # TODO: 实现二次变差计算
    increments = np.diff(W)
    quad_var = np.sum(increments**2)
    return quad_var


def first_passage_time(barrier, T, n_steps, n_simulations=10000):
    """
    首次穿越时间

    τ = inf{t : W(t) = barrier}

    参数:
        barrier: 障碍水平
        T: 最大时间
        n_steps: 步数
        n_simulations: 模拟次数

    返回:
        平均首次穿越时间
    """
    # TODO: 实现首次穿越时间
    passage_times = []

    for _ in range(n_simulations):
        t, W = standard_brownian_motion(T, n_steps, n_paths=1)
        W = W[0]

        # 找到首次穿越时间
        crossed = np.where(np.abs(W) >= barrier)[0]
        if len(crossed) > 0:
            passage_times.append(t[crossed[0]])

    if len(passage_times) == 0:
        return np.inf

    return np.mean(passage_times)


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    np.random.seed(42)

    print("\n测试 1: 标准布朗运动")
    T, n_steps = 1.0, 1000
    t, W = standard_brownian_motion(T, n_steps, n_paths=1)
    W = W[0]

    # 检查 W(0) = 0
    results.append(v.assert_close(W[0], 0.0, name="W(0) = 0"))

    # 检查路径连续性
    is_continuous = len(W) == n_steps + 1
    results.append(is_continuous)
    print(f"  路径连续性: {'✓ 通过' if is_continuous else '✗ 失败'}")

    print("\n测试 2: 布朗运动的方差")
    # E[W(t)²] = t
    np.random.seed(42)
    t_test = 2.0
    _, W_paths = standard_brownian_motion(t_test, 1000, n_paths=10000)
    W_final = W_paths[:, -1]
    variance = np.var(W_final)
    results.append(v.assert_close(variance, t_test, rtol=0.1, name="W(t)的方差"))

    print("\n测试 3: 布朗桥")
    t, B = brownian_bridge(1.0, 1000, W_T=0)
    # 检查端点
    results.append(v.assert_close(B[0], 0.0, atol=1e-10, name="布朗桥起点"))
    results.append(v.assert_close(B[-1], 0.0, rtol=0.01, name="布朗桥终点"))

    print("\n测试 4: 几何布朗运动")
    S0, mu, sigma = 100, 0.1, 0.2
    t, S = geometric_brownian_motion(S0, mu, sigma, 1.0, 1000)
    # 检查初始值
    results.append(v.assert_close(S[0], S0, name="GBM初始值"))
    # 检查价格为正
    all_positive = np.all(S > 0)
    results.append(all_positive)
    print(f"  价格恒正: {'✓ 是' if all_positive else '✗ 否'}")

    print("\n测试 5: Ornstein-Uhlenbeck过程")
    x0, theta, mu_ou, sigma_ou = 0, 1.0, 5.0, 0.5
    t, x = ornstein_uhlenbeck_process(x0, theta, mu_ou, sigma_ou, 10.0, 1000)
    # 长期应该趋向于均值
    mean_value = np.mean(x[-200:])  # 最后部分的均值
    results.append(abs(mean_value - mu_ou) < 2.0)  # 宽松容差
    print(f"  长期均值: {mean_value:.2f} (期望: {mu_ou:.2f})")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 6: Black-Scholes定价")
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    bs_price = black_scholes_call_option(S0, K, T, r, sigma)
    # 检查价格合理性
    is_reasonable = 0 < bs_price < S0
    results.append(is_reasonable)
    print(f"  期权价格: {bs_price:.4f}")
    print(f"  {'✓ 合理' if is_reasonable else '✗ 不合理'}")

    print("\n测试 7: 蒙特卡洛定价 vs Black-Scholes")
    np.random.seed(42)
    mc_price = monte_carlo_option_pricing(S0, K, T, r, sigma, n_paths=100000)
    # 蒙特卡洛应该接近Black-Scholes
    results.append(v.assert_close(mc_price, bs_price, rtol=0.02, name="蒙特卡洛定价"))

    print("\n测试 8: 二次变差")
    np.random.seed(42)
    T_qv = 2.0
    t, W = standard_brownian_motion(T_qv, 10000, n_paths=1)
    qv = quadratic_variation(W[0], t)
    # [W]_t 应该接近 t
    results.append(v.assert_close(qv, T_qv, rtol=0.1, name="二次变差"))

    return all(results)


if __name__ == "__main__":
    test()
