"""
练习 1: 概率分布
================

本练习包含3道题目，由浅入深：
- 初级：离散分布（伯努利、二项、泊松）
- 中级：连续分布和统计矩
- 高级：中心极限定理和分布收敛

学习目标：
- 理解常见概率分布
- 掌握概率密度函数和累积分布函数
- 应用numpy.random生成随机样本
- 理解统计推断的理论基础

参考知识点：
- 概率分布函数和密度函数
- 期望、方差、偏度、峰度
- 大数定律和中心极限定理
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：离散分布
# ============================================================================

def bernoulli_trial(p, n=1):
    """
    初级 - 伯努利试验: 以概率 p 成功

    伯努利试验是最基本的随机试验
    只有两个可能结果：成功(1)或失败(0)

    参数:
        p: 成功概率
        n: 试验次数

    返回:
        成功次数数组（0或1）
    """
    # TODO: 实现伯努利试验
    # 提示: 使用 np.random.rand() < p
    results = (np.random.rand(n) < p).astype(int)
    return results


def binomial_distribution(n, p, size=1000):
    """
    初级 - 二项分布: B(n, p)

    n次独立伯努利试验中成功的次数
    E[X] = np, Var[X] = np(1-p)

    参数:
        n: 试验次数
        p: 单次成功概率
        size: 样本数量

    返回:
        样本数组
    """
    # TODO: 生成二项分布样本
    samples = np.random.binomial(n, p, size)
    return samples


def poisson_distribution(lambda_param=1.0, size=1000):
    """
    初级 - 泊松分布: Poisson(λ)

    用于建模单位时间内事件发生的次数
    当n很大、p很小但np=λ时，二项分布趋向泊松分布

    参数:
        lambda_param: 平均发生率 λ
        size: 样本数量

    返回:
        样本数组
    """
    # TODO: 生成泊松分布样本
    samples = np.random.poisson(lambda_param, size)
    return samples


# ============================================================================
# 中级题目：连续分布和统计矩
# ============================================================================

def normal_distribution(mu=0, sigma=1, size=1000):
    """
    中级 - 正态分布: N(μ, σ²)

    最重要的连续分布，由中心极限定理保证
    68-95-99.7规则: 68%在1σ内, 95%在2σ内, 99.7%在3σ内

    参数:
        mu: 均值
        sigma: 标准差
        size: 样本数量

    返回:
        样本数组
    """
    # TODO: 生成正态分布样本
    samples = np.random.normal(mu, sigma, size)
    return samples


def exponential_distribution(lambda_param=1.0, size=1000):
    """
    中级 - 指数分布: Exp(λ)

    用于建模事件之间的等待时间
    无记忆性: P(X>s+t | X>s) = P(X>t)
    PDF: f(x) = λe^(-λx), x ≥ 0

    参数:
        lambda_param: 率参数 λ
        size: 样本数量

    返回:
        样本数组
    """
    # TODO: 生成指数分布样本
    samples = np.random.exponential(1/lambda_param, size)
    return samples


def compute_moments(samples):
    """
    中级 - 计算样本的矩

    矩是分布的数值特征:
    - 一阶矩: 均值（中心位置）
    - 二阶中心矩: 方差（分散程度）
    - 三阶标准化矩: 偏度（对称性）
    - 四阶标准化矩: 峰度（尾部厚度）

    参数:
        samples: 样本数组

    返回:
        (mean, variance, skewness, kurtosis)
    """
    # TODO: 计算统计矩
    mean = np.mean(samples)
    variance = np.var(samples, ddof=1)  # 无偏估计
    skewness = stats.skew(samples)
    kurtosis = stats.kurtosis(samples)

    return mean, variance, skewness, kurtosis


def empirical_cdf(samples, x):
    """
    中级 - 经验累积分布函数: F(x) = P(X ≤ x)

    用样本估计真实的CDF
    根据Glivenko-Cantelli定理，当样本数趋向无穷时
    经验CDF几乎必然收敛到真实CDF

    参数:
        samples: 样本数组
        x: 计算CDF的点

    返回:
        CDF值 ∈ [0, 1]
    """
    # TODO: 计算经验CDF
    cdf_value = np.mean(samples <= x)
    return cdf_value


def quantile_function(samples, q):
    """
    中级 - 分位数函数（CDF的逆）

    q分位数是使得 P(X ≤ x_q) = q 的值
    例如: 中位数是0.5分位数

    参数:
        samples: 样本数组
        q: 分位数水平 ∈ (0, 1)

    返回:
        q分位数
    """
    # TODO: 计算分位数
    return np.quantile(samples, q)


# ============================================================================
# 高级题目：中心极限定理和分布收敛
# ============================================================================

def central_limit_theorem_demo(distribution_func, n_samples=30, n_experiments=1000):
    """
    高级 - 中心极限定理演示

    中心极限定理(CLT): 设X₁, X₂, ..., Xₙ独立同分布,
    E[Xᵢ]=μ, Var[Xᵢ]=σ²，则当n→∞时:

    (X̄ - μ) / (σ/√n) → N(0, 1)

    这意味着样本均值的分布趋向正态分布，
    无论原始分布是什么

    参数:
        distribution_func: 生成样本的函数
        n_samples: 每次实验的样本数
        n_experiments: 实验次数

    返回:
        样本均值数组
    """
    # TODO: 实现中心极限定理演示
    means = []
    for _ in range(n_experiments):
        samples = distribution_func(size=n_samples)
        means.append(np.mean(samples))

    return np.array(means)


def monte_carlo_integration(f, a, b, n_samples=10000):
    """
    高级 - 蒙特卡洛积分

    使用随机采样估计定积分:
    ∫ₐᵇ f(x)dx ≈ (b-a) * E[f(X)], X ~ Uniform(a, b)

    参数:
        f: 被积函数
        a, b: 积分区间
        n_samples: 采样数

    返回:
        积分的估计值
    """
    # TODO: 实现蒙特卡洛积分
    # 在[a, b]上均匀采样
    x_samples = np.random.uniform(a, b, n_samples)

    # 计算函数值的平均
    f_values = np.array([f(x) for x in x_samples])

    # 积分估计
    integral_estimate = (b - a) * np.mean(f_values)

    return integral_estimate


def kolmogorov_smirnov_test(samples, reference_cdf):
    """
    高级 - Kolmogorov-Smirnov检验

    检验样本是否来自给定分布
    计算经验CDF和理论CDF之间的最大距离

    KS统计量: D = sup|F_n(x) - F(x)|

    参数:
        samples: 样本数组
        reference_cdf: 参考CDF函数

    返回:
        KS统计量
    """
    # TODO: 实现KS检验
    # 对样本排序
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)

    # 计算经验CDF和理论CDF的最大差异
    max_diff = 0
    for i, x in enumerate(sorted_samples):
        empirical = (i + 1) / n
        theoretical = reference_cdf(x)
        diff = abs(empirical - theoretical)
        max_diff = max(max_diff, diff)

    return max_diff


def importance_sampling(target_pdf, proposal_sample_func, proposal_pdf, n_samples=10000):
    """
    高级 - 重要性采样

    当直接从目标分布采样困难时，
    从提议分布采样并调整权重:

    E_p[f(X)] ≈ (1/n) Σ f(Xᵢ) * p(Xᵢ) / q(Xᵢ)

    其中 Xᵢ ~ q

    参数:
        target_pdf: 目标分布的PDF
        proposal_sample_func: 从提议分布采样的函数
        proposal_pdf: 提议分布的PDF
        n_samples: 样本数

    返回:
        加权样本的均值（期望估计）
    """
    # TODO: 实现重要性采样
    # 从提议分布采样
    samples = proposal_sample_func(n_samples)

    # 计算重要性权重
    weights = np.array([target_pdf(x) / proposal_pdf(x) if proposal_pdf(x) > 0 else 0
                        for x in samples])

    # 加权平均
    weighted_mean = np.mean(samples * weights) / np.mean(weights) if np.mean(weights) > 0 else 0

    return weighted_mean


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：离散分布")
    print("="*60)

    print("\n测试 1.1: 伯努利试验")
    np.random.seed(42)
    trials = bernoulli_trial(p=0.5, n=10000)
    success_rate = np.mean(trials)
    results.append(v.assert_close(success_rate, 0.5, rtol=0.05, name="伯努利成功率"))

    print("\n测试 1.2: 二项分布")
    np.random.seed(42)
    samples = binomial_distribution(n=10, p=0.5, size=10000)
    mean_binom = np.mean(samples)
    expected_mean = 10 * 0.5  # n * p
    results.append(v.assert_close(mean_binom, expected_mean, rtol=0.05, name="二项分布均值"))

    print("\n测试 1.3: 泊松分布")
    np.random.seed(42)
    samples = poisson_distribution(lambda_param=3.0, size=10000)
    mean_poisson = np.mean(samples)
    results.append(v.assert_close(mean_poisson, 3.0, rtol=0.05, name="泊松分布均值"))

    print("\n" + "="*60)
    print("中级题目：连续分布和统计矩")
    print("="*60)

    print("\n测试 2.1: 正态分布")
    np.random.seed(42)
    samples = normal_distribution(mu=5, sigma=2, size=10000)
    mean, var, _, _ = compute_moments(samples)
    results.append(v.assert_close(mean, 5, rtol=0.05, name="正态分布均值"))
    results.append(v.assert_close(var, 4, rtol=0.1, name="正态分布方差"))

    print("\n测试 2.2: 指数分布")
    np.random.seed(42)
    samples = exponential_distribution(lambda_param=2.0, size=10000)
    mean_exp = np.mean(samples)
    expected_mean_exp = 1 / 2.0  # 1/λ
    results.append(v.assert_close(mean_exp, expected_mean_exp, rtol=0.05, name="指数分布均值"))

    print("\n测试 2.3: 经验CDF")
    np.random.seed(42)
    samples = normal_distribution(mu=0, sigma=1, size=10000)
    cdf_0 = empirical_cdf(samples, 0)
    # 对于标准正态分布，F(0) = 0.5
    results.append(v.assert_close(cdf_0, 0.5, rtol=0.05, name="经验CDF"))

    print("\n测试 2.4: 分位数函数")
    median = quantile_function(samples, 0.5)
    results.append(v.assert_close(median, 0.0, atol=0.1, name="中位数"))

    print("\n" + "="*60)
    print("高级题目：中心极限定理和分布收敛")
    print("="*60)

    print("\n测试 3.1: 中心极限定理")
    np.random.seed(42)
    # 使用指数分布（非正态）
    means = central_limit_theorem_demo(
        lambda size: exponential_distribution(1.0, size),
        n_samples=30,
        n_experiments=1000
    )
    # 均值的分布应该接近正态
    _, p_value = stats.normaltest(means)
    is_normal = p_value > 0.01  # 不拒绝正态性
    results.append(is_normal)
    print(f"  正态性检验 p-value: {p_value:.4f}")
    print(f"  {'✓ 通过（接近正态）' if is_normal else '✗ 失败'}")

    print("\n测试 3.2: 蒙特卡洛积分")
    np.random.seed(42)
    # 计算 ∫₀¹ x² dx = 1/3
    f = lambda x: x**2
    integral_est = monte_carlo_integration(f, 0, 1, n_samples=10000)
    results.append(v.assert_close(integral_est, 1/3, rtol=0.05, name="蒙特卡洛积分"))

    print("\n测试 3.3: Kolmogorov-Smirnov检验")
    np.random.seed(42)
    samples = normal_distribution(mu=0, sigma=1, size=1000)
    ref_cdf = lambda x: stats.norm.cdf(x, 0, 1)
    ks_stat = kolmogorov_smirnov_test(samples, ref_cdf)
    # KS统计量应该很小（样本确实来自该分布）
    results.append(ks_stat < 0.1)
    print(f"  KS统计量: {ks_stat:.4f}")
    print(f"  {'✓ 通过（小统计量）' if results[-1] else '✗ 失败'}")

    print("\n测试 3.4: 重要性采样")
    np.random.seed(42)
    # 估计标准正态分布的期望（应该是0）
    target_pdf = lambda x: stats.norm.pdf(x, 0, 1)
    proposal_sample = lambda n: np.random.normal(0, 1, n)
    proposal_pdf = lambda x: stats.norm.pdf(x, 0, 1)
    estimate = importance_sampling(target_pdf, proposal_sample, proposal_pdf, n_samples=1000)
    results.append(abs(estimate) < 0.5)  # 应该接近0
    print(f"  期望估计: {estimate:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
