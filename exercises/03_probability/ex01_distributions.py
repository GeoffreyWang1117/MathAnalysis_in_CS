"""
练习 1: 概率分布
================

学习目标：
- 理解常见概率分布
- 掌握概率密度函数和累积分布函数
- 应用numpy.random生成随机样本

任务：
实现和可视化各种概率分布
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def bernoulli_trial(p, n=1):
    """
    伯努利试验: 以概率 p 成功

    参数:
        p: 成功概率
        n: 试验次数

    返回:
        成功次数数组
    """
    # TODO: 实现伯努利试验
    # 提示: 使用 np.random.rand() < p
    results = (np.random.rand(n) < p).astype(int)
    return results


def binomial_distribution(n, p, size=1000):
    """
    二项分布: B(n, p)
    n次独立伯努利试验中成功的次数

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


def normal_distribution(mu=0, sigma=1, size=1000):
    """
    正态分布: N(μ, σ²)

    参数:
        mu: 均值
        sigma: 标准差
        size: 样本数量
    """
    # TODO: 生成正态分布样本
    samples = np.random.normal(mu, sigma, size)
    return samples


def exponential_distribution(lambda_param=1.0, size=1000):
    """
    指数分布: Exp(λ)

    用于建模事件之间的等待时间
    PDF: f(x) = λ * e^(-λx), x ≥ 0
    """
    # TODO: 生成指数分布样本
    samples = np.random.exponential(1/lambda_param, size)
    return samples


def poisson_distribution(lambda_param=1.0, size=1000):
    """
    泊松分布: Poisson(λ)

    用于建模单位时间内事件发生的次数
    """
    # TODO: 生成泊松分布样本
    samples = np.random.poisson(lambda_param, size)
    return samples


def compute_moments(samples):
    """
    计算样本的矩

    返回:
        mean: 均值（一阶矩）
        variance: 方差（二阶中心矩）
        skewness: 偏度（三阶标准化矩）
        kurtosis: 峰度（四阶标准化矩）
    """
    # TODO: 计算统计矩
    mean = np.mean(samples)
    variance = np.var(samples, ddof=1)  # 无偏估计
    skewness = stats.skew(samples)
    kurtosis = stats.kurtosis(samples)

    return mean, variance, skewness, kurtosis


def empirical_cdf(samples, x):
    """
    经验累积分布函数: F(x) = P(X ≤ x)

    参数:
        samples: 样本数组
        x: 计算CDF的点

    返回:
        CDF值
    """
    # TODO: 计算经验CDF
    cdf_value = np.mean(samples <= x)
    return cdf_value


def central_limit_theorem_demo(distribution_func, n_samples=30, n_experiments=1000):
    """
    中心极限定理演示

    从任意分布中抽取n个样本，计算均值
    重复多次，观察均值的分布趋向正态分布

    参数:
        distribution_func: 生成样本的函数
        n_samples: 每次实验的样本数
        n_experiments: 实验次数

    返回:
        均值数组
    """
    # TODO: 实现中心极限定理演示
    means = []
    for _ in range(n_experiments):
        samples = distribution_func(size=n_samples)
        means.append(np.mean(samples))

    return np.array(means)


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: 伯努利试验")
    np.random.seed(42)
    trials = bernoulli_trial(p=0.5, n=10000)
    success_rate = np.mean(trials)
    results.append(v.assert_close(success_rate, 0.5, rtol=0.05, name="伯努利成功率"))

    print("\n测试 2: 二项分布")
    np.random.seed(42)
    samples = binomial_distribution(n=10, p=0.5, size=10000)
    mean_binom = np.mean(samples)
    expected_mean = 10 * 0.5  # n * p
    results.append(v.assert_close(mean_binom, expected_mean, rtol=0.05, name="二项分布均值"))

    print("\n测试 3: 正态分布")
    np.random.seed(42)
    samples = normal_distribution(mu=5, sigma=2, size=10000)
    mean, var, _, _ = compute_moments(samples)
    results.append(v.assert_close(mean, 5, rtol=0.05, name="正态分布均值"))
    results.append(v.assert_close(var, 4, rtol=0.1, name="正态分布方差"))

    print("\n测试 4: 指数分布")
    np.random.seed(42)
    samples = exponential_distribution(lambda_param=2.0, size=10000)
    mean_exp = np.mean(samples)
    expected_mean_exp = 1 / 2.0  # 1/λ
    results.append(v.assert_close(mean_exp, expected_mean_exp, rtol=0.05, name="指数分布均值"))

    print("\n测试 5: 泊松分布")
    np.random.seed(42)
    samples = poisson_distribution(lambda_param=3.0, size=10000)
    mean_poisson = np.mean(samples)
    results.append(v.assert_close(mean_poisson, 3.0, rtol=0.05, name="泊松分布均值"))

    print("\n测试 6: 经验CDF")
    np.random.seed(42)
    samples = normal_distribution(mu=0, sigma=1, size=10000)
    cdf_0 = empirical_cdf(samples, 0)
    # 对于标准正态分布，F(0) = 0.5
    results.append(v.assert_close(cdf_0, 0.5, rtol=0.05, name="经验CDF"))

    print("\n测试 7: 中心极限定理")
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
    print(f"  {'✓ 通过' if is_normal else '✗ 失败'}")

    return all(results)


if __name__ == "__main__":
    test()
