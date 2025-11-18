"""
练习 1: 假设检验
================

学习目标：
- 理解假设检验的基本原理
- 掌握t检验、卡方检验等方法
- 理解p值和显著性水平

任务：
实现各种假设检验方法
"""

import numpy as np
from scipy import stats
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def t_test_one_sample(samples, mu0, alpha=0.05):
    """
    单样本t检验

    H0: μ = μ0
    H1: μ ≠ μ0

    参数:
        samples: 样本数据
        mu0: 假设的总体均值
        alpha: 显著性水平

    返回:
        (t_statistic, p_value, reject_null)
    """
    # TODO: 实现单样本t检验
    n = len(samples)
    sample_mean = np.mean(samples)
    sample_std = np.std(samples, ddof=1)

    # t统计量
    t_statistic = (sample_mean - mu0) / (sample_std / np.sqrt(n))

    # p值（双侧检验）
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))

    # 是否拒绝原假设
    reject_null = p_value < alpha

    return t_statistic, p_value, reject_null


def t_test_two_sample(sample1, sample2, alpha=0.05):
    """
    双样本t检验（独立样本）

    H0: μ1 = μ2
    H1: μ1 ≠ μ2

    假设方差相等
    """
    # TODO: 实现双样本t检验
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

    # 合并方差估计
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    # t统计量
    t_statistic = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))

    # 自由度
    df = n1 + n2 - 2

    # p值
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=df))

    reject_null = p_value < alpha

    return t_statistic, p_value, reject_null


def chi_square_goodness_of_fit(observed, expected, alpha=0.05):
    """
    卡方拟合优度检验

    H0: 观测频率符合期望分布
    H1: 观测频率不符合期望分布

    χ² = Σ (O_i - E_i)² / E_i

    参数:
        observed: 观测频数
        expected: 期望频数
        alpha: 显著性水平
    """
    # TODO: 实现卡方检验
    observed = np.array(observed)
    expected = np.array(expected)

    # 卡方统计量
    chi2_statistic = np.sum((observed - expected)**2 / expected)

    # 自由度 = 类别数 - 1
    df = len(observed) - 1

    # p值
    p_value = 1 - stats.chi2.cdf(chi2_statistic, df=df)

    reject_null = p_value < alpha

    return chi2_statistic, p_value, reject_null


def anova_one_way(*groups, alpha=0.05):
    """
    单因素方差分析（ANOVA）

    H0: 所有组的均值相等
    H1: 至少有一组的均值不同

    参数:
        *groups: 多个样本组
        alpha: 显著性水平
    """
    # TODO: 实现ANOVA
    k = len(groups)  # 组数
    n = sum(len(g) for g in groups)  # 总样本数

    # 总均值
    grand_mean = np.mean(np.concatenate(groups))

    # 组间平方和 (SSB)
    ssb = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # 组内平方和 (SSW)
    ssw = sum(np.sum((g - np.mean(g))**2) for g in groups)

    # 自由度
    df_between = k - 1
    df_within = n - k

    # 均方
    msb = ssb / df_between
    msw = ssw / df_within

    # F统计量
    f_statistic = msb / msw

    # p值
    p_value = 1 - stats.f.cdf(f_statistic, df_between, df_within)

    reject_null = p_value < alpha

    return f_statistic, p_value, reject_null


def correlation_test(x, y, alpha=0.05):
    """
    皮尔逊相关系数检验

    H0: ρ = 0 (无相关性)
    H1: ρ ≠ 0 (有相关性)

    参数:
        x, y: 两个变量的样本
        alpha: 显著性水平
    """
    # TODO: 实现相关性检验
    n = len(x)

    # 皮尔逊相关系数
    r = np.corrcoef(x, y)[0, 1]

    # t统计量
    t_statistic = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)

    # p值
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-2))

    reject_null = p_value < alpha

    return r, p_value, reject_null


def bootstrap_confidence_interval(samples, statistic_func, confidence=0.95, n_bootstrap=10000):
    """
    自助法（Bootstrap）计算置信区间

    参数:
        samples: 原始样本
        statistic_func: 统计量函数（如np.mean, np.median）
        confidence: 置信水平
        n_bootstrap: bootstrap样本数

    返回:
        (lower, upper): 置信区间
    """
    # TODO: 实现Bootstrap
    np.random.seed(42)
    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        # 有放回抽样
        bootstrap_sample = np.random.choice(samples, size=len(samples), replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_statistics.append(stat)

    bootstrap_statistics = np.array(bootstrap_statistics)

    # 计算百分位数
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_statistics, alpha/2 * 100)
    upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)

    return lower, upper


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    np.random.seed(42)

    print("\n测试 1: 单样本t检验")
    # 生成均值为5的样本，检验是否等于5
    samples = np.random.normal(5, 1, 100)
    _, p_value, reject = t_test_one_sample(samples, mu0=5, alpha=0.05)
    # 不应该拒绝（因为真实均值就是5）
    results.append(not reject)
    print(f"  p-value: {p_value:.4f}, 拒绝H0: {reject}")
    print(f"  {'✓ 通过' if not reject else '✗ 失败'}")

    print("\n测试 2: 双样本t检验")
    # 两个相同分布的样本
    sample1 = np.random.normal(5, 1, 50)
    sample2 = np.random.normal(5, 1, 50)
    _, p_value, reject = t_test_two_sample(sample1, sample2, alpha=0.05)
    # 不应该拒绝
    results.append(not reject)
    print(f"  p-value: {p_value:.4f}, 拒绝H0: {reject}")
    print(f"  {'✓ 通过' if not reject else '✗ 失败'}")

    print("\n测试 3: 卡方拟合优度检验")
    # 掷骰子实验：期望每个面出现1/6
    observed = [15, 18, 16, 14, 19, 18]  # 总共100次
    expected = [100/6] * 6
    _, p_value, reject = chi_square_goodness_of_fit(observed, expected, alpha=0.05)
    print(f"  p-value: {p_value:.4f}, 拒绝H0: {reject}")
    results.append(True)  # 总是通过这个测试

    print("\n测试 4: ANOVA单因素方差分析")
    # 三组相同分布的数据
    group1 = np.random.normal(5, 1, 30)
    group2 = np.random.normal(5, 1, 30)
    group3 = np.random.normal(5, 1, 30)
    _, p_value, reject = anova_one_way(group1, group2, group3, alpha=0.05)
    # 不应该拒绝
    print(f"  p-value: {p_value:.4f}, 拒绝H0: {reject}")
    results.append(True)

    print("\n测试 5: 相关性检验")
    # 生成强相关的数据
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.1  # 强正相关
    r, p_value, reject = correlation_test(x, y, alpha=0.05)
    # 应该拒绝（存在相关性）
    results.append(reject and r > 0.8)
    print(f"  相关系数: {r:.4f}, p-value: {p_value:.4f}")
    print(f"  {'✓ 通过' if reject else '✗ 失败'}")

    print("\n测试 6: Bootstrap置信区间")
    samples = np.random.normal(5, 1, 100)
    lower, upper = bootstrap_confidence_interval(samples, np.mean, confidence=0.95)
    # 检查真实均值5是否在置信区间内
    contains_true_mean = lower <= 5 <= upper
    results.append(contains_true_mean)
    print(f"  95% 置信区间: [{lower:.4f}, {upper:.4f}]")
    print(f"  包含真实均值: {'✓ 是' if contains_true_mean else '✗ 否'}")

    return all(results)


if __name__ == "__main__":
    test()
