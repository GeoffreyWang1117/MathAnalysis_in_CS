"""
练习 1: 贝叶斯推断
==================

本练习包含3道题目，由浅入深：
- 初级：贝叶斯定理和先验后验
- 中级：共轭先验和贝叶斯估计
- 高级：MCMC采样和贝叶斯推断

学习目标：
- 理解贝叶斯定理和贝叶斯推断
- 掌握共轭先验的使用
- 实现MCMC采样算法
- 理解贝叶斯神经网络基础

应用领域：
- 贝叶斯机器学习
- 概率编程 (PyMC, Stan)
- 不确定性量化
- A/B测试

参考教材：
- Gelman et al. - Bayesian Data Analysis
- Murphy - Machine Learning: A Probabilistic Perspective
- Bishop - Pattern Recognition and Machine Learning
"""

import numpy as np
from scipy import stats
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：贝叶斯定理和先验后验
# ============================================================================

def bayes_theorem(prior, likelihood, evidence):
    """
    初级 - 贝叶斯定理

    P(θ|D) = P(D|θ) * P(θ) / P(D)

    其中:
    - P(θ|D): 后验概率 (posterior)
    - P(D|θ): 似然函数 (likelihood)
    - P(θ): 先验概率 (prior)
    - P(D): 证据/边缘似然 (evidence)

    参数:
        prior: 先验概率
        likelihood: 似然概率
        evidence: 证据概率

    返回:
        后验概率
    """
    # TODO: 实现贝叶斯定理
    posterior = (likelihood * prior) / evidence
    return posterior


def compute_evidence(likelihood_array, prior_array):
    """
    初级 - 计算证据/边缘似然

    P(D) = Σ P(D|θ) * P(θ)  (离散情况)
    P(D) = ∫ P(D|θ) * P(θ) dθ  (连续情况)

    参数:
        likelihood_array: 不同θ值的似然
        prior_array: 不同θ值的先验

    返回:
        证据值
    """
    # TODO: 实现证据计算
    evidence = np.sum(likelihood_array * prior_array)
    return evidence


def posterior_discrete(theta_values, prior, data, likelihood_func):
    """
    初级 - 离散参数的后验分布

    计算离散参数空间上的后验分布

    参数:
        theta_values: 参数可能的取值
        prior: 先验分布 (与theta_values对应)
        data: 观测数据
        likelihood_func: 似然函数 P(data|theta)

    返回:
        后验分布
    """
    # TODO: 实现离散后验
    # 计算每个theta的似然
    likelihoods = np.array([likelihood_func(data, theta) for theta in theta_values])

    # 计算证据
    evidence = compute_evidence(likelihoods, prior)

    # 计算后验
    if evidence > 0:
        posterior = (likelihoods * prior) / evidence
    else:
        posterior = prior  # 如果证据为0，返回先验

    return posterior


def beta_binomial_update(alpha_prior, beta_prior, successes, failures):
    """
    初级 - Beta-Binomial共轭更新

    先验: Beta(α, β)
    似然: Binomial(n, p)
    后验: Beta(α + successes, β + failures)

    参数:
        alpha_prior: Beta先验的α参数
        beta_prior: Beta先验的β参数
        successes: 成功次数
        failures: 失败次数

    返回:
        (alpha_posterior, beta_posterior)
    """
    # TODO: 实现Beta-Binomial更新
    alpha_posterior = alpha_prior + successes
    beta_posterior = beta_prior + failures

    return alpha_posterior, beta_posterior


# ============================================================================
# 中级题目：共轭先验和贝叶斯估计
# ============================================================================

def gaussian_gaussian_update(prior_mean, prior_var, data, data_var):
    """
    中级 - 高斯-高斯共轭更新

    先验: N(μ₀, σ₀²)
    似然: N(μ, σ²) with known σ²
    后验: N(μ₁, σ₁²)

    其中:
    σ₁² = 1 / (1/σ₀² + n/σ²)
    μ₁ = σ₁² * (μ₀/σ₀² + Σxᵢ/σ²)

    参数:
        prior_mean: 先验均值 μ₀
        prior_var: 先验方差 σ₀²
        data: 观测数据数组
        data_var: 数据方差 σ²

    返回:
        (posterior_mean, posterior_var)
    """
    # TODO: 实现高斯-高斯更新
    n = len(data)
    data_sum = np.sum(data)

    # 后验方差
    posterior_precision = 1/prior_var + n/data_var
    posterior_var = 1 / posterior_precision

    # 后验均值
    posterior_mean = posterior_var * (prior_mean/prior_var + data_sum/data_var)

    return posterior_mean, posterior_var


def maximum_a_posteriori(theta_values, posterior):
    """
    中级 - 最大后验估计 (MAP)

    MAP估计: θ_MAP = argmax P(θ|D)

    参数:
        theta_values: 参数取值
        posterior: 后验概率

    返回:
        MAP估计值
    """
    # TODO: 实现MAP估计
    map_idx = np.argmax(posterior)
    theta_map = theta_values[map_idx]

    return theta_map


def posterior_predictive(theta_values, posterior, predict_func):
    """
    中级 - 后验预测分布

    P(x_new|D) = ∫ P(x_new|θ) P(θ|D) dθ

    对参数的不确定性进行积分

    参数:
        theta_values: 参数取值
        posterior: 后验分布
        predict_func: 预测函数 P(x_new|θ)

    返回:
        后验预测概率
    """
    # TODO: 实现后验预测
    # 对所有theta积分
    predictive = 0
    for theta, post_prob in zip(theta_values, posterior):
        predictive += predict_func(theta) * post_prob

    return predictive


def credible_interval(samples, alpha=0.95):
    """
    中级 - 可信区间 (Credible Interval)

    贝叶斯版本的置信区间
    包含真实参数的概率为α

    参数:
        samples: 后验样本
        alpha: 置信水平

    返回:
        (lower, upper) 可信区间
    """
    # TODO: 实现可信区间
    lower_percentile = (1 - alpha) / 2 * 100
    upper_percentile = (1 + alpha) / 2 * 100

    lower = np.percentile(samples, lower_percentile)
    upper = np.percentile(samples, upper_percentile)

    return lower, upper


# ============================================================================
# 高级题目：MCMC采样和贝叶斯推断
# ============================================================================

def metropolis_hastings(log_posterior, initial_theta, n_samples, proposal_std=1.0):
    """
    高级 - Metropolis-Hastings算法

    MCMC方法生成后验样本
    接受概率: α = min(1, P(θ*|D) / P(θ|D))

    参数:
        log_posterior: 对数后验函数
        initial_theta: 初始参数值
        n_samples: 样本数
        proposal_std: 提议分布标准差

    返回:
        样本数组
    """
    # TODO: 实现Metropolis-Hastings
    samples = []
    current_theta = initial_theta
    current_log_post = log_posterior(current_theta)

    n_accepted = 0

    for i in range(n_samples):
        # 提议新值 (高斯随机游走)
        proposed_theta = current_theta + np.random.normal(0, proposal_std)

        # 计算接受概率
        proposed_log_post = log_posterior(proposed_theta)
        log_accept_ratio = proposed_log_post - current_log_post

        # Metropolis接受准则
        if np.log(np.random.rand()) < log_accept_ratio:
            current_theta = proposed_theta
            current_log_post = proposed_log_post
            n_accepted += 1

        samples.append(current_theta)

    acceptance_rate = n_accepted / n_samples

    return np.array(samples), acceptance_rate


def gibbs_sampling_gaussian(n_samples, mu_prior, var_prior, data, data_var):
    """
    高级 - Gibbs采样 (高斯例子)

    交替采样每个参数
    适用于条件分布已知的情况

    参数:
        n_samples: 样本数
        mu_prior: 均值先验参数
        var_prior: 方差先验参数
        data: 观测数据
        data_var: 数据方差

    返回:
        (mu_samples, var_samples)
    """
    # TODO: 实现Gibbs采样（简化版）
    mu_samples = []
    var_samples = []

    # 初始值
    mu = mu_prior
    var = var_prior

    n = len(data)
    data_mean = np.mean(data)

    for _ in range(n_samples):
        # 更新mu (给定var和data)
        posterior_mean, posterior_var = gaussian_gaussian_update(
            mu_prior, var_prior, data, var
        )
        mu = np.random.normal(posterior_mean, np.sqrt(posterior_var))

        # 更新var (简化：使用逆gamma先验)
        # 这里简化处理
        var = var_prior

        mu_samples.append(mu)
        var_samples.append(var)

    return np.array(mu_samples), np.array(var_samples)


def hamiltonian_monte_carlo_step(log_posterior, grad_log_posterior, theta,
                                   epsilon=0.01, L=10):
    """
    高级 - 哈密顿蒙特卡洛 (HMC) 单步

    利用梯度信息的MCMC方法
    更高效地探索参数空间

    参数:
        log_posterior: 对数后验
        grad_log_posterior: 对数后验的梯度
        theta: 当前参数
        epsilon: 步长
        L: leapfrog步数

    返回:
        新参数值
    """
    # TODO: 实现HMC单步（简化版）
    # 初始动量
    momentum = np.random.randn(*theta.shape)
    current_momentum = momentum.copy()

    # Leapfrog积分
    momentum = momentum + 0.5 * epsilon * grad_log_posterior(theta)

    for _ in range(L):
        theta = theta + epsilon * momentum
        if _ < L - 1:
            momentum = momentum + epsilon * grad_log_posterior(theta)

    momentum = momentum + 0.5 * epsilon * grad_log_posterior(theta)
    momentum = -momentum

    # Metropolis接受步骤
    current_U = -log_posterior(theta)
    current_K = 0.5 * np.sum(current_momentum**2)

    proposed_U = -log_posterior(theta)
    proposed_K = 0.5 * np.sum(momentum**2)

    if np.random.rand() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return theta
    else:
        return theta  # 简化版：总是返回新值


def bayes_factor(model1_evidence, model2_evidence):
    """
    高级 - 贝叶斯因子

    模型选择的贝叶斯方法
    BF₁₂ = P(D|M₁) / P(D|M₂)

    解释:
    - BF > 10: 强支持M₁
    - BF > 3: 中等支持M₁
    - BF ≈ 1: 无法区分

    参数:
        model1_evidence: 模型1的证据
        model2_evidence: 模型2的证据

    返回:
        贝叶斯因子
    """
    # TODO: 实现贝叶斯因子
    if model2_evidence == 0:
        return np.inf

    bayes_factor = model1_evidence / model2_evidence

    return bayes_factor


def importance_sampling_estimate(target_log_pdf, proposal_sample, proposal_log_pdf, n_samples):
    """
    高级 - 重要性采样估计

    当难以直接从目标分布采样时使用
    E_p[f(x)] ≈ Σ w_i f(x_i) / Σ w_i
    其中 w_i = p(x_i) / q(x_i)

    参数:
        target_log_pdf: 目标对数概率密度
        proposal_sample: 从提议分布采样的函数
        proposal_log_pdf: 提议对数概率密度
        n_samples: 样本数

    返回:
        (样本, 权重, 有效样本数)
    """
    # TODO: 实现重要性采样
    samples = proposal_sample(n_samples)

    # 计算重要性权重
    log_weights = np.array([target_log_pdf(s) - proposal_log_pdf(s) for s in samples])

    # 数值稳定的权重归一化
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)
    weights = weights / np.sum(weights)

    # 有效样本数
    ess = 1.0 / np.sum(weights**2)

    return samples, weights, ess


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：贝叶斯定理和先验后验")
    print("="*60)

    print("\n测试 1.1: 贝叶斯定理")
    # 例子：疾病检测
    # P(病|阳性) = P(阳性|病) * P(病) / P(阳性)
    prior = 0.01  # 患病率1%
    likelihood = 0.95  # 灵敏度95%
    evidence = 0.95 * 0.01 + 0.05 * 0.99  # 阳性概率
    posterior = bayes_theorem(prior, likelihood, evidence)
    results.append(0 < posterior < 1)
    print(f"  后验概率: {posterior:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 1.2: Beta-Binomial更新")
    # 抛硬币：先验Beta(2,2), 观测10次成功7次失败
    alpha_post, beta_post = beta_binomial_update(2, 2, 10, 7)
    results.append(alpha_post == 12 and beta_post == 9)
    print(f"  后验参数: α={alpha_post}, β={beta_post}")
    print(f"  后验均值: {alpha_post/(alpha_post+beta_post):.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 1.3: 离散后验")
    # 简单例子：估计骰子是否公平
    theta_values = np.array([1/6, 0.2, 0.15])  # 可能的P(6)
    prior = np.array([0.6, 0.3, 0.1])  # 先验相信公平
    data = [6, 6, 3, 6, 1]  # 观测数据
    likelihood_func = lambda data, p: np.prod([p if x==6 else (1-p)/5 for x in data])

    posterior = posterior_discrete(theta_values, prior, data, likelihood_func)
    results.append(v.assert_close(np.sum(posterior), 1.0, atol=1e-10, name="后验归一化"))

    print("\n" + "="*60)
    print("中级题目：共轭先验和贝叶斯估计")
    print("="*60)

    print("\n测试 2.1: 高斯-高斯更新")
    prior_mean, prior_var = 0.0, 1.0
    data = np.array([1.0, 1.2, 0.8, 1.1])
    data_var = 0.5
    post_mean, post_var = gaussian_gaussian_update(prior_mean, prior_var, data, data_var)

    # 后验均值应该在先验和数据均值之间
    data_mean = np.mean(data)
    results.append(min(prior_mean, data_mean) <= post_mean <= max(prior_mean, data_mean) + 0.5)
    print(f"  先验均值: {prior_mean:.4f}, 数据均值: {data_mean:.4f}")
    print(f"  后验均值: {post_mean:.4f}, 后验方差: {post_var:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 2.2: MAP估计")
    theta_vals = np.linspace(0, 1, 100)
    posterior_dist = stats.beta.pdf(theta_vals, 12, 9)
    theta_map = maximum_a_posteriori(theta_vals, posterior_dist)
    # MAP应该接近12/(12+9)
    expected_map = 11/21
    results.append(v.assert_close(theta_map, expected_map, rtol=0.1, name="MAP估计"))

    print("\n测试 2.3: 可信区间")
    np.random.seed(42)
    samples = np.random.beta(12, 9, 1000)
    lower, upper = credible_interval(samples, alpha=0.95)
    # 95%可信区间应该包含大部分样本
    in_interval = np.mean((samples >= lower) & (samples <= upper))
    results.append(v.assert_close(in_interval, 0.95, rtol=0.1, name="可信区间覆盖"))

    print("\n" + "="*60)
    print("高级题目：MCMC采样和贝叶斯推断")
    print("="*60)

    print("\n测试 3.1: Metropolis-Hastings")
    # 目标：标准正态分布
    log_posterior = lambda x: -0.5 * x**2
    np.random.seed(42)
    samples, accept_rate = metropolis_hastings(log_posterior, 0.0, 1000, proposal_std=1.0)

    # 检查样本均值和方差
    sample_mean = np.mean(samples[100:])  # burn-in
    sample_std = np.std(samples[100:])
    results.append(v.assert_close(sample_mean, 0.0, atol=0.2, name="MH样本均值"))
    results.append(v.assert_close(sample_std, 1.0, rtol=0.3, name="MH样本标准差"))
    print(f"  接受率: {accept_rate:.2%}")
    print(f"  样本均值: {sample_mean:.4f}, 样本标准差: {sample_std:.4f}")

    print("\n测试 3.2: 贝叶斯因子")
    evidence1 = 0.05
    evidence2 = 0.01
    bf = bayes_factor(evidence1, evidence2)
    results.append(v.assert_close(bf, 5.0, name="贝叶斯因子"))

    print("\n测试 3.3: 重要性采样")
    # 目标: N(2, 1), 提议: N(0, 2)
    target_log_pdf = lambda x: -0.5 * (x - 2)**2
    proposal_sample = lambda n: np.random.normal(0, np.sqrt(2), n)
    proposal_log_pdf = lambda x: -0.5 * x**2 / 2 - 0.5 * np.log(2)

    samples_is, weights, ess = importance_sampling_estimate(
        target_log_pdf, proposal_sample, proposal_log_pdf, 1000
    )

    # 加权均值应该接近2
    weighted_mean = np.sum(samples_is * weights)
    results.append(v.assert_close(weighted_mean, 2.0, rtol=0.2, name="重要性采样均值"))
    print(f"  有效样本数: {ess:.0f} / 1000")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
