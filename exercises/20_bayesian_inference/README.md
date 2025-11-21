# 贝叶斯推断 (Bayesian Inference)

## 📚 理论概述

贝叶斯推断是统计推断的一种方法，它使用贝叶斯定理来更新关于参数的概率信念。与频率学派不同，贝叶斯学派将参数视为随机变量，并使用概率分布来描述对参数的不确定性。

---

## 🎯 初级：贝叶斯定理和先验后验

### 1. 贝叶斯定理 (Bayes' Theorem)

**定理陈述**：

$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

其中：
- $P(\theta | D)$：**后验概率** (Posterior) - 观察到数据$D$后，对参数$\theta$的更新信念
- $P(D | \theta)$：**似然函数** (Likelihood) - 给定参数$\theta$时，观察到数据$D$的概率
- $P(\theta)$：**先验概率** (Prior) - 观察数据前，对参数$\theta$的初始信念
- $P(D)$：**边缘似然** (Evidence/Marginal Likelihood) - 数据的归一化常数

**比例形式**（更常用）：

$$P(\theta | D) \propto P(D | \theta) P(\theta)$$

读作："后验 ∝ 似然 × 先验"

**例子：医疗诊断**

假设某种疾病的患病率为1%，测试的灵敏度（真阳性率）为99%，特异度（真阴性率）为99%。如果测试结果为阳性，真正患病的概率是多少？

$$P(\text{病} | \text{阳性}) = \frac{P(\text{阳性} | \text{病}) \cdot P(\text{病})}{P(\text{阳性})}$$

$$= \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.01 \times 0.99} = \frac{0.0099}{0.0198} = 0.5$$

结果仅为50%！这个反直觉的结果说明了先验概率的重要性。

### 2. Beta-Binomial 共轭模型

**共轭先验**的定义：如果先验分布和后验分布属于同一分布族，则称先验为共轭先验。

**Beta分布**作为**二项分布**的共轭先验：

**先验**：
$$\theta \sim \text{Beta}(\alpha, \beta)$$
$$p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

其中$B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$是Beta函数。

**似然**（观察到$n$次试验中$k$次成功）：
$$p(k | \theta, n) = \binom{n}{k} \theta^k (1-\theta)^{n-k}$$

**后验**：
$$\theta | k, n \sim \text{Beta}(\alpha + k, \beta + n - k)$$

**后验均值**：
$$E[\theta | k, n] = \frac{\alpha + k}{\alpha + \beta + n}$$

**解释**：
- $\alpha$可以理解为"先验成功次数"
- $\beta$可以理解为"先验失败次数"
- 后验就是将先验计数和观察数据相加

**例子：投硬币**

使用均匀先验$\text{Beta}(1, 1)$（相当于没有先验信息），观察到10次投掷中6次正面：

后验：$\text{Beta}(1+6, 1+4) = \text{Beta}(7, 5)$

后验均值：$\frac{7}{12} = 0.583$

### 3. 离散后验计算

对于有限个离散假设$\{\theta_1, \theta_2, ..., \theta_n\}$：

$$P(\theta_i | D) = \frac{P(D | \theta_i) P(\theta_i)}{\sum_{j=1}^n P(D | \theta_j) P(\theta_j)}$$

**算法步骤**：
1. 计算每个假设的先验概率$P(\theta_i)$
2. 计算每个假设下的似然$P(D | \theta_i)$
3. 计算未归一化后验：$P(\theta_i) \times P(D | \theta_i)$
4. 归一化：除以总和使概率和为1

---

## 🎯 中级：共轭先验和贝叶斯估计

### 1. Gaussian-Gaussian 共轭模型

**场景**：已知方差$\sigma^2$，推断均值$\mu$

**先验**：
$$\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$$

**似然**（$n$个独立观测$\{x_1, ..., x_n\}$）：
$$p(\mathbf{x} | \mu) = \prod_{i=1}^n \mathcal{N}(x_i | \mu, \sigma^2)$$

**后验**：
$$\mu | \mathbf{x} \sim \mathcal{N}(\mu_n, \sigma_n^2)$$

其中：

**后验方差**：
$$\frac{1}{\sigma_n^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}$$

（精度相加！）

**后验均值**：
$$\mu_n = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

这是先验均值和样本均值的**精度加权平均**。

**简化形式**：
$$\mu_n = \frac{\sigma^2}{\sigma^2 + n\sigma_0^2} \mu_0 + \frac{n\sigma_0^2}{\sigma^2 + n\sigma_0^2} \bar{x}$$

**解释**：
- 当$n \to \infty$时，$\mu_n \to \bar{x}$（数据主导）
- 当$\sigma_0^2 \to \infty$时（先验不确定性大），$\mu_n \to \bar{x}$
- 当$\sigma_0^2 \to 0$时（先验很确定），$\mu_n \to \mu_0$

### 2. 最大后验估计 (MAP - Maximum A Posteriori)

**定义**：
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta | D) = \arg\max_\theta p(D | \theta) p(\theta)$$

对数形式（更常用）：
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta [\log p(D | \theta) + \log p(\theta)]$$

**与MLE的关系**：
- MLE（最大似然估计）：$\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(D | \theta)$
- MAP = MLE + 先验正则化

**例子：带L2正则化的线性回归**

似然：$p(\mathbf{y} | \mathbf{X}, \mathbf{w}) = \mathcal{N}(\mathbf{X}\mathbf{w}, \sigma^2 I)$

先验：$p(\mathbf{w}) = \mathcal{N}(0, \tau^2 I)$

MAP估计等价于：
$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_\mathbf{w} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2$$

其中$\lambda = \sigma^2 / \tau^2$（Ridge回归！）

### 3. 可信区间 (Credible Interval)

**定义**：对于后验分布$p(\theta | D)$，$(1-\alpha)$可信区间$[L, U]$满足：

$$P(L \leq \theta \leq U | D) = 1 - \alpha$$

**与置信区间的区别**：
- **置信区间**（频率学派）：重复实验，95%的区间会包含真实参数
- **可信区间**（贝叶斯）：给定数据，参数有95%的概率在此区间

**最高后验密度区间** (HPD - Highest Posterior Density)：

最短的可信区间，满足：
- 区间内的每个点的后验密度都高于区间外的点
- 总概率为$1-\alpha$

**计算方法**（对称情况）：
$$[Q_{\alpha/2}, Q_{1-\alpha/2}]$$

其中$Q_p$是后验分布的$p$分位数。

---

## 🎯 高级：MCMC采样和贝叶斯推断

### 1. Markov Chain Monte Carlo (MCMC)

**问题**：后验分布$p(\theta | D)$通常没有解析形式，如何采样？

**MCMC思想**：
1. 构造一个马尔可夫链，其平稳分布为目标后验$p(\theta | D)$
2. 运行马尔可夫链直到收敛
3. 收敛后的样本近似来自后验分布

#### 1.1 Metropolis-Hastings 算法

**算法**：

给定当前状态$\theta^{(t)}$：

1. **提议新状态**：从提议分布中采样
   $$\theta^* \sim q(\theta^* | \theta^{(t)})$$

2. **计算接受概率**：
   $$\alpha = \min\left(1, \frac{p(\theta^* | D) q(\theta^{(t)} | \theta^*)}{p(\theta^{(t)} | D) q(\theta^* | \theta^{(t)})}\right)$$

3. **接受或拒绝**：
   $$\theta^{(t+1)} = \begin{cases}
   \theta^* & \text{with probability } \alpha \\
   \theta^{(t)} & \text{with probability } 1-\alpha
   \end{cases}$$

**对称提议分布**（如$q(\theta^* | \theta^{(t)}) = \mathcal{N}(\theta^{(t)}, \sigma^2)$）：

$$\alpha = \min\left(1, \frac{p(\theta^* | D)}{p(\theta^{(t)} | D)}\right)$$

**为什么有效？**

满足**细致平衡条件** (Detailed Balance)：
$$p(\theta) q(\theta' | \theta) \alpha(\theta, \theta') = p(\theta') q(\theta | \theta') \alpha(\theta', \theta)$$

这保证了$p(\theta | D)$是马尔可夫链的平稳分布。

#### 1.2 Gibbs 采样

**适用场景**：多元参数$\boldsymbol{\theta} = (\theta_1, ..., \theta_d)$，条件分布$p(\theta_i | \theta_{-i}, D)$已知。

**算法**（一次迭代）：

For $i = 1, ..., d$:
$$\theta_i^{(t+1)} \sim p(\theta_i | \theta_1^{(t+1)}, ..., \theta_{i-1}^{(t+1)}, \theta_{i+1}^{(t)}, ..., \theta_d^{(t)}, D)$$

**优点**：
- 不需要调整提议分布
- 总是接受新样本（接受率100%）

**缺点**：
- 需要知道条件分布
- 参数高度相关时收敛慢

#### 1.3 Hamiltonian Monte Carlo (HMC)

**物理启发**：将参数空间看作位置，引入动量变量，利用Hamilton动力学。

**优势**：
- 在高维空间中比随机游走更高效
- 减少相关性，更快探索参数空间
- PyMC3、Stan等现代贝叶斯软件的核心

### 2. 贝叶斯因子 (Bayes Factor)

**模型选择**：比较两个模型$M_1$和$M_2$

**贝叶斯因子定义**：
$$BF_{12} = \frac{P(D | M_1)}{P(D | M_2)} = \frac{\int p(D | \theta_1, M_1) p(\theta_1 | M_1) d\theta_1}{\int p(D | \theta_2, M_2) p(\theta_2 | M_2) d\theta_2}$$

**后验模型概率**：
$$P(M_1 | D) = \frac{P(D | M_1) P(M_1)}{P(D | M_1) P(M_1) + P(D | M_2) P(M_2)}$$

如果先验相等$P(M_1) = P(M_2)$：
$$\frac{P(M_1 | D)}{P(M_2 | D)} = BF_{12}$$

**Kass & Raftery 解释标准**：

| $BF_{12}$ | $\log_{10} BF_{12}$ | 证据强度 |
|-----------|---------------------|----------|
| 1 - 3 | 0 - 0.5 | 几乎没有 |
| 3 - 20 | 0.5 - 1.3 | 正面 |
| 20 - 150 | 1.3 - 2.2 | 强 |
| > 150 | > 2.2 | 非常强 |

**与p值的区别**：
- p值只能拒绝零假设，不能量化支持程度
- 贝叶斯因子可以量化对任一模型的支持

### 3. 重要性采样 (Importance Sampling)

**问题**：想计算$E_{p(\theta)}[f(\theta)]$，但从$p(\theta)$采样困难。

**解决方案**：从提议分布$q(\theta)$采样，并使用重要性权重修正：

$$E_p[f(\theta)] = \int f(\theta) p(\theta) d\theta = \int f(\theta) \frac{p(\theta)}{q(\theta)} q(\theta) d\theta = E_q\left[f(\theta) w(\theta)\right]$$

其中**重要性权重**：
$$w(\theta) = \frac{p(\theta)}{q(\theta)}$$

**实践中**（归一化版本）：

1. 从$q(\theta)$采样：$\theta_1, ..., \theta_N \sim q(\theta)$
2. 计算权重：$w_i = \frac{p(\theta_i)}{q(\theta_i)}$
3. 归一化权重：$\tilde{w}_i = \frac{w_i}{\sum_j w_j}$
4. 估计期望：$\hat{E}[f(\theta)] = \sum_i \tilde{w}_i f(\theta_i)$

**有效样本量** (Effective Sample Size)：

$$\text{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2} = \frac{1}{\sum_i \tilde{w}_i^2}$$

ESS衡量样本的有效性：
- 如果所有权重相等：$\text{ESS} = N$（完美）
- 如果一个权重为1，其余为0：$\text{ESS} = 1$（最差）

**选择提议分布的原则**：
- $q$的尾部应该比$p$重（避免权重无穷大）
- $q$应该与$p$形状相似
- $q$应该易于采样

---

## 📊 应用示例

### 1. 贝叶斯A/B测试

**场景**：测试两个版本的转化率

**模型**：
- 版本A：$\theta_A \sim \text{Beta}(\alpha_A, \beta_A)$
- 版本B：$\theta_B \sim \text{Beta}(\alpha_B, \beta_B)$

**观察数据后**：
- A：$n_A$次展示，$k_A$次转化
- B：$n_B$次展示，$k_B$次转化

**后验**：
- $\theta_A | \text{data} \sim \text{Beta}(\alpha_A + k_A, \beta_A + n_A - k_A)$
- $\theta_B | \text{data} \sim \text{Beta}(\alpha_B + k_B, \beta_B + n_B - k_B)$

**决策**：$P(\theta_B > \theta_A | \text{data})$ > 0.95 → 选择B

### 2. 贝叶斯线性回归

**模型**：
$$y = \mathbf{x}^T \mathbf{w} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

**先验**：
$$p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \alpha^{-1} I)$$

**后验**：
$$p(\mathbf{w} | \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mathbf{m}_N, \mathbf{S}_N)$$

其中：
$$\mathbf{S}_N = (\alpha I + \beta \mathbf{X}^T \mathbf{X})^{-1}$$
$$\mathbf{m}_N = \beta \mathbf{S}_N \mathbf{X}^T \mathbf{y}$$

**预测分布**：
$$p(y^* | \mathbf{x}^*, \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mathbf{m}_N^T \mathbf{x}^*, \sigma_N^2(\mathbf{x}^*))$$

其中：
$$\sigma_N^2(\mathbf{x}^*) = \frac{1}{\beta} + {\mathbf{x}^*}^T \mathbf{S}_N \mathbf{x}^*$$

**优势**：预测不仅给出均值，还给出不确定性！

### 3. 贝叶斯神经网络

**思想**：将神经网络权重视为随机变量

**挑战**：
- 参数维度极高（百万级）
- 后验分布复杂，无解析形式

**现代方法**：
- **变分推断** (Variational Inference)：用简单分布近似后验
- **Dropout作为近似贝叶斯推断** (Gal & Ghahramani, 2016)
- **随机梯度Langevin动力学** (SGLD)

---

## 📚 参考教材

1. **Gelman et al. - Bayesian Data Analysis (3rd ed.)**
   - 经典教材，理论与应用并重

2. **Bishop - Pattern Recognition and Machine Learning**
   - Chapter 2: Probability Distributions
   - Chapter 3: Linear Models for Regression

3. **Murphy - Machine Learning: A Probabilistic Perspective**
   - Chapter 5: Bayesian Statistics

4. **Kruschke - Doing Bayesian Data Analysis**
   - 入门友好，直观解释

5. **MacKay - Information Theory, Inference, and Learning Algorithms**
   - 信息论视角的贝叶斯推断

---

## 🔗 相关练习

- **概率论** (Exercise 03): 概率分布基础
- **统计学** (Exercise 04): 假设检验与回归分析
- **信息论** (Exercise 11): KL散度与互信息
- **优化理论** (Exercise 05): 梯度下降与凸优化

---

## 💻 实践建议

1. **从共轭先验开始**：Beta-Binomial, Gaussian-Gaussian
2. **可视化后验分布**：理解先验如何被数据更新
3. **实现简单的MCMC**：理解采样过程
4. **使用现代工具**：PyMC3, Stan, TensorFlow Probability
5. **阅读贝叶斯论文**：理解实际应用中的建模选择

**调试MCMC的技巧**：
- 检查trace plot（轨迹图）
- 计算$\hat{R}$统计量（Gelman-Rubin诊断）
- 计算有效样本量（ESS）
- 检查接受率（Metropolis: 20-40%最优）
