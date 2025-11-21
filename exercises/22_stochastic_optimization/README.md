# 随机优化 (Stochastic Optimization)

## 📚 理论概述

随机优化是处理大规模机器学习问题的核心技术。与传统的批量优化方法不同，随机优化方法在每次迭代中只使用一个或少量样本，从而大幅降低计算复杂度。这些方法是现代深度学习的基石。

**核心思想**：用样本梯度的随机估计代替真实梯度，通过大量迭代逼近最优解。

---

## 🎯 初级：SGD和Mini-batch梯度下降

### 1. 批量梯度下降 (Batch Gradient Descent, BGD)

**优化问题**：

$$\min_{\mathbf{w}} f(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n f_i(\mathbf{w})$$

其中$f_i(\mathbf{w})$是第$i$个样本的损失。

**批量梯度下降**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t) = \mathbf{w}_t - \eta \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{w}_t)$$

**特点**：
- 每次迭代使用**全部**$n$个样本
- 梯度计算准确
- 计算开销：$O(n)$每次迭代

**优点**：
- 稳定收敛
- 理论保证（凸函数）

**缺点**：
- 大数据集（$n$大）时非常慢
- 内存需求大
- 容易陷入局部最优（非凸问题）

**收敛率**（强凸函数）：

$$f(\mathbf{w}_t) - f(\mathbf{w}^*) \leq O\left(\frac{1}{t}\right)$$

使用合适学习率$\eta = O(1/t)$。

### 2. 随机梯度下降 (Stochastic Gradient Descent, SGD)

**思想**：每次迭代只使用**一个**随机样本。

**算法**：

For epoch $= 1, 2, ...$:
  - 随机打乱数据
  - For $i = 1, ..., n$:
    $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f_i(\mathbf{w}_t)$$

**梯度估计**：

$$\mathbb{E}[\nabla f_i(\mathbf{w})] = \nabla f(\mathbf{w})$$

即单个样本梯度是真实梯度的**无偏估计**。

**特点**：
- 计算开销：$O(1)$每次迭代
- 梯度估计噪声大

**优点**：
- 计算快速（大数据集）
- 噪声有助于逃离局部最优（非凸问题）
- 内存友好（一次一个样本）

**缺点**：
- 收敛不稳定（梯度噪声）
- 需要学习率衰减
- 最终不收敛到精确最优（在最优点附近震荡）

**收敛率**（强凸函数，适当学习率）：

$$\mathbb{E}[f(\mathbf{w}_t)] - f(\mathbf{w}^*) \leq O\left(\frac{1}{t}\right)$$

**学习率选择**：
- 常用：$\eta_t = \frac{\eta_0}{1 + \lambda t}$
- 或：$\eta_t = \frac{\eta_0}{\sqrt{t}}$

### 3. Mini-batch SGD

**中间方案**：每次使用$b$个样本（mini-batch）。

**算法**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{b}\sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{w}_t)$$

其中$\mathcal{B}_t$是大小为$b$的随机mini-batch。

**batch size选择**：

| Batch Size | 优点 | 缺点 |
|-----------|------|------|
| $b = 1$ (SGD) | 快速更新，逃离局部最优 | 噪声大，不稳定 |
| $b = n$ (BGD) | 稳定，准确 | 慢，内存大 |
| $b \in [32, 256]$ | 平衡速度与稳定性，GPU并行 | 需要调参 |

**方差与batch size**：

梯度估计的方差：
$$\text{Var}[\nabla f_{\mathcal{B}}] = \frac{1}{b} \text{Var}[\nabla f_i]$$

Batch size越大，方差越小（$\propto 1/b$）。

**实践建议**：
- 图像任务：$b = 32, 64, 128$
- NLP任务：$b = 16, 32$（序列长度不同）
- 小数据集：$b = 10-100$
- 大数据集：$b = 256-1024$

**GPU加速**：
- Mini-batch可以并行计算
- Batch size = 32/64最大化GPU利用率

---

## 🎯 中级：Momentum和Nesterov加速

### 1. 动量SGD (SGD with Momentum)

**问题**：SGD在峡谷地形（某些方向梯度大，某些方向梯度小）时震荡严重。

**物理直觉**：把优化想象成小球滚下山坡，积累动量。

**算法**：

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$$

其中：
- $\mathbf{v}_t$：速度（动量）
- $\beta \in [0, 1)$：动量系数（通常0.9）
- $\eta$：学习率

**展开形式**：

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{w}_t) = \sum_{i=0}^t \beta^{t-i} \nabla f(\mathbf{w}_i)$$

即速度是历史梯度的**指数加权移动平均** (EWMA)。

**效果**：
- 加速相关方向（梯度方向一致）
- 减少震荡方向（梯度方向相反）

**数学解释**：

等价于最小化：
$$\min_{\mathbf{w}} f(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w} - \mathbf{w}_0\|^2$$

其中$\lambda$由$\beta$决定。

**超参数选择**：
- $\beta = 0.9$：经典选择
- $\beta = 0.99$：更多历史（长记忆）
- $\beta = 0$：退化为标准SGD

**Polyak形式**（另一种表述）：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t) + \beta(\mathbf{w}_t - \mathbf{w}_{t-1})$$

### 2. Nesterov加速梯度 (Nesterov Accelerated Gradient, NAG)

**改进思想**："向前看"，在预测位置计算梯度。

**算法**：

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{w}_t - \beta \mathbf{v}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$$

**区别**：
- Momentum：在当前位置$\mathbf{w}_t$计算梯度
- NAG：在"向前看"位置$\mathbf{w}_t - \beta \mathbf{v}_t$计算梯度

**直觉**：
1. 根据动量跳到$\mathbf{w}_t - \beta \mathbf{v}_t$
2. 在那里评估梯度
3. 修正方向

**优势**：
- 更快收敛（凸优化）
- 更稳定（提前看到"错误"）

**收敛率**（强凸）：

$$f(\mathbf{w}_t) - f(\mathbf{w}^*) \leq O\left(\frac{1}{t^2}\right)$$

比标准GD的$O(1/t)$更快！

**另一种形式**（Sutskever等，2013）：

$$\tilde{\mathbf{w}}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \tilde{\mathbf{w}}_{t+1} + \beta(\tilde{\mathbf{w}}_{t+1} - \tilde{\mathbf{w}}_t)$$

### 3. 学习率衰减 (Learning Rate Decay)

**必要性**：
- 初期：需要大学习率快速靠近最优
- 后期：需要小学习率精细调整

**常见策略**：

#### 3.1 指数衰减 (Exponential Decay)

$$\eta_t = \eta_0 \cdot \gamma^t$$

其中$\gamma \in (0, 1)$（如0.95, 0.99）。

#### 3.2 阶梯衰减 (Step Decay)

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}$$

每$k$个epoch衰减一次（如$k=10$, $\gamma=0.5$）。

#### 3.3 逆时间衰减 (Inverse Time Decay)

$$\eta_t = \frac{\eta_0}{1 + kt}$$

其中$k$是衰减率。

#### 3.4 余弦退火 (Cosine Annealing)

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

其中$T$是总迭代次数。

平滑下降，常用于训练Transformer。

#### 3.5 Warm Restart (SGDR)

周期性重启学习率：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)$$

每隔$T_i$重启一次，帮助逃离局部最优。

**选择建议**：
- 简单任务：指数/阶梯衰减
- 深度学习：余弦退火
- 难优化问题：Warm Restart

---

## 🎯 高级：自适应学习率方法

**核心思想**：为每个参数自适应调整学习率。

### 1. AdaGrad

**问题**：不同参数更新频率不同（稀疏特征 vs 稠密特征）。

**算法**：

$$G_t = G_{t-1} + \nabla f(\mathbf{w}_t) \odot \nabla f(\mathbf{w}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla f(\mathbf{w}_t)$$

其中：
- $G_t$：累积平方梯度（元素wise）
- $\odot$：元素级乘法
- $\epsilon$：数值稳定项（$10^{-8}$）

**直觉**：
- 梯度大的参数 → $G_t$大 → 学习率小
- 梯度小的参数 → $G_t$小 → 学习率大

**优点**：
- 适用于稀疏数据（如NLP）
- 不需要手动调学习率

**缺点**：
- 学习率单调递减：$\eta_t \propto 1/\sqrt{t}$
- 训练后期可能过早停止（学习率趋于0）

**应用**：
- 词嵌入训练（稀疏梯度）
- 广告点击率预测

### 2. RMSProp

**改进AdaGrad**：使用移动平均代替累积。

**算法**：

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) \nabla f(\mathbf{w}_t) \odot \nabla f(\mathbf{w}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot \nabla f(\mathbf{w}_t)$$

其中$\beta \approx 0.9$（衰减率）。

**对比AdaGrad**：
- AdaGrad：$G_t = \sum_{i=1}^t g_i^2$（所有历史）
- RMSProp：$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$（指数加权）

**效果**：
- 避免学习率单调递减
- 适应非平稳问题

**Hinton提出**（未发表论文，Coursera课程）。

### 3. Adam (Adaptive Moment Estimation)

**思想**：结合Momentum和RMSProp。

**算法**：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla f(\mathbf{w}_t) \quad \text{(一阶矩估计)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) \nabla f(\mathbf{w}_t) \odot \nabla f(\mathbf{w}_t) \quad \text{(二阶矩估计)}$$

**偏差修正**：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**更新**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

**默认超参数**（Kingma & Ba, 2015）：
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\eta = 0.001$
- $\epsilon = 10^{-8}$

**为什么需要偏差修正？**

初始时$m_0 = 0, v_0 = 0$，导致：
- $m_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i$（偏向0）
- $v_t$同理

修正：除以$(1-\beta_1^t)$，当$t \to \infty$时趋于1（无修正）。

**优势**：
- 收敛快
- 稳定
- 超参数鲁棒
- 广泛适用

**现状**：深度学习默认优化器（90%以上）。

**变体**：
- **AMSGrad**：修复Adam不收敛问题
- **AdaBound**：逐渐变为SGD
- **RAdam**：修正热身阶段

### 4. AdamW

**Adam的问题**：权重衰减（L2正则化）与自适应学习率交互。

**标准Adam + L2**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot (\hat{m}_t + \lambda \mathbf{w}_t)$$

权重衰减被自适应学习率缩放！

**AdamW解耦权重衰减**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t - \eta \lambda \mathbf{w}_t$$

权重衰减独立于自适应学习率。

**效果**：
- 更好的泛化（Transformer训练）
- 与SGD+Momentum的权重衰减一致

**BERT, GPT等模型的标配**。

### 5. 梯度裁剪 (Gradient Clipping)

**问题**：梯度爆炸（RNN、深度网络）。

**方法1：按值裁剪** (Clip by Value)

$$g_i \leftarrow \max(\min(g_i, \theta), -\theta)$$

**方法2：按范数裁剪** (Clip by Norm)

$$\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \leq \theta \\
\frac{\theta}{\|\mathbf{g}\|} \mathbf{g} & \text{otherwise}
\end{cases}$$

**效果**：
- 保持梯度方向
- 限制步长

**超参数**：
- $\theta = 1.0$（RNN常用）
- $\theta = 5.0$（Transformer）

**应用**：
- 循环神经网络（LSTM, GRU）
- Transformer训练
- 强化学习（PPO）

---

## 📊 优化器对比

| 优化器 | 计算开销 | 内存开销 | 超参数数量 | 收敛速度 | 泛化能力 | 适用场景 |
|-------|---------|---------|-----------|---------|---------|---------|
| SGD | 低 | 低 | 1-2 | 慢 | 好 | 凸优化、简单任务 |
| SGD+Momentum | 低 | 低 | 2-3 | 中 | 好 | 图像分类、ResNet |
| AdaGrad | 中 | 中 | 1 | 中 | 中 | 稀疏数据、NLP |
| RMSProp | 中 | 中 | 2 | 快 | 中 | RNN、非平稳问题 |
| Adam | 中 | 中 | 3 | 快 | 中 | 通用、深度学习 |
| AdamW | 中 | 中 | 4 | 快 | 好 | Transformer、BERT |

**经验法则**：
- **研究/竞赛**：尝试多种，选最好的
- **快速原型**：Adam（收敛快）
- **最终模型**：SGD+Momentum或AdamW（泛化好）
- **NLP/Transformer**：AdamW
- **CV/ResNet**：SGD+Momentum

---

## 🧮 数学分析

### 1. SGD收敛性分析

**假设**：
1. $f$是$L$-光滑的：$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|$
2. $f$是$\mu$-强凸的：$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$
3. 梯度方差有界：$\mathbb{E}[\|\nabla f_i(\mathbf{w}) - \nabla f(\mathbf{w})\|^2] \leq \sigma^2$

**定理**（非渐近收敛）：

使用学习率$\eta = \frac{1}{\mu t}$，SGD满足：

$$\mathbb{E}[\|\mathbf{w}_t - \mathbf{w}^*\|^2] \leq \frac{L\sigma^2}{\mu^2 t}$$

**解释**：
- 收敛率：$O(1/t)$
- 依赖于条件数$\kappa = L/\mu$
- 梯度方差$\sigma^2$影响常数

### 2. Momentum加速原理

**Polyak重球法**的等价微分方程：

$$\ddot{\mathbf{x}}(t) + 2\sqrt{\mu}\dot{\mathbf{x}}(t) + \nabla f(\mathbf{x}(t)) = 0$$

这是一个**阻尼振荡系统**（物理类比）。

**Nesterov加速**的收敛率：

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \frac{2L\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{t^2}$$

比GD的$O(1/t)$快！

### 3. Adam收敛性

**原始Adam论文**声称收敛，但**Reddi等（2018）**指出反例。

**问题**：二阶矩的指数移动平均可能"忘记"历史最大值。

**AMSGrad修复**：

$$v_t = \max(\hat{v}_t, v_{t-1})$$

保证$v_t$单调递增。

**实践中**：原始Adam仍广泛使用（问题罕见）。

---

## 📊 实践技巧

### 1. 学习率调优

**经验法则**：
- Adam：$\eta = 10^{-3}$或$10^{-4}$
- SGD：$\eta = 0.1$或$0.01$（配合momentum）

**Learning Rate Finder** (Leslie Smith, 2017)：

1. 从很小的学习率开始（$10^{-6}$）
2. 每个batch指数增加学习率
3. 记录loss
4. 选择loss下降最快的学习率

**1cycle策略**：
1. 从小学习率线性增加到最大
2. 线性下降到更小
3. 最后极速下降

### 2. Batch Size选择

**大batch的问题**（泛化差）：
- 收敛到sharp minima（泛化差）
- 小batch收敛到flat minima（泛化好）

**解决方案**：
- 大batch + 更长训练
- 大batch + 学习率线性缩放（$\eta \propto b$）
- Warm-up策略

**Goyal等（2017）**：训练ResNet-50，batch=8192：
- 学习率$\eta = 0.1 \times \frac{b}{256}$
- 5 epoch warm-up

### 3. 优化器组合

**分层学习率**：不同层使用不同学习率
- 预训练层：小学习率
- 新增层：大学习率

**例子**（BERT fine-tuning）：
```python
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

### 4. 调试技巧

**检查清单**：
- [ ] 损失是否下降？
- [ ] 梯度是否合理（不是NaN/Inf）？
- [ ] 学习率是否合适（太大/太小）？
- [ ] Batch size是否合理？

**可视化**：
- Loss曲线（训练/验证）
- 梯度范数
- 参数更新范数
- 学习率schedule

**常见问题**：

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| Loss不下降 | 学习率太小 | 增大学习率 |
| Loss震荡 | 学习率太大 | 减小学习率 |
| Loss=NaN | 梯度爆炸 | 梯度裁剪、降低学习率 |
| 快速过拟合 | Batch size太小 | 增大batch、正则化 |
| 收敛慢 | 优化器不合适 | 尝试Adam |

---

## 📚 参考教材

1. **Goodfellow, Bengio & Courville - Deep Learning (2016)**
   - Chapter 8: Optimization for Training Deep Models

2. **Ruder - An Overview of Gradient Descent Optimization Algorithms (2016)**
   - 综述博客，非常清晰

3. **Bottou, Curtis & Nocedal - Optimization Methods for Large-Scale Machine Learning (2018)**
   - SIAM Review，权威综述

4. **Kingma & Ba - Adam: A Method for Stochastic Optimization (2015)**
   - Adam原始论文

5. **Loshchilov & Hutter - Decoupled Weight Decay Regularization (2019)**
   - AdamW论文

6. **You, Gitman & Ginsburg - Large Batch Training of Convolutional Networks (2017)**
   - 大batch训练技巧

---

## 🔗 相关练习

- **微积分** (Exercise 01): 梯度计算基础
- **优化理论** (Exercise 05): 凸优化、梯度下降
- **线性代数** (Exercise 02): 矩阵运算
- **概率论** (Exercise 03): 随机估计

---

## 💻 PyTorch实现示例

### SGD with Momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
```

### Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

### 学习率scheduler

```python
# StepLR
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)

# CosineAnnealingLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 使用
for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

---

## 🎯 练习建议

1. **实现基础优化器**：
   - 从头实现SGD、Momentum、Adam
   - 在简单函数上测试（如Rosenbrock函数）

2. **可视化优化路径**：
   - 2D函数上可视化不同优化器的轨迹
   - 对比收敛速度

3. **超参数实验**：
   - 学习率：$10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}$
   - Batch size：$16, 32, 64, 128, 256$
   - Momentum：$0, 0.5, 0.9, 0.99$

4. **真实任务**：
   - MNIST手写数字识别
   - CIFAR-10图像分类
   - 对比SGD vs Adam

5. **阅读论文**：
   - Adam论文（理解偏差修正）
   - AdamW论文（理解权重衰减）
   - 大batch训练论文

**调试流程**：
1. 从小学习率开始（$10^{-4}$）
2. 逐步增大，观察loss
3. 找到最大不发散的学习率
4. 使用该学习率的0.1-0.5倍

**实验记录**：使用Weights & Biases或TensorBoard记录：
- Loss曲线
- 学习率变化
- 梯度范数
- 参数直方图

**现代最佳实践**（2024）：
- Transformer：AdamW + Cosine Annealing + Warmup
- CV (ResNet)：SGD + Momentum + Step Decay
- 小数据集：Adam（快速）
- 大数据集：SGD（泛化好，但需要调参）
