# 核方法 (Kernel Methods)

## 📚 理论概述

核方法是机器学习中的一类强大技术，通过**核技巧** (kernel trick) 将数据隐式映射到高维特征空间，从而使线性算法能够学习非线性模式。核方法的核心思想是：不需要显式计算高维特征映射，只需要计算特征空间中的内积。

**核心优势**：
- 将非线性问题转化为线性问题
- 计算高效（避免显式高维映射）
- 理论基础深厚（RKHS理论）
- 广泛应用（SVM、核PCA、高斯过程等）

---

## 🎯 初级：核函数和核矩阵

### 1. 核函数的定义

**定义**：核函数是一个满足特定条件的二元函数：

$$k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$$

**核技巧的本质**：存在一个特征映射$\phi: \mathcal{X} \to \mathcal{H}$，使得

$$k(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_\mathcal{H}$$

**关键洞察**：我们可以计算$k(\mathbf{x}, \mathbf{x}')$而无需显式计算$\phi(\mathbf{x})$！

### 2. 常见核函数

#### 2.1 线性核 (Linear Kernel)

$$k_{\text{linear}}(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$$

- 最简单的核函数
- 等价于原始特征空间的内积
- 用于线性可分问题

#### 2.2 多项式核 (Polynomial Kernel)

$$k_{\text{poly}}(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + c)^d$$

参数：
- $d$：多项式阶数
- $\gamma$：缩放系数
- $c$：常数项

**例子**（$d=2, \gamma=1, c=0$, 二维输入）：

$$k(\mathbf{x}, \mathbf{x}') = (x_1 x_1' + x_2 x_2')^2 = x_1^2 {x_1'}^2 + 2x_1 x_2 x_1' x_2' + x_2^2 {x_2'}^2$$

对应特征映射：
$$\phi(\mathbf{x}) = (x_1^2, \sqrt{2} x_1 x_2, x_2^2)^T$$

特征空间维度从2增加到3！

**一般情况**（$d$阶，$n$维输入）：

特征空间维度：$\binom{n+d}{d} = O(n^d)$

但计算核函数只需$O(n)$时间！

#### 2.3 径向基函数核 / 高斯核 (RBF / Gaussian Kernel)

$$k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$

其中$\gamma = \frac{1}{2\sigma^2}$是带宽参数。

**展开**：
$$k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x}\|^2) \exp(2\gamma \mathbf{x}^T\mathbf{x}') \exp(-\gamma \|\mathbf{x}'\|^2)$$

**特征空间**：无限维！

利用泰勒展开：
$$\exp(\mathbf{x}^T\mathbf{x}') = \sum_{n=0}^\infty \frac{(\mathbf{x}^T\mathbf{x}')^n}{n!}$$

每一项对应一个多项式特征。

**性质**：
- $k(\mathbf{x}, \mathbf{x}) = 1$（自身相似度为1）
- $k(\mathbf{x}, \mathbf{x}') \in (0, 1]$
- 当$\|\mathbf{x} - \mathbf{x}'\| \to \infty$时，$k(\mathbf{x}, \mathbf{x}') \to 0$

**$\gamma$的作用**：
- $\gamma$大：核函数窄，局部性强，易过拟合
- $\gamma$小：核函数宽，全局性强，易欠拟合

#### 2.4 其他常用核

**Sigmoid核**（类似神经网络）：
$$k_{\text{sigmoid}}(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x}^T \mathbf{x}' + c)$$

**Laplacian核**：
$$k_{\text{Laplacian}}(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|_1\right)$$

### 3. 核矩阵 (Kernel Matrix / Gram Matrix)

**定义**：给定数据集$\{\mathbf{x}_1, ..., \mathbf{x}_n\}$，核矩阵$\mathbf{K}$为：

$$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$$

$$\mathbf{K} = \begin{bmatrix}
k(\mathbf{x}_1, \mathbf{x}_1) & k(\mathbf{x}_1, \mathbf{x}_2) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\
k(\mathbf{x}_2, \mathbf{x}_1) & k(\mathbf{x}_2, \mathbf{x}_2) & \cdots & k(\mathbf{x}_2, \mathbf{x}_n) \\
\vdots & \vdots & \ddots & \vdots \\
k(\mathbf{x}_n, \mathbf{x}_1) & k(\mathbf{x}_n, \mathbf{x}_2) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n)
\end{bmatrix}$$

**与特征矩阵的关系**：

如果$\mathbf{\Phi} = [\phi(\mathbf{x}_1), ..., \phi(\mathbf{x}_n)]^T$，则：

$$\mathbf{K} = \mathbf{\Phi} \mathbf{\Phi}^T$$

### 4. 正定核 (Positive Definite Kernels)

**Mercer定理**：函数$k$是有效核函数 ⟺ 对任意数据集，核矩阵$\mathbf{K}$是**对称半正定**的。

**对称**：$K_{ij} = K_{ji}$（显然，因为$k(\mathbf{x}_i, \mathbf{x}_j) = k(\mathbf{x}_j, \mathbf{x}_i)$）

**半正定**：对所有向量$\mathbf{c} \in \mathbb{R}^n$，

$$\sum_{i=1}^n \sum_{j=1}^n c_i c_j k(\mathbf{x}_i, \mathbf{x}_j) \geq 0$$

等价于：
$$\mathbf{c}^T \mathbf{K} \mathbf{c} \geq 0, \quad \forall \mathbf{c}$$

**验证方法**：
1. 计算核矩阵$\mathbf{K}$
2. 计算特征值：$\lambda_1, ..., \lambda_n$
3. 检查$\lambda_i \geq 0$ for all $i$

**核函数的组合规则**：

如果$k_1, k_2$是有效核，则以下也是有效核：
- $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}')$
- $k(\mathbf{x}, \mathbf{x}') = \alpha k_1(\mathbf{x}, \mathbf{x}')$，$\alpha > 0$
- $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') \cdot k_2(\mathbf{x}, \mathbf{x}')$
- $k(\mathbf{x}, \mathbf{x}') = f(\mathbf{x}) f(\mathbf{x}')$，任意函数$f$

---

## 🎯 中级：核技巧和RKHS

### 1. 核中心化 (Kernel Centering)

**问题**：在特征空间中，数据可能不是零均值的。

**目标**：中心化特征：$\tilde{\phi}(\mathbf{x}) = \phi(\mathbf{x}) - \frac{1}{n}\sum_{i=1}^n \phi(\mathbf{x}_i)$

**核技巧实现**（不显式计算$\phi$）：

$$\tilde{\mathbf{K}} = \mathbf{K} - \mathbf{1}_n \mathbf{K} - \mathbf{K} \mathbf{1}_n + \mathbf{1}_n \mathbf{K} \mathbf{1}_n$$

其中$\mathbf{1}_n = \frac{1}{n} \mathbf{1}\mathbf{1}^T$是$n \times n$的均值投影矩阵。

**简化形式**：

$$\tilde{K}_{ij} = K_{ij} - \frac{1}{n}\sum_{k=1}^n K_{ik} - \frac{1}{n}\sum_{k=1}^n K_{kj} + \frac{1}{n^2}\sum_{k=1}^n\sum_{l=1}^n K_{kl}$$

**验证**：中心化后，$\sum_i \tilde{K}_{ij} = 0$ for all $j$

### 2. 核对齐 (Kernel Alignment)

**目的**：衡量两个核函数（或核与目标）的相似度。

**定义**：核$k_1$和$k_2$的对齐度：

$$A(k_1, k_2) = \frac{\langle \mathbf{K}_1, \mathbf{K}_2 \rangle_F}{\|\mathbf{K}_1\|_F \|\mathbf{K}_2\|_F}$$

其中$\langle \mathbf{A}, \mathbf{B} \rangle_F = \sum_{ij} A_{ij} B_{ij}$是Frobenius内积。

**值域**：$A \in [-1, 1]$
- $A = 1$：完全对齐
- $A = 0$：正交
- $A = -1$：完全反对齐

**应用**：
- 选择最适合任务的核函数
- 多核学习中的核权重

**目标对齐**：

对于分类任务，理想核矩阵：

$$\mathbf{K}_{\text{ideal}} = \mathbf{y}\mathbf{y}^T$$

其中$\mathbf{y} \in \{-1, 1\}^n$是标签向量。

计算$A(\mathbf{K}, \mathbf{K}_{\text{ideal}})$来评估核的质量。

### 3. 核岭回归 (Kernel Ridge Regression)

**原始岭回归**：

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2$$

解：
$$\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

**核化版本**：

在特征空间$\mathcal{H}$中：
$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2$$

**表示定理** (Representer Theorem)：

最优解可以表示为训练样本的线性组合：

$$\mathbf{w}^* = \sum_{i=1}^n \alpha_i \phi(\mathbf{x}_i) = \mathbf{\Phi}^T \boldsymbol{\alpha}$$

代入目标函数：
$$\min_{\boldsymbol{\alpha}} \|\mathbf{y} - \mathbf{K}\boldsymbol{\alpha}\|^2 + \lambda \boldsymbol{\alpha}^T \mathbf{K} \boldsymbol{\alpha}$$

解：
$$\boldsymbol{\alpha} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}$$

**预测**（新样本$\mathbf{x}^*$）：

$$f(\mathbf{x}^*) = \mathbf{w}^T \phi(\mathbf{x}^*) = \sum_{i=1}^n \alpha_i k(\mathbf{x}_i, \mathbf{x}^*)$$

$$= \mathbf{k}^T (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}$$

其中$\mathbf{k} = [k(\mathbf{x}_1, \mathbf{x}^*), ..., k(\mathbf{x}_n, \mathbf{x}^*)]^T$。

**复杂度**：
- 训练：$O(n^3)$（矩阵求逆）
- 预测：$O(n)$（向量乘法）

### 4. 核技巧的通用应用

**一般原则**：如果算法只通过内积访问数据，则可以核化。

**核化步骤**：
1. 将所有$\mathbf{x}_i^T \mathbf{x}_j$替换为$k(\mathbf{x}_i, \mathbf{x}_j)$
2. 将所有$\|\mathbf{x}_i - \mathbf{x}_j\|^2$替换为核距离

**核距离公式**：

$$\|\phi(\mathbf{x}) - \phi(\mathbf{x}')\|^2 = k(\mathbf{x}, \mathbf{x}) - 2k(\mathbf{x}, \mathbf{x}') + k(\mathbf{x}', \mathbf{x}')$$

对于RBF核：$k(\mathbf{x}, \mathbf{x}) = 1$，所以
$$\|\phi(\mathbf{x}) - \phi(\mathbf{x}')\|^2 = 2(1 - k(\mathbf{x}, \mathbf{x}'))$$

### 5. 再生核希尔伯特空间 (RKHS)

**定义**：函数空间$\mathcal{H}$是RKHS，如果：

1. $\mathcal{H}$是希尔伯特空间（完备的内积空间）
2. 存在再生核$k$，使得：
   - $k(\cdot, \mathbf{x}) \in \mathcal{H}$ for all $\mathbf{x}$
   - **再生性质**：$f(\mathbf{x}) = \langle f, k(\cdot, \mathbf{x}) \rangle_\mathcal{H}$ for all $f \in \mathcal{H}$

**Moore-Aronszajn定理**：

对称正定核 ⟺ RKHS

**RKHS范数**：

$$\|f\|_{\mathcal{H}}^2$$

控制函数的复杂度（平滑度）。

**正则化解释**：

$$\min_{f \in \mathcal{H}} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i)) + \lambda \|f\|_{\mathcal{H}}^2$$

- 第一项：经验风险
- 第二项：RKHS范数正则化

**例子**：
- RBF核的RKHS：光滑函数空间
- 线性核的RKHS：线性函数$f(\mathbf{x}) = \mathbf{w}^T\mathbf{x}$，范数$\|\mathbf{w}\|^2$

---

## 🎯 高级：核PCA和核方法应用

### 1. 核主成分分析 (Kernel PCA)

**传统PCA**：

在原始空间中找主成分：
1. 中心化数据
2. 计算协方差矩阵$\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$
3. 特征分解：$\mathbf{C}\mathbf{v} = \lambda \mathbf{v}$

**核PCA**：

在特征空间$\mathcal{H}$中找主成分。

**推导**：

特征空间协方差矩阵：
$$\mathbf{C}_\phi = \frac{1}{n}\sum_{i=1}^n \phi(\mathbf{x}_i) \phi(\mathbf{x}_i)^T$$

特征值问题：
$$\mathbf{C}_\phi \mathbf{v} = \lambda \mathbf{v}$$

**关键洞察**：特征向量必在$\text{span}\{\phi(\mathbf{x}_1), ..., \phi(\mathbf{x}_n)\}$中：

$$\mathbf{v} = \sum_{i=1}^n \alpha_i \phi(\mathbf{x}_i) = \mathbf{\Phi}^T \boldsymbol{\alpha}$$

代入特征值方程：
$$\frac{1}{n}\mathbf{\Phi}^T\mathbf{\Phi}\mathbf{\Phi}^T\boldsymbol{\alpha} = \lambda \mathbf{\Phi}^T\boldsymbol{\alpha}$$

两边左乘$\mathbf{\Phi}$：
$$\frac{1}{n}\mathbf{K}^2\boldsymbol{\alpha} = \lambda \mathbf{K}\boldsymbol{\alpha}$$

简化（假设$\mathbf{K}$可逆）：
$$\mathbf{K}\boldsymbol{\alpha} = n\lambda \boldsymbol{\alpha}$$

**算法**：

1. 计算核矩阵$\mathbf{K}$
2. 中心化$\mathbf{K}$：$\tilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}$，其中$\mathbf{H} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T$
3. 特征分解：$\tilde{\mathbf{K}}\boldsymbol{\alpha}^{(i)} = \lambda_i \boldsymbol{\alpha}^{(i)}$
4. 归一化：$\boldsymbol{\alpha}^{(i)} \leftarrow \frac{\boldsymbol{\alpha}^{(i)}}{\sqrt{\lambda_i}}$（使$\|\mathbf{v}^{(i)}\| = 1$）

**投影**（新样本$\mathbf{x}^*$）：

第$i$个主成分：
$$z_i = \mathbf{v}^{(i)T} \phi(\mathbf{x}^*) = \sum_{j=1}^n \alpha_j^{(i)} k(\mathbf{x}_j, \mathbf{x}^*)$$

**应用**：
- 非线性降维
- 特征提取
- 数据可视化
- 异常检测（重构误差）

### 2. 核k-means聚类

**传统k-means**：

$$\min_{\{\mathbf{c}_k\}, \{r_{ik}\}} \sum_{i=1}^n \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \mathbf{c}_k\|^2$$

**核k-means**：

在特征空间中聚类：
$$\min \sum_{i=1}^n \sum_{k=1}^K r_{ik} \|\phi(\mathbf{x}_i) - \boldsymbol{\mu}_k\|^2$$

**核技巧**：

$$\|\phi(\mathbf{x}_i) - \boldsymbol{\mu}_k\|^2 = k(\mathbf{x}_i, \mathbf{x}_i) - \frac{2}{|C_k|}\sum_{j \in C_k} k(\mathbf{x}_i, \mathbf{x}_j) + \frac{1}{|C_k|^2}\sum_{j,l \in C_k} k(\mathbf{x}_j, \mathbf{x}_l)$$

其中$C_k$是第$k$个簇，$|C_k|$是簇大小。

**算法**：
1. 初始化簇分配
2. 重复：
   - 对每个点$i$，计算到每个簇$k$的距离（使用上述公式）
   - 将点分配到最近的簇
   - 直到收敛

**优势**：能发现非凸形状的簇（如同心圆）。

### 3. 高斯过程回归 (Gaussian Process Regression)

**高斯过程定义**：

函数$f(\mathbf{x})$服从高斯过程，记为$f \sim \mathcal{GP}(m, k)$，如果对任意有限点集$\{\mathbf{x}_1, ..., \mathbf{x}_n\}$：

$$\begin{bmatrix} f(\mathbf{x}_1) \\ \vdots \\ f(\mathbf{x}_n) \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} m(\mathbf{x}_1) \\ \vdots \\ m(\mathbf{x}_n) \end{bmatrix}, \begin{bmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\ \vdots & \ddots & \vdots \\ k(\mathbf{x}_n, \mathbf{x}_1) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n) \end{bmatrix}\right)$$

- $m(\mathbf{x})$：均值函数（通常为0）
- $k(\mathbf{x}, \mathbf{x}')$：协方差函数（核函数！）

**GP回归模型**：

$$y = f(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2)$$

**先验**：
$$f \sim \mathcal{GP}(0, k)$$

**训练数据**：$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$

**预测分布**（新点$\mathbf{x}^*$）：

$$p(f^* | \mathbf{x}^*, \mathcal{D}) = \mathcal{N}(\mu^*, \sigma^{*2})$$

其中：

**预测均值**：
$$\mu^* = \mathbf{k}^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

**预测方差**：
$$\sigma^{*2} = k(\mathbf{x}^*, \mathbf{x}^*) - \mathbf{k}^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}$$

其中：
- $\mathbf{k} = [k(\mathbf{x}_1, \mathbf{x}^*), ..., k(\mathbf{x}_n, \mathbf{x}^*)]^T$
- $\mathbf{K}$是训练数据的核矩阵

**核函数选择**：

- **SE (Squared Exponential) / RBF**：平滑函数
  $$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2l^2}\right)$$

- **Matérn**：可控平滑度
  $$k(\mathbf{x}, \mathbf{x}') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{l}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{l}\right)$$

- **周期核**：周期性函数
  $$k(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{2\sin^2(\pi|\mathbf{x}-\mathbf{x}'|/p)}{l^2}\right)$$

**超参数学习**：

最大化边缘似然（证据）：
$$\log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^T(\mathbf{K}_\theta + \sigma_n^2\mathbf{I})^{-1}\mathbf{y} - \frac{1}{2}\log|\mathbf{K}_\theta + \sigma_n^2\mathbf{I}| - \frac{n}{2}\log(2\pi)$$

使用梯度下降优化$\boldsymbol{\theta} = \{l, \sigma_f, \sigma_n, ...\}$。

**优势**：
- 提供预测不确定性
- 灵活建模（通过核函数）
- 小样本效果好
- 可解释性强

**挑战**：
- $O(n^3)$复杂度
- 核函数选择需要领域知识

### 4. 最大平均差异 (Maximum Mean Discrepancy, MMD)

**问题**：如何度量两个分布$P$和$Q$的差异？

**MMD定义**：

$$\text{MMD}^2(P, Q) = \left\| \mathbb{E}_{\mathbf{x} \sim P}[\phi(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim Q}[\phi(\mathbf{y})] \right\|_{\mathcal{H}}^2$$

即特征空间中均值的距离。

**核技巧计算**：

$$\text{MMD}^2(P, Q) = \mathbb{E}_{P,P}[k(\mathbf{x}, \mathbf{x}')] + \mathbb{E}_{Q,Q}[k(\mathbf{y}, \mathbf{y}')] - 2\mathbb{E}_{P,Q}[k(\mathbf{x}, \mathbf{y})]$$

**无偏估计**（样本$\{\mathbf{x}_i\}_{i=1}^n \sim P$, $\{\mathbf{y}_i\}_{i=1}^m \sim Q$）：

$$\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i \neq j} k(\mathbf{x}_i, \mathbf{x}_j) + \frac{1}{m(m-1)}\sum_{i \neq j} k(\mathbf{y}_i, \mathbf{y}_j) - \frac{2}{nm}\sum_{i,j} k(\mathbf{x}_i, \mathbf{y}_j)$$

**性质**：
- $\text{MMD}(P, Q) = 0$ ⟺ $P = Q$（如果核是特征的）
- $\text{MMD}$是度量（满足三角不等式）

**应用**：
- 两样本检验（是否来自同一分布）
- 生成模型评估（GAN）
- 域适应 (Domain Adaptation)
- 分布匹配

**特征核** (Characteristic Kernel)：

如果$\text{MMD}(P, Q) = 0 \Rightarrow P = Q$，则称核为特征的。

常见特征核：RBF、Laplacian等。

---

## 📊 应用示例

### 1. 支持向量机 (SVM)

**线性SVM**：

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i$$

s.t. $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$

**对偶形式**：

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

s.t. $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$

**核SVM**（将$\mathbf{x}_i^T\mathbf{x}_j$替换为$k(\mathbf{x}_i, \mathbf{x}_j)$）：

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j)$$

**决策函数**：

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b\right)$$

只有支持向量（$\alpha_i > 0$）贡献！

### 2. 核Fisher判别分析

**目标**：在特征空间中最大化类间距离/类内距离。

**核化**：类似核PCA，通过核技巧实现。

**应用**：人脸识别、基因分类。

### 3. 核密度估计

$$\hat{p}(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n k_h(\mathbf{x} - \mathbf{x}_i)$$

其中$k_h$是带宽为$h$的核（通常用高斯核）。

---

## 📚 参考教材

1. **Schölkopf & Smola - Learning with Kernels (2002)**
   - 核方法的圣经级教材

2. **Rasmussen & Williams - Gaussian Processes for Machine Learning (2006)**
   - 高斯过程权威教材（免费在线）

3. **Shawe-Taylor & Cristianini - Kernel Methods for Pattern Analysis (2004)**
   - 理论与应用并重

4. **Hofmann, Schölkopf & Smola - Kernel Methods in Machine Learning (2008)**
   - 综述文章，Annals of Statistics

5. **Berlinet & Thomas-Agnan - Reproducing Kernel Hilbert Spaces (2004)**
   - RKHS理论

---

## 🔗 相关练习

- **线性代数** (Exercise 02): 特征值、SVD
- **优化理论** (Exercise 05): 凸优化、KKT条件
- **统计学** (Exercise 04): 回归分析
- **概率论** (Exercise 03): 概率分布

---

## 💻 实践建议

1. **从简单核开始**：线性核 → 多项式核 → RBF核
2. **可视化核函数**：理解参数（$\gamma$, $d$, $c$）的作用
3. **实现玩具数据集**：XOR、同心圆等非线性可分问题
4. **核参数选择**：
   - 交叉验证选择$\gamma$
   - 网格搜索 + CV
   - 启发式：$\gamma = 1/(d \cdot \text{var}(\mathbf{x}))$
5. **使用工具库**：
   - scikit-learn: `sklearn.svm`, `sklearn.gaussian_process`
   - GPy, GPyTorch: 高斯过程
   - shogun: 多种核方法

**常见陷阱**：
- RBF核$\gamma$过大 → 过拟合（每个样本成为支持向量）
- 核矩阵非正定 → 数值问题（添加正则化$\lambda I$）
- 大规模数据 → $O(n^3)$不可行（使用近似方法：Nyström, Random Features）

**调试技巧**：
- 检查核矩阵对角线是否全为正
- 检查核矩阵是否对称
- 可视化核矩阵（热图）
- 从小数据集开始测试
