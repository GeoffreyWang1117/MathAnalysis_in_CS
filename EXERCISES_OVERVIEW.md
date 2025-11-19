# 📚 练习总览

## MathGym - 数学学习平台

**完整练习列表**: 30个练习 | **总题目数**: 90+道题 | **难度体系**: 三级渐进

---

## 🎯 三级难度体系说明

每个练习现在都包含**三道题目**，由浅入深：

- **初级（⭐）**: 基础概念和定义，适合入门学习
- **中级（⭐⭐）**: 定理应用和中等复杂度问题
- **高级（⭐⭐⭐）**: 深入理论和高级应用

所有练习遵循 **UTM (Undergraduate Texts in Mathematics)** 和 **GTM (Graduate Texts in Mathematics)** 教材体系。

---

## 📖 本科数学基础 (UTM Level)

### 📐 01. 微积分（Calculus）- 5个练习

#### ex01_limits.py - 极限计算 ⭐⭐⭐
**三级结构**：
- **初级**: 基本极限的数值计算 (sin(x)/x, 多项式比值)
- **中级**: 重要极限和极限定理 (e的定义, 夹逼定理, 复合函数极限)
- **高级**: Richardson外推法, 收敛速度分析, ε-δ验证

**知识点**: 极限的ε-δ定义 | 重要极限 | 数值逼近
**应用**: 理解函数的局部行为和连续性

#### ex02_derivatives.py - 导数与梯度 ⭐⭐
**知识点**:
- 中心差分法数值微分
- 二元函数梯度计算
- 梯度下降优化
- 牛顿法求根

**应用**: 机器学习优化算法基础

#### ex03_integration.py - 数值积分 ⭐⭐
**知识点**:
- 黎曼和 | 梯形法则 | 辛普森法则
- 蒙特卡洛积分 | 二重积分

**应用**: 概率计算、物理模拟

#### ex04_series.py - 级数与泰勒展开 ⭐⭐
**知识点**:
- 几何级数、调和级数
- 泰勒级数（e^x, sin, cos）
- 莱布尼茨π公式
- 傅里叶级数

**应用**: 函数近似、信号处理

#### ex05_multivariable_calculus.py - 多元微积分 ⭐⭐⭐
**知识点**:
- 偏导数和梯度
- 多元积分
- Jacobian行列式
- 向量场

**应用**: 优化理论、物理场论

---

### 🔲 02. 线性代数（Linear Algebra）- 3个练习

#### ex01_matrix_operations.py - 矩阵运算基础 ⭐⭐⭐
**三级结构**：
- **初级**: 矩阵基本运算 (乘法, 转置, 迹)
- **中级**: 矩阵性质检验 (对称性, 正交性, 范数, 条件数)
- **高级**: 矩阵分解 (Gram-Schmidt正交化, Givens旋转, 幂迭代法)

**知识点**: 矩阵乘法 | 线性变换 | QR分解
**应用**: 所有机器学习算法的基础

#### ex02_eigenvalues.py - 特征值与特征向量 ⭐⭐⭐
**知识点**:
- 幂迭代法 | 特征值验证
- 矩阵对角化 | 矩阵指数
- 谱半径 | 条件数

**应用**: PCA、图算法、动力系统

#### ex03_svd.py - 奇异值分解 ⭐⭐⭐
**知识点**:
- SVD分解 | 低秩近似
- Moore-Penrose伪逆 | PCA主成分分析
- 图像压缩

**应用**: 降维、推荐系统、图像处理

---

### 🎲 03. 概率论（Probability）- 1个练习

#### ex01_distributions.py - 概率分布 ⭐⭐⭐
**三级结构**：
- **初级**: 离散分布 (伯努利, 二项, 泊松)
- **中级**: 连续分布和统计矩 (正态, 指数, CDF, 分位数)
- **高级**: 中心极限定理, 蒙特卡洛积分, KS检验, 重要性采样

**知识点**: 概率分布 | 统计矩 | 大数定律 | CLT
**应用**: 统计建模、随机模拟、机器学习

---

### 📊 04. 统计学（Statistics）- 2个练习

#### ex01_hypothesis_testing.py - 假设检验 ⭐⭐
**知识点**:
- t检验 | 卡方检验
- ANOVA方差分析
- p值和置信区间

**应用**: 实验设计、A/B测试

#### ex02_regression_analysis.py - 回归分析 ⭐⭐⭐
**知识点**:
- 线性回归 | 最小二乘法
- Ridge/Lasso正则化
- 多项式回归 | 模型评估

**应用**: 预测建模、因果推断

---

### 🎯 05. 优化理论（Optimization）- 2个练习

#### ex01_gradient_descent.py - 梯度下降 ⭐⭐
**知识点**:
- 批量梯度下降
- 随机梯度下降 (SGD)
- 动量法 | Adam优化器

**应用**: 深度学习训练

#### ex02_convex_optimization.py - 凸优化 ⭐⭐⭐
**知识点**:
- 凸函数判定
- KKT条件
- 拉格朗日对偶
- 内点法

**应用**: SVM、组合优化

---

### 📐 09. 数值方法（Numerical Methods）- 1个练习

#### ex01_ode_solvers.py - ODE求解器 ⭐⭐⭐
**知识点**:
- Euler方法
- Runge-Kutta方法
- 多步法
- 稳定性分析

**应用**: 物理模拟、动力系统

---

## 🎓 研究生数学 (GTM Level)

### 🌐 06. 复分析（Complex Analysis）- 1个练习

#### ex01_complex_functions.py - 复函数 ⭐⭐⭐
**知识点**:
- Cauchy-Riemann方程
- 复积分 | 留数定理
- 解析函数 | 共形映射

**应用**: 信号处理、量子力学

**参考教材**: Ahlfors - Complex Analysis

---

### 📏 07. 实分析（Real Analysis）- 2个练习

#### ex01_measure_theory.py - 测度论 ⭐⭐⭐
**知识点**:
- Lebesgue测度
- 可测集 | 可测函数
- Cantor集

**应用**: 概率论基础、随机过程

**参考教材**: Rudin - Real and Complex Analysis

#### ex02_lp_spaces.py - Lp空间 ⭐⭐⭐
**知识点**:
- Lp范数 | Hölder不等式
- Minkowski不等式
- 完备性

**应用**: 泛函分析、信号处理

**参考教材**: Folland - Real Analysis

---

### 🧮 08. 泛函分析（Functional Analysis）- 1个练习

#### ex01_banach_spaces.py - Banach空间 ⭐⭐⭐
**三级结构**：
- **初级**: 赋范空间基本概念 (范数公理, 范数等价, 单位球)
- **中级**: 完备性和压缩映射定理 (Cauchy序列, 不动点迭代)
- **高级**: 线性泛函和对偶空间 (算子范数, Riesz表示定理)

**知识点**: 赋范空间 | 完备性 | Hahn-Banach定理
**应用**: 量子力学、优化理论

**参考教材**: Rudin - Functional Analysis | Conway | Brezis

---

### 🌊 10. 随机过程（Stochastic Processes）- 2个练习

#### ex01_markov_chains.py - 马尔可夫链 ⭐⭐⭐
**知识点**:
- 转移矩阵
- 平稳分布
- 遍历性

**应用**: PageRank、MCMC采样

**参考教材**: Norris - Markov Chains

#### ex02_brownian_motion.py - 布朗运动 ⭐⭐⭐
**知识点**:
- Wiener过程
- 随机微积分
- Itô积分

**应用**: 金融数学、物理扩散

**参考教材**: Øksendal - Stochastic Differential Equations

---

### 📡 11. 信息论（Information Theory）- 1个练习

#### ex01_entropy_and_information.py - 熵与信息 ⭐⭐⭐
**知识点**:
- Shannon熵
- 互信息 | KL散度
- 信道容量

**应用**: 数据压缩、通信理论

**参考教材**: Cover & Thomas - Elements of Information Theory

---

### 🔷 12. 拓扑学（Topology）- 1个练习

#### ex01_point_set_topology.py - 点集拓扑基础 ⭐⭐⭐
**三级结构**：
- **初级**: 基本拓扑构造 (拓扑验证, 离散拓扑, 平凡拓扑)
- **中级**: 拓扑性质和连续映射 (子空间拓扑, 连续性, 闭包)
- **高级**: 紧性和连通性 (紧空间, 连通性, 连通分支)

**知识点**: 拓扑空间 | 连续映射 | 紧性 | 连通性
**应用**: 数据分析拓扑(TDA)、机器人学

**参考教材**: Munkres - Topology | Kelley - General Topology

---

### 🔶 13. 抽象代数（Abstract Algebra）- 1个练习

#### ex01_group_theory.py - 群论基础 ⭐⭐⭐
**三级结构**：
- **初级**: 群的定义和基本性质 (群验证, 群的阶, 元素的阶)
- **中级**: 子群、陪集和Lagrange定理 (子群验证, 左陪集, 指数)
- **高级**: 群同态和同构定理 (同态验证, 核, 同构)

**知识点**: 群公理 | Lagrange定理 | 同态基本定理
**应用**: 密码学、对称性分析、编码理论

**参考教材**: Dummit & Foote - Abstract Algebra | Artin - Algebra

---

### 📐 14. 微分几何（Differential Geometry）- 1个练习

#### ex01_curves_and_surfaces.py - 曲线与曲面 ⭐⭐⭐
**三级结构**：
- **初级**: 参数曲线基本性质 (弧长, 切向量, 主法向量)
- **中级**: 曲率和挠率 (曲率计算, 挠率计算, Frenet标架)
- **高级**: 曲面的基本形式 (第一基本形式, 曲面法向量, Gauss曲率)

**知识点**: 曲率 | 挠率 | Frenet标架 | Gauss曲率
**应用**: 计算机图形学、机器人路径规划

**参考教材**: do Carmo - Differential Geometry | Spivak

---

### 🌀 15. 偏微分方程（Partial Differential Equations）- 1个练习

#### ex01_pde_basics.py - 偏微分方程基础 ⭐⭐⭐
**三级结构**：
- **初级**: PDE分类和特征线 (PDE分类, 输运方程, 特征曲线)
- **中级**: 分离变量法 (热方程级数解, 波动方程, Laplace方程)
- **高级**: 有限差分法 (数值求解Laplace方程, 热方程, CFL条件)

**知识点**: PDE分类 | 分离变量法 | 有限差分法
**应用**: 物理建模、图像处理、流体力学

**参考教材**: Evans - PDE | Strauss - PDE: An Introduction

---

## 🤖 AI数学基础 (AI-Specific Mathematics)

### 📊 16. 图论（Graph Theory）- 2个练习

#### ex01_graph_basics.py - 图论基础 ⭐⭐⭐
**三级结构**：
- **初级**: 图的基本表示和性质 (邻接矩阵/表, 度序列, 连通性)
- **中级**: 图的遍历和连通性 (DFS, BFS, 连通分量, 拉普拉斯矩阵)
- **高级**: 最短路径和图算法 (Dijkstra, PageRank, Floyd-Warshall, MST)

**知识点**: 图表示 | 遍历算法 | 最短路径 | PageRank
**应用**: 图神经网络(GNN) | 社交网络分析 | 推荐系统

**参考教材**: West - Introduction to Graph Theory | Newman - Networks

#### ex02_spectral_graph_theory.py - 谱图理论 ⭐⭐⭐
**三级结构**：
- **初级**: 图拉普拉斯矩阵和谱性质 (非归一化/归一化拉普拉斯, 谱计算)
- **中级**: 谱聚类和图切割 (Fiedler向量, 谱聚类, 归一化切割)
- **高级**: 图卷积神经网络数学基础 (GCN传播矩阵, Cheeger不等式, 图傅里叶变换)

**知识点**: 图谱理论 | 谱聚类 | 图卷积 | 图信号处理
**应用**: 图神经网络(GCN) | 谱聚类 | 社区发现

**参考教材**: Chung - Spectral Graph Theory | Kipf & Welling - GCN论文

---

### 📐 17. 矩阵微积分（Matrix Calculus）- 1个练习

#### ex01_matrix_derivatives.py - 矩阵导数 ⭐⭐⭐
**三级结构**：
- **初级**: 向量和矩阵的基本导数 (二次型梯度, 线性函数雅可比, Softmax梯度)
- **中级**: 雅可比矩阵和海塞矩阵 (数值雅可比, Hessian, 链式法则, 方向导数)
- **高级**: 反向传播和自动微分 (计算图, MLP反向传播, Fisher信息矩阵)

**知识点**: 矩阵求导 | 雅可比/Hessian | 反向传播 | 自动微分
**应用**: 深度学习反向传播 | 优化算法 | 变分推断

**参考教材**: Petersen & Pedersen - Matrix Cookbook | Goodfellow - Deep Learning Ch.6

---

### 🔷 18. 张量代数（Tensor Algebra）- 1个练习

#### ex01_tensor_operations.py - 张量运算 ⭐⭐⭐
**三级结构**：
- **初级**: 张量的基本操作 (重塑, 转置, 外积, 切片)
- **中级**: 张量缩并和Einstein求和 (张量缩并, Einstein求和, 模式n乘积)
- **高级**: 张量分解 (张量展开/折叠, HOSVD, CP分解, 重构误差)

**知识点**: 张量运算 | Einstein求和 | Tucker分解 | CP分解
**应用**: 深度学习框架 | 张量分解 | 推荐系统 | 多线性代数

**参考教材**: Kolda & Bader - Tensor Decompositions | Cichocki - Tensor Methods

---

### 📈 19. 凸分析（Convex Analysis）- 1个练习

#### ex01_convex_sets_and_functions.py - 凸集与凸函数 ⭐⭐⭐
**三级结构**：
- **初级**: 凸集的判定和性质 (凸组合, 凸包, 单纯形投影)
- **中级**: 凸函数和共轭函数 (凸性验证, Hessian判定, Fenchel共轭, 邻近算子)
- **高级**: 次梯度和凸优化对偶 (次梯度, 拉格朗日函数, 对偶函数, KKT条件)

**知识点**: 凸集/凸函数 | Fenchel共轭 | 次梯度 | KKT条件
**应用**: 凸优化 | 机器学习优化 | 信号处理 | SVM

**参考教材**: Boyd & Vandenberghe - Convex Optimization | Rockafellar - Convex Analysis

---

## 📊 学习路径推荐

### 🔰 新手路径 (3-6个月)
1. **微积分**: ex01 → ex02 → ex03
2. **线性代数**: ex01 → ex02
3. **概率论**: ex01

### 🎯 进阶路径 (6-12个月)
完成新手路径后：
4. **统计学**: ex01 → ex02
5. **优化**: ex01 → ex02
6. **数值方法**: ex01
7. **复分析**: ex01

### 🚀 专家路径 (12-24个月)
完成进阶路径后：
8. **实分析**: ex01 → ex02
9. **泛函分析**: ex01
10. **拓扑学**: ex01
11. **抽象代数**: ex01
12. **微分几何**: ex01
13. **随机过程**: ex01 → ex02
14. **信息论**: ex01
15. **偏微分方程**: ex01

---

## 📈 难度分布

- **⭐ 初级题目**: 30道 (每个练习的第一部分)
- **⭐⭐ 中级题目**: 30道 (每个练习的第二部分)
- **⭐⭐⭐ 高级题目**: 30道 (每个练习的第三部分)

**总计**: 90+道题目，覆盖150+个数学概念

---

## 🎓 教材对照表

| 练习主题 | 对应教材系列 | 推荐教材 |
|---------|------------|---------|
| 微积分 | UTM | Spivak - Calculus |
| 线性代数 | UTM | Strang - Linear Algebra |
| 概率论 | UTM | Ross - A First Course in Probability |
| 实分析 | GTM 3 | Rudin - Principles of Mathematical Analysis |
| 复分析 | GTM 11 | Ahlfors - Complex Analysis |
| 泛函分析 | GTM 3 | Rudin - Functional Analysis |
| 拓扑学 | GTM 27 | Munkres - Topology |
| 抽象代数 | GTM 73 | Dummit & Foote - Abstract Algebra |
| 微分几何 | GTM 73 | do Carmo - Differential Geometry |
| 偏微分方程 | GTM 19 | Evans - Partial Differential Equations |
| 随机过程 | GTM | Norris - Markov Chains |

---

## 🔍 按应用领域查找

### 🧠 深度学习/神经网络
- **核心数学**:
  - 线性代数 (全部)
  - 矩阵微积分 (ex01) - 反向传播必备
  - 张量代数 (ex01) - PyTorch/TensorFlow基础
  - 凸分析 (ex01) - 优化理论
- **优化**:
  - 优化 (全部)
  - 微积分: ex02 (梯度)
- **其他**:
  - 概率论: ex01
  - 信息论: ex01

### 📊 图神经网络(GNN)
- **核心数学**:
  - 图论: ex01 (图基础), ex02 (谱图理论)
  - 线性代数: ex01, ex02 (特征值)
  - 凸分析 (ex01)
- **应用**:
  - 社交网络分析
  - 推荐系统
  - 知识图谱

### 🖼️ 计算机视觉
- 线性代数: ex03 (SVD), 张量代数
- 微积分: ex05 (多元)
- 凸优化 (ex01)
- 偏微分方程: ex01
- 矩阵微积分: ex01

### 📝 自然语言处理
- 线性代数: ex02 (特征值), ex03 (SVD)
- 张量代数: ex01
- 概率论: ex01
- 信息论: ex01
- 优化: ex01 (SGD)
- 矩阵微积分: ex01

### 🎮 强化学习
- 线性代数: ex01, ex02
- 概率论: ex01
- 随机过程 (全部)
- 优化 (全部)
- 图论: ex01 (MDP图结构)

### 🎨 计算机图形学
- 线性代数 (全部)
- 微积分: ex05
- 微分几何: ex01
- 张量代数: ex01

---

## ✅ 完成标准

每个练习完成后应该：
- [ ] 所有TODO部分已实现
- [ ] 通过所有测试用例
- [ ] 理解三个难度等级的核心概念
- [ ] 能够解释代码背后的数学原理

---

**最后更新**: 2025年
**总学时估计**: 60-80小时（精通全部内容）
**难度梯度**: UTM (本科) → GTM (研究生) → AI专项数学
