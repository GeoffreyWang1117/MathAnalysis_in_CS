# 🎓 MathGym - 数学分析交互式学习平台

<div align="center">

**面向AI+CS研究生的数学编程练习系统**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*类似 Rustlings 的数学学习平台，通过填空式编程掌握数学知识*

</div>

---

## 📖 简介

**MathGym** 是一个交互式数学编程学习平台，专为计算机科学特别是AI方向的研究生设计。通过填写代码空白（类似 Rustlings），学生可以：

- ✅ 掌握AI+CS领域必需的数学知识
- ✅ 通过Python实现数学概念
- ✅ 使用NumPy、SciPy、Matplotlib等科学计算工具
- ✅ 自动测试和即时反馈
- ✅ 渐进式学习，从基础到高级

---

## 🎯 知识体系

### 📚 **本科阶段数学**

1. **微积分（Calculus）**
   - 极限与连续性
   - 导数与梯度
   - 数值积分（黎曼和、梯形法则、辛普森法则）
   - 级数与泰勒展开

2. **线性代数（Linear Algebra）**
   - 矩阵运算
   - 特征值与特征向量（幂迭代法）
   - SVD与降维
   - PCA主成分分析

3. **概率论（Probability）**
   - 常见概率分布
   - 中心极限定理
   - 蒙特卡洛方法

4. **统计学（Statistics）**
   - 假设检验（t检验、卡方检验）
   - ANOVA方差分析
   - Bootstrap自助法

### 🎓 **研究生阶段数学**

5. **优化理论（Optimization）**
   - 梯度下降及其变体（SGD、Momentum、Adam）
   - 凸优化
   - KKT条件
   - 线性规划与二次规划

6. **复变函数（Complex Analysis）**
   - 复数运算与性质
   - Cauchy-Riemann方程
   - 莫比乌斯变换
   - 留数定理

7. **数值方法（Numerical Methods）**
   - ODE求解器（Euler、RK4）
   - 刚性方程
   - Lorenz混沌系统

8. **实分析（Real Analysis）** - 待扩展
9. **泛函分析（Functional Analysis）** - 待扩展
10. **随机过程（Stochastic Processes）** - 待扩展

---

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/MathAnalysis_in_CS.git
cd MathAnalysis_in_CS

# 安装依赖
pip install -r requirements.txt
```

### 开始学习

```bash
# 运行学习平台
python src/runner.py
```

你会看到：

```
🎓 欢迎来到 MathGym - 数学分析学习平台！
============================================================
这是一个交互式数学编程学习系统，类似于 Rustlings
请填写代码中的 TODO 部分，完成所有练习
============================================================

📊 学习进度
============================================================
已完成: 0/30 (0.0%)
进度条: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

主题进度：
  01_calculus: 0/4
  02_linear_algebra: 0/3
  03_probability: 0/1
  04_statistics: 0/1
  05_optimization: 0/2
  06_complex_analysis: 0/1
  ...
============================================================

📝 下一个练习: exercises/01_calculus/ex01_limits.py
主题: 01_calculus
```

### 编辑练习文件

打开 `exercises/01_calculus/ex01_limits.py`，你会看到：

```python
def limit_sin_x_over_x(epsilon=1e-10):
    """
    计算 lim(x→0) sin(x)/x
    """
    # TODO: 实现极限计算
    x = epsilon
    result = np.sin(x) / x  # 请修改这一行
    return result
```

填写代码，保存后再次运行：

```bash
python src/runner.py
```

---

## 📂 项目结构

```
MathAnalysis_in_CS/
├── exercises/                  # 📝 练习题目
│   ├── 01_calculus/           # 微积分
│   │   ├── ex01_limits.py
│   │   ├── ex02_derivatives.py
│   │   ├── ex03_integration.py
│   │   └── ex04_series.py
│   ├── 02_linear_algebra/     # 线性代数
│   │   ├── ex01_matrix_operations.py
│   │   ├── ex02_eigenvalues.py
│   │   └── ex03_svd.py
│   ├── 03_probability/        # 概率论
│   ├── 04_statistics/         # 统计学
│   ├── 05_optimization/       # 优化理论
│   ├── 06_complex_analysis/   # 复变函数
│   ├── 09_numerical_methods/  # 数值方法
│   └── ...
├── src/                       # 🔧 核心系统
│   ├── runner.py              # 主运行器
│   ├── validator.py           # 验证工具
│   └── utils.py               # 工具函数
├── tests/                     # 🧪 测试用例
├── README.md                  # 📖 本文档
└── requirements.txt           # 📦 依赖列表
```

---

## 🛠️ 命令行工具

```bash
# 查看所有练习
python src/runner.py list

# 查看学习进度
python src/runner.py progress

# 重置进度
python src/runner.py reset

# 交互式学习（默认）
python src/runner.py
```

---

## 💡 练习示例

### 示例 1: 极限计算

```python
# exercises/01_calculus/ex01_limits.py

def limit_sin_x_over_x(epsilon=1e-10):
    """计算 lim(x→0) sin(x)/x"""
    x = epsilon
    result = np.sin(x) / x
    return result
```

### 示例 2: 梯度下降

```python
# exercises/05_optimization/ex01_gradient_descent.py

def gradient_descent(grad_f, x0, learning_rate=0.01, max_iter=1000):
    """标准梯度下降"""
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - learning_rate * grad  # 核心更新规则
    return x
```

### 示例 3: SVD分解

```python
# exercises/02_linear_algebra/ex03_svd.py

def low_rank_approximation(A, k):
    """低秩近似"""
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_k = S.copy()
    S_k[k:] = 0  # 只保留前k个奇异值
    A_k = U @ np.diag(S_k) @ VT
    return A_k
```

---

## 🎯 学习路径建议

### 初学者路径
1. 01_calculus → 02_linear_algebra → 03_probability → 04_statistics

### 进阶路径
1. 05_optimization → 09_numerical_methods → 06_complex_analysis

### AI/ML专项
1. 02_linear_algebra (SVD, PCA) → 05_optimization (Adam, SGD) → 03_probability

---

## 📊 进度跟踪

系统自动跟踪你的学习进度，保存在 `.mathgym_progress.json` 文件中：

```json
{
  "completed": [
    "exercises/01_calculus/ex01_limits.py",
    "exercises/01_calculus/ex02_derivatives.py"
  ],
  "last_updated": "2025-11-18T12:00:00"
}
```

---

## 🧪 测试系统

每个练习都包含自动测试：

```python
@create_test_decorator
def test():
    v = Validator()
    results = []

    print("\n测试 1: lim(x→0) sin(x)/x = 1")
    result = limit_sin_x_over_x()
    results.append(v.assert_close(result, 1.0, name="极限"))

    return all(results)
```

---

## 📦 技术栈

- **Python 3.8+**: 编程语言
- **NumPy**: 数值计算
- **SciPy**: 科学计算
- **Matplotlib**: 数据可视化
- **Seaborn**: 统计可视化
- **SymPy**: 符号计算（可选）

---

## 🤝 贡献指南

欢迎贡献新的练习和改进！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/new-exercise`
3. 提交更改：`git commit -m 'Add new exercise'`
4. 推送分支：`git push origin feature/new-exercise`
5. 提交 Pull Request

### 添加新练习

1. 在适当的目录创建 `ex0X_topic.py`
2. 包含清晰的文档字符串
3. 实现 `test()` 函数
4. 使用 `Validator` 类进行验证

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- 灵感来源于 [Rustlings](https://github.com/rust-lang/rustlings)
- 感谢所有贡献者和使用者

---

## 📧 联系方式

- 问题反馈：[GitHub Issues](https://github.com/yourusername/MathAnalysis_in_CS/issues)
- 邮箱：your.email@example.com

---

<div align="center">

**🎓 开始你的数学编程之旅！**

*学习数学，掌握AI+CS的核心基础*

</div>
