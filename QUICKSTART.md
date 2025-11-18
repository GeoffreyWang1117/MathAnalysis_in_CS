# 🚀 快速开始指南

## 5分钟上手 MathGym

### 步骤 1: 安装依赖

```bash
pip install numpy scipy matplotlib seaborn
```

### 步骤 2: 查看所有练习

```bash
python src/runner.py list
```

输出：
```
📚 所有练习：
  ⬜ exercises/01_calculus/ex01_limits.py
  ⬜ exercises/01_calculus/ex02_derivatives.py
  ⬜ exercises/01_calculus/ex03_integration.py
  ...
```

### 步骤 3: 运行学习平台

```bash
python src/runner.py
```

输出会显示：
```
📝 下一个练习: exercises/01_calculus/ex01_limits.py
主题: 01_calculus
```

### 步骤 4: 编辑练习文件

打开 `exercises/01_calculus/ex01_limits.py`

**练习内容示例：**
```python
def limit_sin_x_over_x(epsilon=1e-10):
    """
    计算 lim(x→0) sin(x)/x
    提示：这是一个重要的极限，结果应该接近 1
    """
    # TODO: 实现极限计算
    x = epsilon
    result = np.sin(x) / x  # 这一行已经正确实现
    return result
```

### 步骤 5: 测试你的答案

方法1 - 直接运行练习文件：
```bash
python exercises/01_calculus/ex01_limits.py
```

方法2 - 使用平台运行器：
```bash
python src/runner.py
```

### 步骤 6: 查看测试结果

**通过示例：**
```
============================================================
🧪 运行测试...
============================================================

测试 1: lim(x→0) sin(x)/x = 1
✓ sin(x)/x 的极限正确

测试 2: lim(n→∞) (1 + 1/n)^n = e
✓ e 的极限定义正确

测试 3: lim(x→0) x^2 * sin(1/x) = 0
✓ 夹逼定理极限正确
============================================================
🎉 所有测试通过！
============================================================
```

**未通过示例：**
```
============================================================
🧪 运行测试...
============================================================

测试 1: lim(x→0) sin(x)/x = 1
✗ sin(x)/x 的极限不正确
  期望: 1.0
  实际: 0.5
============================================================
❌ 部分测试未通过，请检查您的实现
============================================================
```

---

## 📝 学习建议

### 第一天：微积分基础
1. `ex01_limits.py` - 极限计算
2. `ex02_derivatives.py` - 导数与梯度
3. `ex03_integration.py` - 数值积分

### 第二天：线性代数
1. `ex01_matrix_operations.py` - 矩阵运算
2. `ex02_eigenvalues.py` - 特征值分解
3. `ex03_svd.py` - 奇异值分解与PCA

### 第三天：概率统计
1. `ex01_distributions.py` - 概率分布
2. `ex01_hypothesis_testing.py` - 假设检验

### 第四天及以后：高级主题
1. 优化理论 - 梯度下降、凸优化
2. 数值方法 - ODE求解
3. 复变函数 - 复数运算

---

## 💡 常见问题

### Q1: 如何知道哪些部分需要填写？
A: 查找代码中的 `# TODO:` 注释和 `pass  # 请实现` 标记

### Q2: 测试总是失败怎么办？
A:
1. 仔细阅读函数文档字符串中的提示
2. 检查数学公式是否正确实现
3. 查看测试用例了解期望输出
4. 运行单个练习文件查看详细错误信息

### Q3: 可以跳过某些练习吗？
A: 可以！但建议按顺序学习，因为后面的练习可能依赖前面的概念

### Q4: 如何查看进度？
A:
```bash
python src/runner.py progress
```

### Q5: 如何重置进度？
A:
```bash
python src/runner.py reset
```

---

## 🎯 完整示例：完成第一个练习

### 1. 查看练习
```bash
cat exercises/01_calculus/ex01_limits.py
```

### 2. 理解任务
- 计算 lim(x→0) sin(x)/x
- 使用数值方法逼近极限
- 预期结果：1.0

### 3. 实现代码
代码已经提供了正确实现：
```python
x = epsilon
result = np.sin(x) / x
return result
```

### 4. 运行测试
```bash
python exercises/01_calculus/ex01_limits.py
```

### 5. 确认通过
看到 "🎉 所有测试通过！" 表示完成！

### 6. 继续下一个
```bash
python src/runner.py
```

---

## 🔧 调试技巧

### 技巧1: 打印中间结果
```python
def my_function(x):
    result = x ** 2
    print(f"调试: x={x}, result={result}")  # 添加调试信息
    return result
```

### 技巧2: 使用 NumPy 的验证函数
```python
import numpy as np
# 检查数组形状
print(f"Shape: {array.shape}")
# 检查数值范围
print(f"Min: {np.min(array)}, Max: {np.max(array)}")
```

### 技巧3: 单独测试小部分
```python
# 将复杂函数分解为小步骤
def complex_function(x):
    step1 = x ** 2
    step2 = np.sin(step1)
    step3 = step2 / x
    print(f"Step1: {step1}, Step2: {step2}, Step3: {step3}")
    return step3
```

---

## 📚 推荐学习资源

1. **NumPy官方文档**: https://numpy.org/doc/
2. **SciPy教程**: https://docs.scipy.org/doc/scipy/tutorial/
3. **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/

---

**🎓 开始你的数学编程之旅吧！**
