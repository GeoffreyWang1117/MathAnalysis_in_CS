"""
练习 1: 矩阵微积分
==================

本练习包含3道题目，由浅入深：
- 初级：向量和矩阵的基本导数
- 中级：雅可比矩阵和海塞矩阵
- 高级：反向传播和自动微分

学习目标：
- 掌握矩阵对矩阵的求导
- 理解雅可比矩阵和海塞矩阵
- 实现神经网络的反向传播
- 理解自动微分的原理

应用领域：
- 深度学习反向传播 (Backpropagation)
- 优化算法 (梯度下降)
- 敏感性分析
- 变分推断

参考教材：
- Petersen & Pedersen - The Matrix Cookbook
- Magnus & Neudecker - Matrix Differential Calculus
- Goodfellow et al. - Deep Learning (Chapter 6)
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：向量和矩阵的基本导数
# ============================================================================

def gradient_quadratic_form(x, A, b):
    """
    初级 - 二次型的梯度

    计算 f(x) = x^T A x + b^T x 的梯度

    导数公式:
    ∇f = (A + A^T)x + b

    如果A是对称的，则 ∇f = 2Ax + b

    参数:
        x: n维向量
        A: n×n矩阵
        b: n维向量

    返回:
        梯度向量
    """
    # TODO: 实现二次型梯度
    # f(x) = x^T A x + b^T x
    # ∇f = (A + A^T)x + b
    gradient = (A + A.T) @ x + b
    return gradient


def jacobian_linear(A, x):
    """
    初级 - 线性函数的雅可比矩阵

    计算 f(x) = Ax 的雅可比矩阵

    对于 y = Ax，其中 A 是 m×n 矩阵:
    J[i,j] = ∂y_i/∂x_j = A[i,j]

    因此 J = A

    参数:
        A: m×n矩阵
        x: n维向量

    返回:
        雅可比矩阵 (m×n)
    """
    # TODO: 实现线性函数雅可比
    # 线性函数 f(x) = Ax 的雅可比就是 A
    return A


def gradient_softmax(x):
    """
    初级 - Softmax函数的梯度

    Softmax: σ(x)_i = exp(x_i) / Σ_j exp(x_j)

    雅可比矩阵: J[i,j] = σ_i(δ_ij - σ_j)
    其中 δ_ij 是Kronecker delta

    参数:
        x: n维向量

    返回:
        雅可比矩阵 (n×n)
    """
    # TODO: 实现Softmax梯度
    # 先计算softmax
    exp_x = np.exp(x - np.max(x))  # 数值稳定
    softmax = exp_x / np.sum(exp_x)

    # 雅可比矩阵
    n = len(x)
    jacobian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i, j] = softmax[i] * (1 - softmax[i])
            else:
                jacobian[i, j] = -softmax[i] * softmax[j]

    return jacobian


# ============================================================================
# 中级题目：雅可比矩阵和海塞矩阵
# ============================================================================

def jacobian_numerical(f, x, epsilon=1e-7):
    """
    中级 - 数值雅可比矩阵

    使用有限差分法计算雅可比矩阵:
    J[i,j] ≈ (f_i(x + ε*e_j) - f_i(x - ε*e_j)) / (2ε)

    参数:
        f: 函数 R^n -> R^m
        x: n维输入向量
        epsilon: 差分步长

    返回:
        m×n 雅可比矩阵
    """
    # TODO: 实现数值雅可比
    x = np.array(x, dtype=float)
    n = len(x)

    # 先计算f(x)来确定输出维度
    f_x = f(x)
    m = len(f_x) if hasattr(f_x, '__len__') else 1

    if m == 1:
        # 标量输出 - 返回梯度
        jacobian = np.zeros(n)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            f_plus = f(x + epsilon * e_j)
            f_minus = f(x - epsilon * e_j)
            jacobian[j] = (f_plus - f_minus) / (2 * epsilon)
        return jacobian
    else:
        # 向量输出
        jacobian = np.zeros((m, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            f_plus = f(x + epsilon * e_j)
            f_minus = f(x - epsilon * e_j)
            jacobian[:, j] = (f_plus - f_minus) / (2 * epsilon)
        return jacobian


def hessian_matrix(f, x, epsilon=1e-5):
    """
    中级 - 海塞矩阵 (Hessian Matrix)

    海塞矩阵是二阶导数矩阵:
    H[i,j] = ∂²f/∂x_i∂x_j

    用于:
    - 二阶优化方法 (牛顿法)
    - 凸性判定 (正定 => 严格凸)
    - 泰勒展开

    参数:
        f: 标量函数 R^n -> R
        x: n维向量
        epsilon: 差分步长

    返回:
        n×n 海塞矩阵
    """
    # TODO: 实现海塞矩阵计算
    x = np.array(x, dtype=float)
    n = len(x)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # 计算二阶偏导数
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1.0
            e_j[j] = 1.0

            # 使用中心差分
            f_pp = f(x + epsilon * e_i + epsilon * e_j)
            f_pm = f(x + epsilon * e_i - epsilon * e_j)
            f_mp = f(x - epsilon * e_i + epsilon * e_j)
            f_mm = f(x - epsilon * e_i - epsilon * e_j)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)

    return hessian


def chain_rule_composition(g_jacobian, f_jacobian):
    """
    中级 - 链式法则

    对于复合函数 h(x) = g(f(x)):
    J_h = J_g @ J_f

    这是反向传播的核心

    参数:
        g_jacobian: g的雅可比矩阵 (k×m)
        f_jacobian: f的雅可比矩阵 (m×n)

    返回:
        复合函数的雅可比矩阵 (k×n)
    """
    # TODO: 实现链式法则
    return g_jacobian @ f_jacobian


def directional_derivative(grad_f, direction):
    """
    中级 - 方向导数

    在方向v上的方向导数:
    D_v f = ∇f · v

    参数:
        grad_f: 梯度向量
        direction: 方向向量 (已归一化)

    返回:
        方向导数（标量）
    """
    # TODO: 实现方向导数
    # 归一化方向
    v = direction / np.linalg.norm(direction)
    return np.dot(grad_f, v)


# ============================================================================
# 高级题目：反向传播和自动微分
# ============================================================================

class ComputationGraph:
    """
    高级 - 计算图实现自动微分

    实现简单的前向传播和反向传播
    用于神经网络训练
    """

    def __init__(self):
        self.operations = []
        self.gradients = {}

    def forward_linear(self, x, W, b, name):
        """
        线性层前向传播: y = Wx + b

        参数:
            x: 输入向量
            W: 权重矩阵
            b: 偏置向量
            name: 操作名称

        返回:
            输出向量
        """
        # TODO: 实现线性层前向传播
        y = W @ x + b
        self.operations.append({
            'type': 'linear',
            'name': name,
            'inputs': {'x': x, 'W': W, 'b': b},
            'output': y
        })
        return y

    def forward_relu(self, x, name):
        """
        ReLU激活前向传播: y = max(0, x)

        参数:
            x: 输入向量
            name: 操作名称

        返回:
            输出向量
        """
        # TODO: 实现ReLU前向传播
        y = np.maximum(0, x)
        self.operations.append({
            'type': 'relu',
            'name': name,
            'inputs': {'x': x},
            'output': y
        })
        return y

    def backward(self, loss_grad):
        """
        反向传播计算梯度

        从输出的梯度开始，反向计算所有参数的梯度

        参数:
            loss_grad: 损失函数对输出的梯度

        返回:
            参数梯度字典
        """
        # TODO: 实现反向传播
        grad = loss_grad
        param_grads = {}

        # 反向遍历操作
        for op in reversed(self.operations):
            if op['type'] == 'relu':
                # ReLU梯度: grad * (x > 0)
                x = op['inputs']['x']
                grad = grad * (x > 0)

            elif op['type'] == 'linear':
                # 线性层梯度
                x = op['inputs']['x']
                W = op['inputs']['W']
                b = op['inputs']['b']

                # ∂L/∂W = grad ⊗ x^T
                param_grads[f"{op['name']}_W"] = np.outer(grad, x)

                # ∂L/∂b = grad
                param_grads[f"{op['name']}_b"] = grad

                # ∂L/∂x = W^T @ grad
                grad = W.T @ grad

        return param_grads


def backprop_mlp(X, y, W1, b1, W2, b2, learning_rate=0.01):
    """
    高级 - 多层感知机(MLP)的反向传播

    网络结构: X -> Linear -> ReLU -> Linear -> Output
    损失函数: MSE

    参数:
        X: 输入 (n_features,)
        y: 目标 (n_outputs,)
        W1, b1: 第一层参数
        W2, b2: 第二层参数
        learning_rate: 学习率

    返回:
        (loss, 更新后的参数)
    """
    # TODO: 实现MLP反向传播

    # 前向传播
    z1 = W1 @ X + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = W2 @ a1 + b2
    output = z2

    # 损失函数 (MSE)
    loss = 0.5 * np.sum((output - y)**2)

    # 反向传播
    # ∂L/∂z2
    dz2 = output - y

    # 第二层梯度
    dW2 = np.outer(dz2, a1)
    db2 = dz2

    # ∂L/∂a1
    da1 = W2.T @ dz2

    # ReLU梯度
    dz1 = da1 * (z1 > 0)

    # 第一层梯度
    dW1 = np.outer(dz1, X)
    db1 = dz1

    # 参数更新
    W1_new = W1 - learning_rate * dW1
    b1_new = b1 - learning_rate * db1
    W2_new = W2 - learning_rate * dW2
    b2_new = b2 - learning_rate * db2

    return loss, (W1_new, b1_new, W2_new, b2_new)


def fisher_information_matrix(log_likelihood_grad, samples):
    """
    高级 - Fisher信息矩阵

    Fisher信息矩阵衡量参数的信息量:
    I(θ) = E[(∇log p(x|θ))(∇log p(x|θ))^T]

    用于:
    - 自然梯度下降
    - Cramér-Rao界
    - 模型不确定性估计

    参数:
        log_likelihood_grad: 对数似然梯度函数
        samples: 样本列表

    返回:
        Fisher信息矩阵
    """
    # TODO: 实现Fisher信息矩阵
    # 计算所有样本的梯度
    grads = [log_likelihood_grad(sample) for sample in samples]
    grads = np.array(grads)

    # Fisher信息矩阵 = 梯度外积的期望
    n_samples = len(samples)
    fisher = np.zeros((grads.shape[1], grads.shape[1]))

    for grad in grads:
        fisher += np.outer(grad, grad)

    fisher /= n_samples

    return fisher


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：向量和矩阵的基本导数")
    print("="*60)

    print("\n测试 1.1: 二次型梯度")
    A = np.array([[2, 1], [1, 2]])
    x = np.array([1, 1])
    b = np.array([1, 1])
    grad = gradient_quadratic_form(x, A, b)
    # f(x) = x^T A x + b^T x
    # ∇f = (A + A^T)x + b = 2Ax + b (A对称)
    expected_grad = 2 * A @ x + b
    results.append(v.assert_array_equal(grad, expected_grad, name="二次型梯度"))

    print("\n测试 1.2: 线性函数雅可比")
    A = np.array([[1, 2], [3, 4], [5, 6]])
    x = np.array([1, 1])
    J = jacobian_linear(A, x)
    results.append(v.assert_array_equal(J, A, name="线性函数雅可比"))

    print("\n测试 1.3: Softmax梯度")
    x = np.array([1.0, 2.0, 3.0])
    J_softmax = gradient_softmax(x)
    # 验证雅可比矩阵每行和为0 (因为softmax和为1)
    row_sums = np.sum(J_softmax, axis=1)
    results.append(v.assert_array_equal(row_sums, np.zeros(3), atol=1e-10, name="Softmax雅可比行和"))

    print("\n" + "="*60)
    print("中级题目：雅可比矩阵和海塞矩阵")
    print("="*60)

    print("\n测试 2.1: 数值雅可比矩阵")
    # 测试函数 f(x) = [x1^2, x1*x2]
    f = lambda x: np.array([x[0]**2, x[0]*x[1]])
    x = np.array([2.0, 3.0])
    J_num = jacobian_numerical(f, x)
    # 解析雅可比: [[2*x1, 0], [x2, x1]]
    J_analytic = np.array([[2*x[0], 0], [x[1], x[0]]])
    results.append(v.assert_array_equal(J_num, J_analytic, rtol=1e-5, name="数值雅可比"))

    print("\n测试 2.2: 海塞矩阵")
    # 测试函数 f(x) = x1^2 + x1*x2 + x2^2
    f = lambda x: x[0]**2 + x[0]*x[1] + x[1]**2
    x = np.array([1.0, 1.0])
    H = hessian_matrix(f, x)
    # 解析海塞矩阵: [[2, 1], [1, 2]]
    H_analytic = np.array([[2, 1], [1, 2]])
    results.append(v.assert_array_equal(H, H_analytic, rtol=1e-4, name="海塞矩阵"))

    print("\n测试 2.3: 链式法则")
    # g: R^2 -> R^3, f: R^3 -> R^2
    J_g = np.array([[1, 2], [3, 4], [5, 6]])
    J_f = np.array([[1, 0, 1], [0, 1, 1]])
    J_composed = chain_rule_composition(J_f, J_g)
    expected = J_f @ J_g
    results.append(v.assert_array_equal(J_composed, expected, name="链式法则"))

    print("\n测试 2.4: 方向导数")
    grad = np.array([3, 4])
    direction = np.array([1, 0])
    dir_deriv = directional_derivative(grad, direction)
    results.append(v.assert_close(dir_deriv, 3.0, name="方向导数"))

    print("\n" + "="*60)
    print("高级题目：反向传播和自动微分")
    print("="*60)

    print("\n测试 3.1: 计算图自动微分")
    graph = ComputationGraph()
    x = np.array([1.0, 2.0])
    W = np.array([[1, 0], [0, 1], [1, 1]])
    b = np.array([0.5, 0.5, 0.5])

    # 前向传播
    y1 = graph.forward_linear(x, W, b, 'layer1')
    y2 = graph.forward_relu(y1, 'relu1')

    # 反向传播
    loss_grad = np.array([1.0, 1.0, 1.0])
    grads = graph.backward(loss_grad)

    # 检查梯度是否存在
    results.append('layer1_W' in grads and 'layer1_b' in grads)
    print(f"  计算图梯度: {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.2: MLP反向传播")
    # 简单的2-3-2网络
    np.random.seed(42)
    X = np.array([1.0, 0.5])
    y_true = np.array([0.8, 0.2])
    W1 = np.random.randn(3, 2) * 0.1
    b1 = np.zeros(3)
    W2 = np.random.randn(2, 3) * 0.1
    b2 = np.zeros(2)

    loss_before, (W1_new, b1_new, W2_new, b2_new) = backprop_mlp(X, y_true, W1, b1, W2, b2, 0.1)

    # 用新参数计算损失
    z1 = W1_new @ X + b1_new
    a1 = np.maximum(0, z1)
    z2 = W2_new @ a1 + b2_new
    loss_after = 0.5 * np.sum((z2 - y_true)**2)

    # 损失应该下降
    results.append(loss_after < loss_before)
    print(f"  损失下降: {loss_before:.4f} -> {loss_after:.4f}")
    print(f"  {'✓ 通过（损失下降）' if results[-1] else '✗ 失败'}")

    print("\n测试 3.3: Fisher信息矩阵")
    # 简单的高斯分布
    # log p(x|μ,σ²) = -0.5*log(2πσ²) - (x-μ)²/(2σ²)
    # 对μ的梯度: (x-μ)/σ²
    def log_lik_grad(x, mu=0, sigma=1):
        return np.array([(x - mu) / sigma**2])

    samples = np.random.randn(1000)
    fisher = fisher_information_matrix(lambda x: log_lik_grad(x), samples)
    # 对于高斯分布，Fisher信息 = 1/σ² = 1
    results.append(v.assert_close(fisher[0, 0], 1.0, rtol=0.1, name="Fisher信息矩阵"))

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
