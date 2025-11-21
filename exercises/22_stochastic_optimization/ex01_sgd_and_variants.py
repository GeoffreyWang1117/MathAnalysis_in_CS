"""
练习 1: 随机优化与深度学习优化器
==================================

本练习包含3道题目，由浅入深：
- 初级：SGD和Mini-batch梯度下降
- 中级：Momentum和Nesterov加速
- 高级：自适应学习率方法 (AdaGrad, RMSProp, Adam)

学习目标：
- 理解随机梯度下降的理论
- 掌握动量方法的原理
- 实现现代深度学习优化器
- 理解学习率调度策略

应用领域：
- 深度学习训练
- 大规模机器学习
- 在线学习
- 强化学习

参考教材：
- Bottou - Stochastic Gradient Descent Tricks
- Ruder - An Overview of Gradient Descent Optimization Algorithms
- Goodfellow et al. - Deep Learning (Chapter 8)
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：SGD和Mini-batch梯度下降
# ============================================================================

def batch_gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iter=100, tol=1e-6):
    """
    初级 - 批量梯度下降 (Batch GD)

    使用全部数据计算梯度:
    x_{t+1} = x_t - η ∇f(x_t)

    参数:
        f: 目标函数
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率η
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        (最优解, 损失历史)
    """
    # TODO: 实现批量梯度下降
    x = x0.copy()
    loss_history = []

    for iteration in range(max_iter):
        loss = f(x)
        loss_history.append(loss)

        grad = grad_f(x)

        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break

        # 更新
        x = x - learning_rate * grad

    return x, np.array(loss_history)


def stochastic_gradient_descent(f_single, grad_f_single, x0, data, learning_rate=0.01,
                                 n_epochs=10):
    """
    初级 - 随机梯度下降 (SGD)

    每次使用单个样本:
    x_{t+1} = x_t - η ∇f_i(x_t)

    参数:
        f_single: 单样本损失函数 f(x, data_i)
        grad_f_single: 单样本梯度函数
        x0: 初始点
        data: 数据集
        learning_rate: 学习率
        n_epochs: 训练轮数

    返回:
        (最优解, 损失历史)
    """
    # TODO: 实现SGD
    x = x0.copy()
    loss_history = []

    n_samples = len(data)

    for epoch in range(n_epochs):
        # 打乱数据
        indices = np.random.permutation(n_samples)

        epoch_loss = 0
        for idx in indices:
            # 单个样本的梯度
            grad = grad_f_single(x, data[idx])

            # 更新
            x = x - learning_rate * grad

            # 记录损失
            epoch_loss += f_single(x, data[idx])

        loss_history.append(epoch_loss / n_samples)

    return x, np.array(loss_history)


def mini_batch_sgd(f_batch, grad_f_batch, x0, data, batch_size=32,
                   learning_rate=0.01, n_epochs=10):
    """
    初级 - Mini-batch SGD

    使用小批量数据:
    x_{t+1} = x_t - η (1/B) Σ_{i∈B} ∇f_i(x_t)

    参数:
        f_batch: 批量损失函数
        grad_f_batch: 批量梯度函数
        x0: 初始点
        data: 数据集
        batch_size: 批量大小
        learning_rate: 学习率
        n_epochs: 训练轮数

    返回:
        (最优解, 损失历史)
    """
    # TODO: 实现Mini-batch SGD
    x = x0.copy()
    loss_history = []

    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        # 打乱数据
        indices = np.random.permutation(n_samples)

        epoch_loss = 0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_data = [data[i] for i in batch_indices]

            # 批量梯度
            grad = grad_f_batch(x, batch_data)

            # 更新
            x = x - learning_rate * grad

            # 记录损失
            epoch_loss += f_batch(x, batch_data) * len(batch_data)

        loss_history.append(epoch_loss / n_samples)

    return x, np.array(loss_history)


# ============================================================================
# 中级题目：Momentum和Nesterov加速
# ============================================================================

def sgd_with_momentum(grad_f, x0, learning_rate=0.01, momentum=0.9,
                      max_iter=100, tol=1e-6):
    """
    中级 - 动量SGD

    动量方法累积过去的梯度:
    v_{t+1} = β v_t + ∇f(x_t)
    x_{t+1} = x_t - η v_{t+1}

    帮助加速收敛并减少震荡

    参数:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率η
        momentum: 动量系数β
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        (最优解, 速度历史)
    """
    # TODO: 实现动量SGD
    x = x0.copy()
    v = np.zeros_like(x)  # 速度初始化为0

    velocity_history = []

    for iteration in range(max_iter):
        grad = grad_f(x)

        # 更新速度
        v = momentum * v + grad

        # 更新参数
        x = x - learning_rate * v

        velocity_history.append(np.linalg.norm(v))

        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break

    return x, np.array(velocity_history)


def nesterov_accelerated_gradient(grad_f, x0, learning_rate=0.01, momentum=0.9,
                                   max_iter=100, tol=1e-6):
    """
    中级 - Nesterov加速梯度 (NAG)

    "向前看"的动量方法:
    v_{t+1} = β v_t + ∇f(x_t - β v_t)
    x_{t+1} = x_t - η v_{t+1}

    比标准动量方法更稳定

    参数:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率
        momentum: 动量系数
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        最优解
    """
    # TODO: 实现NAG
    x = x0.copy()
    v = np.zeros_like(x)

    for iteration in range(max_iter):
        # 在"向前看"的位置计算梯度
        grad = grad_f(x - momentum * v)

        # 更新速度
        v = momentum * v + learning_rate * grad

        # 更新参数
        x = x - v

        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break

    return x


def learning_rate_decay(initial_lr, iteration, decay_type='exponential', decay_rate=0.95):
    """
    中级 - 学习率衰减

    常见策略:
    - 指数衰减: lr = lr_0 * γ^t
    - 阶梯衰减: lr = lr_0 * γ^⌊t/k⌋
    - 逆时间衰减: lr = lr_0 / (1 + kt)

    参数:
        initial_lr: 初始学习率
        iteration: 当前迭代次数
        decay_type: 衰减类型
        decay_rate: 衰减率

    返回:
        当前学习率
    """
    # TODO: 实现学习率衰减
    if decay_type == 'exponential':
        lr = initial_lr * (decay_rate ** iteration)
    elif decay_type == 'step':
        # 每10步衰减一次
        lr = initial_lr * (decay_rate ** (iteration // 10))
    elif decay_type == 'inverse_time':
        lr = initial_lr / (1 + decay_rate * iteration)
    else:
        lr = initial_lr

    return lr


# ============================================================================
# 高级题目：自适应学习率方法
# ============================================================================

def adagrad(grad_f, x0, learning_rate=0.01, max_iter=100, epsilon=1e-8, tol=1e-6):
    """
    高级 - AdaGrad优化器

    自适应学习率，对每个参数使用不同的学习率:
    G_t = G_{t-1} + g_t ⊙ g_t
    x_{t+1} = x_t - η / √(G_t + ε) ⊙ g_t

    适用于稀疏梯度

    参数:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 初始学习率
        max_iter: 最大迭代次数
        epsilon: 数值稳定项
        tol: 收敛容差

    返回:
        最优解
    """
    # TODO: 实现AdaGrad
    x = x0.copy()
    G = np.zeros_like(x)  # 累积平方梯度

    for iteration in range(max_iter):
        grad = grad_f(x)

        # 累积平方梯度
        G += grad ** 2

        # 自适应学习率更新
        x = x - learning_rate * grad / (np.sqrt(G) + epsilon)

        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break

    return x


def rmsprop(grad_f, x0, learning_rate=0.001, decay_rate=0.9,
            max_iter=100, epsilon=1e-8, tol=1e-6):
    """
    高级 - RMSProp优化器

    使用移动平均的平方梯度:
    E[g²]_t = β E[g²]_{t-1} + (1-β) g_t²
    x_{t+1} = x_t - η / √(E[g²]_t + ε) g_t

    解决AdaGrad学习率单调递减的问题

    参数:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率
        decay_rate: 衰减率β
        max_iter: 最大迭代次数
        epsilon: 数值稳定项
        tol: 收敛容差

    返回:
        最优解
    """
    # TODO: 实现RMSProp
    x = x0.copy()
    E_g_sq = np.zeros_like(x)  # 平方梯度的移动平均

    for iteration in range(max_iter):
        grad = grad_f(x)

        # 更新平方梯度的移动平均
        E_g_sq = decay_rate * E_g_sq + (1 - decay_rate) * (grad ** 2)

        # 更新参数
        x = x - learning_rate * grad / (np.sqrt(E_g_sq) + epsilon)

        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break

    return x


def adam(grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999,
         max_iter=100, epsilon=1e-8, tol=1e-6):
    """
    高级 - Adam优化器

    结合动量和RMSProp:
    m_t = β₁ m_{t-1} + (1-β₁) g_t         (一阶矩估计)
    v_t = β₂ v_{t-1} + (1-β₂) g_t²        (二阶矩估计)
    m̂_t = m_t / (1 - β₁^t)                (偏差修正)
    v̂_t = v_t / (1 - β₂^t)
    x_{t+1} = x_t - η m̂_t / (√v̂_t + ε)

    目前最流行的深度学习优化器

    参数:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        max_iter: 最大迭代次数
        epsilon: 数值稳定项
        tol: 收敛容差

    返回:
        (最优解, 一阶矩, 二阶矩)
    """
    # TODO: 实现Adam
    x = x0.copy()
    m = np.zeros_like(x)  # 一阶矩
    v = np.zeros_like(x)  # 二阶矩

    for t in range(1, max_iter + 1):
        grad = grad_f(x)

        # 更新一阶和二阶矩估计
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # 偏差修正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # 更新参数
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break

    return x, m, v


def adamw(grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999,
          weight_decay=0.01, max_iter=100, epsilon=1e-8, tol=1e-6):
    """
    高级 - AdamW优化器

    Adam with decoupled weight decay:
    与Adam相同，但权重衰减分离:
    x_{t+1} = x_t - η (m̂_t / (√v̂_t + ε) + λ x_t)

    在Transformer等模型中表现更好

    参数:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率
        beta1, beta2: 矩估计衰减率
        weight_decay: 权重衰减系数λ
        max_iter: 最大迭代次数
        epsilon: 数值稳定项
        tol: 收敛容差

    返回:
        最优解
    """
    # TODO: 实现AdamW
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, max_iter + 1):
        grad = grad_f(x)

        # 更新矩估计
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # 偏差修正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # 更新参数（解耦权重衰减）
        x = x - learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * x)

        # 检查收敛
        if np.linalg.norm(grad) < tol:
            break

    return x


def gradient_clipping(grad, max_norm=1.0):
    """
    高级 - 梯度裁剪

    防止梯度爆炸:
    if ||g|| > max_norm:
        g = g * max_norm / ||g||

    参数:
        grad: 梯度向量
        max_norm: 最大范数

    返回:
        裁剪后的梯度
    """
    # TODO: 实现梯度裁剪
    grad_norm = np.linalg.norm(grad)

    if grad_norm > max_norm:
        grad = grad * max_norm / grad_norm

    return grad


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    validator = Validator()
    results = []

    # 测试函数: f(x) = x^2 + y^2 (最小值在(0,0))
    f = lambda x: np.sum(x**2)
    grad_f = lambda x: 2 * x

    print("\n" + "="*60)
    print("初级题目：SGD和Mini-batch梯度下降")
    print("="*60)

    print("\n测试 1.1: 批量梯度下降")
    x0 = np.array([5.0, 5.0])
    x_opt, loss_hist = batch_gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=100)
    # 应该收敛到(0,0)附近
    results.append(validator.assert_close(np.linalg.norm(x_opt), 0.0, atol=0.1, name="BGD收敛"))

    print("\n测试 1.2: Mini-batch SGD")
    # 简化的数据集
    data = [np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([0.5, -0.5])]
    f_batch = lambda x, batch: np.mean([np.sum((x - d)**2) for d in batch])
    grad_f_batch = lambda x, batch: np.mean([2*(x - d) for d in batch], axis=0)

    x_opt_mb, loss_hist_mb = mini_batch_sgd(f_batch, grad_f_batch, x0, data,
                                              batch_size=2, learning_rate=0.1, n_epochs=20)
    # 损失应该下降
    results.append(loss_hist_mb[-1] < loss_hist_mb[0])
    print(f"  初始损失: {loss_hist_mb[0]:.4f}, 最终损失: {loss_hist_mb[-1]:.4f}")
    print(f"  {'✓ 通过（损失下降）' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("中级题目：Momentum和Nesterov加速")
    print("="*60)

    print("\n测试 2.1: 动量SGD")
    x_opt_momentum, vel_hist = sgd_with_momentum(grad_f, x0, learning_rate=0.1,
                                                   momentum=0.9, max_iter=100)
    results.append(validator.assert_close(np.linalg.norm(x_opt_momentum), 0.0, atol=0.1,
                                  name="动量SGD收敛"))

    print("\n测试 2.2: Nesterov加速梯度")
    x_opt_nag = nesterov_accelerated_gradient(grad_f, x0, learning_rate=0.1,
                                                momentum=0.9, max_iter=100)
    results.append(validator.assert_close(np.linalg.norm(x_opt_nag), 0.0, atol=0.1,
                                  name="NAG收敛"))

    print("\n测试 2.3: 学习率衰减")
    lr_0 = 0.1
    lr_exp = learning_rate_decay(lr_0, 10, 'exponential', 0.95)
    # 指数衰减后应该更小
    results.append(lr_exp < lr_0)
    print(f"  初始学习率: {lr_0}, 衰减后: {lr_exp:.6f}")
    print(f"  {'✓ 通过（学习率减小）' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：自适应学习率方法")
    print("="*60)

    print("\n测试 3.1: AdaGrad")
    x_opt_adagrad = adagrad(grad_f, x0, learning_rate=1.0, max_iter=200)
    results.append(validator.assert_close(np.linalg.norm(x_opt_adagrad), 0.0, atol=0.1,
                                  name="AdaGrad收敛"))

    print("\n测试 3.2: RMSProp")
    x_opt_rmsprop = rmsprop(grad_f, x0, learning_rate=0.1, max_iter=100)
    results.append(validator.assert_close(np.linalg.norm(x_opt_rmsprop), 0.0, atol=0.1,
                                  name="RMSProp收敛"))

    print("\n测试 3.3: Adam")
    x_opt_adam, m, v = adam(grad_f, x0, learning_rate=0.1, max_iter=100)
    results.append(validator.assert_close(np.linalg.norm(x_opt_adam), 0.0, atol=0.1,
                                  name="Adam收敛"))
    print(f"  最优解: {x_opt_adam}")

    print("\n测试 3.4: AdamW")
    x_opt_adamw = adamw(grad_f, x0, learning_rate=0.1, weight_decay=0.01, max_iter=100)
    results.append(validator.assert_close(np.linalg.norm(x_opt_adamw), 0.0, atol=0.1,
                                  name="AdamW收敛"))

    print("\n测试 3.5: 梯度裁剪")
    large_grad = np.array([10.0, 10.0])
    clipped_grad = gradient_clipping(large_grad, max_norm=1.0)
    # 裁剪后范数应该≤1
    results.append(validator.assert_close(np.linalg.norm(clipped_grad), 1.0, atol=0.01,
                                  name="梯度裁剪"))

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
