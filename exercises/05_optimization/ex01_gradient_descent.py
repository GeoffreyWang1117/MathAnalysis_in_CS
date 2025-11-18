"""
练习 1: 梯度下降算法
====================

学习目标：
- 掌握各种梯度下降变体
- 理解学习率的影响
- 实现动量、AdaGrad、Adam等优化器

任务：
实现现代深度学习中的优化算法
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def gradient_descent(grad_f, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    标准梯度下降

    x_{t+1} = x_t - α * ∇f(x_t)

    参数:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率 α
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        (x_opt, history): 最优点和历史轨迹
    """
    # TODO: 实现梯度下降
    x = x0.copy()
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)

        # 更新
        x_new = x - learning_rate * grad

        history.append(x_new.copy())

        # 检查收敛
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(history)


def momentum_gradient_descent(grad_f, x0, learning_rate=0.01, momentum=0.9,
                               max_iter=1000, tol=1e-6):
    """
    动量梯度下降

    v_{t+1} = β * v_t + ∇f(x_t)
    x_{t+1} = x_t - α * v_{t+1}

    动量帮助加速收敛并避免震荡
    """
    # TODO: 实现动量梯度下降
    x = x0.copy()
    v = np.zeros_like(x0)  # 速度初始化为0
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)

        # 更新速度
        v = momentum * v + grad

        # 更新位置
        x_new = x - learning_rate * v

        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(history)


def adagrad(grad_f, x0, learning_rate=0.1, max_iter=1000, tol=1e-6, epsilon=1e-8):
    """
    AdaGrad优化器

    G_t = G_{t-1} + (∇f(x_t))²
    x_{t+1} = x_t - α / √(G_t + ε) * ∇f(x_t)

    自适应学习率，对稀疏梯度很有效
    """
    # TODO: 实现AdaGrad
    x = x0.copy()
    G = np.zeros_like(x0)  # 累积平方梯度
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)

        # 累积平方梯度
        G = G + grad**2

        # 自适应学习率更新
        x_new = x - learning_rate / (np.sqrt(G) + epsilon) * grad

        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(history)


def adam(grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999,
         max_iter=1000, tol=1e-6, epsilon=1e-8):
    """
    Adam优化器 (Adaptive Moment Estimation)

    m_t = β1 * m_{t-1} + (1-β1) * ∇f(x_t)           # 一阶矩估计
    v_t = β2 * v_{t-1} + (1-β2) * (∇f(x_t))²        # 二阶矩估计
    m̂_t = m_t / (1 - β1^t)                          # 偏差修正
    v̂_t = v_t / (1 - β2^t)
    x_{t+1} = x_t - α * m̂_t / (√v̂_t + ε)

    结合了动量和RMSProp的优点，是目前最流行的优化器
    """
    # TODO: 实现Adam
    x = x0.copy()
    m = np.zeros_like(x0)  # 一阶矩
    v = np.zeros_like(x0)  # 二阶矩
    history = [x.copy()]

    for t in range(1, max_iter + 1):
        grad = grad_f(x)

        # 更新矩估计
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # 偏差修正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # 更新参数
        x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(history)


def line_search_backtracking(f, grad_f, x, p, alpha=1.0, rho=0.5, c=1e-4):
    """
    回溯线搜索（Armijo准则）

    找到满足 Armijo 条件的步长:
    f(x + α*p) ≤ f(x) + c*α*∇f(x)^T*p

    参数:
        f: 目标函数
        grad_f: 梯度函数
        x: 当前点
        p: 搜索方向
        alpha: 初始步长
        rho: 收缩因子
        c: Armijo常数
    """
    # TODO: 实现回溯线搜索
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha = rho * alpha

    return alpha


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    # 测试函数: Rosenbrock函数（香蕉函数）
    # f(x,y) = (1-x)² + 100(y-x²)²
    # 最小值在 (1, 1)，值为 0
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad_rosenbrock(x):
        dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dfdy = 200 * (x[1] - x[0]**2)
        return np.array([dfdx, dfdy])

    x0 = np.array([0.0, 0.0])
    target = np.array([1.0, 1.0])

    print("\n测试 1: 标准梯度下降")
    x_opt, history = gradient_descent(grad_rosenbrock, x0, learning_rate=0.001, max_iter=10000)
    results.append(v.assert_close(x_opt, target, rtol=0.1, name="GD最优点"))

    print("\n测试 2: 动量梯度下降")
    x_opt, history = momentum_gradient_descent(grad_rosenbrock, x0,
                                                learning_rate=0.001, momentum=0.9,
                                                max_iter=10000)
    results.append(v.assert_close(x_opt, target, rtol=0.1, name="Momentum最优点"))

    print("\n测试 3: AdaGrad")
    x_opt, history = adagrad(grad_rosenbrock, x0, learning_rate=0.5, max_iter=10000)
    results.append(v.assert_close(x_opt, target, rtol=0.1, name="AdaGrad最优点"))

    print("\n测试 4: Adam优化器")
    x_opt, history = adam(grad_rosenbrock, x0, learning_rate=0.01, max_iter=10000)
    results.append(v.assert_close(x_opt, target, rtol=0.01, name="Adam最优点"))
    print(f"  收敛到: {x_opt}")

    print("\n测试 5: 回溯线搜索")
    # 简单的二次函数
    f_simple = lambda x: np.sum(x**2)
    grad_simple = lambda x: 2 * x
    x = np.array([2.0, 2.0])
    p = -grad_simple(x)  # 负梯度方向
    alpha = line_search_backtracking(f_simple, grad_simple, x, p)
    results.append(alpha > 0 and alpha <= 1.0)
    print(f"  找到的步长: {alpha:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    return all(results)


if __name__ == "__main__":
    test()
