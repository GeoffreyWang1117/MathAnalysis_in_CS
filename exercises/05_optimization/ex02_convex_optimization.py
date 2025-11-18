"""
练习 2: 凸优化
==============

学习目标：
- 理解凸函数和凸集
- 掌握KKT条件
- 实现约束优化问题

任务：
解决凸优化问题
"""

import numpy as np
from scipy.optimize import minimize, linprog
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def is_convex_function(f, x_samples):
    """
    数值检验函数的凸性

    凸函数满足: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

    参数:
        f: 函数
        x_samples: 测试样本点列表
    """
    # TODO: 实现凸性检验
    n_tests = 100
    is_convex = True

    for _ in range(n_tests):
        # 随机选择两个点
        i, j = np.random.choice(len(x_samples), 2, replace=False)
        x, y = x_samples[i], x_samples[j]

        # 随机选择 λ
        lambda_val = np.random.rand()

        # 检查凸性条件
        z = lambda_val * x + (1 - lambda_val) * y
        lhs = f(z)
        rhs = lambda_val * f(x) + (1 - lambda_val) * f(y)

        if lhs > rhs + 1e-6:  # 容差
            is_convex = False
            break

    return is_convex


def projected_gradient_descent(f, grad_f, x0, project_fn, learning_rate=0.01,
                                max_iter=1000, tol=1e-6):
    """
    投影梯度下降法（用于约束优化）

    x_{t+1} = Proj(x_t - α∇f(x_t))

    参数:
        f: 目标函数
        grad_f: 梯度函数
        x0: 初始点
        project_fn: 投影函数（投影到可行域）
        learning_rate: 学习率
        max_iter: 最大迭代次数
        tol: 收敛容差
    """
    # TODO: 实现投影梯度下降
    x = x0.copy()
    history = [x.copy()]

    for i in range(max_iter):
        # 梯度步
        grad = grad_f(x)
        x_unconstrained = x - learning_rate * grad

        # 投影到可行域
        x_new = project_fn(x_unconstrained)

        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(history)


def project_to_simplex(x):
    """
    投影到单纯形: {x | x ≥ 0, Σx_i = 1}

    这在概率分布优化中很常见
    """
    # TODO: 实现单纯形投影
    # 简化算法: 投影到标准单纯形
    n = len(x)
    x_sorted = np.sort(x)[::-1]

    # 找到阈值
    cumsum = np.cumsum(x_sorted)
    rho = np.where(x_sorted > (cumsum - 1) / np.arange(1, n + 1))[0][-1]
    theta = (cumsum[rho] - 1) / (rho + 1)

    return np.maximum(x - theta, 0)


def proximal_gradient_descent(grad_f, prox_g, x0, learning_rate=0.01,
                               max_iter=1000, tol=1e-6):
    """
    近端梯度下降法

    用于求解: min f(x) + g(x)
    其中 f 可微，g 有简单的近端算子

    x_{t+1} = prox_{αg}(x_t - α∇f(x_t))

    参数:
        grad_f: f 的梯度
        prox_g: g 的近端算子
        x0: 初始点
    """
    # TODO: 实现近端梯度下降
    x = x0.copy()
    history = [x.copy()]

    for i in range(max_iter):
        # 梯度步
        grad = grad_f(x)
        x_intermediate = x - learning_rate * grad

        # 近端步
        x_new = prox_g(x_intermediate, learning_rate)

        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, np.array(history)


def soft_threshold(x, lambda_param):
    """
    软阈值算子（L1正则的近端算子）

    prox_{λ|·|}(x) = sign(x) * max(|x| - λ, 0)

    用于LASSO等稀疏优化问题
    """
    # TODO: 实现软阈值
    return np.sign(x) * np.maximum(np.abs(x) - lambda_param, 0)


def linear_programming_example():
    """
    线性规划示例

    maximize: 3x + 2y
    subject to:
        2x + y ≤ 18
        2x + 3y ≤ 42
        3x + y ≤ 24
        x ≥ 0, y ≥ 0

    使用 scipy.optimize.linprog
    """
    # TODO: 使用linprog求解
    # linprog求最小值，所以目标函数取负
    c = [-3, -2]  # 目标函数系数（取负）

    # 不等式约束: A_ub @ x <= b_ub
    A_ub = np.array([[2, 1],
                     [2, 3],
                     [3, 1]])
    b_ub = np.array([18, 42, 24])

    # 变量边界: x >= 0, y >= 0
    bounds = [(0, None), (0, None)]

    # 求解
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    return result.x, -result.fun  # 返回最优解和最优值


def quadratic_programming(Q, p, x0):
    """
    二次规划: min (1/2)x^T Q x + p^T x

    使用scipy.optimize.minimize

    参数:
        Q: Hessian矩阵（正定）
        p: 线性项系数
        x0: 初始点
    """
    # TODO: 实现二次规划求解
    def objective(x):
        return 0.5 * x.T @ Q @ x + p.T @ x

    def gradient(x):
        return Q @ x + p

    result = minimize(objective, x0, jac=gradient, method='CG')

    return result.x


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: 凸性检验")
    # 凸函数: f(x) = x^2
    f_convex = lambda x: x**2
    samples = [np.array([x]) for x in np.linspace(-5, 5, 20)]
    is_conv = is_convex_function(f_convex, samples)
    results.append(is_conv)
    print(f"  f(x) = x^2 是凸函数: {'✓ 是' if is_conv else '✗ 否'}")

    print("\n测试 2: 单纯形投影")
    x = np.array([1.0, 2.0, 3.0])
    x_proj = project_to_simplex(x)
    # 检查约束
    is_valid = np.all(x_proj >= 0) and abs(np.sum(x_proj) - 1.0) < 1e-6
    results.append(is_valid)
    print(f"  投影结果: {x_proj}")
    print(f"  满足约束: {'✓ 是' if is_valid else '✗ 否'}")

    print("\n测试 3: 软阈值算子")
    x = np.array([3.0, -2.0, 0.5, -0.3])
    lambda_param = 1.0
    result = soft_threshold(x, lambda_param)
    expected = np.array([2.0, -1.0, 0.0, 0.0])
    results.append(v.assert_close(result, expected, name="软阈值"))

    print("\n测试 4: 线性规划")
    x_opt, f_opt = linear_programming_example()
    print(f"  最优解: x = {x_opt}")
    print(f"  最优值: f = {f_opt:.2f}")
    # 检查解的合理性
    results.append(f_opt > 0 and len(x_opt) == 2)

    print("\n测试 5: 二次规划")
    # min (1/2)x^T Q x + p^T x
    Q = np.array([[2, 0], [0, 2]])  # 2*I
    p = np.array([-2, -2])
    x0 = np.array([0.0, 0.0])
    x_opt = quadratic_programming(Q, p, x0)
    expected = np.array([1.0, 1.0])  # 解析解
    results.append(v.assert_close(x_opt, expected, rtol=0.01, name="二次规划"))

    print("\n测试 6: 投影梯度下降")
    # 在单位球内最小化 f(x) = (x-a)^T(x-a)
    a = np.array([2.0, 2.0])
    f = lambda x: np.sum((x - a)**2)
    grad_f = lambda x: 2 * (x - a)
    project = lambda x: x / max(1.0, np.linalg.norm(x))  # 投影到单位球

    x0 = np.array([0.0, 0.0])
    x_opt, _ = projected_gradient_descent(f, grad_f, x0, project,
                                          learning_rate=0.1, max_iter=1000)

    # 最优解应该在单位球边界上
    norm = np.linalg.norm(x_opt)
    results.append(v.assert_close(norm, 1.0, rtol=0.01, name="投影GD（单位球约束）"))

    return all(results)


if __name__ == "__main__":
    test()
