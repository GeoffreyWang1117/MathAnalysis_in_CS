"""
练习 1: 凸集与凸函数
====================

本练习包含3道题目，由浅入深：
- 初级：凸集的判定和基本性质
- 中级：凸函数和共轭函数
- 高级：次梯度和凸优化对偶

学习目标：
- 理解凸集和凸函数的定义
- 掌握凸性判定方法
- 理解Fenchel共轭和对偶性
- 掌握次梯度和KKT条件

应用领域：
- 凸优化 (Convex Optimization)
- 机器学习优化
- 信号处理
- 控制理论

参考教材：
- Boyd & Vandenberghe - Convex Optimization
- Rockafellar - Convex Analysis
- Bubeck - Convex Optimization: Algorithms and Complexity
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：凸集的判定和基本性质
# ============================================================================

def is_convex_combination(x, points, weights):
    """
    初级 - 验证点是否为凸组合

    凸组合: x = Σᵢ λᵢ xᵢ，其中 λᵢ ≥ 0, Σᵢ λᵢ = 1

    参数:
        x: 待验证点
        points: 点列表
        weights: 权重列表

    返回:
        布尔值，是否为凸组合
    """
    # TODO: 实现凸组合验证
    # 检查权重非负且和为1
    if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
        return False

    # 检查x = Σ λᵢ xᵢ
    combination = np.zeros_like(x)
    for point, weight in zip(points, weights):
        combination += weight * point

    return np.allclose(x, combination)


def convex_hull_2d(points):
    """
    初级 - 计算2D点集的凸包

    凸包: 包含所有点的最小凸集

    使用Graham扫描算法

    参数:
        points: n×2点数组

    返回:
        凸包顶点索引列表
    """
    # TODO: 实现凸包算法（简化版）
    points = np.array(points)
    n = len(points)

    if n < 3:
        return list(range(n))

    # 找到最下面的点（y最小）
    start = np.argmin(points[:, 1])

    # 按极角排序
    def polar_angle(p, origin):
        dx = p[0] - origin[0]
        dy = p[1] - origin[1]
        return np.arctan2(dy, dx)

    origin = points[start]
    angles = [polar_angle(p, origin) for p in points]
    sorted_indices = sorted(range(n), key=lambda i: (angles[i], np.linalg.norm(points[i] - origin)))

    # Graham扫描
    hull = []
    for i in sorted_indices:
        while len(hull) >= 2:
            # 检查是否左转
            o, a, b = hull[-2], hull[-1], i
            cross = np.cross(points[a] - points[o], points[b] - points[o])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)

    return hull


def project_onto_simplex(v):
    """
    初级 - 投影到概率单纯形

    概率单纯形: Δⁿ = {x : xᵢ ≥ 0, Σᵢ xᵢ = 1}

    找到v在单纯形上的欧几里得投影

    参数:
        v: n维向量

    返回:
        投影后的向量
    """
    # TODO: 实现单纯形投影
    n = len(v)
    u = np.sort(v)[::-1]  # 降序排列

    # 找到合适的阈值
    cumsum = np.cumsum(u)
    rho = np.where((u * np.arange(1, n+1) > (cumsum - 1)))[0][-1]

    theta = (cumsum[rho] - 1) / (rho + 1)

    # 投影
    w = np.maximum(v - theta, 0)

    return w


# ============================================================================
# 中级题目：凸函数和共轭函数
# ============================================================================

def is_convex_function(f, domain, epsilon=1e-5):
    """
    中级 - 验证函数是否凸

    凸函数: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
    对所有 x, y ∈ dom(f), λ ∈ [0,1]

    参数:
        f: 函数
        domain: 定义域内的测试点列表
        epsilon: 数值容差

    返回:
        布尔值，是否凸
    """
    # TODO: 实现凸性验证
    # 随机采样检验
    np.random.seed(42)
    n_tests = 50

    for _ in range(n_tests):
        # 随机选择两个点和λ
        if len(domain) < 2:
            return True

        x, y = domain[np.random.choice(len(domain), 2, replace=False)]
        lambda_val = np.random.rand()

        # 计算凸组合点
        z = lambda_val * x + (1 - lambda_val) * y

        # 验证凸性不等式
        f_z = f(z)
        f_convex_bound = lambda_val * f(x) + (1 - lambda_val) * f(y)

        if f_z > f_convex_bound + epsilon:
            return False

    return True


def check_convexity_by_hessian(hessian):
    """
    中级 - 通过Hessian矩阵判断凸性

    二次可微函数f是凸的当且仅当:
    Hessian矩阵 ∇²f(x) 在定义域内处处半正定

    参数:
        hessian: Hessian矩阵

    返回:
        (is_convex, min_eigenvalue)
    """
    # TODO: 实现基于Hessian的凸性判定
    eigenvalues = np.linalg.eigvalsh(hessian)
    min_eigenvalue = np.min(eigenvalues)

    # 半正定: 所有特征值 ≥ 0
    is_convex = min_eigenvalue >= -1e-10  # 数值容差

    return is_convex, min_eigenvalue


def fenchel_conjugate(f, y, domain):
    """
    中级 - Fenchel共轭函数

    共轭函数: f*(y) = sup_x (y^T x - f(x))

    性质:
    - f** = f (当f是凸且闭的)
    - Fenchel不等式: f(x) + f*(y) ≥ x^T y

    参数:
        f: 原函数
        y: 共轭函数的参数
        domain: 原函数定义域的采样点

    返回:
        f*(y) 的近似值
    """
    # TODO: 实现Fenchel共轭
    # 计算 sup_x (y^T x - f(x))
    conjugate_values = []

    for x in domain:
        value = np.dot(y, x) - f(x)
        conjugate_values.append(value)

    # 取最大值
    f_conjugate = np.max(conjugate_values)

    return f_conjugate


def proximal_operator(f_grad, x, lambda_param, learning_rate=0.01, iterations=100):
    """
    中级 - 邻近算子(Proximal Operator)

    prox_λf(v) = argmin_x (f(x) + (1/2λ)||x - v||²)

    用于:
    - 邻近梯度下降
    - ADMM算法
    - 正则化优化

    参数:
        f_grad: f的梯度函数
        x: 初始点
        lambda_param: 邻近参数λ
        learning_rate: 梯度下降学习率
        iterations: 迭代次数

    返回:
        邻近算子的近似值
    """
    # TODO: 实现邻近算子（使用梯度下降近似）
    v = x.copy()
    x_prox = x.copy()

    for _ in range(iterations):
        # 梯度: ∇f(x) + (x - v)/λ
        grad = f_grad(x_prox) + (x_prox - v) / lambda_param

        # 梯度下降更新
        x_prox = x_prox - learning_rate * grad

    return x_prox


# ============================================================================
# 高级题目：次梯度和凸优化对偶
# ============================================================================

def subgradient(f, x, epsilon=1e-6):
    """
    高级 - 计算次梯度

    次梯度 g ∈ ∂f(x) 满足:
    f(y) ≥ f(x) + g^T(y - x) for all y

    对于不可微点，次梯度是一个集合

    参数:
        f: 凸函数
        x: 计算点
        epsilon: 数值差分步长

    返回:
        一个次梯度（数值近似）
    """
    # TODO: 实现次梯度计算
    # 使用单边差分近似
    n = len(x)
    subgrad = np.zeros(n)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0

        # 单边导数
        forward_diff = (f(x + epsilon * e_i) - f(x)) / epsilon
        backward_diff = (f(x) - f(x - epsilon * e_i)) / epsilon

        # 取平均作为次梯度
        subgrad[i] = (forward_diff + backward_diff) / 2

    return subgrad


def lagrangian(f, constraints, x, lambda_vec, mu_vec):
    """
    高级 - 拉格朗日函数

    标准形式优化问题:
    minimize f(x)
    subject to gᵢ(x) ≤ 0, hⱼ(x) = 0

    拉格朗日函数:
    L(x, λ, μ) = f(x) + Σᵢ λᵢgᵢ(x) + Σⱼ μⱼhⱼ(x)

    参数:
        f: 目标函数
        constraints: {'ineq': [g1, g2, ...], 'eq': [h1, h2, ...]}
        x: 决策变量
        lambda_vec: 不等式约束的拉格朗日乘子(≥0)
        mu_vec: 等式约束的拉格朗日乘子

    返回:
        拉格朗日函数值
    """
    # TODO: 实现拉格朗日函数
    L = f(x)

    # 不等式约束
    if 'ineq' in constraints:
        for i, g in enumerate(constraints['ineq']):
            L += lambda_vec[i] * g(x)

    # 等式约束
    if 'eq' in constraints:
        for j, h in enumerate(constraints['eq']):
            L += mu_vec[j] * h(x)

    return L


def dual_function(f, constraints, lambda_vec, mu_vec, x_domain):
    """
    高级 - 对偶函数

    对偶函数:
    g(λ, μ) = inf_x L(x, λ, μ)

    性质:
    - g是凹函数（即使原问题非凸）
    - 弱对偶: g(λ, μ) ≤ p* (原问题最优值)

    参数:
        f: 目标函数
        constraints: 约束字典
        lambda_vec: 不等式乘子
        mu_vec: 等式乘子
        x_domain: x的采样点

    返回:
        对偶函数值
    """
    # TODO: 实现对偶函数
    # 计算 inf_x L(x, λ, μ)
    lagrangian_values = []

    for x in x_domain:
        L_val = lagrangian(f, constraints, x, lambda_vec, mu_vec)
        lagrangian_values.append(L_val)

    # 取下确界
    dual_val = np.min(lagrangian_values)

    return dual_val


def check_kkt_conditions(x, lambda_vec, mu_vec, grad_f, grad_constraints, constraints):
    """
    高级 - 检验KKT条件

    Karush-Kuhn-Tucker (KKT) 条件是凸优化最优性的充要条件:

    1. 定常性: ∇f(x*) + Σᵢ λᵢ∇gᵢ(x*) + Σⱼ μⱼ∇hⱼ(x*) = 0
    2. 原问题可行性: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
    3. 对偶可行性: λᵢ ≥ 0
    4. 互补松弛: λᵢgᵢ(x*) = 0

    参数:
        x: 候选解
        lambda_vec: 不等式乘子
        mu_vec: 等式乘子
        grad_f: 目标函数梯度
        grad_constraints: 约束梯度字典
        constraints: 约束函数字典

    返回:
        (满足KKT, 违反的条件列表)
    """
    # TODO: 实现KKT条件检验
    violations = []
    tol = 1e-6

    # 1. 定常性条件
    stationarity = grad_f(x).copy()

    if 'ineq' in grad_constraints:
        for i, grad_g in enumerate(grad_constraints['ineq']):
            stationarity += lambda_vec[i] * grad_g(x)

    if 'eq' in grad_constraints:
        for j, grad_h in enumerate(grad_constraints['eq']):
            stationarity += mu_vec[j] * grad_h(x)

    if np.linalg.norm(stationarity) > tol:
        violations.append('stationarity')

    # 2. 原问题可行性
    if 'ineq' in constraints:
        for g in constraints['ineq']:
            if g(x) > tol:
                violations.append('primal_feasibility_ineq')
                break

    if 'eq' in constraints:
        for h in constraints['eq']:
            if abs(h(x)) > tol:
                violations.append('primal_feasibility_eq')
                break

    # 3. 对偶可行性
    if np.any(lambda_vec < -tol):
        violations.append('dual_feasibility')

    # 4. 互补松弛
    if 'ineq' in constraints:
        for i, g in enumerate(constraints['ineq']):
            if abs(lambda_vec[i] * g(x)) > tol:
                violations.append('complementary_slackness')
                break

    satisfies_kkt = len(violations) == 0

    return satisfies_kkt, violations


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：凸集的判定和基本性质")
    print("="*60)

    print("\n测试 1.1: 凸组合验证")
    points = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1])]
    weights = [0.3, 0.5, 0.2]
    x = 0.3 * points[0] + 0.5 * points[1] + 0.2 * points[2]
    is_convex_comb = is_convex_combination(x, points, weights)
    results.append(is_convex_comb)
    print(f"  凸组合验证: {'✓ 通过' if is_convex_comb else '✗ 失败'}")

    print("\n测试 1.2: 凸包计算")
    points_2d = np.array([[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1]])
    hull = convex_hull_2d(points_2d)
    # 凸包应该有4个顶点
    results.append(len(hull) >= 3)
    print(f"  凸包顶点数: {len(hull)}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 1.3: 单纯形投影")
    v_test = np.array([1.0, 2.0, -0.5])
    w = project_onto_simplex(v_test)
    # 检查投影在单纯形上
    on_simplex = np.all(w >= -1e-10) and np.isclose(np.sum(w), 1.0)
    results.append(on_simplex)
    print(f"  投影: {w}, 和: {np.sum(w):.4f}")
    print(f"  {'✓ 通过（在单纯形上）' if on_simplex else '✗ 失败'}")

    print("\n" + "="*60)
    print("中级题目：凸函数和共轭函数")
    print("="*60)

    print("\n测试 2.1: 凸函数验证")
    # 测试凸函数 f(x) = x^2
    f_convex = lambda x: x**2
    domain_1d = np.linspace(-2, 2, 20)
    is_conv = is_convex_function(f_convex, domain_1d)
    results.append(is_conv)
    print(f"  f(x)=x² 是凸函数: {'✓ 通过' if is_conv else '✗ 失败'}")

    print("\n测试 2.2: Hessian判定凸性")
    # f(x,y) = x² + y² 的Hessian是 [[2, 0], [0, 2]]
    H = np.array([[2, 0], [0, 2]])
    is_conv, min_eig = check_convexity_by_hessian(H)
    results.append(is_conv)
    print(f"  Hessian半正定: {is_conv}, 最小特征值: {min_eig:.4f}")
    print(f"  {'✓ 通过' if is_conv else '✗ 失败'}")

    print("\n测试 2.3: Fenchel共轭")
    # f(x) = x²/2 的共轭是 f*(y) = y²/2
    f = lambda x: 0.5 * x**2
    domain = np.linspace(-5, 5, 100)
    y = 1.0
    f_star = fenchel_conjugate(f, np.array([y]), domain.reshape(-1, 1))
    expected = 0.5 * y**2
    results.append(v.assert_close(f_star, expected, rtol=0.1, name="Fenchel共轭"))

    print("\n" + "="*60)
    print("高级题目：次梯度和凸优化对偶")
    print("="*60)

    print("\n测试 3.1: 次梯度计算")
    # f(x) = |x| 在 x=0 处的次梯度 ∈ [-1, 1]
    f_abs = lambda x: np.linalg.norm(x, 1)
    x = np.array([0.0])
    subgrad = subgradient(f_abs, x)
    # 次梯度应该在[-1, 1]内
    results.append(-1 <= subgrad[0] <= 1)
    print(f"  |x|在x=0的次梯度: {subgrad[0]:.4f}")
    print(f"  {'✓ 通过（在[-1,1]内）' if results[-1] else '✗ 失败'}")

    print("\n测试 3.2: 拉格朗日函数")
    f = lambda x: x[0]**2 + x[1]**2
    constraints = {'ineq': [lambda x: x[0] + x[1] - 1]}  # x + y ≤ 1
    x = np.array([0.5, 0.5])
    lambda_vec = np.array([1.0])
    mu_vec = np.array([])

    L_val = lagrangian(f, constraints, x, lambda_vec, mu_vec)
    # L = x² + y² + λ(x + y - 1) = 0.5 + 1.0 * 0 = 0.5
    results.append(L_val >= 0)
    print(f"  拉格朗日函数值: {L_val:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.3: KKT条件检验")
    # 简单问题: min x² s.t. x ≥ 1
    # 最优解: x* = 1, λ* = -2
    x_opt = np.array([1.0])
    lambda_opt = np.array([2.0])  # 注意约束改写为 1-x ≤ 0
    mu_opt = np.array([])

    grad_f = lambda x: 2 * x
    constraints_kkt = {'ineq': [lambda x: 1 - x[0]]}
    grad_constraints_kkt = {'ineq': [lambda x: np.array([-1.0])]}

    satisfies, violations = check_kkt_conditions(
        x_opt, lambda_opt, mu_opt,
        grad_f, grad_constraints_kkt, constraints_kkt
    )
    results.append(satisfies)
    print(f"  KKT条件满足: {satisfies}")
    if not satisfies:
        print(f"  违反条件: {violations}")
    print(f"  {'✓ 通过' if satisfies else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
