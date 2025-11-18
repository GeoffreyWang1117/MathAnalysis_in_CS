"""
练习 5: 多元微积分
==================

学习目标：
- 掌握偏导数和梯度
- 理解Hessian矩阵
- 应用链式法则
- 计算方向导数和梯度下降

任务：
实现多元微积分的核心概念
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def partial_derivative(f, x, y, var='x', h=1e-5):
    """
    偏导数

    ∂f/∂x 或 ∂f/∂y

    参数:
        f: 二元函数 f(x, y)
        x, y: 计算点
        var: 'x' 或 'y'
        h: 步长
    """
    # TODO: 实现偏导数
    if var == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elif var == 'y':
        return (f(x, y + h) - f(x, y - h)) / (2 * h)


def gradient_vector(f, x, y, h=1e-5):
    """
    梯度向量

    ∇f = (∂f/∂x, ∂f/∂y)

    参数:
        f: 二元函数
        x, y: 计算点
    """
    # TODO: 实现梯度向量
    df_dx = partial_derivative(f, x, y, 'x', h)
    df_dy = partial_derivative(f, x, y, 'y', h)
    return np.array([df_dx, df_dy])


def hessian_matrix(f, x, y, h=1e-4):
    """
    Hessian矩阵（二阶偏导数矩阵）

    H = [[∂²f/∂x², ∂²f/∂x∂y],
         [∂²f/∂y∂x, ∂²f/∂y²]]

    参数:
        f: 二元函数
        x, y: 计算点
    """
    # TODO: 实现Hessian矩阵
    # 二阶偏导数
    d2f_dx2 = (f(x + h, y) - 2*f(x, y) + f(x - h, y)) / h**2
    d2f_dy2 = (f(x, y + h) - 2*f(x, y) + f(x, y - h)) / h**2

    # 混合偏导数
    d2f_dxdy = (f(x + h, y + h) - f(x + h, y - h) -
                f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)

    H = np.array([[d2f_dx2, d2f_dxdy],
                  [d2f_dxdy, d2f_dy2]])

    return H


def directional_derivative(f, x, y, direction, h=1e-5):
    """
    方向导数

    D_u f = ∇f · u

    其中 u 是单位方向向量

    参数:
        f: 函数
        x, y: 计算点
        direction: 方向向量（会被归一化）
    """
    # TODO: 实现方向导数
    # 归一化方向向量
    u = np.array(direction) / np.linalg.norm(direction)

    # 梯度
    grad = gradient_vector(f, x, y, h)

    # 方向导数 = 梯度 · 方向
    return np.dot(grad, u)


def critical_points_2d(f, x_range, y_range, n=50):
    """
    寻找临界点（梯度为零的点）

    使用网格搜索方法

    参数:
        f: 函数
        x_range, y_range: 搜索范围
        n: 网格密度
    """
    # TODO: 实现临界点搜索
    x_vals = np.linspace(x_range[0], x_range[1], n)
    y_vals = np.linspace(y_range[0], y_range[1], n)

    critical_pts = []

    for x in x_vals:
        for y in y_vals:
            grad = gradient_vector(f, x, y)
            # 如果梯度的模很小，认为是临界点
            if np.linalg.norm(grad) < 0.01:
                critical_pts.append((x, y, f(x, y)))

    return critical_pts


def classify_critical_point(f, x, y):
    """
    用Hessian矩阵分类临界点

    - det(H) > 0, H[0,0] > 0: 局部最小值
    - det(H) > 0, H[0,0] < 0: 局部最大值
    - det(H) < 0: 鞍点
    - det(H) = 0: 无法判定

    参数:
        f: 函数
        x, y: 临界点
    """
    # TODO: 实现临界点分类
    H = hessian_matrix(f, x, y)
    det_H = np.linalg.det(H)

    if abs(det_H) < 1e-8:
        return "degenerate"
    elif det_H > 0:
        if H[0, 0] > 0:
            return "minimum"
        else:
            return "maximum"
    else:
        return "saddle"


def chain_rule_2d(f, g, h, t):
    """
    链式法则

    如果 z = f(x, y), x = g(t), y = h(t)
    则 dz/dt = ∂f/∂x · dx/dt + ∂f/∂y · dy/dt

    参数:
        f: z = f(x, y)
        g: x = g(t)
        h: y = h(t)
        t: 参数值
    """
    # TODO: 实现链式法则
    dt = 1e-5

    # 计算 x, y 的值
    x = g(t)
    y = h(t)

    # ∂f/∂x 和 ∂f/∂y
    df_dx = partial_derivative(f, x, y, 'x')
    df_dy = partial_derivative(f, x, y, 'y')

    # dx/dt 和 dy/dt
    dx_dt = (g(t + dt) - g(t - dt)) / (2 * dt)
    dy_dt = (h(t + dt) - h(t - dt)) / (2 * dt)

    # 链式法则
    dz_dt = df_dx * dx_dt + df_dy * dy_dt

    return dz_dt


def line_integral(F, curve, t_range, n=1000):
    """
    线积分（向量场沿曲线）

    ∫_C F · dr = ∫_a^b F(r(t)) · r'(t) dt

    参数:
        F: 向量场 F(x, y) = (F_x, F_y)
        curve: 参数曲线 r(t) = (x(t), y(t))
        t_range: (t_start, t_end)
        n: 采样点数
    """
    # TODO: 实现线积分
    t = np.linspace(t_range[0], t_range[1], n)
    dt = (t_range[1] - t_range[0]) / n

    integral = 0
    for i in range(n - 1):
        # 计算曲线上的点
        x, y = curve(t[i])

        # 计算切向量 dr/dt
        x_next, y_next = curve(t[i + 1])
        dr_dt = np.array([(x_next - x) / dt, (y_next - y) / dt])

        # 计算向量场
        F_val = F(x, y)

        # F · dr/dt * dt
        integral += np.dot(F_val, dr_dt) * dt

    return integral


def green_theorem_verify(F, region_boundary, area):
    """
    验证Green定理

    ∮_C F · dr = ∬_D (∂Q/∂x - ∂P/∂y) dA

    其中 F = (P, Q)

    参数:
        F: 向量场 (P, Q)
        region_boundary: 区域边界曲线
        area: 区域面积（简化）
    """
    # TODO: 实现Green定理验证（简化版）
    # 这里只计算线积分作为左边
    # 实际应用中需要计算二重积分
    line_int = line_integral(F, region_boundary, (0, 2*np.pi))
    return line_int


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: 偏导数")
    f = lambda x, y: x**2 + y**2
    # ∂f/∂x = 2x, at (3, 4): ∂f/∂x = 6
    df_dx = partial_derivative(f, 3, 4, 'x')
    results.append(v.assert_close(df_dx, 6.0, rtol=1e-4, name="∂f/∂x"))

    # ∂f/∂y = 2y, at (3, 4): ∂f/∂y = 8
    df_dy = partial_derivative(f, 3, 4, 'y')
    results.append(v.assert_close(df_dy, 8.0, rtol=1e-4, name="∂f/∂y"))

    print("\n测试 2: 梯度向量")
    grad = gradient_vector(f, 3, 4)
    expected_grad = np.array([6.0, 8.0])
    results.append(v.assert_close(grad, expected_grad, rtol=1e-4, name="梯度"))

    print("\n测试 3: Hessian矩阵")
    # f = x² + y², Hessian = [[2, 0], [0, 2]]
    H = hessian_matrix(f, 3, 4)
    expected_H = np.array([[2, 0], [0, 2]])
    results.append(v.assert_close(H, expected_H, rtol=1e-2, name="Hessian矩阵"))

    print("\n测试 4: 方向导数")
    # 沿梯度方向的方向导数 = ||∇f||
    direction = grad / np.linalg.norm(grad)
    dir_deriv = directional_derivative(f, 3, 4, direction)
    expected = np.linalg.norm(grad)
    results.append(v.assert_close(dir_deriv, expected, rtol=1e-4, name="方向导数"))

    print("\n测试 5: 临界点分类")
    # f = x² + y² 的临界点在 (0, 0)，是最小值
    classification = classify_critical_point(f, 0, 0)
    results.append(classification == "minimum")
    print(f"  f(x,y) = x² + y² 在 (0,0): {classification}")
    print(f"  {'✓ 正确' if results[-1] else '✗ 错误'}")

    # 鞍点示例: f = x² - y²
    f_saddle = lambda x, y: x**2 - y**2
    classification_saddle = classify_critical_point(f_saddle, 0, 0)
    results.append(classification_saddle == "saddle")
    print(f"  f(x,y) = x² - y² 在 (0,0): {classification_saddle}")
    print(f"  {'✓ 正确' if results[-1] else '✗ 错误'}")

    print("\n测试 6: 链式法则")
    # z = x² + y², x = t, y = t²
    # dz/dt = 2x·1 + 2y·2t = 2t + 4t·t² = 2t + 4t³
    f_chain = lambda x, y: x**2 + y**2
    g = lambda t: t
    h = lambda t: t**2
    t_val = 2.0
    dz_dt = chain_rule_2d(f_chain, g, h, t_val)
    expected_dz_dt = 2 * t_val + 4 * t_val**3  # = 4 + 32 = 36
    results.append(v.assert_close(dz_dt, expected_dz_dt, rtol=1e-3, name="链式法则"))

    print("\n测试 7: 线积分")
    # 保守场 F = (x, y) 沿圆周的线积分应该为 0
    F = lambda x, y: np.array([y, -x])  # 旋度非零
    curve = lambda t: (np.cos(t), np.sin(t))  # 单位圆
    integral = line_integral(F, curve, (0, 2*np.pi))
    # 这个场的旋度为 -2，圆的面积为 π，所以积分为 -2π
    expected_integral = -2 * np.pi
    results.append(v.assert_close(integral, expected_integral, rtol=0.1, name="线积分"))

    return all(results)


if __name__ == "__main__":
    test()
