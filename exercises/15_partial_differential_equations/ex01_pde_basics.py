"""
练习 1: 偏微分方程基础
======================

本练习包含3道题目，由浅入深：
- 初级：偏微分方程的分类和特征线
- 中级：分离变量法和Fourier级数解
- 高级：有限差分法数值求解

参考教材：
- Evans - Partial Differential Equations
- Strauss - Partial Differential Equations: An Introduction
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：PDE分类和特征线
# ============================================================================

def classify_pde_2d(A, B, C):
    """
    初级 - 分类二阶PDE

    A u_xx + B u_xy + C u_yy + ... = 0

    分类：
    - B² - 4AC < 0: 椭圆型 (如Laplace方程)
    - B² - 4AC = 0: 抛物型 (如热方程)
    - B² - 4AC > 0: 双曲型 (如波动方程)

    参数:
        A, B, C: 二阶偏导数系数

    返回:
        "elliptic", "parabolic", "hyperbolic"
    """
    # TODO: 实现PDE分类
    discriminant = B**2 - 4*A*C

    if discriminant < -1e-10:
        return "elliptic"
    elif abs(discriminant) <= 1e-10:
        return "parabolic"
    else:
        return "hyperbolic"


def transport_equation_solution(c, initial_condition, x, t):
    """
    初级 - 输运方程的解

    ∂u/∂t + c ∂u/∂x = 0
    u(x, 0) = φ(x)

    解: u(x, t) = φ(x - ct)

    参数:
        c: 传播速度
        initial_condition: 初始条件函数 φ
        x: 空间位置
        t: 时间

    返回:
        u(x, t)
    """
    # TODO: 实现输运方程解
    return initial_condition(x - c * t)


def characteristic_curves(a, b, x0, t0, t_end, n=100):
    """
    初级 - 计算特征曲线

    对于方程 a u_x + b u_t = 0
    特征曲线: dx/dt = a/b

    参数:
        a, b: 系数
        x0, t0: 初始点
        t_end: 终止时间
        n: 采样点数

    返回:
        (t, x): 特征曲线上的点
    """
    # TODO: 实现特征曲线
    t = np.linspace(t0, t_end, n)

    if abs(b) < 1e-10:
        # 垂直特征线
        x = np.full(n, x0)
    else:
        # dx/dt = a/b
        x = x0 + (a / b) * (t - t0)

    return t, x


# ============================================================================
# 中级题目：分离变量法
# ============================================================================

def heat_equation_1d(x, t, L, alpha, n_terms=50):
    """
    中级 - 一维热方程的级数解

    ∂u/∂t = α ∂²u/∂x²
    u(0, t) = u(L, t) = 0
    u(x, 0) = sin(πx/L)

    解: u(x,t) = Σ b_n sin(nπx/L) exp(-α(nπ/L)²t)

    参数:
        x: 空间位置
        t: 时间
        L: 区域长度
        alpha: 热扩散系数
        n_terms: 级数项数

    返回:
        u(x, t)
    """
    # TODO: 实现热方程解
    u = 0

    for n in range(1, n_terms + 1):
        # 初始条件为 sin(πx/L)，所以只有 n=1 项的系数不为0
        if n == 1:
            b_n = 1
        else:
            b_n = 0

        u += b_n * np.sin(n * np.pi * x / L) * np.exp(-alpha * (n * np.pi / L)**2 * t)

    return u


def wave_equation_1d(x, t, L, c, n_terms=50):
    """
    中级 - 一维波动方程的d'Alembert解

    ∂²u/∂t² = c² ∂²u/∂x²
    u(0, t) = u(L, t) = 0
    u(x, 0) = sin(πx/L), ∂u/∂t(x, 0) = 0

    解: u(x,t) = Σ [A_n cos(cnπt/L) + B_n sin(cnπt/L)] sin(nπx/L)

    参数:
        x: 空间位置
        t: 时间
        L: 区域长度
        c: 波速
        n_terms: 级数项数
    """
    # TODO: 实现波动方程解
    u = 0

    for n in range(1, n_terms + 1):
        # 初始位移 sin(πx/L)
        A_n = 1 if n == 1 else 0
        # 初始速度为0
        B_n = 0

        u += (A_n * np.cos(c * n * np.pi * t / L) +
              B_n * np.sin(c * n * np.pi * t / L)) * np.sin(n * np.pi * x / L)

    return u


def laplace_2d_rectangular(x, y, Lx, Ly, boundary_values, n_terms=20):
    """
    中级 - 矩形区域上的Laplace方程

    ∂²u/∂x² + ∂²u/∂y² = 0
    在矩形 [0, Lx] × [0, Ly] 上

    参数:
        x, y: 位置
        Lx, Ly: 区域尺寸
        boundary_values: 边界条件（简化为顶部边界）
        n_terms: 级数项数
    """
    # TODO: 实现Laplace方程解（简化版）
    # 假设三边为0，顶部边界为 f(x)
    u = 0

    for n in range(1, n_terms + 1):
        # Fourier系数（简化：假设顶部为 sin(πx/Lx)）
        b_n = 1 if n == 1 else 0

        u += b_n * np.sin(n * np.pi * x / Lx) * np.sinh(n * np.pi * y / Lx) / np.sinh(n * np.pi * Ly / Lx)

    return u


# ============================================================================
# 高级题目：有限差分法
# ============================================================================

def finite_difference_laplace_2d(Lx, Ly, nx, ny, max_iter=1000, tol=1e-6):
    """
    高级 - 有限差分法求解2D Laplace方程

    ∂²u/∂x² + ∂²u/∂y² = 0

    使用五点差分格式:
    (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h² = 0

    参数:
        Lx, Ly: 区域尺寸
        nx, ny: 网格点数
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        u: 解的网格
    """
    # TODO: 实现有限差分法
    # 初始化网格
    u = np.zeros((nx, ny))

    # 边界条件（简化：四边为0，内部初始化为随机）
    u[1:-1, 1:-1] = 0.1

    # Gauss-Seidel迭代
    for iteration in range(max_iter):
        u_old = u.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

        # 检查收敛
        if np.max(np.abs(u - u_old)) < tol:
            break

    return u


def finite_difference_heat_1d(L, T, nx, nt, alpha):
    """
    高级 - 有限差分法求解1D热方程

    ∂u/∂t = α ∂²u/∂x²

    使用向前Euler + 中心差分:
    (u_i^{n+1} - u_i^n) / Δt = α (u_{i+1}^n - 2u_i^n + u_{i-1}^n) / Δx²

    参数:
        L: 空间区域长度
        T: 时间长度
        nx: 空间网格点数
        nt: 时间步数
        alpha: 热扩散系数

    返回:
        (x, t, u): 网格和解
    """
    # TODO: 实现热方程有限差分
    dx = L / (nx - 1)
    dt = T / (nt - 1)

    # 稳定性条件: α dt / dx² ≤ 0.5
    r = alpha * dt / dx**2
    if r > 0.5:
        print(f"  警告: 不稳定! r = {r:.4f} > 0.5")

    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, nx))

    # 初始条件
    u[0, :] = np.sin(np.pi * x / L)

    # 边界条件
    u[:, 0] = 0
    u[:, -1] = 0

    # 时间推进
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

    return x, t, u


def cfl_condition_check(dx, dt, c):
    """
    高级 - 检查CFL（Courant-Friedrichs-Lewy）条件

    对于波动方程，稳定性条件: c dt / dx ≤ 1

    参数:
        dx: 空间步长
        dt: 时间步长
        c: 波速

    返回:
        是否满足CFL条件
    """
    # TODO: 实现CFL条件检查
    cfl_number = c * dt / dx
    return cfl_number <= 1.0, cfl_number


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：PDE分类和特征线")
    print("="*60)

    print("\n测试 1.1: PDE分类")
    # Laplace方程: u_xx + u_yy = 0 (A=1, B=0, C=1)
    pde_type = classify_pde_2d(1, 0, 1)
    results.append(pde_type == "elliptic")
    print(f"  Laplace方程类型: {pde_type} (期望: elliptic)")

    # 热方程: u_t = u_xx (A=1, B=0, C=0)
    pde_type2 = classify_pde_2d(1, 0, 0)
    results.append(pde_type2 == "parabolic")
    print(f"  热方程类型: {pde_type2} (期望: parabolic)")

    print("\n测试 1.2: 输运方程")
    initial = lambda x: np.sin(x)
    u = transport_equation_solution(c=1.0, initial_condition=initial, x=np.pi, t=np.pi/2)
    expected = initial(np.pi - np.pi/2)  # sin(π/2) = 1
    results.append(v.assert_close(u, expected, rtol=0.01, name="输运方程解"))

    print("\n" + "="*60)
    print("中级题目：分离变量法")
    print("="*60)

    print("\n测试 2.1: 热方程解")
    u_heat = heat_equation_1d(x=0.5, t=0.1, L=1.0, alpha=0.1, n_terms=50)
    # 热方程应该使初始波衰减
    results.append(abs(u_heat) < 1.0)
    print(f"  u(0.5, 0.1) = {u_heat:.6f}")
    print(f"  {'✓ 通过（合理范围）' if results[-1] else '✗ 失败'}")

    print("\n测试 2.2: 波动方程解")
    u_wave = wave_equation_1d(x=0.5, t=0, L=1.0, c=1.0, n_terms=50)
    # t=0时应该等于初始条件
    expected_wave = np.sin(np.pi * 0.5)  # sin(π/2) = 1
    results.append(v.assert_close(u_wave, expected_wave, rtol=0.01, name="波动方程初值"))

    print("\n测试 2.3: Laplace方程")
    u_laplace = laplace_2d_rectangular(x=0.5, y=0.5, Lx=1.0, Ly=1.0,
                                        boundary_values=None, n_terms=20)
    # Laplace方程的解应该在边界值之间
    results.append(abs(u_laplace) < 10.0)  # 合理性检查
    print(f"  u(0.5, 0.5) = {u_laplace:.6f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：有限差分法")
    print("="*60)

    print("\n测试 3.1: 有限差分Laplace")
    u_fd = finite_difference_laplace_2d(Lx=1.0, Ly=1.0, nx=11, ny=11, max_iter=100)
    # 边界为0的Laplace方程，内部解应该也接近0
    interior_max = np.max(np.abs(u_fd[1:-1, 1:-1]))
    results.append(interior_max < 0.5)
    print(f"  内部最大值: {interior_max:.6f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 3.2: 有限差分热方程")
    x, t, u = finite_difference_heat_1d(L=1.0, T=0.1, nx=21, nt=101, alpha=0.01)
    # 验证能量衰减
    initial_energy = np.sum(u[0, :]**2)
    final_energy = np.sum(u[-1, :]**2)
    energy_decreased = final_energy < initial_energy
    results.append(energy_decreased)
    print(f"  初始能量: {initial_energy:.6f}")
    print(f"  最终能量: {final_energy:.6f}")
    print(f"  {'✓ 能量衰减' if energy_decreased else '✗ 能量未衰减'}")

    print("\n测试 3.3: CFL条件")
    stable, cfl_num = cfl_condition_check(dx=0.1, dt=0.05, c=1.0)
    results.append(stable)
    print(f"  CFL数: {cfl_num:.4f}")
    print(f"  {'✓ 稳定' if stable else '✗ 不稳定'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
