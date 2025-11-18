"""
练习 1: 常微分方程（ODE）求解
==============================

学习目标：
- 掌握Euler方法、Runge-Kutta方法
- 理解数值稳定性
- 应用ODE求解器

任务：
实现经典的ODE数值求解方法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def euler_method(f, y0, t_span, dt):
    """
    Euler方法求解ODE: dy/dt = f(t, y)

    y_{n+1} = y_n + dt * f(t_n, y_n)

    参数:
        f: 右端函数 f(t, y)
        y0: 初始条件
        t_span: (t_start, t_end) 时间区间
        dt: 时间步长

    返回:
        t: 时间点数组
        y: 对应的解数组
    """
    # TODO: 实现Euler方法
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n = len(t)

    # 初始化解数组
    if np.isscalar(y0):
        y = np.zeros(n)
        y[0] = y0
    else:
        y = np.zeros((n, len(y0)))
        y[0] = y0

    # Euler迭代
    for i in range(n - 1):
        y[i + 1] = y[i] + dt * f(t[i], y[i])

    return t, y


def runge_kutta_4(f, y0, t_span, dt):
    """
    4阶Runge-Kutta方法（RK4）

    经典的RK4公式:
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + dt*k1/2)
    k3 = f(t_n + dt/2, y_n + dt*k2/2)
    k4 = f(t_n + dt, y_n + dt*k3)
    y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4)

    RK4比Euler方法精度更高（4阶 vs 1阶）
    """
    # TODO: 实现RK4方法
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n = len(t)

    if np.isscalar(y0):
        y = np.zeros(n)
        y[0] = y0
    else:
        y = np.zeros((n, len(y0)))
        y[0] = y0

    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
        k3 = f(t[i] + dt/2, y[i] + dt*k2/2)
        k4 = f(t[i] + dt, y[i] + dt*k3)

        y[i + 1] = y[i] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y


def solve_harmonic_oscillator(omega=1.0, y0=(1.0, 0.0), t_end=10.0, dt=0.01):
    """
    求解谐振子方程: d²y/dt² + ω²y = 0

    转换为一阶系统:
    dy1/dt = y2
    dy2/dt = -ω²y1

    其中 y1 = y, y2 = dy/dt

    参数:
        omega: 角频率
        y0: (y初值, dy/dt初值)
        t_end: 结束时间
        dt: 时间步长

    返回:
        t, y1, y2
    """
    # TODO: 使用RK4求解谐振子
    def f(t, y):
        y1, y2 = y
        return np.array([y2, -omega**2 * y1])

    t, y = runge_kutta_4(f, np.array(y0), (0, t_end), dt)

    return t, y[:, 0], y[:, 1]


def solve_lorenz_system(sigma=10, rho=28, beta=8/3, y0=(1, 1, 1), t_end=50, dt=0.01):
    """
    求解Lorenz混沌系统:

    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

    Lorenz系统展示混沌行为

    参数:
        sigma, rho, beta: Lorenz系统参数
        y0: 初始条件 (x0, y0, z0)
        t_end: 结束时间
        dt: 时间步长
    """
    # TODO: 实现Lorenz系统求解
    def f(t, y):
        x, y_val, z = y
        dx = sigma * (y_val - x)
        dy = x * (rho - z) - y_val
        dz = x * y_val - beta * z
        return np.array([dx, dy, dz])

    t, y = runge_kutta_4(f, np.array(y0), (0, t_end), dt)

    return t, y


def solve_stiff_ode_example():
    """
    求解刚性ODE示例

    dy/dt = -1000y + 3000 - 2000e^(-t)

    这个方程是"刚性"的，需要特殊处理
    使用scipy的solve_ivp with method='BDF'（向后微分）
    """
    # TODO: 使用solve_ivp求解刚性ODE
    def f(t, y):
        return -1000 * y + 3000 - 2000 * np.exp(-t)

    y0 = [0]
    t_span = (0, 0.3)
    t_eval = np.linspace(0, 0.3, 100)

    # BDF方法适合刚性问题
    sol = solve_ivp(f, t_span, y0, method='BDF', t_eval=t_eval)

    return sol.t, sol.y[0]


def verify_energy_conservation(t, y1, y2, omega=1.0):
    """
    验证谐振子的能量守恒

    E = (1/2)(dy/dt)² + (1/2)ω²y²

    对于无阻尼谐振子，能量应该守恒
    """
    # TODO: 计算能量并检查守恒
    kinetic = 0.5 * y2**2
    potential = 0.5 * omega**2 * y1**2
    total_energy = kinetic + potential

    # 检查能量变化
    energy_variation = np.std(total_energy) / np.mean(total_energy)

    return total_energy, energy_variation


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: Euler方法求解 dy/dt = -y, y(0) = 1")
    # 解析解: y = e^(-t)
    f1 = lambda t, y: -y
    t, y = euler_method(f1, 1.0, (0, 2), 0.01)
    y_exact = np.exp(-t)
    # Euler方法误差较大，使用较宽松的容差
    error = np.mean(np.abs(y - y_exact))
    results.append(error < 0.1)
    print(f"  平均误差: {error:.6f}")
    print(f"  {'✓ 通过' if error < 0.1 else '✗ 失败'}")

    print("\n测试 2: RK4方法求解 dy/dt = -y, y(0) = 1")
    t, y = runge_kutta_4(f1, 1.0, (0, 2), 0.01)
    y_exact = np.exp(-t)
    error = np.mean(np.abs(y - y_exact))
    # RK4应该更精确
    results.append(error < 0.001)
    print(f"  平均误差: {error:.6f}")
    print(f"  {'✓ 通过' if error < 0.001 else '✗ 失败'}")

    print("\n测试 3: 谐振子能量守恒")
    t, y1, y2 = solve_harmonic_oscillator(omega=1.0, y0=(1.0, 0.0), t_end=20.0, dt=0.01)
    energy, variation = verify_energy_conservation(t, y1, y2, omega=1.0)
    # 能量变化应该很小
    results.append(variation < 0.01)
    print(f"  能量相对变化: {variation:.6f}")
    print(f"  {'✓ 通过（能量守恒）' if variation < 0.01 else '✗ 失败'}")

    print("\n测试 4: Lorenz系统求解")
    t, y = solve_lorenz_system(t_end=10, dt=0.01)
    # 检查解的合理性（没有NaN或Inf）
    is_valid = np.all(np.isfinite(y))
    results.append(is_valid)
    print(f"  解的有效性: {'✓ 通过' if is_valid else '✗ 失败'}")
    print(f"  最终状态: x={y[-1, 0]:.2f}, y={y[-1, 1]:.2f}, z={y[-1, 2]:.2f}")

    print("\n测试 5: 刚性ODE求解")
    t, y = solve_stiff_ode_example()
    # 检查解是否收敛到稳态值
    steady_state = 3.0 - 2.0 * np.exp(-t[-1])
    results.append(abs(y[-1] - steady_state) < 0.1)
    print(f"  最终值: {y[-1]:.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    return all(results)


if __name__ == "__main__":
    test()
