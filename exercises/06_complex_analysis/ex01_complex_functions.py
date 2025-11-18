"""
练习 1: 复变函数基础
====================

学习目标：
- 掌握复数运算
- 理解复变函数的性质
- 实现复数可视化

任务：
实现复变函数的计算和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


def complex_exponential(z):
    """
    复指数函数: e^z = e^(x+iy) = e^x (cos(y) + i*sin(y))

    欧拉公式: e^(iθ) = cos(θ) + i*sin(θ)
    """
    # TODO: 实现复指数（不直接使用np.exp）
    x = z.real
    y = z.imag
    return np.exp(x) * (np.cos(y) + 1j * np.sin(y))


def complex_logarithm(z):
    """
    复对数（主值）: log(z) = log|z| + i*arg(z)

    其中 arg(z) ∈ (-π, π]
    """
    # TODO: 实现复对数
    r = np.abs(z)
    theta = np.angle(z)  # 主辐角
    return np.log(r) + 1j * theta


def complex_power(z, w):
    """
    复数幂: z^w = e^(w*log(z))

    参数:
        z: 底数（复数）
        w: 指数（复数）
    """
    # TODO: 实现复数幂
    return np.exp(w * np.log(z))


def mobius_transformation(z, a, b, c, d):
    """
    莫比乌斯变换: f(z) = (az + b) / (cz + d)

    其中 ad - bc ≠ 0

    莫比乌斯变换将圆（或直线）映射到圆（或直线）
    """
    # TODO: 实现莫比乌斯变换
    if abs(a*d - b*c) < 1e-10:
        raise ValueError("ad - bc 必须非零")

    return (a * z + b) / (c * z + d)


def cauchy_riemann_check(f, z, h=1e-5):
    """
    检查Cauchy-Riemann方程

    对于解析函数 f(z) = u(x,y) + i*v(x,y):
    ∂u/∂x = ∂v/∂y
    ∂u/∂y = -∂v/∂x

    参数:
        f: 复变函数
        z: 检查点
        h: 数值微分步长

    返回:
        是否满足C-R方程
    """
    # TODO: 实现C-R方程检验
    x, y = z.real, z.imag

    # 计算函数值
    f_z = f(z)
    f_x_plus = f(z + h)
    f_x_minus = f(z - h)
    f_y_plus = f(z + 1j*h)
    f_y_minus = f(z - 1j*h)

    # 数值微分
    df_dx = (f_x_plus - f_x_minus) / (2*h)
    df_dy = (f_y_plus - f_y_minus) / (2*h)

    # 提取实部和虚部的偏导数
    du_dx = df_dx.real
    dv_dx = df_dx.imag
    du_dy = df_dy.real
    dv_dy = df_dy.imag

    # 检查C-R方程
    condition1 = abs(du_dx - dv_dy) < 1e-4
    condition2 = abs(du_dy + dv_dx) < 1e-4

    return condition1 and condition2


def complex_conjugate_properties(z, w):
    """
    验证共轭的性质:
    1. conj(z + w) = conj(z) + conj(w)
    2. conj(z * w) = conj(z) * conj(w)
    3. conj(conj(z)) = z

    返回:
        (property1_holds, property2_holds, property3_holds)
    """
    # TODO: 验证共轭性质
    z_conj = np.conj(z)
    w_conj = np.conj(w)

    # 性质1
    prop1 = np.allclose(np.conj(z + w), z_conj + w_conj)

    # 性质2
    prop2 = np.allclose(np.conj(z * w), z_conj * w_conj)

    # 性质3
    prop3 = np.allclose(np.conj(z_conj), z)

    return prop1, prop2, prop3


def plot_complex_function(f, x_range=(-2, 2), y_range=(-2, 2), n=20):
    """
    可视化复变函数（使用颜色映射）

    将复数 f(z) 映射为颜色:
    - 幅角 → 色调
    - 模 → 亮度

    参数:
        f: 复变函数
        x_range: x范围
        y_range: y范围
        n: 网格密度
    """
    # TODO: 实现复变函数可视化
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # 计算函数值
    W = f(Z)

    # 幅角映射到色调，模映射到亮度
    angle = np.angle(W)
    magnitude = np.abs(W)

    # 创建图形
    plt.figure(figsize=(10, 8))
    plt.imshow(angle, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
               origin='lower', cmap='hsv')
    plt.colorbar(label='arg(f(z))')
    plt.title('Complex Function Visualization')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')

    return plt.gcf()


def residue_at_pole(f, z0, order=1):
    """
    计算函数在极点的留数（简单极点）

    对于简单极点: Res(f, z0) = lim_{z→z0} (z-z0)f(z)

    参数:
        f: 复变函数
        z0: 极点位置
        order: 极点阶数（这里只处理order=1）
    """
    # TODO: 数值计算留数
    h = 1e-6
    z = z0 + h
    residue = (z - z0) * f(z)
    return residue


@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n测试 1: 复指数函数")
    z = 1 + 1j * np.pi
    result = complex_exponential(z)
    expected = np.exp(z)
    results.append(v.assert_close(result, expected, name="复指数"))

    print("\n测试 2: 欧拉公式 e^(iπ) + 1 = 0")
    result = complex_exponential(1j * np.pi)
    results.append(v.assert_close(result, -1, rtol=1e-10, name="欧拉公式"))

    print("\n测试 3: 复对数")
    z = 1 + 1j
    result = complex_logarithm(z)
    # 验证: e^(log(z)) = z
    verification = np.exp(result)
    results.append(v.assert_close(verification, z, name="复对数验证"))

    print("\n测试 4: 复数幂")
    z = 2 + 1j
    w = 1 - 1j
    result = complex_power(z, w)
    expected = z ** w
    results.append(v.assert_close(result, expected, name="复数幂"))

    print("\n测试 5: 莫比乌斯变换")
    # f(z) = (z-1)/(z+1) 将上半平面映射到单位圆
    z = 1j
    result = mobius_transformation(z, 1, -1, 1, 1)
    # f(i) = (i-1)/(i+1) = i
    expected = 1j
    results.append(v.assert_close(result, expected, rtol=0.01, name="莫比乌斯变换"))

    print("\n测试 6: Cauchy-Riemann方程检验")
    # f(z) = z^2 是解析的
    f_analytic = lambda z: z**2
    z = 1 + 1j
    is_analytic = cauchy_riemann_check(f_analytic, z)
    results.append(is_analytic)
    print(f"  f(z) = z^2 满足C-R方程: {'✓ 是' if is_analytic else '✗ 否'}")

    print("\n测试 7: 共轭性质")
    z = 2 + 3j
    w = 1 - 2j
    prop1, prop2, prop3 = complex_conjugate_properties(z, w)
    results.append(prop1 and prop2 and prop3)
    print(f"  所有共轭性质成立: {'✓ 是' if results[-1] else '✗ 否'}")

    print("\n测试 8: 留数计算")
    # f(z) = 1/z 在 z=0 的留数是 1
    f = lambda z: 1 / z if abs(z) > 1e-10 else 1e10
    z0 = 0 + 0j
    # 由于数值问题，我们在接近z0的点计算
    res = residue_at_pole(f, 1e-6)
    results.append(abs(res - 1.0) < 0.1)
    print(f"  Res(1/z, 0) ≈ {res:.4f} (期望: 1)")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    return all(results)


if __name__ == "__main__":
    test()
