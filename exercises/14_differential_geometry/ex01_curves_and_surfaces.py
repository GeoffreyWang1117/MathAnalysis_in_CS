"""
练习 1: 曲线与曲面
==================

本练习包含3道题目，由浅入深：
- 初级：参数曲线的基本性质
- 中级：曲率和挠率
- 高级：曲面的第一和第二基本形式

参考教材：
- do Carmo - Differential Geometry of Curves and Surfaces
- Spivak - A Comprehensive Introduction to Differential Geometry
"""

import numpy as np
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：参数曲线的基本性质
# ============================================================================

def arc_length(curve, t_range, n=1000):
    """
    初级 - 计算曲线的弧长

    L = ∫||r'(t)||dt

    参数:
        curve: 参数曲线 r(t) = (x(t), y(t), z(t))
        t_range: (t0, t1)
        n: 采样点数
    """
    # TODO: 实现弧长计算
    t = np.linspace(t_range[0], t_range[1], n)
    dt = (t_range[1] - t_range[0]) / n

    length = 0
    for i in range(n - 1):
        r1 = np.array(curve(t[i]))
        r2 = np.array(curve(t[i + 1]))
        length += np.linalg.norm(r2 - r1)

    return length


def tangent_vector(curve, t, h=1e-5):
    """
    初级 - 计算切向量 T = r'(t) / ||r'(t)||

    参数:
        curve: 参数曲线
        t: 参数值
        h: 数值微分步长
    """
    # TODO: 实现切向量
    r_t_plus = np.array(curve(t + h))
    r_t_minus = np.array(curve(t - h))

    # 数值导数
    dr_dt = (r_t_plus - r_t_minus) / (2 * h)

    # 归一化
    T = dr_dt / np.linalg.norm(dr_dt)

    return T


def unit_normal_vector(curve, t, h=1e-5):
    """
    初级 - 计算主法向量 N

    N = T' / ||T'||

    参数:
        curve: 参数曲线
        t: 参数值
    """
    # TODO: 实现主法向量
    T1 = tangent_vector(curve, t - h, h)
    T2 = tangent_vector(curve, t + h, h)

    dT_dt = (T2 - T1) / (2 * h)

    if np.linalg.norm(dT_dt) < 1e-10:
        return np.array([0, 0, 0])

    N = dT_dt / np.linalg.norm(dT_dt)

    return N


# ============================================================================
# 中级题目：曲率和挠率
# ============================================================================

def curvature(curve, t, h=1e-5):
    """
    中级 - 计算曲线的曲率

    κ = ||r' × r''|| / ||r'||³

    参数:
        curve: 参数曲线
        t: 参数值
    """
    # TODO: 实现曲率计算
    # 一阶导数
    r_plus = np.array(curve(t + h))
    r_minus = np.array(curve(t - h))
    r_prime = (r_plus - r_minus) / (2 * h)

    # 二阶导数
    r_t = np.array(curve(t))
    r_double_prime = (r_plus - 2 * r_t + r_minus) / h**2

    # 曲率公式
    cross_product = np.cross(r_prime, r_double_prime)
    numerator = np.linalg.norm(cross_product)
    denominator = np.linalg.norm(r_prime) ** 3

    if denominator < 1e-10:
        return 0

    return numerator / denominator


def torsion(curve, t, h=1e-5):
    """
    中级 - 计算曲线的挠率

    τ = (r' × r'') · r''' / ||r' × r''||²

    挠率度量曲线离开密切平面的程度

    参数:
        curve: 参数曲线
        t: 参数值
    """
    # TODO: 实现挠率计算
    # 一阶、二阶、三阶导数
    r_m2 = np.array(curve(t - 2*h))
    r_m1 = np.array(curve(t - h))
    r_0 = np.array(curve(t))
    r_p1 = np.array(curve(t + h))
    r_p2 = np.array(curve(t + 2*h))

    r_prime = (r_p1 - r_m1) / (2 * h)
    r_double_prime = (r_p1 - 2*r_0 + r_m1) / h**2
    r_triple_prime = (r_p2 - 2*r_p1 + 2*r_m1 - r_m2) / (2 * h**3)

    cross_prod = np.cross(r_prime, r_double_prime)
    denominator = np.linalg.norm(cross_prod) ** 2

    if denominator < 1e-10:
        return 0

    numerator = np.dot(cross_prod, r_triple_prime)

    return numerator / denominator


def frenet_frame(curve, t, h=1e-5):
    """
    中级 - 计算Frenet标架 (T, N, B)

    T: 切向量
    N: 主法向量
    B: 副法向量 B = T × N

    返回:
        (T, N, B): Frenet标架
    """
    # TODO: 实现Frenet标架
    T = tangent_vector(curve, t, h)
    N = unit_normal_vector(curve, t, h)
    B = np.cross(T, N)

    return T, N, B


# ============================================================================
# 高级题目：曲面的基本形式
# ============================================================================

def first_fundamental_form(surface, u, v, h=1e-5):
    """
    高级 - 计算曲面的第一基本形式

    I = E du² + 2F du dv + G dv²

    其中:
    E = ⟨r_u, r_u⟩
    F = ⟨r_u, r_v⟩
    G = ⟨r_v, r_v⟩

    参数:
        surface: 参数曲面 r(u, v)
        u, v: 参数值
    """
    # TODO: 实现第一基本形式
    # 计算偏导数
    r_u = (np.array(surface(u + h, v)) - np.array(surface(u - h, v))) / (2 * h)
    r_v = (np.array(surface(u, v + h)) - np.array(surface(u, v - h))) / (2 * h)

    E = np.dot(r_u, r_u)
    F = np.dot(r_u, r_v)
    G = np.dot(r_v, r_v)

    return E, F, G


def surface_normal(surface, u, v, h=1e-5):
    """
    高级 - 计算曲面的单位法向量

    n = (r_u × r_v) / ||r_u × r_v||

    参数:
        surface: 参数曲面
        u, v: 参数值
    """
    # TODO: 实现曲面法向量
    r_u = (np.array(surface(u + h, v)) - np.array(surface(u - h, v))) / (2 * h)
    r_v = (np.array(surface(u, v + h)) - np.array(surface(u, v - h))) / (2 * h)

    n = np.cross(r_u, r_v)
    return n / np.linalg.norm(n)


def gaussian_curvature(surface, u, v, h=1e-5):
    """
    高级 - 计算Gauss曲率

    K = (LN - M²) / (EG - F²)

    Gauss曲率是内蕴几何量

    参数:
        surface: 参数曲面
        u, v: 参数值
    """
    # TODO: 实现Gauss曲率
    # 第一基本形式
    E, F, G = first_fundamental_form(surface, u, v, h)

    # 第二基本形式（简化计算）
    # L, M, N对应二阶偏导数
    r = np.array(surface(u, v))
    r_uu = (np.array(surface(u + h, v)) - 2*r + np.array(surface(u - h, v))) / h**2
    r_vv = (np.array(surface(u, v + h)) - 2*r + np.array(surface(u, v - h))) / h**2
    r_uv = (np.array(surface(u + h, v + h)) - np.array(surface(u + h, v - h)) -
            np.array(surface(u - h, v + h)) + np.array(surface(u - h, v - h))) / (4 * h**2)

    n = surface_normal(surface, u, v, h)

    L = np.dot(r_uu, n)
    M = np.dot(r_uv, n)
    N = np.dot(r_vv, n)

    # Gauss曲率
    denominator = E * G - F**2
    if abs(denominator) < 1e-10:
        return 0

    K = (L * N - M**2) / denominator

    return K


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：参数曲线的基本性质")
    print("="*60)

    print("\n测试 1.1: 弧长 - 单位圆")
    # 单位圆周长 = 2π
    circle = lambda t: (np.cos(t), np.sin(t), 0)
    L = arc_length(circle, (0, 2*np.pi))
    results.append(v.assert_close(L, 2*np.pi, rtol=0.01, name="圆周长"))

    print("\n测试 1.2: 切向量")
    T = tangent_vector(circle, 0)
    expected_T = np.array([0, 1, 0])  # 在t=0处，切向量应指向y方向
    results.append(v.assert_close(T, expected_T, rtol=0.01, name="切向量"))

    print("\n测试 1.3: 主法向量")
    N = unit_normal_vector(circle, 0)
    # 圆的法向量应指向圆心
    results.append(np.linalg.norm(N) > 0.9)  # 应该是单位向量
    print(f"  主法向量模: {np.linalg.norm(N):.4f}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print("中级题目：曲率和挠率")
    print("="*60)

    print("\n测试 2.1: 曲率 - 单位圆")
    kappa = curvature(circle, 0)
    # 单位圆的曲率 = 1
    results.append(v.assert_close(kappa, 1.0, rtol=0.1, name="圆的曲率"))

    print("\n测试 2.2: 挠率 - 平面曲线")
    # 平面曲线的挠率 = 0
    tau = torsion(circle, 0)
    results.append(abs(tau) < 0.1)
    print(f"  平面曲线挠率: {tau:.6f} (期望: ~0)")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 2.3: Frenet标架正交性")
    T, N, B = frenet_frame(circle, np.pi/4)
    # T, N, B应该两两正交
    orthogonal = (abs(np.dot(T, N)) < 0.1 and
                  abs(np.dot(N, B)) < 0.1 and
                  abs(np.dot(T, B)) < 0.1)
    results.append(orthogonal)
    print(f"  T·N = {np.dot(T, N):.4f}")
    print(f"  N·B = {np.dot(N, B):.4f}")
    print(f"  T·B = {np.dot(T, B):.4f}")
    print(f"  {'✓ 通过（两两正交）' if orthogonal else '✗ 失败'}")

    print("\n" + "="*60)
    print("高级题目：曲面的基本形式")
    print("="*60)

    print("\n测试 3.1: 第一基本形式 - 球面")
    # 单位球面: r(u,v) = (sin(u)cos(v), sin(u)sin(v), cos(u))
    sphere = lambda u, v: (np.sin(u)*np.cos(v), np.sin(u)*np.sin(v), np.cos(u))
    E, F, G = first_fundamental_form(sphere, np.pi/4, np.pi/4)
    # 球面的第一基本形式: E=1, F=0, G=sin²(u)
    results.append(abs(F) < 0.1)  # F应该为0
    print(f"  E = {E:.4f}, F = {F:.4f}, G = {G:.4f}")
    print(f"  {'✓ 通过（F≈0）' if results[-1] else '✗ 失败'}")

    print("\n测试 3.2: 曲面法向量")
    n = surface_normal(sphere, np.pi/4, np.pi/4)
    # 球面的法向量应该是单位向量
    results.append(v.assert_close(np.linalg.norm(n), 1.0, rtol=0.01, name="法向量单位性"))

    print("\n测试 3.3: Gauss曲率 - 球面")
    K = gaussian_curvature(sphere, np.pi/4, np.pi/4, h=1e-4)
    # 单位球面的Gauss曲率 = 1
    results.append(abs(K - 1.0) < 0.5)  # 宽松容差（数值计算不稳定）
    print(f"  球面Gauss曲率: {K:.4f} (期望: ~1)")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
