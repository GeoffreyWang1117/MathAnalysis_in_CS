import sympy as sp

t = sp.symbols('t')
s = sp.symbols('s')
f = sp.exp(-2 * t) * sp.sin(3 * t)

# 拉普拉斯变换
F = sp.laplace_transform(f, t, s)

print(f"f(t) 的拉普拉斯变换: {F}")
