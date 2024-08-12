import numpy as np

# Example function: Sinc function
def f(x):
    # np.sinc(x) is normalized sinc function (sin(pi*x)/(pi*x)), so we rescale it
    return np.sinc(x / np.pi)

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = (h / 2) * (y[0] + 2 * sum(y[1:n]) + y[n])
    return integral

def simpsons_rule(f, a, b, n):
    if n % 2 == 1:
        n += 1  # n must be even
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 3 * (y[0] + 2 * sum(y[2:n:2]) + 4 * sum(y[1:n:2]) + y[n])
    return integral

# Parameters
a = 0
b = np.pi
n = 1000

# Compute the integral using Trapezoidal Rule
trap_result = trapezoidal_rule(f, a, b, n)

# Compute the integral using Simpson's Rule
simp_result = simpsons_rule(f, a, b, n)

print(f"Trapezoidal Rule: {trap_result}")
print(f"Simpson's Rule: {simp_result}")