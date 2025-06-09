import numpy as np
import scipy.integrate as spi
import scipy.linalg as spa
import matplotlib.pyplot as plt

# 参数设置
S = 100
r = 0.05
sigma = 0.2
T = 1
q = 0.2
K = 100

# 求解偏微分方程
def black_scholes_pde(S, r, sigma, T, q, K):
    def f(S, t):
        return np.exp(-q * T + (r - 0.5 * sigma**2) * t) * np.maximum(S - K, 0)
    def g(S, t):
        return np.exp(-q * T + (r - 0.5 * sigma**2) * t) * np.maximum(K - S, 0)
    def h(S, t):
        return np.exp(-q * T + (r - 0.5 * sigma**2) * t)
    def pde(S, t, u, v):
        return u * (r - 0.5 * sigma**2) * v + 0.5 * sigma**2 * v**2 - q * u * v
    return pde

# 定义有限差分方法
def finite_difference_method(S, r, sigma, T, q, K, N, M):
    dt = T / M
    dx = S / N
    u = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))
    for m in range(M + 1):
        for n in range(N + 1):
            S_ = S + n * dx
            u[m, n] = black_scholes_pde(S_, m * dt, r, T, q, K)
            v[m, n] = np.exp(-q * T + (r - 0.5 * sigma**2) * m * dt) * np.maximum(S_ - K, 0)
    return u, v, dt, dx

# 求解期权价格
def option_price(S, r, sigma, T, q, K, N, M):
    u, v, dt, dx = finite_difference_method(S, r, sigma, T, q, K, N, M)
    return np.sum(np.exp(-q * T + (r - 0.5 * sigma**2) * dt) * (u[m, n] - u[m - 1, n]) * dx for m in range(1, M + 1) for n in range(N))

# 计算期权价格
S = np.linspace(50, 150, 100)
option_prices = [option_price(S, r, sigma, T, q, K, N, M) for N in range(1, 10) for M in range(1, 10)]

# 绘制期权价格曲线
plt.plot(S, option_prices)
plt.xlabel('S')
plt.ylabel('Option Price')
plt.title('Black-Scholes Option Price Curve')
plt.show()
