import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def progon(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n - 1):
        c_prime[i] = c[i] / (b[i] - a[i - 1] * c_prime[i - 1])

    for i in range(1, n):
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / (b[i] - a[i - 1] * c_prime[i - 1])

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def green_function(x, xi, t):
    if t == 0:
        x_0 = np.zeros(N)
        j = np.argmin(np.abs(x_grid - xi))
        x_0[j] = 1.0 / h
        return x_0
    result = np.zeros_like(x)
    result = 1 / (2 * np.sqrt(np.pi * kappa * t)) * np.exp(-(x - xi) ** 2 / (4 * kappa * t))
    return result


# Параметры задачи
A = 0.1
B = 20.0
N = 501
T = 1.0
K = 1000
xi = 11.0
v = 0.5
kappa = 10.0

x_grid, h = np.linspace(A, B, N, retstep=True)
tau = T / K
t_grid = np.linspace(0.1, T, K + 1)
print(h, tau)

u_num = np.zeros((K + 1, N))
j_xi = np.argmin(np.abs(x_grid - xi))
u_num[0, j_xi] = 1.0 / h
err = [np.max(np.abs(u_num[0] - green_function(x_grid, xi, 0)))]

u_num[:, 0] = 0.0
u_num[:, -1] = 0.0

for k in range(K):
    a = np.zeros(N - 2)
    b = np.zeros(N - 2)
    c = np.zeros(N - 2)
    d = np.zeros(N - 2)

    for j in range(1, N - 1):
        x_j = x_grid[j]
        coef_implicit = v * tau * kappa / h ** 2
        coef_explicit = (1 - v) * tau * kappa / h ** 2

        if j == 1:
            a[j - 1] = 0.0
            b[j - 1] = 1 + coef_implicit
            c[j - 1] = -coef_implicit / 2
            explicit_part = (coef_explicit / 2) * (u_num[k, j + 1] - 2 * u_num[k, j] + u_num[k, j - 1])
            d[j - 1] = u_num[k, j] + explicit_part
        elif j == N - 2:
            a[j - 1] = -coef_implicit / 2
            b[j - 1] = 1 + coef_implicit
            c[j - 1] = 0.0
            explicit_part = (coef_explicit / 2) * (u_num[k, j + 1] - 2 * u_num[k, j] + u_num[k, j - 1])
            d[j - 1] = u_num[k, j] + explicit_part
        else:
            a[j - 1] = -coef_implicit / 2
            b[j - 1] = 1 + coef_implicit
            c[j - 1] = -coef_implicit / 2
            explicit_part = (coef_explicit / 2) * (u_num[k, j + 1] - 2 * u_num[k, j] + u_num[k, j - 1])
            d[j - 1] = u_num[k, j] + explicit_part

    solution = progon(a, b, c, d)
    u_num[k + 1, 1:-1] = solution
    current_error = np.max(np.abs(u_num[k + 1] - green_function(x_grid, xi, t_grid[k + 1])))
    err.append(current_error)


fig, ax = plt.subplots(dpi=150)
line_num, = ax.plot(x_grid, u_num[0], 'r--', label='Численное решение')
line_true, = ax.plot(x_grid, green_function(x_grid, xi, 0), 'b-', label='Аналитическое решение')
ax.set_xlim(A, B)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend()


def update(frame):
    t = t_grid[frame]
    line_num.set_ydata(u_num[frame])
    line_true.set_ydata(green_function(x_grid, xi, t))
    ax.set_title(f"t = {t:.2f}")
    return line_num, line_true,



ani = FuncAnimation(fig, update, frames=K + 1, interval=50)
print(f"Максимальная ошибка на последнем шаге: {err[-1]:.6f}")
plt.show()