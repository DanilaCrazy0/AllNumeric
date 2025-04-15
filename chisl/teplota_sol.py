import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded

A = 0.1
B = 10.0
N = 101
T = 1.0
K = 201
xi = 4.0

kappa = 0.5

x_grid, h = np.linspace(A, B, N, retstep=True)
t_grid, tau = np.linspace(0, T, K, retstep=True)

alpha = kappa * tau / h ** 2


def der_2(u_i):
    padded_f = np.pad(u_i, (1, 1), mode='constant', constant_values=0)
    padded_f[1:-1] = (padded_f[2:] - 2 * padded_f[1:-1] + padded_f[:-2]) / h ** 2
    return padded_f[1:-1]


def initial_condition(x):
    x_0 = np.zeros_like(x)
    j_xi = np.argmin(np.abs(x_grid - xi))
    x_0[j_xi] = 1.0 / h
    return x_0


def true_solution(x, t):
    if t == 0:
        return initial_condition(x)
    return 1.0 / np.sqrt(4.0 * np.pi * kappa * t) * np.exp(- (x - xi) ** 2 / (4.0 * kappa * t))


def left_bc(t):
    return 0.0


def right_bc(t):
    return 0.0


u = np.zeros((K, N))
u[0, :] = initial_condition(x_grid)

def crank_nicolson_step(u_old, t_old):
    rhs = u_old[1:-1].copy()

    D2 = (u_old[2:] - 2 * u_old[1:-1] + u_old[:-2])
    rhs += 0.5 * alpha * D2

    lhs_bc_left = left_bc(t_old + tau)
    lhs_bc_right = right_bc(t_old + tau)

    # rhs[0] += 0.5 * alpha * lhs_bc_left
    # rhs[-1] += 0.5 * alpha * lhs_bc_right

    main_diag = (1 + alpha) * np.ones(N - 2)
    lower_diag = -0.5 * alpha * np.ones(N - 3)
    upper_diag = -0.5 * alpha * np.ones(N - 3)

    A = np.zeros(N - 2)
    B = np.zeros(N - 2)

    A[0] = upper_diag[0] / main_diag[0]
    B[0] = rhs[0] / main_diag[0]

    for i in range(1, N - 3):
        denom = main_diag[i] - lower_diag[i - 1] * A[i - 1]
        A[i] = upper_diag[i] / denom
        B[i] = (rhs[i] - lower_diag[i - 1] * B[i - 1]) / denom

    sol_interior = np.zeros(N - 2)
    sol_interior[-1] = (rhs[-1] - lower_diag[-1] * B[-2]) / (main_diag[-1] - lower_diag[-1] * A[-2])

    for i in range(N - 4, -1, -1):
        sol_interior[i] = B[i] - A[i] * sol_interior[i + 1]

    # Собираем решение
    u_new = u_old.copy()
    u_new[0] = lhs_bc_left
    u_new[-1] = lhs_bc_right
    u_new[1:-1] = sol_interior

    return u_new


# Анимация
fig, ax = plt.subplots()
line_true, = ax.plot(x_grid, true_solution(x_grid, 0), 'b-', label='Точное')
line_num, = ax.plot(x_grid, u[0], 'r--', label='Численное')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('t = 0.0')
ax.legend()
ax.set_ylim(-0.1, 2.0)


def init():
    # Инициализация графики
    line_true.set_data(x_grid, true_solution(x_grid, 0.0))
    line_num.set_data(x_grid, u[0])
    return line_true, line_num


def update(n):
    t_n = t_grid[n]
    if n < K - 1:
        u[n + 1] = crank_nicolson_step(u[n], t_grid[n])
        line_true.set_data(x_grid, true_solution(x_grid, t_n))
        line_num.set_data(x_grid, u[n + 1])
    else:
        err = np.max(np.abs(u[n] - true_solution(x_grid, t_n)))
        print(err)
    # Обновляем линии
    ax.set_title(f"t = {t_n:.3f}")
    return line_true, line_num


ani = FuncAnimation(fig, update, frames=K, init_func=init, blit=False, interval=50)
plt.show()

# 0.00963618627453769
# 0.0014745817047524812
