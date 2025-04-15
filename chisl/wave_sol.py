import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def true_func(x, t):
    return x ** 2 + np.arccos(x * t / 2)


def der_2(u_i):
    padded_f = np.pad(u_i, (1, 1), mode='constant', constant_values=0)
    padded_f[1:-1] = (padded_f[2:] - 2 * padded_f[1:-1] + padded_f[:-2]) / h ** 2
    return padded_f[1:-1]


def grid(A, B, N, T, K):
    x_grid, h = np.linspace(A, B, N, retstep=True)
    t_grid, tau = np.linspace(0, T, K, retstep=True)
    return x_grid, t_grid, h, tau


A = 0
B = 1
N = 101
T = 1
K = 101

a_2 = 0.5
alpha_l = 0
beta_l = -1
alpha_r = 1
beta_r = 1

def f(x, t):
    return x * t * (t ** 2 - 2 * x ** 2) / (2 * (4 - (x ** 2) * (t ** 2)) ** 1.5) - 1

def phi(x):
    return x ** 2 + np.pi / 2

def psi(x):
    return -x / 2

def mu_l(t):
    return -t / 2

def mu_r(t):
    return 3 + np.arccos(t / 2) - t / (4 - t ** 2) ** 0.5


def fill_boards(u_k, k):
    u_k[0] = (2 * mu_l(tau * k) * h + 4 * beta_l * u_k[1] - beta_l * u_k[2]) / (2 * h * alpha_l + 3 * beta_l)
    u_k[-1] = (2 * mu_r(tau * k) * h + 4 * beta_r * u_k[-2] - beta_r * u_k[-3]) / (2 * h * alpha_r + 3 * beta_r)


x_grid, t_grid, h, tau = grid(A, B, N, T, K)
u = np.zeros((3, x_grid.shape[0]))

u[0] = phi(x_grid)
fill_boards(u[0], 0)
u[1] = phi(x_grid) + tau * psi(x_grid) + tau ** 2 / 2 * (a_2 * der_2(phi(x_grid)) + f(x_grid, 0))
fill_boards(u[1], 1)
u_true = true_func(x_grid, 0.0)

k = 2
err = [np.max(u[0]-u_true)]

def solution_step():
    global k, u_true, u
    u[2][1:-1] = 2 * u[1, 1:-1] - u[0, 1:-1] + tau ** 2 * a_2 / h ** 2 * (u[1, 2:] - 2 * u[1, 1:-1] + u[1, :-2]) + tau ** 2 * f(x_grid[1:-1], (k-1) * tau)
    fill_boards(u[2], k)
    u[0] = u[1].copy()
    u[1] = u[2].copy()

    u_true = true_func(x_grid, tau * (k-1))

    if k - 2 <= K:
        k += 1
    else:
        ani.event_source.stop()


fig = plt.figure(1, dpi=150)
ax = plt.gca()

# Создали заголовок
title = plt.title("t = 0.0")

# Создали линии с разными цветами и метками
line, = plt.plot(x_grid, u_true, color='blue', label='Точное решение')  # Синий цвет для точного решения
line2, = plt.plot(x_grid, u[0], color='red', label='Численное решение', linestyle='--')  # Красный цвет для численного решения

# Настроили оси
ax.set_xlim(xmin=A, xmax=B)
ax.set_ylim(ymin=-1.2, ymax=1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")

# Добавили легенду
plt.legend()

fig.tight_layout()

def update(frame):
    global err
    solution_step()
    # Обновляем заголовок
    title.set_text("t = %.3f" % ((k - 1) * tau))

    # Адаптируем ось y под решение
    y_min = np.min(np.minimum(u[0], u_true))
    y_max = np.max(np.maximum(u[0], u_true))
    ax.set_ylim(y_min, y_max)

    # Заменяем данные линии
    line.set_ydata(u_true)
    line2.set_ydata(u[0])

    err.append(np.max(np.abs(u[0] - u_true)))

    if k >= K:
        print(err[-1])
        return line, line2, title


print(f'Число Куранта: {tau * a_2 ** 0.5 / h}')
ani = FuncAnimation(fig, update, frames=K, interval=50)
plt.show()