import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры задачи
A = -1.0
B = 1.0
N = 101
N1 = 1001
T = 2.0
K = 1001
t0 = 0.0
x0 = 0.0
eps_x = 0.2
eps_t = 0.2
a = 1.0

x_grid, h = np.linspace(A, B, N, retstep=True)
t_grid, tau = np.linspace(0, T, K, retstep=True)
x_grid1, h1 = np.linspace(A, B, N1, retstep=True)
l = (B - A) / 2
center = (A + B) / 2

# Середины ячеек
x_half = (x_grid[:-1] + x_grid[1:]) / 2
x_half1 = (x_grid1[:-1] + x_grid1[1:]) / 2

print(f"tau*a = {tau*a}, h = {h / a}")
print(f"tau*a = {tau*a}, h1 = {h1 / a}")


# Вспомогательные функции
def hevisaid(x):
    return np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))


def ksi(x, x0, eps):
    return np.abs(x - x0) / eps


def phi_1(x, x0, eps):
    return hevisaid(1 - ksi(x, x0, eps))

def phi_2(x, x0, eps):
    return phi_1(x, x0, eps) * (1 - ksi(x, x0, eps) ** 2)


def phi_3(x, x0, eps):
    return phi_1(x, x0, eps) * np.cos(np.pi * ksi(x, x0, eps) / 2) ** 3


# Граничные условия и начальное условие
def mu_L(t, t0=t0, eps=eps_t):
    return phi_1(t, t0, eps)


def mu_R(t, t0=t0, eps=eps_t):
    return phi_1(t, t0, eps)


def phi_func(x, x0=x0, eps=eps_x):
    return phi_2(x, x0, eps)


def true_solution(x, t, l, a):
    X = l * np.arcsinh(np.sinh(1) * np.exp(-a * t / l))
    u0 = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi < -X:
            arg = t + (l / a) * np.log(np.sinh(np.abs(xi) / l) / np.sinh(1))
            u0[i] = -np.tanh(1) / np.tanh(xi / l) * mu_L(arg)
        elif np.abs(xi) <= X:
            sinh_xi_l = np.sinh(xi / l)
            exp_at_l = np.exp(a * t / l)
            arg_phi = l * np.arcsinh(sinh_xi_l * exp_at_l)
            numerator = np.cosh(xi / l)
            denominator = np.sqrt(np.sinh(xi / l) ** 2 + np.exp(-2 * a * t / l))
            u0[i] = (numerator / denominator) * phi_func(arg_phi)
        else:  # xi > X
            arg = t + (l / a) * np.log(np.sinh(np.abs(xi) / l) / np.sinh(1))
            u0[i] = np.tanh(1) / np.tanh(xi / l) * mu_R(arg)

    return u0


# Функция потока
def flux(u, x):
    return a * np.tanh(x / l) * u


# Консервативная схема (аналог схемы Годунова)
def conservative_scheme_step(u_old, t, tau, h, x_grid, x_half):
    u_new = u_old.copy()

    # Вычисляем скорости на серединах ячеек
    a_half = -a * np.tanh(x_half / l)

    # Вычисляем значения u на границах ячеек с учетом направления скорости
    u_half = np.where(a_half > 0, u_old[:-1], u_old[1:])

    # Вычисляем потоки на границах ячеек
    F_half = -flux(u_half, x_half)

    # Обновляем решение
    u_new[1:-1] = u_old[1:-1] - (tau / h) * (F_half[1:] - F_half[:-1])

    # Граничные условия с учетом направления скорости
    # Левая граница
    a_left = a_half[0]
    if a_left > 0:  # Характеристика входит слева
        u_new[0] = mu_L(t)
    else:           # Характеристика выходит, берем значение изнутри
        u_new[0] = u_new[1]

    # Правая граница
    a_right = a_half[-1]
    if a_right < 0: # Характеристика входит справа
        u_new[-1] = mu_R(t)
    else:           # Характеристика выходит, берем значение изнутри
        u_new[-1] = u_new[-2]

    return u_new


# Инициализация решения
u = np.zeros((K, N))
u1 = np.zeros((K, N1))
u[0, :] = phi_func(x_grid)
u1[0, :] = phi_func(x_grid1)

error_h = np.zeros(K)
error_h1 = np.zeros(K)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


line_true, = ax1.plot(x_grid, true_solution(x_grid, 0, l, a), 'b-', label='Точное решение')
line_num, = ax1.plot(x_grid, u[0], 'r--', label=f'h={h:.3f}')
line_num1, = ax1.plot(x_grid1, u1[0], 'g--', label=f'h={h1:.3f}')
ax1.set_ylabel('u')
ax1.set_title('t = 0.0')
ax1.legend()
ax1.set_ylim(-0.1, 1.5)
ax1.grid()


line_err, = ax2.plot([], [], 'r-', label=f'Ошибка (h={h:.3f})')
line_err1, = ax2.plot([], [], 'g-', label=f'Ошибка (h={h1:.3f})')
ax2.set_xlabel('Время')
ax2.set_ylabel('Макс. ошибка')
ax2.legend()
ax2.grid()
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(tau, T)
ax2.set_ylim(1e-4, 3e0)


def init():
    # Инициализация верхнего графика
    line_true.set_data(x_grid, true_solution(x_grid, 0.0, l, a))
    line_num.set_data(x_grid, u[0])
    line_num1.set_data(x_grid1, u1[0])

    # Инициализация нижнего графика
    line_err.set_data([], [])
    line_err1.set_data([], [])
    return line_true, line_num, line_num1, line_err, line_err1


def update(n):
    t_n = t_grid[n]

    # Вычисляем точные решения для обеих сеток
    u_true = true_solution(x_grid, t_n, l, a)
    u_true1 = true_solution(x_grid1, t_n, l, a)

    if n < K - 1:
        # Обновляем решения
        u[n + 1] = conservative_scheme_step(u[n], t_n, tau, h, x_grid, x_half)
        u1[n + 1] = conservative_scheme_step(u1[n], t_n, tau, h1, x_grid1, x_half1)

        # Вычисляем ошибки
        error_h[n] = np.mean(np.abs(u[n] - u_true))
        error_h1[n] = np.mean(np.abs(u1[n] - u_true1))

        # Обновляем линии
        line_true.set_data(x_grid, u_true)
        line_num.set_data(x_grid, u[n + 1])
        line_num1.set_data(x_grid1, u1[n + 1])
        line_err.set_data(t_grid[:n + 1], error_h[:n + 1])
        line_err1.set_data(t_grid[:n + 1], error_h1[:n + 1])
    else:
        error_h[n] = np.max(np.abs(u[n] - u_true))
        error_h1[n] = np.max(np.abs(u1[n] - u_true1))

    # Обновление заголовка и пределов
    ax1.set_title(f"t = {t_n:.3f}")
    y_min = np.min([np.min(u[n]), np.min(u_true)])
    y_max = np.max([np.max(u[n]), np.max(u_true)])
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(tau, T)
    ax2.set_ylim(1e-4, 3e0)

    return line_true, line_num, line_num1, line_err, line_err1


ani = FuncAnimation(fig, update, frames=K, init_func=init, blit=False, interval=50)
plt.show()