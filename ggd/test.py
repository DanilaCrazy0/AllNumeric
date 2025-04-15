import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def calculate_velocity_field(U, V, Gamma, a=1.0, xlim=(-5, 5), ylim=(-5, 5), resolution=30):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Комплексный потенциал
    def complex_potential(z):
        if np.abs(z) <= a:
            return np.nan + 1j * np.nan
        return (Gamma / (2 * np.pi * 1j)) * np.log(z) - (U + 1j * V) * a ** 2 / z

    # Комплексная скорость (производная потенциала)
    def complex_velocity(z):
        if np.abs(z) <= a:
            return np.nan + 1j * np.nan
        return (Gamma / (2 * np.pi * 1j)) / z + (U + 1j * V) * a ** 2 / z ** 2

    # Вычисляем скорости
    V_complex = np.vectorize(complex_velocity)(Z)
    Vx = np.real(V_complex)
    Vy = -np.imag(V_complex)

    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, Vx, Vy, density=2, color='blue', linewidth=1, arrowsize=1.5)

    circle = plt.Circle((0, 0), a, color='red', fill=True, alpha=0.3)
    plt.gca().add_patch(circle)

    plt.title(f'Поле скоростей (U={U}, V={V}, Γ={Gamma})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


# Примеры использования:
calculate_velocity_field(U=0, V=1, Gamma=0)