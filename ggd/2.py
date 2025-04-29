import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Параметры задачи
y_max = 10  # Максимальное расстояние вдоль струи
x_max = 2   # Максимальное поперечное расстояние
num_points = 50  # Количество точек для расчёта (меньше для лучшей читаемости стрелок)

# Сетка координат
y = np.linspace(0.1, y_max, num_points)  # Чтобы избежать деления на 0
x = np.linspace(-x_max, x_max, num_points)
Y, X = np.meshgrid(y, x)

# Модель скорости (u) в турбулентной струе
# Для плоской струи: u ~ 1/sqrt(y) * exp(-x² / (2*b(y)²)), где b(y) ~ y
u0 = 1.0  # Масштаб скорости
b0 = 0.5  # Масштаб ширины

# Ширина струи растёт линейно с y
b = b0 * Y

# Профиль скорости (гауссовский в поперечном сечении)
U = (u0 / np.sqrt(Y)) * np.exp(-(X**2) / (2 * b**2))

# Компоненты вектора скорости:
# Ux (поперечная) = 0 (струя симметрична)
# Uy (продольная) = U(x,y)
Ux = np.zeros_like(U)  # Поперечная компонента (пренебрегаем)
Uy = U  # Продольная компонента

# Нормализация векторов для лучшей визуализации
magnitude = np.sqrt(Ux**2 + Uy**2)
Ux_norm = Ux / magnitude  # Нормированные компоненты
Uy_norm = Uy / magnitude

# Создание графика
plt.figure(figsize=(12, 6))

# Цветовая карта скорости
speed = np.sqrt(Ux**2 + Uy**2)
plt.pcolormesh(X, Y, speed, cmap=cm.viridis, shading='auto')
plt.colorbar(label='Скорость |u|')

# Векторы скорости (избегаем слишком частого отображения)
skip = 2  # Пропуск точек для стрелок
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           Ux_norm[::skip, ::skip], Uy_norm[::skip, ::skip],
           color='white', scale=30, width=0.002)

# Настройка графика
plt.xlabel('Поперечное направление (x)')
plt.ylabel('Продольное направление (y)')
plt.title('Поле скоростей в турбулентной затопленной струе')
plt.grid(alpha=0.3)
plt.xlim(-x_max, x_max)
plt.ylim(0, y_max)
plt.show()