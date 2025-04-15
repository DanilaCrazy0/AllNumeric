import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def func(x, t):
    return x ** 2 + np.arccos(x * t / 2)


A = 0.0
B = 1.0
N = 301

# Строим сетку и сразу возращаем шаг
x, h = np.linspace(A, B, N, retstep=True)
print(h / 0.5 ** 0.5)

# Текущее время
curr_time = 0.0

# Инициализируем функцию
u1 = func(x, 0.0)


# Шаг условного решателя
def solution_step():
    # Будем тупо обновлять глобальные переменные
    global curr_time, u1

    if curr_time <= 1:
        u1 = func(x, curr_time)
        curr_time += 0.01


fig = plt.figure(1, dpi=200)
ax = plt.gca()

# Создали заголовок
title = plt.title("t = 0.0")

line, = plt.plot(x, u1) # Создали линию

# Настроили оси
ax.set_xlim(xmin=A, xmax=B)
ax.set_ylim(ymin=-1.2, ymax=1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")

fig.tight_layout()


def update(frame):
    solution_step()
        
    # Обновляем заголовок
    title.set_text("t = %.3f" % curr_time)
    
    # Адаптируем ось y под решение
    ax.set_ylim(np.min(u1), np.max(u1))

    # Заменяем данные линии
    line.set_ydata(u1)    


ani = FuncAnimation(fig, update, frames=1000, interval=50)
plt.show()

