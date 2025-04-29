import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation



def heaviside(x):
    if x < 0:
        return 0.0
    elif x == 0:
        return 0.5
    else:
        return 1.0

l = 2
a = 5
omega = 2 * np.pi
T = 0.1

h = 0.01

tau = 0.8 * h / a

# abs(a * cos(omega * t)) * tau/h <=1

x = np.linspace(0, l, int(l/h) + 1)


def phi(x):
    #return np.sin(2 * np.pi * x / l)
    return np.where((x > 0.3*l) & (x < 0.7*l), 1.0, 0.0)


def u_ext(x, t):
    return phi((x - (a / omega) * np.sin(omega * t)) % l)


u_curr = phi(x)

exact = u_ext(x, 0)


fig, ax = plt.subplots(figsize=(16, 7))
line1, = ax.plot(x, u_curr, '-', label="Численное решение")
line2, = ax.plot(x, exact, '-.', label="Точное решение")
ax.set_xlim(0, l)
ax.set_ylim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("u")
title = ax.set_title("Решение при t = 0.0")
ax.grid(True)
ax.legend()


moment = tau

def update(frame):
    global u_curr, moment, exact

    if moment >= T + tau:
        return line1, line2, title

    c = a * np.cos(omega * (moment + tau / 2))

    F = np.zeros(len(u_curr) + 1)

    F[1:-1] = c * (heaviside(c) * u_curr[:-1] + heaviside(-c) * u_curr[1:])

    F[0] = F[-3]
    F[-1] = F[2]

    #print(F_max)
    #print(F_min)

    exact = u_ext(x, moment)


    u_curr = u_curr - tau * (F[1:] - F[:-1]) / h

    moment += tau

    line1.set_ydata(u_curr)
    line2.set_ydata(exact)
    title.set_text("Решение при t = %.3f" % moment)

    return line1, line2, title


ani = FuncAnimation(fig, update, frames=int(T / tau), interval=100, blit=True)

plt.show()

print(np.mean(abs(u_curr - exact)))