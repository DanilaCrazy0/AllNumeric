#dz2
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

x0 = 1.0
y0 = 1.0
z0 = 1.0
t_span = [0.0, 1000.0]
n_points = 100000
dt = (t_span[1] - t_span[0]) / n_points
t_eval = np.linspace(t_span[0], t_span[1], n_points)

def equation(t, xyz):
    x, y, z = xyz
    dxdt = -x - 4 * y
    dydt = x + z ** 2
    dzdt = 1 + x
    return dxdt, dydt, dzdt

sol = solve_ivp(lambda t, xyz: equation(t, xyz),
                t_span, [x0, y0, z0],
                t_eval=t_eval, method='RK45', dense_output=True)

fig, ax = plt.subplots(2, 2, figsize=(15,13))
ax[0, 0] = fig.add_subplot(2, 2, 1, projection='3d')
ax[0,0].plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5)
ax[0,0].scatter(sol.y[0][0], sol.y[1][0], sol.y[2][0], marker='^', c='g', label='start')
ax[0,0].scatter(sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], marker='^', c='b', label='end')
ax[0,0].scatter(-1, 0.25, -1, marker='o', c='r', label='stationar')
ax[0,0].scatter(-1, 0.25, 1, marker='o', c='r')
ax[0,0].legend()
ax[0,0].set_title('Phase')
ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('y')
ax[0,0].set_zlabel('z')

ax[0,1].plot(sol.y[0], sol.y[1], linewidth=0.5)
ax[0,1].scatter(sol.y[0][0], sol.y[1][0], marker='^', c='g', label='start')
ax[0,1].scatter(sol.y[0][-1], sol.y[1][-1], marker='^', c='r', label='end')
ax[0,1].scatter(-1, 0.25, marker='o', c='r', label='stationar')
ax[0,1].legend()
ax[0,1].grid()
ax[0,1].set_title('Phase x-y')
ax[0,1].set_xlabel('x')
ax[0,1].set_ylabel('y')

ax[1,0].plot(sol.y[1], sol.y[2], linewidth=0.5)
ax[1,0].scatter(sol.y[1][0], sol.y[2][0], marker='^', c='g', label='start')
ax[1,0].scatter(sol.y[1][-1], sol.y[2][-1], marker='^', c='b', label='end')
ax[1,0].scatter(0.25, -1, marker='o', c='r', label='stationar')
ax[1,0].scatter(0.25, 1, marker='o', c='r', label='stationar')
ax[1,0].legend()
ax[1,0].grid()
ax[1,0].set_title('Phase y-z')
ax[1,0].set_xlabel('y')
ax[1,0].set_ylabel('z')

ax[1,1].plot(sol.y[0], sol.y[2], linewidth=0.5)
ax[1,1].scatter(sol.y[0][0], sol.y[2][0], marker='^', c='g', label='start')
ax[1,1].scatter(sol.y[0][-1], sol.y[2][-1], marker='^', c='b', label='end')
ax[1,1].scatter(-1, -1, marker='o', c='r', label='stationar')
ax[1,1].scatter(-1, 1, marker='o', c='r', label='stationar')
ax[1,1].legend()
ax[1,1].grid()
ax[1,1].set_title('Phase x-z')
ax[1,1].set_xlabel('x')
ax[1,1].set_ylabel('z')


plt.show()