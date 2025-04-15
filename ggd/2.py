import numpy as np
import matplotlib.pyplot as plt

a = 1.0
Gamma = 1.0
U = 0.0
V = 1.0

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

mask = R >= a
vx = np.zeros_like(X)
vy = np.zeros_like(Y)

vx[mask] = (np.sin(Theta[mask])) / (2 * np.pi * R[mask]) + (a**2 * np.sin(2 * Theta[mask])) / (R[mask]**2)
vy[mask] = (np.cos(Theta[mask])) / (2 * np.pi * R[mask]) - (a**2 * np.cos(2 * Theta[mask])) / (R[mask]**2)

plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, vx, vy, density=2, color='blue', linewidth=1, arrowsize=1.5)

circle = plt.Circle((0, 0), a, color='red', fill=True, alpha=0.3)
plt.gca().add_patch(circle)

plt.title('Векторное поле скорости для цилиндра (U=0, V=1, Γ=1)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()