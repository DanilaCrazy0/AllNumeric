import numpy as np
from scipy.linalg import expm, inv

# Исходные данные
A = np.array([
    [-4, 10, -1, -7, 3],
    [-3, 10, -1, -6, 2],
    [-4, 11, -1, -5, 1],
    [-4, 10, 0, -4, 0],
    [-4, 10, -1, -2, -1]
], dtype=float)

S = np.array([
    [11, 1, 1, 5, 31],
    [6, 1, 2, 9, 46],
    [9, 2, 3, 12, 53],
    [8, 2, 4, 14, 56],
    [9, 3, 5, 15, 57]
], dtype=float)

x0 = np.array([1, 1, 1, 1, 1], dtype=float)

# (a) QR-алгоритм для нахождения собственных значений
def qr_algorithm(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    eigvals = []
    for _ in range(max_iter):
        Q, R = np.linalg.qr(A)
        A_new = R @ Q
        if np.allclose(A, A_new, atol=tol):
            break
        A = A_new
    # Извлекаем собственные значения из диагонали
    eigvals = np.diag(A)
    return np.sort(eigvals)

# Выполнение QR-алгоритма
computed_eigenvalues = qr_algorithm(A.copy())
print("Собственные значения через QR-алгоритм:", computed_eigenvalues)

# (b) Проверка A = SΛS⁻¹
Lambda = np.diag(np.sort(computed_eigenvalues))  # Сортируем собственные значения
S_inv = inv(S)
A_reconstructed = S @ Lambda @ S_inv

# Проверка с учётом погрешностей
print("\nПроверка A = SΛS⁻¹ (максимальная погрешность):",
      np.max(np.abs(A - A_reconstructed)))

# (c) Решение системы и устойчивость
def solution_x(t):
    Lambda_t = np.diag(np.exp(np.diag(Lambda) * t))
    return S @ Lambda_t @ S_inv @ x0

# Анализ устойчивости
real_parts = np.real(np.diag(Lambda))
is_stable = np.all(real_parts < 0)
print("\nСистема устойчива?" , "Да" if is_stable else "Нет")

# (d) Модификация матрицы A для устойчивости
# Вариант i: Замена положительных λ на -λ
Lambda_i = np.diag([-λ if λ > 0 else λ for λ in np.diag(Lambda)])
A_i = S @ Lambda_i @ S_inv

# Вариант ii: Сдвиг Λ на α = 32
alpha = 32
Lambda_ii = np.diag(np.diag(Lambda) - alpha)
A_ii = S @ Lambda_ii @ S_inv

# Проверка устойчивости модификаций
real_parts_i = np.real(np.diag(Lambda_i))
is_stable_i = np.all(real_parts_i < 0)
print("\nПосле модификации (i): Устойчива?", "Да" if is_stable_i else "Нет")

real_parts_ii = np.real(np.diag(Lambda_ii))
is_stable_ii = np.all(real_parts_ii < 0)
print("После модификации (ii): Устойчива?", "Да" if is_stable_ii else "Нет")

# Пример использования решения (t = 1)
t = 1.0
x_t = solution_x(t)
print("\nРешение x(t) при t =", t, ":", x_t)