import numpy as np

A = np.array([[-4, 10, -1, -7, 3],
              [-3, 10, -1, -6, 2],
              [-4, 11, -1, -5, 1],
              [-4, 10, 0, -4, 0],
              [-4, 10, -1, -2, -1]])

def QR(A):
    n = A.shape[1]
    Q = np.zeros_like(A, dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def qr_algorithm(A, max_iter=1000, tol=1e-10):
    n = A.shape[0]
    A_k = A.copy().astype(float)
    sv = np.eye(n)

    for k in range(max_iter):
        shift = A_k[-1, -1]
        Q_k, R_k = QR(A_k - shift * np.eye(n))
        A_k = R_k @ Q_k + shift * np.eye(n)
        sv = sv @ Q_k

        # Проверка сходимости
        off_diag = np.sum(np.abs(np.tril(A_k, -1)))
        if off_diag < tol:
            break

    # Сортировка собственных значений и векторов
    idx = np.argsort(np.diag(A_k))
    sigma = np.diag(np.diag(A_k)[idx])
    sv = sv[:, idx]

    return sigma, sv


Q, R = QR(A)
sigma, sv = qr_algorithm(A)

print("Приближённые собственные значения:")
print(sigma)

print("Приближённые собственные векторы:")
print(sv)

print("A = S * sigma * S.inv:")
print(sv @ sigma @ np.linalg.inv(sv))