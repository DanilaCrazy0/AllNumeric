import scipy
import matplotlib.pyplot as plt
import numpy as np

n = 5
num = 10000
matricies = []
errs = []
alphas = np.logspace(-7, 3, num=num)

def LU(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for k in range(n-1):
        if U[k, k] == 0:
            print('Не годится')
            return None
        for i in range(k+1, n):
            l = U[i, k] / U[k, k]
            L[i, k] = l
            U[i] = U[i] - l * U[k]
    return L, U

for alpha in alphas:
    A = np.random.randint(1, 10, size=(n, n)).astype(float)
    for i in range(n):
        A[i, i] = alpha * (np.sum(np.abs(A[i, :i])) + np.sum(np.abs(A[i, i+1:])))
    matricies.append(A)

for M in matricies:
    L, U = LU(M)
    err = np.linalg.norm(M-L@U, ord='fro') / np.linalg.norm(M, ord='fro')
    errs.append(err)

fid, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].plot(alphas, errs)
ax[0].set_xscale('log')
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('Error')
ax[0].grid()

ax[1].plot(alphas[int(0.65*num):], errs[int(0.65*num):])
ax[1].set_xscale('log')
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('Error')
ax[1].grid()
plt.show()