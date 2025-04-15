import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

pd.set_option('display.max_columns', None)
k = 5
n_max = 400
n_matrices = 100

df = pd.read_csv('data.csv')
df = df.iloc[:, 1:]
df.columns = ['norm_f', 'norm_1', 'norm_2', 'norm_inf', 'norm_max', 'norm_f * n ** k']
out = pd.DataFrame(np.zeros((6, 6)), index=['norm_f', 'norm_1', 'norm_2', 'norm_inf', 'norm_max', 'norm_f * n ** k'],
                   columns=['norm_f', 'norm_1', 'norm_2', 'norm_inf', 'norm_max', 'norm_f * n ** k'])

fig, ax = plt.subplots(6, 6, figsize=(30, 15))

for i, col in enumerate(df.columns.tolist()):
    for j, col_out in enumerate(out.columns.tolist()):
        if col == col_out:
            ax[i, j].axis('off')
            continue

        f_n = df[col] / df[col_out]

        x = np.log(np.arange(2, len(df) + 2))
        y = np.log(f_n)

        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        out.loc[col, col_out] = round(slope, 2)

        ax[i, j].scatter(x, y)
        ax[i, j].plot(x, x * slope + intercept, c='r')
        ax[i, j].set_title(f'{col} / {col_out}')
        ax[i, j].grid()

print(out)
out.to_excel('out.xlsx')
fig.tight_layout()
plt.show()
