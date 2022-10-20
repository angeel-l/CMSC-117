import numpy as np
import methods as mp

# number of entries
n = 100

# constructing arrays a, b, c
a = np.ones(n, dtype=float)
b = np.arange(1, n + 1, dtype=float)
c = np.tile([1, 3], 50)

# block function that returns upper triangular matrix U
def block(n, a, b, c):  
    U = np.zeros((n,n))
    U = np.diag(a) + np.diag(b[:-1], 1) + np.diag(c[:-2], 2) + np.diag(a[:-3], 3)
    return U

# constructing matrix A for item 1a
A = block(100, a, b, c)
print("The matrix A has been constructed.\n", A)

# using BackwardSubRow
A_backrow = mp.BackwardSubRow(A, a)
print("\nSolving for x using backward substitution with rows:")
print(A_backrow)

# computation of residual max norm
max_n = mp.max_norm(A, A_backrow, a)
print("Residual Max Norm:", max_n)

# using BackwardSubCol
print("\nUsing backward substitution with columns:")
A_backcol = mp.BackwardSubCol(A, a)
print(A_backcol)

# computation of residual max norm
max_n = mp.max_norm(A, A_backcol, a)
print("Residual Max Norm:", max_n)

# transposing matrix A for item 1b
A = A.transpose()
print("\nThe matrix A has been transposed.\n", A)

# using ForwardSubRow
print("\nUsing forward substitution with rows:")
A_forrow = mp.ForwardSubRow(A, b)
print(A_forrow)

# computation of residual max norm
max_n = mp.max_norm(A, A_forrow, b)
print("Residual Max Norm:", max_n)

# using ForwardSubCol
print("\nUsing forward substitution with columns:")
A_forcol = mp.ForwardSubCol(A, b)
print(A_forcol)

# computation of residual max norm
max_n = mp.max_norm(A, A_forcol, b)
print("Residual Max Norm:", max_n)