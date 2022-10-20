import numpy as np
import methods as mp

# defining A1 as the matrix given in 2a
A1 = [[50, 107, 36], [25, 54, 20], [31, 66, 21]]
b = np.ones(3)

print("For item 2a, the given matrix A is: ")
print(A1)

# Solving for A1 = L1U1 and storing each triangular matrix in the respective variables
L1, U1 = mp.GetLU(A1)
print("\nThe LU factorization of matrix A1 is: ")
print(mp.LUKJI(A1))

print("\nIts lower triangular matrix is: ")
print(L1)
print("Its upper triangular matrix is: ")
print(U1)

# Solving for L1y = b using forward substitution
y = mp.ForwardSubRow(L1, b)
print("\nSolving for Ly = b we obtain the value of y: ")
print(y)

# Solving for U1x = y using backward substitution
x = mp.BackwardSubRow(U1, y)
print("\nSolving for Ux = b we obtain the value of x: ")
print(x)

# computation of residual max norm
max_n = mp.max_norm(A1, x, b)
print("\nResidual Max Norm:", max_n)

# defining A2 as the matrix given in 2b
A2 = [[10, 2, 1], [2, 20, -2], [-2, 3, 10]]
print("\nFor item 2b, the given matrix A is: ")
print(A2)

# Solving for A2 = L2U2 and storing each triangular matrix in the respective variables
L2, U2 = mp.GetLU(A2)
print("\nThe LU factorization of matrix A2 is: ")
print(mp.LUKJI(A2))

print("\nIts lower triangular matrix is: ")
print(L2)
print("Its upper triangular matrix is: ")
print(U2)

# Solving for L2y = b using forward substitution
y = mp.ForwardSubRow(L2, b)
print("\nSolving for Ly = b we obtain the value of y: ")
print(y)

# Solving for U2x = y using backward substitution
x = mp.BackwardSubRow(U2, y)
print("\nSolving for Ux = b we obtain the value of x: ")
print(x)

# computation of residual max norm
max_n = mp.max_norm(A2, x, b)
print("\nResidual Max Norm:", max_n)