import numpy as np

def LUKJI(A):
    """
    Solving for the LU factorization in A stored in A

    Parameters:
    A: matrix
       n x n

    Output
    ------
    A: matrix
       n x n, LU factorization of the input A
    """
    n = len(A)
    for k in range(0, n):
        for j in range(k + 1, n):
            A[j][k] = A[j][k] / A[k][k]
        for j in range(k + 1, n):
            for i in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
    return A

def GetLU(A):
    """
    Solving for the LU factors of A

    Parameters:
    A: matrix
       n x n

    Output
    ------
    L: lower triangular matrix
    U: upper triangular matrix
    """
    A = LUKJI(A)
    n = len(A)
    L = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    U = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(0, n):
        L[i][i] = 1
        for j in range(0, i):
            L[i][j] = A[i][j]
        for j in range(i, n):
            U[i][j] = A[i][j]
    return (L, U)

def ForwardSubRow(L,b):
    """
    Solving Lx = b using forward substitution by rows.
    
    Parameters:
    L : matrix
        n x n lower triangular matrix
    b : vector
        n x 1   
        
    Output
    ------
    x : vector
        n x 1, solution to Lx = b 
    """
    n = len(L)
    x = np.zeros(n, dtype=float)
    x[0] = b[0]/L[0][0]
    for i in range(1,n):
        s = 0
        for j in range(0,i):
            s = s + L[i][j]*x[j]
        x[i] = (b[i]-s)/L[i][i]
    return x

def ForwardSubCol(L,b):
    """
    Solving Lx = b using foward substitution by columns.
    
    Parameters:
    L : matrix
        n x n lower riangular matrix
    b : vector
        n x 1   
        
    Output
    ------
    x : vector
        n x 1, solution to Lx = b 
    """
    n = len(L)
    for j in range(0,n-1):
        b[j] = b[j]/L[j][j]
        for i in range(j+1,n):
            b[i] = b[i] - L[i][j]*b[j]
    b[n-1] = b[n-1]/L[n-1][n-1]
    return b

def BackwardSubRow(U, b):
    """
    Solving Ux = b using backward substitution by rows.
    
    Parameters:
    U : matrix
        n x n upper triangular matrix
    b : vector
        n x 1   
        
    Output
    ------
    x : vector
        n x 1, solution to Ux = b 
    """
    n = len(U)
    x = np.zeros(n)
    x[n-1] = b[n-1]/U[n-1][n-1] 
    for i in range(n-2,-1,-1):
        s = 0
        for j in range(i+1,n):
            s = s + U[i][j]*x[j]
        x[i] = (b[i]-s)/U[i][i]   
    return x

def BackwardSubCol(U, b):
    """
    Solving Ux = b using backward substitution by columns.
    
    Parameters
    ----------
    U : matrix
        n x n upper triangular matrix
    b : vector
        n x 1   
        
    Output
    ------
    b : Vector
        Solution to Ux = b.
    """
    n = len(U)
    for j in range(n - 1, 0, -1):
        b[j] = b[j] / U[j][j]
        for i in range(0, j):
            b[i] = b[i] - U[i][j] * b[j]
    b[0] = b[0] / U[0][0]
    return b

def max_norm(A, x, d):
    """
    Solving for the Residual Max Norm

    Parameters
    ------
    A : matrix
        n x n
    x : vector
        n x 1
    d : vector
        n x 1

    Output
    ------
    max_n : computed residual max norm
    """
    max_n = 0
    maxn = d - np.dot(A, x)
    for i in maxn:
        max_n += abs(i)
    return max_n