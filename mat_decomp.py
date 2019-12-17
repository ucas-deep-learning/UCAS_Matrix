import numpy as np

def mat_decomp(A, type ='LU'):
    """
    mat_decomp函数用于实现矩阵的分解，包括LU、QR（Gram-Schmidt）、
    Orthogonal Reduction (Householder reduction 和 Givens reduction)
    :param A:       输入矩阵
    :param type:    选择矩阵分解的方式，可选参数如下:
            ‘LU’:               LU分解,A=LU,L为下三角矩阵，U为上三角矩阵
            ‘GramSchmidt’:      Gram-Schmidt分解,A=QR,Q为正交矩阵，R为上三角矩阵
            ‘Householder’:      Householder约减,A=QR,Q为正交矩阵，R为上三角矩阵
            ‘Givens’:           Givens,A=QR,Q为正交矩阵，R为上三角矩阵
            default:            LU分解
    :return:
    """
    print("输入矩阵:",A)
    if type == 'LU':
        LU(A)
    elif type =='GramSchmidt':
        GramSchmidt(A)
    elif type =='Householder':
        Householder(A)
    elif type =='Givens':
        Givens(A)
    else:
        print("请重新输入矩阵的分解方式！可选:‘LU’,‘GramSchmidt’,‘Householder’,‘Givens’,默认为LU")

def LU(A):

    """
    对矩阵A进行LU分解,A=LU,L为下三角矩阵，U为上三角矩阵
    :param A: 输入矩阵，要求是一个方阵
    :return: P,L,U
            P:对A行交换的初等行变换矩阵
            L:下三角矩阵
            U:上三角矩阵
    """
    A = np.array(A)
    m, n = np.shape(A)

    # 初始化P为单位阵
    P = np.identity(n)

    L = np.zeros((m, n))
    U = np.zeros((m, n))
    temp = np.zeros((m, n))
    # A should be a square matrix.
    if m != n:
        print("请输入一个方阵！")
        return

    # 判断A是否是奇异矩阵，如果是则返回。
    if np.linalg.matrix_rank(A) != n:
        print("输入矩阵是奇异矩阵，无法LU分解")
        return

    # 判断A的各阶子矩阵A_sub 是否是奇异矩阵，如果是则对A行交换，即PA=LU。
    else:
        for i in range(n):
            A_sub = np.zeros((i + 1, i + 1))
            for j in range(i + 1):
                for k in range(i + 1):
                    A_sub[j][k] = A[j][k]
            r = np.linalg.matrix_rank(A_sub)
            t = j + 1

            while r != (i + 1):
                print("A的子矩阵为奇异矩阵，需要对A行变换，使其变为非奇异矩阵，即采用PLU分解。")
                # 对A行交换
                for q in range(j + 1):
                    A_sub[j, q] = A[t, q]
                for p in range(n):
                    temp[t, p] = A[t, p]
                    A[t, p] = A[j, p]
                    A[j, p] = temp[t, p]
                T = np.zeros((m, n))
                for e in range(n):
                    T[e][e] = 1
                T[j][j] = 0
                T[t][t] = 0
                T[j][t] = 1
                T[t][j] = 1
                P = np.dot(T, P)
                t = t + 1
                r = np.linalg.matrix_rank(A_sub)

    # 计算L,U矩阵
    for i in range(n):
        L[i][i] = 1     # L矩阵主对角元素上都为1
        if i == 0:
            # 计算L矩阵的第一列和U矩阵第一行
            for j in range(n):
                L[j][0] = A[j][0] / A[0][0]
                U[0][j] = A[0][j]

        else:
            # 计算 U
            for j in range(i, n):
                temp = 0
                for k in range(i):
                    temp = temp + L[i][k] * U[k][j]
                U[i][j] = A[i][j] - temp
            # 计算 L
            for j in range(i + 1, n):
                temp = 0
                for k in range(i):
                    temp = temp + L[j][k] * U[k][i]
                L[j][i] = (A[j][i] - temp) / U[i][i]

    # 圆整，保留三位有效数字
    L = np.around(L, decimals=3)
    U = np.around(U, decimals=3)
    if (P == np.identity(n)).all():
        print("LU分解")
        print("L: {}\nU: {}\n".format(L, U))
    else:
        print("PLU分解")
        print("P: {}\nL: {}\nU: {}\n".format(P,L,U))
    return P,L,U

def GramSchmidt(A):
    """
    对矩阵A进行施密特正交化
    :param A: 输入矩阵，要求是一个方阵。
    :return: Q,R
             GramSchmidt正交化得到后矩阵Q为正交矩阵，A=QR,R为上三角矩阵
    """
    print("GramSchmidt:")
    A = np.array(A)
    m,n = np.shape(A)
    Q = np.zeros((m,n))

    # 判断是否是方阵
    if m !=n:
        print("请输入一个方阵！")
        return 0

    for i in range (n):
        if i != 0:
            x = A[:, i]
            for j in range (i):
                x = x - np.dot(Q[:,j].T, A[:, i]) * Q[:,j] #减去前面每个维度的分量
            Q[:,i] = x / np.linalg.norm(x) #标准化
        #     初始化
        else:
            x = A[:, 0]
            Q[:,0] = x / np.linalg.norm(x)

    # A=QR,R=Q的转置乘以A
    R = np.dot(Q.T,A)

    # 圆整，保留三位小数
    Q = np.around(Q, decimals=3)
    R = np.around(R, decimals=3)

    print("Q:", Q)
    print("R:", R)
    return Q,R

def Householder(A):
    """
    对矩阵A进行Householder约减
    :param A: 输入矩阵，要求是一个方阵。
    :return: Q,R
             Householder约减得到一个上三角矩阵R,A=QR,Q为正交矩阵
    """

    print("Householder:")
    A = np.array(A)
    m, n = np.shape(A)
    # 判断是否是方阵
    if m != n:
        print("请输入一个方阵！")
        return 0

    Q = np.identity(n)
    for i in range(n-1):
        U = np.array(A[i:,i])   #第一列完成后，第二列只需考虑主对角线下方的元素
        U[0] = U[0]-np.linalg.norm(U)
        U = np.mat(U).T

        P_ = np.zeros((n,n))
        P_[i:n,i:n] = 2*np.dot(U,U.T)/np.dot(U.T,U)
        P = np.identity(n)-P_   # 得到反射算子

        A = np.dot(P,A)
        Q = np.dot(P,Q)

    R = A
    Q = Q.T
    # 圆整，保留三位有效数字
    R = np.around(R, decimals=3)
    Q = np.around(Q, decimals=3)

    print("Q:", Q)
    print("R:", R)
    return Q,R

def Givens(A):
    """
    对矩阵A进行Givens约减
    :param A: 输入矩阵，要求是一个方阵。
    :return: Q,R Givens约减得到一个上三角矩阵R,A=QR,Q为正交矩阵
    """

    print("Givens:")
    A = np.array(A)
    m, n = np.shape(A)
    # 判断是否是方阵
    if m != n:
        print("请输入一个方阵！")
        return 0

    Q = np.identity(n)
    for i in range(n-1):
        for j in range (i+1,n):     #j比i大，根据[i,i]和其下方的元素计算旋转矩阵，使[i,i]下方元素全为0。
            P = np.zeros((n, n))
            U = np.array(A[0:, i])
            c = U[i] / (U[i] * U[i] + U[j] * U[j]) ** 0.5
            s = U[j] / (U[i] * U[i] + U[j] * U[j]) ** 0.5
            P[i, i] = c
            P[j, j] = c
            P[i, j] = s
            P[j, i] = -s
            P[n - i - j, n - i - j] = 1 #得到旋转矩阵P
            A = np.dot(P, A)

            Q = np.dot(P, Q)

    R = A
    Q = Q.T
    # 圆整，保留三位有效数字
    R = np.around(R, decimals=3)
    Q = np.around(Q, decimals=3)

    print("Q:", Q)
    print("R:", R)
    return Q,R

if __name__ == "__main__":
    A1 = np.array([[2,2,2],
                  [4,7,7],
                  [6,18,22]])
    A2 = np.array([[1, 4, 5, 6],
              [1, 4, 6, 7],
              [1, 4, 6, 8],
              [5, 1, 5, 12]])
    A3 = np.array([[0,-20,-14],
                   [3,27,-4],
                   [4,11,-2]])
    mat_decomp(A1,'LU')
    mat_decomp(A2,'LU')
    mat_decomp(A3,'GramSchmidt')
    mat_decomp(A3,'Householder')
    mat_decomp(A3,'Givens')
