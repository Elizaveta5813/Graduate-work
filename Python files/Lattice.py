import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def cP(N):

    r = 0.5
    L = (N)**(1/3.) * 2 * r

    x = np.array([])
    y = np.array([])
    z = np.array([])

    s = -1

    for i in range(int(L/r)):
        for j in range(int(L/r)):
            for k in range(int(L/r)):
                if (((i * 2 * r) ** 2 + (j * 2 * r)**2 + (k * 2 * r)**2) <= L**2):
                    s = s + 1
                    x = np.append(x, i * 2 * r)
                    y = np.append(y, j * 2 * r)
                    z = np.append(z, k * 2 * r)

    for i in range(int(x.shape[0])):
        if (((x[i]!=0) & (y[i]!=0))):
            x = np.append(x, [-x[i], x[i], -x[i]])
            y = np.append(y, [y[i], -y[i], -y[i]])
            z = np.append(z, [z[i], z[i], z[i]])
        if ((x[i]==0) & (y[i]!=0)):
            x = np.append(x, [x[i]])
            y = np.append(y, [-y[i]])
            z = np.append(z, [z[i]])
        if ((y[i]==0) & (x[i]!=0)):
            x = np.append(x, [-x[i]])
            y = np.append(y, [-y[i]])
            z = np.append(z, [z[i]])

    for i in range(x.shape[0]):
        if (z[i]!=0):
            x = np.append(x, [x[i]])
            y = np.append(y, [y[i]])
            z = np.append(z, [-z[i]])

    return x, y, z

def one_particle(G):

    return np.array(sqrt(1/G)*np.random.normal( 0.0 , 1.0, size = 1)), np.array(sqrt(1/G)*np.random.normal( 0.0 , 1.0, size = 1)), np.array(sqrt(1/G)*np.random.normal( 0.0 , 1.0, size = 1))

def HCP_old(N):

    x, y, z = np.array([]), np.array([]), np.array([])
    a = 2.
    L = (N * 2) ** (1 / 3.)
    num = int(L)
    # print(num)

    for k in range(-num, num):
        for j in range(-num, num):
            for i in range(-num, num):
                if ((k % 2 == 0) & (j % 2 == 0)):
                    x0 = i * a
                    y0 = j * a
                    z0 = k * a
                if ((k % 2 == 1) & (j % 2 == 0)):
                    x0 = i * a
                    y0 = (j + 0.5) * a
                    z0 = k * a
                if ((k % 2 == 1) & (j % 2 == 1)):
                    x0 = (i + 0.5) * a
                    y0 = (j + 0.5) * a
                    z0 = k * a
                if ((k % 2 == 0) & (j % 2 == 1)):
                    x0 = (i + 0.5) * a
                    y0 = j * a
                    z0 = k * a
                if ((x0 ** 2 + y0 ** 2 + z0 ** 2 <= L ** 2)):
                     x = np.append(x, x0)
                     y = np.append(y, y0)
                     z = np.append(z, z0)


    return x, y, z

def HCP(N):
    x, y, z = np.array([]), np.array([]), np.array([])
    a = 1.
    L = (N*1.5) ** (1 / 3.)
    num = int(L)
    # print(num)

    for k in range(-num, num):
        for j in range(-num, num):
            for i in range(-num, num):
                x0 = 2 * i + ((j + k) % 2)
                x0 = x0 * a
                y0 = sqrt(3) * (j + (k % 2) / 3)
                y0 = y0 * a
                z0 = 2 * sqrt(6) / 3 * k
                z0 = z0 * a
                if ((x0 ** 2 + y0 ** 2 + z0 ** 2 <= L ** 2) & (x.shape[0] < N)):
                    x = np.append(x, x0)
                    y = np.append(y, y0)
                    z = np.append(z, z0)

    return x, y, z

def FCC(N):
    x = np.array([])
    y = np.array([])
    z = np.array([])

    L = (N/2.) ** (1 / 3.)
    num = int(L)

    r = 1

    for i in range(-num, num + 1):
        for j in range(-num, num + 1):
            for k in range(-num, num + 1):
                if ((((i * 2 * r) ** 2 + (j * 2 * r) ** 2 + (k * 2 * r) ** 2) <= L ** 2) & (x.shape[0]<N)):
                    x = np.append(x, i * 2 * r)
                    y = np.append(y, j * 2 * r)
                    z = np.append(z, k * 2 * r)
                if ((((i * 2 * r ) ** 2 + (j * 2 * r + r) ** 2 + (k * 2 * r + r) ** 2) <= L ** 2) & (x.shape[0]<N)):
                    x = np.append(x, i * 2 * r )
                    y = np.append(y, j * 2 * r + r)
                    z = np.append(z, k * 2 * r + r)
                if ((((i * 2 * r + r) ** 2 + (j * 2 * r ) ** 2 + (k * 2 * r + r) ** 2) <= L ** 2) & (x.shape[0]<N)):
                    x = np.append(x, i * 2 * r + r)
                    y = np.append(y, j * 2 * r )
                    z = np.append(z, k * 2 * r + r)
                if ((((i * 2 * r + r) ** 2 + (j * 2 * r + r) ** 2 + (k * 2 * r ) ** 2) <= L ** 2) & (x.shape[0]<N)):
                    x = np.append(x, i * 2 * r + r)
                    y = np.append(y, j * 2 * r + r)
                    z = np.append(z, k * 2 * r )

    return x, y, z

def BCC(N):

    x = np.array([])
    y = np.array([])
    z = np.array([])

    L = (N) ** (1 / 3.)
    num = int(L)

    r = 1

    for i in range(-num, num+1):
        for j in range(-num, num+1):
            for k in range(-num,  num+1):
                if ((((i * 2 * r) ** 2 + (j * 2 * r) ** 2 + (k * 2 * r) ** 2) <= L ** 2)&(x.shape[0]<N)):
                    x = np.append(x, i * 2 * r)
                    y = np.append(y, j * 2 * r)
                    z = np.append(z, k * 2 * r)
                if ((((i * 2 * r + r) ** 2 + (j * 2 * r + r) ** 2 + (k * 2 * r + r) ** 2) <= L ** 2)&(x.shape[0]<N)):
                    x = np.append(x, i * 2 * r + r)
                    y = np.append(y, j * 2 * r + r)
                    z = np.append(z, k * 2 * r + r)

    return x, y, z