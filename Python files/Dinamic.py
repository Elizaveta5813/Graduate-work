import numpy as np
import numba
from time import time
import matplotlib.pyplot as plt
import os
import Lattice
import File_save
from math import sqrt

@numba.njit
def force(x, y, z, N, d, T): #вычисление матрицы сил f/m по 1-ой координате

	f_x, f_y, f_z = np.zeros(N), np.zeros(N), np.zeros(N)
	for i in range(N):
		r0 = sqrt(x[i]**2 + y[i]**2 + z[i]**2)
		# f_x[i] = -x[i]
		# f_y[i] = -y[i]
		# f_z[i] = -z[i]
		if ((r0 > N**(1/3.))):
			f_x[i] = -x[i] * N / (r0 ** 3)
			f_y[i] = -y[i] * N / (r0 ** 3)
			f_z[i] = -z[i] * N / (r0 ** 3)
		else:
			f_x[i] = -x[i]
			f_y[i] = -y[i]
			f_z[i] = -z[i]

	for i in range(0, N):
		for j in range(0, N):
			if (j != i):
				dx = x[i] - x[j]
				dy = y[i] - y[j]
				dz = z[i] - z[j]

				xi_ij = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

				f_x[i] = f_x[i] + dx / (xi_ij ** 3)
				f_y[i] = f_y[i] + dy / (xi_ij ** 3)
				f_z[i] = f_z[i] + dz / (xi_ij ** 3)

	fx_0_norm, fy_0_norm, fz_0_norm = np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64), np.zeros(N,
	                                                                                                         dtype=np.float64)

	fx_0_norm[0:int(N / 2)], fy_0_norm[0:int(N / 2)], fz_0_norm[0:int(N / 2)] = \
		d * np.random.normal(0, sigma, size=int(N / 2)), d * np.random.normal(0, sigma,
		                                                                      size=int(N / 2)), d * np.random.normal(0,
		                                                                                                             sigma,
		                                                                                                             size=int(
			                                                                                                             N / 2))
	fx_0_norm[int(N / 2):int(N / 2) * 2], fy_0_norm[int(N / 2):int(N / 2) * 2], fz_0_norm[int(N / 2):int(N / 2) * 2] = \
		-fx_0_norm[0:int(N / 2)], -fy_0_norm[0:int(N / 2)], -fz_0_norm[0:int(N / 2)]
	np.random.shuffle(fx_0_norm)
	np.random.shuffle(fy_0_norm)
	np.random.shuffle(fz_0_norm)
	if (T < 500000):
		f_x = f_x + fx_0_norm
		f_y = f_y + fy_0_norm
		f_z = f_z + fz_0_norm
		return f_x, f_y, f_z
	else:
		return f_x, f_y, f_z

@numba.njit
def distribution(fx_0, fy_0, fz_0, x, y, z, V_x, V_y, V_z, N, d, T): #вычисление новой координаты и скорости

	if (T < 500000):
		coordinates_x = x + h * V_x + (fx_0 - gamma * V_x) * (h * h) / 2.
		coordinates_y = y + h * V_y + (fy_0 - gamma * V_y) * (h * h) / 2.
		coordinates_z = z + h * V_z + (fz_0 - gamma * V_z) * (h * h) / 2.

		fx, fy, fz = force(coordinates_x, coordinates_y, coordinates_z, N, d, T)

		velocity_x = (V_x + h * (fx_0 - gamma * V_x + fx) / 2.) / (1 + gamma * h / 2.)
		velocity_y = (V_y + h * (fy_0 - gamma * V_y + fy) / 2.) / (1 + gamma * h / 2.)
		velocity_z = (V_z + h * (fz_0 - gamma * V_z + fz) / 2.) / (1 + gamma * h / 2.)
	else:
		coordinates_x = x + h * V_x + (fx_0 ) * (h * h) / 2.
		coordinates_y = y + h * V_y + (fy_0 ) * (h * h) / 2.
		coordinates_z = z + h * V_z + (fz_0 ) * (h * h) / 2.

		fx, fy, fz = force(coordinates_x, coordinates_y, coordinates_z, N, d, T)

		velocity_x = (V_x + h * (fx_0 + fx) / 2.)
		velocity_y = (V_y + h * (fy_0  + fy) / 2.)
		velocity_z = (V_z + h * (fz_0  + fz) / 2.)

	return coordinates_x, coordinates_y, coordinates_z, velocity_x, velocity_y, velocity_z, fx, fy, fz

h = 0.01
H_t = 50000
sigma = 1.
gamma = 0.09
G_date = np.arange(100,210, 10)
G_date = np.append(G_date, np.array([250]))
N = 1075
n_date = np.arange(31, 33, dtype='int')

for n in n_date:
	print(n)

	for G in G_date:

		d = sqrt(2 * gamma/ (h * G))
		print(G, d)

		P = np.zeros(H_t)
		E = np.zeros(H_t)
		K = np.zeros(H_t)
		t = np.arange(H_t)

		x = np.array([])
		y = np.array([])
		z = np.array([])

		num = G
		dir_name = "/Users/elizavetashpilko/Desktop/Work/test/HCP/N="+str(N)+'/'+str(n)+'/'
		plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

		x, y, z = Lattice.HCP(N)
		N = x.shape[0]
		alpha = max(np.sqrt(x**2 + y**2 + z**2)) / (N**(1/3.))
		x, y, z = x  / alpha, y / alpha, z / alpha
		num_0 = np.argmin(x**2 + y**2 + z**2)
		V_x, V_y, V_z = np.zeros(N), np.zeros(N), np.zeros(N)
		none = np.array([])

		if (N%2 == 1):
			d = d * sqrt(N / (N - 1))

		f_x, f_y, f_z = File_save.File_Coordinate(num)
		# v_x_, v_y_, v_z_ = File_save.File_Velocity(num)
		# file_x, file_y, file_z = File_save.File_Force(num)

		V_x = np.random.normal(0, sigma, N) * sqrt(1./G/N)
		V_y = np.random.normal(0, sigma, N) * sqrt(1./G/N)
		V_z = np.random.normal(0, sigma, N) * sqrt(1./G/N)

		fx_0, fy_0, fz_0 = force(x, y, z, N, d, 10)

		start = time()

		for i in range(H_t):
			x, y, z, V_x, V_y, V_z, fx_0, fy_0, fz_0 = distribution(fx_0, fy_0, fz_0, x, y, z, V_x, V_y, V_z, N, d, i)
			K[i] = np.mean(V_x**2 + V_y ** 2 + V_z ** 2)/3

			if (i % 100 == 0):
				File_save.Coordinate_Save(f_x, f_y, f_z, x, y, z)
				# File_save.Velocity_Save(v_x_, v_y_, v_z_, V_x, V_y, V_z)
				# File_save.Force_Save(file_x, file_y, file_z, fx_0-gamma*V_x, fy_0-gamma*V_y, fz_0-gamma*V_z)

		end = time()
		print(end - start, G, 1./np.mean(K[int(H_t/2.):]) , np.abs(G - 1/np.mean(K[int(H_t/2.):]))/G*100)