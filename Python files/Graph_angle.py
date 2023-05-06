import numpy as np
import os
from matplotlib import pyplot as plt
import numba

@numba.njit
def Six_neighbors_all_shells(N, data_x, data_y, data_z, t, G, n):

	dts = np.zeros((n-1, 6))
	for nn in range(2, n+1, 1):
		mins, means, maxs, stds = np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000)
		for it in range(0, data_x.shape[0] - 2000, 1):
			x0, y0, z0 = data_x[2000 + it, :], data_y[2000 + it, :], data_z[2000 + it, :]
			r = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
			xlist = np.arange(r.shape[0])
			Rm = max(r[r < N**(1./3)*1.1])
			cnt = 0

			xlist = xlist[(r <= nn * Rm * t) & (r >= (nn - 1) * Rm * t)]
			R = np.mean(r[(r <= nn * Rm * t) & (r >= (nn - 1) * Rm * t)])
			if xlist.shape[0] > 6:
				mins_t, means_t, maxs_t, stds_t = np.zeros(xlist.shape[0]), np.zeros(xlist.shape[0]), \
				                                  np.zeros(xlist.shape[0]), np.zeros(xlist.shape[0])
				for i in xlist:
					dr = np.sqrt((x0 - x0[i]) ** 2 + (y0 - y0[i]) ** 2 + (z0 - z0[i]) ** 2)
					dr[i] = 10 ** 10
					n = np.argsort(dr)

					angle = np.zeros(12)
					for j in range(6):
						u = [x0[i] - x0[n[j]], y0[i] - y0[n[j]], z0[i] - z0[n[j]]]
						u_norm = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
						ang = np.zeros(6)
						ang[j] = 10**10
						for k in range(6):
							if (k != j):
								v = [x0[i] - x0[n[k]], y0[i] - y0[n[k]], z0[i] - z0[n[k]]]
								v_norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
								cos = (u[0]*v[0]+u[1]*v[1]+u[2]*v[2]) / (u_norm * v_norm)
								l = min([u_norm, v_norm])

								if np.abs(cos) > 1:
									cos = np.sign(cos)

								sin = np.sin(np.arccos(cos) / 2.) / np.sqrt(1 - (l / 2 / R) ** 2)
								if np.abs(sin) > 1:
									sin = np.sign(sin)
								ang_0 = 2 * np.arcsin(sin)
								ang[k] = ang_0

						angle[j * 2] = np.sort(ang)[0]
						angle[j * 2 + 1] = np.sort(ang)[1]

					mins_t[cnt] = np.nanmin(angle)
					maxs_t[cnt] = np.nanmax(angle)
					means_t[cnt] = np.nanmean(angle)
					stds_t[cnt] = np.nanstd(angle)
					cnt += 1
				mins[it] = np.nanmean(mins_t)
				maxs[it] = np.nanmean(maxs_t)
				means[it] = np.nanmean(means_t)
				stds[it] = np.nanmean(stds_t)
			else:
				mins[it] = np.nan
				maxs[it] = np.nan
				means[it] = np.nan
				stds[it] = np.nan
		dts[nn-2,:] = [G, nn, np.nanmean(means), np.nanmean(mins), np.nanmean(maxs), np.nanmean(stds)]

	return dts
G_date = np.arange(10,210,10)
G_date = np.append(G_date, [250, 500])
N_date = [155, 335, 665]
n_date = [4, 5, 6]
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname('/Users/elizavetashpilko/Work/test/HCP/Angles/'))
for iN, N in enumerate(N_date):
	print(N)
	file = 'Angles_'+str(N)+', six.txt'
	open(file, 'w').close()
	file_save = open(file, 'a')
	for G in G_date:
		print(G)
		if N == 155:
			dir_name = '/Users/elizavetashpilko/Work/test/HCP/N='+str(N)+'/'
		else:
			dir_name = '/Users/elizavetashpilko/Work/test/HCP/N='+str(N)+'/'+str(G)+'/'
		data_x = np.loadtxt(dir_name+'x_'+str(G)+'.txt', dtype='float64')
		data_y = np.loadtxt(dir_name+'y_'+str(G)+'.txt', dtype='float64')
		data_z = np.loadtxt(dir_name+'z_'+str(G)+'.txt', dtype='float64')
		dts = Six_neighbors_all_shells(N, data_x, data_y, data_z, 1./n_date[iN], G, n_date[iN])
		np.savetxt(file_save, dts, newline="\n", fmt='%f')
		# file_save.write("\n")
		# print(dts)