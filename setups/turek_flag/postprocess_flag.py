import numpy as np
import matplotlib.pyplot as plt

folder = 'results/turek_flag/dynamic'


t = np.load(folder + '/StVenantKirchhoff/time.npy')
disp_lin = np.load(folder + '/LinearElastic/A.npy')
disp_svk = np.load(folder + '/StVenantKirchhoff/A.npy')

a = 0  # 0 <= a <= b <= Nt
b = None
L = 0.35

x = ['x', 'y']

for i in [0, 1]:
    mean_svk = np.mean(disp_svk[:,i])
    amp_svk = np.sqrt(np.mean((disp_svk[i]-mean_svk)**2))
    
    plt.figure()
    plt.plot(t[a:b], disp_lin[a:b,i]/L, label='Linear')
    plt.plot(t[a:b], disp_svk[a:b,i]/L, label='StVenantKirchhoff')
    plt.plot(t[a:b], mean_svk/L+0.0*t[a:b], label='Mean StVenantKirchhoff')
    plt.title('{}-deformation of point A(t)'.format(x[i]))
    plt.legend()
    plt.savefig(folder + '/disp_{}.png'.format(x[i]))
