import numpy as np
import matplotlib.pyplot as plt

folder = 'results/turek_flag'

t = np.load(folder + '_svk/time.npy')
disp_lin = np.load(folder + '/disp.npy')
disp_svk = np.load(folder + '_svk/disp.npy')

a = 0  # 0 <= a <= b <= Nt
b = None
L = 1#0.35

for i in [0, 1]:
    mean_svk = np.mean(disp_svk[:,i])
    amp_svk = np.sqrt(np.mean((disp_svk[i]-mean_svk)**2))
    max_svk = max(disp_svk[:,i])
    min_svk = min(disp_svk[:,i])


    print('u_{} = {} +- {}'.format(i, mean_svk, amp_svk))
    print('u_{} = {} +- {}'.format(i, 0.5*(max_svk+min_svk), 0.5*(max_svk-min_svk)))


    plt.figure()
    plt.plot(t[a:b], disp_lin[a:b,i]/L, label='Linear')
    plt.plot(t[a:b], disp_svk[a:b,i]/L, label='StVenantKirchhoff')
    plt.plot(t[a:b], mean_svk+0.0*t[a:b], label='Mean StVenantKirchhoff')
    plt.plot(t[a:b], mean_svk+amp_svk+0.0*t[a:b])
    plt.plot(t[a:b], mean_svk-amp_svk+0.0*t[a:b])

    plt.legend()
    plt.savefig(folder + '/disp_{}.png'.format(i))
