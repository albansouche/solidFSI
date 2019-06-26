import numpy as np
import matplotlib.pyplot as plt

eps_lin, sig_lin = np.loadtxt('results/PW3d/LinearElastic.out')
eps_SVK, sig_SVK = np.loadtxt('results/PW3d/StVenantKirchhoff.out')
eps_neo, sig_neo = np.loadtxt('results/PW3d/neoHookean.out')

plt.figure()
plt.plot(eps_lin, sig_lin, label='Linear Elastic')
plt.plot(eps_SVK, sig_SVK, label='StVenant-Kirchhoff')
#plt.plot(eps_neo, sig_neo, label='neoHookean')
#plt.xlabel('Green-Lagrange Strain $E_{zz}$')
plt.xlabel('Infinitesimal Strain $\\epsilon_{zz}$')
#plt.ylabel('$\\operatorname{tr}(\\sigma)$')
plt.ylabel('$\\sigma_{zz}$')
plt.title('Stress-strain in the point $(0.11, 0.0, 0.0)$')
plt.legend()
plt.savefig('results/PW3d/stress-strain.png')
