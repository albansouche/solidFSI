import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    return pointwise

folder = 'results/law/'

eps_lin, sig_lin = np.loadtxt(folder+'LinearElastic.out')
eps_SVK, sig_SVK = np.loadtxt(folder+'StVenantKirchhoff.out')
#eps_neo, sig_neo = np.loadtxt(folder+'neoHookean.out')


SVKint = interp1d(eps_SVK, sig_SVK)
SVK = np.vectorize(extrap1d(SVKint))
err_SVK = np.abs(SVK(eps_lin)-sig_lin)
relerr_SVK = err_SVK[1:]/sig_lin[1:]

"""
neoint = interp1d(eps_neo, sig_neo)
neo = np.vectorize(extrap1d(neoint))
err_neo = np.abs(neo(eps_lin)-sig_lin)
relerr_neo = err_neo[1:]/sig_lin[1:]
"""

plt.figure(figsize=[15, 11])
plt.plot(eps_lin, sig_lin, label='Linear Elastic')
#plt.plot(eps_lin, neo(eps_lin), label='SVK interp')
plt.plot(eps_SVK, sig_SVK, label='StVenant-Kirchhoff')
#plt.plot(eps_neo, sig_neo, label='Neo-Hookean')
#plt.xlabel('Green-Lagrange Strain $E_{zz}$')
#plt.xlabel('Infinitesimal Strain $\\epsilon_{zz}$')
plt.xlabel('Norm of displacment')
plt.ylabel('$\\operatorname{tr}(\\sigma)$')
#plt.ylabel('$\\sigma_{zz}$')
plt.title('Stress-strain in the point $(0.11, 0.0, 0.0)$')
plt.legend()
plt.savefig(folder+'stress-strain.png')




plt.figure(figsize=[15, 11])
plt.plot(eps_lin[1:], relerr_SVK, label='StVenantKirchhoff')
#plt.plot(eps_lin[1:], relerr_neo, label='Neo-Hookean')
#plt.xlabel('Infinitesimal Strain $\\epsilon_{zz}$')
plt.xlabel('Norm of displacment')
plt.ylabel('$\\operatorname{tr}(\\sigma)$')
#plt.ylabel('$rel error \\sigma_{zz}$')
plt.title('Error: $\\frac{hyper-lin}{lin}$')
#plt.legend()
plt.savefig(folder+'error.png')
