import numpy as np
import numpy.linalg as npl
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# Inter- and extrapolator
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

### PARAMETERS ################################################################

folder = 'results/tube/PW3d/static'
point_name = "midpoint"

models = ['LinearElastic', 'StVenantKirchhoff']
N_models = len(models)

#xlabel = 'Infinitesimal volumetric Strain $\\operatorname{tr}(\\epsilon)$'
xlabel = 'Infinitesimal Strain $\\epsilon_{zz}$'
#xlabel = 'Green-Lagrange Strain $E_{zz}$'
#xlabel = 'Displacement'

#ylabel = '$\\operatorname{tr}(\\sigma)/3$'
ylabel = 'Traction pressure'
#ylabel = '$\\sigma_{zz}$'
#ylabel = '$Von Mises stress'

ylabel += ' [mmHg]'

R_inner = 0.002
R_outer = 0.0023
R_midpoint = 0.5*(R_inner+R_outer)

###############################################################################

mmHg = 133.322387415
#mmHg = 1

eps = [0] * N_models
sig = [0] * N_models
relation_point = [0] * N_models
relation = [0] * N_models
err = [0] * (N_models-1)
relerr = [0] * (N_models-1)

# Extract data
for i in range(N_models):
    d = np.load("{}/{}/{}_displacement.npy".format(folder, models[i], point_name))
    d_norm = np.array([npl.norm(d[j]) for j in range(np.shape(d)[0])])
    print(d_norm)
    import sys; sys.exit()
    eps[i] = d_norm/R_midpoint
    #eps[i] = np.load("{}/{}/{}_strain.npy".format(folder, models[i], point_name))
    #sig[i] = np.load("{}/{}/{}_stress.npy".format(folder, models[i], point_name))/3/mmHg
    sig[i] = np.load("{}/{}/{}_traction.npy".format(folder, models[i], point_name))/mmHg
    relation_point[i] = interp1d(eps[i], sig[i])
    relation[i] = np.vectorize(extrap1d(relation_point[i]))



# Difference between linear model (0) and the others (1->N_models)
for i in range(1, N_models):
    err[i-1] = np.abs(relation[i](eps[0])-sig[0])
    relerr[i-1] = err[i-1][1:]/sig[0][1:]

# Plot stress-strain relation for all models
plt.figure()
for i in range(N_models):
    #if i >= 1: plt.plot(eps[0], relation[i](eps[0]), label=models[i]+' interpolated')
    plt.plot(eps[i], sig[i], label=models[i])
plt.scatter(0,0)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title('Stress-strain in midpoint')
plt.legend()
plt.savefig(folder+'/stress-strain.png')


# Plot relative difference between linear model and the others
plt.figure()
for i in range(1, N_models):
    plt.plot(eps[0][1:], relerr[i-1], label=models[i])
plt.xlabel(xlabel)
plt.ylabel('Relative error of '+ylabel)
plt.title('Relative difference: $\\frac{y_{hyper}-y_{lin}}{y_{lin}}$')
plt.legend()
plt.savefig(folder+'/error.png')
