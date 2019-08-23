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

folder = 'results/tube/PW3d/static'#+'_prestress'
point_name = 'midpoint'

models = ['LinearElastic', 'StVenantKirchhoff']# + ['LinearElastic_prestress', 'StVenantKirchhoff_prestress']
N_models = len(models)

xlabel = 'displacement'
xlabel = 'strain'

ylabel = 'traction'
#ylabel = 'stress'

R_inner = 0.002
R_outer = 0.0023
R_midpoint = 0.5*(R_inner+R_outer)

###############################################################################

mmHg = 133.322387415
#mmHg = 1

X = [0] * N_models
Y = [0] * N_models
relation_point = [0] * N_models
relation = [0] * N_models
err = [0] * (N_models-1)
relerr = [0] * (N_models-1)

# Extract data
for i in range(N_models):
    X[i] = np.load('{}/{}/{}_{}.npy'.format(folder, models[i], point_name, xlabel))#[1*(i<2):]
    if xlabel == 'displacement':
        d_norm = np.array([npl.norm(X[i][j]) for j in range(np.shape(X[i])[0])])
        X[i] = d_norm/R_midpoint
    Y[i] = np.load('{}/{}/{}_{}.npy'.format(folder, models[i], point_name, ylabel))#[1*(i<2):]
    Y[i] /= mmHg
    if ylabel == 'stress':  # tr(sigma)
        Y[i] /= 3  # p = 1/3 tr(sigma)
    relation_point[i] = interp1d(X[i], Y[i])
    relation[i] = np.vectorize(extrap1d(relation_point[i]))

if 0:#xlabel == 'displacement':
    shift = X[2][0] - X[0][0]
    for i in range(2, N_models):
        X[i] -= shift

# Difference between linear model (0) and the others (1->N_models)
for i in range(1, N_models):
    err[i-1] = np.abs(relation[i](X[0])-Y[0])
    relerr[i-1] = err[i-1][1:]/Y[0][1:]



#xlabel = 'Infinitesimal volumetric Strain $\\operatorname{tr}(\\epsilon)$'
#xlabel = 'Infinitesimal Strain $\\epsilon_{zz}$'
#xlabel = 'Green-Lagrange Strain $E_{zz}$'
#ylabel = '$\\operatorname{tr}(\\sigma)/3$'
#ylabel = 'Traction pressure'
#ylabel = '$\\sigma_{zz}$'
#ylabel = '$Von Mises stress'

# Plot stress-strain relation for all models
plt.figure()
for i in range(N_models):
    #if i >= 1: plt.plot(X[0], relation[i](X[0]), label=models[i]+' interpolated')
    plt.plot(X[i], Y[i], label=models[i])
#plt.scatter(0,0)
plt.xlabel(xlabel)
plt.ylabel(ylabel+' [mmHg]')
plt.title('{}{} in midpoint'.format(xlabel, ylabel))
plt.legend()
plt.savefig(folder+'/{}-{}.png'.format(xlabel, ylabel))


# Plot relative difference between linear model and the others
plt.figure()
for i in range(1, N_models):
    plt.plot(X[0][1:], relerr[i-1], label=models[i])
plt.xlabel(xlabel)
plt.ylabel('Relative error of '+ylabel)
plt.title('Relative difference: $\\frac{y_{hyper}-y_{lin}}{y_{lin}}$')
plt.legend()
plt.savefig(folder+'/error.png')
