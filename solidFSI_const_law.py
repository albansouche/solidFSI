#!/usr/bin/env python

__author__ = "Alban Souche <alban@simula.no>"
__date__ = "2019-13-03"
__copyright__ = "Copyright (C) 2019 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

"""
Pressure wave benchmark 3D
Solid solver for a number of elastic models (linear and non-linear)
from the CBC.twist library.
"""

# IMPORTS ######################################################################
from dolfin import *
import sys
import os
import shutil
import numpy as np
import time as tm
from IPython import embed

# READ USER INPUTS #############################################################
from library.argpar import *
args = parse()

# READ SETUP FILE ##############################################################
exec("from setups.%s.%s import *" % (args.setup, args.setup))
setup = Setup()

# Update variables from commandline
for key, value in list(args.__dict__.items()):
    if value is None:
        args.__dict__.pop(key)
setup.__dict__.update(args.__dict__)  # update if any given argument parameters

# Check if CBC.solve module is in the system path
if setup.CBCsolve_path not in sys.path:
    print("CBC.solve path added to sys.path")
    sys.path.insert(0, setup.CBCsolve_path)
from library.structure_problem import *
from library.fsi_coupling import *

# MESHES #######################################################################
print("Extract fluid and solid submeshes")
subM = FSISubmeshes()
if setup.mesh_split:  # create submeshes from a parent mesh
    setup.get_parent_mesh()
    subM.split(setup.mesh_folder, setup.parent_mesh, setup.dom_f_id, setup.dom_s_id)
    setup.reshape_parent_mesh()
    subM.read(setup.mesh_folder)  # FIXME: this should not be necessary
else:  # read submeshes from files
    subM.read(setup.mesh_folder)

# Function space and functions #################################################
de = VectorElement('CG', subM.mesh_s.ufl_cell(), setup.d_deg)
D = FunctionSpace(subM.mesh_s, de)
Nd = TestFunction(D)
d_ = Function(D)  # sol disp. at t = n
tract_S = Function(D)  # solid traction
V_dum = FunctionSpace(subM.mesh_f, de)  # this is a hack of the FSI solver
d_scalar = FiniteElement('DG', subM.mesh_s.ufl_cell(), setup.d_deg-1)
D_scalar = FunctionSpace(subM.mesh_s, d_scalar)
#d_tensor = TensorElement('CG', subM.mesh_s.ufl_cell(), setup.d_deg)
#D_tensor = FunctionSpace(subM.mesh_s, d_tensor)
eps_scalar = Function(D_scalar)
sig_scalar = Function(D_scalar)

# DOFS mapping #################################################################
print("Extract FSI DOFs")
subM.DOFs_fsi(V_dum, D, setup.fsi_id)

# Boundary conditions ##########################################################
print("Read boundary conditions from setup")
setup.setup_Dirichlet_BCs()

# Solid solver, based on cbc.twist #############################################
print("Prepare structure solver")
_struct = StructureSolver(setup, subM, tract_S)
################################################################################

# Traction vector as Neumann bds
de1 = VectorElement('CG', subM.mesh_s.ufl_cell(), 1)
D1 = FunctionSpace(subM.mesh_s, de1)
Nd1 = TestFunction(D1)
ds_s = Measure("ds", domain=subM.mesh_s, subdomain_data=subM.boundaries_s)
n_s = FacetNormal(subM.mesh_s)
T = Function(D1)
Tn = Function(D1)
Tn.vector()[:] = assemble(inner(Constant((1., 1., 1.)), Nd1) * ds_s(subdomain_id=setup.fsi_id))
Tnvec = Tn.vector().get_local()
np.place(Tnvec, Tnvec < 1e-9, 1.0)
Tn.vector()[:] = Tnvec

# Numpy array for saving stress-strain
eps_array = [0.0]
sig_array = [0.0]
point = (0.011, 0.0, 0.0)

# TIME LOOP START ##############################################################
t = 0.0
counter = 1
tic = tm.clock()
print("ENTERING TIME LOOP")
while t < setup.T:

    # time update
    print("Solving for timestep %g of %g   " % (t, setup.T), end='\r')
    #print("\n Solving for timestep %g" % t)
    t += setup.dt
    setup.p_exp.t = t

    if setup.p_exp.value_shape() == (1,):
        T.vector()[:] = assemble(inner(setup.p_exp[0]*n_s, Nd1) * ds_s(subdomain_id=setup.fsi_id))
    else:
        T.vector()[:] = assemble(inner(setup.p_exp*n_s, Nd1) * ds_s(subdomain_id=setup.fsi_id))
    T.vector()[:] = - np.divide(T.vector().get_local(), Tn.vector().get_local())
    TT = project(T, D)
    tract_S.vector()[subM.fsi_dofs_s] = TT.vector()[subM.fsi_dofs_s]

    # Solve for solid deformation
    if setup.solid_solver_scheme == 'HHT':
        assign(d_, _struct.step(setup.dt))  # pass displacements
    elif setup.solid_solver_scheme == 'CG1':
        assign(d_, _struct.step(setup.dt)[0])  # pass displacements / not velocities

    # save data -------------------------------
    if counter % setup.save_step == 0:

        # Stress
        sig = _struct.material.SecondPiolaKirchhoffStress(d_)

        # Strain
        eps = _struct.material.epsilon

        # Scalar strain-like variable to export
        #assign(eps_scalar, project(tr(eps), D_scalar))
        assign(eps_scalar, project(eps[2, 2], D_scalar))
        #disp = d_(point); eps_scalar = np.sqrt(np.sum(disp*disp))
        #assign(eps_scalar, project(PrincipalStretches(d_)[0], D_scalar))

        # Scalar stress-like variable to export
        assign(sig_scalar, project(tr(sig), D_scalar))
        #assign(sig_scalar, project(sig[2, 2], D_scalar))
        #assign(sig_scalar, project(vonMises(sig), D_scalar))

        # Store stress-strain
        eps_array += [eps_scalar(point)]
        sig_array += [sig_scalar(point)]

    # update solution vectors -----------------
    _struct.update()

    # update time loop counter
    counter += 1
################################################################################

# Save stress-strain relation to file
np.savetxt(setup.save_path+'/'+setup.solid_solver_model+'.out', (eps_array, sig_array))
