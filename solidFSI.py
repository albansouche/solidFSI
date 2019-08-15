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

# FUNCTIONS ###################################################################
def assign_characteristic(space, scalar_characteristic, obs, quant):
    assign(scalar_characteristic, project(obs(quant), space))

def choose_obs(obs_string, dim=3):
    for i in range(dim):
        for j in range(dim):
            if obs_string == "[{},{}]".format(i,j):
                return lambda x : x[i,j]
    if obs_string == "tr":
        return tr
    elif obs_string == "vonMises":
        return lambda x : vonMises(x, dim)

def choose_quant(quant, struct=None):
    if quant == "strain":
        return InfinitesimalStrain
    elif quant == "stress":
        return struct.material.SecondPiolaKirchhoffStress

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
de = VectorElement("CG", subM.mesh_s.ufl_cell(), setup.d_deg)
D = FunctionSpace(subM.mesh_s, de)
Nd = TestFunction(D)
d_ = Function(D)  # sol disp. at t = n
tract_S = Function(D)  # solid traction
V_dum = FunctionSpace(subM.mesh_f, de)  # this is a hack of the FSI solver
d_scalar = FiniteElement("CG", subM.mesh_s.ufl_cell(), setup.d_deg)
D_scalar = FunctionSpace(subM.mesh_s, d_scalar)

scalar_characteristic = Function(D_scalar)

# DOFS mapping #################################################################
print("Extract FSI DOFs")
subM.DOFs_fsi(V_dum, D, setup.fsi_id)

# Boundary conditions ##########################################################
print("Read boundary conditions from setup")
setup.setup_Dirichlet_BCs()

# Solid solver, based on cbc.twist #############################################
print("Prepare structure solver")
_struct = StructureSolver(setup, subM, tract_S)

# Initialization of data file ##################################################
print("Prepare output files")
if MPI.rank(mpi_comm_world()) == 0:
    if os.path.isdir(setup.save_path):
        shutil.rmtree(setup.save_path)
    os.makedirs(setup.save_path)
MPI.barrier(mpi_comm_world())
sol_d_file = XDMFFile(mpi_comm_world(), "{}/disp_{}.xdmf".format(setup.save_path, setup.extension))
sol_d_file.parameters["flush_output"] = True
sol_d_file.parameters["rewrite_function_mesh"] = False
sol_d_file.parameters["functions_share_mesh"] = True
sol_char_file = XDMFFile(mpi_comm_world(), "{}/quant_{}.xdmf".format(setup.save_path, setup.extension))
sol_char_file.parameters["flush_output"] = True
sol_char_file.parameters["rewrite_function_mesh"] = False
sol_char_file.parameters["functions_share_mesh"] = True

################################################################################

# If no pressure expression is defined, assume it is zero
if setup.p_exp == []:
    setup.p_exp = Expression("0.0", degree=0, t=0)

# Traction vector as Neumann bds
de1 = VectorElement("CG", subM.mesh_s.ufl_cell(), 1)
D1 = FunctionSpace(subM.mesh_s, de1)
Nd1 = TestFunction(D1)
ds_s = Measure("ds", domain=subM.mesh_s, subdomain_data=subM.boundaries_s)
n_s = FacetNormal(subM.mesh_s)
T = Function(D1)
Tn = Function(D1)
Tn.vector()[:] = assemble(inner(Constant((1.0,)*subM.mesh_s.geometry().dim()), Nd1) * ds_s(subdomain_id=setup.fsi_id))
Tnvec = Tn.vector().get_local()
np.place(Tnvec, Tnvec < 1e-9, 1.0)
Tn.vector()[:] = Tnvec

# Write first value, before first time step
t = 0.0
counter = 1
tic = tm.clock()


if setup.u0 == []:
    assign(d_, project(Constant((0.0,)*subM.mesh_s.geometry().dim()), D))
else:
    assign(d_, project(setup.u0, D))

# Define functions for computing scalar characteristic
obs = choose_obs(setup.obs, subM.mesh_s.geometry().dim())
quant = choose_quant(setup.quant, _struct)
assign_characteristic(D_scalar, scalar_characteristic, obs, quant(d_))

# Save first timestep
time_list = []
time_list.append(t)
sol_d_file.write(d_, t)
sol_char_file.write(scalar_characteristic, t)

# Observe displacement at observation points
for obs_point in setup.obs_points:
    obs_point.append(d_(obs_point.point))

# TIME LOOP START ##############################################################
print("ENTERING TIME LOOP")
while t < setup.T:

    # time update
    print("Solving for timestep %g of %g   " % (t, setup.T), end="\r")
    #print("\n Solving for timestep %g" % t)
    t += setup.dt
    setup.p_exp.t = t

    if setup.p_exp.value_shape() == (1,):  # Python subclass of Dolfin Expression
        T.vector()[:] = assemble(inner(setup.p_exp[0]*n_s, Nd1) * ds_s(subdomain_id=setup.fsi_id))
    else:  # Direct Expression object
        T.vector()[:] = assemble(inner(setup.p_exp*n_s, Nd1) * ds_s(subdomain_id=setup.fsi_id))
    T.vector()[:] = - np.divide(T.vector().get_local(), Tn.vector().get_local())
    TT = project(T, D)
    tract_S.vector()[subM.fsi_dofs_s] = TT.vector()[subM.fsi_dofs_s]

    # Solve for solid deformation
    if setup.solid_solver_scheme == "HHT":
        assign(d_, _struct.step(setup.dt))  # pass displacements
    elif setup.solid_solver_scheme == "CG1":
        assign(d_, _struct.step(setup.dt)[0])  # pass displacements / not velocities

    # save data -------------------------------
    if counter % setup.save_step == 0:

        # Compute scalar characteristic for exportation
        assign_characteristic(D_scalar, scalar_characteristic, obs, quant(d_))

        sol_d_file.write(d_, t)
        sol_char_file.write(scalar_characteristic, t)

        time_list.append(t)
        for obs_point in setup.obs_points:
            obs_point.append(d_(obs_point.point))

    # update solution vectors -----------------
    _struct.update()

    # update time loop counter
    counter += 1
################################################################################

np.save(setup.save_path + "/time.npy", time_list)
for obs_point in setup.obs_points:
    obs_point.save(setup.save_path)
