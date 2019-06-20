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
sol_d_file = XDMFFile(mpi_comm_world(), setup.save_path + "/solid_def.xdmf")
sol_d_file.parameters["flush_output"] = True
sol_d_file.parameters["rewrite_function_mesh"] = False
sol_d_file.parameters["functions_share_mesh"] = True
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

# TIME LOOP START ##############################################################
t = 0.0
counter = 1
tic = tm.clock()
print("ENTERING TIME LOOP")
while t < setup.T:

    # time update
    print("\n Solving for timestep %g" % t)
    t += setup.dt
    try:
        setup.p_time_exp.t = t  # FIXME : Need improvement, update time in boundary conditions expression
    except:
        pass

    # TODO: Make a pressure expression instead of using scalar*t
    T.vector()[:] = assemble(inner(Constant(1E4*t)*n_s, Nd1) * ds_s(subdomain_id=setup.fsi_id))
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
        sol_d_file.write(d_, t)

    # update solution vectors -----------------
    _struct.update()

    # update time loop counter
    counter += 1
################################################################################
