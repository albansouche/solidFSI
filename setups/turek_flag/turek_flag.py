#!/usr/bin/env python

"""
Setup for a 2d pressure wave problem in a pipe
Comparing different elastic constitutive law on finite deformation amplitude
"""

from dolfin import *
import numpy as np
import os
from setups import *
from IPython import embed  # for debugging
from mshr import *


class Setup(Setup_base):

    def __init__(self):

        # print setup
        print("Loading Turek flag setup")

        # FE order
        self.d_deg = 1  # Deformation degree (solid)

        # Physical properties #################################################

        # Time
        self.T = 0.5  # End time s.
        self.dt = 0.001  # Time step s.
        t = 0.0

        # Solid prop.
        self.rho_s = 1.0E3  # density
        self.nu_s = 0.4  # Poisson ratio
        self.mu_s = 0.5E6 #2.0E6 #  # Shear modulus or 2nd Lame Coef.
        self.lamda_s = self.nu_s*2.*self.mu_s/(1. - 2.*self.nu_s)  # Young's modulus
        self.young = 2*self.mu_s*(1+self.nu_s)

        # Pre-stress
        self.pre_stress_tract = 1.0E+4  # Traction to induce horizontal pre-stress
        self.pre_stress = Constant(((self.pre_stress_tract, 0.0),
                                    (0.0, 0.0)))
        #self.pre_stress = []

        # FSI pressure expression
        self.p_exp = Expression('0.0', degree=0, t=t)
        #self.p_exp = Expression('0.6 - tol < x[0] ? value : 0.0', degree=2, tol=1.0E-14, value=-0.0E3, t=t)  # Traction

        # Body forces
        self.grav = 2.0
        deform = 0.001  # Deformation (u(L)/L) with static horizontal body force linear elasticity
        self.body_force = Constant((0.0, -self.rho_s*self.grav))
        #self.body_force = Constant((deform*2*self.young/0.35, 0.0))
        #self.body_force = Constant((deform*2*self.young/0.35, -self.rho_s*self.grav))
        #self.body_force = []

        # Initial conditions
        A = self.pre_stress_tract*(self.lamda_s+2*self.mu_s)/4/self.mu_s/(self.lamda_s+self.mu_s)
        B = -self.pre_stress_tract*self.lamda_s/4/self.mu_s/(self.lamda_s+self.mu_s)
        self.u0 = Expression(('A*(x[0]-0.25)', 'B*(x[1]-0.2)'), degree=2, A=A, B=B)  # Initial displacement
        #self.u0 = Expression(('0.0', '-pow(x[0]-0.25, 2.0)'), degree=2)  # Initial displacement
        #self.u0 = []  # Initial displacement
        #self.v0 = Expression(('0.0', 'x[0]-0.25'), degree=2) # Initial velocity
        self.v0 = []  # Initial velocity

        # Solver properties ###################################################

        # Dynamic or stationnary
        self.is_dynamic = True

        # path to CBC.solve
        self.CBCsolve_path = "library/external/cbc.solve"

        # saving data
        self.save_path = "results/turek_flag"
        self.save_step = 1  # saving solution every "n" steps

        # parent mesh info. !! values can be redefined in get_parent_mesh() !!
        self.mesh_folder = "setups/turek_flag/mesh"
        self.mesh_split = True  # spliting the mesh True or False. Submesh are systematically saved in the folder
        self.parent_mesh = []  # if parent mesh provided by a xml or xml.gz file
        self.dom_f_id = 0  # value of the cell id for the fluid domain in the parent mesh file
        self.dom_s_id = 1  # value of the cell id for the solid domain in the parent mesh file
        self.inlet_id = 0  # fluid inlet id
        self.outlet_id = 1  # fluid outlet id
        self.noslip_f_id = 2 # fluid wall id
        self.noslip_s_id = 3  # solid clamp id
        self.fsi_id = 4  # IMPORTANT VARIABLE value of the FSI facet id in the parent mesh file

        # Material constitutive law
        self.solid_solver_model = "StVenantKirchhoff"

        # Shortcut for puttiing StVenantKirchhoff results in separate folder from LinearElastic
        if self.solid_solver_model == "StVenantKirchhoff": self.save_path += "_svk"

        # solvers
        self.solid_solver_scheme = "HHT"  # "CG1" or "HHT"

        # set compiler arguments
        parameters["form_compiler"]["quadrature_degree"] = 6
        os.environ['OMP_NUM_THREADS'] = '4'

        # set log outputs from dolfin
        set_log_level(40)  # 0 to 100 / more info >> lower value

    ############################################################################

    def get_parent_mesh(self):

        # read mesh from mesh file
        self.parent_mesh = Mesh(self.mesh_folder + "/mesh.xml")
        subdomains = MeshFunction("size_t", self.parent_mesh, self.mesh_folder + "/subdomains.xml")
        boundaries = MeshFunction("size_t", self.parent_mesh, self.mesh_folder + "/boundaries.xml")

        # read domains and boundaries from mesh file
        #self.parent_domains = MeshFunction("size_t", self.parent_mesh, 2, self.parent_mesh.domains())
        #self.parent_domains.set_values(self.parent_domains.array()+1)  # in order to have fluid==1 and solid==2

        # send back the modification to the parent mesh file.
        for i in range(len(subdomains.array())):
            val = subdomains.array()[i]
            dic = (i, val)
            self.parent_mesh.domains().set_marker(dic, 2)

        for (i, val) in enumerate(boundaries.array()):
            dic = (i, val)
            self.parent_mesh.domains().set_marker(dic, 1)

        self.parent_bds = MeshFunction("size_t", self.parent_mesh, 1, self.parent_mesh.domains())


        return

    ############################################################################

    def reshape_parent_mesh(self):

        return

    ############################################################################

    def setup_Dirichlet_BCs(self):

        noslip = Constant((0.0, 0.0))

        # Fluid velocity BCs

        # Dirichlet conditions for the fluid pressure problem

        # Mesh problem bcs

        # Dirichlet conditions for the solid problem
        self.bcs_s_vals = [noslip]
        self.bcs_s_ids = [self.noslip_s_id]
        self.bcs_s_fct_sps = ['xyz']  # 'x', 'y', 'z' (freeslip) or 'xyz' (noslip)

        return

    ############################################################################

    def post_process(self, t,  **namespace):

        #self.p_time_exp.t = t

        pass
