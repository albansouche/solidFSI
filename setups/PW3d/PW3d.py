#!/usr/bin/env python

"""
Setup for a 3d pressure wave problem in a pipe
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
        print("Loading PW3d setup")

        # Initiate Setup_base, setting all unspecified arguments to []
        super().__init__()


        # SOLVER PROPERTIES ###################################################

        # FE order
        self.d_deg = 1  # Deformation degree (solid)

        # Dynamic or stationnary
        self.is_dynamic = True

        # Material constitutive law
        self.solid_solver_model = "StVenantKirchhoff"

        # Solvers
        self.solid_solver_scheme = "HHT"  # "HHT" or "CG1"

        # Set compiler arguments
        parameters["form_compiler"]["quadrature_degree"] = 6
        os.environ["OMP_NUM_THREADS"] = "8"#"4"

        # Set log outputs from dolfin
        set_log_level(90)  # 0 to 100 / more info >> lower value


        # PHYSICAL PROPERTIES #################################################

        # Time
        self.T = 10  # End time s.
        self.dt = 0.1  # Time step s.

        # Solid properties
        self.rho_s = 1.0E3  # density
        self.nu_s = 0.45  # Poisson ratio
        self.mu_s = 345E3  # Shear modulus or 2nd Lame Coef.
        self.lamda_s = self.nu_s*2.*self.mu_s/(1. - 2.*self.nu_s)  # Young's modulus

        # Pressure
        p_max = 5.0E+4  # Max inlet pressure Pa
        p_wave = "p_max*exp(-0.5*pow((x[1]+0.5*L+4.0*s-a*t)/s, 2.0))"
        p_inlet = "p_max*0.5*(1.0+sin(2.0*pi*f*t))*exp(-0.5*pow((x[1]+0.5*L)/s, 2.0))"
        p_exp = "p_max*t/T_max"
        self.p_exp = Expression(p_wave, degree=2, p_max=p_max, f=10, L=0.1, a=0.02, s=0.003, t=0)

        # Body forces
        pass

        # Prestress
        pass

        # Initial conditions
        pass


        # OBSERVATION PARAMETERS ##############################################

        # Observe strain or stress ("strain", "stress")
        self.quant = "stress"

        # Operator on observed quantity (scalar) "[i,j]", "tr" or "vonMises"
        self.obs = "vonMises"

        self.obs_points.append(obs_point("midpoint", Point(0.011, 0, 0)))


        # DATA PARAMETERS #####################################################

        # Path to CBC.solve
        self.CBCsolve_path = "library/external/cbc.solve"

        # Saving data
        self.extension = self.solid_solver_model
        self.save_path = "results/tube/PW3d/{}".format(self.extension)
        self.save_step = 1  # saving solution every "n" steps

        # Parent mesh info. !! values can be redefined in get_parent_mesh() !!
        self.mesh_folder = "setups/PW3d/mesh"
        self.mesh_split = True  # spliting the mesh True or False. Submesh are systematically saved in the folder
        self.parent_mesh = []  # if parent mesh provided by a xml or xml.gz file
        self.dom_f_id = 1  # value of the cell id for the fluid domain in the parent mesh file
        self.dom_s_id = 2  # value of the cell id for the solid domain in the parent mesh file
        self.fsi_id = 21  # IMPORTANT VARIABLE value of the FSI facet id in the parent mesh file
        self.inlet_id = 1  # fluid inlet id
        self.outlet_id = 2  # fluid outlet id
        self.inlet_s_fixed_id = 9 # solid inlet fixed facet id (one single facet attached)
        self.inlet_s_id = 10  # solid inlet id
        self.outlet_s_id = 11  # solid outlet id



    ############################################################################

    def get_parent_mesh(self):

        # Read mesh from mesh file
        self.parent_mesh = Mesh(self.mesh_folder + "/mesh_single.xml")
        #self.parent_mesh = refine(self.parent_mesh)

        # Read domains and boundaries from mesh file
        self.parent_domains = MeshFunction("size_t", self.parent_mesh, 3, self.parent_mesh.domains())
        self.parent_domains.set_values(self.parent_domains.array()+1)  # in order to have fluid==1 and solid==2

        # Send back the modification to the parent mesh file.
        for i in range(len(self.parent_domains.array())):
            val = self.parent_domains.array()[i]
            dic = (i, val)
            self.parent_mesh.domains().set_marker(dic, 3)

        self.parent_bds = MeshFunction("size_t", self.parent_mesh, 2, self.parent_mesh.domains())

        return

    ############################################################################

    def reshape_parent_mesh(self):

        return

    ############################################################################

    def setup_Dirichlet_BCs(self):

        noslip = Constant((0.0, 0.0, 0.0))
        freeslip = Constant(0.0)
        fixed_outlet = True

        # Fluid velocity BCs

        # Dirichlet conditions for the fluid pressure problem

        # Mesh problem bcs

        # Dirichlet conditions for the solid problem
        if fixed_outlet:
            self.bcs_s_vals = [freeslip, freeslip, noslip]
            self.bcs_s_ids = [self.inlet_s_fixed_id, self.inlet_s_id, self.outlet_s_id]
            self.bcs_s_fct_sps = ["y", "y", "xyz"]  # "x", "y", "z" (freeslip) or "xyz" (noslip)
        else:
            self.bcs_s_vals = [noslip, freeslip, freeslip]
            self.bcs_s_ids = [self.inlet_s_fixed_id, self.inlet_s_id, self.outlet_s_id]
            self.bcs_s_fct_sps = ["xyz", "y", "y"]  # "x", "y", "z" (freeslip) or "xyz" (noslip)

        return

    ############################################################################

    def post_process(self, t,  **namespace):

        #self.p_time_exp.t = t

        pass
