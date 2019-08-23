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
        self.solid_solver_model = "LinearElastic" # "LinearElastic", "StVenantKirchhoff"

        # Solvers
        self.solid_solver_scheme = "HHT"  # "HHT" or "CG1"

        # Set compiler arguments
        parameters["form_compiler"]["quadrature_degree"] = 6
        os.environ["OMP_NUM_THREADS"] = "4"

        # Set log outputs from dolfin
        set_log_level(90)  # 0 to 100 / more info >> lower value


        # DATA PARAMETERS #####################################################

        # Path to CBC.solve
        self.CBCsolve_path = "library/external/cbc.solve"

        # Saving data
        self.extension = self.solid_solver_model  # Filename extension
        self.save_path = "results/tube/PW3d/{}{}/{}".format(self.is_dynamic*"dynamic", (not self.is_dynamic)*"static", self.extension)
        self.save_step = 1  # saving solution every "n" steps

        # Parent mesh info. !! values can be redefined in get_parent_mesh() !!
        self.mesh_folder = "setups/PW3d/mesh"
        self.mesh_file = "{}{}.xml".format("cyl10x2cm", "")
        #self.mesh_file = "{}{}.xml".format("cyl25x4mm", "_double")

        self.mesh_split = True  # spliting the mesh True or False. Submesh are systematically saved in the folder
        self.parent_mesh = []  # if parent mesh provided by a xml or xml.gz file
        self.dom_f_id = 1  # value of the cell id for the fluid domain in the parent mesh file
        self.dom_s_id = 2  # value of the cell id for the solid domain in the parent mesh file
        self.fsi_id = 21  # IMPORTANT VARIABLE value of the FSI facet id in the parent mesh file
        self.inlet_id = 1  # fluid inlet id
        self.outlet_id = 2  # fluid outlet id
        self.inlet_s_fixed_id = 9  # solid inlet fixed facet id (one single facet attached)  # Initially  10, manually set to 9
        self.inlet_s_id = 10  # solid inlet id # Initially 10, not modified
        self.outlet_s_id = 11  # solid outlet id # Initially 10, manually set to 11 through script


        # PHYSICAL PROPERTIES #################################################

        # Tube geometry
        if "cyl10x2cm" in self.mesh_file:
            R_inner = 0.01
            R_outer = 0.012
            L = 0.1
        elif "cyl25x4mm" in self.mesh_file:
            R_inner = 0.002
            R_outer = 0.0023
            L = 0.025
        inlet = - 0.5*L

        # Time
        self.T  = 10  # End time s.
        self.dt = 0.1 # Time step s.

        # Solid properties
        self.rho_s = 1.0E3  # density
        self.nu_s = 0.45  # Poisson ratio
        self.mu_s = 345E3  # Shear modulus or 2nd Lame Coef.
        self.lamda_s = self.nu_s*2.*self.mu_s/(1. - 2.*self.nu_s)  # Young's modulus

        # Pressure
        mmHg = 133.322387415  # 1mmHg in Pa
        p_0 = 60*mmHg  # Diastolic pressure, Pa
        p_max = 120*mmHg  # Maximum pressure, Pa
        p_const_0 = "p_0"
        p_const_max = "p_max"
        p_increase = "p_0 + (p_max-p_0)*t/T_max"  # Linearely increasing
        p_wave = "p_0 + (p_max-p_0)*exp(-0.5*pow((x[1]-inlet+4.0*s-a*t)/s, 2.0))"  # Moving Gaussian
        p_inlet = "p_0 + (p_max-p_0)*0.5*(1.0+sin(2.0*pi*f*t))*exp(-0.5*pow((x[1]-inlet)/s, 2.0))"  # Oscillating inlet pressure

        p_string = p_increase  # Choose string for pressure expression
        self.p_exp = Expression(p_string, degree=2, p_max=p_max, p_0=p_0,
                                T_max=self.T, L=L, inlet=inlet, f=10, a=0.02,
                                s=0.03*L, t=0)

        # Body forces
        # self.body_force = Constant/Expression... ("x", "y", "z")

        # Pressure on FSI-interface that will automatically find u0 and change initial mesh
        self.pre_press_val = p_0  # Constant

        # Prestress (prescribed, without changing initial mesh)
        # self.prestress = Constant/Expression... (("xx", "xy", "xz"), ("yx", "yy", "yz"), ("zx", "zy", "zz"))

        # Initial conditions
        u0_string = ("factor*x[0]*exp(-0.5*pow((x[1]-inlet)/s, 2.0))",
                     "0.0",
                     "factor*x[2]*exp(-0.5*pow((x[1]-inlet)/s, 2.0))")
        #self.u0 = Expression(u0_string, degree=2, R=R_inner, factor=0.3, inlet=inlet, s=0.02*L)


        # OBSERVATION PARAMETERS ##############################################

        # Store quantities everywhere
        self.store_everywhere = [True, True]

        # Observe strain or stress ("displacement", "strain", "stress")
        self.quantities = ["strain", "stress"]

        # Operator on observed quantity (scalar) "[i,j]", "tr" or "vonMises"
        self.observators = ["[2,2]", "tr"]

        # Observation points
        not_quants = []
        not_quants.append("displacement")
        not_quants.append("traction")
        self.obs_points.append(obs_point("midpoint", Point(R_inner+0.5*(R_outer-R_inner), 0, 0), not_quants+self.quantities))
        #self.obs_points.append(obs_point("name", FEniCS-Point, quantities))


    ############################################################################

    def get_parent_mesh(self):

        # Read mesh from mesh file
        self.parent_mesh = Mesh("{}/{}".format(self.mesh_folder, self.mesh_file))
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

        # Fluid velocity BCs

        # Dirichlet conditions for the fluid pressure problem

        # Mesh problem bcs

        # Dirichlet conditions for the solid problem
        if 0:  # Freeslip inlet surface, fixed outlet
            self.bcs_s_vals = [freeslip, freeslip, noslip]
            self.bcs_s_ids = [self.inlet_s_fixed_id, self.inlet_s_id, self.outlet_s_id]
            self.bcs_s_fct_sps = ["y", "y", "xyz"]  # "x", "y", "z" (freeslip) or "xyz" (noslip)
        elif 0:  # One inlet facet attached, freeslip inlet surface, freeslip outlet
            self.bcs_s_vals = [noslip, freeslip, freeslip]
            self.bcs_s_ids = [self.inlet_s_fixed_id, self.inlet_s_id, self.outlet_s_id]
            self.bcs_s_fct_sps = ["xyz", "y", "y"]  # "x", "y", "z" (freeslip) or "xyz" (noslip)
        elif 0:  # Free inlet surface, fixed outlet
            self.bcs_s_vals = [noslip]
            self.bcs_s_ids = [self.outlet_s_id]
            self.bcs_s_fct_sps = ["xyz"]  # "x", "y", "z" (freeslip) or "xyz" (noslip)
        elif 1:   # Fixed inlet- and outlet surface
            self.bcs_s_vals = [noslip, noslip, noslip]
            self.bcs_s_ids = [self.inlet_s_fixed_id, self.inlet_s_id, self.outlet_s_id]
            self.bcs_s_fct_sps = ["xyz", "xyz", "xyz"]  # "x", "y", "z" (freeslip) or "xyz" (noslip)
        return

    ############################################################################

    def post_process(self, t,  **namespace):

        #self.p_time_exp.t = t

        pass
