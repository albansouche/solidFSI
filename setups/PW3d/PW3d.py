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



class P_exp(Expression):

   def __init__(self, p_max, f, L, t, **kwargs):
       self.p_max = p_max
       self.f = f
       self.L = L
       self.t = t

   def eval(self, value, x):
        value[0] = 0.5*self.p_max*self.t*(1.0+sin(2.0*pi*self.f*self.t))*(1.0-0.1*(x[1]+0.5*self.L)/self.L)

   def value_shape(self):
       return (1,)


class P_wave_exp(Expression):

   def __init__(self, p_max, L, velocity, width, t, **kwargs):
       self.p_max = p_max
       self.L = L
       self.a = velocity
       self.s = width
       self.t = t

   def eval(self, value, x):
        #value[0] = self.p_max*(abs(x[1]+0.5*self.L+4.0*self.s-self.a*self.t)<self.s)
        value[0] = self.p_max*exp(-0.5*pow((x[1]+0.5*self.L+4.0*self.s-self.a*self.t)/self.s, 2.0))

   def value_shape(self):
       return (1,)


class Setup(Setup_base):

    def __init__(self):

        # print setup
        print("Loading PW3d setup")

        # Initiate Setup_base, setting all unspecified arguments to []
        super().__init__()

        # FE order
        self.d_deg = 1  # Deformation degree (solid)

        # Time
        self.T = 0.1  # End time s.
        self.dt = 0.01  # Time step s.

        # Pressure
        self.p_in_max = 5.0E+5  # Max inlet pressure Pa
        f = 10.0
        L = 0.1
        a = 0.02
        sigma = 0.003
        t = 0.0
        #self.t_ramp = 0.1  # s.
        #self.p_exp = P_exp(self.p_in_max, f, L, t, degree=2)
        #self.p_exp = P_wave_exp(self.p_in_max, L, a, sigma, t, degree=1)
        self.p_exp = Expression('p_max*exp(-0.5*pow((x[1]+0.5*L+4.0*s-a*t)/s, 2.0))', p_max = self.p_in_max, L=L, a=a, s=sigma, t=t, degree = 2)
        #self.p_exp = Expression('p_max*0.5*(1.0+sin(2.0*pi*f*t))*exp(-0.5*pow((x[1]+0.5*L)/s, 2.0))', p_max=self.p_in_max, f=f, L=L, s=sigma, t=t, degree=1)
        #self.p_exp = Expression('p_max*t/T_max', p_max=self.p_in_max, T_max=self.T, t=t, degree=1)

        # Solid prop.
        self.rho_s = 1.0E3  # density
        self.nu_s = 0.45  # Poisson ratio
        self.mu_s = 345E3  # Shear modulus or 2nd Lame Coef.
        self.lamda_s = self.nu_s*2.*self.mu_s/(1. - 2.*self.nu_s)  # Young's modulus

        # Solver properties ###################################################

        # Dynamic or stationnary
        self.is_dynamic = True

        # path to CBC.solve
        self.CBCsolve_path = "library/external/cbc.solve"

        # saving data
        #self.save_path = "results/stress-strain"
        self.save_path = "results/tube/PW3d"
        self.save_step = 1  # saving solution every "n" steps

        # parent mesh info. !! values can be redefined in get_parent_mesh() !!
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

        # Material constitutive law
        # "LinearElastic" or "StVenantKirchhoff", "MooneyRivlin", "neoHookean", "Isihara", "Biderman", "GentThomas" or "Ogden"
        self.solid_solver_model = "StVenantKirchhoff"

        # solvers
        self.solid_solver_scheme = "HHT"  # "CG1" or "HHT"

        # set compiler arguments
        parameters["form_compiler"]["quadrature_degree"] = 6
        os.environ['OMP_NUM_THREADS'] = '4'

        # set log outputs from dolfin
        set_log_level(90)  # 0 to 100 / more info >> lower value

    ############################################################################

    def get_parent_mesh(self):

        # read mesh from mesh file
        self.parent_mesh = Mesh(self.mesh_folder + "/mesh_single.xml")
        #self.parent_mesh = refine(self.parent_mesh)

        # read domains and boundaries from mesh file
        self.parent_domains = MeshFunction("size_t", self.parent_mesh, 3, self.parent_mesh.domains())
        self.parent_domains.set_values(self.parent_domains.array()+1)  # in order to have fluid==1 and solid==2

        # send back the modification to the parent mesh file.
        for i in range(len(self.parent_domains.array())):
            val = self.parent_domains.array()[i]
            dic = (i, val)
            self.parent_mesh.domains().set_marker(dic, 3)

        #toto = File('toto.pvd')
        #toto << self.parent_mesh
        #toto << self.parent_domains
        self.parent_bds = MeshFunction("size_t", self.parent_mesh, 2, self.parent_mesh.domains())
        #toto << self.parent_bds


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
            self.bcs_s_fct_sps = ['y', 'y', 'xyz']  # 'x', 'y', 'z' (freeslip) or 'xyz' (noslip)
        else:
            self.bcs_s_vals = [noslip, freeslip, freeslip]
            self.bcs_s_ids = [self.inlet_s_fixed_id, self.inlet_s_id, self.outlet_s_id]
            self.bcs_s_fct_sps = ['xyz', 'y', 'y']  # 'x', 'y', 'z' (freeslip) or 'xyz' (noslip)

        return

    ############################################################################

    def post_process(self, t,  **namespace):

        #self.p_time_exp.t = t

        pass
