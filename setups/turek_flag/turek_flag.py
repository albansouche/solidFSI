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


def stationnary_prestress(E, nu, model, A, common="prestress"):
    """Return analytical prestress and initial displacement.
    :raise: ValueError
    """
    C = A + 0.5*A**2
    D = -nu/(1-nu)*C
    B = np.sqrt(1+2*D) - 1
    Alin = (1+A)*C
    Blin = -nu/(1-nu)*Alin
    prestress_tract = E/(1-nu**2)*(1+A)*C


    u0 = Expression(("A*(x[0]-0.25)", "B*(x[1]-0.2)"), degree=2, A=A, B=B)  # Initial displacement

    if common == "prestress" and model == "LinearElastic":
        u0.A = Alin
        u0.B = Blin
    elif common == "displacement" and model == "LinearElastic":
        prestress_tract = E/(1-nu**2)*A

    P0 = Constant(((-prestress_tract, 0.0), (0.0, 0.0)))

    # Compatibility condition
    if Blin < -1  or D < -0.5:
        raise ValueError("Prestress traction too high.")

    return P0, u0




class Setup(Setup_base):
    """Setup configuration to be loaded by solidFSI.
    Unspecified paramters will be set to zero.
    """

    def __init__(self):

        # print setup
        print("Loading Turek flag setup")

        # IMPORTANT: Initiatialize Setup_base, setting all unspecified arguments to []
        super().__init__()


        # SOLVER PROPERTIES ###################################################

        # FE order
        self.d_deg = 1  # Deformation degree (solid)

        # Dynamic or stationnary
        self.is_dynamic = True

        # Material constitutive law
        self.solid_solver_model = "StVenantKirchhoff"

        # solvers
        self.solid_solver_scheme = "HHT"  #  "HHT" or "CG1"

        # set compiler arguments
        parameters["form_compiler"]["quadrature_degree"] = 6
        os.environ["OMP_NUM_THREADS"] = "8"#"4"

        # set log outputs from dolfin
        set_log_level(40)  # 0 to 100 / more info >> lower value


        # PHYSICAL PROPERTIES #################################################

        # Time
        self.T  = 0.1  # End time s.
        self.dt = 0.01  # Time step s.
        t = 0.0

        # Solid prop.
        self.rho_s = 1.0E3  # density
        self.nu_s = 0.4  # Poisson ratio
        self.mu_s = 0.5E6 #2.0E6 #  # Shear modulus or 2nd Lame Coef.
        self.lamda_s = self.nu_s*2.*self.mu_s/(1. - 2.*self.nu_s)
        self.young = 2*self.mu_s*(1+self.nu_s) # Young's modulus

        # FSI pressure expression
        #self.p_exp = Expression("0.6 - tol < x[0] ? value : 0.0", degree=2, tol=1.0E-14, value=-1.0E3, t=t)  # Traction

        # Body forces
        g = 2.0
        gravity = Constant((0.0, -self.rho_s*g))
        #factor = 100  # Horizontal to vertical body force ratio in static test
        #gravity = Constant((factor*self.rho_s*g, -self.rho_s*g))
        self.body_force = gravity

        # Stationnary prestress
        P0, u0 = [], []
        A_svk = 0.01  # Beam strain StVenantKirchhoff
        #P0, u0 = stationnary_prestress(E=self.young, nu=self.nu_s, model=self.solid_solver_model, A=A_svk, common="prestress")

        # Prestress
        self.prestress = P0

        # Initial conditions
        if self.is_dynamic:
            self.u0 = u0  # Inititial displacement
            #self.u0 = Expression(("0.0", "-pow(x[0]-0.25, 2.0)"), degree=2)  # Initial displacement
            #self.v0 = Expression(("0.0", "x[0]-0.25"), degree=2) # Initial velocity




        # OBSERVATION PARAMETERS ##############################################

        # Store quantities everywhere
        self.store_everywhere = [True]

        # Observe strain or stress ("strain", "stress")
        self.quantities = ["stress"]

        # Operator on observed quantity (scalar) "[i,j]", "tr" or "vonMises"
        self.observators = ["tr"]

        # Observation points
        not_quants = []
        not_quants.append("displacement")
        self.obs_points.append(obs_point("A", Point(0.6, 0.2), not_quants+self.quantities))



        # DATA PARAMETERS #####################################################

        # Flag or box imitating flag
        flag_or_box = "box"

        # path to CBC.solve
        self.CBCsolve_path = "library/external/cbc.solve"

        # saving data
        self.save_step = 1  # saving solution every "n" steps
        self.save_path = "results/turek_flag/{}/".format("dynamic"*self.is_dynamic + "static"*(not self.is_dynamic))
        self.extension = self.solid_solver_model
        #self.extension += "_{}".format(factor)
        self.save_path += self.extension

        # parent mesh info. !! values can be redefined in get_parent_mesh() !!
        self.mesh_folder = "setups/turek_flag/mesh_{}".format(flag_or_box)
        self.mesh_split = True  # spliting the mesh True or False. Submesh are systematically saved in the folder
        self.parent_mesh = []  # if parent mesh provided by a xml or xml.gz file
        self.dom_f_id = 0  # value of the cell id for the fluid domain in the parent mesh file
        self.dom_s_id = 1  # value of the cell id for the solid domain in the parent mesh file
        self.inlet_id = 0  # fluid inlet id
        self.outlet_id = 1  # fluid outlet id
        self.noslip_f_id = 2 # fluid wall id
        self.noslip_s_id = 3  # solid clamp id
        self.fsi_id = 4  # IMPORTANT VARIABLE value of the FSI facet id in the parent mesh file

        # "Box-flag" mesh does not have a fluid domain.
        if flag_or_box == "box":
            self.dom_f_id = self.dom_s_id


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
        freeslip = Constant(0.0)

        # Fluid velocity BCs

        # Dirichlet conditions for the fluid pressure problem

        # Mesh problem bcs

        # Dirichlet conditions for the solid problem
        self.bcs_s_vals = [noslip]
        self.bcs_s_ids = [self.noslip_s_id]
        self.bcs_s_fct_sps = ["xyz"]  # "x", "y", "z" (freeslip) or "xyz" (noslip)

        return

    ############################################################################

    def post_process(self, t,  **namespace):

        #self.p_time_exp.t = t

        pass
