#!/usr/bin/env python

__author__ = "Alban Souche <alban@simula.no>"
__date__ = "2019-13-03"
__copyright__ = "Copyright (C) 2019 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

""" structure_problem.py module executes the cbc.twist solver to compute the
structure defromation of the fsi problem.
"""

from cbc.twist import *
import numpy as np


class StructureSolver(Hyperelasticity):

    def __init__(self, setup, subM, tract_S):
        # Initialize base class
        Hyperelasticity.__init__(self)

        # FSI Setup parameters
        self.mesh_s = subM.mesh_s
        self.T = setup.T
        self.dt = setup.dt
        self.dynamic = setup.is_dynamic

        self.rho_s = setup.rho_s  # kg/m3
        self.mu_s = setup.mu_s  # N/m2 == (kg/m/s2) == Pa
        self.lmbda = setup.lamda_s  # N/m2

        self.body_force_exp = setup.body_force
        self.initial_displacement = setup.u0
        self.initial_velocity = setup.v0
        self.pre_stress_exp = setup.pre_stress


        if setup.solid_solver_model == 'LinearElastic':
            self.material = LinearElastic([self.mu_s, self.lmbda])
        elif setup.solid_solver_model == 'StVenantKirchhoff':
            self.material = StVenantKirchhoff([self.mu_s, self.lmbda])
        elif setup.solid_solver_model == 'MooneyRivlin':
            self.C1 = self.mu_s/2
            self.C2 = self.mu_s/2
            self.material = MooneyRivlin([self.C1, self.C2])
        elif setup.solid_solver_model == 'neoHookean':
            self.half_nkT = self.mu_s/2
            self.material = neoHookean([self.half_nkT])
        elif setup.solid_solver_model == 'Isihara':
            self.C10 = self.mu_s/2
            self.C01 = self.mu_s/2
            self.C20 = self.mu_s/2
            self.material = Isihara([self.C10, self.C01, self.C20])
        elif setup.solid_solver_model == 'Biderman':
            self.C10 = self.mu_s/2
            self.C01 = self.mu_s/2
            self.C20 = self.mu_s/2
            self.C30 = self.mu_s/2
            self.material = Biderman([self.C10, self.C01, self.C20, self.C30])
        elif setup.solid_solver_model == 'GentThomas':
            self.C1 = self.mu_s/2
            self.C2 = self.mu_s/2
            self.material = GentThomas([self.C1, self.C2])
        elif setup.solid_solver_model == 'Ogden':
            # neoHookean
            self.alpha1 = 2
            self.alpha2 = 1
            self.alpha3 = 1
            self.mu1 = 2*self.mu_s
            self.mu2 = 0
            self.mu3 = 0
            self.material = Ogden([self.alpha1, self.alpha2, self.alpha3, self.mu1, self.mu2, self.mu3])

        self.d_deg = setup.d_deg
        self.D_bcs_vals = setup.bcs_s_vals  # Dirichlet values
        self.D_bcs_ids = [(subM.boundaries_s, bcs_s_id) for bcs_s_id in setup.bcs_s_ids]  # Dirichlet ids
        self.D_bcs_fct_sps = setup.bcs_s_fct_sps # Dirichlet function space types

        self.N_bcs_vals = [tract_S]  # Neumann values
        self.N_bcs_ids = ["on_boundary"]  # Neumann ids

        self.solver_scheme = setup.solid_solver_scheme  # MomentumBalanceSolver: "CG1" or "HHT" (Hilber-Hughes-Taylor)

        # Don't plot and save solution in subsolvers
        self.parameters["solver_parameters"]["plot_solution"] = False
        self.parameters["solver_parameters"]["save_solution"] = False
        self.parameters["solver_parameters"]["store_solution_data"] = False
        self.parameters["solver_parameters"]["element_degree"] = setup.d_deg

    def body_force(self):
        return self.body_force_exp

    def mesh(self):
        return self.mesh_s

    def end_time(self):
        return self.T

    def time_step(self):
        return self.dt

    def is_dynamic(self):
        return self.dynamic

    def dirichlet_values(self):
        return self.D_bcs_vals

    def dirichlet_boundaries(self):
        return self.D_bcs_ids  # [(MeshFunction, index1), (MeshFunction, index2)]

    def dirichlet_function_spaces(self):
        return self.D_bcs_fct_sps

    def initial_conditions(self):
        return self.initial_displacement, self.initial_velocity

    def neumann_conditions(self):
        return self.N_bcs_vals

    def neumann_boundaries(self):
        return self.N_bcs_ids

    def material_model(self):
        return self.material

    def pre_stress(self):
        return self.pre_stress_exp

    def reference_density(self):
        return self.rho_s

    def time_stepping(self):
        return self.solver_scheme

    def __str__(self):
        return "Solid domain for FSI semi-implicit coupling"

################################################################################
