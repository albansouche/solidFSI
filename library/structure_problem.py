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

        self.rho_s = setup.rho_s  # kg/m3
        self.mu_s = setup.mu_s  # N/m2 == (kg/m/s2) == Pa
        self.lmbda = setup.lamda_s  # N/m2

        if setup.solid_solver_model == 'LinearElastic':
            self.material = LinearElastic([self.mu_s, self.lmbda])
        elif setup.solid_solver_model == 'StVenantKirchhoff':
            self.material = StVenantKirchhoff([self.mu_s, self.lmbda])
        # TODO: add them all from CBC material models

        self.D_bcs_vals = setup.bcs_s_vals  # Dirichlet values #FIXME, for more than 1 boundrary condition
        self.D_bcs_ids = [(subM.boundaries_s, setup.bcs_s_ids[0])]  # Dirichlet ids #FIXME, for more than 1 boundrary condition

        self.N_bcs_vals = [tract_S]  # Neumann values
        self.N_bcs_ids = ["on_boundary"]  # Neumann ids

        self.solver_scheme = setup.solid_solver_scheme  # MomentumBalanceSolver: "CG1" or "HHT" (Hilber-Hughes-Taylor)

        # Don't plot and save solution in subsolvers
        self.parameters["solver_parameters"]["plot_solution"] = False
        self.parameters["solver_parameters"]["save_solution"] = False
        self.parameters["solver_parameters"]["store_solution_data"] = False
        self.parameters["solver_parameters"]["element_degree"] = setup.d_deg

    def mesh(self):
        return self.mesh_s

    def end_time(self):
        return self.T

    def time_step(self):
        return self.dt

    def is_dynamic(self):
        return True

    def dirichlet_values(self):
        return self.D_bcs_vals

    def dirichlet_boundaries(self):
        return self.D_bcs_ids  # [(MeshFunction, index1), (MeshFunction, index2)]

    def neumann_conditions(self):
        return self.N_bcs_vals

    def neumann_boundaries(self):
        return self.N_bcs_ids

    def material_model(self):
        return self.material

    def reference_density(self):
        return self.rho_s

    def time_stepping(self):
        return self.solver_scheme

    def __str__(self):
        return "Solid domain for FSI semi-implicit coupling"

################################################################################
