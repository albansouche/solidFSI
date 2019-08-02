#!/usr/bin/env python

__author__ = "Alban Souche <alban@simula.no>"
__date__ = "2019-13-03"
__copyright__ = "Copyright (C) 2019 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

""" fsi_coupling.py module collects functions and classes used for the
fluid-structure coupling:
>> Stress Operators:
    def _Sigma_F
    def _Sigma_S
>> class FSISubmeshes:
    def read
    def split
    def DOFs_fsi
>> class AitkenRelaxation
    def __init__
    def init
    def step
"""

################################################################################

from dolfin import *
import numpy as np
import os

from IPython import embed

## OPERATORS from CBC.twist ####################################################
from cbc.twist import DeformationGradient as F
from cbc.twist import Jacobian as J
from cbc.twist import GreenLagrangeStrain as E
from cbc.twist.kinematics import SecondOrderIdentity as I


def _Sigma_F(U_F, P_F, U_M, mu_F):
    "Return fluid stress in reference domain (not yet Piola mapped)"
    return mu_F*(grad(U_F)*inv(F(U_M)) + inv(F(U_M)).T*grad(U_F).T) - P_F*I(U_F)


def _Sigma_S(U_S, mu_S, lmbda_S):
    "Return structure stress in reference domain"
    return dot(F(U_S), 2*mu_S*E(U_S) + lmbda_S*tr(E(U_S))*I(U_S))


################################################################################


class FSISubmeshes:

    def read(self, mesh_folder):
        """
        mesh_folder: where to read the data (best place in ProjectionFSI/setups/meshes/...)
        """

        # Read files if previously created
        self.mesh_f = Mesh()
        self.mesh_s = Mesh()
        self.mesh_m = Mesh()
        fr = HDF5File(mpi_comm_world(), mesh_folder + "/subMeshes.h5", 'r')
        fr.read(self.mesh_f, 'mesh_f', False)
        fr.read(self.mesh_s, 'mesh_s', False)
        fr.read(self.mesh_m, 'mesh_f', False)
        self.boundaries_f = MeshFunction('size_t', self.mesh_f, self.mesh_f.geometry().dim() - 1)
        self.boundaries_s = MeshFunction('size_t', self.mesh_s, self.mesh_s.geometry().dim() - 1)
        self.boundaries_m = MeshFunction('size_t', self.mesh_m, self.mesh_m.geometry().dim() - 1)
        fr.read(self.boundaries_f, 'boundaries_f')
        fr.read(self.boundaries_s, 'boundaries_s')
        fr.read(self.boundaries_m, 'boundaries_f')

        print('Fluid Mesh: %d vertices:' % self.mesh_f.num_vertices(), '%d cells:' % self.mesh_f.num_cells())
        print('Solid Mesh: %d vertices:' % self.mesh_s.num_vertices(), '%d cells:' % self.mesh_s.num_cells())

        return

    def split(self, mesh_folder, mesh, dom_f_id, dom_s_id):
        """
        mesh_folder: where to read the data (best place in ProjectionFSI/setups/setup_name/mesh/)
        parent_mesh: Mesh()  OR  .xml(.gz) file in mesh_folder
        dom_f_id: domain id used for the fluid
        dom_s_id: domain id used for the solid
        fsi_boundary_id: boundary id along the fluid and solid interface
        """

        # read parent mesh
        if isinstance(mesh, str):
            mesh = Mesh(mesh_folder + mesh)
        elif isinstance(mesh, Mesh) != True:
            print('Error with import of the parent_mesh!')
        print('Parent Mesh: %d vertices:' % mesh.num_vertices(), '%d cells:' % mesh.num_cells())

        bd_vals = mesh.domains().markers(mesh.geometry().dim()-1).values()
        bd_indx = mesh.domains().markers(mesh.geometry().dim()-1).keys()
        dom_vals = mesh.domains().markers(mesh.geometry().dim()).values()
        dom_indx = mesh.domains().markers(mesh.geometry().dim()).keys()

        domains = MeshFunction('size_t', mesh, mesh.geometry().dim(), mesh.domains())

        mesh_f = SubMesh(mesh, domains, dom_f_id)  # subdivide of the parent mesh
        mesh_s = SubMesh(mesh, domains, dom_s_id)  # subdivide of the parent mesh

        bd_f_vals = mesh_f.domains().markers(mesh.geometry().dim()-1).values()
        bd_f_indx = mesh_f.domains().markers(mesh.geometry().dim()-1).keys()
        bd_s_vals = mesh_s.domains().markers(mesh.geometry().dim()-1).values()
        bd_s_indx = mesh_s.domains().markers(mesh.geometry().dim()-1).keys()

        boundaries_f = MeshFunction('size_t', mesh_f, mesh.geometry().dim()-1)
        boundaries_s = MeshFunction('size_t', mesh_s, mesh.geometry().dim()-1)
        boundaries_f.set_all(0)
        boundaries_s.set_all(0)

        boundaries_f.array()[np.array(list(bd_f_indx))] = np.array(list(bd_f_vals))
        boundaries_s.array()[np.array(list(bd_s_indx))] = np.array(list(bd_s_vals))

        print('Fluid Mesh: %d vertices:' % mesh_f.num_vertices(), '%d cells:' % mesh_f.num_cells())
        print('Solid Mesh: %d vertices:' % mesh_s.num_vertices(), '%d cells:' % mesh_s.num_cells())

        # Save the files for reuse, of for later parallel execution (SubMesh runs serial only!)
        if os.path.isfile(mesh_folder + "/subMeshes.h5"):
            os.remove(mesh_folder + "/subMeshes.h5")
        fw = HDF5File(mpi_comm_world(), mesh_folder + "/subMeshes.h5", "w")
        # fw.flush()
        fw.write(mesh_f, "mesh_f")
        fw.write(boundaries_f, "boundaries_f")
        fw.write(mesh_s, "mesh_s")
        fw.write(boundaries_s, "boundaries_s")

        # update self variables
        self.mesh_f = mesh_f
        self.mesh_s = mesh_s
        self.boundaries_f = boundaries_f
        self.boundaries_s = boundaries_s

        self.mesh_m = Mesh()
        fr = HDF5File(mpi_comm_world(), mesh_folder + "/subMeshes.h5", 'r')
        #fr = XDMFFile(mpi_comm_world(), mesh_folder + "/subMeshes.xdmf", 'r')
        fr.read(self.mesh_m, 'mesh_f', False)
        self.boundaries_m = MeshFunction('size_t', self.mesh_m, self.mesh_m.geometry().dim() - 1)
        fr.read(self.boundaries_m, 'boundaries_f')

        return

    def DOFs_fsi(self, V, D, fsi_id):
        """
        Mapping of the fsi degrees of freedom from V and D.
        V and D should have same element type and order!
        """

        # DOFS MAPPING # parallel ready ########################################

        # FSI nodes from V vectorspace (hack of DirichletBC outputs)
        tmp = DirichletBC(V, Constant((1.0,)*self.mesh_f.geometry().dim()), self.boundaries_f, fsi_id)
        dum = Function(V)
        tmp.apply(dum.vector())
        fsi_dofs_f = np.array(np.where(dum.vector().gather_on_zero() > 0.0)).flatten()

        # FSI nodes from D vectorspace (hack of DirichletBC outputs)
        tmp = DirichletBC(D, Constant((1.0,)*self.mesh_s.geometry().dim()), self.boundaries_s, fsi_id)
        dum = Function(D)
        tmp.apply(dum.vector())
        fsi_dofs_s = np.array(np.where(dum.vector().gather_on_zero() > 0.0)).flatten()

        # 2017.2.0 # parallel ready
        comm = self.mesh_f.mpi_comm().tompi4py()
        local_dofs_x = V.tabulate_dof_coordinates()
        X = np.concatenate(comm.allgather(local_dofs_x)).reshape((-1, self.mesh_f.geometry().dim()))
        X_fsi_f = X[fsi_dofs_f.astype(int), ]

        # 2017.2.0 # parallel ready
        comm = self.mesh_s.mpi_comm().tompi4py()
        local_dofs_x = D.tabulate_dof_coordinates()
        X = np.concatenate(comm.allgather(local_dofs_x)).reshape((-1, self.mesh_s.geometry().dim()))
        X_fsi_s = X[fsi_dofs_s.astype(int), ]

        order2_s = np.zeros(len(X_fsi_s), "int")
        cpt = 0
        for i in range(0, len(X_fsi_s), self.mesh_s.geometry().dim()):
            tmp = np.where(np.sum((X_fsi_s[i, ] == X_fsi_f), 1) == self.mesh_s.geometry().dim())
            fsi_idx = tmp[0]
            if self.mesh_s.geometry().dim() == 3:
                order2_s[[i, i+1, i+2]] = fsi_idx.astype(int)
            elif self.mesh_s.geometry().dim() == 2:
                order2_s[[i, i+1]] = fsi_idx  # store map in vector
        fsi_dofs_f = fsi_dofs_f[order2_s.astype(int), ]  # reodering corresponding to fsi map fluid<>solid

        self.fsi_dofs_f = fsi_dofs_f
        self.fsi_dofs_s = fsi_dofs_s

        return


################################################################################
