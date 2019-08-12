from dolfin import *
import mshr
import matplotlib.pyplot as plt

Ny = 5
Nx = 100

mesh = RectangleMesh(Point(0.25,0.19), Point(0.6,0.21), Nx, Ny)
mesh = refine(mesh)

solid_string = 'true'
fluid_string = 'false'

inlet_f_string = 'false'
outlet_f_string = 'false'
noslip_f_string = 'false'
noslip_s_string = '(x[0] < 0.25+tol)'
interface_fsi_string = 'x[0] > 0.6-tol || x[1] < 0.19+tol || x[1] > 0.21-tol'

tol = 1.e-14

fluid = CompiledSubDomain(fluid_string, tol=tol)
solid = CompiledSubDomain(solid_string, tol=tol)

inlet_f = CompiledSubDomain(inlet_f_string, tol=tol)
outlet_f = CompiledSubDomain(outlet_f_string, tol=tol)
noslip_f = CompiledSubDomain(noslip_f_string, tol=tol)
noslip_s = CompiledSubDomain(noslip_s_string, tol=tol)
interface_fsi = CompiledSubDomain(interface_fsi_string, tol=tol)

subdomains = MeshFunction('size_t', mesh, 2)
subdomains.set_all(0)
fluid.mark(subdomains, 0)
solid.mark(subdomains, 1)

boundaries = MeshFunction('size_t', mesh, 1)
boundaries.set_all(5)
inlet_f.mark(boundaries, 0)
outlet_f.mark(boundaries, 1)
noslip_f.mark(boundaries, 2)
noslip_s.mark(boundaries, 3)
interface_fsi.mark(boundaries, 4)

File('mesh.xml') << mesh
File('subdomains.xml') << subdomains
File('boundaries.xml') << boundaries

File('boundaries.pvd') << boundaries

#File('mesh_fsi.pvd') << mesh
