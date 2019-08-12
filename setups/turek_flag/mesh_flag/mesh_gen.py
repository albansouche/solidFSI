from dolfin import *
import mshr
import matplotlib.pyplot as plt

mesh = Mesh('TF_fsi.xml.gz')
mesh = refine(mesh)
mesh = refine(mesh)

solid_string = '(x[0] > 0.2 && x[1] > 0.19 - tol && x[0] < 0.6 + tol && x[1] < 0.21 + tol)'
fluid_string = '(x[0] < 0.2 || x[1] < 0.19 + tol || x[0] > 0.6 - tol || x[1] > 0.21 - tol)'

inlet_f_string = '(x[0] < tol)'
outlet_f_string = '(x[0] > 2.5 - tol)'
wall_string = '(x[1] < tol || x[1] > 0.41 - tol)'
box_string = '({} || {} || {})'.format(inlet_f_string, outlet_f_string, wall_string)
#circle_string = '(pow(x[0]-0.2,2.0) + pow(x[1]-0.2,2.0) < pow(0.05,2.0) + tol)'
circle_string = '(on_boundary && ! {})'.format(box_string)
arc_f_string = '({} && {})'.format(fluid_string, circle_string)
arc_s_string = '({} && {})'.format(solid_string, circle_string)
noslip_f_string = '({} || {})'.format(wall_string, arc_f_string)
#print(noslip_f_string); import sys; sys.exit()
noslip_s_string = arc_s_string
interface_fsi_string = '({} && {})'.format(solid_string, fluid_string)

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
