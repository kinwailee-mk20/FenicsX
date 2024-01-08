from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.io.utils import VTKFile, XDMFFile
import gmsh
import numpy as np
from dolfinx.fem import (Constant, Function, FunctionSpace, VectorFunctionSpace, dirichletbc, locate_dofs_topological,
                         locate_dofs_geometrical, Expression)
from petsc4py.PETSc import ScalarType
from dolfinx import mesh
from ufl import TestFunction, TrialFunction, dot, dx, nabla_grad, inner
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.utils import XDMFFile
import sys

#######################################################################################################################
# space
w_space = 1.3
h_space = 1

# tube dimension
w_tube = 0.5
h_tube = 0.1

# tube location
x_tube = w_space / 2.5
y_tube = h_space / 2.5

file_path = "new"

# object dimension
r_obj_1 = 0.02

# object location
# L1 > x = 0.04, y = 0.03
# L2 > x = 0.24 y = 0.05
# L3 > x = 0.46, y = 0.07

x_obj_1 = x_tube + 0.04
y_obj_1 = y_tube + 0.03


# define permittivity #################################################################################################
space_perm = 1 * 8.854e-11
tube_perm = 6 * 8.854e-11
obj1_perm = 3 * 8.854e-11


# starting ############################################################################################################
gmsh.initialize()
gmsh.model.add("2D Electrode Rings")


#define background
space = gmsh.model.occ.addRectangle(0, 0, 0, w_space, h_space)
gmsh.model.occ.synchronize()

#define tube
tube = gmsh.model.occ.addRectangle(x_tube, y_tube, 0, w_tube, h_tube)
gmsh.model.occ.synchronize()

#define foreign object
obj_1 = gmsh.model.occ.addDisk(x_obj_1, y_obj_1, 0, r_obj_1, r_obj_1)
gmsh.model.occ.synchronize()

objects = [(2, obj_1)]
all_surfaces = [(2, tube)]
all_surfaces.extend(objects)

gmsh.model.occ.fragment([(2, space)], all_surfaces)
gmsh.model.occ.synchronize()


# com_space
x_com_centre_space, y_com_centre_space = w_space / 2, h_space / 2  # a
x_com_left_space, y_com_left_space = 0, h_space / 2  # b
x_com_right_space, y_com_right_space = w_space, h_space / 2  # c
x_com_top_space, y_com_top_space = w_space / 2, h_space  # d
x_com_bottom_space, y_com_bottom_space = w_space / 2, 0  # e

# com_tube
x_com_centre_tube, y_com_centre_tube = x_tube + (w_tube / 2), y_tube + (h_tube / 2)  # A
x_com_left_tube, y_com_left_tube = x_tube, y_tube + (h_tube / 2)  # B
x_com_right_tube, y_com_right_tube = x_tube + w_tube, y_tube + (h_tube / 2)  # C
x_com_top_tube, y_com_top_tube = x_tube + (w_tube / 2), y_tube + h_tube  # D
x_com_bottom_tube, y_com_bottom_tube = x_tube + (w_tube / 2), y_tube  # E

# com_object
x_com_centre_object, y_com_centre_object = x_obj_1, y_obj_1

lines = gmsh.model.occ.getEntities(dim=1)
surfaces = gmsh.model.occ.getEntities(dim=2)

line_border = []
line_pos_terminal = []
line_neg_terminal = []
line_object = []
line_space = []

surface_space = []
surface_medium = []
surface_object = []

pos_marker, neg_marker, medium_marker, space_marker, object_marker = 1, 3, 5, 7, 9

for line in lines:
    com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
    if np.allclose(com, [x_com_left_space, y_com_left_space, 0]) or np.allclose(com, [x_com_right_space, y_com_right_space, 0]) or np.allclose(com, [x_com_top_space, y_com_top_space, 0]) or np.allclose(com, [x_com_bottom_space, y_com_bottom_space, 0]):
        line_space.append(line[1])
    elif np.allclose(com, [x_com_left_tube, y_com_left_tube, 0]):
        line_pos_terminal.append(line[1])
    elif np.allclose(com, [x_com_right_tube, y_com_right_tube, 0]):
        line_neg_terminal.append(line[1])
    elif np.allclose(com, [x_com_top_tube, y_com_top_tube, 0]) or np.allclose(com,[x_com_bottom_tube, y_com_bottom_tube, 0]):
        line_border.append(line[1])
    else:
        line_object.append(line[1])

for surface in surfaces:
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    if np.allclose(com, [x_com_centre_space, y_com_centre_space, 0], atol=1.e-2):
        surface_space.append(surface[1])
    elif np.allclose(com, [x_com_centre_object, y_com_centre_object, 0]):
        surface_object.append(surface[1])
    elif np.allclose(com, [x_com_centre_tube, y_com_centre_tube, 0], atol=1.e-3):
        surface_medium.append(surface[1])
    else:
        surface_medium.append(surface[1])

gmsh.model.addPhysicalGroup(1, line_space, tag=space_marker, name='space')
gmsh.model.addPhysicalGroup(1, line_neg_terminal, tag=neg_marker, name='neg')
gmsh.model.addPhysicalGroup(1, line_pos_terminal, tag=pos_marker, name='pos')
gmsh.model.addPhysicalGroup(1, line_border, tag=medium_marker, name='medium')
gmsh.model.addPhysicalGroup(1, line_object, tag=object_marker, name='obj1')

gmsh.model.addPhysicalGroup(2, surface_space, tag=space_marker, name='space')
gmsh.model.addPhysicalGroup(2, surface_medium, tag=medium_marker, name='medium')
gmsh.model.addPhysicalGroup(2, surface_object, tag=object_marker, name='obj1')


# for illustration
#phantom_tube = gmsh.model.occ.addRectangle(x_tube, y_tube, 0, w_tube, h_tube)
#gmsh.model.occ.synchronize()
#gmsh.model.addPhysicalGroup(2, [phantom_tube], tag=medium_marker+50, name="tube_phantom")

#o1 = gmsh.model.occ.addDisk(x_obj_1, y_obj_1, 0, r_obj_1, r_obj_1)
#gmsh.model.occ.synchronize()
#gmsh.model.addPhysicalGroup(2, [o1], tag=object_marker+50, name="obj1_phantom")


# define mesh sizes
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.006)


# generate mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
gmsh.model.mesh.optimize("Netgen")
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
gmsh.finalize()


# define function space for electric potential
V = FunctionSpace(domain, ("P", 1))


# define function and function space for permittivity
tdim = domain.topology.dim
fdim = tdim - 1
Q_eps = FunctionSpace(domain, ("DG", 0))
material_tags = np.unique(cell_markers.values)
eps = Function(Q_eps)


# assign permittivity
eps.x.array[:] = tube_perm
for tag in material_tags:
    cells = cell_markers.find(tag)
    if tag == object_marker:
        eps_ = obj1_perm
    elif tag == medium_marker:
        eps_ = tube_perm
    else:
        eps_ = space_perm
    eps.x.array[cells] = np.full_like(cells, eps_, dtype=ScalarType)

domain.topology.create_connectivity(tdim, fdim)

print('line_border: ', line_border)
print('line_pos_terminal: ', line_pos_terminal)
print('line_neg_terminal: ', line_neg_terminal)
print('line_object: ', line_object)
print('line_space: ', line_space)
print('surface_space: ', surface_space)
print('surface_medium: ', surface_medium)
print('surface_object: ', surface_object)
print('material_tags: ', material_tags)


# define boundary conditions
v_1 = ScalarType(10)
v_0 = ScalarType(-10)


def negative(x):
    return np.logical_and(np.isclose(x[0], x_tube + w_tube), np.logical_and(x[1] <= y_tube + h_tube, y_tube < x[1]))


def positive(x):
    return np.logical_and(np.isclose(x[0], x_tube), np.logical_and(x[1] <= y_tube + h_tube, y_tube < x[1]))


positive_bc = locate_dofs_geometrical(V, positive)
negative_bc = locate_dofs_geometrical(V, negative)

bcs_pos = dirichletbc(v_1, positive_bc, V)
bcs_neg = dirichletbc(v_0, negative_bc, V)
bcs = [bcs_pos, bcs_neg]


# solve problem
u = TrialFunction(V)
v = TestFunction(V)

a = dot(nabla_grad(u), nabla_grad(v)) * eps * dx
L = Constant(domain, 1e-16) * v * dx

uh = Function(V)
problem = LinearProblem(a, L, u=uh, bcs=bcs)
problem.solve()


# generate electric field
E_field = -nabla_grad(uh)

V_E_field = VectorFunctionSpace(domain, ("DG", 1))
E_field_expr = Expression(E_field, V_E_field.element.interpolation_points())
E_field_projected = Function(V_E_field)
E_field_projected.interpolate(E_field_expr)


# export solution
print("Export the solution to file...")
with XDMFFile(domain.comm, file_path + ".xdmf", "w") as xdmf:
    domain.name = 'cylinder geometry'
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers)
    xdmf.write_meshtags(facet_markers)
    uh.name = "Electric potential"
    xdmf.write_function(uh)
    E_field_projected.name = "Electric field"
    xdmf.write_function(E_field_projected)

