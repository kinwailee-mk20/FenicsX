from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.io.utils import VTKFile, XDMFFile
import gmsh
import numpy as np
from dolfinx.fem import (Constant, Function, FunctionSpace, VectorFunctionSpace, dirichletbc, locate_dofs_topological,
                         locate_dofs_geometrical, Expression)
from petsc4py.PETSc import ScalarType
from dolfinx import mesh
from ufl import TestFunction, TrialFunction, dot, dx, nabla_grad, inner, grad
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.utils import XDMFFile
import sys
import math

#######################################################################################################################

# tube dimension
w_tube = 0.5
h_tube = 0.1

# tube coordinate
x_tube = 0
y_tube = 0

# object dimension
x_obj_rectangle, y_obj_rectangle = w_tube / 2, h_tube / 2

w_obj_rectangle, h_obj_rectangle = 0.01, 0.005


# conductivity
# conductor: 1.12e7
# insulator: 1e-15
# milk(tube): 1

cond_tube = 1
cond_obj = 1.12e7


# com_tube
x_com_centre_tube, y_com_centre_tube = x_tube + (w_tube / 2), y_tube + (h_tube / 2)  # A
x_com_left_tube, y_com_left_tube = x_tube, y_tube + (h_tube / 2)  # B
x_com_right_tube, y_com_right_tube = x_tube + w_tube, y_tube + (h_tube / 2)  # C
x_com_top_tube, y_com_top_tube = x_tube + (w_tube / 2), y_tube + h_tube  # D
x_com_bottom_tube, y_com_bottom_tube = x_tube + (w_tube / 2), y_tube  # E

# com_object
x_com_centre_object, y_com_centre_object = x_obj_rectangle + (w_obj_rectangle/2), y_obj_rectangle + (h_obj_rectangle/2)

file_path = "cond"

# starting ############################################################################################################

gmsh.initialize()
gmsh.model.add("2D Electrode Rings")

#define tube
tube = gmsh.model.occ.addRectangle(x_tube, y_tube, 0, w_tube, h_tube)
gmsh.model.occ.synchronize()

#define foreign object
obj_1 = gmsh.model.occ.addRectangle(x_obj_rectangle, y_obj_rectangle, 0, w_obj_rectangle, h_obj_rectangle)
gmsh.model.occ.synchronize()

objects = [(2, obj_1)]

gmsh.model.occ.fragment([(2, tube)], objects)
gmsh.model.occ.synchronize()

# find COM
line_border = []
line_pos_terminal = []
line_neg_terminal = []
line_object = []


surface_medium = []
surface_object = []
lines = gmsh.model.occ.getEntities(dim=1)
surfaces = gmsh.model.occ.getEntities(dim=2)

    # assign tag
neg_marker, pos_marker, box_marker, space_marker, medium_marker, object_marker = 1, 3, 5, 7, 9, 11


for line in lines:
    com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
    if np.allclose(com, [x_com_left_tube, y_com_left_tube, 0]):
        line_pos_terminal.append(line[1])
    elif np.allclose(com, [x_com_right_tube, y_com_right_tube, 0]):
        line_neg_terminal.append(line[1])
    elif np.allclose(com, [x_com_top_tube, y_com_top_tube, 0]) or np.allclose(com,
                                                                                  [x_com_bottom_tube, y_com_bottom_tube,
                                                                                   0]):
        line_border.append(line[1])
    else:
        line_object.append(line[1])


gmsh.model.addPhysicalGroup(1, line_neg_terminal, tag=neg_marker, name='neg')
gmsh.model.addPhysicalGroup(1, line_pos_terminal, tag=pos_marker, name='pos')
gmsh.model.addPhysicalGroup(1, line_border, tag=medium_marker, name='medium')
gmsh.model.addPhysicalGroup(1, line_object, tag=object_marker, name='obj1')


surface_object.append(surfaces[0][1])
surface_medium.append(surfaces[1][1])

gmsh.model.addPhysicalGroup(2, surface_medium, tag=medium_marker, name='medium')
gmsh.model.addPhysicalGroup(2, surface_object, tag=object_marker, name='obj1')



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
Q_cond = FunctionSpace(domain, ("DG", 0))
material_tags = np.unique(cell_markers.values)
facet_tags = np.unique(facet_markers.values)
cond = Function(Q_cond)


# assign conductivity (cell)
cond.x.array[:] = cond_tube
for tag in material_tags:
    cells = cell_markers.find(tag)
    if tag == object_marker:
        cond_ = cond_obj
    else:
        cond_ = cond_tube
    #print('marker', tag)
    cond.x.array[cells] = np.full_like(cells, cond_, dtype=ScalarType)


domain.topology.create_connectivity(tdim, fdim)

# define boundary conditions
v_1 = ScalarType(10)
v_0 = ScalarType(-10)


def negative(x):
    return np.isclose(x[0], x_tube + w_tube)

def positive(x):
    return np.isclose(x[0], 0)


# solve problem
u = TrialFunction(V)
v = TestFunction(V)

positive_bc = locate_dofs_geometrical(V, positive)
negative_bc = locate_dofs_geometrical(V, negative)


bcs_pos = dirichletbc(v_1, positive_bc, V)
bcs_neg = dirichletbc(v_0, negative_bc, V)
bcs = [bcs_pos, bcs_neg]


f = Constant(domain, 1e-16)
a = inner(cond * grad(u), grad(v)) * dx
L = f * v * dx

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


