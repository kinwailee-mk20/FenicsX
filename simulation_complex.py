from dolfinx.io.utils import XDMFFile, VTKFile


from dolfinx.fem import (Constant, Function, functionspace, Expression, VectorFunctionSpace,
                         dirichletbc, locate_dofs_topological, locate_dofs_geometrical)
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, dot, dx, nabla_grad, curl, inner, conj, grad
from mpi4py import MPI
from dolfinx.io import gmshio
import numpy as np
import gmsh


# complex setup
frequency = 10e3
omega = 2 * np.pi * frequency    # frequency = 10kHz


# tube dimension (m)
w_tube = 0.5
h_tube = 0.1

x_tube = 0                          # tube at the centre of the space
y_tube = 0

# object dimension (m)
w_obj_rectangle, h_obj_rectangle = 0.01, 0.01


# object_coordinates
K = 0
y_obj_rectangle = (h_tube / 2) - (h_obj_rectangle / 2) + K
x_obj_rectangle = (w_tube / 2) - (w_obj_rectangle / 2)


# conductivity (S/m)
# conductor: 1.12e7
# insulator: 1e-14
# milk: 1
epsilon_0 = 8.854e-12
tube_perm = 6 * epsilon_0
tube_cond = 1 + omega * tube_perm * 1j

object_perm_cond = 5000 * epsilon_0                      # permittivity of iron = 5000
object_cond_cond = 1e8 + omega * object_perm_cond * 1j

object_perm_ins = 3 * epsilon_0
object_cond_ins = 1e-12 + omega * object_perm_ins * 1j      # permittivity of plastic = 3


mesh_size = 0.0003125

# com_tube
x_com_centre_tube, y_com_centre_tube = x_tube + (w_tube / 2), y_tube + (h_tube / 2)  # A
x_com_left_tube, y_com_left_tube = x_tube, y_tube + (h_tube / 2)  # B
x_com_right_tube, y_com_right_tube = x_tube + w_tube, y_tube + (h_tube / 2)  # C
x_com_top_tube, y_com_top_tube = x_tube + (w_tube / 2), y_tube + h_tube  # D
x_com_bottom_tube, y_com_bottom_tube = x_tube + (w_tube / 2), y_tube  # E


# gmsh initialization
gmsh.initialize()
gmsh.model.add("2D Electrode Rings")

file_path = "cond_{min}".format(min=mesh_size)

# define tube
tube = gmsh.model.occ.addRectangle(x_tube, y_tube, 0, w_tube, h_tube)
gmsh.model.occ.synchronize()

# define foreign object
obj_1 = gmsh.model.occ.addRectangle(x_obj_rectangle, y_obj_rectangle, 0, w_obj_rectangle, h_obj_rectangle)

# conformal surface
objects = [(2, obj_1)]
gmsh.model.occ.fragment([(2, tube)], objects)
gmsh.model.occ.synchronize()

# find COM
line_border = []
line_pos_terminal = []
line_neg_terminal = []
line_object = []
line_space = []

surface_space = []
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
    elif np.allclose(com, [x_com_top_tube, y_com_top_tube, 0]) or np.allclose(com, [x_com_bottom_tube, y_com_bottom_tube, 0]):
        line_border.append(line[1])
    else:
        line_object.append(line[1])


# define physical group
gmsh.model.addPhysicalGroup(1, line_neg_terminal, tag=neg_marker, name='neg')
gmsh.model.addPhysicalGroup(1, line_pos_terminal, tag=pos_marker, name='pos')
gmsh.model.addPhysicalGroup(1, line_border, tag=medium_marker, name='medium')
gmsh.model.addPhysicalGroup(1, line_object, tag=object_marker, name='obj1')


surface_object.append(surfaces[0][1])
surface_medium.append(surfaces[1][1])

gmsh.model.addPhysicalGroup(2, surface_medium, tag=medium_marker, name='medium')
gmsh.model.addPhysicalGroup(2, surface_object, tag=object_marker, name='obj1')


# define mesh sizes
gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

gmsh.model.occ.removeAllDuplicates
gmsh.model.mesh.removeDuplicateNodes
gmsh.model.mesh.removeDuplicateElements

# generate mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
gmsh.model.mesh.optimize("Netgen")
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
gmsh.finalize()

# define function space for electric potential
V = functionspace(domain, ("Lagrange", 1))

# define function and function space for permittivity
tdim = domain.topology.dim
fdim = tdim - 1
Q_sigma = functionspace(domain, ("DG", 0))
material_tags = np.unique(cell_markers.values)
sigma = Function(Q_sigma)

# assign conductivity (cell)
sigma.x.array[:] = tube_perm
for tag in material_tags:
    cells = cell_markers.find(tag)
    if tag == object_marker:
        sigma_ = object_cond_cond
    elif tag == medium_marker:
        sigma_ = tube_cond
    else:
        sigma_ = tube_cond

    sigma.x.array[cells] = np.full_like(cells, sigma_, dtype=ScalarType)

domain.topology.create_connectivity(tdim, fdim)

# define boundary conditions
r, i = 10 * np.cos(omega), 10 * np.sin(omega) * 1j
v_pos = Constant(domain, ScalarType(r + i))
r, i = 10 * np.cos(omega - np.pi), 10 * np.sin(omega - np.pi) * 1j
v_neg = Constant(domain, ScalarType(r + i))


# solve problem
u = TrialFunction(V)
v = TestFunction(V)

# find dof
pos_facet = facet_markers.find(pos_marker)
neg_facet = facet_markers.find(neg_marker)
dof_positive = locate_dofs_topological(V, fdim, pos_facet)
dof_negative = locate_dofs_topological(V, fdim, neg_facet)

# assign conditions
bcs_pos = dirichletbc(v_pos, dof_positive, V)
bcs_neg = dirichletbc(v_neg, dof_negative, V)
bcs = [bcs_pos, bcs_neg]

# piece-wise conductivity
a = sigma * inner(grad(u), grad(v)) * dx
L = Constant(domain, 0 + 0j) * conj(v) * dx

uh = Function(V)
problem = LinearProblem(a, L, u=uh, bcs=bcs)
problem.solve()


# generate electric field
E_field = -nabla_grad(uh)

V_E_field = functionspace(domain, ("Lagrange", 1, (2,)))
E_field_expr = Expression(E_field, V_E_field.element.interpolation_points())
E_field_projected = Function(V_E_field)
E_field_projected.interpolate(E_field_expr)
E_field_projected.name = "Electric field"
uh.name = "Electric potential"

# export solution
print("Export the solution to file...")
with XDMFFile(domain.comm, file_path + ".xdmf", "w") as xdmf:
    domain.name = 'cylinder geometry'
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers, domain.geometry)
    xdmf.write_meshtags(facet_markers, domain.geometry)
    xdmf.write_function(uh)
    xdmf.write_function(E_field_projected)


