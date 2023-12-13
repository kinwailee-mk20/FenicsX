from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.io.utils import VTKFile
import gmsh
import numpy as np
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc, locate_dofs_topological, locate_dofs_geometrical)
from petsc4py.PETSc import ScalarType
from dolfinx import mesh
from ufl import TestFunction, TrialFunction, dot, dx, grad
from dolfinx.fem.petsc import LinearProblem

R = 5 # radius of tube
r = 0.25 # radius of electrode

gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)

# Create model
main_model = gmsh.model()
main_model.add('main_mesh')
main_model.setCurrent('main_mesh')

# add 2D tube
pos_term = gmsh.model.occ.addDisk(0, 0, 0, R, R, tag=1) # positive terminal


# add -ve terminal
neg_term = gmsh.model.occ.addDisk(0, 0, 0, r, r, tag=2) # negative terminal

main_model.occ.fragment(gmsh.model.occ.getEntities(3), [])
space = gmsh.model.occ.cut([(2, pos_term)], [(2, neg_term)])
# syncrhonise
gmsh.model.occ.synchronize()

# boundary conditions
gmsh.model.addPhysicalGroup(1, [1], name = 'tube')
gmsh.model.addPhysicalGroup(2, [1], name = 'medium')
gmsh.model.addPhysicalGroup(1, [2], name = 'negative')
gmsh.model.addPhysicalGroup(2, [2], name = 'empty')


# generate a 2D mesh
gmsh.model.mesh.generate(2)

# model to mesh
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(main_model, mesh_comm, gmsh_model_rank, gdim=2)
gmsh.finalize()
# tdim = domain.topology.dim
# fdim = tdim - 1
# domain.topology.create_connectivity(tdim, fdim)

# The function space for the boundary conditions and the electric potential
V = FunctionSpace(domain, ("P", 1))

v_1 = ScalarType(10)
v_0 = ScalarType(0)

# define boundary on circles
def on_positive(x):
  return np.isclose(np.sqrt(x[0]**2 + x[1]**2), R)

def on_negative(x):
  return np.isclose(np.sqrt(x[0]**2 + x[1]**2), r)

positive_bc = locate_dofs_geometrical(V, on_positive)
negative_bc = locate_dofs_geometrical(V, on_negative)

bcs_pos = dirichletbc(v_1, positive_bc, V)
bcs_neg = dirichletbc(v_0, negative_bc, V)
bcs = [bcs_pos, bcs_neg]

# set u and v
u = TrialFunction(V)
v = TestFunction(V)

# solve
a = dot(grad(u), grad(v)) * dx
L = Constant(domain, ScalarType(0)) * v * dx
uh = Function(V)
problem = LinearProblem(a, L, u=uh, bcs=bcs)
problem.solve()

with VTKFile(MPI.COMM_WORLD, "new_model.bp", "w") as vtk:
    vtk.write_mesh(domain)
    vtk.write_function(uh)
