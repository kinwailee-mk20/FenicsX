from dolfinx.io.utils import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import (Constant, Function, FunctionSpace, Expression, VectorFunctionSpace,
                         dirichletbc, locate_dofs_topological, locate_dofs_geometrical)
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from dolfinx.io import gmshio
import numpy as np
import gmsh
from dolfinx import geometry
import matplotlib.pyplot as plt
import os
from ufl import TestFunction, TrialFunction, dot, dx, nabla_grad, curl, inner, conj, grad
from dolfinx import geometry

w_tube, h_tube = 0.5, 0.1

x_tube, y_tube = 0, 0

# conductivity (S/m)
# conductor: 1.12e7
# insulator: 1e-14
# milk: 1
tube_cond = 1

# com_tube
x_com_centre_tube, y_com_centre_tube = x_tube + (w_tube / 2), y_tube + (h_tube / 2)  # A
x_com_left_tube, y_com_left_tube = x_tube, y_tube + (h_tube / 2)  # B
x_com_right_tube, y_com_right_tube = x_tube + w_tube, y_tube + (h_tube / 2)  # C
x_com_top_tube, y_com_top_tube = x_tube + (w_tube / 2), y_tube + h_tube  # D
x_com_bottom_tube, y_com_bottom_tube = x_tube + (w_tube / 2), y_tube  # E

# assign marker
neg_marker, pos_marker, box_marker, space_marker, medium_marker, object_marker = 1, 3, 5, 7, 9, 11

def get_mesh(r_obj, mesh_size, K, save_mesh: str = None):

    if save_mesh is not None and os.path.exists(save_mesh):
        print("Loading existing mesh from file:", save_mesh)
        domain, cell_markers, facet_markers = gmshio.read_from_msh(save_mesh, MPI.COMM_WORLD, gdim=2)
        return domain, cell_markers, facet_markers


    '''
    h_obj_square, w_obj_square = 0.01, 0.25
    x_obj_square = (w_tube / 2) - (w_obj_square / 2) + K
    y_obj_square = (h_tube / 2) - (h_obj_square / 2) - 0.02
    '''
    x_obj_disc = w_tube / 2 + K
    y_obj_disc = (h_tube / 2) - 0.020


    gmsh.initialize()
    model = gmsh.model()
    model.add('main_mesh')
    model.setCurrent('main_mesh')


    # define tube
    tube = model.occ.addRectangle(x_tube, y_tube, 0, w_tube, h_tube)
    model.occ.synchronize()

    #obj_1 = gmsh.model.occ.addRectangle(x_obj_square, y_obj_square, 0, w_obj_square, h_obj_square)
    obj_1 = model.occ.addDisk(x_obj_disc, y_obj_disc, 0, r_obj, r_obj)

    # conformal surface
    objects = [(2, obj_1)]
    model.occ.fragment([(2, tube)], objects)
    model.occ.synchronize()

    # find COM
    line_border = []
    line_pos_terminal = []
    line_neg_terminal = []
    line_object = []
    line_space = []

    surface_space = []
    surface_medium = []
    surface_object = []
    lines = model.occ.getEntities(dim=1)
    surfaces = model.occ.getEntities(dim=2)


    for line in lines:
        com = model.occ.getCenterOfMass(line[0], line[1])
        if np.allclose(com, [x_com_left_tube, y_com_left_tube, 0]):
            line_neg_terminal.append(line[1])
        elif np.allclose(com, [x_com_right_tube, y_com_right_tube, 0]):
            line_pos_terminal.append(line[1])
        elif np.allclose(com, [x_com_top_tube, y_com_top_tube, 0]) or np.allclose(com,
                                                                                  [x_com_bottom_tube, y_com_bottom_tube,
                                                                                   0]):
            line_border.append(line[1])
        else:
            line_object.append(line[1])

    # define physical group
    model.addPhysicalGroup(1, line_pos_terminal, tag=pos_marker, name='pos')
    model.addPhysicalGroup(1, line_neg_terminal, tag=neg_marker, name='neg')
    model.addPhysicalGroup(1, line_border, tag=medium_marker, name='medium')

    surface_object.append(surfaces[0][1])
    surface_medium.append(surfaces[1][1])

    model.addPhysicalGroup(2, surface_medium, tag=medium_marker, name='medium')
    model.addPhysicalGroup(2, surface_object, tag=object_marker, name='obj1')

    # define mesh sizes
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    model.occ.removeAllDuplicates
    model.mesh.removeDuplicateNodes
    model.mesh.removeDuplicateElements

    # generate mesh
    model.occ.synchronize()
    model.mesh.generate(2)
    model.mesh.optimize("Netgen")
    if save_mesh is not None:
        gmsh.write(save_mesh)
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    print("Writing new mesh file:", save_mesh)
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(model, mesh_comm, gmsh_model_rank, gdim=2)
    gmsh.finalize()
    
    return domain, cell_markers, facet_markers


def get_field(conductivity, domain, cell_markers, facet_markers):

    if conductivity == 'conductor':
        object_cond = 1e8
    if conductivity == 'insulator':
        object_cond = 1e-12
    if conductivity == 'same':
        object_cond = 1

    # define function space for electric potential
    V = FunctionSpace(domain, ("Lagrange", 1))

    # define function and function space for permittivity
    tdim = domain.topology.dim
    fdim = tdim - 1
    Q_sigma = FunctionSpace(domain, ("DG", 0))
    material_tags = np.unique(cell_markers.values)
    sigma = Function(Q_sigma)

    # assign conductivity (cell)
    sigma.x.array[:] = tube_cond
    for tag in material_tags:
        cells = cell_markers.find(tag)
        if tag == object_marker:
            sigma_ = object_cond
        elif tag == medium_marker:
            sigma_ = tube_cond
        else:
            sigma_ = tube_cond

        sigma.x.array[cells] = np.full_like(cells, sigma_, dtype=ScalarType)

    domain.topology.create_connectivity(tdim, fdim)

    # define boundary conditions
    v_1 = ScalarType(-10)
    v_0 = ScalarType(10)

    # solve problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # find dof
    pos_facet = facet_markers.find(pos_marker)
    neg_facet = facet_markers.find(neg_marker)
    dof_positive = locate_dofs_topological(V, fdim, pos_facet)
    dof_negative = locate_dofs_topological(V, fdim, neg_facet)

    # assign conditions
    bcs_neg = dirichletbc(v_1, dof_negative, V)
    bcs_pos = dirichletbc(v_0, dof_positive, V)
    bcs = [bcs_pos, bcs_neg]

    # piece-wise conductivity
    f = Constant(domain, 1e-16)
    a = dot(sigma * grad(u), grad(v)) * dx
    L = f * v * dx

    uh = Function(V)
    problem = LinearProblem(a, L, u=uh, bcs=bcs)
    problem.solve()

    # generate electric field
    E_field = -nabla_grad(uh)

    V_E_field = VectorFunctionSpace(domain, ("Lagrange", 1))
    E_field_expr = Expression(E_field, V_E_field.element.interpolation_points())
    E_field_projected = Function(V_E_field)
    E_field_projected.interpolate(E_field_expr)
    E_field_projected.name = "Electric field"
    uh.name = "Electric potential"

    tol = 0.001  # Avoid hitting the outside of the domain
    z = np.linspace(0 + tol, 0.5 - tol, 101)
    points = np.zeros((3, 101))
    points[0] = z
    u_values = []

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = uh.eval(points_on_proc, cells)

    return uh, E_field_projected, points_on_proc, u_values


def save_xdmf(domain, cell_markers, facet_markers, uh, E_field_projected, conductivity,r_obj, K, mesh_size):

    file_name = "{p}_disk_{i}_K{j}_{min}".format(p=conductivity, i=r_obj, j=K, min=mesh_size)
    print("Export the solution to file...")
    with XDMFFile(domain.comm, file_name + ".xdmf", "w") as xdmf:
        domain.name = '2D_electrode_ring'
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_markers, domain.geometry)
        xdmf.write_meshtags(facet_markers, domain.geometry)
        xdmf.write_function(uh)
        xdmf.write_function(E_field_projected)


def get_solution(r_obj, mesh_size, conductivity, K):
    mesh_name = "disk_{i}_K{j}_{min}.msh".format(i=r_obj, j=K, min=mesh_size)
    domain, cell_markers, facet_markers = get_mesh(r_obj, mesh_size, K, save_mesh=mesh_name)
    uh, E_field_projected, points_on_proc, u_values = get_field(conductivity, domain, cell_markers, facet_markers)
    save_xdmf(domain, cell_markers, facet_markers, uh, E_field_projected, conductivity,r_obj, K, mesh_size)

    return points_on_proc, u_values


if __name__ == "__main__":
    mesh_size = 0.005
    conductivity = 'conductor'
    K = 0
    r_obj = 0.005

    mesh_name = "disk_{i}_K{j}_{min}.msh".format(i=r_obj, j=K, min=mesh_size)
    domain, cell_markers, facet_markers = get_mesh(r_obj, mesh_size, K, save_mesh=mesh_name)
    uh, E_field_projected, points_on_proc, u_values = get_field(conductivity, domain, cell_markers, facet_markers)
    save_xdmf(domain, cell_markers, facet_markers, uh, E_field_projected, conductivity,r_obj, K, mesh_size)

    fig = plt.figure(constrained_layout=True)
    plt.plot(points_on_proc[:, 0] * 1000, u_values * 1000, "b", linewidth=2, label="conductor")


    plt.grid(True)
    plt.title("Conducting disk (5 mm * 5 mm) at (-20, 0)")
    plt.ylabel("Potential Difference (mV)")
    plt.xlabel("z-coordinate (mm)")
    plt.legend()
    plt.savefig(f"test_1.png")
    

