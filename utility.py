
from mpi4py import MPI
from dolfinx.io import gmshio
import numpy as np
import gmsh
import os
from dolfinx.mesh import Mesh, MeshTags, locate_entities_boundary
from dolfinx.io.utils import XDMFFile
from ufl import TestFunction, TrialFunction, dot, dx, nabla_grad, curl, inner, conj, ds, inv
from dolfinx.fem import (Constant, Function, functionspace, Expression,
                         dirichletbc, locate_dofs_topological, DirichletBC, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, locate_entities_boundary
from petsc4py.PETSc import ScalarType
from dolfinx import geometry
from typing import Tuple, List
from visualisation import *
import pandas as pd
import math


tube_height, tube_radius = 500, 50
x_tube, y_tube, z_tube, dx_tube, dy_tube, dz_tube = 0, 0, 0, 0, 0, tube_height

tube_cond = 1
pos_tag, neg_tag, medium_tag, obj_tag = 1, 3, 5, 7
mesh_resolution = 1

def get_mesh_cylinder(mesh_size, object: Tuple[Tuple[float, float, float], float] = None, save_mesh: str = None):

    if save_mesh is not None and os.path.exists(save_mesh):
        print("Loading existing mesh from file:", save_mesh)
        domain, cell_markers, facet_markers = gmshio.read_from_msh(save_mesh, MPI.COMM_WORLD, gdim=3)
        return domain, cell_markers, facet_markers

    gmsh.initialize()
    model = gmsh.model()
    model.add('main_mesh')
    model.setCurrent('main_mesh')

    tube = model.occ.addCylinder(x_tube, y_tube, z_tube, dx_tube, dy_tube, tube_height, tube_radius)
    if object is not None:
        obj1 = model.occ.addSphere(object[0][0], object[0][1], object[0][2], object[1])
        objects = [(3, obj1)]
        ov, ovv = model.occ.fragment([(3, tube)], objects)
        model.occ.synchronize()
        model.addPhysicalGroup(3, [ov[1][1]], tag=medium_tag, name="medium")
        model.addPhysicalGroup(3, [ov[0][1]], tag=obj_tag, name="obj1")
    else:
        model.occ.synchronize()
        model.addPhysicalGroup(3, [tube], tag=medium_tag, name="medium")

    surface_medium = []
    surface_pos = []
    surface_neg = []

    for surface in model.getEntities(dim=2):
        com = model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [0, 0, 0]):
            model.addPhysicalGroup(2, [surface[1]], tag=neg_tag, name="bottom_disk")
            surface_neg.append(surface)
        elif np.allclose(com, [0, 0, tube_height]):
            model.addPhysicalGroup(2, [surface[1]], tag=pos_tag, name="top_disk")
            surface_pos.append(surface)
        elif np.allclose(com, [0, 0, tube_height / 2]):
            model.addPhysicalGroup(2, [surface[1]], tag=medium_tag, name="side_surface")
            surface_medium.append(surface)

    model.occ.removeAllDuplicates()
    model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    model.mesh.generate(3)
    model.mesh.optimize("Netgen")
    if save_mesh is not None:
        gmsh.write(save_mesh)
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    print("Writing new mesh file:", save_mesh)
    domain, ct, ft = gmshio.model_to_mesh(model, mesh_comm, gmsh_model_rank, gdim=3)
    gmsh.finalize()

    return domain, ct, ft


def get_field(conductivity, domain, ct, ft):

    if conductivity == 'conductor':
        object_cond = 1e8
    if conductivity == 'insulator':
        object_cond = 1e-12
    if conductivity == 'same':
        object_cond = 1

    V = functionspace(domain, ("Lagrange", 1))

    # Define the material parameters
    tdim = domain.topology.dim
    fdim = tdim - 1

    Q_cond = functionspace(domain, ("DG", 0))
    material_tags = np.unique(ct.values)
    cond = Function(Q_cond)
    # As we only set some values in eps, initialize all as vacuum

    for tag in material_tags:
        cells = ct.find(tag)
        if tag == obj_tag:
            cond_ = object_cond
        else:
            cond_ = tube_cond
        cond.x.array[cells] = np.full_like(cells, cond_, dtype=ScalarType)

    domain.topology.create_connectivity(tdim, fdim)

    # Set the boundary conditions
    # The top boundary is set to 10V
    frequency = 10e3
    omega = 2 * np.pi * frequency    # frequency = 10kHz
    r, i = 10 * np.cos(omega), 10 * np.sin(omega) * 1j
    v_pos = Constant(domain, ScalarType(r + i))
    r, i = 10 * np.cos(omega - np.pi), 10 * np.sin(omega - np.pi) * 1j
    v_neg = Constant(domain, ScalarType(r + i))


    pos_facet = ft.find(pos_tag)
    neg_facet = ft.find(neg_tag)
    dof_positive = locate_dofs_topological(V, fdim, pos_facet)
    dof_negative = locate_dofs_topological(V, fdim, neg_facet)

    # assign conditions
    bcs_pos = dirichletbc(v_pos, dof_positive, V)
    bcs_neg = dirichletbc(v_neg, dof_negative, V)
    bcs = [bcs_pos, bcs_neg]


    u = TrialFunction(V)
    v = TestFunction(V)

    # Construct the PDE and solve for the electric potential
    a = inner(nabla_grad(u), nabla_grad(v)) * cond * dx
    L = Constant(domain, 0 + 0j) * conj(v) * dx
    # Solve the linear problem
    uh = Function(V)
    options = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = LinearProblem(a, L, u=uh, bcs=bcs, petsc_options=options)
    print("Now solving the electric potential PDE...")
    problem.solve()
    solver = problem.solver
    E_field = -nabla_grad(uh)
    J_field = cond * E_field

    # project to mesh
    V_E_field = functionspace(domain, ("Lagrange", 1, (3,)))
    V_J_field = functionspace(domain, ("Lagrange", 1, (3,)))
    E_field_expr = Expression(E_field, V_E_field.element.interpolation_points())
    J_field_expr = Expression(J_field, V_J_field.element.interpolation_points())
    E_field_projected = Function(V_E_field)
    J_field_projected = Function(V_J_field)
    E_field_projected.name = 'Electric_Field'
    E_field_projected.interpolate(E_field_expr)
    J_field_projected.interpolate(J_field_expr)
    J_field_projected.name = 'Current_Field'
    uh.name = "Electric potential"

    Potential_abs = Function(V, dtype=float)
    Potential_angle = Function(V, dtype=float)
    Potential_abs.name = 'Electric potential magnitude'
    Potential_angle.name = 'Electric potential phase'
    E_field_abs = Function(V_E_field, dtype=float)
    E_field_angle = Function(V_E_field, dtype=float)
    E_field_abs.name = 'Electric field magnitude'
    E_field_angle.name = 'Electric field phase'
    J_field_abs = Function(V_J_field, dtype=float)
    J_field_angle = Function(V_J_field, dtype=float)
    J_field_abs.name = 'Current field magnitude'
    J_field_angle.name = 'Current field phase'
    Potential_abs.x.array[:] = np.abs(uh.x.array)
    Potential_angle.x.array[:] = np.angle(uh.x.array, deg=True)
    E_field_abs.x.array[:] = np.abs(E_field_projected.x.array)
    E_field_angle.x.array[:] = np.angle(E_field_projected.x.array, deg=True)
    J_field_abs.x.array[:] = np.abs(J_field_projected.x.array)
    J_field_angle.x.array[:] = np.angle(J_field_projected.x.array, deg=True)

    return uh, E_field_abs, J_field_abs


def save_solution(domain, ct, ft, uh, E_field_projected, J_field_projected, conductivity, mesh_size, object: Tuple[Tuple[float, float, float], float] = None):

    if object is not None:
        file_name = "{p}_sphere_{i}_{j}_{k}_{q}_{min}".format(p=conductivity, i=object[0][0], j=object[0][1], k=object[0][2], q=object[1], min=mesh_size)
    else:
        file_name = "no_object_{}".format(mesh_size)

    # Export solution
    print("Export the solution to file...")
    with XDMFFile(domain.comm, file_name + ".xdmf", "w") as xdmf:
        domain.name = '3D Tube'
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(ct, domain.geometry)
        xdmf.write_meshtags(ft, domain.geometry)
        uh.name = "Electric potential"
        xdmf.write_function(uh)
        E_field_projected.name = "Electric field"
        xdmf.write_function(E_field_projected)
        J_field_projected.name = "Current Density"
        xdmf.write_function(J_field_projected)


def get_solution_disk(mesh_size, conductivity, object: Tuple[Tuple[float, float, float], float] = None):
    if object is not None:
        mesh_name = "disk_{}_{}_{}_{}_{}.msh".format(object[0][0], object[0][1], object[0][2], object[1], mesh_size)
        domain, cell_markers, facet_markers = get_mesh_cylinder(mesh_size, object, save_mesh=mesh_name)
        uh, E_field_projected, points_on_proc, u_values = get_field(conductivity, domain, cell_markers, facet_markers)
        save_solution(domain, cell_markers, facet_markers, uh, E_field_projected, conductivity, mesh_size, object)
    else:
        mesh_name = "no_object_{}.msh".format(mesh_size)
        domain, cell_markers, facet_markers = get_mesh_cylinder(mesh_size, object, save_mesh=mesh_name)
        uh, E_field_projected, points_on_proc, u_values = get_field(conductivity, domain, cell_markers, facet_markers)
        save_solution(domain, cell_markers, facet_markers, uh, E_field_projected, conductivity, mesh_size, object)


    return points_on_proc, u_values


def single_obj(n, conductivity, no_object: bool, difference: bool, plot:bool):
    mesh_size = 5

    for i in range(n):

        obj_r = np.random.uniform(5, 25)  # sphere radius varies between 5mm to 25mm
        obj_i = np.random.uniform(0, 360)  # location varies between 0 to 360 degrees
        obj_angle = np.pi * obj_i / 180

        obj_dist = (tube_radius - obj_r - 0.1) * math.sqrt(np.random.uniform(0.01, 1))
        xc = obj_dist * np.cos(obj_angle)
        yc = obj_dist * np.sin(obj_angle)
        zc = np.random.uniform(170, 330)  # correspond to -80 and + 80


        obj_pos = None if no_object else ((xc, yc, zc), obj_r)
        #mesh_name = "{}_sphere_{}_{}_{}_{}.msh".format(i, round(obj_r, 2), round(obj_dist, 2), round(obj_angle, 3), round(zc, 2))
        domain, cell_markers, facet_markers = get_mesh_cylinder(mesh_size, obj_pos)
        generate_data(i, domain, cell_markers, facet_markers, conductivity, obj_r, obj_i, obj_dist, zc, get_difference=difference, get_plot=plot)



def generate_data(id, domain, cell_markers, facet_markers, conductivity, obj_radius, obj_angle, obj_dist, obj_z,
                  get_difference: bool = True, get_plot: bool = False):
    if get_difference:
        uh, E_field_projected, J_field_projected = get_field(conductivity, domain, cell_markers, facet_markers)
        uh_same, _, _ = get_field('same', domain, cell_markers, facet_markers)
    else:
        uh, _, _ = get_field(conductivity, domain, cell_markers, facet_markers)

    values_array = []
    for i in range(0, 16, 1):
        angle = np.pi * i * 22.5 / 180

        x_vals = np.ones(8) * (tube_radius - 0.1) * np.cos(angle)
        y_vals = np.ones(8) * (tube_radius - 0.1) * np.sin(angle)
        z_vals = np.linspace(170, 330, 8)
        coords = np.stack((x_vals, y_vals, z_vals)).T
        points, values = get_points(domain, uh, coords)
        _, values_same = get_points(domain, uh_same, coords)
        diff = values - values_same
        diff = np.real(diff) * 1000  # since we use mV
        values_array.append(diff)

    data = np.array(values_array)
    data_x = np.squeeze(data)
    data_y = np.array([[conductivity], [str(obj_radius)], [str(obj_angle)], [str(obj_dist)], [str(obj_z)]])
    data_y = data_y.reshape(1, 5)

    np.savetxt("{}_single_{}_sphere_x.csv".format(id, conductivity), data_x, delimiter=",")
    np.savetxt("{}_single_{}_sphere_y.csv".format(id, conductivity), data_y, delimiter=',', fmt='%s')

    if get_plot:
        fig = plt.figure(constrained_layout=True)
        x = np.linspace(-80, 80, 8)

        plt.plot(x, data[0, :], color='orange', label='k=0')
        plt.plot(x, data[1, :], color='red', label='k=1')
        plt.plot(x, data[2, :], color='blue', label='k=2')
        plt.plot(x, data[3, :], color='green', label='k=3')
        plt.plot(x, data[4, :], color='gray', label='k=4')
        plt.plot(x, data[5, :], color='brown', label='k=5')
        plt.plot(x, data[6, :], color='pink', label='k=6')
        plt.plot(x, data[7, :], color='purple', label='k=7')
        plt.plot(x, data[8, :], color='orange', label='k=8')
        plt.plot(x, data[9, :], color='red', label='k=9')
        plt.plot(x, data[10, :], color='blue', label='k=10')
        plt.plot(x, data[11, :], color='green', label='k=11')
        plt.plot(x, data[12, :], color='gray', label='k=12')
        plt.plot(x, data[13, :], color='brown', label='k=13')
        plt.plot(x, data[14, :], color='pink', label='k=14')
        plt.plot(x, data[15, :], color='purple', label='k=15')

        if get_difference:
            z = obj_z - 250
            title = str(round(obj_radius, 2)) + 'mm ' + str(conductivity) + ' sphere (r=' + str(
                round(obj_dist, 3)) + ', a=' + str(round(obj_angle, 3)) + ', z=' + str(round(z, 3)) + ')'
            plt.title(title)
            plt.ylabel("Potential Difference (mV)")
            plt.xlabel("z-coordinate (mm)")
            plt.legend()
            plt.savefig('sphere_' + str(id) + '.png')
        else:
            title = str(round(obj_radius, 2)) + 'mm ' + str(conductivity) + ' sphere (r=' + str(
                round(obj_dist, 3)) + ', a=' + str(round(obj_angle, 3)) + ', z=' + str(round(obj_z, 3)) + ')'
            plt.title(title)
            plt.ylabel("Absolute Potential (mV)")
            plt.xlabel("z-coordinate (mm)")
            plt.legend()
            plt.savefig(str(id) + '.png')



def get_points(domain, function, coords):

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_mesh = []
    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, coords)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, coords)
    for i, point in enumerate(coords):
        if len(colliding_cells.links(i)) > 0:
            points_on_mesh.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_mesh = np.array(points_on_mesh, dtype=np.float64)
    values = function.eval(points_on_mesh, cells)

    return points_on_mesh, values


def get_noise(n:int, scale):

    #np.random.seed(57)
    data = np.zeros((n, 16, 8))         # surface potential for undisturbed case is 0V

    array = []
    for i in range(data.shape[0]):
        x = data[i]
        #sd = np.random.uniform(a, b)
        noise = np.random.normal(0, 1, x.shape)
        noise_scaled = scale * noise
        x_noise = x + noise_scaled
        array.append(x_noise)

    array = np.array(array)
    word_array = np.array(['undisturbed'], dtype='<U11')
    data_y = np.tile(word_array, (500, 5))

    return array, data_y

def add_noise(data, scale):

    #np.random.seed(57)
    array = []
    noise_only = []

    for i in range(data.shape[0]):
        x = data[i]
        noise = np.random.normal(0, 1, x.shape)
        noise_scaled = scale * noise
        x_noise = x + noise_scaled
        array.append(x_noise)
        noise_only.append(noise_scaled)

    noise_only = np.array(noise_only)
    noise_data = np.array(array)

    return noise_only, noise_data



if __name__ == "__main__":
    no_object = False
    conductivity = 'insulator'
    single_obj(2, conductivity, False, True, False)

    conductivity = 'conductor'
    single_obj(2, conductivity, False, True, False)





