from mpi4py import MPI
from dolfinx.io import gmshio
import numpy as np
import gmsh
import os

# specified in mm
Tube_hight = 250
Tube_radius = 50

top_electrode_height = 230
bottom_electrode_height = 20
radius_tol = 5
# Tags: 
tube_interior_tag = 10 # 10 = tube interior
tube_sides_tag = 101 # 101 = tube sides

top_disk_tag = 102 # 102 = top disk
bottom_disk_tag = 100 # 100 = bottom disk

top_elec_1_tag = 201 # 201 = top electrode 1
top_elec_2_tag = 202 # 202 = top electrode 2
bottom_elec_1_tag = 301 # 201 = top electrode 1
bottom_elec_2_tag = 302 # 202 = top electrode 2

obj1_tag = 103 # 103 = obj1 (large perm)
obj2_tag = 104 # 104 = obj2 (small perm)

mesh_file = "mesh3D.msh"

def get_model_domain():
    """
    Builds a model and mesh based on a cylindrical geometry
    Returns: a tuple of (the domain mesh, cell tags, facet tags) 
    """
    # check if mesh3D.msh exists as file
    if os.path.exists(mesh_file):
        # load the mesh instead of creating it again.
        print("Loading existing mesh from file:", mesh_file)
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
        return mesh, cell_tags, facet_tags 
    
    gmsh.initialize()
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2)
    
    max_mesh = Tube_radius * 0.1
    min_mesh = Tube_radius * 0.05
    min_dist = Tube_radius * 0.1
    max_dist = Tube_radius * 0.2

    # Create model
    main_model = gmsh.model()
    main_model.add('main_mesh')
    main_model.setCurrent('main_mesh')

    cylinder = main_model.occ.addCylinder(0, 0, 0, 0, 0, Tube_hight, Tube_radius)
    # Add an object with large permitivity
    obj1 = main_model.occ.addSphere(0, 0, Tube_hight // 2 - 50, 5)
    # Add an object with large permitivity
   # obj2 = main_model.occ.addSphere(0, 0, Tube_hight // 2 + 50, 5)
    # Use fragment to make sure that all interfaces are conformal.
    main_model.occ.fragment(gmsh.model.occ.getEntities(3), [])
    main_model.occ.synchronize()
    '''
    important_areas = []
    important_surfaces = []
    # Find the volumes based on COM
    for vol in main_model.getEntities(dim=3):
            com = main_model.occ.getCenterOfMass(vol[0], vol[1])
            if np.allclose(com, [0, 0, Tube_hight // 2 - 50]):
                main_model.addPhysicalGroup(3, [vol[1]], tag=obj1_tag, name="obj1")
                important_areas.append(vol)
            elif np.allclose(com, [0, 0, Tube_hight // 2 + 50]):
                main_model.addPhysicalGroup(3, [vol[1]], tag=obj2_tag, name="obj2")
                important_areas.append(vol)
            else:
                main_model.addPhysicalGroup(3, [vol[1]], tag=tube_interior_tag, name="volume")
    for vol in main_model.getEntities(dim=2):
            com = main_model.occ.getCenterOfMass(vol[0], vol[1])
            if np.allclose(com, [0, 0, 0]):
                main_model.addPhysicalGroup(2, [vol[1]], tag=bottom_disk_tag, name="bottom_disk")
                important_surfaces.append(vol)
            elif np.allclose(com, [0, 0, Tube_hight]):
                main_model.addPhysicalGroup(2, [vol[1]], tag=top_disk_tag, name="top_disk")
                important_surfaces.append(vol)
            elif np.allclose(com, [0, 0, Tube_hight / 2]):
                main_model.addPhysicalGroup(2, [vol[1]], tag=tube_sides_tag, name="side_surface")
                important_surfaces.append(vol)
    '''

    # Add some spheres for illustrative purposes to show the location of the objects
    o1 = main_model.occ.addSphere(0, 0, Tube_hight // 2 - 50, 5)
    #o2 = main_model.occ.addSphere(0, 0, Tube_hight // 2 + 50, 5)
    main_model.occ.synchronize()
    main_model.addPhysicalGroup(3, [o1], tag=obj1_tag+50, name="obj1_phantom")
    #main_model.addPhysicalGroup(3, [o2], tag=obj2_tag+50, name="obj2_phantom")

    main_model.occ.removeAllDuplicates
    main_model.mesh.removeDuplicateNodes
    main_model.mesh.removeDuplicateElements

    main_model.occ.synchronize()
    main_model.mesh.generate(3) 
    main_model.mesh.optimize("Netgen")
    gmsh.write(mesh_file)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, ct, ft = gmshio.model_to_mesh(main_model, mesh_comm, gmsh_model_rank, gdim=3)
    gmsh.finalize()
    return domain, ct, ft
