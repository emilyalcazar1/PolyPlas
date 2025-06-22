import numpy as np
import pytest
import src.polymesher.PolyGeometryAndBCs as polygeometry
import src.polymesher.PolyMesher as polymesher
import src.polytop.PolyTopUtilities as polytop_utils
import src.polyplas.PolyPlasUtilities as polyplas_utils

@pytest.fixture(scope="session")
def boundary_value_problem():
    return polygeometry.CantileverBeam(x1=0.0, x2=3.0, y1=0.0, y2=1.0, applied_displacement_magnitude=1.0)


@pytest.fixture(scope="session")
def polymesh_and_precomputed_data(boundary_value_problem):
    polymesh_object = polymesher.PolyMesher(
        boundary_value_problem,
        number_of_elements=100,
        maximum_iterations=20
    )
    polymesh_object.run()
    # polymesh_object.plot()
    # import matplotlib.pyplot as plt
    # plt.show()
    precomputed_data = polytop_utils.get_precomputed_data(
        polymesh_object,
        elastic_modulus=100.0,
        poissons_ratio=0.25
    )
    return polymesh_object, precomputed_data

@pytest.fixture(scope="session")
def polymesh_and_polyplas_precomputed_data(boundary_value_problem):
    polymesh_object = polymesher.PolyMesher(
        boundary_value_problem,
        number_of_elements=20,
        maximum_iterations=20
    )
    polymesh_object.run()
    
    plasticity_parameters = dict(elastic_modulus=74633.0,
                        poissons_ratio = 0.3,
                        hardening_modulus = 2000.0,
                        initial_yield_stress = 344.0) 
    
    polyplas_precomputed_data = polyplas_utils.get_precomputed_data_filter_and_heaviside_projection(
        polymesh_object,
        plasticity_parameters
    )
    return polymesh_object, polyplas_precomputed_data, plasticity_parameters


@pytest.fixture(scope="session")
def regularmesh_and_polyplas_precomputed_data(boundary_value_problem):
    # Specify regular element seeds to get quadrilaterals
    x = np.linspace(0.0, 3.0, 4)
    x = 0.5 * np.diff(x) + x[:-1]
    y = np.linspace(0.0, 1.0, 2)
    y = 0.5 * np.diff(y) + y[:-1]
    X, Y = np.meshgrid(x, y)
    regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
    polymesh_object = polymesher.PolyMesher(
        boundary_value_problem,
        maximum_iterations=1,
        element_center_points=regular_grid_element_centroids
    )
    polymesh_object.run()
    

    plasticity_parameters = dict(elastic_modulus=74633.0,
                        poissons_ratio = 0.3,
                        hardening_modulus = 2000.0,
                        initial_yield_stress = 344.0) 
    
    polyplas_precomputed_data = polyplas_utils.get_precomputed_data_filter_and_heaviside_projection(
        polymesh_object,
        plasticity_parameters
    )

    return polymesh_object, polyplas_precomputed_data

@pytest.fixture(scope="session")
def square_mesh_and_polyplas_precomputed_data():
    #######################################################################################################################
    #######################################################################################################################
    class UnitSquare(polygeometry.BoundaryValueProblem):
        '''
        Class for defining a simple unit square domain for testing purposes.
        '''
        def __init__(self, load_magnitude: float = 1.0):
            super().__init__(applied_displacement_magnitude=load_magnitude)
            self.rectangle = polygeometry.Rectangle(x1=0.0, x2=1.0, y1=0.0, y2=1.0)

        def get_bounding_box(self) -> np.ndarray:
            return self.rectangle.get_bounding_box()

        def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
            return self.rectangle.distance_function(point_coordinates)

        def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> polygeometry.BoundaryConditions:
            x_coordinates = nodal_coordinates[:, 0]
            y_coordinates = nodal_coordinates[:, 1]
            bounding_box = self.get_bounding_box()
            bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
            number_of_nodes = float(nodal_coordinates.shape[0])
            nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

            left_edge = np.argwhere(x_coordinates < nodal_tolerance).ravel()
            right_edge = np.argwhere(x_coordinates > (1.0 - nodal_tolerance)).ravel()
            node_upper_right_edge = np.argwhere((x_coordinates > (1.0 - nodal_tolerance)).ravel()\
                                                 & (y_coordinates > (1.0 - nodal_tolerance)).ravel())

            bottom_edge = np.argwhere(y_coordinates < nodal_tolerance).ravel()

            number_of_loaded_nodes = node_upper_right_edge.size
            displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
            displacement_components[:, 0] = self.applied_displacement_magnitude 

            applied_displacement_1 = polygeometry.AppliedDisplacement(nodal_indices=node_upper_right_edge,
                                                      applied_displacement_components=displacement_components)
            boundary_conditions = polygeometry.BoundaryConditions(fixed_x_nodal_indices=left_edge,
                                                                  fixed_y_nodal_indices=bottom_edge,
                                                                  applied_displacements=[applied_displacement_1])
            return boundary_conditions
        
    number_of_elements = 16
    boundary_value_problem = UnitSquare()
    boundary_value_problem.applied_displacement_magnitude = 0.01
    polymesh_object = polymesher.PolyMesher(
        boundary_value_problem,
        number_of_elements = number_of_elements,
        maximum_iterations=30,
        use_regular_grid_if_implemented=False,
    )
    polymesh_object.run()
    polymesh_object.plot()
    import matplotlib.pyplot as plt
    plt.show()
    plasticity_parameters = dict(elastic_modulus=10000.0,
                        poissons_ratio = 0.3,
                        hardening_modulus = 100.0,
                        initial_yield_stress = 10.0) 
    
    polyplas_precomputed_data = polyplas_utils.get_precomputed_data_filter_and_heaviside_projection(
        polymesh_object,
        plasticity_parameters)
    return polymesh_object, polyplas_precomputed_data

@pytest.fixture(scope="session")
def square_mesh_and_polyplas_precomputed_data_and_material_dict():
    #######################################################################################################################
    #######################################################################################################################
    class UnitSquare(polygeometry.BoundaryValueProblem):
        '''
        Class for defining a simple unit square domain for testing purposes.
        '''
        def __init__(self, load_magnitude: float = 1.0):
            super().__init__(applied_displacement_magnitude=load_magnitude)
            self.rectangle = polygeometry.Rectangle(x1=0.0, x2=1.0, y1=0.0, y2=1.0)

        def get_bounding_box(self) -> np.ndarray:
            return self.rectangle.get_bounding_box()

        def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
            return self.rectangle.distance_function(point_coordinates)

        def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> polygeometry.BoundaryConditions:
            x_coordinates = nodal_coordinates[:, 0]
            y_coordinates = nodal_coordinates[:, 1]
            bounding_box = self.get_bounding_box()
            bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
            number_of_nodes = float(nodal_coordinates.shape[0])
            nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

            left_edge = np.argwhere(x_coordinates < nodal_tolerance).ravel()
            right_edge = np.argwhere(x_coordinates > (1.0 - nodal_tolerance)).ravel()
            node_upper_right_edge = np.argwhere((x_coordinates > (1.0 - nodal_tolerance)).ravel()\
                                                 & (y_coordinates > (1.0 - nodal_tolerance)).ravel())

            bottom_edge = np.argwhere(y_coordinates < nodal_tolerance).ravel()

            number_of_loaded_nodes = node_upper_right_edge.size
            displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
            displacement_components[:, 0] = self.applied_displacement_magnitude 

            applied_displacement_1 = polygeometry.AppliedDisplacement(nodal_indices=node_upper_right_edge,
                                                      applied_displacement_components=displacement_components)
            boundary_conditions = polygeometry.BoundaryConditions(fixed_x_nodal_indices=left_edge,
                                                                  fixed_y_nodal_indices=bottom_edge,
                                                                  applied_displacements=[applied_displacement_1])
            return boundary_conditions
        
    number_of_elements = 16
    boundary_value_problem = UnitSquare()
    boundary_value_problem.applied_displacement_magnitude = 0.01
    polymesh_object = polymesher.PolyMesher(
        boundary_value_problem,
        number_of_elements = number_of_elements,
        maximum_iterations=30,
        use_regular_grid_if_implemented=False,
    )
    polymesh_object.run()
    polymesh_object.plot()
    import matplotlib.pyplot as plt
    plt.show()
    plasticity_parameters = dict(elastic_modulus=10000.0,
                        poissons_ratio = 0.3,
                        hardening_modulus = 100.0,
                        initial_yield_stress = 10.0) 
    
    polyplas_precomputed_data = polyplas_utils.get_precomputed_data_filter_and_heaviside_projection(
        polymesh_object,
        plasticity_parameters)
    return polymesh_object, polyplas_precomputed_data, plasticity_parameters


@pytest.fixture(scope="session")
def square_mesh_and_polyplas_precomputed_data_uniaxial_deformation():
    #######################################################################################################################
    #######################################################################################################################
    class UnitSquare(polygeometry.BoundaryValueProblem):
        '''
        Class for defining a simple unit square domain for testing purposes.
        '''
        def __init__(self, displacement_magnitude: float = 1.0):
            super().__init__(applied_displacement_magnitude=displacement_magnitude)
            self.rectangle = polygeometry.Rectangle(x1=0.0, x2=1.0, y1=0.0, y2=1.0)

        def get_bounding_box(self) -> np.ndarray:
            return self.rectangle.get_bounding_box()

        def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
            return self.rectangle.distance_function(point_coordinates)

        def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> polygeometry.BoundaryConditions:
            x_coordinates = nodal_coordinates[:, 0]
            y_coordinates = nodal_coordinates[:, 1]
            bounding_box = self.get_bounding_box()
            bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
            number_of_nodes = float(nodal_coordinates.shape[0])
            nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

            left_edge = np.argwhere(x_coordinates < nodal_tolerance).ravel()
            right_edge = np.argwhere(x_coordinates > (1.0 - nodal_tolerance)).ravel()
            node_upper_right_edge = np.argwhere((x_coordinates > (1.0 - nodal_tolerance)).ravel()\
                                                 & (y_coordinates > (1.0 - nodal_tolerance)).ravel())

            bottom_edge = np.argwhere(y_coordinates < nodal_tolerance).ravel()

            number_of_loaded_nodes = right_edge.size
            displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
            displacement_components[:, 0] = self.applied_displacement_magnitude

            applied_displacement_1 = polygeometry.AppliedDisplacement(nodal_indices=right_edge,
                                                      applied_displacement_components=displacement_components)
            boundary_conditions = polygeometry.BoundaryConditions(fixed_x_nodal_indices=left_edge,
                                                                  fixed_y_nodal_indices=bottom_edge,
                                                                  applied_displacements=[applied_displacement_1])
            return boundary_conditions

    regular_grid_element_centroids = np.array([[0.5, 0.5]]) #single element
    boundary_value_problem = UnitSquare()
    boundary_value_problem.applied_displacement_magnitude = 0.05
    polymesh_object = polymesher.PolyMesher(
        UnitSquare(),
        number_of_elements=10,
        maximum_iterations=30,
        element_center_points=regular_grid_element_centroids
    )
    polymesh_object.run()
    polymesh_object.plot()
    import matplotlib.pyplot as plt
    plt.show()
    plasticity_parameters = dict(elastic_modulus=10000.0,
                        poissons_ratio = 0.3,
                        hardening_modulus = 100.0,
                        initial_yield_stress = 10.0) 
    
    polyplas_precomputed_data = polyplas_utils.get_precomputed_data_filter_and_heaviside_projection(
        polymesh_object,
        plasticity_parameters)
    return polymesh_object, polyplas_precomputed_data



