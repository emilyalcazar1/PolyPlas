import logging
import os
import meshio
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, List, NamedTuple, Optional, Tuple
from time import perf_counter
from dataclasses import dataclass
from src.polymesher.PolyMesher import PolyMesher

#######################################################################################################################
#######################################################################################################################
class PolyPlasPrecomputedData(NamedTuple):
    '''This class is used to store precomputed data for PolyPlas.'''
    number_of_nodes: int
    number_of_elements: int
    number_of_dofs_per_node: int
    number_of_dofs_per_element_type: Dict[int, int]
    shear_modulus:float
    bulk_modulus:float
    hardening_modulus: float
    initial_yield_stress: float
    space_dimension: int
    number_of_global_dofs: int
    number_of_quad_points_per_element: Dict[int, int]
    global_dof_indices_elementwise: Dict[int, np.ndarray]
    delaunay_simplices: np.ndarray
    element_relative_volumes: np.ndarray
    element_volumes_per_elem_type: Dict[int, np.ndarray]
    nodal_coordinates: np.ndarray
    element_connectivity_arrays: Dict[int, np.ndarray]
    shape_function_values: Dict[int, np.ndarray]    
    shape_function_gradients: Dict[int, np.ndarray] 
    jacobian_determinant_x_quadrature_weights: Dict[int, np.ndarray]   
    node_to_element_map: sp.sparse.csr_matrix
    virtual_displacement_symmetric_gradients: Dict[int, np.ndarray]
    virtual_displacement_volume_weighted_divergence: Dict[int, np.ndarray]
    fixed_dof_indices: np.ndarray
    internal_force_indices_array: np.ndarray 
    unique_zero_dirichlet_dof_indices: np.ndarray 
    unique_nonzero_dirichlet_dof_indices: np.ndarray 
    unique_nonzero_dirichlet_dof_indices_x:  np.ndarray 
    unique_nonzero_dirichlet_dof_indices_y: np.ndarray
    coo_matrix_row_indices: np.ndarray
    coo_matrix_column_indices: np.ndarray 
    coo_matrix_all_dirichlet_dof_indices: np.ndarray 
    coo_matrix_unique_dirichlet_diagonal_dof_indices: np.ndarray 
    coo_matrix_nonzero_dirichlet_dof_indices: np.ndarray
    coo_matrix_nonzero_dirichlet_dof_indices_x: np.ndarray
    coo_matrix_nonzero_dirichlet_dof_indices_y: np.ndarray
    coo_matrix_nonzero_dirichlet_indices: np.ndarray
    coo_matrix_nonzero_dirichlet_indices_x: np.ndarray
    coo_matrix_nonzero_dirichlet_indices_y: np.ndarray
    total_applied_displacement_x: float
    total_applied_displacement_y: float
    IxI: np.ndarray
    deviatoric_projection_tensor: np.ndarray
    volumetric_projection_tensor: np.ndarray
    linear_elastic_constitutive_tensor: np.ndarray
    fourth_order_symmetric_identity_tensor: np.ndarray
    passive_design_variable_indices: np.ndarray
    passive_local_element_indices: Dict[int, np.ndarray]
    passive_global_element_indices: np.ndarray

#######################################################################################################################
#######################################################################################################################
class currentPlasticState:
    def __init__(self):
        self.nodal_displacement_vector = None
        self.current_total_strain_tensors = None
        self.previous_total_strain_tensors = None
        self.current_plastic_strain_tensors = None
        self.previous_plastic_strain_tensors = None
        self.current_stress_tensors = None
        self.previous_stress_tensors = None
        self.current_deviatoric_stress_tensors = None
        self.previous_deviatoric_stress_tensors = None
        self.current_pressure = None
        self.current_accumulated_plastic_strains = None
        self.previous_accumulated_plastic_strains = None
        self.plastic_multiplier_increments = None

    def transfer_state(self, precomputed_data: PolyPlasPrecomputedData):
        unique_numbers_of_vertices = precomputed_data.element_connectivity_arrays.keys()
        for number_of_vertices in unique_numbers_of_vertices:
            self.previous_total_strain_tensors[number_of_vertices][:,:,:,:] = self.current_total_strain_tensors[number_of_vertices][:,:,:,:]
            self.previous_plastic_strain_tensors[number_of_vertices][:,:,:,:] = self.current_plastic_strain_tensors[number_of_vertices][:,:,:,:]
            self.previous_stress_tensors[number_of_vertices][:,:,:,:] = self.current_stress_tensors[number_of_vertices][:,:,:,:]
            self.previous_accumulated_plastic_strains[number_of_vertices][:,:] = self.current_accumulated_plastic_strains[number_of_vertices][:,:]
            self.previous_deviatoric_stress_tensors[number_of_vertices][:,:,:,:] = self.current_deviatoric_stress_tensors[number_of_vertices][:,:,:,:]

    def save_state(self, time_step_index: int):
        print('Saving state at time step:', {time_step_index})
        state_filename = f"state_at_time_step_{time_step_index:03d}.npz"
        state_file_directory = os.path.join(os.getcwd(), "saved_states")
        if not os.path.exists(state_file_directory):
            os.makedirs(state_file_directory)
        state_file_path = os.path.join(state_file_directory, state_filename)
        np.savez(state_file_path,
                 nodal_displacement_vector=self.nodal_displacement_vector,
                 current_total_strain_tensors=self.current_total_strain_tensors,
                 previous_total_strain_tensors=self.previous_total_strain_tensors,
                 current_plastic_strain_tensors=self.current_plastic_strain_tensors,
                 previous_plastic_strain_tensors=self.previous_plastic_strain_tensors,
                 current_stress_tensors=self.current_stress_tensors,
                 previous_stress_tensors=self.previous_stress_tensors,
                 current_deviatoric_stress_tensors=self.current_deviatoric_stress_tensors,
                 previous_deviatoric_stress_tensors=self.previous_deviatoric_stress_tensors,
                 current_pressure=self.current_pressure,
                 current_accumulated_plastic_strains=self.current_accumulated_plastic_strains,
                 previous_accumulated_plastic_strains=self.previous_accumulated_plastic_strains,
                 plastic_multiplier_increments=self.plastic_multiplier_increments)
        
    def load_state(self, time_step_index: int):
        self.state
    
#######################################################################################################################
#######################################################################################################################
@dataclass
class MMA_Optimizer_State:
    def __init__(self):
        self.current_design_variables = None
        self.previous_design_variables = None
        self.previous_previous_design_variables = None
        self.lower_asymptotes = None
        self.upper_asymptotes = None
        self.design_variable_lower_bounds = None
        self.design_variable_upper_bounds = None
        self.current_objective_scale_factor = None
        self.mma_move_limit = None

    def set_number_of_design_variables(self, number_of_design_variables: int, volume_fraction_upper_bound: float, 
                                       mma_move_limit: float = 0.15, passive_design_variable_indices: np.ndarray = None):
        if passive_design_variable_indices.size > 0:
            variables_for_optimization_mask = np.ones(number_of_design_variables, dtype=bool)
            variables_for_optimization_mask[passive_design_variable_indices] = False
            self.current_design_variables = volume_fraction_upper_bound * np.ones((number_of_design_variables, 1), dtype=float)[variables_for_optimization_mask]
            self.previous_design_variables = np.zeros((number_of_design_variables, 1), dtype=float)[variables_for_optimization_mask]
            self.previous_previous_design_variables = np.zeros((number_of_design_variables, 1), dtype=float)[variables_for_optimization_mask]
            self.lower_asymptotes = np.zeros((number_of_design_variables, 1), dtype=float)[variables_for_optimization_mask]
            self.upper_asymptotes = np.ones((number_of_design_variables, 1), dtype=float)[variables_for_optimization_mask]
            self.design_variable_lower_bounds = np.zeros((number_of_design_variables, 1), dtype=float)[variables_for_optimization_mask]
            self.design_variable_upper_bounds = np.ones((number_of_design_variables, 1), dtype=float)[variables_for_optimization_mask]
            self.mma_move_limit = mma_move_limit
        else:
            self.current_design_variables = volume_fraction_upper_bound * np.ones((number_of_design_variables, 1), dtype=float)
            self.previous_design_variables = np.zeros((number_of_design_variables, 1), dtype=float)
            self.previous_previous_design_variables = np.zeros((number_of_design_variables, 1), dtype=float)
            self.lower_asymptotes = np.zeros((number_of_design_variables, 1), dtype=float)
            self.upper_asymptotes = np.ones((number_of_design_variables, 1), dtype=float)
            self.design_variable_lower_bounds = np.zeros((number_of_design_variables, 1), dtype=float)
            self.design_variable_upper_bounds = np.ones((number_of_design_variables, 1), dtype=float)
            self.mma_move_limit = mma_move_limit

    def transfer_history(self, new_design_variables: np.ndarray):
        self.previous_previous_design_variables[:] = self.previous_design_variables[:]
        self.previous_design_variables[:] = self.current_design_variables[:]
        self.current_design_variables[:] = new_design_variables[:]

    def save_restart_file(self, iteration_number: int = 0):
        np.savez(f"mma_restart_iteration_{iteration_number:04d}.npz",
                 current_design_variables=self.current_design_variables,
                 previous_design_variables=self.previous_design_variables,
                 previous_previous_design_variables=self.previous_previous_design_variables,
                 lower_asymptotes=self.lower_asymptotes,
                 upper_asymptotes=self.upper_asymptotes,
                 current_objective_scale_factor=self.current_objective_scale_factor,
                 design_variable_lower_bounds=self.design_variable_lower_bounds,
                 design_variable_upper_bounds=self.design_variable_upper_bounds)

    def load_restart_file(self, iteration_number: int = 0):
        restart_filename = f"mma_restart_iteration_{iteration_number:04d}.npz"
        if not os.path.exists(restart_filename):
            raise IOError(f"The restart file '{restart_filename}' does not exist!")
        data = np.load(restart_filename)
        self.current_design_variables = data["current_design_variables"]
        self.previous_design_variables = data["previous_design_variables"]
        self.previous_previous_design_variables = data["previous_previous_design_variables"]
        self.lower_asymptotes = data["lower_asymptotes"]
        self.upper_asymptotes = data["upper_asymptotes"]
        self.current_objective_scale_factor = float(data["current_objective_scale_factor"])
        self.design_variable_lower_bounds = data["design_variable_lower_bounds"]
        self.design_variable_upper_bounds = data["design_variable_upper_bounds"]


#######################################################################################################################
#######################################################################################################################
class LumpedMassProjection(NamedTuple):
    '''This class is used to efficiently project element-based quantities to the nodes using a lumped mass matrix.'''
    right_hand_side_vector_indices: tuple[np.ndarray, np.ndarray]
    right_hand_side_vector_data: np.ndarray
    right_hand_side_vector_element_indices: np.ndarray
    lumped_mass_matrix_inverse: sp.sparse.csr_matrix

    def apply_projection(self, element_based_vector: np.ndarray) -> np.ndarray:
        '''This function applies the lumped mass projection to a vector of element-based scalars.

        Args:
            element_based_vector: A numpy array containing one scalar for each element in the finite element mesh.

        Returns:
            A numpy array containing the scalar values projected to the nodes of the finite element mesh.
        '''
        right_hand_side_data = \
            self.right_hand_side_vector_data * element_based_vector[self.right_hand_side_vector_element_indices]
        right_hand_side_vector = \
            sp.sparse.coo_matrix((right_hand_side_data, self.right_hand_side_vector_indices)).toarray().ravel()
        projected_data_vector = self.lumped_mass_matrix_inverse @ right_hand_side_vector
        return projected_data_vector


#######################################################################################################################
#######################################################################################################################
def setup_logger(logging_level=logging.INFO):
    '''This function sets up the logger for writing data to the console.

    Args:
        logging_level: The logging level as a logging.LEVEL constant.
            Options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL

    Returns:
        None
    '''
    logger = logging.getLogger("Poly")
    logger.setLevel(logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    first_line = "Logged Columns: Time (Hour:Minute:Second), Logger Name, Logging Level, Message"
    formatter = logging.Formatter('%(asctime)s, %(name)s, %(levelname)s, %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info(first_line)


#######################################################################################################################
#######################################################################################################################
def get_polygonal_triangulation(number_of_vertices: int,
                                point_in_natural_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angles = 2.0 * np.pi * np.arange(1.0, number_of_vertices + 1, dtype=float) / number_of_vertices
    x = np.concatenate(( np.cos(angles), np.array([point_in_natural_coordinates[0]]) ))
    y = np.concatenate(( np.sin(angles), np.array([point_in_natural_coordinates[1]]) ))
    natural_coordinates_of_vertices = np.column_stack((x, y))
    triangle_element_connectivity = np.zeros((number_of_vertices, 3), dtype=int)
    triangle_element_connectivity[:, 0] = number_of_vertices
    triangle_element_connectivity[:, 1] = np.arange(number_of_vertices, dtype=int)
    triangle_element_connectivity[:, 2] = np.arange(1, number_of_vertices + 1, dtype=int)
    triangle_element_connectivity[-1, 2] = 0
    return natural_coordinates_of_vertices, triangle_element_connectivity


#######################################################################################################################
#######################################################################################################################
def get_polygonal_element_quadrature(number_of_vertices: int, number_of_triangle_quadrature_points: int) -> Tuple[np.ndarray, np.ndarray]:

    if number_of_triangle_quadrature_points != 1 and number_of_triangle_quadrature_points !=3:
        raise ValueError(f"Number of quadrature points per triangle not supported. Please enter 1 or 3, user specified '{number_of_triangle_quadrature_points}.'")
    
    if number_of_triangle_quadrature_points == 3:
        one_sixth = 1.0 / 6.0
        two_thirds = 2.0 / 3.0
        triangle_quadrature_weights = np.array([one_sixth, one_sixth, one_sixth])
        triangle_quadrature_points = np.array([[one_sixth, one_sixth], [two_thirds, one_sixth], [one_sixth, two_thirds]])
    elif number_of_triangle_quadrature_points == 1:
        one_half = 1.0 / 2.0
        one_third = 1.0 / 3.0
        triangle_quadrature_weights = np.array([one_half])
        triangle_quadrature_points = np.array([one_third, one_third]).reshape((-1,2))
    triangle_nodal_coordinates, triangle_element_connectivity = \
        get_polygonal_triangulation(number_of_vertices, np.array([0.0, 0.0]))
    number_of_polygonal_quadrature_points = number_of_vertices * number_of_triangle_quadrature_points
    polygonal_quadrature_points  = np.zeros((number_of_polygonal_quadrature_points, 2), dtype=float)
    polygonal_quadrature_weights = np.zeros((number_of_polygonal_quadrature_points,),   dtype=float)
    for triangle_index in range(triangle_element_connectivity.shape[0]):
        triangle_nodal_indices = triangle_element_connectivity[triangle_index, :]
        for quadrature_point_index in range(number_of_triangle_quadrature_points):
            xi, eta = triangle_quadrature_points[quadrature_point_index, :]
            triangle_shape_function_values = np.array([1.0 - xi - eta, xi, eta])
            triangle_shape_function_gradients = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
            jacobian_of_the_mapping = \
                triangle_nodal_coordinates[triangle_nodal_indices, :].T @ triangle_shape_function_gradients
            index = triangle_index * number_of_triangle_quadrature_points + quadrature_point_index
            polygonal_quadrature_points[index, :] = \
                triangle_shape_function_values @ triangle_nodal_coordinates[triangle_nodal_indices, :]
            jacobian_determinant = np.linalg.det(jacobian_of_the_mapping)
            triangle_quadrature_weight = triangle_quadrature_weights[quadrature_point_index]
            polygonal_quadrature_weights[index] = jacobian_determinant * triangle_quadrature_weight
    return polygonal_quadrature_weights, polygonal_quadrature_points


#######################################################################################################################
#######################################################################################################################
def get_polygonal_shape_function_values_and_gradients(number_of_vertices: int,
                                                      point_in_natural_coordinates: np.ndarray) -> \
                                                      Tuple[np.ndarray, np.ndarray]:
    polygonal_shape_function_values = np.zeros((number_of_vertices), dtype=float)
    polygonal_shape_function_gradients = np.zeros((number_of_vertices, 2), dtype=float)
    alpha_values = np.zeros((number_of_vertices), dtype=float)
    alpha_gradients = np.zeros((number_of_vertices, 2), dtype=float)
    alpha_values_sum = 0.0
    alpha_gradients_sum = np.zeros((2), dtype=float)
    area_values = np.zeros((number_of_vertices), dtype=float)
    area_gradients = np.zeros((number_of_vertices, 2), dtype=float)
    triangle_nodal_coordinates, triangle_element_connectivity = \
        get_polygonal_triangulation(number_of_vertices, point_in_natural_coordinates)
    for triangle_index in range(triangle_element_connectivity.shape[0]):
        triangle_nodal_indices = triangle_element_connectivity[triangle_index, :]
        nodal_coordinates = triangle_nodal_coordinates[triangle_nodal_indices, :]
        area_values[triangle_index] = 0.5 * np.linalg.det(np.hstack((nodal_coordinates, np.ones((3, 1)))))
        area_gradients[triangle_index, 0] = 0.5 * (nodal_coordinates[2, 1] - nodal_coordinates[1, 1])
        area_gradients[triangle_index, 1] = 0.5 * (nodal_coordinates[1, 0] - nodal_coordinates[2, 0])
    area_values = np.concatenate((np.array([area_values[-1]]), area_values))
    area_gradients = np.vstack((area_gradients[-1, :], area_gradients))
    for triangle_index in range(triangle_element_connectivity.shape[0]):
        alpha_values[triangle_index] = 1.0 / (area_values[triangle_index] * area_values[triangle_index + 1])
        alpha_gradients[triangle_index, 0] = -1.0 * alpha_values[triangle_index] * \
            (area_gradients[triangle_index, 0] / area_values[triangle_index] +
             area_gradients[triangle_index + 1, 0] / area_values[triangle_index + 1])
        alpha_gradients[triangle_index, 1] = -1.0 * alpha_values[triangle_index] * \
            (area_gradients[triangle_index, 1] / area_values[triangle_index] +
             area_gradients[triangle_index + 1, 1] / area_values[triangle_index + 1])
        alpha_values_sum += alpha_values[triangle_index]
        alpha_gradients_sum += alpha_gradients[triangle_index, :].ravel()
    for vertex_index in range(number_of_vertices):
        N = alpha_values[vertex_index] / alpha_values_sum
        polygonal_shape_function_values[vertex_index] = N
        polygonal_shape_function_gradients[vertex_index, :] = \
            (alpha_gradients[vertex_index, :] - N * alpha_gradients_sum[:]) / alpha_values_sum
    return polygonal_shape_function_values, polygonal_shape_function_gradients


#######################################################################################################################
#######################################################################################################################
def get_shape_function_tables_and_quadrature_weights(unique_numbers_of_vertices: List[int], number_of_quadrature_points_per_triangle: int) -> \
                                                        Tuple[Dict[int, np.ndarray],
                                                              Dict[int, np.ndarray],
                                                              Dict[int, np.ndarray]]:
    shape_function_values = {}
    shape_function_gradients = {}
    quadrature_weights = {}
    for number_of_vertices in unique_numbers_of_vertices:
        polygonal_quadrature_weights, polygonal_quadrature_points = \
            get_polygonal_element_quadrature(number_of_vertices, number_of_quadrature_points_per_triangle)
        number_of_quadrature_points = polygonal_quadrature_weights.size
        quadrature_weights[number_of_vertices] = polygonal_quadrature_weights
        shape_function_values[number_of_vertices] = \
            np.zeros((number_of_quadrature_points, number_of_vertices), dtype=float)
        shape_function_gradients[number_of_vertices] = \
            np.zeros((number_of_quadrature_points, number_of_vertices, 2), dtype=float)
        for quadrature_point_index in range(number_of_quadrature_points):
            polygonal_quadrature_point = polygonal_quadrature_points[quadrature_point_index, :]
            polygonal_shape_function_values, polygonal_shape_function_gradients = \
                get_polygonal_shape_function_values_and_gradients(number_of_vertices,
                                                                  polygonal_quadrature_point)
            shape_function_values[number_of_vertices][quadrature_point_index, :] = \
                polygonal_shape_function_values
            shape_function_gradients[number_of_vertices][quadrature_point_index, :, :] = \
                polygonal_shape_function_gradients
    return shape_function_values, shape_function_gradients, quadrature_weights
   
#######################################################################################################################
#######################################################################################################################
def get_constant_constitutive_tensors(tensor_space_dim: int = 3, elastic_modulus: float = 1.0, poissons_ratio: float = 0.25):
    """Get a constant constitutive tensors used in plane strain plasticity formulation.

    Args:
        Tensor_space_dim: The computations are done in einstien notation for a 3D tensor space. 
        elastic_modulus: The elastic modulus as a float.
        poissons_ratio: Poisson's ratio as a float.

    Returns:
        IxI outer product of identity tensor with itself, volumetric projector, deviatoric projector,
        and the linear elastic constitutive tensor.
    """
    if elastic_modulus <= 0.0:
        raise ValueError(f"The elastic modulus must be positive. User specified '{elastic_modulus}'")
    if (poissons_ratio <= -1.0) or (poissons_ratio > 0.5):
        raise ValueError(f"Poissons ratio must be > -1 and < 0.5. User specified '{poissons_ratio}'")
    bulk_modulus = elastic_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio))
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poissons_ratio))

    identity_tensor = np.eye(tensor_space_dim, dtype=np.float64)
    IxI = np.einsum("ij, kl -> ijkl", identity_tensor, identity_tensor)
    temp1 = np.einsum("im,jn->ijmn", identity_tensor, identity_tensor)
    temp2 = np.einsum("in,jm->ijmn", identity_tensor, identity_tensor)
    fourth_order_symmetric_identity_tensor = 0.5 * (temp1 + temp2)
    volumetric_projection_tensor = (1.0/3.0)*IxI
    deviatoric_projection_tensor = fourth_order_symmetric_identity_tensor - volumetric_projection_tensor
    linear_elastic_constitutive_tensor = bulk_modulus * IxI + 2.0 * shear_modulus * deviatoric_projection_tensor    #plane strain
    return IxI, deviatoric_projection_tensor, volumetric_projection_tensor, linear_elastic_constitutive_tensor, fourth_order_symmetric_identity_tensor


#######################################################################################################################
#######################################################################################################################
def get_nodal_field_to_element_average_map(element_connectivity_arrays: Dict[int, np.ndarray]) -> sp.sparse.csr_matrix:
    '''This function returns a sparse matrix that maps nodal fields to the average value in each element.

    The sparse matrix, A, is constructed such that A @ scalar_nodal_field = element_average_field_values. This is
    useful for computing the average filtered design variables for each element as an average of the nodal values.

    Args:
        element_connectivity_arrays: A dictionary of numpy arrays containing the element connectivity for each element
            type. The keys are the number of vertices per element and the values are the element connectivity arrays.

    Returns:
        A sparse matrix with shape (number_of_elements, number_of_nodes) that maps nodal fields to element averages.
    '''
    rows, cols, data = [], [], []
    element_index = 0
    for number_of_vertices, element_connectivity in element_connectivity_arrays.items():
        number_of_elements = element_connectivity.shape[0]
        scale = 1.0 / float(number_of_vertices)
        nodal_weights = scale * np.ones((number_of_elements, number_of_vertices), dtype=float)
        element_index_range = np.arange(element_index, element_index + number_of_elements, dtype=int)
        element_index_range = element_index_range.reshape((number_of_elements, 1))
        element_indices = np.tile(element_index_range, (1, number_of_vertices))
        rows.append(element_indices.ravel())
        cols.append(element_connectivity.ravel())
        data.append(nodal_weights.ravel())
        element_index += number_of_elements
    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_data = np.concatenate(data)
    node_to_element_map = sp.sparse.coo_matrix((all_data, (all_rows, all_cols))).tocsr()
    node_to_element_map.sort_indices()
    return node_to_element_map


#######################################################################################################################
#######################################################################################################################
def get_lumped_mass_projection(precomputed_data: PolyPlasPrecomputedData) -> LumpedMassProjection:
    '''This function returns a lumped mass projection object.

    The lumped mass projection is useful for efficiently projecting element-based quantities to the nodes. In
    particular, this is often used to project the element densities to the nodes for making contour plots that are
    smoother than the discontinuous element-based field.

    Args:
        precomputed_data: A precomputed data object for the finite element mesh.

    Returns:
        A LumpedMassProjection object containing the lumped mass projector.
    '''
    logger = logging.getLogger("Poly.TopUtilities")
    start_time = perf_counter()
    number_of_nodes = precomputed_data.number_of_nodes
    number_of_projection_dofs_per_node = 1
    number_of_projection_dofs = number_of_nodes * number_of_projection_dofs_per_node
    projection_dof_indices = np.arange(number_of_projection_dofs, dtype=int)

    matrix_rows = []
    matrix_columns = []
    matrix_data = []
    right_hand_side_integration_data = []
    right_hand_side_integration_indices = []
    right_hand_side_element_indices = []
    element_index = 0
    for number_of_vertices, element_connectivity in precomputed_data.element_connectivity_arrays.items():
        number_of_projection_dofs_per_element = number_of_vertices * number_of_projection_dofs_per_node
        number_of_elements = element_connectivity.shape[0]
        matrix_entries = np.einsum(
            'qm,qn,eq->emn',
            precomputed_data.shape_function_values[number_of_vertices],
            precomputed_data.shape_function_values[number_of_vertices],
            precomputed_data.jacobian_determinant_x_quadrature_weights[number_of_vertices],
            optimize=True
        )
        matrix_data.append(matrix_entries.flatten())

        elementwise_dof_indices = projection_dof_indices[element_connectivity]
        right_hand_side_integration = np.einsum(
            'qn,eq->en',
            precomputed_data.shape_function_values[number_of_vertices],
            precomputed_data.jacobian_determinant_x_quadrature_weights[number_of_vertices]
        )
        right_hand_side_integration_data.append(right_hand_side_integration.flatten())
        right_hand_side_integration_indices.append(elementwise_dof_indices.flatten())
        element_indices = np.arange(element_index, element_index + number_of_elements, dtype=int)
        element_indices = element_indices.reshape((number_of_elements, 1))
        right_hand_side_element_indices.append(
            np.tile(element_indices, (1, number_of_projection_dofs_per_element)).ravel()
        )

        column_indices = np.tile(elementwise_dof_indices, (number_of_projection_dofs_per_element, 1, 1))
        column_indices = column_indices.transpose((1, 0, 2))
        row_indices = column_indices.transpose((0, 2, 1))

        matrix_rows.append(row_indices.flatten())
        matrix_columns.append(column_indices.flatten())
        element_index += number_of_elements

    right_hand_side_integration_data = np.concatenate(right_hand_side_integration_data).ravel()
    right_hand_side_integration_cols = np.zeros((right_hand_side_integration_data.size,), dtype=int)
    right_hand_side_integration_rows = np.concatenate(right_hand_side_integration_indices).ravel()
    right_hand_side_integration_indices = (right_hand_side_integration_rows, right_hand_side_integration_cols)
    right_hand_side_element_indices = np.concatenate(right_hand_side_element_indices).ravel()

    rows = np.concatenate(matrix_rows)
    cols = np.concatenate(matrix_columns)
    data = np.concatenate(matrix_data)
    mass_projection_matrix = sp.sparse.coo_matrix((data, (rows, cols)), dtype=float).tocsr()
    matrix_row_sum = mass_projection_matrix.sum(axis=1)
    if isinstance(matrix_row_sum, np.matrix):
        matrix_row_sum = matrix_row_sum.A1
    lumped_mass_matrix_inverse = sp.sparse.diags(1.0 / matrix_row_sum, offsets=0, format='csr')

    lumped_mass_projection = LumpedMassProjection(
        right_hand_side_vector_data=right_hand_side_integration_data,
        right_hand_side_vector_indices=right_hand_side_integration_indices,
        right_hand_side_vector_element_indices=right_hand_side_element_indices,
        lumped_mass_matrix_inverse=lumped_mass_matrix_inverse
    )

    elapsed_time = perf_counter() - start_time
    message = f"Lumped mass projector construction required {elapsed_time:0.2f} seconds."
    logger.info(message)
    return lumped_mass_projection


#######################################################################################################################
#######################################################################################################################
def get_precomputed_data_filter_and_heaviside_projection(polymesh_object: PolyMesher, material_parameters: Dict):
    '''This function returns a PolyTopPrecomputedData object for a given mesh as well as the filter and heaviside projection.

    The PolyTopPrecomputedData object contains many quantities that are needed to run an optimization problem.
    This includes the nodal coordinates, element connectivity, boundary conditions, shape function values /
    gradients, and several other arrays including index arrays for efficient operations.

    Args:
        polymesh_object: A PolyMesher object containing the mesh data and boundary conditions.
        material_parameters: A dictionary containing the material parameters for the mesh.

        materials_parameters: A Dict containing all the relevant material properties for the mesh.

    Returns:
        A PolyTopPrecomputedData object containing the precomputed data for the mesh.
    '''
    logger = logging.getLogger("Poly.TopUtilities")
    nodal_coordinates, element_connectivity_arrays, boundary_conditions = \
        polymesh_object.get_mesh_and_boundary_conditions()

    start_time = perf_counter()
    elastic_modulus = material_parameters["elastic_modulus"]
    poissons_ratio = material_parameters["poissons_ratio"]
    hardening_modulus = material_parameters["hardening_modulus"]
    initial_yield_stress = material_parameters["initial_yield_stress"]

    IxI, deviatoric_projection_tensor, volumetric_projection_tensor, linear_elastic_constitutive_tensor, fourth_order_symmetric_identity_tensor = \
        get_constant_constitutive_tensors(elastic_modulus=elastic_modulus, poissons_ratio=poissons_ratio)
    
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poissons_ratio))
    bulk_modulus = elastic_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio))
    number_of_triangle_quadrature_points = 1    #Can be 1 or 3

    number_of_dofs_per_node = 2
    number_of_nodes = nodal_coordinates.shape[0]
    total_number_of_nodal_dofs = number_of_nodes * number_of_dofs_per_node
    new_shape = (number_of_nodes, number_of_dofs_per_node)
    nodal_dof_indices = np.arange(total_number_of_nodal_dofs, dtype=int).reshape(new_shape)

    fixed_x_dof_indices = nodal_dof_indices[boundary_conditions.fixed_x_nodal_indices, 0].ravel()
    fixed_y_dof_indices = nodal_dof_indices[boundary_conditions.fixed_y_nodal_indices, 1].ravel()
    fixed_dof_indices = np.concatenate((fixed_x_dof_indices, fixed_y_dof_indices)).ravel()
    
    for applied_displacement in boundary_conditions.applied_displacements:
        x_dof_indices = nodal_dof_indices[applied_displacement.nodal_indices, 0].ravel()
        y_dof_indices = nodal_dof_indices[applied_displacement.nodal_indices, 1].ravel()
    if np.all(applied_displacement.applied_displacement_components[:,:] != 0.0): #in both x and y direction
        total_applied_displacement_x = applied_displacement.applied_displacement_components[0,0]
        total_applied_displacement_y = applied_displacement.applied_displacement_components[0,1]
        unique_nonzero_dirichlet_dof_indices_x = np.asarray(x_dof_indices,dtype=np.int32)
        unique_nonzero_dirichlet_dof_indices_y = np.asarray(y_dof_indices,dtype=np.int32)
        unique_nonzero_dirichlet_dof_indices = np.concatenate((unique_nonzero_dirichlet_dof_indices_x, unique_nonzero_dirichlet_dof_indices_y))
    elif np.all(applied_displacement.applied_displacement_components[:,0]) != 0.0: # in x direction
        total_applied_displacement_x = applied_displacement.applied_displacement_components[0,0]
        total_applied_displacement_y = 0.0
        unique_nonzero_dirichlet_dof_indices_x = np.asarray(x_dof_indices,dtype=np.int32)
        unique_nonzero_dirichlet_dof_indices_y = None
        unique_nonzero_dirichlet_dof_indices = np.asanyarray(x_dof_indices, dtype=np.int32)
    elif np.all(applied_displacement.applied_displacement_components[:,1]) != 0.0: # in y direction
        total_applied_displacement_x = 0.0
        total_applied_displacement_y = applied_displacement.applied_displacement_components[0,1]
        unique_nonzero_dirichlet_dof_indices_x = None
        unique_nonzero_dirichlet_dof_indices_y = np.asarray(y_dof_indices,dtype=np.int32)
        unique_nonzero_dirichlet_dof_indices = np.asarray(y_dof_indices,dtype=np.int32)
    
    

    
    unique_numbers_of_vertices = element_connectivity_arrays.keys()
    #POLYGONAL ELEMENTS
    shape_function_values, shape_function_gradients_in_natural_coordinates, quadrature_weights = get_shape_function_tables_and_quadrature_weights(unique_numbers_of_vertices, number_of_triangle_quadrature_points)
    
    element_index = 0
    total_area = 0.0
    stiffness_matrix_rows = []
    stiffness_matrix_columns = []
    stiffness_matrix_element_indices = []
    element_relative_volume_arrays = []
    elementwise_dof_index_arrays = {}
    shape_function_gradients_in_physical_coordinates = {}
    jacobian_determinant_x_quadrature_weights = {}
    number_of_quadrature_points_per_element = {}
    element_volumes_per_elem_type = {}
    global_dof_indices_elementwise = {}
    virtual_displacement_symmetric_gradients = {}
    virtual_displacement_volume_weighted_divergence = {}
    number_of_dofs_per_elem_type = {}
    total_global_dof_indices_elementwise_temp = []
    nodal_dof_indices = np.arange(total_number_of_nodal_dofs, dtype=int).reshape((number_of_nodes, number_of_dofs_per_node))

    for number_of_vertices in unique_numbers_of_vertices:
        number_of_dofs_per_elem_type[number_of_vertices] = number_of_vertices * number_of_dofs_per_node
        number_of_quadrature_points_per_element[number_of_vertices] = \
            number_of_vertices * number_of_triangle_quadrature_points
        element_connectivity = element_connectivity_arrays[number_of_vertices]
        nodal_coordinates_elementwise = nodal_coordinates[element_connectivity, :]
        jacobian_of_the_mapping = \
            np.einsum('eni,qnj->eqij',
                      nodal_coordinates_elementwise,
                      shape_function_gradients_in_natural_coordinates[number_of_vertices])
        jacobian_determinants = np.linalg.det(jacobian_of_the_mapping)
        jacobian_determinant_x_quadrature_weights[number_of_vertices] = \
            np.einsum('eq,q->eq',
                      jacobian_determinants,
                      quadrature_weights[number_of_vertices])
        inverse_jacobian = np.linalg.inv(jacobian_of_the_mapping)
        shape_function_gradients_in_physical_coordinates[number_of_vertices] = \
            np.einsum('eqji,qnj->eqni',
                      inverse_jacobian,
                      shape_function_gradients_in_natural_coordinates[number_of_vertices])
        element_areas = np.sum(jacobian_determinant_x_quadrature_weights[number_of_vertices], axis=1)
        element_volumes_per_elem_type[number_of_vertices] = element_areas
        total_area += np.sum(element_areas)
        element_relative_volume_arrays.append(element_areas)

        # Compute virtual displacement symmetric gradients
        number_of_dofs_per_element = number_of_dofs_per_node * number_of_vertices
        virtual_displacements = np.zeros((number_of_dofs_per_element, number_of_vertices, number_of_dofs_per_node))
        
        
        for element_node_index in range(number_of_vertices):
            for nodal_dof_index in range(number_of_dofs_per_node):
                element_dof_index = element_node_index * number_of_dofs_per_node + nodal_dof_index
                virtual_displacements[element_dof_index, element_node_index, nodal_dof_index] = 1.0
        virtual_displacement_gradients = np.einsum(
            'inx,eqnd->eqixd',
            virtual_displacements,
            shape_function_gradients_in_physical_coordinates[number_of_vertices]
        )
        virtual_displacement_gradients_transpose = virtual_displacement_gradients.transpose((0, 1, 2, 4, 3))
        virtual_displacement_symmetric_gradients[number_of_vertices] = \
            0.5 * (virtual_displacement_gradients + virtual_displacement_gradients_transpose)
        trace_virtual_disp_symmetric_gradients = np.einsum("eqijj->eqi", virtual_displacement_symmetric_gradients[number_of_vertices])
        virtual_displacement_volume_weighted_divergence[number_of_vertices] = np.einsum("eqi,eq->ei", trace_virtual_disp_symmetric_gradients, jacobian_determinant_x_quadrature_weights[number_of_vertices])


        # Compute arrays of indices
        number_of_elements_per_element_type = element_connectivity.shape[0]
        global_dof_indices_elementwise[number_of_vertices] = nodal_dof_indices[element_connectivity, :].reshape((number_of_elements_per_element_type, number_of_dofs_per_elem_type[number_of_vertices]))
        np.array(total_global_dof_indices_elementwise_temp.append(global_dof_indices_elementwise[number_of_vertices].ravel()))

        my_shape = (number_of_elements_per_element_type, number_of_dofs_per_element)
        elementwise_dof_index_arrays[number_of_vertices] = nodal_dof_indices[element_connectivity, :].reshape(my_shape)
        column_indices = np.tile(elementwise_dof_index_arrays[number_of_vertices],
                                 (number_of_dofs_per_element, 1, 1)).transpose((1, 0, 2))
        row_indices = column_indices.transpose((0, 2, 1))
        
        coo_matrix_row_indices = row_indices.flatten()
        coo_matrix_column_indices = column_indices.flatten()

        element_indices = np.arange(element_index, element_index + number_of_elements_per_element_type, dtype=int)
        element_indices = element_indices.reshape((number_of_elements_per_element_type, 1))
        matrix_element_indices = np.tile(element_indices, (1, number_of_dofs_per_element**2)).ravel()

        stiffness_matrix_rows.append(coo_matrix_row_indices)
        stiffness_matrix_columns.append(coo_matrix_column_indices)
        stiffness_matrix_element_indices.append(matrix_element_indices)

        element_index += number_of_elements_per_element_type

    total_global_dof_indices_elementwise = np.concatenate(total_global_dof_indices_elementwise_temp)
    internal_force_indices_array = (total_global_dof_indices_elementwise, np.zeros((total_global_dof_indices_elementwise.size,), dtype=int))

    total_number_of_elements = element_index
    all_stiffness_matrix_rows = np.concatenate(stiffness_matrix_rows).ravel()
    all_stiffness_matrix_columns = np.concatenate(stiffness_matrix_columns).ravel()
    all_stiffness_matrix_element_indices = np.concatenate(stiffness_matrix_element_indices).ravel()

    all_dirichlet_dof_indices = np.sort(np.unique(np.concatenate([fixed_dof_indices, unique_nonzero_dirichlet_dof_indices]))) #indices of the fixed and "displaced" dof
    all_dirichlet_dof_coo_mask = np.in1d(all_stiffness_matrix_rows,    all_dirichlet_dof_indices) | \
                                 np.in1d(all_stiffness_matrix_columns, all_dirichlet_dof_indices)
    coo_matrix_all_dirichlet_dof_indices = np.sort(np.argwhere(all_dirichlet_dof_coo_mask))

    # for building the rhs vector contribution from nonzero dofs
    coo_matrix_nonzero_dirichlet_dof_indices = np.unique(np.argwhere(np.isin(all_stiffness_matrix_columns,  unique_nonzero_dirichlet_dof_indices)).ravel())
    coo_matrix_nonzero_dirichlet_rows = all_stiffness_matrix_rows[coo_matrix_nonzero_dirichlet_dof_indices]
    coo_matrix_nonzero_dirichlet_cols = np.zeros((coo_matrix_nonzero_dirichlet_rows.size,), dtype=int)
    coo_matrix_nonzero_dirichlet_indices = (coo_matrix_nonzero_dirichlet_rows, coo_matrix_nonzero_dirichlet_cols)

    if np.all(applied_displacement.applied_displacement_components[:,:] != 0.0): #in both x and y direction
        coo_matrix_nonzero_dirichlet_dof_indices_x = np.unique(np.argwhere(np.isin(all_stiffness_matrix_columns,  unique_nonzero_dirichlet_dof_indices_x)).ravel())
        coo_matrix_nonzero_dirichlet_rows_x = all_stiffness_matrix_rows[coo_matrix_nonzero_dirichlet_dof_indices_x]
        coo_matrix_nonzero_dirichlet_cols_x = np.zeros((coo_matrix_nonzero_dirichlet_rows_x.size,), dtype=int)
        coo_matrix_nonzero_dirichlet_indices_x = (coo_matrix_nonzero_dirichlet_rows_x, coo_matrix_nonzero_dirichlet_cols_x)

        coo_matrix_nonzero_dirichlet_dof_indices_y = np.unique(np.argwhere(np.isin(all_stiffness_matrix_columns,  unique_nonzero_dirichlet_dof_indices_y)).ravel())
        coo_matrix_nonzero_dirichlet_rows_y = all_stiffness_matrix_rows[coo_matrix_nonzero_dirichlet_dof_indices_y]
        coo_matrix_nonzero_dirichlet_cols_y = np.zeros((coo_matrix_nonzero_dirichlet_rows_y.size,), dtype=int)
        coo_matrix_nonzero_dirichlet_indices_y = (coo_matrix_nonzero_dirichlet_rows_y, coo_matrix_nonzero_dirichlet_cols_y)
    else:
        coo_matrix_nonzero_dirichlet_dof_indices_x = None
        coo_matrix_nonzero_dirichlet_indices_x = None
        coo_matrix_nonzero_dirichlet_dof_indices_y = None
        coo_matrix_nonzero_dirichlet_indices_y = None

    # for setting the diagonal component of a coo_matrix to 1 by finding a single unique diagonal component to set to 1 and all others equal to 0
    diagonal_entries_mask = (all_stiffness_matrix_rows == all_stiffness_matrix_columns)
    fixed_diagonal_entries_mask = diagonal_entries_mask & all_dirichlet_dof_coo_mask
    row = all_stiffness_matrix_rows[fixed_diagonal_entries_mask]
    _, unique_row_indices = np.unique(row, return_index=True)
    fixed_diagonal_indices = np.argwhere(fixed_diagonal_entries_mask).ravel()
    coo_matrix_unique_dirichlet_diagonal_dof_indices = np.sort(fixed_diagonal_indices[unique_row_indices])

    element_volumes = np.concatenate(element_relative_volume_arrays).ravel()
    element_relative_volumes = element_volumes / total_area

    delaunay_simplices = polymesh_object.get_delaunay_simplices()

    node_to_element_map = get_nodal_field_to_element_average_map(element_connectivity_arrays)

    _, largest_element_edge_length = polymesh_object.get_smallest_and_largest_edge_length()
    passive_design_variable_indices, passive_local_element_indices = \
        polymesh_object.boundary_value_problem.get_passive_region_indices(
            nodal_coordinates,
            element_connectivity_arrays,
            largest_element_edge_length
        )
    
    first_element_index = 0
    element_indices = {}
    for number_of_vertices, element_connectivity in element_connectivity_arrays.items():
        number_of_elements_of_this_type = element_connectivity.shape[0]
        last_element_index = first_element_index + number_of_elements_of_this_type
        element_indices[number_of_vertices] = np.arange(first_element_index, last_element_index, dtype=int)
        first_element_index = last_element_index

    passive_global_element_indices = []
    at_least_one_array_is_nonempty = False
    for number_of_vertices, local_element_indices in passive_local_element_indices.items():
        if local_element_indices.size > 0:
            at_least_one_array_is_nonempty = True
        global_element_indices = element_indices[number_of_vertices][local_element_indices]
        passive_global_element_indices.append(global_element_indices)
    passive_global_element_indices = np.concatenate(passive_global_element_indices, dtype=int).ravel() \
        if at_least_one_array_is_nonempty else np.array([], dtype=int)
    
    precomputed_data = PolyPlasPrecomputedData(
        number_of_nodes = number_of_nodes,
        number_of_elements = total_number_of_elements,
        number_of_dofs_per_node = number_of_dofs_per_node,
        number_of_dofs_per_element_type = number_of_dofs_per_elem_type,
        shear_modulus = shear_modulus,
        bulk_modulus = bulk_modulus,
        hardening_modulus = hardening_modulus,
        initial_yield_stress = initial_yield_stress,
        space_dimension = number_of_dofs_per_node,
        number_of_global_dofs = total_number_of_nodal_dofs,
        number_of_quad_points_per_element = number_of_quadrature_points_per_element,
        global_dof_indices_elementwise= global_dof_indices_elementwise,
        delaunay_simplices = delaunay_simplices,
        element_relative_volumes = element_relative_volumes, #for volume constraint in top opt
        element_volumes_per_elem_type = element_volumes_per_elem_type,
        nodal_coordinates = nodal_coordinates,
        element_connectivity_arrays = element_connectivity_arrays,
        shape_function_values = shape_function_values,  
        shape_function_gradients = shape_function_gradients_in_physical_coordinates,   
        jacobian_determinant_x_quadrature_weights = jacobian_determinant_x_quadrature_weights,
        node_to_element_map = node_to_element_map,
        virtual_displacement_symmetric_gradients = virtual_displacement_symmetric_gradients,
        virtual_displacement_volume_weighted_divergence=virtual_displacement_volume_weighted_divergence,
        fixed_dof_indices=fixed_dof_indices,
        internal_force_indices_array = internal_force_indices_array,
        unique_zero_dirichlet_dof_indices = np.asarray(fixed_dof_indices,dtype=np.int32),
        unique_nonzero_dirichlet_dof_indices = unique_nonzero_dirichlet_dof_indices,
        unique_nonzero_dirichlet_dof_indices_x = unique_nonzero_dirichlet_dof_indices_x,
        unique_nonzero_dirichlet_dof_indices_y = unique_nonzero_dirichlet_dof_indices_y,
        coo_matrix_row_indices = all_stiffness_matrix_rows,
        coo_matrix_column_indices = all_stiffness_matrix_columns,
        coo_matrix_all_dirichlet_dof_indices = coo_matrix_all_dirichlet_dof_indices,
        coo_matrix_unique_dirichlet_diagonal_dof_indices = coo_matrix_unique_dirichlet_diagonal_dof_indices,
        coo_matrix_nonzero_dirichlet_dof_indices = coo_matrix_nonzero_dirichlet_dof_indices,
        coo_matrix_nonzero_dirichlet_dof_indices_x = coo_matrix_nonzero_dirichlet_dof_indices_x,
        coo_matrix_nonzero_dirichlet_dof_indices_y = coo_matrix_nonzero_dirichlet_dof_indices_y,
        coo_matrix_nonzero_dirichlet_indices = coo_matrix_nonzero_dirichlet_indices,
        coo_matrix_nonzero_dirichlet_indices_x = coo_matrix_nonzero_dirichlet_indices_x,
        coo_matrix_nonzero_dirichlet_indices_y = coo_matrix_nonzero_dirichlet_indices_y,
        total_applied_displacement_x=total_applied_displacement_x,
        total_applied_displacement_y=total_applied_displacement_y,
        IxI=IxI,
        deviatoric_projection_tensor=deviatoric_projection_tensor,
        volumetric_projection_tensor=volumetric_projection_tensor,
        linear_elastic_constitutive_tensor=linear_elastic_constitutive_tensor,
        fourth_order_symmetric_identity_tensor=fourth_order_symmetric_identity_tensor,
        passive_design_variable_indices=passive_design_variable_indices,
        passive_local_element_indices=passive_local_element_indices,
        passive_global_element_indices=passive_global_element_indices
    )
   
    elapsed_time = perf_counter() - start_time
    message = f"Precomputation of numerical quantities required {elapsed_time:0.1f} seconds."
    logger.info(message)
    return precomputed_data

#######################################################################################################################
#######################################################################################################################
def get_initial_plastic_state(precomputed_data: PolyPlasPrecomputedData):
    state = currentPlasticState()
    number_of_global_dofs = precomputed_data.number_of_nodes * precomputed_data.number_of_dofs_per_node
    state.nodal_displacement_vector = np.zeros((number_of_global_dofs,), dtype=np.float64)

    state.current_total_strain_tensors = {}
    state.previous_total_strain_tensors = {}
    state.current_plastic_strain_tensors = {}
    state.previous_plastic_strain_tensors = {}
    state.current_stress_tensors = {}
    state.previous_stress_tensors = {}
    state.current_deviatoric_stress_tensors = {}
    state.previous_deviatoric_stress_tensors = {}
    state.current_pressure = {}
    state.current_accumulated_plastic_strains = {}
    state.previous_accumulated_plastic_strains = {}
    state.plastic_multiplier_increments = {}

    unique_numbers_of_vertices = precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        number_of_quad_points_per_element = precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        number_of_elements = precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        state.current_total_strain_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.previous_total_strain_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.current_plastic_strain_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.previous_plastic_strain_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.current_stress_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.previous_stress_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.current_deviatoric_stress_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.previous_deviatoric_stress_tensors[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element, 3, 3), dtype=np.float64)
        state.current_pressure[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element), dtype=np.float64)
        state.current_accumulated_plastic_strains[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element), dtype=np.float64)
        state.previous_accumulated_plastic_strains[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element), dtype=np.float64)
        state.plastic_multiplier_increments[number_of_vertices] = np.zeros((number_of_elements, number_of_quad_points_per_element), dtype=np.float64)
    return state


#######################################################################################################################
#######################################################################################################################
def write_vtk_output(nodal_coordinates: np.ndarray,
                     element_connectivity_arrays: Dict[int, np.ndarray],
                     optimization_iteration_number: Optional[np.ndarray],
                     current_time: float = 0.0,
                     output_directory: str = "./",
                     point_data: Optional[Dict[str, np.ndarray]] = None,
                     cell_data: Optional[List[Dict[str, np.ndarray]]] = None):
    '''This function outputs a VTK file for visualization of the mesh and associated data in an external \
    software package like Paraview or VisIt.

    Args:
        nodal_coordinates: A numpy array of nodal coordinates for the finite element mesh.
        element_connectivity_arrays: A dictionary of numpy arrays containing the element connectivity for \
            each element type. The keys are the number of vertices per element and the values are the element \
            connectivity.
        optimization_iteration_number: The optimization iteration number as an integer.
        current_time: The current time as a float.
        output_directory: The directory to which the VTK file will be written.
        point_data: A dictionary of numpy arrays containing the point data to be written to the vtk file. The keys
            are the names of the point data fields and the values are the numpy arrays containing the data.
        cell_data: A list of dictionaries of numpy arrays containing the cell data to be written to the vtk file.
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    number_of_nodes, space_dimension = nodal_coordinates.shape
    nodal_coordinates_3D = np.zeros((number_of_nodes, 3), dtype=float)
    nodal_coordinates_3D[:, :space_dimension] = nodal_coordinates
    cell_blocks = [meshio.CellBlock("polygon", element_connectivity)
                   for element_connectivity in element_connectivity_arrays.values()]
    output_filepath = os.path.join(output_directory, f"output_iter_.{optimization_iteration_number:04f}.vtk")
    meshio.write_points_cells(
        filename=output_filepath,
        points=nodal_coordinates_3D,
        cells=cell_blocks,
        point_data=point_data,
        cell_data=cell_data
    )

#######################################################################################################################
#######################################################################################################################
def plot_scalar_nodal_field_contours(precomputed_data,
                                     scalar_nodal_field: np.ndarray,
                                     figure_title: str = "",
                                     colorbar_label: str = "Scalar Field",
                                     colormap: str = "binary",
                                     colorbar_lower_bound: float = 0.0,
                                     colorbar_upper_bound: float = 1.0):
    """Create a contour plot of a scalar nodal field.

    Args:
        precomputed_data: A precomputed data object for the finite element mesh.
        scalar_nodal_field: A numpy array containing the scalar nodal field to be plot.
        figure_title: The title of the figure as a string.
        colorbar_label: The label for the colorbar as a string.
        colormap: The name of the colormap as a string.
        colorbar_lower_bound: The lower bound of the colorbar as a float.
        colorbar_upper_bound: The upper bound of the colorbar as a float.
    """
    matplotlib_triangulation = mpl.tri.Triangulation(precomputed_data.nodal_coordinates[:, 0],
                                                     precomputed_data.nodal_coordinates[:, 1],
                                                     precomputed_data.delaunay_simplices)

    maximum_nodal_value = np.amax(scalar_nodal_field)
    extend = "neither" if maximum_nodal_value <= colorbar_upper_bound else 'max'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    number_of_contour_levels = 256
    my_colormap = plt.colormaps[colormap].with_extremes(bad="yellow", under="white", over="magenta")
    levels = np.linspace(colorbar_lower_bound, colorbar_upper_bound, number_of_contour_levels, endpoint=True)
    contour_plot = ax.tricontourf(matplotlib_triangulation,
                                  scalar_nodal_field,
                                  levels,
                                  cmap=my_colormap,
                                  extend=extend)
    ax.axis('equal')
    ax.axis('off')
    ax.set_title(figure_title, fontsize=18)
    contour_plot.set_clim([colorbar_lower_bound, colorbar_upper_bound])
    color_bar = fig.colorbar(contour_plot,
                             ticks=[colorbar_lower_bound, colorbar_upper_bound],
                             format="%0.1f",
                             extend=extend)
    color_bar.set_label(label=colorbar_label, size=18)
    color_bar.ax.tick_params(labelsize=16)
    fig.tight_layout()