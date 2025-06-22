import os
import logging
import sys
import numpy as np
import scipy as sp
import pandas as pd
import copy
import warnings
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import opt_einsum as oe
from time import perf_counter
from scipy.sparse import coo_matrix
try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve
from src.polyplas.MMA import perform_design_update
import src.polyplas.PolyPlasUtilities as polyplas_utils
import src.polytop.PolyFilter as polyfilter
import src.polytop.PolyInterpolation as polyinterpolation
import src.polytop.PolyProjection as polyprojection
from typing import Optional, Tuple
from time import perf_counter
try:
    from pypardiso import factorized
except ImportError:
    from scipy.sparse.linalg import factorized


#######################################################################################################################
#######################################################################################################################
class PolyPlas:
    '''PolyPlas class for von-Mises based topology optimization.

    Attributes:
        design_variable_filter (polyfilter.FilterBase): Design variable filter object.
        elastic_material_interpolation (polyinterpolation.MaterialInterpolationFunction):
            Material interpolation function for elastic parameters which includes an interpolation function (e.g., SIMP, RAMP).
        plastic_material_interpolation (polyinterpolation.MaterialInterpolationFunction):
            Material interpolation function for plastic parameters which includes an interpolation function (e.g., SIMP, RAMP).
        projection (polyprojection.ProjectionBase): Projection object for projecting filtered design variables.
        maximum_projection_strength (float): Maximum projection strength for the projection operation.
        precomputed_data (polyplas_utils.PolyPlasPrecomputedData):
            Precomputed numerical quantities for the finite element problem.
        volume_fraction_upper_bound (float): Volume fraction constraint upper bound.
        number_of_time_steps (int): Number of time steps.
        directory_name (str): Directory name which will be created to store all output files 
        (paraview .vtk, convergence plots, deviatoric plane plots)

        Raises:
            TypeError: If any of the input arguments do not have the correct type.
            ValueError: If any of the input arguments do not have an acceptable value.
    '''

    def __init__(self,
                 design_variable_filter: polyfilter.FilterBase = polyfilter.IdentityFilter(),
                 elastic_material_interpolation: polyinterpolation.MaterialInterpolationFunction = polyinterpolation.SIMP(),
                 plastic_material_interpolation: polyinterpolation.MaterialInterpolationFunction = polyinterpolation.SIMP(),
                 projection: polyprojection.ProjectionBase = polyprojection.IdentityProjection(),
                 maximum_projection_strength: float = 1.25,
                 precomputed_data: polyplas_utils.PolyPlasPrecomputedData = None,
                 volume_fraction_upper_bound: float = 0.5,
                 number_of_time_steps: int = 100,
                 directory_name: str = 'default_directory'):
        
        self.logger = logging.getLogger("Poly.Top")

        if not isinstance(design_variable_filter, polyfilter.FilterBase):
            raise TypeError(f"User supplied 'design_variable_filter' of type '{type(design_variable_filter)}'.")
        self.design_variable_filter = design_variable_filter

        if not isinstance(elastic_material_interpolation, polyinterpolation.MaterialInterpolationFunction):
            raise TypeError(f"User supplied 'elastic material_interpolation' of type '{type(elastic_material_interpolation)}'.")
        self.elastic_material_interpolation = elastic_material_interpolation

        if not isinstance(plastic_material_interpolation, polyinterpolation.MaterialInterpolationFunction):
            raise TypeError(f"User supplied 'plastic material_interpolation' of type '{type(plastic_material_interpolation)}'.")
        self.plastic_material_interpolation = plastic_material_interpolation

        if not isinstance(projection, polyprojection.ProjectionBase):
            raise TypeError(f"User supplied 'projection' of type '{type(projection)}'.")
        self.projection = projection

        if not isinstance(maximum_projection_strength, float):
            raise TypeError(f"User supplied 'maximum_projection_strength' of type '{type(maximum_projection_strength)}'.")
        
        if isinstance(maximum_projection_strength, float):
            if maximum_projection_strength <= 0.0:
                raise ValueError("Maximum projection strength must be greater than 0.")
    
        if precomputed_data is None:
            raise ValueError("User must supply 'precomputed_data' which can be created "
                             "using the 'get_precomputed_data' function from PolyPlasUtilities.")
        if not isinstance(precomputed_data, polyplas_utils.PolyPlasPrecomputedData):
            raise TypeError(f"User supplied 'precomputed_data' of type '{type(precomputed_data)}'.")
        
        if not isinstance(volume_fraction_upper_bound, float):
            raise TypeError(f"User supplied 'volume_fraction_upper_bound' of type '{type(volume_fraction_upper_bound)}', should be float.")
        
        if volume_fraction_upper_bound <= 0.0 or volume_fraction_upper_bound > 1.0:
            raise ValueError(f"User supplied 'volume_fraction_upper_bound' of value '{volume_fraction_upper_bound}', should be between 0 and 1.")
        
        if not isinstance(directory_name, str):
            raise TypeError(f"User supplied 'directory_name' of type '{type(directory_name)}', should be string.")    

        if isinstance(number_of_time_steps, int):
            if number_of_time_steps <= 0:
                raise ValueError("Number of time steps must be greater than 0.")
        
        if not isinstance(number_of_time_steps, int):
            raise TypeError(f"User supplied 'number_of_time_steps' of type '{type(number_of_time_steps)}', should be int.")
           
        
        self.precomputed_data = precomputed_data

        self.volume_fraction_upper_bound = volume_fraction_upper_bound
        self.maximum_projection_strength = maximum_projection_strength
        minimum_projection_strength = self.projection.projection_strength
        projection_strengths = np.linspace(minimum_projection_strength, self.maximum_projection_strength, 5)
        projection_strengths = projection_strengths[1:]
        self.projection_strengths_continuation = projection_strengths
        self.objective_scale_factor = 1.0
        self.lumped_mass_projection = polyplas_utils.get_lumped_mass_projection(precomputed_data)
        self.logged_values_data_frame = None
        self.optimization_iteration_number = 0
        self.element_densities = np.ones((self.precomputed_data.number_of_elements,), dtype=float)
        self.average_design_variable_change = 1.0
        self.number_of_time_steps = number_of_time_steps
        self.total_applied_displacement_x = self.precomputed_data.total_applied_displacement_x
        self.total_applied_displacement_y = self.precomputed_data.total_applied_displacement_y
        self.filtered_and_projected_design_variables = np.ones((self.precomputed_data.number_of_elements,), dtype=float)
        self.directory_name = directory_name

        self.filtered_and_projected_design_variables_per_elem_type = {}
        unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()
        element_index = 0
        for number_of_vertices in unique_numbers_of_vertices:
            number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
            self.filtered_and_projected_design_variables_per_elem_type[number_of_vertices] = self.filtered_and_projected_design_variables\
            [element_index:(element_index+number_of_elements_per_elem_type)]
            element_index += number_of_elements_per_elem_type
        
        self.state = polyplas_utils.get_initial_plastic_state(self.precomputed_data)
     
    @property
    def volume_fraction_upper_bound(self) -> float:
        return self.__volume_fraction_upper_bound

    @volume_fraction_upper_bound.setter
    def volume_fraction_upper_bound(self, value: float):
        if not isinstance(value, float):
            raise TypeError(f"Volume fraction upper bound must be a float. User specified type '{type(value)}'.")
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Volume fraction upper bound must be between 0 and 1. User specified value '{value}'.")
        self.__volume_fraction_upper_bound = value
    
    def run_top_opt_problem(self, MMA_state, maximum_optimization_iterations: int, convergence_tolerance: float, check_grad: bool):
        '''Run the topology optimization problem.

        Args:
            MMA_state: Instance of the dataclass storing MMA parameters.
            maximum_optimization_iterations (int): Maximum number of optimization iterations to perform.
            convergence_tolerance (Optional[float]): Convergence tolerance for the L_infty norm of the current and the previous design variables
            check_grad (Optional[bool]): Flag to check the gradients before using them in the MMA. 
        Returns:
            Logger with the design variables (np.ndarray), the objective function value (np.ndarray),
            volume fraction value (np.ndarray), and volume constraint upper bound.
        '''

        if check_grad == True:
            np.random.seed(777)
            initial_design_variables = np.random.uniform(low=0.75, high=0.95, size=MMA_state.current_design_variables.size)
            self.check_gradients_MMA_input(initial_design_variables)
            sys.exit(0)

        if not isinstance(maximum_optimization_iterations, int):
            raise TypeError(f"Maximum optimization iterations must be an int. "
                            f"User specified type '{type(maximum_optimization_iterations)}'.")

        if maximum_optimization_iterations is not None:
            self.maximum_optimization_iterations = maximum_optimization_iterations

        if not isinstance(convergence_tolerance, float):
            raise TypeError(f"Convergence tolerance must be a float. "
                            f"User specified type '{type(convergence_tolerance)}'.")
        if convergence_tolerance < 0.0:
            raise ValueError(f"Convergence tolerance must be >= 0.0. "
                             f"User specified value '{convergence_tolerance}'.")
        
        number_of_nodes = self.precomputed_data.number_of_nodes
        number_of_elements = self.precomputed_data.number_of_elements

        initial_design_variables = MMA_state.current_design_variables
        if self.precomputed_data.passive_design_variable_indices.size == 0:
            if initial_design_variables.size != number_of_nodes:
                raise ValueError("The initial_design_variables array must have size equal to the number of nodes.")
        
        filtered_nodal_design_variables = np.zeros((number_of_nodes), dtype=float)
        filtered_element_design_variables = np.zeros((number_of_elements), dtype=float)
        design_variable_change_measure = np.ones((1), dtype=float)

        optimization_iteration_number = 0
        while optimization_iteration_number < maximum_optimization_iterations and design_variable_change_measure > convergence_tolerance:
            if self.precomputed_data.passive_design_variable_indices.size > 0:
                variables_for_optimization_mask = np.ones(self.precomputed_data.number_of_nodes, dtype=bool)
                variables_for_optimization_mask[self.precomputed_data.passive_design_variable_indices] = False 
                design_variables = np.ones(self.precomputed_data.number_of_nodes, dtype=float)
                design_variables[variables_for_optimization_mask] = MMA_state.current_design_variables.flatten()
            else:
                design_variables = MMA_state.current_design_variables.flatten()
           
            filtered_nodal_design_variables[:] = self.design_variable_filter.apply_filter(design_variables)
            filtered_element_design_variables[:] = \
                self.precomputed_data.node_to_element_map @ filtered_nodal_design_variables
            self.element_densities[:] = self.projection.apply_projection(filtered_element_design_variables)

            #get objection and constraint function values and sensitivities
            (objective_function_value, objective_function_sensitivity), (volume_fraction_value, volume_constraint_sensitivity) = \
                self.get_function_values_and_gradients(design_variables, optimization_iteration_number)
            
            if optimization_iteration_number == 0:
                self.objective_scale_factor = 1.0 / abs(objective_function_value)

            scaled_objective_function = self.objective_scale_factor * objective_function_value
            objective_function_sensitivity[:] = self.objective_scale_factor * objective_function_sensitivity

            constraint_value = \
                (float(volume_fraction_value) - self.volume_fraction_upper_bound) / self.volume_fraction_upper_bound
            
            objective_function_sensitivity_MMA_input = objective_function_sensitivity.reshape((self.precomputed_data.number_of_nodes, 1))
            
            constraint_value_sensitivity = np.zeros((1, volume_constraint_sensitivity.size))
            constraint_value_sensitivity[0, :] = (1.0 / abs(self.volume_fraction_upper_bound)) * volume_constraint_sensitivity

            print(f"Optimization Iteration {optimization_iteration_number:4d}, Objective {scaled_objective_function}, Constraint {constraint_value}") 

            if self.precomputed_data.passive_design_variable_indices.size > 0:
                objective_function_sensitivity_MMA_input = objective_function_sensitivity_MMA_input[variables_for_optimization_mask]
                constraint_value_sensitivity = constraint_value_sensitivity[:,variables_for_optimization_mask]

            perform_design_update(MMA_state, scaled_objective_function, objective_function_sensitivity_MMA_input, 
                                  np.asarray(constraint_value), constraint_value_sensitivity, 
                                  MMA_state.mma_move_limit, optimization_iteration_number)
            
            logged_values = {"Optimization Iteration Number": optimization_iteration_number,
                             "Objective Function Value": objective_function_value,
                             "Volume Fraction Value": volume_fraction_value,
                             "Constraint Upper Bound": self.volume_fraction_upper_bound}
            self.update_logged_values_data_frame(logged_values)
            design_variable_change_measure = np.max(np.abs(MMA_state.current_design_variables - MMA_state.previous_design_variables))
            optimization_iteration_number += 1
        if self.precomputed_data.passive_design_variable_indices.size > 0:
            final_design_variables = np.ones((self.precomputed_data.number_of_nodes,1), dtype=float)
            final_design_variables[variables_for_optimization_mask,:] = MMA_state.current_design_variables
        else:
            final_design_variables = MMA_state.current_design_variables
        filtered_nodal_design_variables[:] = self.design_variable_filter.apply_filter(final_design_variables[:,0])
        filtered_element_design_variables[:] = \
            self.precomputed_data.node_to_element_map @ filtered_nodal_design_variables
        filtered_element_densities = self.precomputed_data.node_to_element_map @ filtered_nodal_design_variables
        self.filtered_and_projected_design_variables[:] = self.projection.apply_projection(filtered_element_densities)
        self.element_densities[:] = self.projection.apply_projection(filtered_element_design_variables)

        _, _, _, _ = self.run_forward_analysis() 
        current_working_directory = os.getcwd()[:]
        os.chdir(self.directory_name)
        Final_design_variables = MMA_state.current_design_variables
        np.save('Final_design_variables.npy', Final_design_variables)
        os.chdir(current_working_directory)


    def check_gradients_MMA_input(self, initial_design_variables, perturbation=1.0e-7):
        '''Final check of the gradients before the use of them in the MMA. Numerical check via FD for the objective function 
            (scaled or unscaled) and the volume constraint.

        Args:
            initial_design_variables: A numpy array of initial design variables.
            perturbation: A float representing the finite difference step size to use.

        Returns:
            .txt output files containing the numerical and analytical objective function gradient and their percent difference, 
        '''
        number_of_design_variables = initial_design_variables.size
        num_design_variables_to_check = min(8, number_of_design_variables)
        (_, objective_gradient), (_, constraint_gradient) = \
            self.get_function_values_and_gradients(initial_design_variables.ravel(),
                                                optimization_iteration_number=0)
        
        numerical_objective_gradients = np.zeros((objective_gradient.size, ), dtype=float)
        design_variables = initial_design_variables.copy()
        for variable_index in range(num_design_variables_to_check):
            design_variables[:] = initial_design_variables[:]
            design_variables[variable_index] += 2.0 * perturbation
            (objective_value_fwd2, _), (constraint_value_fwd2, _) = \
                self.get_function_values_and_gradients(design_variables.ravel(),
                                                        optimization_iteration_number=0)

            design_variables[:] = initial_design_variables[:]
            design_variables[variable_index] += perturbation
            (objective_value_fwd, _), (constraint_value_fwd, _) = \
                self.get_function_values_and_gradients(design_variables.ravel(),
                                                        optimization_iteration_number=0)

            design_variables[:] = initial_design_variables[:]
            design_variables[variable_index] -= perturbation
            (objective_value_bwd, _), (constraint_value_bwd, _) = \
                self.get_function_values_and_gradients(design_variables.ravel(),
                                                        optimization_iteration_number=0)

            design_variables[:] = initial_design_variables[:]
            design_variables[variable_index] -= 2.0 * perturbation
            (objective_value_bwd2, _), (constraint_value_bwd2, _) = \
                self.get_function_values_and_gradients(design_variables.ravel(),
                                                        optimization_iteration_number=0)
            
            numerical_objective_gradient = (objective_value_bwd2 - objective_value_fwd2 + 8.0 * (objective_value_fwd - objective_value_bwd)) / (12.0 * perturbation)
            numerical_constraint_gradient = (constraint_value_bwd2 - constraint_value_fwd2 + 8.0 * (constraint_value_fwd - constraint_value_bwd)) / (12.0 * perturbation)
            numerical_objective_gradients[variable_index] = numerical_objective_gradient

            percent_difference = 100.0 * (objective_gradient[variable_index] - numerical_objective_gradient) / abs(numerical_objective_gradient)
            print(f"Objective : A {objective_gradient[variable_index]:0.3e}, N {numerical_objective_gradient:0.3e}, PD {percent_difference:0.4f}")

            percent_difference = 100.0 * (constraint_gradient[variable_index] - numerical_constraint_gradient) / abs(numerical_constraint_gradient)
            print(f" || Constraint: A {constraint_gradient[variable_index]:0.3e}, N {numerical_constraint_gradient:0.3e}, PD {percent_difference:0.4f}")

        np.savetxt('numerical_objective_gradient.txt', numerical_objective_gradients)
        np.savetxt('analytical_objective_gradient.txt', objective_gradient)
        np.savetxt('percent_difference.txt', 100.0 * (objective_gradient - numerical_objective_gradients) / np.maximum(np.abs(numerical_objective_gradients), 1.0e-15))
        print('Done with MMA gradient checker')

    def run_forward_analysis(self):
        '''
        This function conducts the complete forward analysis of the domain considering von Mises plasticity. This includes the time stepping, 
        displacement control (the Newton-Raphson iterations), and the output of the results.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
                   
        Returns:
            The objective function, the nodal displacements, and the reaction forces.
        ''' 
        total_time = 1.0
        time_increment = total_time / float(self.number_of_time_steps)
  
    
        nodal_displacement_output_array = np.zeros((self.precomputed_data.number_of_nodes, self.precomputed_data.number_of_dofs_per_node), dtype=float)
        applied_displacements_x = []
        applied_displacements_y = []
        reaction_forces = []
        objective_function_value = np.zeros((1,), dtype=float)
       
        #Computing the interpolation function values and derivatives for each optimization iteration
        unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()
        self.elastic_interpolation_function_values = {}
        self.plastic_interpolation_function_values = {}
        self.elastic_interpolation_function_derivatives = {}
        self.plastic_interpolation_function_derivatives = {}
        for number_of_vertices in unique_numbers_of_vertices:
            number_of_quadrature_pts_per_element_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
            densities_by_quadrature_point = np.column_stack([self.filtered_and_projected_design_variables_per_elem_type[number_of_vertices] \
                                                             for _ in range(number_of_quadrature_pts_per_element_type)]).ravel()
            self.elastic_interpolation_function_values[number_of_vertices] = self.elastic_material_interpolation.get_value(densities_by_quadrature_point)
            self.plastic_interpolation_function_values[number_of_vertices] = self.plastic_material_interpolation.get_value(densities_by_quadrature_point)
            self.elastic_interpolation_function_derivatives[number_of_vertices] = self.elastic_material_interpolation.get_derivative(densities_by_quadrature_point)
            self.plastic_interpolation_function_derivatives[number_of_vertices] = self.plastic_material_interpolation.get_derivative(densities_by_quadrature_point)

        self.state = polyplas_utils.get_initial_plastic_state(self.precomputed_data)
        self.states = []
        
        for time_step_index in range(self.number_of_time_steps):
            current_time = (1.0 + time_step_index) * time_increment
            print(f"\nCurrent time step {current_time}")
            if (self.precomputed_data.unique_nonzero_dirichlet_dof_indices_x is not None) and \
                (self.precomputed_data.unique_nonzero_dirichlet_dof_indices_y is not None): #x and y applied displacements
                current_applied_displacement_x = current_time * self.total_applied_displacement_x
                current_applied_displacement_y = current_time * self.total_applied_displacement_y
            elif self.precomputed_data.unique_nonzero_dirichlet_dof_indices_x is not None: #then applied x displacement
                current_applied_displacement_x = current_time * self.total_applied_displacement_x
                current_applied_displacement_y = None
            elif self.precomputed_data.unique_nonzero_dirichlet_dof_indices_y is not None: #then applied y displacement
                current_applied_displacement_x = None
                current_applied_displacement_y = current_time * self.total_applied_displacement_y
            

            reaction_force = 0.0
            newton_iteration_number = 0
            initial_residual_norm = 0.0
            relative_residual_norm = 1.0
            print(f"\nBeginning time step {time_step_index}")
            while relative_residual_norm > 1.0e-7:
                previous_nodal_displacements = self.state.nodal_displacement_vector.copy() 
                residual_norm, reaction_force = self.update_state(current_applied_displacement_x, 
                                                                  current_applied_displacement_y, line_search_step_size = 1.0) 
                if newton_iteration_number == 0:
                    initial_residual_norm = max(residual_norm, 1.0)
                relative_residual_norm = residual_norm / initial_residual_norm
                if newton_iteration_number > 0:
                    line_search_step_size = 0.8
                    max_line_search_iterations = 10
                    for i in range(max_line_search_iterations):
                        self.state.nodal_displacement_vector = previous_nodal_displacements.copy()
                        residual_norm, _ = self.update_state(current_applied_displacement_x, 
                                                             current_applied_displacement_y, 
                                                             line_search_step_size = line_search_step_size)
                        if residual_norm < previous_residual_norm:
                            break
                        line_search_step_size *= line_search_step_size
                previous_residual_norm = residual_norm
                print(f"newton iteration = {newton_iteration_number}, relative residual norm = {relative_residual_norm:0.3e}, residual norm = {residual_norm:0.3e}")
                newton_iteration_number += 1
                if newton_iteration_number > 25:
                    raise ValueError("Newton-Raphson iterations did not converge.")
            applied_displacements_x.append(current_applied_displacement_x)
            applied_displacements_y.append(current_applied_displacement_y)
            reaction_forces.append(reaction_force)
            nodal_displacement_output_array[:, :] = self.state.nodal_displacement_vector.reshape((self.precomputed_data.number_of_nodes, 
                                                                                                  self.precomputed_data.number_of_dofs_per_node))
            nodal_densities = self.lumped_mass_projection.apply_projection(self.element_densities)
            point_data = {"nodal_displacements": nodal_displacement_output_array, "Density": nodal_densities}
            unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()
            average_element_stress = {}
            average_element_pressure = {}
            average_accumulated_plastic_strain = {}
            average_element_plastic_strain = {}
            average_von_mises_stress = {}
            von_mises_stress = {}
            stress_xx = []
            stress_yy = []
            stress_xy = []
            stress_zz = []
            von_mises_stress_for_output = []
            pressure = []
            plastic_strain_xx = []
            plastic_strain_yy = []
            plastic_strain_xy = []
            accumulated_plastic_strain = []
            
            
            #Computing the update of the plastic work objective function value
            for number_of_vertices in unique_numbers_of_vertices:
                deviatoric_stress_tensor_sum = \
                    self.state.current_deviatoric_stress_tensors[number_of_vertices] + \
                    self.state.previous_deviatoric_stress_tensors[number_of_vertices]
                
                plastic_strain_difference = \
                    self.state.current_plastic_strain_tensors[number_of_vertices] - \
                    self.state.previous_plastic_strain_tensors[number_of_vertices]
                
                JxW = self.precomputed_data.jacobian_determinant_x_quadrature_weights \
                    [number_of_vertices]
                          
                objective_function_value += oe.contract('...,eqij,eqij,eq->', 
                                                        -0.5, 
                                                        deviatoric_stress_tensor_sum, 
                                                        plastic_strain_difference, 
                                                        JxW) 
                
                von_mises_stress_temp = oe.contract("eqij,eqij->eq", 
                                        self.state.current_deviatoric_stress_tensors[number_of_vertices],
                                        self.state.current_deviatoric_stress_tensors[number_of_vertices])
                von_mises_stress[number_of_vertices] = ((3.0/2.0) * von_mises_stress_temp)**0.5
                
            if time_step_index == self.number_of_time_steps - 1:    #if at final time step
                for number_of_vertices in unique_numbers_of_vertices:
                    average_element_stress[number_of_vertices] = np.mean(self.state.current_stress_tensors[number_of_vertices][:,:,:,:], axis=1)  
                    average_accumulated_plastic_strain[number_of_vertices] = np.mean(self.state.current_accumulated_plastic_strains[number_of_vertices][:,:], axis=1)
                    average_element_pressure[number_of_vertices] = np.mean(self.state.current_pressure[number_of_vertices][:,:], axis=1)
                    average_element_plastic_strain[number_of_vertices] = np.mean(self.state.current_plastic_strain_tensors[number_of_vertices][:,:,:,:], axis=1)
                    average_von_mises_stress[number_of_vertices] = np.mean(von_mises_stress[number_of_vertices][:,:], axis=1)
                    np.array(stress_xx.append(average_element_stress[number_of_vertices][:,0,0].ravel()))   
                    np.array(stress_yy.append(average_element_stress[number_of_vertices][:,1,1].ravel()))
                    np.array(stress_xy.append(average_element_stress[number_of_vertices][:,0,1].ravel()))
                    np.array(stress_zz.append(average_element_stress[number_of_vertices][:,2,2].ravel()))
                    np.array(von_mises_stress_for_output.append(average_von_mises_stress[number_of_vertices].ravel()))
                    np.array(pressure.append(average_element_pressure[number_of_vertices].ravel()))
                    np.array(plastic_strain_xx.append(average_element_plastic_strain[number_of_vertices][:,0,0].ravel()))
                    np.array(plastic_strain_yy.append(average_element_plastic_strain[number_of_vertices][:,1,1].ravel()))
                    np.array(plastic_strain_xy.append(average_element_plastic_strain[number_of_vertices][:,0,1].ravel()))
                    np.array(accumulated_plastic_strain.append(average_accumulated_plastic_strain[number_of_vertices].ravel()))

                cell_data = {"stress_xx": stress_xx, "stress_yy": stress_yy, "stress_xy": stress_xy, "pressure": pressure, \
                             "von_Mises_stress": von_mises_stress_for_output, "plastic_strain_xx:": plastic_strain_xx, \
                             "plastic_strain_yy": plastic_strain_yy, "plastic_strain_xy": plastic_strain_xy,\
                             "accumulated_plastic_strain:": accumulated_plastic_strain}
                polyplas_utils.write_vtk_output(self.precomputed_data.nodal_coordinates, self.precomputed_data.element_connectivity_arrays,
                                 optimization_iteration_number=self.optimization_iteration_number, current_time=current_time, 
                                 output_directory = self.directory_name, point_data=point_data, cell_data=cell_data)      
            
            self.states.append(copy.deepcopy(self.state))
            self.state.transfer_state(self.precomputed_data)  
        return  objective_function_value, applied_displacements_x, applied_displacements_y, reaction_forces 
    
    def update_state(self, current_applied_displacement_x, current_applied_displacement_y, line_search_step_size = 1.0):
        '''
        This function returns the n+1 plastic state of the system, given the current n plastic state and current applied displacement.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            current_applied_displacement_x: The current applied displacement in the x-direction.
            current_applied_displacement_y: The current applied displacement in the y-direction.
            line_search_step_size: The predetermined line search step size.

        Returns:
            residual_norm: The norm of the residual vector at the n+1 iteration.
            reaction_force: The reaction force at the n+1 iteration.
        '''
        start_time = perf_counter()
        current_mean_applied_displacement = np.mean(self.state.nodal_displacement_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices])
        current_strain_tensors = {}
        current_volume_avg_volumetric_strain = {}
        unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()
        for number_of_vertices in unique_numbers_of_vertices:
            nodal_displacements_elementwise_per_elem_type = self.state.nodal_displacement_vector[self.precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
            current_volume_avg_volumetric_strain[number_of_vertices] = oe.contract("e,ei,ei->e", 
                                                                                 1.0 / self.precomputed_data.element_volumes_per_elem_type[number_of_vertices], 
                                                                                 self.precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices], 
                                                                                 nodal_displacements_elementwise_per_elem_type)
            current_strain_tensors[number_of_vertices] = oe.contract('eqimn,ei->eqmn', 
                                                                   self.precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices], 
                                                                   nodal_displacements_elementwise_per_elem_type)
        element_residual_vectors, element_jacobian_matrices = self.get_element_quantities(current_strain_tensors, current_volume_avg_volumetric_strain) 


        element_jacobian_temp = []
        element_residual_temp = []
        for number_of_vertices in unique_numbers_of_vertices:       
            np.array(element_jacobian_temp.append(element_jacobian_matrices[number_of_vertices].ravel()))
            np.array(element_residual_temp.append(element_residual_vectors[number_of_vertices].ravel()))

        element_jacobian_data = np.concatenate(element_jacobian_temp)
        element_residual_data = np.concatenate(element_residual_temp)
        coo_matrix_nonzero_dirichlet_data = (element_jacobian_data[([self.precomputed_data.coo_matrix_nonzero_dirichlet_dof_indices])]).reshape((-1,))  

        vector_shape = (self.precomputed_data.number_of_global_dofs, 1)

        if (self.precomputed_data.unique_nonzero_dirichlet_dof_indices_x is not None) and (self.precomputed_data.unique_nonzero_dirichlet_dof_indices_y is not None): #x and y applied displacements
            current_mean_applied_displacement_x = np.mean(self.state.nodal_displacement_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices_x])
            current_mean_applied_displacement_y = np.mean(self.state.nodal_displacement_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices_y])
            applied_displacement_increment_x = current_applied_displacement_x - current_mean_applied_displacement_x
            applied_displacement_increment_y = current_applied_displacement_y - current_mean_applied_displacement_y
            coo_matrix_nonzero_dirichlet_data_x = (element_jacobian_data[([self.precomputed_data.coo_matrix_nonzero_dirichlet_dof_indices_x])]).reshape((-1,)) 
            coo_matrix_nonzero_dirichlet_data_y = (element_jacobian_data[([self.precomputed_data.coo_matrix_nonzero_dirichlet_dof_indices_y])]).reshape((-1,)) 
            nonzero_dirichlet_rhs_contribution_x = (-1.0 * applied_displacement_increment_x) * coo_matrix((coo_matrix_nonzero_dirichlet_data_x, self.precomputed_data.coo_matrix_nonzero_dirichlet_indices_x), shape=vector_shape).toarray().ravel()
            nonzero_dirichlet_rhs_contribution_y = (-1.0 * applied_displacement_increment_y) * coo_matrix((coo_matrix_nonzero_dirichlet_data_y, self.precomputed_data.coo_matrix_nonzero_dirichlet_indices_y), shape=vector_shape).toarray().ravel()
            global_internal_force_vector = coo_matrix((element_residual_data, self.precomputed_data.internal_force_indices_array), shape=vector_shape).toarray().ravel()
            right_hand_side_vector = (nonzero_dirichlet_rhs_contribution_x + nonzero_dirichlet_rhs_contribution_y) - global_internal_force_vector
            right_hand_side_vector[self.precomputed_data.unique_zero_dirichlet_dof_indices] = 0.0
            reaction_force = np.sum(global_internal_force_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices])  
            right_hand_side_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices_x] = applied_displacement_increment_x
            right_hand_side_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices_y] = applied_displacement_increment_y
        elif self.precomputed_data.unique_nonzero_dirichlet_dof_indices_x is not None: #then applied x displacement
            current_mean_applied_displacement_x = np.mean(self.state.nodal_displacement_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices])
            applied_displacement_increment_x = current_applied_displacement_x - current_mean_applied_displacement_x
            coo_matrix_nonzero_dirichlet_data = (element_jacobian_data[([self.precomputed_data.coo_matrix_nonzero_dirichlet_dof_indices])]).reshape((-1,)) 
            nonzero_dirichlet_rhs_contribution = (-1.0 * applied_displacement_increment_x) * coo_matrix((coo_matrix_nonzero_dirichlet_data, self.precomputed_data.coo_matrix_nonzero_dirichlet_indices), shape=vector_shape).toarray().ravel()
            global_internal_force_vector = coo_matrix((element_residual_data, self.precomputed_data.internal_force_indices_array), shape=vector_shape).toarray().ravel()
            right_hand_side_vector = nonzero_dirichlet_rhs_contribution - global_internal_force_vector
            right_hand_side_vector[self.precomputed_data.unique_zero_dirichlet_dof_indices] = 0.0
            reaction_force = np.sum(global_internal_force_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices])  
            right_hand_side_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices] = applied_displacement_increment_x
        elif self.precomputed_data.unique_nonzero_dirichlet_dof_indices_y is not None: #then applied y displacement
            current_mean_applied_displacement_y = np.mean(self.state.nodal_displacement_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices])
            applied_displacement_increment_y = current_applied_displacement_y - current_mean_applied_displacement_y
            coo_matrix_nonzero_dirichlet_data = (element_jacobian_data[([self.precomputed_data.coo_matrix_nonzero_dirichlet_dof_indices])]).reshape((-1,)) 
            nonzero_dirichlet_rhs_contribution = (-1.0 * applied_displacement_increment_y) * coo_matrix((coo_matrix_nonzero_dirichlet_data, self.precomputed_data.coo_matrix_nonzero_dirichlet_indices), shape=vector_shape).toarray().ravel()
            global_internal_force_vector = coo_matrix((element_residual_data, self.precomputed_data.internal_force_indices_array), shape=vector_shape).toarray().ravel()
            right_hand_side_vector = nonzero_dirichlet_rhs_contribution - global_internal_force_vector
            right_hand_side_vector[self.precomputed_data.unique_zero_dirichlet_dof_indices] = 0.0
            reaction_force = np.sum(global_internal_force_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices])  
            right_hand_side_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices] = applied_displacement_increment_y

        element_jacobian_data[self.precomputed_data.coo_matrix_all_dirichlet_dof_indices] = 0.0
        element_jacobian_data[self.precomputed_data.coo_matrix_unique_dirichlet_diagonal_dof_indices] = 1.0

        global_jacobian_matrix = coo_matrix((element_jacobian_data, (self.precomputed_data.coo_matrix_row_indices, self.precomputed_data.coo_matrix_column_indices))).tocsc()

        assembly_time = perf_counter() - start_time

        start_time = perf_counter()
        nodal_displacement_update = spsolve(global_jacobian_matrix, right_hand_side_vector)
        solve_time = perf_counter() - start_time

        print(f"Assembly required {assembly_time:0.3e} seconds. Solving the linear system required {solve_time:0.3e} seconds.")

        self.state.nodal_displacement_vector += nodal_displacement_update * line_search_step_size

        right_hand_side_vector[self.precomputed_data.unique_zero_dirichlet_dof_indices] = 0.0
        right_hand_side_vector[self.precomputed_data.unique_nonzero_dirichlet_dof_indices] = 0.0
        residual_norm = np.linalg.norm(right_hand_side_vector)

        return residual_norm, reaction_force
    
    def get_element_quantities(self, current_strain_tensors, current_volume_avg_volumetric_strain):
        '''
        This function returns the elementwise residual vectors and jacobian matrices, and the updated plastic state of the system.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            current_strain_tensors: The current strain tensors at the ith iteration.
            current_volume_avg_volumetric_strain: The current volume averaged volumetric strain at the ith step.

        Returns:
            element_residual_vectors: The elementwise residual vectors at i+1 iteration.
            element_jacobian_matrices: The elementwise jacobian matrices at i+1 iteration.
        '''
        unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()
        total_strain_tensors = {}
        for number_of_vertices in unique_numbers_of_vertices:
            number_of_elements = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
            total_number_of_quad_pts_per_element_type = number_of_elements * self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
            total_strain_tensors[number_of_vertices] = np.zeros((total_number_of_quad_pts_per_element_type,3,3), dtype=float)
            total_strain_tensors[number_of_vertices][:, :2, :2] = current_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 2, 2))

        constitutive_tensors = self.get_J2_update(total_strain_tensors, current_volume_avg_volumetric_strain)
        element_jacobian_matrices = {}
        element_residual_vectors = {}
     
        for number_of_vertices in unique_numbers_of_vertices:
            elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices]
            total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
            number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
            elastic_interpolation_reshaped = elastic_interpolation.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type))
            elastic_interpolation_per_element = elastic_interpolation_reshaped[:,0]
            virtual_displacement_symmetric_gradients = self.precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices][:,:,:,:,:]
            jacobian_determinant_x_quadrature_weights = self.precomputed_data.jacobian_determinant_x_quadrature_weights[number_of_vertices][:,:]
            current_deviatoric_stress_tensors = self.state.current_deviatoric_stress_tensors[number_of_vertices][:, :, :2, :2]
            virtual_displacement_volume_weighted_divergence = self.precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices][:,:]
            element_volumes_per_elem_type = self.precomputed_data.element_volumes_per_elem_type[number_of_vertices][:]
            bulk_modulus = self.precomputed_data.bulk_modulus

            pressure = self.state.current_pressure[number_of_vertices][:,0]

            element_jacobian_matrices[number_of_vertices] = oe.contract('eqmij,eqijkl,eqnkl,eq->emn',
                                                                        virtual_displacement_symmetric_gradients,
                                                                        constitutive_tensors[number_of_vertices],
                                                                        virtual_displacement_symmetric_gradients,
                                                                        jacobian_determinant_x_quadrature_weights, 
                                                                        optimize='optimal')
            
            element_jacobian_matrices[number_of_vertices] += oe.contract("...,e,em,en->emn", 
                                                                         bulk_modulus, 
                                                                         elastic_interpolation_per_element / element_volumes_per_elem_type, 
                                                                         virtual_displacement_volume_weighted_divergence,
                                                                         virtual_displacement_volume_weighted_divergence, 
                                                                         optimize='optimal')
            
            element_residual_vectors[number_of_vertices] = oe.contract('eqmij,eqij,eq->em',
                                                                        virtual_displacement_symmetric_gradients,
                                                                        current_deviatoric_stress_tensors,
                                                                        jacobian_determinant_x_quadrature_weights, 
                                                                        optimize='optimal')
            element_residual_vectors[number_of_vertices] += oe.contract('e,em->em',
                                                                        pressure, 
                                                                        virtual_displacement_volume_weighted_divergence, 
                                                                        optimize='optimal')
            
        return element_residual_vectors, element_jacobian_matrices
    
    def get_J2_update(self, total_strain_tensor, current_volume_avg_volumetric_strain):
        '''
        This function computes the return mapping algorithm for von Mises (J2) plasticity.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            total_strain_tensor: The total strain tensor computed by the current displacement field.
            current_volume_avg_volumetric_strain: The current volume averaged volumetric strain at the ith step.

        Returns:
            constitutive tensor: The consistent tangent constitutive tensor at the ith iteration.
            (computes and stores update of the self.state variables)
        '''

        constitutive_tensors = {}
        stress_tensors = {}
        deviatoric_stress_tensors = {}
        pressure_stress = {}
        plastic_strain_tensors = {}

        unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()       
    
        for number_of_vertices in unique_numbers_of_vertices:
            elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices]
            plastic_interpolation = self.plastic_interpolation_function_values[number_of_vertices]

            total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
            number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
            total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type
            trial_elastic_strain_tensor = total_strain_tensor[number_of_vertices] - self.state.previous_plastic_strain_tensors[number_of_vertices]\
                                                                                .reshape((total_number_of_quad_pts_per_element_type,3,3))
            trial_deviatoric_elastic_strain = oe.contract("ijkl,qkl->qij", self.precomputed_data.deviatoric_projection_tensor, 
                                                          trial_elastic_strain_tensor, 
                                                          optimize='optimal')
            trial_deviatoric_constitutive_tensor = oe.contract("...,q,ijkl->qijkl", 
                                                               2.0*self.precomputed_data.shear_modulus, elastic_interpolation, 
                                                               self.precomputed_data.deviatoric_projection_tensor, 
                                                               optimize='optimal')
            trial_deviatoric_stress_tensor = oe.contract("...,q,qij->qij", 
                                                         2.0*self.precomputed_data.shear_modulus, 
                                                         elastic_interpolation, 
                                                         trial_deviatoric_elastic_strain, optimize='optimal')
            trial_deviatoric_stress_norms = np.linalg.norm(trial_deviatoric_stress_tensor, axis=(1, 2), ord='fro')
            trial_current_stress = ((3.0/2.0)**0.5) * trial_deviatoric_stress_norms
            penalized_initial_yield_stress = plastic_interpolation * self.precomputed_data.initial_yield_stress
            penalized_hardening_modulus = elastic_interpolation * self.precomputed_data.hardening_modulus
            current_yield_stress = penalized_initial_yield_stress + penalized_hardening_modulus *\
                                   self.state.previous_accumulated_plastic_strains[number_of_vertices]\
                                    .reshape((total_number_of_quad_pts_per_element_type,))
            
            #Yield function:
            trial_yield_function = trial_current_stress - current_yield_stress
            elastic = trial_yield_function <= 1.0e-10   #creating elastic mask
            plastic = ~elastic

            #Variables for update computations
            stress_tensors[number_of_vertices] = self.state.current_stress_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
            deviatoric_stress_tensors[number_of_vertices] = self.state.current_deviatoric_stress_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
            plastic_strain_tensors[number_of_vertices] = self.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
            previous_accumulated_plastic_strains = self.state.previous_accumulated_plastic_strains[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,))
            accumulated_plastic_strains = self.state.current_accumulated_plastic_strains[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,))
            constitutive_tensors[number_of_vertices] = np.zeros((total_number_of_quad_pts_per_element_type, 2, 2, 2, 2), dtype=float)
            elastic_interpolation_reshaped = elastic_interpolation.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type))
            pressure_stress[number_of_vertices] = oe.contract("...,eq,e->eq", 
                                                              self.precomputed_data.bulk_modulus, 
                                                              elastic_interpolation_reshaped, 
                                                              current_volume_avg_volumetric_strain[number_of_vertices]).ravel()

            #Elastic update
            previous_plastic_strain_tensors = self.state.previous_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))
            stress_tensors[number_of_vertices][elastic, :, :] = trial_deviatoric_stress_tensor[elastic, :, :] 
            stress_tensors[number_of_vertices][elastic, :, :] += oe.contract('q,ij->qij', 
                                                                             pressure_stress[number_of_vertices][elastic], 
                                                                             np.eye(3), 
                                                                             optimize='optimal')
            deviatoric_stress_tensors[number_of_vertices][elastic, :, :] = trial_deviatoric_stress_tensor[elastic, :, :]
            
            plastic_strain_tensors[number_of_vertices][elastic, :, :] = previous_plastic_strain_tensors[elastic, :, :]
            accumulated_plastic_strains[elastic] = previous_accumulated_plastic_strains[elastic]
            constitutive_tensors[number_of_vertices][elastic, :, :, :, :] = trial_deviatoric_constitutive_tensor[elastic, :2, :2, :2, :2]
            plastic_multiplier_increments = np.zeros((total_number_of_quad_pts_per_element_type,), dtype=float)

            #Plastic update      
            if np.any(plastic):
                plastic_indices = np.argwhere(plastic).ravel()
                print("In plastic loading. Number of plastic points: ", len(plastic_indices))
                plastic_multiplier_increments[plastic_indices] = trial_yield_function[plastic_indices] / \
                                                                (3.0 * self.precomputed_data.shear_modulus*elastic_interpolation[plastic_indices] \
                                                                 + penalized_hardening_modulus[plastic_indices])
                yield_surface_normals = trial_deviatoric_stress_tensor[plastic_indices, :, :] / trial_deviatoric_stress_norms[plastic_indices, None, None]
                plastic_step_plastic_strain_increments = (((3.0/2.0)**0.5) * plastic_multiplier_increments[plastic_indices, None, None]) * yield_surface_normals
                plastic_strain_tensors[number_of_vertices][plastic_indices, :, :] = previous_plastic_strain_tensors[plastic_indices, :, :] + plastic_step_plastic_strain_increments
                accumulated_plastic_strains[plastic_indices] = previous_accumulated_plastic_strains[plastic_indices] + plastic_multiplier_increments[plastic_indices]
                
                stress_tensors[number_of_vertices][plastic_indices, :, :] = trial_deviatoric_stress_tensor[plastic_indices, :, :] \
                                                                            - ((2.0 * (self.precomputed_data.shear_modulus*elastic_interpolation[plastic_indices,None,None])) \
                                                                               * plastic_step_plastic_strain_increments)
                stress_tensors[number_of_vertices][plastic_indices, :, :] += oe.contract('q,ij->qij', pressure_stress[number_of_vertices][plastic_indices], 
                                                                                         np.eye(3), 
                                                                                         optimize='optimal')

                deviatoric_stress_tensors[number_of_vertices][plastic_indices, :, :] = trial_deviatoric_stress_tensor[plastic_indices, :, :]\
                                                                            - ((2.0 * (self.precomputed_data.shear_modulus*elastic_interpolation[plastic_indices,None,None])) \
                                                                               * plastic_step_plastic_strain_increments)

                c1 = 2.0 * (self.precomputed_data.shear_modulus*elastic_interpolation[plastic_indices]) * \
                    (1.0 - (3.0 * (self.precomputed_data.shear_modulus*elastic_interpolation[plastic_indices])\
                      * plastic_multiplier_increments[plastic_indices]) / trial_current_stress[plastic_indices])
                A = oe.contract('q,ijkl->qijkl', 
                                c1, 
                                self.precomputed_data.deviatoric_projection_tensor[:2, :2, :2, :2], 
                                optimize='optimal')
                c2 = 6.0 * (self.precomputed_data.shear_modulus*elastic_interpolation[plastic_indices])**2 \
                    * (plastic_multiplier_increments[plastic_indices] / trial_current_stress[plastic_indices] \
                    - 1.0 / (3.0 * (self.precomputed_data.shear_modulus*elastic_interpolation[plastic_indices]) + penalized_hardening_modulus[plastic_indices]))
                B = oe.contract('q,qij,qkl->qijkl', 
                                c2, 
                                yield_surface_normals[:, :2, :2], 
                                yield_surface_normals[:, :2, :2], 
                                optimize='optimal')
                constitutive_tensors[number_of_vertices][plastic_indices, :, :, :, :] += (A + B)


            self.state.current_total_strain_tensors[number_of_vertices][:,:, :, :] = total_strain_tensor[number_of_vertices].reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, 3, 3))
            self.state.current_stress_tensors[number_of_vertices][:, :, :, :] = stress_tensors[number_of_vertices].reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, 3, 3))
            self.state.current_deviatoric_stress_tensors[number_of_vertices][:, :, :, :] = deviatoric_stress_tensors[number_of_vertices].reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, 3, 3))
            self.state.current_pressure[number_of_vertices][:, :] = pressure_stress[number_of_vertices].reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type))
            self.state.current_plastic_strain_tensors[number_of_vertices][:, :, :, :] = plastic_strain_tensors[number_of_vertices].reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, 3, 3))
            self.state.current_accumulated_plastic_strains[number_of_vertices][:, :] = accumulated_plastic_strains.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type))
            self.state.plastic_multiplier_increments[number_of_vertices][:, :] = plastic_multiplier_increments.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type))

            constitutive_tensors[number_of_vertices] = constitutive_tensors[number_of_vertices].reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, 2, 2, 2, 2))

        return constitutive_tensors       

    def compute_objective_function_sensitivity(self):
        '''
        Computes the path dependent sensitivity of the augmented objective function of the plastic work with respect 
        to the design variables.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            
        Returns:
            objective_function_sensitivity_wrt_design_variables.
        '''
        objective_function_sensitivity_wrt_densities = np.zeros((self.precomputed_data.number_of_elements), dtype=float)
        
        total_number_of_time_steps = self.number_of_time_steps
        self.state =  self.states[total_number_of_time_steps-1]
        

        #Computing the F_c and F_u r.h.s. of the Schur complement for final time step, n
        F_c, F_u = self.compute_sensitivity_contribution(total_number_of_time_steps-1, 
                                                              objective_function_sensitivity_wrt_densities, 
                                                              F_c=None, F_u = None, at_final_time_step=True)
        
        #Computing the F_c and F_u and updating the adjoint contributions for the remainder of the time steps
        #Note: The objective function sensitivity is being updated and computed in the compute_sensitivity_contribution function
        for time_step_index in range(total_number_of_time_steps-2, -1, -1):
            self.state = self.states[time_step_index]
            F_c, F_u = self.compute_sensitivity_contribution(time_step_index, 
                                                         objective_function_sensitivity_wrt_densities, 
                                                         F_c, F_u, at_final_time_step=False)
            
        objective_function_sensitivity_wrt_filtered_densities = self.projection.apply_chain_rule(objective_function_sensitivity_wrt_densities)
        nodal_sensitivity = \
            self.precomputed_data.node_to_element_map.T @ objective_function_sensitivity_wrt_filtered_densities
        objective_function_sensitivity_wrt_design_variables = self.design_variable_filter.apply_chain_rule(nodal_sensitivity)
        
        return -1.0 * objective_function_sensitivity_wrt_design_variables
     
    def compute_sensitivity_contribution(self, time_step_index, objective_function_sensitivity_wrt_densities, F_c, F_u, at_final_time_step: bool):
        '''
        Computes the residual derivatives at the current time_step_index and uses this information to form the Schur complement to solve for the
        local and global adjoint vectors at the current time_step_index. Updates the objective function sensitivity with respect to the design variables.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            time_step_index: The current time step index. 
            objective_function_sensitivity_wrt_densities: The current objective function sensitivity with respect to the densities (is updated during the function call).
            F_c: The current path dependent term in the adjoint equation for the stress tensor.
            F_u: The current path dependent term in the adjoint equation for the displacement field.
            at_final_time_step: A boolean flag indicating whether the current time step is the final time step or not.
            
        Returns:
            F_c: A portion of the F_c^{i-1} update of the term consisting of the partial of the local residual with respect to the previous local variables.
            F_u: A portion of the F_u^{i-1} update of the term consisting of the partial of the local residual with respect to the previous local variables.
            (Terms F_c^{i-1} and F_u^{i-1} are PARTIALLY updated and are used in the next time step)
        '''
        if at_final_time_step:
            F_c_final_time_step = {}
            F_u_final_time_step = {}
            F_c_updated = {}
            F_u_updated = {}
        else:
            F_c_updated = {}
            F_u_updated = {}
                       
        element_jacobian_data = []
        global_adjoint_vector_rhs_data = []

        internal_force_derivative_wrt_densities = {}
        internal_force_derivative_wrt_current_displacements = {}
        internal_force_derivative_wrt_current_local_variables = {}

        local_residual_derivative_wrt_densities = {}
        local_residual_derivative_wrt_previous_displacements = {}
        local_residual_derivative_wrt_current_displacements = {}
        local_residual_derivative_wrt_previous_local_variables = {}
        local_residual_derivative_wrt_current_local_variables = {}
        inverse_local_residual_derivative_wrt_current_local_variables = {}

        plastic_work_obj_derivative_wrt_densities = {}
        plastic_work_obj_derivative_wrt_current_displacements = {}
        plastic_work_obj_derivative_wrt_current_local_variables = {}


        unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()
        for number_of_vertices in unique_numbers_of_vertices:

            internal_force_derivative_wrt_densities[number_of_vertices], internal_force_derivative_wrt_current_displacements[number_of_vertices],\
                  internal_force_derivative_wrt_current_local_variables[number_of_vertices] = self.compute_global_residual_sensitivity_per_elem_type(number_of_vertices)
            
            number_of_local_dofs = 6 #accumulated plastic strain, plastic multiplier, 4 components of plastic strain tensor
            number_of_dofs_per_element_type = self.precomputed_data.number_of_dofs_per_element_type[number_of_vertices]

            total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
            number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
            total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type

            local_residual_derivative_wrt_previous_displacements[number_of_vertices] = np.zeros((total_number_of_quad_pts_per_element_type, 
                                                                                                 number_of_local_dofs, number_of_dofs_per_element_type), dtype=float)

            local_residual_derivative_wrt_densities[number_of_vertices], local_residual_derivative_wrt_current_displacements[number_of_vertices],\
               local_residual_derivative_wrt_current_local_variables[number_of_vertices], local_residual_derivative_wrt_previous_local_variables[number_of_vertices] \
               = self.compute_local_residual_sensitivity_per_elem_type(number_of_vertices)
            
            plastic_work_obj_derivative_wrt_densities[number_of_vertices], plastic_work_obj_derivative_wrt_current_displacements[number_of_vertices],\
                plastic_work_obj_derivative_wrt_current_local_variables[number_of_vertices] = self.get_plastic_work_obj_partials_per_elem_type(number_of_vertices,
                                                                                                                                                time_step_index, 
                                                                                                                                                at_final_time_step)


            #Now compose Schur complement to find the global and local adjoint vectors at each time step
            inverse_local_residual_derivative_wrt_current_local_variables[number_of_vertices] = np.linalg.inv(local_residual_derivative_wrt_current_local_variables[number_of_vertices])\
                .reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_dofs, number_of_local_dofs)) #qij ->eqij
            local_residual_derivative_wrt_current_displacements[number_of_vertices] = local_residual_derivative_wrt_current_displacements[number_of_vertices].\
                reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, 
                         number_of_local_dofs, 
                         number_of_dofs_per_element_type))
            
            element_jacobian_matrices = oe.contract("...,eqdn,eqcd,eqmc->emn", -1.0, 
                                                  local_residual_derivative_wrt_current_displacements[number_of_vertices], 
                                                  inverse_local_residual_derivative_wrt_current_local_variables[number_of_vertices],
                                                   internal_force_derivative_wrt_current_local_variables[number_of_vertices], optimize='optimal')
        
            element_jacobian_matrices += internal_force_derivative_wrt_current_displacements[number_of_vertices]
           
            if at_final_time_step:
                F_c_final_time_step[number_of_vertices] = -1.0 * plastic_work_obj_derivative_wrt_current_local_variables[number_of_vertices]\
                    .reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_dofs))
                F_u_final_time_step[number_of_vertices] = -1.0 * plastic_work_obj_derivative_wrt_current_displacements[number_of_vertices]   
                global_adjoint_vector_rhs_data_per_element_type = oe.contract("...,eqcm,eqdc,eqd->em", 
                                                                            -1.0, 
                                                                            local_residual_derivative_wrt_current_displacements[number_of_vertices], 
                                                                            inverse_local_residual_derivative_wrt_current_local_variables[number_of_vertices], 
                                                                            F_c_final_time_step[number_of_vertices], optimize='optimal')
                global_adjoint_vector_rhs_data_per_element_type += F_u_final_time_step[number_of_vertices]
            else:
                #Use the updated F_c path dependent term that is an input argument
                F_c[number_of_vertices] -= plastic_work_obj_derivative_wrt_current_local_variables[number_of_vertices]\
                    .reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_dofs)) 
                F_u[number_of_vertices] -= plastic_work_obj_derivative_wrt_current_displacements[number_of_vertices]
                global_adjoint_vector_rhs_data_per_element_type = oe.contract("...,eqcm,eqdc,eqd->em", -1.0, 
                                                                            local_residual_derivative_wrt_current_displacements[number_of_vertices], 
                                                                            inverse_local_residual_derivative_wrt_current_local_variables[number_of_vertices],
                                                                             F_c[number_of_vertices], optimize='optimal')
                global_adjoint_vector_rhs_data_per_element_type += F_u[number_of_vertices]

            np.array(element_jacobian_data.append(element_jacobian_matrices.ravel()))
            np.array(global_adjoint_vector_rhs_data.append(global_adjoint_vector_rhs_data_per_element_type.ravel()))


        element_jacobian_data_final = np.concatenate(element_jacobian_data)
        global_adjoint_vector_rhs_final = np.concatenate(global_adjoint_vector_rhs_data)

        element_jacobian_data_final[self.precomputed_data.coo_matrix_all_dirichlet_dof_indices] = 0.0
        element_jacobian_data_final[self.precomputed_data.coo_matrix_unique_dirichlet_diagonal_dof_indices] = 1.0
        
        element_jacobian_matrices_tranpose = coo_matrix((element_jacobian_data_final, 
                                                         (self.precomputed_data.coo_matrix_column_indices, 
                                                          self.precomputed_data.coo_matrix_row_indices))).tocsr()
        solve_function = factorized(element_jacobian_matrices_tranpose)

        vector_shape = (self.precomputed_data.number_of_global_dofs, 1)

        global_adjoint_vector_rhs = coo_matrix((global_adjoint_vector_rhs_final, 
                                                (self.precomputed_data.internal_force_indices_array)), shape = vector_shape).toarray().ravel()
        global_adjoint_vector_rhs[self.precomputed_data.unique_zero_dirichlet_dof_indices] = 0.0
        global_adjoint_vector_rhs[self.precomputed_data.unique_nonzero_dirichlet_dof_indices] = 0.0
        global_adjoint_vector = solve_function(global_adjoint_vector_rhs)

        element_index = 0
        for number_of_vertices in unique_numbers_of_vertices:
            global_adjoint_vector_elementwise = global_adjoint_vector[self.precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
            local_adjoint_vector_rhs = oe.contract("...,eqmc,em->eqc", -1.0, 
                                                 internal_force_derivative_wrt_current_local_variables[number_of_vertices], 
                                                 global_adjoint_vector_elementwise, optimize='optimal')
            if at_final_time_step:
                local_adjoint_vector_rhs += F_c_final_time_step[number_of_vertices]
            else: 
                local_adjoint_vector_rhs += F_c[number_of_vertices]

            number_of_dofs_per_element_type = self.precomputed_data.number_of_dofs_per_element_type[number_of_vertices]
            total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
            number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
            
            local_adjoint_vector = oe.contract("eqdc,eqd->eqc", 
                                               inverse_local_residual_derivative_wrt_current_local_variables[number_of_vertices], 
                                               local_adjoint_vector_rhs, 
                                               optimize='optimal')
            plastic_work_obj_derivative_wrt_densities_elementwise = oe.contract("eq->e", 
                                                                                plastic_work_obj_derivative_wrt_densities[number_of_vertices]
                                                                                .reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)), 
                                                                                optimize='optimal')

            objective_function_sensitivity_wrt_densities[element_index:(element_index+total_number_of_elements_per_elem_type)] += plastic_work_obj_derivative_wrt_densities_elementwise
            objective_function_sensitivity_wrt_densities[element_index:(element_index+total_number_of_elements_per_elem_type)] += oe.contract("em,em->e", 
                                                                                                                                  internal_force_derivative_wrt_densities[number_of_vertices], 
                                                                                                                                  global_adjoint_vector_elementwise, 
                                                                                                                                  optimize='optimal')
            objective_function_sensitivity_wrt_densities[element_index:(element_index+total_number_of_elements_per_elem_type)] += oe.contract("eqm,eqm->e", 
                                                                                                                                            local_residual_derivative_wrt_densities[number_of_vertices]
                                                                                                                                            .reshape((total_number_of_elements_per_elem_type,\
                                                                                                                                            number_of_quadrature_pts_per_elem_type, number_of_local_dofs)), 
                                                                                                                                            local_adjoint_vector, 
                                                                                                                                            optimize='optimal')
            
            F_c_updated[number_of_vertices] = oe.contract("eqdc,eqd->eqc", 
                                                           local_residual_derivative_wrt_previous_local_variables[number_of_vertices].\
                                                           reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, \
                                                           number_of_local_dofs, number_of_local_dofs)), 
                                                           local_adjoint_vector, 
                                                           optimize='optimal')
            F_c_updated[number_of_vertices] *= -1.0

            F_u_updated[number_of_vertices] = oe.contract("eqcm,eqc->em", local_residual_derivative_wrt_previous_displacements[number_of_vertices].\
                                                          reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type,\
                                                          number_of_local_dofs, number_of_dofs_per_element_type)), 
                                                          local_adjoint_vector, 
                                                          optimize='optimal')
            F_u_updated[number_of_vertices] *= -1.0
            element_index += total_number_of_elements_per_elem_type

        return  F_c_updated, F_u_updated
    

    def compute_global_residual_sensitivity_per_elem_type(self, number_of_vertices: int):
        '''
        Computes all of the global residual partial derivatives for the specified polygonal element type.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            number_of_vertices: The number of vertices in the polygon element type.
                    
        Returns:
            internal_force_derivative_wrt_densities_per_elem_type: The derivatives of the global residual with respect to the densities.
            internal_force_derivative_wrt_current_displacements_per_elem_type: The derivatives of the global residual with respect to the current displacements.
            internal_force_derivative_wrt_current_plastic_state_per_elem_type: The derivatives of the global residual with respect to the current plastic state.
        '''
        

        elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices]
        

        plastic_interpolation = self.plastic_interpolation_function_values[number_of_vertices]
        elastic_interpolation_derivative = self.elastic_interpolation_function_derivatives[number_of_vertices]

        total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type
        elastic_interpolation_reshaped = elastic_interpolation.reshape((total_number_of_elements_per_elem_type, \
                                                                         number_of_quadrature_pts_per_elem_type))
        elastic_interpolation_for_computations = elastic_interpolation_reshaped[:,0]
      
        number_of_local_dofs = 6 #accumulated plastic strain, plastic multiplier, 4 components of plastic strain tensor
        number_of_dofs_per_element_type = self.precomputed_data.number_of_dofs_per_element_type[number_of_vertices]
        virtual_displacement_symmetric_gradients = self.precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices]
        virtual_displacement_volume_weighted_divergence = self.precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices]

        elastic_strain_tensors = self.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3)) \
                                - self.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        uninterpolated_stress_tensor = oe.contract("ijkl,qkl->qij", 
                                                   self.precomputed_data.linear_elastic_constitutive_tensor, 
                                                   elastic_strain_tensors)
        stress_tensor_derivatives_wrt_densities = oe.contract("q,qij->qij", 
                                                              elastic_interpolation_derivative, 
                                                              uninterpolated_stress_tensor)\
                                                              .reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, 3, 3))    
        jacobian_determinant_x_quadrature_weights = self.precomputed_data.jacobian_determinant_x_quadrature_weights[number_of_vertices]  

        bulk_modulus_derivative_wrt_densities = oe.contract('...,q->q', 
                                                            self.precomputed_data.bulk_modulus, 
                                                            elastic_interpolation_derivative)\
                                                            .reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type))
        
        shear_modulus_derivative_wrt_densities = oe.contract('...,q->q', 
                                                             self.precomputed_data.shear_modulus, 
                                                             elastic_interpolation_derivative, 
                                                             optimize='optimal')\
                                                            .reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type))
        
        elastic_strain = self.state.current_total_strain_tensors[number_of_vertices] - self.state.current_plastic_strain_tensors[number_of_vertices]
        nodal_displacements_elementwise_per_elem_type = self.state.nodal_displacement_vector[self.precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
        current_volume_avg_volumetric_strain_per_elem_type = oe.contract("e,ei,ei->e", 
                                                                        1.0 / self.precomputed_data.element_volumes_per_elem_type[number_of_vertices], 
                                                                        self.precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices], 
                                                                        nodal_displacements_elementwise_per_elem_type, 
                                                                        optimize='optimal')

        internal_force_derivative_wrt_densities_per_elem_type = oe.contract("...,eq,eqmij,ijkl,eqkl,eq->em", 
                                                                            2.0, 
                                                                            shear_modulus_derivative_wrt_densities,
                                                                            virtual_displacement_symmetric_gradients, 
                                                                            self.precomputed_data.deviatoric_projection_tensor[:2, :2, :, :],
                                                                            elastic_strain[:,:,:,:], 
                                                                            jacobian_determinant_x_quadrature_weights, 
                                                                            optimize='optimal') 
        bulk_modulus_derivative_for_computations = bulk_modulus_derivative_wrt_densities[:,0] 
        internal_force_derivative_wrt_densities_per_elem_type += oe.contract("e,e,em->em", 
                                                                             bulk_modulus_derivative_for_computations,
                                                                             current_volume_avg_volumetric_strain_per_elem_type,
                                                                             virtual_displacement_volume_weighted_divergence, 
                                                                             optimize='optimal')
        
        
        internal_force_derivative_wrt_current_displacements_per_elem_type = oe.contract("...,eq,eqmij,ijkl,eqnkl,eq->emn", 
                                                                                      2.0 * self.precomputed_data.shear_modulus,
                                                                                      elastic_interpolation.
                                                                                      reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)),
                                                                                      virtual_displacement_symmetric_gradients,
                                                                                      self.precomputed_data.deviatoric_projection_tensor[:2,:2,:2,:2],
                                                                                      virtual_displacement_symmetric_gradients,
                                                                                      jacobian_determinant_x_quadrature_weights, 
                                                                                      optimize='optimal')
        
        internal_force_derivative_wrt_current_displacements_per_elem_type += oe.contract("...,e,em,en->emn", 
                                                                                         self.precomputed_data.bulk_modulus, 
                                                                                         elastic_interpolation_for_computations 
                                                                                         / self.precomputed_data.element_volumes_per_elem_type[number_of_vertices], 
                                                                                         self.precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices],
                                                                                         self.precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices], 
                                                                                         optimize='optimal')
                                                                                      
        internal_force_derivative_wrt_current_plastic_strains = oe.contract("...,eq,eqmij,ijkl,eq->eqmkl", 
                                                                            -2.0 * self.precomputed_data.shear_modulus,
                                                                            elastic_interpolation.
                                                                            reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)),
                                                                            virtual_displacement_symmetric_gradients, 
                                                                            self.precomputed_data.deviatoric_projection_tensor[:2, :2, :, :],
                                                                            jacobian_determinant_x_quadrature_weights, 
                                                                            optimize='optimal')
                                                                          
        
        internal_force_derivative_wrt_current_plastic_state_per_elem_type = np.zeros((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, \
                                                                                      number_of_dofs_per_element_type, number_of_local_dofs), dtype=float)
        internal_force_derivative_wrt_current_plastic_state_per_elem_type[:, :, :, 2] = internal_force_derivative_wrt_current_plastic_strains[:, :, :, 0, 0]
        internal_force_derivative_wrt_current_plastic_state_per_elem_type[:, :, :, 3] = internal_force_derivative_wrt_current_plastic_strains[:, :, :, 1, 1]
        internal_force_derivative_wrt_current_plastic_state_per_elem_type[:, :, :, 4] = internal_force_derivative_wrt_current_plastic_strains[:, :, :, 0, 1]
        internal_force_derivative_wrt_current_plastic_state_per_elem_type[:, :, :, 5] = internal_force_derivative_wrt_current_plastic_strains[:, :, :, 2, 2]

        return internal_force_derivative_wrt_densities_per_elem_type, internal_force_derivative_wrt_current_displacements_per_elem_type, internal_force_derivative_wrt_current_plastic_state_per_elem_type

    def compute_local_residual_sensitivity_per_elem_type(self, number_of_vertices):
        '''
        Computes all of the local residual partial derivatives for the specified polygonal element type.
        Computes the local reisudal partial derivatives for both the elastic and plastic residual.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            number_of_vertices: The number of vertices in the polygon element type.
                    
        Returns:
            local_residual_derivative_wrt_densities_per_elem_type
            local_residual_derivative_wrt_current_displacements_per_elem_type
            local_residual_derivative_wrt_current_local_variables_per_elem_type
            local_residual_derivative_wrt_previous_local_variables_per_elem_type
        '''
               
        elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices]
        plastic_interpolation = self.plastic_interpolation_function_values[number_of_vertices]
        elastic_interpolation_derivative = self.elastic_interpolation_function_derivatives[number_of_vertices]
        plastic_interpolation_derivative = self.plastic_interpolation_function_derivatives[number_of_vertices]

        total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type
        elastic_strain_tensor = self.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3)) \
            - self.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        deviatoric_elastic_strain = oe.contract("ijkl,qkl->qij", 
                                                self.precomputed_data.deviatoric_projection_tensor, 
                                                elastic_strain_tensor, 
                                                optimize='optimal')
        deviatoric_stress_tensor = oe.contract("...,q,qij->qij", 
                                               2.0*self.precomputed_data.shear_modulus, 
                                               elastic_interpolation, 
                                               deviatoric_elastic_strain, 
                                               optimize='optimal')
        uninterpolated_deviatoric_stress_tensor = oe.contract("...,qij->qij", 
                                                              2.0*self.precomputed_data.shear_modulus, 
                                                              deviatoric_elastic_strain, 
                                                              optimize='optimal')
        uninterpolated_deviatoric_stress_norms = np.linalg.norm(uninterpolated_deviatoric_stress_tensor, axis=(1, 2), ord='fro')
        deviatoric_stress_norms = np.linalg.norm(deviatoric_stress_tensor, axis=(1, 2), ord='fro')
        current_stress = ((3.0/2.0)**0.5) * deviatoric_stress_norms
        penalized_initial_yield_stress = plastic_interpolation * self.precomputed_data.initial_yield_stress
        penalized_hardening_modulus = elastic_interpolation * self.precomputed_data.hardening_modulus
        penalized_shear_modulus = elastic_interpolation * self.precomputed_data.shear_modulus
        current_yield_stress = penalized_initial_yield_stress + penalized_hardening_modulus \
            * self.state.previous_accumulated_plastic_strains[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,))
        current_accumulated_plastic_strain = self.state.current_accumulated_plastic_strains[number_of_vertices]\
                                            .reshape((total_number_of_quad_pts_per_element_type,))
        #Yield function
        yield_function = current_stress - current_yield_stress
        elastic = yield_function <= 1.0e-10   #creating mask
        plastic = ~elastic

        number_of_local_dofs = 6 #accumulated plastic strain, plastic multiplier, 4 components of plastic strain tensor
        dim = self.precomputed_data.space_dimension
        number_of_dofs_per_element_type = self.precomputed_data.number_of_dofs_per_element_type[number_of_vertices]
        virtual_displacement_symmetric_gradients = self.precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices]
        local_residual_derivative_wrt_current_local_variables_per_elem_type = np.zeros((total_number_of_quad_pts_per_element_type, number_of_local_dofs, number_of_local_dofs), dtype=float)
        local_residual_derivative_wrt_previous_local_variables_per_elem_type = np.zeros((total_number_of_quad_pts_per_element_type, number_of_local_dofs, number_of_local_dofs), dtype=float)
        local_residual_derivative_wrt_densities_per_elem_type = np.zeros((total_number_of_quad_pts_per_element_type, number_of_local_dofs), dtype=float)
        local_residual_derivative_wrt_current_displacements_per_elem_type = np.zeros((total_number_of_quad_pts_per_element_type, number_of_local_dofs, number_of_dofs_per_element_type), dtype=float)
        fourth_order_symmetric_identity_tensor = self.precomputed_data.fourth_order_symmetric_identity_tensor
        fourth_order_deviatoric_projection_tensor = self.precomputed_data.deviatoric_projection_tensor
        plastic_multiplier_increments = self.state.plastic_multiplier_increments[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,))

        if np.any(elastic):
            elastic_indices = np.argwhere(elastic).ravel()
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,0,0] = 1.0 #derivative of the update of the accumulated plastic
                                                                                                           #strain wrt the accumulated plastic strain
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,1,1] = 1.0 #derivative of the yield function wrt the plastic 
                                                                                                           #multiplier

            #Derivatives of the e_11 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,2, 2] = fourth_order_symmetric_identity_tensor[0, 0, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,2, 3] = fourth_order_symmetric_identity_tensor[0, 0, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,2, 4] = fourth_order_symmetric_identity_tensor[0, 0, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,2, 5] = fourth_order_symmetric_identity_tensor[0, 0, 2, 2]

            #Derivatives of the 22 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,3, 2] = fourth_order_symmetric_identity_tensor[1, 1, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,3, 3] = fourth_order_symmetric_identity_tensor[1, 1, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,3, 4] = fourth_order_symmetric_identity_tensor[1, 1, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,3, 5] = fourth_order_symmetric_identity_tensor[1, 1, 2, 2]

            #Derivatives of the 12 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,4, 2] = fourth_order_symmetric_identity_tensor[0, 1, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,4, 3] = fourth_order_symmetric_identity_tensor[0, 1, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,4, 4] = fourth_order_symmetric_identity_tensor[0, 1, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,4, 5] = fourth_order_symmetric_identity_tensor[0, 1, 2, 2]

            #Derivatives of the 33 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,5, 2] = fourth_order_symmetric_identity_tensor[2, 2, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,5, 3] = fourth_order_symmetric_identity_tensor[2, 2, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,5, 4] = fourth_order_symmetric_identity_tensor[2, 2, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[elastic_indices,5, 5] = fourth_order_symmetric_identity_tensor[2, 2, 2, 2]

            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,0,0] = -1.0 #derivative of the update of the accumulated plastic strain wrt the accumulated plastic strain

            #Derivatives of the 11 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,2, 2] = -fourth_order_symmetric_identity_tensor[0, 0, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,2, 3] = -fourth_order_symmetric_identity_tensor[0, 0, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,2, 4] = -fourth_order_symmetric_identity_tensor[0, 0, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,2, 5] = -fourth_order_symmetric_identity_tensor[0, 0, 2, 2]

            #Derivatives of the 22 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,3, 2] = -fourth_order_symmetric_identity_tensor[1, 1, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,3, 3] = -fourth_order_symmetric_identity_tensor[1, 1, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,3, 4] = -fourth_order_symmetric_identity_tensor[1, 1, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,3, 5] = -fourth_order_symmetric_identity_tensor[1, 1, 2, 2]

            #Derivatives of the 12 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,4, 2] = -fourth_order_symmetric_identity_tensor[0, 1, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,4, 3] = -fourth_order_symmetric_identity_tensor[0, 1, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,4, 4] = -fourth_order_symmetric_identity_tensor[0, 1, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,4, 5] = -fourth_order_symmetric_identity_tensor[0, 1, 2, 2]

            #Derivatives of the 33 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,5, 2] = -fourth_order_symmetric_identity_tensor[2, 2, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,5, 3] = -fourth_order_symmetric_identity_tensor[2, 2, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,5, 4] = -fourth_order_symmetric_identity_tensor[2, 2, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[elastic_indices,5, 5] = -fourth_order_symmetric_identity_tensor[2, 2, 2, 2]
 
        if np.any(plastic):
            plastic_indices = np.argwhere(plastic).ravel()

            virtual_displacement_symmetric_gradients_plastic = virtual_displacement_symmetric_gradients\
                .reshape((total_number_of_quad_pts_per_element_type, number_of_dofs_per_element_type, dim, dim))[plastic_indices, :, :, :]
            deviatoric_stress_derivative_wrt_densities = oe.contract("...,q,ijkl,qkl->qij", 
                                                                     2.0*self.precomputed_data.shear_modulus, 
                                                                     elastic_interpolation_derivative[plastic_indices], 
                                                                     self.precomputed_data.deviatoric_projection_tensor, 
                                                                     elastic_strain_tensor[plastic_indices,:,:], 
                                                                     optimize='optimal')
            yield_function_derivative_wrt_deviatoric_stress = oe.contract("...,q,qij->qij", 
                                                                          (3.0/2.0)**(0.5), 
                                                                           1.0 / uninterpolated_deviatoric_stress_norms[plastic_indices], 
                                                                           uninterpolated_deviatoric_stress_tensor[plastic_indices,:,:], 
                                                                           optimize='optimal')
            yield_function_derivative_wrt_densities = oe.contract("qij,qij->q", 
                                                                  yield_function_derivative_wrt_deviatoric_stress, 
                                                                  deviatoric_stress_derivative_wrt_densities, 
                                                                  optimize='optimal')

            flow_rule_derivative_wrt_plastic_multiplier = oe.contract("...,q,qij->qij",
                                                                       -(3.0/2.0)**0.5, 
                                                                        1.0/uninterpolated_deviatoric_stress_norms[plastic_indices], 
                                                                        uninterpolated_deviatoric_stress_tensor[plastic_indices],   
                                                                        optimize='optimal')
            yield_function_derivative_wrt_current_plastic_strain = oe.contract("...,q,qij->qij", 
                                                                               -1.0*(6.0)**0.5, 
                                                                                penalized_shear_modulus[plastic_indices] / 
                                                                                uninterpolated_deviatoric_stress_norms[plastic_indices], 
                                                                                uninterpolated_deviatoric_stress_tensor[plastic_indices,:,:], 
                                                                                optimize='optimal')
    
            deviatoric_stress_over_deviatoric_stress_norm = oe.contract("q,qij->qij", 
                                                                      1.0 / uninterpolated_deviatoric_stress_norms[plastic_indices], 
                                                                      uninterpolated_deviatoric_stress_tensor[plastic_indices], 
                                                                      optimize='optimal')
            temp_1 = oe.contract("qij,qkl->qijkl", 
                                 deviatoric_stress_over_deviatoric_stress_norm, 
                                 deviatoric_stress_over_deviatoric_stress_norm, 
                                 optimize='optimal')
            temp_2 = fourth_order_deviatoric_projection_tensor[np.newaxis, :, :, :, :] - temp_1
            temp_3 = oe.contract("...,q->q",  
                                 (6.0)**0.5 * self.precomputed_data.shear_modulus, 
                                 plastic_multiplier_increments[plastic_indices] 
                                 / uninterpolated_deviatoric_stress_norms[plastic_indices], 
                                 optimize='optimal')
            A_i = oe.contract("q,qijkl->qijkl", temp_3, temp_2, optimize='optimal')
            flow_rule_derivative_wrt_current_plastic_strain = fourth_order_symmetric_identity_tensor[np.newaxis, :, :, :, :] + A_i
            yield_function_derivative_wrt_current_displacements = oe.contract("...,q,qij,qmij->qm", 
                                                                              (6.0)**0.5, penalized_shear_modulus[plastic_indices] , 
                                                                              deviatoric_stress_over_deviatoric_stress_norm[:,:2,:2], 
                                                                              virtual_displacement_symmetric_gradients_plastic, 
                                                                              optimize='optimal')
            flow_rule_derivative_wrt_current_displacements = oe.contract("...,qijkl,qmkl->qmij", 
                                                                         -1.0, 
                                                                         A_i[:,:,:, :dim, :dim], 
                                                                         virtual_displacement_symmetric_gradients_plastic, 
                                                                         optimize='optimal')


            local_residual_derivative_wrt_densities_per_elem_type[plastic_indices, 1] = yield_function_derivative_wrt_densities - \
                (self.precomputed_data.hardening_modulus * elastic_interpolation_derivative[plastic_indices]*current_accumulated_plastic_strain[plastic_indices] + \
                 self.precomputed_data.initial_yield_stress * plastic_interpolation_derivative[plastic_indices])
            
            local_residual_derivative_wrt_current_displacements_per_elem_type[plastic_indices, 2, :] = flow_rule_derivative_wrt_current_displacements[:,:,0,0]
            local_residual_derivative_wrt_current_displacements_per_elem_type[plastic_indices, 3, :] = flow_rule_derivative_wrt_current_displacements[:,:,1,1]
            local_residual_derivative_wrt_current_displacements_per_elem_type[plastic_indices, 4, :] = flow_rule_derivative_wrt_current_displacements[:,:,0,1]
            local_residual_derivative_wrt_current_displacements_per_elem_type[plastic_indices, 5, :] = flow_rule_derivative_wrt_current_displacements[:,:,2,2]
            local_residual_derivative_wrt_current_displacements_per_elem_type[plastic_indices, 1, :] = yield_function_derivative_wrt_current_displacements[:,:]
                
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 0, 0] = 1.0 #Derivative of the update of the accumulated plastic strain wrt the accumulated plastic strain
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 1, 0] = -penalized_hardening_modulus[plastic_indices] #Derivative of the yield function wrt the accumulated plastic strain
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 0, 1] = -1.0 #Derivative of the update of the accumulated plastic strain wrt the plastic multiplier
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 2, 1] = flow_rule_derivative_wrt_plastic_multiplier[:, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 3, 1] = flow_rule_derivative_wrt_plastic_multiplier[:, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 4, 1] = flow_rule_derivative_wrt_plastic_multiplier[:, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 5, 1] = flow_rule_derivative_wrt_plastic_multiplier[:, 2, 2]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 1, 2] = yield_function_derivative_wrt_current_plastic_strain[:, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 1, 3] = yield_function_derivative_wrt_current_plastic_strain[:, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 1, 4] = yield_function_derivative_wrt_current_plastic_strain[:, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 1, 5] = yield_function_derivative_wrt_current_plastic_strain[:, 2, 2]       
            #Derivative of the 11 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 2, 2] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 0, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 2, 3] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 0, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 2, 4] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 0, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 2, 5] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 0, 2, 2]       
            #Derivative of the 22 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 3, 2] = flow_rule_derivative_wrt_current_plastic_strain[:, 1, 1, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 3, 3] = flow_rule_derivative_wrt_current_plastic_strain[:, 1, 1, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 3, 4] = flow_rule_derivative_wrt_current_plastic_strain[:, 1, 1, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 3, 5] = flow_rule_derivative_wrt_current_plastic_strain[:, 1, 1, 2, 2]       
            #Derivative of the 12 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 4, 2] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 1, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 4, 3] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 1, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 4, 4] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 1, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 4, 5] = flow_rule_derivative_wrt_current_plastic_strain[:, 0, 1, 2, 2]       
            #Derivative of the 33 component of the flow rule wrt the components of the plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 5, 2] = flow_rule_derivative_wrt_current_plastic_strain[:, 2, 2, 0, 0]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 5, 3] = flow_rule_derivative_wrt_current_plastic_strain[:, 2, 2, 1, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 5, 4] = flow_rule_derivative_wrt_current_plastic_strain[:, 2, 2, 0, 1]
            local_residual_derivative_wrt_current_local_variables_per_elem_type[plastic_indices, 5, 5] = flow_rule_derivative_wrt_current_plastic_strain[:, 2, 2, 2, 2]       
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,0,0] = -1.0 #derivative of the update of the accumulated plastic strain wrt the accumulated plastic strain       
            #Derivative of the 11 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,2, 2] = -fourth_order_symmetric_identity_tensor[0, 0, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,2, 3] = -fourth_order_symmetric_identity_tensor[0, 0, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,2, 4] = -fourth_order_symmetric_identity_tensor[0, 0, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,2, 5] = -fourth_order_symmetric_identity_tensor[0, 0, 2, 2]       
            #Derivative of the 22 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,3, 2] = -fourth_order_symmetric_identity_tensor[1, 1, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,3, 3] = -fourth_order_symmetric_identity_tensor[1, 1, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,3, 4] = -fourth_order_symmetric_identity_tensor[1, 1, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,3, 5] = -fourth_order_symmetric_identity_tensor[1, 1, 2, 2]       
            #Derivative of the 12 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,4, 2] = -fourth_order_symmetric_identity_tensor[0, 1, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,4, 3] = -fourth_order_symmetric_identity_tensor[0, 1, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,4, 4] = -fourth_order_symmetric_identity_tensor[0, 1, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,4, 5] = -fourth_order_symmetric_identity_tensor[0, 1, 2, 2]       
            #Derivative of the 33 component of the flow rule wrt the components of the PREVIOUS plastic strain e_11, e_22, e_12, e_33
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,5, 2] = -fourth_order_symmetric_identity_tensor[2, 2, 0, 0]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,5, 3] = -fourth_order_symmetric_identity_tensor[2, 2, 1, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,5, 4] = -fourth_order_symmetric_identity_tensor[2, 2, 0, 1]
            local_residual_derivative_wrt_previous_local_variables_per_elem_type[plastic_indices,5, 5] = -fourth_order_symmetric_identity_tensor[2, 2, 2, 2]


        return local_residual_derivative_wrt_densities_per_elem_type, local_residual_derivative_wrt_current_displacements_per_elem_type,\
               local_residual_derivative_wrt_current_local_variables_per_elem_type, local_residual_derivative_wrt_previous_local_variables_per_elem_type

    def get_plastic_work_obj_partials_per_elem_type(self, number_of_vertices: int, time_step_index: int, at_final_time_step: bool):
        '''
        Computes all of the plastic work partial derivatives for the specified polygonal element type.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            number_of_vertices: The number of vertices in the polygon element type.
            time_step_index: The index of the current time step.
            at_final_time_step: A boolean flag indicating if the current time step is the final time step. 
            (Different partial derivative expressions used at final time step vs the remaining time steps)
                    
        Returns:
            plastic_work_obj_derivative_wrt_densities
            plastic_work_obj_derivative_wrt_current_displacements
            plastic_work_obj_derivative_wrt_current_local_variables
        '''
        
        elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices][:]
        elastic_interpolation_derivative = self.elastic_interpolation_function_derivatives[number_of_vertices][:]  
        total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type
        penalized_shear_modulus = elastic_interpolation * self.precomputed_data.shear_modulus   
        number_of_local_dofs = 6 #accumulated plastic strain, plastic multiplier, 4 components of plastic strain tensor
        number_of_dofs_per_element_type = self.precomputed_data.number_of_dofs_per_element_type[number_of_vertices]
        virtual_displacement_symmetric_gradients = self.precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices][:,:,:,:,:]

        elastic_strain_tensors = self.state.current_total_strain_tensors[number_of_vertices][:,:,:,:] - \
            self.state.current_plastic_strain_tensors[number_of_vertices][:,:,:,:] 
        
        deviatoric_stress_tensor_derivatives_wrt_densities = oe.contract("...,eq,ijkl,eqkl->eqij", 
                                                                         2.0 * self.precomputed_data.shear_modulus,
                                                                         elastic_interpolation_derivative.
                                                                         reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)),
                                                                         self.precomputed_data.deviatoric_projection_tensor,
                                                                         elastic_strain_tensors, 
                                                                         optimize='optimal')     

        jacobian_determinant_x_quadrature_weights = self.precomputed_data.jacobian_determinant_x_quadrature_weights[number_of_vertices][:,:]  

        plastic_work_obj_derivative_wrt_densities = np.zeros((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type), dtype=float)
        plastic_work_obj_derivative_wrt_current_displacements = np.zeros((total_number_of_elements_per_elem_type, number_of_dofs_per_element_type), dtype=float)
        plastic_work_obj_derivative_wrt_current_local_variables = np.zeros((total_number_of_quad_pts_per_element_type, number_of_local_dofs), dtype=float)     


        if at_final_time_step:   
            plastic_strain_diff_tensor_n = self.state.current_plastic_strain_tensors[number_of_vertices][:,:,:,:] \
                                    - self.state.previous_plastic_strain_tensors[number_of_vertices][:,:,:,:]
            
            deviatoric_stress_tensor_value = self.state.current_deviatoric_stress_tensors[number_of_vertices][:,:,:,:] + self.state.previous_deviatoric_stress_tensors[number_of_vertices][:,:,:,:]          
     
            plastic_work_obj_derivative_wrt_densities[:,:] += oe.contract("...,eqij,eqij,eq->eq", 
                                                                          0.5, 
                                                                          deviatoric_stress_tensor_derivatives_wrt_densities[:,:, :, :],
                                                                          plastic_strain_diff_tensor_n[:,:,:,:], 
                                                                          jacobian_determinant_x_quadrature_weights, 
                                                                          optimize='optimal')
        
            plastic_work_derivative_wrt_current_plastic_strain = oe.contract("...,eq,ijkl,eqkl,eq->eqij",  
                                                                             -1.0,
                                                                            penalized_shear_modulus.
                                                                            reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)), 
                                                                            self.precomputed_data.deviatoric_projection_tensor, 
                                                                            plastic_strain_diff_tensor_n[:,:,:,:], 
                                                                            jacobian_determinant_x_quadrature_weights, optimize='optimal').\
                                                                                reshape((total_number_of_quad_pts_per_element_type, 3, 3))
            
            plastic_work_derivative_wrt_current_plastic_strain += oe.contract("...,eq,eqij->eqij", 
                                                                              0.5,
                                                                              jacobian_determinant_x_quadrature_weights,
                                                                              deviatoric_stress_tensor_value[:,:,:,:],
                                                                              optimize='optimal').reshape((total_number_of_quad_pts_per_element_type, 3, 3))
            

            plastic_work_obj_derivative_wrt_current_local_variables[:, 2] = plastic_work_derivative_wrt_current_plastic_strain[:, 0, 0]
            plastic_work_obj_derivative_wrt_current_local_variables[:, 3] = plastic_work_derivative_wrt_current_plastic_strain[:, 1, 1]
            plastic_work_obj_derivative_wrt_current_local_variables[:, 4] = plastic_work_derivative_wrt_current_plastic_strain[:, 0, 1]
            plastic_work_obj_derivative_wrt_current_local_variables[:, 5] = plastic_work_derivative_wrt_current_plastic_strain[:, 2, 2]
        
            plastic_work_obj_derivative_wrt_current_displacements[:,:] = oe.contract("eq,ijkl,eqij,eqmkl,eq->em", 
                                                                                     penalized_shear_modulus.reshape
                                                                                     ((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)),
                                                                                     self.precomputed_data.deviatoric_projection_tensor[:,:,:2,:2], 
                                                                                     plastic_strain_diff_tensor_n[:,:,:,:],
                                                                                     virtual_displacement_symmetric_gradients, 
                                                                                     jacobian_determinant_x_quadrature_weights, 
                                                                                     optimize='optimal')  

        else:
            total_strain_tensor_i_minus_1 = self.state.previous_total_strain_tensors[number_of_vertices][:,:,:,:]
            plastic_strain_tensor_i_minus_1 = self.state.previous_plastic_strain_tensors[number_of_vertices][:,:,:,:]
            deviatoric_stress_tensor_i_minus_1 = self.state.previous_deviatoric_stress_tensors[number_of_vertices][:,:,:,:]
            self.state = self.states[time_step_index + 1]
            total_strain_tensor_i_plus_1 = self.state.current_total_strain_tensors[number_of_vertices][:,:,:,:]
            plastic_strain_tensor_i_plus_1 = self.state.current_plastic_strain_tensors[number_of_vertices][:,:,:,:]
            deviatoric_stress_tensor_i_plus_1 = self.state.current_deviatoric_stress_tensors[number_of_vertices][:,:,:,:]
            deviatoric_stress_tensor_value = -deviatoric_stress_tensor_i_plus_1 + deviatoric_stress_tensor_i_minus_1
            total_strain_diff_tensor_i = total_strain_tensor_i_plus_1 - total_strain_tensor_i_minus_1
            plastic_strain_diff_tensor_i = plastic_strain_tensor_i_plus_1 - plastic_strain_tensor_i_minus_1

            plastic_work_obj_derivative_wrt_densities[:,:] += oe.contract("...,eqij,eqij,eq->eq", 
                                                                          0.5, 
                                                                          deviatoric_stress_tensor_derivatives_wrt_densities[:,:,:,:],
                                                                          plastic_strain_diff_tensor_i[:,:,:,:],
                                                                          jacobian_determinant_x_quadrature_weights, 
                                                                          optimize='optimal')
            
            plastic_work_derivative_wrt_current_plastic_strain = oe.contract("...,eq,ijkl,eqij,eq->eqkl",  
                                                                             -1.0,
                                                                            penalized_shear_modulus.reshape
                                                                            ((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)), 
                                                                            self.precomputed_data.deviatoric_projection_tensor, 
                                                                            plastic_strain_diff_tensor_i[:,:,:,:], 
                                                                            jacobian_determinant_x_quadrature_weights, 
                                                                            optimize='optimal') \
                                                                            .reshape((total_number_of_quad_pts_per_element_type, 3, 3))
            plastic_work_derivative_wrt_current_plastic_strain += oe.contract("...,eqkl,eq->eqkl", 
                                                                              0.5,
                                                                              deviatoric_stress_tensor_value[:,:,:,:],
                                                                              jacobian_determinant_x_quadrature_weights, 
                                                                              optimize='optimal') \
                                                                              .reshape((total_number_of_quad_pts_per_element_type, 3, 3))

            plastic_work_obj_derivative_wrt_current_local_variables[:, 2] = plastic_work_derivative_wrt_current_plastic_strain[:, 0, 0]
            plastic_work_obj_derivative_wrt_current_local_variables[:, 3] = plastic_work_derivative_wrt_current_plastic_strain[:, 1, 1]
            plastic_work_obj_derivative_wrt_current_local_variables[:, 4] = plastic_work_derivative_wrt_current_plastic_strain[:, 0, 1]
            plastic_work_obj_derivative_wrt_current_local_variables[:, 5] = plastic_work_derivative_wrt_current_plastic_strain[:, 2, 2]
                    
            plastic_work_obj_derivative_wrt_current_displacements[:,:] = oe.contract("eq,ijkl,eqij,eqmkl,eq->em", 
                                                                                   penalized_shear_modulus.reshape
                                                                                   ((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)),
                                                                                   self.precomputed_data.deviatoric_projection_tensor[:,:,:2,:2], 
                                                                                   plastic_strain_diff_tensor_i[:,:,:,:], 
                                                                                   virtual_displacement_symmetric_gradients, 
                                                                                   jacobian_determinant_x_quadrature_weights, 
                                                                                   optimize='optimal')
            self.state = self.states[time_step_index]

        return plastic_work_obj_derivative_wrt_densities, plastic_work_obj_derivative_wrt_current_displacements, plastic_work_obj_derivative_wrt_current_local_variables

                  
    def compute_volume_fraction_value_and_sensitivity(self):
        '''Computes the volume fraction value and sensitivity using the current element densities.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.

        Returns:
            The volume fraction value.
            The volume fraction sensitivity with respect to the design variables.
        '''
        volume_fraction_value = np.inner(self.precomputed_data.element_relative_volumes, self.element_densities)
        projection_chain_rule = self.projection.apply_chain_rule(self.precomputed_data.element_relative_volumes)
        nodal_sensitivity = self.precomputed_data.node_to_element_map.T @ projection_chain_rule
        volume_fraction_sensitivity_wrt_design_variables = \
            self.design_variable_filter.apply_chain_rule(nodal_sensitivity)
        return volume_fraction_value, volume_fraction_sensitivity_wrt_design_variables

    def get_function_values_and_gradients(self, design_variables: np.ndarray, optimization_iteration_number: int = 0):
        '''Computes the objective function value and sensitivity with respect to the unfiltered and unprojected design variables.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            design_variables: The vector of design variables.
            optimization_iteration_number: The current optimization iteration number.

        Returns:
            The objective function value.
            The objective function sensitivity with respect to the design variables.
            The volume fraction value.
            The volume fraction sensitivity with respect to the design variables.
        '''
        
        self.optimization_iteration_number = optimization_iteration_number
        # Continuation applied below:
        if optimization_iteration_number < 25:
            self.elastic_material_interpolation.penalization_parameter = 1.0
            self.plastic_material_interpolation.penalization_parameter = 0.5
        elif optimization_iteration_number < 50:
            self.elastic_material_interpolation.penalization_parameter = 2.0
            self.plastic_material_interpolation.penalization_parameter = 1.5
        elif optimization_iteration_number < 75:
            self.elastic_material_interpolation.penalization_parameter = 3.0
            self.plastic_material_interpolation.penalization_parameter = 2.5
        elif optimization_iteration_number < 100:
            self.elastic_material_interpolation.penalization_parameter = 4.0
            self.plastic_material_interpolation.penalization_parameter = 3.5
        elif optimization_iteration_number < 125:
            self.projection.projection_strength = self.projection_strengths_continuation[0]
        elif optimization_iteration_number < 150:
            self.projection.projection_strength = self.projection_strengths_continuation[1]
        elif optimization_iteration_number < 175:
            self.projection.projection_strength = self.projection_strengths_continuation[2]
        else:
            self.projection.projection_strength = self.projection_strengths_continuation[3]

        filtered_nodal_design_variables = self.design_variable_filter.apply_filter(design_variables)
        filtered_element_densities = self.precomputed_data.node_to_element_map @ filtered_nodal_design_variables
        self.filtered_and_projected_design_variables[:] = self.projection.apply_projection(filtered_element_densities)
    
        objective_function_value, _, _, _, = self.run_forward_analysis()

        objective_sensitivity = self.compute_objective_function_sensitivity()
        (volume_fraction, volume_fraction_sensitivity) = self.compute_volume_fraction_value_and_sensitivity()

        return (objective_function_value, objective_sensitivity), (volume_fraction, volume_fraction_sensitivity)
    
    def finite_difference_sensitivity_checker(self, number_of_design_variables_to_check: int = 10, finite_diff_step_size: float=1.0e-6):
        '''Finite difference sensitivity checker for the augmented objective function sensitivity and the volume fraction constraint sensitivity.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            number_of_design_variables_to_check: The number of design variables to check.
            finite_diff_step_size: The finite difference step size.
            
        Returns:
            objective_function_sensitivity_percent_differences 
            volume_fraction_sensitivity_percent_differences
            (Plots of the numerical sensitivities compared against the analytical sensitivities (derivatives in code) for objective and constraint function.)
        '''
        rng = np.random.default_rng(seed=123)

        def get_objective_function_values_and_sensitivities(design_variables):
            filtered_nodal_design_variables = self.design_variable_filter.apply_filter(design_variables)
            filtered_element_densities = self.precomputed_data.node_to_element_map @ filtered_nodal_design_variables
            self.element_densities[:] = self.projection.apply_projection(filtered_element_densities)
            self.filtered_and_projected_design_variables[:] = self.projection.apply_projection(filtered_element_densities)
               
            objective_function_value, _, _, _, = self.run_forward_analysis() 
            objective_function_sensitivity = self.compute_objective_function_sensitivity()
            volume_fraction_value, volume_fraction_sensitivity = self.compute_volume_fraction_value_and_sensitivity()
            return (objective_function_value, volume_fraction_value), (objective_function_sensitivity, volume_fraction_sensitivity)

        random_design_variables = rng.uniform(low=0.4, high=0.9, size=self.precomputed_data.number_of_nodes) 
          
        functions, analytical_sensitivities = get_objective_function_values_and_sensitivities(random_design_variables)
        analytical_objection_function_sensitivity, analytical_volume_fraction_sensitivity = analytical_sensitivities
        objective_function, volume_function = functions

        number_of_design_variabes_to_check = min(number_of_design_variables_to_check, random_design_variables.size)

        numerical_objective_function_sensitivities = np.zeros((number_of_design_variabes_to_check), dtype=float)
        numerical_volume_fraction_sensitivities = np.zeros((number_of_design_variabes_to_check), dtype=float)

        for design_variable_index in range(number_of_design_variabes_to_check):
            message = "Checking sensitivities for design variable " + \
                      f"{design_variable_index:5d} of {number_of_design_variabes_to_check}"
            self.logger.info(message)
            design_variables_forward = random_design_variables.copy()
            design_variables_forward[design_variable_index] += finite_diff_step_size
            function_values, _ = get_objective_function_values_and_sensitivities(design_variables_forward)
            total_work_value_f1, volume_fraction_value_f1 = function_values

            design_variables_backward = random_design_variables.copy()
            design_variables_backward[design_variable_index] -= finite_diff_step_size
            function_values, _ = get_objective_function_values_and_sensitivities(design_variables_backward)
            total_work_value_b1, volume_fraction_value_b1 = function_values

            numerical_objective_function_sensitivities[design_variable_index] = \
                (total_work_value_f1 - total_work_value_b1) / (2.0 * finite_diff_step_size)

            numerical_volume_fraction_sensitivities[design_variable_index] = \
                (volume_fraction_value_f1 - volume_fraction_value_b1) / (2.0 * finite_diff_step_size)

        print(numerical_objective_function_sensitivities)
        print(analytical_objection_function_sensitivity)

        objective_function_sensitivity_percent_differences = 100.0 * \
            (analytical_objection_function_sensitivity[0:number_of_design_variabes_to_check] -\
             numerical_objective_function_sensitivities) / np.abs(numerical_objective_function_sensitivities)

        volume_fraction_sensitivity_percent_differences = 100.0 * \
            (analytical_volume_fraction_sensitivity[0:number_of_design_variabes_to_check] - \
             numerical_volume_fraction_sensitivities) / np.abs(numerical_volume_fraction_sensitivities)

        def plot_sensitivity_comparison(analytical, numerical, percent_difference, title_string):
            fig, axes = plt.subplots(nrows=2, ncols=1)
            axes[0].plot(numerical, 'k', linewidth=1.5, label='Numerical')
            axes[0].plot(analytical, 'ro', markersize=2, label='Analytical')
            axes[0].set_xlabel("Node Index")
            axes[0].set_ylabel(title_string)
            axes[0].legend()
            axes[0].grid(True)

            axes[1].semilogy(np.abs(percent_difference), 'k', linewidth=1.5)
            axes[1].set_xlabel("Node Index")
            axes[1].set_ylabel("Percent Difference")
            axes[1].grid(True)
            fig.tight_layout()

        current_working_directory = os.getcwd()[:]
        os.chdir(self.directory_name)

        plot_sensitivity_comparison(analytical_objection_function_sensitivity[0:number_of_design_variabes_to_check],
                                    numerical_objective_function_sensitivities,
                                    objective_function_sensitivity_percent_differences,
                                    "Objective Function Sensitivity")
        plt.savefig('Objective_FD.eps', format='eps', dpi=1200)

        plot_sensitivity_comparison(analytical_volume_fraction_sensitivity[0:number_of_design_variabes_to_check],
                                    numerical_volume_fraction_sensitivities,
                                    volume_fraction_sensitivity_percent_differences,
                                    "Volume Fraction Sensitivity")
        plt.savefig('volume_constraint_FD.eps', format='eps', dpi=1200)
        os.chdir(current_working_directory)

        return objective_function_sensitivity_percent_differences, volume_fraction_sensitivity_percent_differences


    def get_local_residual_for_elem_type(self, elastic, plastic, number_of_vertices):
        '''Gets the local residual per polygonal element type. 
           Function used in the unit testing of the partial derivatives of the local residual in file test_polyplas.py

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            elastic: A boolean array indicating the quadrature points in the elastic loading.
            plastic: A boolean array indicating the quadrature points in the plastic loading.
            number_of_vertices: The number of vertices in the polygonal element type.
            
        Returns:
            local_residual_vector: The local residual vector for both the elastic and plastic loading. 
        '''
        elastic_indices = np.argwhere(elastic).ravel()
        plastic_indices = np.argwhere(plastic).ravel()
        uninterpolated_linear_elastic_constitutive_tensor = self.precomputed_data.linear_elastic_constitutive_tensor
        
        number_of_elements_per_element_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quad_pts_per_element_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = number_of_elements_per_element_type * number_of_quad_pts_per_element_type
        elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices]
        plastic_interpolation = self.plastic_interpolation_function_values[number_of_vertices]
        total_strain_tensors = self.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))
        current_plastic_strain_tensors = self.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))
        previous_plastic_strain_tensors = self.state.previous_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))
        elastic_strain_tensors = total_strain_tensors - current_plastic_strain_tensors
        previous_plastic_strain_tensors = self.state.previous_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        plastic_multiplier_increments = self.state.plastic_multiplier_increments[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,))
        uninterpolated_stress_tensor = oe.contract("ijkl,qkl->qij", 
                                                   uninterpolated_linear_elastic_constitutive_tensor, 
                                                   elastic_strain_tensors)
        interpolated_stress_tensors = oe.contract("q,qij->qij", 
                                                  elastic_interpolation, 
                                                  uninterpolated_stress_tensor)
        deviatoric_stress_tensors = oe.contract("ijkl,qkl->qij", 
                                                self.precomputed_data.deviatoric_projection_tensor, 
                                                interpolated_stress_tensors)
        deviatoric_stress_norms = np.linalg.norm(deviatoric_stress_tensors, axis=(1,2), ord='fro')
        current_stress = ((3.0/2.0)**0.5) * deviatoric_stress_norms
        penalized_initial_yield_stress = plastic_interpolation * self.precomputed_data.initial_yield_stress
        penalized_hardening_modulus = elastic_interpolation * self.precomputed_data.hardening_modulus
        
        current_accumulated_plastic_strain = self.state.current_accumulated_plastic_strains[number_of_vertices] \
                                                .reshape((total_number_of_quad_pts_per_element_type,))
        previous_accumulated_plastic_strain = self.state.previous_accumulated_plastic_strains[number_of_vertices] \
                                                .reshape((total_number_of_quad_pts_per_element_type,))
        current_yield_stress = penalized_initial_yield_stress + penalized_hardening_modulus * current_accumulated_plastic_strain
        piece_of_flow_rule = oe.contract("...,q,qij->qij", 
                                         -1.0 * (3.0/2.0)**(0.5), 
                                         plastic_multiplier_increments / deviatoric_stress_norms, 
                                         deviatoric_stress_tensors)
        accumulated_plastic_strain_law_elastic = current_accumulated_plastic_strain - previous_accumulated_plastic_strain
        accumulated_plastic_strain_law_plastic = current_accumulated_plastic_strain - previous_accumulated_plastic_strain - plastic_multiplier_increments
        yield_function_elastic = plastic_multiplier_increments
        yield_function_plastic = current_stress - current_yield_stress
        flow_rule_elastic = current_plastic_strain_tensors - previous_plastic_strain_tensors
        flow_rule_plastic = current_plastic_strain_tensors - previous_plastic_strain_tensors + piece_of_flow_rule
        number_of_local_dofs = 6
        local_residual_vector = np.zeros((total_number_of_quad_pts_per_element_type, number_of_local_dofs))
        if any(elastic):
            local_residual_vector[elastic_indices,0] = accumulated_plastic_strain_law_elastic[elastic_indices]
            local_residual_vector[elastic_indices,1] = yield_function_elastic[elastic_indices]
            local_residual_vector[elastic_indices,2] = flow_rule_elastic[elastic_indices,0,0]
            local_residual_vector[elastic_indices,3] = flow_rule_elastic[elastic_indices,1,1]
            local_residual_vector[elastic_indices,4] = flow_rule_elastic[elastic_indices,0,1]
            local_residual_vector[elastic_indices,5] = flow_rule_elastic[elastic_indices,2,2]
        if any(plastic):
            local_residual_vector[plastic_indices,0] = accumulated_plastic_strain_law_plastic[plastic_indices]
            local_residual_vector[plastic_indices,1] = yield_function_plastic[plastic_indices]
            local_residual_vector[plastic_indices,2] = flow_rule_plastic[plastic_indices,0,0]
            local_residual_vector[plastic_indices,3] = flow_rule_plastic[plastic_indices,1,1]
            local_residual_vector[plastic_indices,4] = flow_rule_plastic[plastic_indices,0,1]
            local_residual_vector[plastic_indices,5] = flow_rule_plastic[plastic_indices,2,2]

        return local_residual_vector
    
    

    def get_global_residual_vector_for_elem_type(self, number_of_vertices, current_volume_avg_volumetric_strain):
        '''Gets the global residual per polygonal element type. 
           Function used in the unit testing of the partial derivatives of the global residual in file test_polyplas.py

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            number_of_vertices: The number of vertices in the polygonal element type.
            current_volume_avg_volumetric_strain: The current volume average volumetric strain.
            
        Returns:
            element_residual_vector: The element global residual vector.
        '''
        dim = self.precomputed_data.space_dimension
        elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices]
        plastic_interpolation = self.plastic_interpolation_function_values[number_of_vertices]

        number_of_elements_per_element_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quad_pts_per_element_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = number_of_elements_per_element_type * number_of_quad_pts_per_element_type
        deviatoric_stress_tensor = np.zeros((total_number_of_quad_pts_per_element_type, 3, 3), dtype=float)

        total_strain_tensors = self.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        plastic_strain_tensors = self.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        elastic_strain_tensors = total_strain_tensors - plastic_strain_tensors

        deviatoric_elastic_strain = oe.contract("ijkl,qkl->qij", 
                                                self.precomputed_data.deviatoric_projection_tensor, 
                                                elastic_strain_tensors)
        
        deviatoric_stress_tensor = oe.contract("...,q,qij->qij", 
                                               2.0*self.precomputed_data.shear_modulus, 
                                               elastic_interpolation, 
                                               deviatoric_elastic_strain)\
                                               .reshape((number_of_elements_per_element_type, number_of_quad_pts_per_element_type, 3,3))
       
        elastic_interpolation_reshaped = elastic_interpolation.reshape((number_of_elements_per_element_type, number_of_quad_pts_per_element_type))
        elastic_interpolation_for_computations = elastic_interpolation_reshaped[:,0]
        pressure_stress = oe.contract('...,e,e->e', 
                                      self.precomputed_data.bulk_modulus, 
                                      elastic_interpolation_for_computations, 
                                      current_volume_avg_volumetric_strain)
        
        element_residual_vectors = oe.contract('eqmij,eqij,eq->em',
                                                self.precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices],
                                                deviatoric_stress_tensor[:, :, :dim, :dim],
                                                self.precomputed_data.jacobian_determinant_x_quadrature_weights[number_of_vertices])
        element_residual_vectors += oe.contract("e,em->em", 
                                                pressure_stress, 
                                                self.precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices])
        return element_residual_vectors
    
    def get_plastic_work_objective_for_elem_type(self, number_of_vertices):
        '''Computes the plastic work objective function per polygonal element type. 
           Function used in the unit testing of the partial derivatives of the plastic work objective in file test_polyplas.py
           Note: This function is only valid for a single time step.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            number_of_vertices: The number of vertices in the polygonal element type.
            
        Returns:
            plastic_work_value_per_elem_type: The plastic work objective for the polygonal element type.
        '''

        elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices]       
        
        total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type
        stress_tensors = np.zeros((total_number_of_quad_pts_per_element_type,3,3))

        elastic_strain_tensor = self.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3)) - \
                                self.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        deviatoric_elastic_strain = oe.contract("ijkl,qkl->qij", 
                                                self.precomputed_data.deviatoric_projection_tensor, 
                                                elastic_strain_tensor)       
        deviatoric_stress_tensor = oe.contract("...,q,qij->qij", 
                                               2.0*self.precomputed_data.shear_modulus, 
                                               elastic_interpolation, 
                                               deviatoric_elastic_strain)
                    
        stress_tensors[:, :, :] = deviatoric_stress_tensor[:, :, :] 
        
        stress_value = stress_tensors.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type,3,3))
        plastic_strain_difference = self.state.current_plastic_strain_tensors[number_of_vertices][:,:,:,:]\
                                    - self.state.previous_plastic_strain_tensors[number_of_vertices][:,:,:,:]               
        plastic_work_value_per_elem_type = oe.contract('...,eq,eqij,eqij->', 
                                                       0.5, 
                                                       self.precomputed_data.jacobian_determinant_x_quadrature_weights[number_of_vertices], 
                                                       stress_value, 
                                                       plastic_strain_difference) 
        return plastic_work_value_per_elem_type   
    

    def plot_density_field(self):
        '''Plots the density field using the current element densities.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
        '''
        self.logger.debug("Plot density field begin")
        start_time = perf_counter()

        nodal_densities = self.lumped_mass_projection.apply_projection(self.element_densities)
        polyplas_utils.plot_scalar_nodal_field_contours(
            self.precomputed_data,
            nodal_densities,
            figure_title=f"Optimization Iteration {self.optimization_iteration_number}",
            colorbar_label=r"Density, $\rho$",
            colormap="binary",
            colorbar_lower_bound=0.0,
            colorbar_upper_bound=1.0
        )

        elapsed_time = perf_counter() - start_time
        message = f"Plotting the density field required {elapsed_time:.2f} seconds"
        self.logger.debug(message)
        self.logger.debug("Plot density field end")

    def update_logged_values_data_frame(self, logged_values: dict):
        '''Update the logged values data frame with the provided dictionary of quantities to log.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            logged_values (dict): Dictionary containing quantities to log.
        '''
        logged_values = self.elastic_material_interpolation.update_logged_values(logged_values)
        logged_values = self.plastic_material_interpolation.update_logged_values_plastic(logged_values)
        logged_values = self.projection.update_logged_values(logged_values)
        if self.logged_values_data_frame is None:
            self.logged_values_data_frame = pd.DataFrame(data=logged_values, index=[0])
        else:
            new_data_frame = pd.DataFrame(data=logged_values, index=[0])
            self.logged_values_data_frame = \
                pd.concat((self.logged_values_data_frame, new_data_frame), ignore_index=True)
            

    def plot_optimization_history(self,
                                  figure_size_scale_factor: float = 1.5,
                                  save_figure: bool = False,
                                  save_data_filepath: str = "./logged_data.csv"):
        '''Plots of the objective function convergence, volume constraint history, continuation parameters, 
           and the deviatoric plane at the final state.

        Args:
            self: The reference to the current instance of class (object): PolyPlas.
            figure_size_scale_factor (float): Scale factor for the figure size.
            save_figure (bool): If True, save the figure to a PDF file.
            save_data_filepath (str): Path to the CSV file to save the logged data.

        Returns:
            Saves figures to the self.directory_name.
        '''
        plt.rc('font', family='serif')

        if self.logged_values_data_frame is None:
            if os.path.exists(save_data_filepath):
                self.logged_values_data_frame = pd.read_csv(save_data_filepath)
            else:
                raise ValueError(f"There are no logged values to plot and no CSV file at path '{save_data_filepath}'")
        else:
            self.logged_values_data_frame.to_csv(save_data_filepath,
                                                 float_format="%12.4e",
                                                 index=False)
        line_width = 3.0
        font_size = 8.75 * figure_size_scale_factor

        data_columns = self.logged_values_data_frame.columns.to_series()
        optimization_iterations = self.logged_values_data_frame["Optimization Iteration Number"]

        current_working_directory = os.getcwd()[:]
        os.chdir(self.directory_name)

        # Plot the objective function convergence
        plt.figure(2)
        objective = self.logged_values_data_frame.loc[:, "Objective Function Value"]
        plt.plot(optimization_iterations, objective, color='b', linestyle='solid',
                           linewidth=line_width)

        plt.xlabel("Optimization Iteration Number", fontsize=font_size)
        plt.ylabel("Objective Function Value", fontsize=font_size)

        plt.tick_params(axis='x', labelsize=font_size)
        plt.tick_params(axis='y', labelsize=font_size)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('objective_convergence.eps', format='eps', dpi=1200)

        # Plot the volume fraction constraint value
        plt.figure(3)
        constraint = self.logged_values_data_frame.loc[:, "Volume Fraction Value"]
        constraint_upper_bound = self.logged_values_data_frame.loc[:, "Constraint Upper Bound"]
        plt.plot(optimization_iterations, constraint, color='b', linestyle='solid',
                           linewidth=line_width)
        plt.plot(optimization_iterations, constraint_upper_bound, color='r', linestyle='dashed',
                           linewidth=line_width, label="Constraint Upper Bound")
        plt.xlabel("Optimization Iteration Number", fontsize=font_size)
        plt.ylabel("Volume Fraction Value", fontsize=font_size)

        plt.tick_params(axis='x', labelsize=font_size)
        plt.tick_params(axis='y', labelsize=font_size)
        plt.grid(True)
        plt.legend(loc='best', prop={'size': font_size}, ncol=1)
        lower_y_limit = round(max(0, np.amin(constraint) - 0.1), 1)
        upper_y_limit = round(min(1, np.amax(constraint) + 0.1), 1)
        plt.ylim([lower_y_limit, upper_y_limit])
        yticks = np.linspace(lower_y_limit, upper_y_limit, 5)
        plt.yticks(yticks, [f"{yt:0.2f}" for yt in yticks])
        plt.tight_layout()
        plt.savefig('volume_constraint_value_convergence.eps', format='eps', dpi=1200)

        np.save('constraint.npy', constraint)
        np.save('objective.npy', objective)

        # Plot the continuation parameters
        plt.figure(4)
        mask = data_columns.str.endswith("Projection Strength")
        projection_strength_label = data_columns[mask].values[0]
        projection_strength = self.logged_values_data_frame.loc[:, projection_strength_label]
        mask = data_columns.str.endswith("Penalization Parameter")
        penalization_parameter_label = data_columns[mask].values[0]
        elastic_penalization_parameter = self.logged_values_data_frame.loc[:, penalization_parameter_label]
        mask = data_columns.str.endswith("Plastic Penalization Parameter")
        plastic_penalization_parameter_label = data_columns[mask].values[0]
        plastic_penalization_parameter = self.logged_values_data_frame.loc[:, plastic_penalization_parameter_label]
       
        plt.plot(optimization_iterations, projection_strength, color='k', linestyle='solid',
                           linewidth=line_width, label=projection_strength_label)
        plt.plot(optimization_iterations, elastic_penalization_parameter, color='b', linestyle='dashed',
                           linewidth=line_width, label='SIMP Elastic Penalization')
        plt.plot(optimization_iterations, plastic_penalization_parameter, color='r', linestyle='dashed',
                           linewidth=line_width, label='SIMP Plastic Penalization')
        plt.xlabel("Optimization Iteration Number", fontsize=font_size)
        plt.tick_params(axis='x', labelsize=font_size)
        plt.tick_params(axis='y', labelsize=font_size)
        plt.grid(True)
        plt.legend(loc='best', prop={'size': font_size}, ncol=1)
        plt.tight_layout()
        plt.savefig('projection_and_penalization.eps', format='eps', dpi=1200)

        # Plot the deviatoric plane at the final state
        plt.figure(5)
        center = np.array([0, 0])
        radius = (2.0/3.0)**0.5
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = radius * np.cos(theta) + center[0]
        y_circle = radius * np.sin(theta) + center[1]
        plt.plot(x_circle, y_circle, 'r')
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.1)
        plt.gca().set_aspect('equal', 'box')

        self.state = self.states[-1]
        unique_numbers_of_vertices = self.precomputed_data.element_connectivity_arrays.keys()
        total_number_of_quadrature_points_all_elem_type = 0
        total_number_of_quadrature_points_per_elem_type = {}
        for number_of_vertices in unique_numbers_of_vertices:
            total_number_of_elements_per_elem_type = self.precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
            number_of_quadrature_pts_per_elem_type = self.precomputed_data.number_of_quad_points_per_element[number_of_vertices]
            total_number_of_quadrature_points_per_elem_type[number_of_vertices] = total_number_of_elements_per_elem_type \
                                                                                    * number_of_quadrature_pts_per_elem_type
            total_number_of_quadrature_points_all_elem_type += total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type

        sigma_xx = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigma_yy = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigma_xy = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigma_1 = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigma_2 = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigma_3 = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigmas = np.zeros((total_number_of_quadrature_points_all_elem_type,3), dtype=float)
        sigmas_transformed = np.zeros((total_number_of_quadrature_points_all_elem_type,3), dtype=float)
        sigma_1_transformed = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigma_2_transformed = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        sigma_3_transformed = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)
        current_yield_stress = np.zeros((total_number_of_quadrature_points_all_elem_type,), dtype=float)

        quad_pt_index = 0
        for number_of_vertices in unique_numbers_of_vertices:
            total_number_of_quad_pts_per_elem_type = total_number_of_quadrature_points_per_elem_type[number_of_vertices]
            stress_tensors = self.state.current_stress_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_elem_type, 3, 3))[:,:,:]
            principal_stresses,_ = np.linalg.eig(stress_tensors)
            
            sigma_1[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)] = principal_stresses[:,2]
            sigma_2[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)] = principal_stresses[:,1]
            sigma_3[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)] = principal_stresses[:,0]

            origin_arrow = np.zeros((3,), dtype=float)
            sigma_1_arrow = np.array([1.0, 0.0, 0.0])
            sigma_2_arrow = np.array([0.0, 1.0, 0.0])
            sigma_3_arrow = np.array([0.0, 0.0, 1.0])

            matrix_for_rotation = np.array([[((2.0)**0.5)/2.0, -(2.0**0.5)/(2.0*(3.0**0.5)), ((2.0**0.5)/(2.0)) * (2.0/3.0)**0.5], 
                                            [0, (2.0/3.0)**0.5, 1.0/(3.0**0.5)], 
                                            [-(2.0**0.5)/(2.0), -(2.0**0.5)/(2.0*(3.0**0.5)), ((2.0**0.5)/(2.0)) * (2.0/3.0)**0.5]])
            
            sigma_1_tranformed_arrow = oe.contract("ij,i->j", matrix_for_rotation, sigma_1_arrow)
            sigma_2_tranformed_arrow = oe.contract("ij,i->j", matrix_for_rotation, sigma_2_arrow)
            sigma_3_tranformed_arrow = oe.contract("ij,i->j", matrix_for_rotation, sigma_3_arrow)
            
            matrix_inv = np.linalg.inv(matrix_for_rotation)

            sigmas[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),0] = sigma_1[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)]
            sigmas[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),1] = sigma_2[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)]
            sigmas[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),2] = sigma_3[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)]

            sigmas_transformed[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),:] =  oe.contract("ij,qj->qi", 
                                                                                                                    matrix_inv,
                                                                                                                    sigmas[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),:] )

            
            sigma_1_transformed[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)] = sigmas_transformed[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),0]
            sigma_2_transformed[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)] = sigmas_transformed[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),1]
            sigma_3_transformed[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)] = sigmas_transformed[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type),2]


            elastic_interpolation = self.elastic_interpolation_function_values[number_of_vertices][:]
            plastic_interpolation = self.plastic_interpolation_function_values[number_of_vertices][:]
            current_yield_stress[quad_pt_index:(quad_pt_index+total_number_of_quad_pts_per_elem_type)] = (self.precomputed_data.initial_yield_stress \
                                                                                                          * plastic_interpolation   \
                                                    + self.precomputed_data.hardening_modulus * elastic_interpolation * \
                                                        self.state.current_accumulated_plastic_strains[number_of_vertices].reshape((total_number_of_quad_pts_per_elem_type))) 
            quad_pt_index += total_number_of_quad_pts_per_elem_type
        
        
        
        sigma_1_transformed_over_lim = sigma_1_transformed / current_yield_stress
        sigma_2_transformed_over_lim = sigma_2_transformed / current_yield_stress

        plt.plot(sigma_1_transformed_over_lim, sigma_2_transformed_over_lim, 'ko', markersize=2)
       
        plt.quiver(*origin_arrow[:2], *sigma_1_tranformed_arrow[:2], color='k', scale=0.8, scale_units='xy')
        plt.quiver(*origin_arrow[:2], *sigma_2_tranformed_arrow[:2], color='k', scale=0.8, scale_units='xy')
        plt.quiver(*origin_arrow[:2], *sigma_3_tranformed_arrow[:2], color='k', scale=0.8, scale_units='xy')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.title("Deviatoric Plane")
        plt.grid(False)
        plt.savefig('pi-plane-plot.eps', format='eps', dpi=1200)
        os.chdir(current_working_directory)


    