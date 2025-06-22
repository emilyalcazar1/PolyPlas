import pytest
import numpy as np
import pandas as pd
import copy
import src.polytop
import matplotlib.pyplot as plt
from src.polyplas.PolyPlas import PolyPlas
from src.polytop.PolyFilter import IdentityFilter, PolynomialFilter
from src.polytop.PolyProjection import IdentityProjection, TanhHeavisideProjection
from src.polytop.PolyInterpolation import SIMP, RAMP
from typing import Dict, List, NamedTuple, Optional, Tuple



def test_init_with_valid_parameters(polymesh_and_polyplas_precomputed_data):
    polyplas_precomputed_data = polymesh_and_polyplas_precomputed_data[1]

    design_variable_filter = IdentityFilter()
    elastic_material_interpolation = SIMP()
    plastic_material_interpolation = SIMP()
    projection = IdentityProjection()
    volume_fraction_upper_bound = 0.5
    maximum_optimization_iterations = 50
    number_of_time_steps = 10
    polyplas_object = PolyPlas(design_variable_filter=design_variable_filter,
                        elastic_material_interpolation=elastic_material_interpolation,
                        plastic_material_interpolation=plastic_material_interpolation,
                        projection=projection,
                        precomputed_data=polyplas_precomputed_data,
                        volume_fraction_upper_bound=volume_fraction_upper_bound,
                        number_of_time_steps=number_of_time_steps,
                        directory_name="directory_for_tests")
    
    assert polyplas_object.design_variable_filter == design_variable_filter
    assert polyplas_object.elastic_material_interpolation == elastic_material_interpolation
    assert polyplas_object.plastic_material_interpolation == plastic_material_interpolation
    assert polyplas_object.projection == projection
    assert polyplas_object.precomputed_data == polyplas_precomputed_data
    assert polyplas_object.volume_fraction_upper_bound == volume_fraction_upper_bound
    assert polyplas_object.number_of_time_steps == number_of_time_steps

def test_init_with_invalid_design_variable_filter_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(design_variable_filter="not a filter", precomputed_data=precomputed_data)

def test_init_with_invalid_elastic_material_interpolation_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(elastic_material_interpolation="not a material interpolation function", precomputed_data=precomputed_data)

def test_init_with_invalid_plastic_material_interpolation_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(plastic_material_interpolation="not a material interpolation function", precomputed_data=precomputed_data)

def test_init_with_invalid_projection_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(projection="not a projection", precomputed_data=precomputed_data)

def test_init_with_invalid_maximum_projection_strength_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(maximum_projection_strength="not a number", precomputed_data=precomputed_data)

def test_init_with_invalid_maximum_projection_strength_value(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(ValueError):
        PolyPlas(maximum_projection_strength=-0.5, precomputed_data=precomputed_data)

def test_init_with_invalid_precomputed_data_type():
    with pytest.raises(TypeError):
        PolyPlas(precomputed_data="not precomputed data")

def test_init_with_invalid_precomputed_data_value():
    with pytest.raises(ValueError):
        PolyPlas(precomputed_data=None)

def test_init_with_invalid_volume_fraction_upper_bound_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(volume_fraction_upper_bound="not a number", precomputed_data=precomputed_data)

def test_init_with_negative_volume_fraction_upper_bound(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(ValueError):
        PolyPlas(volume_fraction_upper_bound=-0.5, precomputed_data=precomputed_data)

def test_init_with_negative_number_of_time_steps(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(ValueError):
        PolyPlas(number_of_time_steps=-10, precomputed_data=precomputed_data)

def test_init_with_invalid_number_of_time_steps_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(number_of_time_steps='string', precomputed_data=precomputed_data)


def test_init_invalid_directory_name_type(polymesh_and_polyplas_precomputed_data):
    precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    with pytest.raises(TypeError):
        PolyPlas(directory_name=1, precomputed_data=precomputed_data)

def test_global_residual_derivative_wrt_densities(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]
    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 50
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )

    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()

    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]

        internal_force_derivatives_wrt_densities, _, _  = polyplas_object.compute_global_residual_sensitivity_per_elem_type(number_of_vertices)
        #  check derivatives of internal force wrt densities
        perturbation = 1.0e-5
        saved_elastic_interpolation = polyplas_object.elastic_interpolation_function_values.copy()
        saved_plastic_interpolation = polyplas_object.plastic_interpolation_function_values.copy()
        number_of_quadrature_points_per_element = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        nodal_displacements_elementwise_per_elem_type = polyplas_object.state.nodal_displacement_vector[polyplas_precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
        current_volume_avg_volumetric_strain_per_elem_type = np.einsum("e,ei,ei->e", 
                                                                        1.0 / polyplas_precomputed_data.element_volumes_per_elem_type[number_of_vertices], 
                                                                        polyplas_precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices], 
                                                                        nodal_displacements_elementwise_per_elem_type)
        densities_by_quadrature_point = np.column_stack([polyplas_object.filtered_and_projected_design_variables_per_elem_type[number_of_vertices] for _ in range(number_of_quadrature_points_per_element)]).ravel()
        new_densities = densities_by_quadrature_point.copy()
        new_densities += perturbation
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = polyplas_object.elastic_material_interpolation.get_value(new_densities)
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = polyplas_object.plastic_material_interpolation.get_value(new_densities)
        global_residual_f1 = polyplas_object.get_global_residual_vector_for_elem_type(number_of_vertices, current_volume_avg_volumetric_strain_per_elem_type)
        new_densities -= 2.0 * perturbation
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = polyplas_object.elastic_material_interpolation.get_value(new_densities)
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = polyplas_object.plastic_material_interpolation.get_value(new_densities)
        global_residual_b1 = polyplas_object.get_global_residual_vector_for_elem_type(number_of_vertices, current_volume_avg_volumetric_strain_per_elem_type)
        new_densities += perturbation
        numerical_derivatives = (global_residual_f1 - global_residual_b1) / (2.0 * perturbation)
        numerical_derivative = np.sum(numerical_derivatives, axis=1)
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = saved_elastic_interpolation
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = saved_plastic_interpolation
        print(numerical_derivatives)
        print(internal_force_derivatives_wrt_densities)
        assert np.allclose(numerical_derivatives, internal_force_derivatives_wrt_densities)


def test_global_residual_derivative_wrt_nodal_displacements(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps=50
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )
    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        _, internal_force_derivative_wrt_current_displacements,_ = polyplas_object.compute_global_residual_sensitivity_per_elem_type(number_of_vertices)
        ######### Checking derivatives of global residual wrt current nodal displacements #######################################
        saved_nodal_displacements = polyplas_object.state.nodal_displacement_vector.copy()
        saved_current_total_strain_tensor = copy.deepcopy(polyplas_object.state.current_total_strain_tensors)
        perturbation = 1.0e-6
        nodal_dof=0
        element_index = 0   #has to be equal to the correct corresponding nodal_dof
        new_nodal_displacements = polyplas_object.state.nodal_displacement_vector[polyplas_precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
        new_nodal_displacements[0, nodal_dof] += perturbation
        current_volume_avg_volumetric_strain_per_elem_type = np.einsum("e,ei,ei->e", 
                                                                        1.0 / polyplas_precomputed_data.element_volumes_per_elem_type[number_of_vertices], 
                                                                        polyplas_precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices], 
                                                                        new_nodal_displacements)
        new_current_total_strain_tensor = np.einsum("eqimn,ei->eqmn", polyplas_precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices], new_nodal_displacements)
        polyplas_object.state.current_total_strain_tensors[number_of_vertices][:,:,:2,:2] = new_current_total_strain_tensor
        global_residual_f1 = polyplas_object.get_global_residual_vector_for_elem_type(number_of_vertices, current_volume_avg_volumetric_strain_per_elem_type)
        new_nodal_displacements[0, nodal_dof] -= 2.0 * perturbation
        current_volume_avg_volumetric_strain_per_elem_type = np.einsum("e,ei,ei->e", 
                                                                        1.0 / polyplas_precomputed_data.element_volumes_per_elem_type[number_of_vertices], 
                                                                        polyplas_precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices], 
                                                                        new_nodal_displacements)
        new_current_total_strain_tensor = np.einsum("eqimn,ei->eqmn", polyplas_precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices], new_nodal_displacements)
        polyplas_object.state.current_total_strain_tensors[number_of_vertices][:,:,:2,:2] = new_current_total_strain_tensor
        global_residual_b1 = polyplas_object.get_global_residual_vector_for_elem_type(number_of_vertices, current_volume_avg_volumetric_strain_per_elem_type)
        numerical_derivatives = (global_residual_f1 - global_residual_b1) / (2.0 * perturbation)
        polyplas_object.state.nodal_displacement_vector = saved_nodal_displacements
        polyplas_object.state.current_total_strain_tensors = saved_current_total_strain_tensor
        max_entry = np.max(internal_force_derivative_wrt_current_displacements[:,:,nodal_dof])
        assert numerical_derivatives[element_index,:] == pytest.approx(internal_force_derivative_wrt_current_displacements[element_index,nodal_dof,:], rel=1.0e-6*max_entry)

def test_global_residual_derivative_wrt_current_local_variables(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps=20
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )
    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    _, _, _,_ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]

        _, _,internal_force_derivative_wrt_current_plastic_state = polyplas_object.compute_global_residual_sensitivity_per_elem_type(number_of_vertices)
        # checking derivative of the internal force wrt current plastic strain
        perturbation = 1.0e-5
        base_plastic_strain = polyplas_object.state.current_plastic_strain_tensors.copy()
        nodal_displacements_elementwise_per_elem_type = polyplas_object.state.nodal_displacement_vector[polyplas_precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
        current_volume_avg_volumetric_strain_per_elem_type = np.einsum("e,ei,ei->e", 
                                                                        1.0 / polyplas_precomputed_data.element_volumes_per_elem_type[number_of_vertices], 
                                                                        polyplas_precomputed_data.virtual_displacement_volume_weighted_divergence[number_of_vertices], 
                                                                        nodal_displacements_elementwise_per_elem_type)
        ii = np.array([0, 1, 0, 2], dtype=int)
        jj = np.array([0, 1, 1, 2], dtype=int)
        quadrature_pt_index = 1
        element_index = 0
        k = 0
        local_variable_index = k + 2
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_index, quadrature_pt_index , ii[k], jj[k]] += perturbation
        internal_force_f1 = polyplas_object.get_global_residual_vector_for_elem_type(number_of_vertices, current_volume_avg_volumetric_strain_per_elem_type)
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_index, quadrature_pt_index, ii[k], jj[k]] -= 2.0 * perturbation
        internal_force_b1 = polyplas_object.get_global_residual_vector_for_elem_type(number_of_vertices, current_volume_avg_volumetric_strain_per_elem_type)
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_index,quadrature_pt_index, ii[k], jj[k]] += perturbation
        numerical_derivative = (internal_force_f1[element_index, :] - internal_force_b1[element_index, :]) / (2.0 * perturbation)
        polyplas_object.state.current_plastic_strain_tensors = base_plastic_strain
        max_entry = np.max(internal_force_derivative_wrt_current_plastic_state[element_index, quadrature_pt_index, :, local_variable_index])
        print(numerical_derivative)
        print(internal_force_derivative_wrt_current_plastic_state[element_index, quadrature_pt_index, :, local_variable_index])
        assert numerical_derivative == pytest.approx(internal_force_derivative_wrt_current_plastic_state[element_index, quadrature_pt_index, :, local_variable_index], rel=1.0e-6*max_entry)

def test_local_residual_derivative_wrt_densities(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 20
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )

    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    _, _, _,_ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        local_residual_derivative_wrt_densities, _, _, _ = polyplas_object.compute_local_residual_sensitivity_per_elem_type(number_of_vertices)
        total_number_of_elements_per_elem_type = polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type

        saved_elastic_interpolation = polyplas_object.elastic_interpolation_function_values[number_of_vertices][:]
        saved_plastic_interpolation = polyplas_object.plastic_interpolation_function_values[number_of_vertices][:]
        penalized_initial_yield_stress = saved_plastic_interpolation * polyplas_precomputed_data.initial_yield_stress
        penalized_hardening_modulus = saved_elastic_interpolation * polyplas_precomputed_data.hardening_modulus
        elastic_strain_tensor = polyplas_object.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))\
              - polyplas_object.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        deviatoric_elastic_strain = np.einsum("ijkl,qkl->qij", polyplas_precomputed_data.deviatoric_projection_tensor, elastic_strain_tensor)
        deviatoric_stress_tensor = np.einsum("...,q,qij->qij", 2.0*polyplas_precomputed_data.shear_modulus, saved_elastic_interpolation, deviatoric_elastic_strain)

        deviatoric_stress_norms = np.linalg.norm(deviatoric_stress_tensor, axis=(1, 2), ord='fro')
        current_stress = ((3.0/2.0)**0.5) * deviatoric_stress_norms
        current_yield_stress = penalized_initial_yield_stress + penalized_hardening_modulus * polyplas_object.state.previous_accumulated_plastic_strains[number_of_vertices]\
            .reshape((total_number_of_quad_pts_per_element_type,))

        yield_function = current_stress - current_yield_stress
        elastic = yield_function <= 1.0e-10   #creating mask
        plastic = ~elastic

        perturbation = 1.0e-5
       
        number_of_quadrature_points_per_element = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        densities_by_quadrature_point = np.column_stack([polyplas_object.filtered_and_projected_design_variables_per_elem_type[number_of_vertices] for _ in range(number_of_quadrature_points_per_element)]).ravel()
        new_densities = densities_by_quadrature_point.copy()
        new_densities += perturbation
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = polyplas_object.elastic_material_interpolation.get_value(new_densities)
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = polyplas_object.plastic_material_interpolation.get_value(new_densities)
        local_residual_f1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        new_densities -= 2.0 * perturbation
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = polyplas_object.elastic_material_interpolation.get_value(new_densities)
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = polyplas_object.plastic_material_interpolation.get_value(new_densities)
        local_residual_b1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        new_densities += perturbation
        numerical_derivatives = (local_residual_f1 - local_residual_b1) / (2.0 * perturbation)
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = saved_elastic_interpolation
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = saved_plastic_interpolation
        max_entry = np.max(local_residual_derivative_wrt_densities)
        print(numerical_derivatives)
        print(local_residual_derivative_wrt_densities)
        assert np.allclose(local_residual_derivative_wrt_densities, numerical_derivatives)   

def test_local_residual_derivative_wrt_current_nodal_displacments(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]
    number_of_time_steps = 20
    volume_fraction_upper_bound = 0.5
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )

    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        _, local_residual_derivative_wrt_current_nodal_displacements, _, _ = polyplas_object.compute_local_residual_sensitivity_per_elem_type(number_of_vertices)
        total_number_of_elements_per_elem_type = polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type

        elastic_interpolation = polyplas_object.elastic_interpolation_function_values.copy()
        plastic_interpolation = polyplas_object.plastic_interpolation_function_values.copy()
        penalized_initial_yield_stress = plastic_interpolation[number_of_vertices] * polyplas_precomputed_data.initial_yield_stress
        penalized_hardening_modulus = elastic_interpolation[number_of_vertices] * polyplas_precomputed_data.hardening_modulus
        elastic_strain_tensor = polyplas_object.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))\
              - polyplas_object.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        deviatoric_elastic_strain = np.einsum("ijkl,qkl->qij", polyplas_precomputed_data.deviatoric_projection_tensor, elastic_strain_tensor)
        deviatoric_stress_tensor = np.einsum("...,q,qij->qij", 2.0*polyplas_precomputed_data.shear_modulus, elastic_interpolation[number_of_vertices], deviatoric_elastic_strain)

        deviatoric_stress_norms = np.linalg.norm(deviatoric_stress_tensor, axis=(1, 2), ord='fro')
        current_stress = ((3.0/2.0)**0.5) * deviatoric_stress_norms
        current_yield_stress = penalized_initial_yield_stress + penalized_hardening_modulus * polyplas_object.state.previous_accumulated_plastic_strains[number_of_vertices]\
            .reshape((total_number_of_quad_pts_per_element_type,))

        yield_function = current_stress - current_yield_stress
        elastic = yield_function <= 1.0e-10   #creating mask
        plastic = ~elastic

        saved_nodal_displacements = polyplas_object.state.nodal_displacement_vector.copy()
        saved_current_total_strain_tensor = copy.deepcopy(polyplas_object.state.current_total_strain_tensors)
        perturbation = 1.0e-8
        nodal_dof=5
        element_index = 0
        number_of_local_dofs = 6
       
        new_nodal_displacements = polyplas_object.state.nodal_displacement_vector[polyplas_precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
        new_nodal_displacements[element_index, nodal_dof] += perturbation
        new_current_total_strain_tensor = np.einsum("eqimn,ei->eqmn", polyplas_precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices], new_nodal_displacements)
        polyplas_object.state.current_total_strain_tensors[number_of_vertices][:,:,:2,:2] = new_current_total_strain_tensor
        local_residual_f1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        new_nodal_displacements[element_index, nodal_dof] -= 2.0 * perturbation
        new_current_total_strain_tensor = np.einsum("eqimn,ei->eqmn", polyplas_precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices], new_nodal_displacements)
        polyplas_object.state.current_total_strain_tensors[number_of_vertices][:,:,:2,:2] = new_current_total_strain_tensor
        local_residual_b1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        numerical_derivatives = (local_residual_f1 - local_residual_b1) / (2.0 * perturbation)
        polyplas_object.state.nodal_displacement_vector = saved_nodal_displacements
        polyplas_object.state.current_total_strain_tensors = saved_current_total_strain_tensor
        max_entry = np.max(local_residual_derivative_wrt_current_nodal_displacements[:,:,nodal_dof])
        reshaped_numerical_derivatives = numerical_derivatives.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_dofs))
        reshaped_local_residual_derivative_wrt_current_nodal_displacements = local_residual_derivative_wrt_current_nodal_displacements[:,:,nodal_dof].\
                                                                    reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type,number_of_local_dofs))
        print(reshaped_numerical_derivatives[element_index, :, :])
        print(reshaped_local_residual_derivative_wrt_current_nodal_displacements[element_index,:,:])
        assert reshaped_numerical_derivatives[element_index, :, :] == pytest.approx(reshaped_local_residual_derivative_wrt_current_nodal_displacements[element_index,:,:], rel=1.0e-3*max_entry)


def test_local_residual_derivative_wrt_current_local_variables(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 20
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )
    
    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        _, _, local_residual_derivative_wrt_current_local_variables, _ = polyplas_object.compute_local_residual_sensitivity_per_elem_type(number_of_vertices)
        total_number_of_elements_per_elem_type = polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type
        number_of_local_variables = 6

        elastic_interpolation = polyplas_object.elastic_interpolation_function_values.copy()
        plastic_interpolation = polyplas_object.plastic_interpolation_function_values.copy()
        penalized_initial_yield_stress = plastic_interpolation[number_of_vertices] * polyplas_precomputed_data.initial_yield_stress
        penalized_hardening_modulus = elastic_interpolation[number_of_vertices] * polyplas_precomputed_data.hardening_modulus
        elastic_strain_tensor = polyplas_object.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))\
              - polyplas_object.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        deviatoric_elastic_strain = np.einsum("ijkl,qkl->qij", polyplas_precomputed_data.deviatoric_projection_tensor, elastic_strain_tensor)
        deviatoric_stress_tensor = np.einsum("...,q,qij->qij", 2.0*polyplas_precomputed_data.shear_modulus, elastic_interpolation[number_of_vertices], deviatoric_elastic_strain)

        deviatoric_stress_norms = np.linalg.norm(deviatoric_stress_tensor, axis=(1, 2), ord='fro')
        current_stress = ((3.0/2.0)**0.5) * deviatoric_stress_norms
        current_yield_stress = penalized_initial_yield_stress + penalized_hardening_modulus * polyplas_object.state.previous_accumulated_plastic_strains[number_of_vertices]\
            .reshape((total_number_of_quad_pts_per_element_type,))

        yield_function = current_stress - current_yield_stress
        elastic = yield_function <= 1.0e-10   #creating mask
        plastic = ~elastic

        # checking derivatives wrt accumulated plastic strain
        element_dof = 0
        quadrature_pt_index = 3
        perturbation = 1.0e-6
        polyplas_object.state.current_accumulated_plastic_strains[number_of_vertices][element_dof, quadrature_pt_index] += perturbation
        local_residual_f1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.current_accumulated_plastic_strains[number_of_vertices][element_dof, quadrature_pt_index] -= 2.0 * perturbation
        local_residual_b1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.current_accumulated_plastic_strains[number_of_vertices][element_dof, quadrature_pt_index]  += perturbation
        numerical_derivatives = (local_residual_f1[:,:] - local_residual_b1[:,:]) / (2.0 * perturbation)
        reshaped_numerical_derivatives = numerical_derivatives.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_variables))
        reshaped_analytical_derivatives = local_residual_derivative_wrt_current_local_variables.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type,\
                                                                                                             number_of_local_variables, number_of_local_variables))
        print(reshaped_numerical_derivatives[element_dof, quadrature_pt_index,:])
        print(reshaped_analytical_derivatives[element_dof, quadrature_pt_index,:,0])
        assert np.allclose(reshaped_numerical_derivatives[element_dof, quadrature_pt_index,:], reshaped_analytical_derivatives[element_dof, quadrature_pt_index,:,0])

        # Checking derivatives wrt plastic multiplier increment
        polyplas_object.state.plastic_multiplier_increments[number_of_vertices][element_dof, quadrature_pt_index] += perturbation
        local_residual_f1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.plastic_multiplier_increments[number_of_vertices][element_dof, quadrature_pt_index] -= 2.0 * perturbation
        local_residual_b1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.plastic_multiplier_increments[number_of_vertices][element_dof, quadrature_pt_index] += perturbation
        numerical_derivatives = (local_residual_f1[:,:] - local_residual_b1[:,:]) / (2.0 * perturbation)
        reshaped_numerical_derivatives = numerical_derivatives.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_variables))
        print(reshaped_numerical_derivatives[element_dof, quadrature_pt_index,:])
        print(reshaped_analytical_derivatives[element_dof, quadrature_pt_index,:,1])
        assert np.allclose(reshaped_numerical_derivatives[element_dof,quadrature_pt_index,:], reshaped_analytical_derivatives[element_dof,quadrature_pt_index,:,1])

        # checking derivatives wrt current plastic strain
        strain_index_i = np.array([0, 1, 0, 2], dtype=int)
        strain_index_j = np.array([0, 1, 1, 2], dtype=int)
        k = 3
        local_variable_index = k + 2
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_dof, quadrature_pt_index, strain_index_i[k], strain_index_j[k]] += perturbation
        local_residual_f1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_dof, quadrature_pt_index, strain_index_i[k], strain_index_j[k]] -= 2.0 * perturbation
        local_residual_b1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_dof, quadrature_pt_index, strain_index_i[k], strain_index_j[k]]  += perturbation
        numerical_derivatives = (local_residual_f1[:,:] - local_residual_b1[:,:]) / (2.0 * perturbation)
        reshaped_numerical_derivatives = numerical_derivatives.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_variables))
        print(reshaped_numerical_derivatives[element_dof, quadrature_pt_index, :], reshaped_analytical_derivatives[element_dof, quadrature_pt_index, :, local_variable_index])
        assert np.allclose(reshaped_numerical_derivatives[element_dof, quadrature_pt_index, :], reshaped_analytical_derivatives[element_dof, quadrature_pt_index, :, local_variable_index])

def test_local_residual_derivative_wrt_previous_local_variables(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 50
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )
    
    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state = polyplas_object.states[number_of_time_steps-1]
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        _, _, _, local_residual_derivative_wrt_previous_local_variables = polyplas_object.compute_local_residual_sensitivity_per_elem_type(number_of_vertices)
        total_number_of_elements_per_elem_type = polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        total_number_of_quad_pts_per_element_type = total_number_of_elements_per_elem_type * number_of_quadrature_pts_per_elem_type
        number_of_local_variables = 6

        elastic_interpolation = polyplas_object.elastic_interpolation_function_values.copy()
        plastic_interpolation = polyplas_object.plastic_interpolation_function_values.copy()
        penalized_initial_yield_stress = plastic_interpolation[number_of_vertices] * polyplas_precomputed_data.initial_yield_stress
        penalized_hardening_modulus = elastic_interpolation[number_of_vertices] * polyplas_precomputed_data.hardening_modulus
        elastic_strain_tensor = polyplas_object.state.current_total_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type, 3, 3))\
              - polyplas_object.state.current_plastic_strain_tensors[number_of_vertices].reshape((total_number_of_quad_pts_per_element_type,3,3))
        deviatoric_elastic_strain = np.einsum("ijkl,qkl->qij", polyplas_precomputed_data.deviatoric_projection_tensor, elastic_strain_tensor)
        deviatoric_stress_tensor = np.einsum("...,q,qij->qij", 2.0*polyplas_precomputed_data.shear_modulus, elastic_interpolation[number_of_vertices], deviatoric_elastic_strain)

        deviatoric_stress_norms = np.linalg.norm(deviatoric_stress_tensor, axis=(1, 2), ord='fro')
        current_stress = ((3.0/2.0)**0.5) * deviatoric_stress_norms
        current_yield_stress = penalized_initial_yield_stress + penalized_hardening_modulus * polyplas_object.state.previous_accumulated_plastic_strains[number_of_vertices]\
            .reshape((total_number_of_quad_pts_per_element_type,))

        yield_function = current_stress - current_yield_stress
        elastic = yield_function <= 1.0e-10   #creating mask
        plastic = ~elastic

        # checking derivatives wrt accumulated plastic strain
        element_dof = 0
        quadrature_pt_index = 3
        perturbation = 1.0e-6
        polyplas_object.state.previous_accumulated_plastic_strains[number_of_vertices][element_dof, quadrature_pt_index] += perturbation
        local_residual_f1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.previous_accumulated_plastic_strains[number_of_vertices][element_dof, quadrature_pt_index] -= 2.0 * perturbation
        local_residual_b1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.previous_accumulated_plastic_strains[number_of_vertices][element_dof, quadrature_pt_index]  += perturbation
        numerical_derivatives = (local_residual_f1[:,:] - local_residual_b1[:,:]) / (2.0 * perturbation)
        reshaped_numerical_derivatives = numerical_derivatives.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_variables))
        reshaped_analytical_derivatives = local_residual_derivative_wrt_previous_local_variables.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type,\
                                                                                                             number_of_local_variables, number_of_local_variables))
        assert np.allclose(reshaped_numerical_derivatives[element_dof, quadrature_pt_index,:], reshaped_analytical_derivatives[element_dof, quadrature_pt_index,:,0])

        # Checking derivatives wrt plastic multiplier increment
        assert np.allclose(local_residual_derivative_wrt_previous_local_variables[:, :, 1], 0.0) 

        # checking derivatives wrt previous plastic strain
        strain_index_i = np.array([0, 1, 0, 2], dtype=int)
        strain_index_j = np.array([0, 1, 1, 2], dtype=int)
        k = 0
        local_variable_index = k + 2
        polyplas_object.state.previous_plastic_strain_tensors[number_of_vertices][element_dof, quadrature_pt_index, strain_index_i[k], strain_index_j[k]] += perturbation
        local_residual_f1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.previous_plastic_strain_tensors[number_of_vertices][element_dof, quadrature_pt_index, strain_index_i[k], strain_index_j[k]] -= 2.0 * perturbation
        local_residual_b1 = polyplas_object.get_local_residual_for_elem_type(elastic, plastic, number_of_vertices)
        polyplas_object.state.previous_plastic_strain_tensors[number_of_vertices][element_dof, quadrature_pt_index, strain_index_i[k], strain_index_j[k]]  += perturbation
        numerical_derivatives = (local_residual_f1[:,:] - local_residual_b1[:,:]) / (2.0 * perturbation)
        reshaped_numerical_derivatives = numerical_derivatives.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type, number_of_local_variables))
        assert np.allclose(reshaped_numerical_derivatives[element_dof, quadrature_pt_index, :], reshaped_analytical_derivatives[element_dof, quadrature_pt_index, :, local_variable_index])

def test_plastic_work_obj_derivative_wrt_densities(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 1
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )

    if number_of_time_steps > 1:
      raise ValueError("Test can only be run for single time step due to implementation of the .get_total_work_obj_for_elem_type() function")
    

    design_variables = np.zeros((polyplas_precomputed_data.number_of_nodes,), dtype=float)
    design_variables[:] = volume_fraction_upper_bound
    at_final_time_step = True
    time_step_index = 0
    
    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        plastic_work_obj_derivative_wrt_densities, _, _ = polyplas_object.get_plastic_work_obj_partials_per_elem_type(number_of_vertices, time_step_index, at_final_time_step)
        total_number_of_elements_per_elem_type = polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        plastic_work_obj_derivative_wrt_densities_elementwise = np.einsum("eq->e", plastic_work_obj_derivative_wrt_densities.reshape((total_number_of_elements_per_elem_type, number_of_quadrature_pts_per_elem_type)))
        plastic_work_obj_derivative_wrt_densities_per_elem_type = np.sum(plastic_work_obj_derivative_wrt_densities_elementwise)

        perturbation = 1.0e-5
        saved_elastic_interpolation = polyplas_object.elastic_interpolation_function_values[number_of_vertices][:]
        saved_plastic_interpolation = polyplas_object.plastic_interpolation_function_values[number_of_vertices][:]
        number_of_quadrature_points_per_element = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        densities_by_quadrature_point = np.column_stack([polyplas_object.filtered_and_projected_design_variables_per_elem_type[number_of_vertices] for _ in range(number_of_quadrature_points_per_element)]).ravel()
        new_densities = densities_by_quadrature_point.copy()
        new_densities += perturbation
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = polyplas_object.elastic_material_interpolation.get_value(new_densities)
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = polyplas_object.plastic_material_interpolation.get_value(new_densities)
        plastic_work_f1 = polyplas_object.get_plastic_work_objective_for_elem_type(number_of_vertices)
        new_densities -= 2.0 * perturbation
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = polyplas_object.elastic_material_interpolation.get_value(new_densities)
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = polyplas_object.plastic_material_interpolation.get_value(new_densities)
        plastic_work_b1 = polyplas_object.get_plastic_work_objective_for_elem_type(number_of_vertices)
        new_densities += perturbation
        numerical_derivatives = (plastic_work_f1 - plastic_work_b1) / (2.0 * perturbation)
        print(numerical_derivatives)
        print(plastic_work_obj_derivative_wrt_densities_per_elem_type)
        assert np.allclose(numerical_derivatives, plastic_work_obj_derivative_wrt_densities_per_elem_type)
        polyplas_object.elastic_interpolation_function_values[number_of_vertices] = saved_elastic_interpolation
        polyplas_object.plastic_interpolation_function_values[number_of_vertices] = saved_plastic_interpolation


def test_plastic_work_obj_derivative_wrt_nodal_displacements(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 1
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )
    
    if number_of_time_steps > 1:
        raise ValueError("Test can only be run for single time step due to implementation of the .get_total_work_obj_for_elem_type() function")

    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        at_final_time_step = True
        time_step_index = 0

        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        _, plastic_work_obj_derivative_wrt_current_displacements, _ = polyplas_object.get_plastic_work_obj_partials_per_elem_type(number_of_vertices, time_step_index, at_final_time_step)

        ######### Checking derivatives of total work wrt current nodal displacements #######################################
        saved_nodal_displacements = polyplas_object.state.nodal_displacement_vector.copy()
        saved_current_total_strain_tensor = copy.deepcopy(polyplas_object.state.current_total_strain_tensors)
        perturbation = 1.0e-8
        nodal_dof_index = 6
        element_index = 0 #much be valid index accross all element types
        new_nodal_displacements = polyplas_object.state.nodal_displacement_vector[polyplas_precomputed_data.global_dof_indices_elementwise[number_of_vertices]]
        new_nodal_displacements[element_index,nodal_dof_index] += perturbation
        new_current_total_strain_tensor = np.einsum("eqimn,ei->eqmn", polyplas_precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices], new_nodal_displacements)
        polyplas_object.state.current_total_strain_tensors[number_of_vertices][:,:,:2,:2] = new_current_total_strain_tensor
        plastic_work_f1 = polyplas_object.get_plastic_work_objective_for_elem_type(number_of_vertices)
        new_nodal_displacements[element_index,nodal_dof_index] -= 2.0 * perturbation
        new_current_total_strain_tensor = np.einsum("eqimn,ei->eqmn", polyplas_precomputed_data.virtual_displacement_symmetric_gradients[number_of_vertices], new_nodal_displacements)
        polyplas_object.state.current_total_strain_tensors[number_of_vertices][:,:,:2,:2] = new_current_total_strain_tensor
        plastic_work_b1 = polyplas_object.get_plastic_work_objective_for_elem_type(number_of_vertices)
        numerical_derivatives = (plastic_work_f1 - plastic_work_b1) / (2.0 * perturbation)
        print(numerical_derivatives)
        print(plastic_work_obj_derivative_wrt_current_displacements[:,nodal_dof_index])
        assert np.allclose(numerical_derivatives, plastic_work_obj_derivative_wrt_current_displacements[element_index, nodal_dof_index])
        polyplas_object.state.nodal_displacement_vector = saved_nodal_displacements
        polyplas_object.state.current_total_strain_tensors = saved_current_total_strain_tensor 
        

def test_plastic_work_obj_derivative_wrt_local_variables(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]

    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 1
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-8, penalization_parameter=1.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-4, penalization_parameter=1.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )

    if number_of_time_steps > 1:
      raise ValueError("Test can only be run for single time step due to implementation of the .get_total_work_obj_for_elem_type() function")

    _, _, _, _ = polyplas_object.run_forward_analysis()
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        at_final_time_step = True
        time_step_index = 0

        polyplas_object.state =  polyplas_object.states[number_of_time_steps-1]
        _, _, plastic_work_obj_derivative_wrt_current_local_variables = polyplas_object.get_plastic_work_obj_partials_per_elem_type(number_of_vertices, time_step_index, at_final_time_step)

        ######### Checking derivatives of total work wrt current local variables #######################################
        # checking derivatives wrt current plastic strain
        total_number_of_elements_per_elem_type = polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0]
        number_of_quadrature_pts_per_elem_type = polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]
        number_of_local_dofs = 6
        
        plastic_work_derivative_wrt_local_variables = plastic_work_obj_derivative_wrt_current_local_variables.reshape((total_number_of_elements_per_elem_type, \
                                                                number_of_quadrature_pts_per_elem_type, number_of_local_dofs))
        perturbation = 1.0e-8
        base_plastic_strain = polyplas_object.state.current_plastic_strain_tensors.copy()
        ii = np.array([0, 1, 0, 2], dtype=int)
        jj = np.array([0, 1, 1, 2], dtype=int)
        element_index = 0 #must be valid accorss all element types
        quad_index = 2
        k = 3
        local_variable_index = k + 2
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_index, quad_index, ii[k], jj[k]] += perturbation
        total_work_f1 = polyplas_object.get_plastic_work_objective_for_elem_type(number_of_vertices)
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_index, quad_index, ii[k], jj[k]] -= 2.0 * perturbation
        total_work_b1 = polyplas_object.get_plastic_work_objective_for_elem_type(number_of_vertices)
        polyplas_object.state.current_plastic_strain_tensors[number_of_vertices][element_index, quad_index, ii[k], jj[k]] += perturbation
        numerical_derivatives = (total_work_f1 - total_work_b1) / (2.0 * perturbation)
        polyplas_object.state.current_plastic_strain_tensors = base_plastic_strain.copy()
        print(numerical_derivatives)
        print(plastic_work_derivative_wrt_local_variables[element_index, quad_index, local_variable_index])   
        assert np.allclose(numerical_derivatives, plastic_work_derivative_wrt_local_variables[element_index, quad_index, local_variable_index])

def test_update_logged_values_data_frame(square_mesh_and_polyplas_precomputed_data_uniaxial_deformation):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_uniaxial_deformation[1]
    volume_fraction_upper_bound = 0.5
    number_of_time_steps = 1
    polyplas_object = PolyPlas(
        design_variable_filter=PolynomialFilter(polyplas_precomputed_data, filter_radius=0.25),
        elastic_material_interpolation=SIMP(ersatz_parameter=1.0e-3, penalization_parameter=2.0),
        plastic_material_interpolation=SIMP(ersatz_parameter=1.0e-3, penalization_parameter=2.0),
        projection=TanhHeavisideProjection(projection_strength=1.5, projection_threshold=0.25),
        precomputed_data=polyplas_precomputed_data,
        volume_fraction_upper_bound=volume_fraction_upper_bound,
        number_of_time_steps=number_of_time_steps,
        directory_name="directory_for_tests"
    )
    logged_values = {'Test': 1.125}
    polyplas_object.update_logged_values_data_frame(logged_values)
    assert isinstance(polyplas_object.logged_values_data_frame, pd.DataFrame)
    expected_dictionary = {'Test': 1.125,
                           'SIMP Penalization Parameter': 2.0,
                           'SIMP Ersatz Parameter': 1.0e-3,
                           'Tanh Heaviside Projection Strength': 1.5,
                           'Tanh Heaviside Projection Threshold': 0.25}
    for expected_key, expected_value in expected_dictionary.items():
        assert expected_key in polyplas_object.logged_values_data_frame.columns
        assert polyplas_object.logged_values_data_frame[expected_key].iloc[0] == expected_value
