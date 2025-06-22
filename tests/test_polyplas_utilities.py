import pytest
import sys
import numpy as np
import scipy.sparse as sp

import src.polyplas.PolyPlasUtilities as polyplas_utils
from typing import Dict, List, NamedTuple, Optional, Tuple


def test_valid_material_parameter_input(polymesh_and_polyplas_precomputed_data):
    polyplas_precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    shear_modulus = polyplas_precomputed_data.shear_modulus
    bulk_modulus = polyplas_precomputed_data.bulk_modulus
    hadening_modulus = polyplas_precomputed_data.hardening_modulus
    initial_yield_stress = polyplas_precomputed_data.initial_yield_stress
    assert all(x >= 0 for x in [shear_modulus, bulk_modulus, hadening_modulus, initial_yield_stress])

def test_shape_function_gradient(polymesh_and_polyplas_precomputed_data):
    polyplas_precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    shape_function_gradients = polyplas_precomputed_data.shape_function_gradients
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        number_of_elements_per_element_type = (polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0])
        assert shape_function_gradients[number_of_vertices].shape == (number_of_elements_per_element_type,
                                                                      polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices],
                                                                    number_of_vertices,
                                                                    polyplas_precomputed_data.space_dimension)
        for element_number in range(number_of_elements_per_element_type):
            for quadrature_point in range(polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices]):
                assert np.allclose(np.sum(shape_function_gradients[number_of_vertices][element_number][quadrature_point]), 0.0)
    assert isinstance(shape_function_gradients, Dict)

def test_shape_function_values(polymesh_and_polyplas_precomputed_data):
    polyplas_precomputed_data = polymesh_and_polyplas_precomputed_data[1]
    shape_function_values = polyplas_precomputed_data.shape_function_values
    unique_numbers_of_vertices = polyplas_precomputed_data.element_connectivity_arrays.keys()
    for number_of_vertices in unique_numbers_of_vertices:
        number_of_elements_per_element_type = (polyplas_precomputed_data.element_connectivity_arrays[number_of_vertices].shape[0])
        assert shape_function_values[number_of_vertices].shape == (polyplas_precomputed_data.number_of_quad_points_per_element[number_of_vertices],
                                                                    number_of_vertices)
        
    for number_of_vertices in unique_numbers_of_vertices:
        print(shape_function_values[number_of_vertices])
        assert np.allclose(np.sum(shape_function_values[number_of_vertices], axis=1), 1.0)
    assert isinstance(shape_function_values, Dict)

def test_linear_elastic_constitutive_tensor(square_mesh_and_polyplas_precomputed_data_and_material_dict):
    polyplas_precomputed_data = square_mesh_and_polyplas_precomputed_data_and_material_dict[1]
    plasticity_parameters = square_mesh_and_polyplas_precomputed_data_and_material_dict[2]
    poissons_ratio = plasticity_parameters['poissons_ratio']
    elastic_modulus = plasticity_parameters['elastic_modulus']
    
    constitutive_tensor_to_check = polyplas_precomputed_data.linear_elastic_constitutive_tensor
    bulk_modulus = elastic_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio))
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poissons_ratio))

    assert isinstance(constitutive_tensor_to_check, np.ndarray)
    print(constitutive_tensor_to_check.shape)
    assert constitutive_tensor_to_check.shape == (3, 3, 3, 3)
    identity_tensor = np.eye(3, dtype=np.float64)
    IxI = np.einsum("ij, kl -> ijkl", identity_tensor, identity_tensor)
    temp1 = np.einsum("im,jn->ijmn", identity_tensor, identity_tensor)
    temp2 = np.einsum("in,jm->ijmn", identity_tensor, identity_tensor)
    fourth_order_symmetric_identity_tensor = 0.5 * (temp1 + temp2)
    volumetric_projection_tensor = (1.0/3.0)*IxI
    deviatoric_projection_tensor = fourth_order_symmetric_identity_tensor - volumetric_projection_tensor
    linear_elastic_constitutive_tensor = bulk_modulus * IxI + 2.0 * shear_modulus * deviatoric_projection_tensor   #plane strain
    assert np.allclose(constitutive_tensor_to_check, linear_elastic_constitutive_tensor)





