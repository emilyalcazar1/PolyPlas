import pytest
import numpy as np
import src.polymesher.PolyMesher as polymesher
import src.polyplas.PolyPlasUtilities as polyplas_utils

def test_init_with_number_of_elements(boundary_value_problem):
    number_of_elements = 10
    polymesh = polymesher.PolyMesher(boundary_value_problem,
                                       number_of_elements=number_of_elements)
    assert polymesh.number_of_elements == number_of_elements
    assert polymesh.element_center_points.shape == (number_of_elements, 2)

def test_init_with_element_center_points(boundary_value_problem):
    element_center_points = np.array([[0.0, 0.0],
                                      [1.0, 0.0],
                                      [0.0, 1.0]])
    polymesh = polymesher.PolyMesher(boundary_value_problem,
                                       element_center_points=element_center_points)
    assert polymesh.number_of_elements == element_center_points.shape[0]
    assert np.allclose(polymesh.element_center_points, element_center_points)

def test_init_with_invalid_element_center_points_type(boundary_value_problem):
    with pytest.raises(TypeError):
        polymesher.PolyMesher(boundary_value_problem,
                                element_center_points=[(0.0, 0.0),
                                                       (1.0, 0.0),
                                                       (0.0, 1.0)])

def test_init_with_invalid_element_center_points_shape(boundary_value_problem):
    with pytest.raises(ValueError):
        polymesher.PolyMesher(boundary_value_problem,
                                element_center_points=np.array([[0.0, 0.0, 0.0],
                                                                [1.0, 0.0, 0.0]]))

def test_init_with_neither_element_center_points_nor_number_of_elems(boundary_value_problem):
    with pytest.raises(ValueError):
        polymesher.PolyMesher(boundary_value_problem)

def test_init_with_invalid_number_of_elements_type(boundary_value_problem):
    with pytest.raises(TypeError):
        polymesher.PolyMesher(boundary_value_problem,
                                number_of_elements="not an integer")

def test_init_with_invalid_number_of_elements_value(boundary_value_problem):
    with pytest.raises(ValueError):
        polymesher.PolyMesher(boundary_value_problem,
                                number_of_elements=0)

def test_init_with_invalid_maximum_iterations_type(boundary_value_problem):
    with pytest.raises(TypeError):
        polymesher.PolyMesher(boundary_value_problem,
                                number_of_elements=10,
                                maximum_iterations="not an integer")

def test_init_with_invalid_maximum_iterations_value(boundary_value_problem):
    with pytest.raises(ValueError):
        polymesher.PolyMesher(boundary_value_problem,
                                number_of_elements=10,
                                maximum_iterations=-1)

def test_generate_random_initial_element_centers(boundary_value_problem):
    number_of_elements = 10
    polymesh = polymesher.PolyMesher(boundary_value_problem,
                                       number_of_elements=number_of_elements)
    element_center_points = polymesh.generate_random_initial_element_centers()
    assert element_center_points.shape == (number_of_elements, 2)

def test_generate_mesh(boundary_value_problem):
    number_of_elements = 100
    polymesh_object = polymesher.PolyMesher(boundary_value_problem,
                                       number_of_elements=number_of_elements)
    material_parameters = dict(elastic_modulus=10000.0,
                        poissons_ratio = 0.3,
                        hardening_modulus = 100.0, 
                        initial_yield_stress = 10.0) 
    
    precomputed_data = polyplas_utils.get_precomputed_data_filter_and_heaviside_projection(polymesh_object, material_parameters)
    assert precomputed_data.number_of_elements == number_of_elements
    # Make sure the jacobian determinants are all positive
    for jacobian_x_quadrature_weights in precomputed_data.jacobian_determinant_x_quadrature_weights.values():
        assert np.all(jacobian_x_quadrature_weights > 0.0)

def test_generate_mesh_plot_without_error(boundary_value_problem):
    number_of_elements = 10
    polymesh = polymesher.PolyMesher(boundary_value_problem,
                                       number_of_elements=number_of_elements)
    polymesh.plot()
