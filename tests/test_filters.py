import numpy as np
import pytest
from PolyFilter import get_design_variable_filter, IdentityFilter

def test_get_design_variable_filter_identity():
    identity_filter = get_design_variable_filter(polymesh_object=None,
                                                 precomputed_data=None,
                                                 filter_type="identity")
    assert isinstance(identity_filter, IdentityFilter)
    assert identity_filter.type == "identity"

def test_get_design_variable_filter_filter_type(polymesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    with pytest.raises(ValueError):
        get_design_variable_filter(polymesh_object=polymesh,
                                   precomputed_data=polyplas_precomputed_data,
                                   filter_type="not a filter type",
                                   filter_radius=0.5)

def test_get_design_variable_filter_polymesh_object_type(polymesh_and_polyplas_precomputed_data):
    _, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    with pytest.raises(TypeError):
        get_design_variable_filter(polymesh_object="not a PolyMesher object",
                                   precomputed_data=polyplas_precomputed_data,
                                   filter_type="polynomial",
                                   filter_radius=0.5)

def test_get_design_variable_filter_filter_radius_type(polymesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    with pytest.raises(TypeError):
        get_design_variable_filter(polymesh_object=polymesh,
                                   precomputed_data=polyplas_precomputed_data,
                                   filter_type="polynomial",
                                   filter_radius="not a number")

def test_get_design_variable_filter_filter_radius_value(polymesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    with pytest.raises(ValueError):
        get_design_variable_filter(polymesh_object=polymesh,
                                   precomputed_data=polyplas_precomputed_data,
                                   filter_type="polynomial",
                                   filter_radius=-0.5)

def test_get_design_variable_filter_use_relative_filter_radius_type(polymesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    with pytest.raises(TypeError):
        get_design_variable_filter(polymesh_object=polymesh,
                                   precomputed_data=polyplas_precomputed_data,
                                   filter_type="polynomial",
                                   filter_radius=1.0,
                                   use_relative_filter_radius="not a boolean")

def test_get_design_variable_filter_precomputed_data_type(polymesh_and_polyplas_precomputed_data):
    polymesh, _, _ = polymesh_and_polyplas_precomputed_data
    with pytest.raises(TypeError):
        get_design_variable_filter(polymesh_object=polymesh,
                                   precomputed_data="Not a PrecomputedData object",
                                   filter_type="polynomial",
                                   filter_radius=1.0)

def test_get_design_variable_filter_polynomial(polymesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    filter_radius = 1.5
    _, largest_edge_length = polymesh.get_smallest_and_largest_edge_length()
    polynomial_filter = get_design_variable_filter(polymesh_object=polymesh,
                                                   precomputed_data=polyplas_precomputed_data,
                                                   filter_type="polynomial",
                                                   filter_radius=filter_radius,
                                                   use_relative_filter_radius=True)
    assert polynomial_filter.type == "polynomial"
    assert polynomial_filter.filter_radius == filter_radius * largest_edge_length

def test_set_filter_radius_type(polymesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    polynomial_filter = get_design_variable_filter(polymesh_object=polymesh,
                                                   precomputed_data=polyplas_precomputed_data,
                                                   filter_type="polynomial",
                                                   filter_radius=1.0)
    with pytest.raises(TypeError):
        polynomial_filter.filter_radius = "not a number"

def test_set_filter_radius_value(polymesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data, _ = polymesh_and_polyplas_precomputed_data
    polynomial_filter = get_design_variable_filter(polymesh_object=polymesh,
                                                   precomputed_data=polyplas_precomputed_data,
                                                   filter_type="polynomial",
                                                   filter_radius=1.0)
    with pytest.raises(ValueError):
        polynomial_filter.filter_radius = -1.0

def test_identity_filter():
    identity_filter = get_design_variable_filter(polymesh_object=None,
                                                 precomputed_data=None,
                                                 filter_type="identity")
    random_vector = np.random.rand(10)
    assert np.allclose(identity_filter.apply_filter(random_vector), random_vector)
    assert np.allclose(identity_filter.apply_chain_rule(random_vector), random_vector)

def test_polynomial_filter(regularmesh_and_polyplas_precomputed_data):
    polymesh, polyplas_precomputed_data = regularmesh_and_polyplas_precomputed_data
    _, largest_edge_length = polymesh.get_smallest_and_largest_edge_length()
    filter_radius = 0.95 * largest_edge_length
    polynomial_filter = get_design_variable_filter(polymesh_object=polymesh,
                                                   precomputed_data=polyplas_precomputed_data,
                                                   filter_type="polynomial",
                                                   filter_radius=filter_radius,
                                                   use_relative_filter_radius=False)
    constant_vector = 0.5 * np.ones(polyplas_precomputed_data.number_of_nodes)
    assert np.allclose(polynomial_filter.apply_filter(constant_vector), constant_vector)
    assert np.allclose(polynomial_filter.apply_chain_rule(constant_vector), constant_vector)

    filter_radius = 2.0 * largest_edge_length
    polynomial_filter = get_design_variable_filter(polymesh_object=polymesh,
                                                   precomputed_data=polyplas_precomputed_data,
                                                   filter_type="polynomial",
                                                   filter_radius=filter_radius,
                                                   use_relative_filter_radius=False)
    assert np.allclose(polynomial_filter.apply_filter(constant_vector), constant_vector)
    result = np.array([0.45557272, 0.45557272, 0.45557272, 0.45557272, 0.54442728, 0.54442728, 0.54442728, 0.54442728])
    assert np.allclose(polynomial_filter.apply_chain_rule(constant_vector), result)