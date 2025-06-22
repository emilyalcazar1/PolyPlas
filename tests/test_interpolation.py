import pytest
import numpy as np
import src.polytop.PolyInterpolation as polyinterp

def test_material_interpolation_factory():
    with pytest.raises(ValueError):
        polyinterp.get_material_interpolation_function(material_interpolation_function_name="not a function name")

def test_ersatz_parameter():
    with pytest.raises(ValueError):
        polyinterp.get_material_interpolation_function(ersatz_parameter=-1.0)
    with pytest.raises(ValueError):
        polyinterp.get_material_interpolation_function(ersatz_parameter=1.0)
    with pytest.raises(TypeError):
        polyinterp.get_material_interpolation_function(ersatz_parameter='not a number')

def test_penalization_parameter():
    with pytest.raises(ValueError):
        polyinterp.get_material_interpolation_function(penalization_parameter=-1.0)
    with pytest.raises(ValueError):
        polyinterp.get_material_interpolation_function(penalization_parameter=10.1)
    with pytest.raises(TypeError):
        polyinterp.get_material_interpolation_function(penalization_parameter='not a number')

def test_set_name_invalid_type():
    simp = polyinterp.get_material_interpolation_function(material_interpolation_function_name="SIMP",
                                                          penalization_parameter=2.0,
                                                          ersatz_parameter=0.125)
    with pytest.raises(TypeError):
        simp.name = 1.0 # not a string

def test_get_value():
    simp = polyinterp.get_material_interpolation_function(material_interpolation_function_name="SIMP",
                                                          penalization_parameter=2.0,
                                                          ersatz_parameter=0.125)
    density_vector = np.array([0.0, 0.5, 1.0])
    simp_values = simp.get_value(density_vector)
    assert isinstance(simp_values, np.ndarray)
    assert np.allclose(simp_values, np.array([0.125, 0.125 + 0.875*0.25, 1.0]))

    ramp = polyinterp.get_material_interpolation_function(material_interpolation_function_name="RAMP",
                                                          penalization_parameter=2.0,
                                                          ersatz_parameter=0.0625)
    ramp_values = ramp.get_value(density_vector)
    assert isinstance(ramp_values, np.ndarray)
    assert np.allclose(ramp_values, np.array([0.0625, 0.296875, 1.0]))

def test_get_derivative():
    simp = polyinterp.get_material_interpolation_function(material_interpolation_function_name="SIMP",
                                                          penalization_parameter=2.0,
                                                          ersatz_parameter=0.0625)
    density_vector = np.array([0.0, 0.5, 1.0])
    simp_derivatives = simp.get_derivative(density_vector)
    assert isinstance(simp_derivatives, np.ndarray)
    assert np.allclose(simp_derivatives, np.array([0.0,  0.9375, 1.875]))

    ramp = polyinterp.get_material_interpolation_function(material_interpolation_function_name="RAMP",
                                                          penalization_parameter=2.0,
                                                          ersatz_parameter=0.0625)
    ramp_derivatives = ramp.get_derivative(density_vector)
    assert isinstance(ramp_derivatives, np.ndarray)
    assert np.allclose(ramp_derivatives, np.array([0.3125, 0.703125, 2.8125]))
