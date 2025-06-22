import pytest
import numpy as np
import src.polytop.PolyProjection as polyprojection

def test_init_with_valid_parameters():
    projection_strength = 0.5
    projection_threshold = 0.2
    projection = polyprojection.get_projection_function(projection_function_type="tanh",
                                                        projection_strength=projection_strength,
                                                        projection_threshold=projection_threshold)
    assert projection.projection_strength == projection_strength
    assert projection.projection_threshold == projection_threshold
    assert projection.name == "Tanh Heaviside"

def test_init_with_invalid_projection_strength_type():
    with pytest.raises(TypeError):
        polyprojection.get_projection_function(projection_function_type="tanh",
                                               projection_strength="not a number",
                                               projection_threshold=0.5)

def test_init_with_negative_projection_strength():
    with pytest.raises(ValueError):
        polyprojection.get_projection_function(projection_function_type="tanh",
                                               projection_strength=-1.0,
                                               projection_threshold=0.5)

def test_init_with_invalid_projection_threshold_type():
    with pytest.raises(TypeError):
        polyprojection.get_projection_function(projection_function_type="tanh",
                                               projection_strength=1.0,
                                               projection_threshold="not a number")

def test_init_with_negative_projection_threshold():
    with pytest.raises(ValueError):
        polyprojection.get_projection_function(projection_function_type="tanh",
                                               projection_strength=1.0,
                                               projection_threshold=-1.0)

def test_invalid_name_type():
    projection = polyprojection.get_projection_function(projection_function_type="tanh",
                                                        projection_strength=1.0,
                                                        projection_threshold=0.5)
    with pytest.raises(TypeError):
        projection.name = 1.0

def test_init_with_invalid_projection_function_name():
    with pytest.raises(ValueError):
        polyprojection.get_projection_function(projection_function_type="not a valid name",
                                               projection_strength=1.0,
                                               projection_threshold=0.5)

def test_update_logged_values():
    projection_strength = 0.5
    projection_threshold = 0.2
    projection = polyprojection.get_projection_function(projection_function_type="tanh",
                                                        projection_strength=projection_strength,
                                                        projection_threshold=projection_threshold)
    logged_values = {}
    updated_logged_values = projection.update_logged_values(logged_values)
    assert updated_logged_values == {f"Tanh Heaviside Projection Strength": projection_strength,
                                     f"Tanh Heaviside Projection Threshold": projection_threshold}

def test_tanh_projection():
    projection_strength = 2.0
    projection_threshold = 0.25
    projection = polyprojection.get_projection_function(projection_function_type="tanh",
                                                        projection_strength=projection_strength,
                                                        projection_threshold=projection_threshold)
    filtered_values = np.linspace(0.0, 1.0, 10)
    projected_values = projection.apply_projection(filtered_values)
    c1 = np.tanh(projection_strength * projection_threshold)
    denominator = c1 + np.tanh(projection_strength * (1.0 - projection_threshold))
    numerator   = c1 + np.tanh(projection_strength * (filtered_values - projection_threshold))
    expected_projected_values = numerator / denominator
    assert np.allclose(projected_values, expected_projected_values)

    projected_derivatives = projection.apply_chain_rule(np.ones_like(filtered_values))
    new_denominator = denominator * np.cosh(projection_strength * (filtered_values - projection_threshold))**2
    expected_projected_derivatives = projection_strength / new_denominator
    assert np.allclose(projected_derivatives, expected_projected_derivatives)

def test_identity_projection():
    projection = polyprojection.get_projection_function(projection_function_type="identity",
                                                        projection_strength=1.0,
                                                        projection_threshold=0.5)
    filtered_values = np.linspace(0.0, 1.0, 10)
    projected_values = projection.apply_projection(filtered_values)
    assert np.allclose(projected_values, filtered_values)

    projected_derivatives = projection.apply_chain_rule(filtered_values)
    assert np.allclose(projected_derivatives, filtered_values)
