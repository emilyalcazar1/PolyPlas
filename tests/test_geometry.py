import pytest
import numpy as np
import src.polymesher.PolyGeometryAndBCs as polygeometry
import src.polymesher.PolyMesher as polymesher

def test_intersection():
    distance_array_1 = np.array([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]])
    distance_array_2 = np.array([[2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0]])
    expected_result = np.zeros((2, 5))
    expected_result[:, 0:2] = distance_array_1[:, :2]
    expected_result[:, 2:4] = distance_array_2[:, :2]
    expected_result[:, 4] = np.array([4.0, 7.0])
    assert np.allclose(polygeometry.intersection(distance_array_1, distance_array_2), expected_result)

def test_union():
    distance_array_1 = np.array([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]])
    distance_array_2 = np.array([[2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0]])
    expected_result = np.zeros((2, 5))
    expected_result[:, 0:2] = distance_array_1[:, :2]
    expected_result[:, 2:4] = distance_array_2[:, :2]
    expected_result[:, 4] = np.array([3.0, 6.0])
    assert np.allclose(polygeometry.union(distance_array_1, distance_array_2), expected_result)

def test_subtract():
    distance_array_1 = np.array([[1.0, 2.0,  3.0],
                                 [4.0, 5.0,  6.0]])
    distance_array_2 = np.array([[2.0, 3.0,  4.0],
                                 [5.0, 6.0, -7.0]])
    expected_result = np.zeros((2, 5))
    expected_result[:, 0:2] = distance_array_1[:, :2]
    expected_result[:, 2:4] = distance_array_2[:, :2]
    expected_result[:, 4] = np.array([3.0, 7.0])
    assert np.allclose(polygeometry.subtract(distance_array_1, distance_array_2), expected_result)

@pytest.fixture(scope="function",
                params=[polygeometry.MbbBeam(applied_displacement_magnitude=2.0),
                        polygeometry.CantileverBeam(applied_displacement_magnitude=2.0),
                        polygeometry.HookDomain(applied_displacement_magnitude=2.0),
                        polygeometry.SuspensionDomain(applied_displacement_magnitude=2.0),
                        polygeometry.SerpentineDomain(applied_displacement_magnitude=2.0),
                        polygeometry.WrenchDomain(applied_displacement_magnitude=2.0),
                        polygeometry.MichellDomain(applied_displacement_magnitude=2.0),
                        polygeometry.CurvedBeam(applied_displacement_magnitude=2.0),
                        polygeometry.Lbracket(applied_displacement_magnitude=2.0),
                        polygeometry.Corbel(applied_displacement_magnitude=2.0),
                        polygeometry.CrackDomain(applied_displacement_magnitude=2.0),
                        polygeometry.PortalFrame(applied_displacement_magnitude=2.0),
                        polygeometry.EyeBar(applied_displacement_magnitude=2.0),
                        polygeometry.HalfPortalFrame(applied_displacement_magnitude=2.0),
                        polygeometry.CooksMembrane(applied_displacement_magnitude = 2.0),
                        polygeometry.UnitSquare(applied_displacement_magnitude = 2.0),
                        polygeometry.PlateWithHole(load_magnitude=2.0),
                        polygeometry.HalfClampedBeam(applied_displacement_magnitude = 2.0),
                        polygeometry.AntennaBracket(applied_displacement_magnitude=2.0)],
                ids=["MbbBeam", "CantileverBeam", "HookDomain", "SuspensionDomain", "SerpentineDomain",
                     "WrenchDomain", "MichellDomain", "CurvedBeam", "Lbracket", "Corbel", "CrackDomain",
                     "PortalFrame", "EyeBar", "HalfPortalFrame", "CooksMembrane", "UnitSquare", "PlateWithHole", 
                     "HalfClampedBeam", "AntennaBracket"])
def boundary_value_problem(request):
    yield request.param

def test_boundary_value_problem_load_magnitude(boundary_value_problem):
    assert boundary_value_problem.load_magnitude == 2.0
    load_magnitude = 4.0
    boundary_value_problem.load_magnitude = load_magnitude
    assert boundary_value_problem.load_magnitude == load_magnitude

def test_boundary_value_problem_invalid_load_magnitude(boundary_value_problem):
    with pytest.raises(ValueError):
        boundary_value_problem.load_magnitude = -1.0
    with pytest.raises(ValueError):
        boundary_value_problem.load_magnitude = 0.0
    with pytest.raises(TypeError):
        boundary_value_problem.load_magnitude = 'not a number'

def test_line():
    my_line = polygeometry.Line(x1=1.0, y1=1.0, x2=2.0, y2=2.0)
    assert my_line.x1 == 1.0
    assert my_line.y1 == 1.0
    assert my_line.x2 == 2.0
    assert my_line.y2 == 2.0

    expected_result = np.array([1.0, 2.0, 1.0, 2.0])
    assert np.allclose(my_line.get_bounding_box(), expected_result)

    point_coordinates = np.array([[-1.0, -1.0],
                                  [ 1.5,  1.5],
                                  [-1.0,  0.0],
                                  [ 0.0, -1.0]])
    expected_result = np.array([0.0, 0.0, -0.5**0.5, 0.5**0.5])
    distance = my_line.distance_function(point_coordinates)
    assert np.allclose(distance[:, -1], expected_result)

    with pytest.raises(TypeError):
        polygeometry.Line(x1='not a number', y1=1.0, x2=2.0, y2=2.0)
    with pytest.raises(TypeError):
        polygeometry.Line(x1=1.0, y1='not a number', x2=2.0, y2=2.0)
    with pytest.raises(TypeError):
        polygeometry.Line(x1=1.0, y1=1.0, x2='not a number', y2=2.0)
    with pytest.raises(TypeError):
        polygeometry.Line(x1=1.0, y1=1.0, x2=2.0, y2='not a number')

def test_rectangle():
    my_rectangle = polygeometry.Rectangle(x1=1.0, y1=1.0, x2=3.0, y2=2.0)
    assert my_rectangle.x1 == 1.0
    assert my_rectangle.y1 == 1.0
    assert my_rectangle.x2 == 3.0
    assert my_rectangle.y2 == 2.0

    expected_result = np.array([1.0, 3.0, 1.0, 2.0])
    assert np.allclose(my_rectangle.get_bounding_box(), expected_result)

    point_coordinates = np.array([[ 1.0,  2.0],
                                  [ 3.0,  1.0],
                                  [ 2.0,  1.5],
                                  [ 0.0,  0.0]])
    expected_result = np.array([0.0, 0.0, -0.5, 1.0])
    distance = my_rectangle.distance_function(point_coordinates)
    assert np.allclose(distance[:, -1], expected_result)

    with pytest.raises(TypeError):
        polygeometry.Rectangle(x1='not a number', y1=1.0, x2=2.0, y2=2.0)
    with pytest.raises(TypeError):
        polygeometry.Rectangle(x1=1.0, y1='not a number', x2=2.0, y2=2.0)
    with pytest.raises(TypeError):
        polygeometry.Rectangle(x1=1.0, y1=1.0, x2='not a number', y2=2.0)
    with pytest.raises(TypeError):
        polygeometry.Rectangle(x1=1.0, y1=1.0, x2=2.0, y2='not a number')
    with pytest.raises(ValueError):
        polygeometry.Rectangle(x1=1.0, y1=1.0, x2=1.0, y2=2.0)
    with pytest.raises(ValueError):
        polygeometry.Rectangle(x1=1.0, y1=1.0, x2=2.0, y2=1.0)

def test_circle():
    my_circle = polygeometry.Circle(center=[1.0, 1.0], radius=2.0)
    assert my_circle.center_x_coordinate == 1.0
    assert my_circle.center_y_coordinate == 1.0
    assert my_circle.radius == 2.0

    expected_result = np.array([-1.0, 3.0, -1.0, 3.0])
    assert np.allclose(my_circle.get_bounding_box(), expected_result)

    point_coordinates = np.array([[ 1.0,  1.0],
                                  [-1.0,  1.0],
                                  [-2.0,  1.0],
                                  [ 1.0,  4.0]])
    expected_result = np.array([-2.0, 0.0, 1.0, 1.0])
    distance = my_circle.distance_function(point_coordinates)
    assert np.allclose(distance[:, -1], expected_result)

    with pytest.raises(TypeError):
        polygeometry.Circle(center='not a list', radius=1.0)
    with pytest.raises(ValueError):
        polygeometry.Circle(center=(1.0, 1.0, 1.0), radius=1.0)
    with pytest.raises(TypeError):
        polygeometry.Circle(center=('not a number', 1.0), radius=1.0)
    with pytest.raises(TypeError):
        polygeometry.Circle(center=(1.0, 'not a number'), radius=1.0)
    with pytest.raises(TypeError):
        polygeometry.Circle(center=(1.0, 1.0), radius='not a number')
    with pytest.raises(ValueError):
        polygeometry.Circle(center=(1.0, 1.0), radius=-1.0)

def test_curved_beam():
    bvp = polygeometry.CurvedBeam(center=(1.0, 0.0), inner_radius=1.0, outer_radius=3.0)
    expected_result = np.array([-2.0, 4.0, 0.0, 3.0])
    assert np.allclose(bvp.get_bounding_box(), expected_result)

    point_coordinates = np.array([[ 1.0,  0.0],
                                  [ 0.0,  0.0],
                                  [ 1.0,  4.0],
                                  [ 1.0,  3.0]])
    expected_result = np.array([1.0, 0.0, 1.0, 0.0])
    distance = bvp.distance_function(point_coordinates)
    assert np.allclose(distance[:, -1], expected_result)

    with pytest.raises(ValueError):
        polygeometry.CurvedBeam(center=(1.0, 0.0), inner_radius=1.0, outer_radius=1.0)

def test_boundary_conditions(boundary_value_problem):
    is_suspension_domain = isinstance(boundary_value_problem, polygeometry.SuspensionDomain)
    polymesh_object = polymesher.PolyMesher(
        boundary_value_problem,
        number_of_elements=100,
        maximum_iterations=20
    )
    nodal_coordinates, _, boundary_conditions = \
        polymesh_object.get_mesh_and_boundary_conditions()
    assert isinstance(boundary_conditions, polygeometry.BoundaryConditions)

    fixed_x_nodal_indices = boundary_conditions.fixed_x_nodal_indices
    assert isinstance(fixed_x_nodal_indices, np.ndarray)
    my_distance = boundary_value_problem.distance_function(nodal_coordinates[fixed_x_nodal_indices, :])
    bounding_box = boundary_value_problem.get_bounding_box()
    bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
    number_of_nodes = float(nodal_coordinates.shape[0])
    nodal_tolerance = 0.25 * (bounding_box_area**0.5) / (number_of_nodes**0.5)
    if is_suspension_domain:
        assert np.all(my_distance[:, -1] < nodal_tolerance)
    else:
        assert np.allclose(my_distance[:, -1], 0.0, atol=nodal_tolerance)

    fixed_y_nodal_indices = boundary_conditions.fixed_y_nodal_indices
    assert isinstance(fixed_y_nodal_indices, np.ndarray)
    my_distance = boundary_value_problem.distance_function(nodal_coordinates[fixed_y_nodal_indices, :])
    if is_suspension_domain:
        assert np.all(my_distance[:, -1] < nodal_tolerance)
    else:
        assert np.allclose(my_distance[:, -1], 0.0, atol=nodal_tolerance)

    assert isinstance(boundary_conditions.applied_loads, list)
    for applied_load in boundary_conditions.applied_loads:
        assert isinstance(applied_load, polygeometry.AppliedLoad)
        assert isinstance(applied_load.nodal_indices, np.ndarray)
        assert isinstance(applied_load.load_components, np.ndarray)
        assert applied_load.load_components.ndim == 2
        assert applied_load.load_components.shape[1] == 2
        loaded_nodal_coordinates = nodal_coordinates[applied_load.nodal_indices, :]
        my_distance = boundary_value_problem.distance_function(loaded_nodal_coordinates)
        if is_suspension_domain:
            assert np.all(my_distance[:, -1] < nodal_tolerance)
        else:
            assert np.allclose(my_distance[:, -1], 0.0, atol=nodal_tolerance)

def test_regular_grid_generation(boundary_value_problem):
    is_regular_grid_implemented = boundary_value_problem.is_regular_grid_implemented()
    assert isinstance(is_regular_grid_implemented, bool)
    if is_regular_grid_implemented:
        polymesh_object = polymesher.PolyMesher(
            boundary_value_problem,
            number_of_elements=500,
            maximum_iterations=30,
            use_regular_grid_if_implemented=True
        )
        _, element_connectivity_arrays, _ = polymesh_object.get_mesh_and_boundary_conditions()
        assert isinstance(element_connectivity_arrays, dict)
        assert len(element_connectivity_arrays) == 1
    else:
        with pytest.raises(NotImplementedError):
             boundary_value_problem.get_regular_grid_element_centroids(100)
