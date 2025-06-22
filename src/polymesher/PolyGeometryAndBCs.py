import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import NamedTuple, List, Tuple, Dict


#######################################################################################################################
#######################################################################################################################
def intersection(distance_array_1: np.ndarray, distance_array_2: np.ndarray) -> np.ndarray:
    return np.column_stack((distance_array_1[:, :-1], distance_array_2[:, :-1],
                            np.maximum(distance_array_1[:, -1], distance_array_2[:, -1])))

def union(distance_array_1: np.ndarray, distance_array_2: np.ndarray) -> np.ndarray:
    return np.column_stack((distance_array_1[:, :-1], distance_array_2[:, :-1],
                            np.minimum(distance_array_1[:, -1], distance_array_2[:, -1])))

def subtract(distance_array_1: np.ndarray, distance_array_2: np.ndarray) -> np.ndarray:
    return np.column_stack((distance_array_1[:, :-1], distance_array_2[:, :-1],
                            np.maximum(distance_array_1[:, -1], -1.0 * distance_array_2[:, -1])))


#######################################################################################################################
#######################################################################################################################
class GeometricEntity(ABC):
    '''Abstract base class for geometry definition.'''
    @abstractmethod
    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def get_bounding_box(self) -> np.ndarray:
        pass


#######################################################################################################################
#######################################################################################################################
class AppliedDisplacement(NamedTuple):
    '''Tuple for defining an applied load.'''
    nodal_indices: np.ndarray
    applied_displacement_components: np.ndarray


#######################################################################################################################
#######################################################################################################################
class BoundaryConditions(NamedTuple):
    '''Tuple for defining boundary conditions including fixed node indices and a list of applied loads.'''
    fixed_x_nodal_indices: np.ndarray
    fixed_y_nodal_indices: np.ndarray
    applied_displacements: List[AppliedDisplacement]


#######################################################################################################################
#######################################################################################################################
class BoundaryValueProblem(GeometricEntity):
    '''Abstract base class for defining boundary value problems.

    The GeometricEntity methods must be implemented by the child class in addition to
    a get_boundary_conditions method.

    The boundary conditions are defined by the following:
        1. An array of nodal indices that are fixed in the x-direction.
        2. An array of nodal indices that are fixed in the y-direction.
        3. A list of applied loads. Each applied load is defined by:
            a. An array of nodal indices that the load is applied to.
            b. A tuple of the x and y components of the load.
    '''
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        self.applied_displacement_magnitude = applied_displacement_magnitude

    @property
    def applied_displacement_magnitude(self) -> float:
        return self._applied_displacement_magnitude

    @applied_displacement_magnitude.setter
    def applied_displacement_magnitude(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError(f"The load magnitude must be a number. User supplied '{value}' of type '{type(value)}'.")
        if value == 0.0:
            raise ValueError(f"The load magnitude must be non-zero. User supplied '{value}'.")
        if value < 0.0:
            raise ValueError(f"The load magnitude must be positive. User supplied '{value}'.")
        self._applied_displacement_magnitude = value

    @abstractmethod
    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        pass

    def is_regular_grid_implemented(self) -> bool:
        return False

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        raise NotImplementedError("The get_regular_grid_element_centroids method has not been implemented.")

    def get_passive_region_indices(self,
                                   nodal_coordinates: np.ndarray,
                                   element_connectivity_arrays: Dict[int, np.ndarray],
                                   largest_element_edge_length: float
                                   ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        passive_design_variable_indices = np.array([], dtype=int)
        passive_local_element_indices = defaultdict(lambda: np.array([], dtype=int))
        return passive_design_variable_indices, passive_local_element_indices


#######################################################################################################################
#######################################################################################################################
class Line(GeometricEntity):
    '''Class for defining a line segment in 2D space from (x1, y1) to (x2, y2).'''

    def __init__(self, x1: float = 0.0, x2: float = 1.0, y1: float = 0.0, y2: float = 1.0):
        if not isinstance(x1, (float, int)):
            raise TypeError(f"The 'x1' parameter must be a number. User specified x1 = '{x1}'")
        if not isinstance(x2, (float, int)):
            raise TypeError(f"The 'x2' parameter must be a number. User specified x2 = '{x2}'")
        if not isinstance(y1, (float, int)):
            raise TypeError(f"The 'y1' parameter must be a number. User specified y1 = '{y1}'")
        if not isinstance(y2, (float, int)):
            raise TypeError(f"The 'y2' parameter must be a number. User specified y2 = '{y2}'")
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)
        self.line_unit_vector = np.array([self.x2 - self.x1, self.y2 - self.y1], dtype=float)
        self.line_unit_vector /= np.linalg.norm(self.line_unit_vector)

    def get_bounding_box(self) -> np.ndarray:
        return np.array([self.x1, self.x2, self.y1, self.y2])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        my_distance = (point_coordinates[:, 0] - self.x1) * self.line_unit_vector[1] - \
                      (point_coordinates[:, 1] - self.y1) * self.line_unit_vector[0]
        return np.column_stack((my_distance, my_distance))


#######################################################################################################################
#######################################################################################################################
class Circle(GeometricEntity):
    '''Class for defining a circle in 2D space at specified center, with specified radius.'''

    def __init__(self, center: Tuple[float, float] = (0.0, 0.0), radius: float = 1.0):
        if not isinstance(radius, (float, int)):
            raise TypeError(f"The 'radius' parameter must be a number. User specified radius = '{radius}'")
        if radius <= 0.0:
            raise ValueError(f"The 'radius' parameter must be positive. User specified radius = '{radius}'")
        self.radius = float(radius)

        if not isinstance(center, (list, tuple, np.ndarray)):
            raise TypeError("The 'center' parameter must be a list, tuple, or array. "
                            f"User specified center = '{center}' of type '{type(center)}'")
        if len(center) != 2:
            raise ValueError("The 'center' parameter must be a list or tuple of length 2. User specified "
                             f"center = '{center}' of length {len(center)}")
        if not isinstance(center[0], (float, int)):
            raise TypeError(f"The 'center[0]' parameter must be a number. User specified center[0] = '{center[0]}'")
        if not isinstance(center[1], (float, int)):
            raise TypeError(f"The 'center[1]' parameter must be a number. User specified center[1] = '{center[1]}'")
        self.center_x_coordinate, self.center_y_coordinate = center

    def get_bounding_box(self) -> np.ndarray:
        return np.array([self.center_x_coordinate - self.radius,
                         self.center_x_coordinate + self.radius,
                         self.center_y_coordinate - self.radius,
                         self.center_y_coordinate + self.radius])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        my_distance = np.sqrt((point_coordinates[:, 0] - self.center_x_coordinate)**2 +
                              (point_coordinates[:, 1] - self.center_y_coordinate)**2) - self.radius
        return np.column_stack((my_distance, my_distance))


#######################################################################################################################
#######################################################################################################################
class Rectangle(GeometricEntity):
    '''Class for defining a rectangle in 2D space from (x1, y1) to (x2, y2)'''

    def __init__(self, x1: float = 0.0, x2: float = 3.0, y1: float = 0.0, y2: float = 1.0):
        if not isinstance(x1, (float, int)):
            raise TypeError(f"The 'x1' parameter must be a number. User specified x1 = '{x1}'")
        if not isinstance(x2, (float, int)):
            raise TypeError(f"The 'x2' parameter must be a number. User specified x2 = '{x2}'")
        if not isinstance(y1, (float, int)):
            raise TypeError(f"The 'y1' parameter must be a number. User specified y1 = '{y1}'")
        if not isinstance(y2, (float, int)):
            raise TypeError(f"The 'y2' parameter must be a number. User specified y2 = '{y2}'")
        if x1 >= x2:
            raise ValueError("The 'x1' parameter must be less than the 'x2' parameter. "
                             f"User specified x1 = '{x1}' and x2 = '{x2}'")
        if y1 >= y2:
            raise ValueError("The 'y1' parameter must be less than the 'y2' parameter. "
                             f"User specified y1 = '{y1}' and y2 = '{y2}'")
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)

    def get_bounding_box(self) -> np.ndarray:
        return np.array([self.x1, self.x2, self.y1, self.y2])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        my_shape = (point_coordinates.shape[0], 1)
        x_points = point_coordinates[:, 0].reshape(my_shape)
        y_points = point_coordinates[:, 1].reshape(my_shape)
        d_x1 = self.x1 - x_points
        d_x2 = x_points - self.x2
        d_y1 = self.y1 - y_points
        d_y2 = y_points - self.y2
        temp = np.column_stack((d_x1, d_x2, d_y1, d_y2))
        my_maximum = np.amax(temp, axis=1).ravel()
        return np.column_stack((d_x1, d_x2, d_y1, d_y2, my_maximum))


#######################################################################################################################
#######################################################################################################################
class Polygon(GeometricEntity):
    '''Class for defining a 2D convex polygon from a list of vertex coordinates'''

    def __init__(self, vertex_coordinates: np.ndarray):
        if not isinstance(vertex_coordinates, np.ndarray):
            raise TypeError("The 'vertex_coordinates' parameter must be a numpy array. "
                            f"User specified vertex_coordinates = '{vertex_coordinates}' "
                            f"of type '{type(vertex_coordinates)}'")
        if len(vertex_coordinates.shape) != 2:
            raise ValueError("The 'vertex_coordinates' parameter must be a 2D numpy array. "
                             f"User specified vertex_coordinates = '{vertex_coordinates}' "
                             f"of shape {vertex_coordinates.shape}")
        if vertex_coordinates.shape[1] != 2:
            raise ValueError("The 'vertex_coordinates' parameter must be a 2D numpy array with 2 columns. "
                             f"User specified vertex_coordinates = '{vertex_coordinates}' "
                             f"of shape {vertex_coordinates.shape}")
        vertex_coordinates = np.row_stack((vertex_coordinates, vertex_coordinates[0, :]))
        self.line_list = []
        number_of_lines = vertex_coordinates.shape[0] - 1
        for line_index in range(number_of_lines):
            x1, y1 = vertex_coordinates[line_index, :]
            x2, y2 = vertex_coordinates[line_index + 1, :]
            self.line_list.append(Line(x1=x1, x2=x2, y1=y1, y2=y2))
        minimum_coordinates = np.amin(vertex_coordinates, axis=0)
        maximum_coordinates = np.amax(vertex_coordinates, axis=0)
        self.bounding_box = np.array([minimum_coordinates[0], maximum_coordinates[0],
                                      minimum_coordinates[1], maximum_coordinates[1]])

    def get_bounding_box(self) -> np.ndarray:
        return self.bounding_box

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        my_distance = self.line_list[0].distance_function(point_coordinates)
        for line in self.line_list[1:]:
            my_distance = intersection(my_distance, line.distance_function(point_coordinates))
        return my_distance

#######################################################################################################################
#######################################################################################################################
class UnitSquare(BoundaryValueProblem):
    '''
    Class for defining a simple unit square domain for testing purposes.
    '''
    def __init__(self, x1: float = -0.5,
                 x2: float = 0.5,
                 y1: float = -0.5,
                 y2: float = 0.5,
                 applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.rectangle = Rectangle(x1=x1, x2=x2, y1=y1, y2=y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    def get_bounding_box(self) -> np.ndarray:
        return self.rectangle.get_bounding_box()
    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return self.rectangle.distance_function(point_coordinates)
    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)
        left_edge = np.argwhere(x_coordinates < (-0.5 + nodal_tolerance)).ravel()
        right_edge = np.argwhere(x_coordinates > (0.5 - nodal_tolerance)).ravel()
        node_upper_right_edge = np.argwhere((x_coordinates > (0.5 - nodal_tolerance)).ravel() & (y_coordinates > (0.5 - nodal_tolerance)).ravel())
        bottom_edge = np.argwhere(y_coordinates < (-0.5 + nodal_tolerance)).ravel()
        top_edge = np.argwhere(y_coordinates > (0.5 - nodal_tolerance)).ravel()
        number_of_loaded_nodes = right_edge.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 0] = self.applied_displacement_magnitude #applied_displacement_magnitude in x direction
        applied_displacement_1 = AppliedDisplacement(nodal_indices=right_edge,
                                                  applied_displacement_components=applied_displacement_components)    
           
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=left_edge,
                                                              fixed_y_nodal_indices=bottom_edge,
                                                              applied_displacements=[applied_displacement_1])
        return boundary_conditions
    
    def is_regular_grid_implemented(self) -> bool:
        return True

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        x_length = self.x2 - self.x1
        y_length = self.y2 - self.y1
        x_is_larger_than_y = x_length >= y_length
        N = approximate_number_of_elements
        L = x_length if x_is_larger_than_y else y_length
        H = y_length if x_is_larger_than_y else x_length
        NL_float = (H - L + (H**2 + L**2 - 2.0*H*L + 4.0*H*L*N)**0.5)/(2.0*H)
        NH_float = N / NL_float
        NL = int(np.ceil(NL_float))
        NH = int(np.ceil(NH_float))
        number_of_x_points = NL + 1 if x_is_larger_than_y else NH + 1
        number_of_y_points = NH + 1 if x_is_larger_than_y else NL + 1
        x = np.linspace(self.x1, self.x2, number_of_x_points)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(self.y1, self.y2, number_of_y_points)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
        return regular_grid_element_centroids

#######################################################################################################################
#######################################################################################################################
class CurvedBeam(BoundaryValueProblem):
    '''
    Class for defining a simple curved beam domain with the lower left edge fixed and
    a load applied at 45-degrees to the lower right edge.
    '''
    def __init__(self,
                 center: Tuple[float, float] = (0.0, 0.0),
                 inner_radius: float = 10.0,
                 outer_radius: float = 20.0,
                 applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        if outer_radius <= inner_radius:
            raise ValueError(f"The 'outer_radius' parameter must be larger than the 'inner_radius' parameter. "
                             f"User specified outer_radius = '{outer_radius}' and inner_radius = '{inner_radius}'")
        self.outer_circle = Circle(center=center, radius=outer_radius)
        self.inner_circle = Circle(center=center, radius=inner_radius)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.center_x_coordinate, self.center_y_coordinate = center
        eps = 1.0e-10
        c = 1.0 + eps
        x1 = self.center_x_coordinate - c * self.outer_radius
        x2 = self.center_x_coordinate + c * self.outer_radius
        y1 = self.center_y_coordinate - c * self.outer_radius
        y2 = self.center_y_coordinate + eps
        self.bottom_rectangle = Rectangle(x1=x1, x2=x2, y1=y1, y2=y2)

    def get_bounding_box(self) -> np.ndarray:
        return np.array([self.center_x_coordinate - self.outer_radius,
                         self.center_x_coordinate + self.outer_radius,
                         self.center_y_coordinate,
                         self.center_y_coordinate + self.outer_radius])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return subtract(subtract(self.outer_circle.distance_function(point_coordinates),
                                 self.bottom_rectangle.distance_function(point_coordinates)),
                        self.inner_circle.distance_function(point_coordinates))

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        displacement_region_width = 0.25 * (self.outer_radius - self.inner_radius)
        displacement_region_center_x_coordinate = self.center_x_coordinate + self.inner_radius + \
            0.5 * (self.outer_radius - self.inner_radius)
        displacement_region_left_x_coordinate  = displacement_region_center_x_coordinate - 0.5 * displacement_region_width - nodal_tolerance
        displacement_region_right_x_coordinate = displacement_region_center_x_coordinate + 0.5 * displacement_region_width + nodal_tolerance

        bottom_edge_mask = y_coordinates < (self.center_y_coordinate + nodal_tolerance)
        left_half_mask = x_coordinates < self.center_x_coordinate
        displacement_region_x_mask = (x_coordinates > displacement_region_left_x_coordinate) & \
                             (x_coordinates < displacement_region_right_x_coordinate)

        center_of_bottom_right_edge_node_indices = np.argwhere(bottom_edge_mask & displacement_region_x_mask).ravel()
        bottom_left_edge_node_indices = np.argwhere(bottom_edge_mask & left_half_mask).ravel()

        number_of_loaded_nodes = center_of_bottom_right_edge_node_indices.size
        while number_of_loaded_nodes == 0:
            displacement_region_left_x_coordinate -= nodal_tolerance
            displacement_region_right_x_coordinate += nodal_tolerance
            displacement_region_x_mask = (x_coordinates > displacement_region_left_x_coordinate) & \
                                 (x_coordinates < displacement_region_right_x_coordinate)
            center_of_bottom_right_edge_node_indices = np.argwhere(bottom_edge_mask & displacement_region_x_mask).ravel()
            number_of_loaded_nodes = center_of_bottom_right_edge_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 0] = self.applied_displacement_magnitude 
        applied_displacement_1 = AppliedDisplacement(nodal_indices=center_of_bottom_right_edge_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=bottom_left_edge_node_indices,
                                                 fixed_y_nodal_indices=bottom_left_edge_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

#######################################################################################################################
#######################################################################################################################
class CooksMembrane(BoundaryValueProblem):
    '''
    Class for the shape of the CooksMembrane domain.
    '''
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        polygon_vertex_list_1 = np.array([[0.0, 0.0],
                                          [48.0, 44.0],
                                          [48.0, 60.0],
                                          [0.0, 44.0]])
                                          
        self.polygon_1 = Polygon(vertex_coordinates=polygon_vertex_list_1)

    def get_bounding_box(self) -> np.ndarray:
        return self.polygon_1.get_bounding_box()
    
    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return self.polygon_1.distance_function(point_coordinates)
    
    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)
        left_edge_mask = x_coordinates < (bounding_box[0] + nodal_tolerance)
        right_edge_mask = x_coordinates > (bounding_box[1] - nodal_tolerance)
        support_node_indices = np.argwhere(left_edge_mask).ravel()
        applied_displacement_node_indices = np.argwhere(right_edge_mask).ravel()

        number_of_applied_displacement_nodes = applied_displacement_node_indices.size
        aplied_displacement_components = np.zeros((number_of_applied_displacement_nodes, 2), dtype=float)
        aplied_displacement_components[:, 1] = self.applied_displacement_magnitude #applied displacement in the y direction

        applied_load_1 = AppliedDisplacement(nodal_indices=applied_displacement_node_indices,
                                     applied_displacement_components=aplied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=support_node_indices,
                                                 fixed_y_nodal_indices=support_node_indices,
                                                 applied_displacements=[applied_load_1])
        
        return boundary_conditions

#######################################################################################################################
#######################################################################################################################
class HalfPortalFrame(BoundaryValueProblem):
    '''
    Class for the Half Portal Frame domain.
    '''
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        polygon_vertex_list_1 = np.array([[0.0, 0.0],
                                          [2.75, 0.0],
                                          [30.0, 17.5],
                                          [30.0, 30.0],
                                          [0.0, 30.0]])
                                          
        self.polygon_1 = Polygon(vertex_coordinates=polygon_vertex_list_1)

    def get_bounding_box(self) -> np.ndarray:
        return self.polygon_1.get_bounding_box()
    
    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return self.polygon_1.distance_function(point_coordinates)
    
    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)
        left_edge_mask = x_coordinates < nodal_tolerance
        bottom_edge_mask = y_coordinates < nodal_tolerance
        top_edge_mask = y_coordinates > (30.0 - nodal_tolerance)
        right_edge_mask = x_coordinates > (bounding_box[1] - nodal_tolerance)
        support_node_indices = np.argwhere(left_edge_mask).ravel()
        fixed_x_node_indices = np.argwhere(right_edge_mask).ravel() 
        fixed_y_node_indices = bottom_edge_mask
        top_edge_mask = y_coordinates > (30.0 - nodal_tolerance)
        center_mask = np.abs(x_coordinates) > (28.0 + nodal_tolerance)
        loaded_node_indices = np.argwhere(top_edge_mask & center_mask).ravel()
        
        number_of_loaded_nodes = loaded_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=loaded_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=fixed_x_node_indices,
                                                 fixed_y_nodal_indices=fixed_y_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        
        return boundary_conditions












#######################################################################################################################
#######################################################################################################################
class MbbBeam(BoundaryValueProblem):
    '''
    Class for defining a simple MBB beam domain with the left edge fixed in the x direction (symmetry) and the
    bottom right vertex fixed in the y direction. The load is applied to the node at the top left corner of the beam.
    '''
    def __init__(self,
                 x1: float = 0.0,
                 x2: float = 3.0,
                 y1: float = 0.0,
                 y2: float = 1.0,
                 applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.rectangle = Rectangle(x1=x1, x2=x2, y1=y1, y2=y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get_bounding_box(self) -> np.ndarray:
        return self.rectangle.get_bounding_box()

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return self.rectangle.distance_function(point_coordinates)

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        load_region_thickness = 0.04 * self.x2
        loaded_node_indices = np.argwhere((x_coordinates < (self.x1 + nodal_tolerance + load_region_thickness)) &
                                          (y_coordinates > (self.y2 - nodal_tolerance))).ravel()
        symmetry_boundary_node_indices = np.argwhere(x_coordinates < (self.x1 + nodal_tolerance)).ravel()
        support_node_indices = np.argwhere((x_coordinates > (self.x2 - nodal_tolerance)) &
                                           (y_coordinates < (self.y1 + nodal_tolerance))).ravel()

        number_of_loaded_nodes = loaded_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude #y-direction

        applied_displacement_1 = AppliedDisplacement(nodal_indices=loaded_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=symmetry_boundary_node_indices,
                                                 fixed_y_nodal_indices=support_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

    def is_regular_grid_implemented(self) -> bool:
        return True

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        x_length = self.x2 - self.x1
        y_length = self.y2 - self.y1
        x_is_larger_than_y = x_length >= y_length
        N = approximate_number_of_elements
        L = x_length if x_is_larger_than_y else y_length
        H = y_length if x_is_larger_than_y else x_length
        NL_float = (H - L + (H**2 + L**2 - 2.0*H*L + 4.0*H*L*N)**0.5)/(2.0*H)
        NH_float = N / NL_float
        NL = int(np.ceil(NL_float))
        NH = int(np.ceil(NH_float))
        number_of_x_points = NL + 1 if x_is_larger_than_y else NH + 1
        number_of_y_points = NH + 1 if x_is_larger_than_y else NL + 1
        x = np.linspace(self.x1, self.x2, number_of_x_points)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(self.y1, self.y2, number_of_y_points)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
        return regular_grid_element_centroids


#######################################################################################################################
#######################################################################################################################
class HalfClampedBeam(BoundaryValueProblem):
    '''
    Class for defining the Half Clamped Beam domain.
    '''
    def __init__(self,
                 x1: float = 0.0,
                 x2: float = 2000.0,
                 y1: float = 0.0,
                 y2: float = 1000.0,
                 applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.rectangle = Rectangle(x1=x1, x2=x2, y1=y1, y2=y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get_bounding_box(self) -> np.ndarray:
        return self.rectangle.get_bounding_box()

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return self.rectangle.distance_function(point_coordinates)

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        y_average = 0.5 * (self.y1 + self.y2)
        scale = 1.0
        mask = (x_coordinates > (self.x2 - nodal_tolerance)) & \
               (y_coordinates > (y_average - scale * nodal_tolerance)) & \
               (y_coordinates < (y_average + scale * nodal_tolerance))
        while not np.any(mask):
            mask = (x_coordinates > (self.x2 - nodal_tolerance)) & \
                   (y_coordinates > (y_average - scale * nodal_tolerance)) & \
                   (y_coordinates < (y_average + scale * nodal_tolerance))
            scale *= 10.0
        loaded_node_indices = np.argwhere(mask).ravel()
        mask_1_x = x_coordinates > (self.x2 - nodal_tolerance)
        mask_2_x = x_coordinates < (self.x1 + nodal_tolerance)
        fixed_x_node_indices = np.argwhere(mask_1_x | mask_2_x).ravel()
        # fixed_x_node_indices = np.argwhere(x_coordinates > (self.x2 - nodal_tolerance)).ravel()
        fixed_y_node_indices = np.argwhere(x_coordinates < (self.x1 + nodal_tolerance)).ravel()

        right_edge_mask = x_coordinates > (bounding_box[1] - nodal_tolerance)
        loaded_node_indices = np.argwhere((x_coordinates > (self.x2 - self.x2/10 - nodal_tolerance)) &
                                          (y_coordinates > (self.y2 - nodal_tolerance))).ravel()
        number_of_loaded_nodes = loaded_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude #y direction

        applied_displacement_1 = AppliedDisplacement(nodal_indices=loaded_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=fixed_x_node_indices,
                                                 fixed_y_nodal_indices=fixed_y_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

    def is_regular_grid_implemented(self) -> bool:
        return True

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        x_length = self.x2 - self.x1
        y_length = self.y2 - self.y1
        x_is_larger_than_y = x_length >= y_length
        N = approximate_number_of_elements
        L = x_length if x_is_larger_than_y else y_length
        H = y_length if x_is_larger_than_y else x_length
        NL_float = (H - L + (H**2 + L**2 - 2.0*H*L + 4.0*H*L*N)**0.5)/(2.0*H)
        NH_float = N / NL_float
        NL = int(np.ceil(NL_float))
        NH = int(np.ceil(NH_float))
        number_of_x_points = NL + 1 if x_is_larger_than_y else NH + 1
        number_of_y_points = NH + 1 if x_is_larger_than_y else NL + 1
        x = np.linspace(self.x1, self.x2, number_of_x_points)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(self.y1, self.y2, number_of_y_points)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
        return regular_grid_element_centroids


#######################################################################################################################
#######################################################################################################################
class CantileverBeam(BoundaryValueProblem):
    '''
    Class for defining a simple cantilever beam domain with the left edge fixed in the x and y directions and
    an applied load to the center of the right edge.
    '''
    def __init__(self,
                 x1: float = 0.0,
                 x2: float = 3.0,
                 y1: float = 0.0,
                 y2: float = 1.0,
                 applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.rectangle = Rectangle(x1=x1, x2=x2, y1=y1, y2=y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get_bounding_box(self) -> np.ndarray:
        return self.rectangle.get_bounding_box()

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return self.rectangle.distance_function(point_coordinates)

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        y_average = 0.5 * (self.y1 + self.y2)
        scale = 1.0
        mask = (x_coordinates > (self.x2 - nodal_tolerance)) & \
               (y_coordinates > (y_average - scale * nodal_tolerance)) & \
               (y_coordinates < (y_average + scale * nodal_tolerance))
        while not np.any(mask):
            mask = (x_coordinates > (self.x2 - nodal_tolerance)) & \
                   (y_coordinates > (y_average - scale * nodal_tolerance)) & \
                   (y_coordinates < (y_average + scale * nodal_tolerance))
            scale *= 10.0
        loaded_node_indices = np.argwhere(mask).ravel()
        fixed_boundary_node_indices = np.argwhere(x_coordinates < (self.x1 + nodal_tolerance)).ravel()

        right_edge_mask = x_coordinates > (bounding_box[1] - nodal_tolerance)
        loaded_node_indices = np.argwhere(right_edge_mask).ravel()
        number_of_loaded_nodes = loaded_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude #y direction

        applied_displacement_1 = AppliedDisplacement(nodal_indices=loaded_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=fixed_boundary_node_indices,
                                                 fixed_y_nodal_indices=fixed_boundary_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

    def is_regular_grid_implemented(self) -> bool:
        return True

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        x_length = self.x2 - self.x1
        y_length = self.y2 - self.y1
        x_is_larger_than_y = x_length >= y_length
        N = approximate_number_of_elements
        L = x_length if x_is_larger_than_y else y_length
        H = y_length if x_is_larger_than_y else x_length
        NL_float = (H - L + (H**2 + L**2 - 2.0*H*L + 4.0*H*L*N)**0.5)/(2.0*H)
        NH_float = N / NL_float
        NL = int(np.ceil(NL_float))
        NH = int(np.ceil(NH_float))
        number_of_x_points = NL + 1 if x_is_larger_than_y else NH + 1
        number_of_y_points = NH + 1 if x_is_larger_than_y else NL + 1
        x = np.linspace(self.x1, self.x2, number_of_x_points)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(self.y1, self.y2, number_of_y_points)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
        return regular_grid_element_centroids


#######################################################################################################################
#######################################################################################################################
class MichellDomain(BoundaryValueProblem):
    '''
    Class for defining a Michell domain corresponding to the Michell solution of 1904.
    '''
    def __init__(self,
                 x1: float = 0.0,
                 x2: float = 5.0,
                 y1: float = -2.0,
                 y2: float = 2.0,
                 hole_radius: float = 1.0,
                 applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.rectangle = Rectangle(x1=x1, x2=x2, y1=y1, y2=y2)
        self.circular_hole = Circle(center=(0.0, 0.0), radius=hole_radius)
        if 2 * hole_radius >= (y2 - y1):
            raise ValueError('Hole radius is too large for the given domain.')
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.hole_radius = hole_radius

    def get_bounding_box(self) -> np.ndarray:
        return self.rectangle.get_bounding_box()

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return subtract(self.rectangle.distance_function(point_coordinates),
                        self.circular_hole.distance_function(point_coordinates))

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        y_average = 0.5 * (self.y1 + self.y2)
        scale = 1.0
        load_region_thickness = 0.08 * self.y2
        mask = (x_coordinates > (self.x2 - nodal_tolerance)) & \
               (y_coordinates > (y_average - scale * nodal_tolerance - load_region_thickness)) & \
               (y_coordinates < (y_average + scale * nodal_tolerance + load_region_thickness))
        while not np.any(mask):
            mask = (x_coordinates > (self.x2 - nodal_tolerance)) & \
                   (y_coordinates > (y_average - scale * nodal_tolerance - load_region_thickness)) & \
                   (y_coordinates < (y_average + scale * nodal_tolerance + load_region_thickness))
            scale *= 2.0
        loaded_node_indices = np.argwhere(mask).ravel()

        radius = np.sqrt((x_coordinates - self.x1)**2 + (y_coordinates - y_average)**2)
        support_node_indices = np.argwhere(radius < (self.hole_radius + nodal_tolerance)).ravel()

        number_of_loaded_nodes = loaded_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude

        applied_displacement_1 = AppliedDisplacement(nodal_indices=loaded_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=support_node_indices,
                                                 fixed_y_nodal_indices=support_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions


#######################################################################################################################
#######################################################################################################################
class WrenchDomain(BoundaryValueProblem):
    '''
    Class for defining a wrench domain with a fixed hole on the right and a hole on the left with a downward load.
    '''
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.line1 = Line(x1=0.0, y1=0.3, x2=0.0, y2=-0.3)
        self.line2 = Line(x1=0.0, y1=-0.3, x2=2.0, y2=-0.5)
        self.line3 = Line(x1=2.0, y1=-0.5, x2=2.0, y2=0.5)
        self.line4 = Line(x1=2.0, y1=0.5, x2=0.0, y2=0.3)
        self.circle5 = Circle(center=(0.0, 0.0), radius=0.3)
        self.circle6 = Circle(center=(2.0, 0.0), radius=0.5)
        self.circle7 = Circle(center=(0.0, 0.0), radius=0.175)
        self.circle8 = Circle(center=(2.0, 0.0), radius=0.3)

    def get_bounding_box(self) -> np.ndarray:
        return np.array([-0.3, 2.5, -0.5, 0.5])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        d1 = self.line1.distance_function(point_coordinates)
        d2 = self.line2.distance_function(point_coordinates)
        d3 = self.line3.distance_function(point_coordinates)
        d4 = self.line4.distance_function(point_coordinates)
        d5 = self.circle5.distance_function(point_coordinates)
        d6 = self.circle6.distance_function(point_coordinates)
        d7 = self.circle7.distance_function(point_coordinates)
        d8 = self.circle8.distance_function(point_coordinates)
        outer = union(d6, union(d5, intersection(d4, intersection(d3, intersection(d2, d1)))))
        inner = union(d8, d7)
        return subtract(outer, inner)

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        fixed_hole_radius = np.sqrt((x_coordinates - 2.0)**2 + y_coordinates**2)
        support_node_indices = np.argwhere(fixed_hole_radius < (0.3 + nodal_tolerance)).ravel()

        radius = np.sqrt(x_coordinates**2 + y_coordinates**2)
        mask = (radius < (0.175 + nodal_tolerance)) & (y_coordinates < 0.0)
        loaded_node_indices = np.argwhere(mask).ravel()

        number_of_loaded_nodes = loaded_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=loaded_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=support_node_indices,
                                                 fixed_y_nodal_indices=support_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions


#######################################################################################################################
#######################################################################################################################
class SerpentineDomain(BoundaryValueProblem):
    '''
    Class for defining a serpentine domain which is a wave-like cantilever beam.
    '''
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        r = 4.0
        l = 3.0
        a = 4.0
        b = a * l / r
        c = a / r * (r**2 - l**2)**0.5
        d = -(-l**2 + r**2)**0.5
        self.bounding_box = np.array([0.0, 3.0 * l + 2.0 * b, -r - a - d, r + a + d], dtype=float)
        self.load_x_coordinate = 3.0 * l + 2.0 * b

        self.circle1 = Circle(center=(0.0, d), radius=r + a)
        self.circle2 = Circle(center=(0.0, d), radius=r)
        self.line3 = Line(x1=0.0, y1=d, x2=l, y2=0.0)
        self.line4 = Line(x1=0.0, y1=1.0, x2=0.0, y2=d)
        self.circle5 = Circle(center=(2.0 * l + b, c - d), radius=r + a)
        self.circle6 = Circle(center=(2.0 * l + b, c - d), radius=r)
        self.line7 = Line(x1=2.0 * l + b, y1=c - d, x2=l + b, y2=c)
        self.line8 = Line(x1=3.0 * l + b, y1=c, x2=2.0 * l + b, y2=c - d)

    def get_bounding_box(self) -> np.ndarray:
        return self.bounding_box.copy()

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        d1 = self.circle1.distance_function(point_coordinates)
        d2 = self.circle2.distance_function(point_coordinates)
        d3 = self.line3.distance_function(point_coordinates)
        d4 = self.line4.distance_function(point_coordinates)
        d5 = self.circle5.distance_function(point_coordinates)
        d6 = self.circle6.distance_function(point_coordinates)
        d7 = self.line7.distance_function(point_coordinates)
        d8 = self.line8.distance_function(point_coordinates)
        d9 = intersection(intersection(subtract(d1, d2), d3), d4)
        d10 = intersection(intersection(subtract(d5, d6), d7), d8)
        return union(d9, d10)

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        support_node_indices = np.argwhere(x_coordinates < nodal_tolerance).ravel()
       
        line_definition = -0.8833333 * x_coordinates + 13.25
        mask_on_line = np.isclose(y_coordinates, line_definition, atol=nodal_tolerance)
        mask_x_coordinate = x_coordinates > 14.60275 + nodal_tolerance
        mask_applied_load = mask_on_line & mask_x_coordinate
        
        load_edge_node_indices = np.argwhere(mask_applied_load).ravel()
        number_of_loaded_nodes = load_edge_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude #downward y displacement


        applied_displacement_1 = AppliedDisplacement(nodal_indices=load_edge_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=support_node_indices,
                                                 fixed_y_nodal_indices=support_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

#######################################################################################################################
#######################################################################################################################
class HookDomain(BoundaryValueProblem):
    """
    Class for defining a hook domain with a downward distributed load.
    """
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.applied_displacement_magnitude = applied_displacement_magnitude
        self.circle1 = Circle(center=(59.9713, 78.7683), radius=80.0)
        self.circle2 = Circle(center=(54.8716, 76.8672), radius=35.0)
        self.circle3 = Circle(center=(0.0, 80.6226), radius=20.0)
        self.circle4 = Circle(center=(0.0, 80.6226), radius=10.0)
        self.circle5 = Circle(center=(14.8842, 1.8605), radius=50.0)
        self.circle6 = Circle(center=(0.0, 0.0), radius=19.0)
        self.circle7 = Circle(center=(-27.0406, 0.0), radius=8.0406)
        self.line1 = Line(x1=65.4346, y1=76.9983, x2=-19.9904, y2=81.2407)
        self.line2 = Line(x1=-25.6060, y1=-27.4746, x2=65.4346, y2=76.9983)
        self.line3 = Line(x1=1.0, y1=0.0, x2=0.0, y2=0.0)

    def get_bounding_box(self) -> np.ndarray:
        return np.array([-35.0812, 64.8842, -48.1395, 100.6226])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        c1 = self.circle1.distance_function(point_coordinates)
        c2 = self.circle2.distance_function(point_coordinates)
        c3 = self.circle3.distance_function(point_coordinates)
        c4 = self.circle4.distance_function(point_coordinates)
        c5 = self.circle5.distance_function(point_coordinates)
        c6 = self.circle6.distance_function(point_coordinates)
        c7 = self.circle7.distance_function(point_coordinates)
        l1 = self.line1.distance_function(point_coordinates)
        l2 = self.line2.distance_function(point_coordinates)
        l3 = self.line3.distance_function(point_coordinates)
        d1 = subtract(union(intersection(subtract(c1, c2),intersection(l1, l2)), c3), c4)
        d2 = union(intersection(subtract(c5, c6), l3), c7)
        d3 = intersection(subtract(c5, c6), -1.0 * l2)
        return union(union(d1, d2), d3)

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0]
        y_coordinates = nodal_coordinates[:, 1]
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        upper_circle_radius = np.sqrt(x_coordinates**2 + (y_coordinates - 80.6226)**2)
        mask = (upper_circle_radius < (10.0 + nodal_tolerance)) & (y_coordinates >= 80.6226)
        upper_half_circle_indices = np.argwhere(mask).ravel()

        lower_circle_radius = np.sqrt(x_coordinates**2 + y_coordinates**2)
        mask = (lower_circle_radius < (19.0 + nodal_tolerance)) & (y_coordinates <= 0.0)
        lower_half_circle_indices = np.argwhere(mask).ravel()

        number_of_loaded_nodes = lower_half_circle_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=lower_half_circle_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=upper_half_circle_indices,
                                                 fixed_y_nodal_indices=upper_half_circle_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions
    
    def get_passive_region_indices(self,
                                   nodal_coordinates: np.ndarray,
                                   element_connectivity_arrays: Dict[int, np.ndarray],
                                   largest_element_edge_length: float
                                   ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        lower_circle_radius = np.sqrt(x_coordinates**2 + y_coordinates**2)
        padding_width = 2.0*largest_element_edge_length + nodal_tolerance
        mask = (lower_circle_radius < (19.0 + padding_width)) & (y_coordinates <= 0.0)
        passive_design_variable_indices = np.argwhere(mask).ravel()
        
        passive_local_element_indices = {}
        for number_of_vertices, element_connectivity in element_connectivity_arrays.items():
            # determine which elements have all of their nodes in the passive region
            mask = np.isin(element_connectivity, passive_design_variable_indices)
            mask = np.all(mask, axis=1)
            passive_local_element_indices[number_of_vertices] = np.argwhere(mask).ravel()

        return passive_design_variable_indices, passive_local_element_indices


#######################################################################################################################
#######################################################################################################################
class Lbracket(BoundaryValueProblem):
    """
    Class for defining a L-bracket domain which is fixed along the top edge and has a downward load at the top corner
    of the right-most edge, distributed over a small region.
    """
    def __init__(self, side_length: float = 30.0, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        if not isinstance(side_length, (float, int)):
            raise TypeError("The 'side_length' parameter must be a number. "
                            f"User specified side_length = '{side_length}'")
        if side_length <= 0.0:
            raise ValueError("The 'side_length' parameter must be greater than zero. "
                             f"User specified side_length = '{side_length}'")
        self.side_length = float(side_length)

        self.outer_rectangle = Rectangle(x1=0.0, x2=side_length, y1=0.0, y2=side_length)

        eps = 1.0e-2
        c = 1.0 + eps
        x1 = 0.4 * side_length
        x2 = c * side_length
        y1 = 0.4 * side_length
        y2 = c * side_length
        self.rectangular_hole = Rectangle(x1=x1, x2=x2, y1=y1, y2=y2)

    def get_bounding_box(self) -> np.ndarray:
        return np.array([0.0, self.side_length, 0.0, self.side_length])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return subtract(self.outer_rectangle.distance_function(point_coordinates),
                        self.rectangular_hole.distance_function(point_coordinates))

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        load_region_thickness = 4.0 * self.side_length / 100.0
        right_edge_mask = x_coordinates > (self.side_length - nodal_tolerance)
        load_region_bottom_y_coordinate = 0.4 * self.side_length - load_region_thickness - nodal_tolerance
        load_region_mask = right_edge_mask & (y_coordinates > load_region_bottom_y_coordinate)
        # load_region_mask = right_edge_mask
        load_region_node_indices = np.argwhere(load_region_mask).ravel()
        top_edge_node_indices = np.argwhere(y_coordinates > (self.side_length - nodal_tolerance)).ravel()

        number_of_loaded_nodes = load_region_node_indices.size
        applied_displacment_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacment_components[:, 1] = -1.0 * self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=load_region_node_indices,
                                     applied_displacement_components=applied_displacment_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=top_edge_node_indices,
                                                 fixed_y_nodal_indices=top_edge_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

    def is_regular_grid_implemented(self) -> bool:
        return True

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        number_of_center_square_elements_along_edge = int(np.ceil(approximate_number_of_elements**0.5 / 2.0))
        element_size = 0.4 / number_of_center_square_elements_along_edge
        x = np.linspace(0.0, 0.4, number_of_center_square_elements_along_edge + 1)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(0.0, 0.4, number_of_center_square_elements_along_edge + 1)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
        #
        number_of_outer_rectangle_elements_along_edge = int(np.ceil(0.6 / element_size))
        x = np.linspace(0.4, 1.0, number_of_outer_rectangle_elements_along_edge + 1)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(0.0, 0.4, number_of_center_square_elements_along_edge + 1)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        new_centroids = np.column_stack((X.ravel(), Y.ravel()))
        regular_grid_element_centroids = np.row_stack((regular_grid_element_centroids, new_centroids))
        #
        x = np.linspace(0.0, 0.4, number_of_center_square_elements_along_edge + 1)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(0.4, 1.0, number_of_outer_rectangle_elements_along_edge + 1)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        new_centroids = np.column_stack((X.ravel(), Y.ravel()))
        regular_grid_element_centroids = np.row_stack((regular_grid_element_centroids, new_centroids))
        #
        regular_grid_element_centroids *= self.side_length
        return regular_grid_element_centroids
    
    def get_passive_region_indices(self,
                                   nodal_coordinates: np.ndarray,
                                   element_connectivity_arrays: Dict[int, np.ndarray],
                                   largest_element_edge_length: float
                                   ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        load_region_thickness = 4.0 * self.side_length / 100.0
        padding_width = 2*largest_element_edge_length + nodal_tolerance
        right_edge_mask = x_coordinates > (self.side_length - padding_width)
        load_region_bottom_y_coordinate = 0.4 * self.side_length - load_region_thickness - padding_width
        load_region_mask = right_edge_mask & (y_coordinates > load_region_bottom_y_coordinate)
        passive_design_variable_indices = np.argwhere(load_region_mask).ravel()
        
        passive_local_element_indices = {}
        for number_of_vertices, element_connectivity in element_connectivity_arrays.items():
            # determine which elements have all of their nodes in the passive region
            mask = np.isin(element_connectivity, passive_design_variable_indices)
            mask = np.all(mask, axis=1)
            passive_local_element_indices[number_of_vertices] = np.argwhere(mask).ravel()

        return passive_design_variable_indices, passive_local_element_indices


#######################################################################################################################
#######################################################################################################################
class Corbel(BoundaryValueProblem):
    """
    Class for defining a corbel domain which is fixed along the top and bottom edges and has a downward load at
    the center of the right-most edge.
    """
    def __init__(self, side_length: float = 40.0, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        if not isinstance(side_length, (float, int)):
            raise TypeError("The 'side_length' parameter must be a number. "
                            f"User specified side_length = '{side_length}'")
        if side_length <= 0.0:
            raise ValueError("The 'side_length' parameter must be greater than zero. "
                             f"User specified side_length = '{side_length}'")
        self.side_length = float(side_length)

        self.left_rectangle = Rectangle(x1=0.0,
                                        x2=side_length,
                                        y1=-1.5 * side_length,
                                        y2= 1.5 * side_length)
        self.right_rectangle = Rectangle(x1= 0.9 * side_length,
                                         x2= 2.0 * side_length,
                                         y1=-0.5 * side_length,
                                         y2= 0.5 * side_length)


    def get_bounding_box(self) -> np.ndarray:
        return np.array([0.0, 2.0 * self.side_length, -1.5 * self.side_length, 1.5 * self.side_length])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return union(self.left_rectangle.distance_function(point_coordinates),
                     self.right_rectangle.distance_function(point_coordinates))

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        load_region_thickness = 0.08 * self.side_length
        load_region_x_mask = x_coordinates > (2.0 * self.side_length - nodal_tolerance)
        load_region_y_mask = np.abs(y_coordinates) < (0.5 * load_region_thickness - 2.0 * nodal_tolerance)
        load_region_mask = load_region_x_mask & load_region_y_mask
        load_region_node_indices = np.argwhere(load_region_mask).ravel()

        top_and_bottom_edge_node_mask = np.abs(y_coordinates) > (1.5 * self.side_length - nodal_tolerance)
        top_and_bottom_edge_node_indices = np.argwhere(top_and_bottom_edge_node_mask).ravel()

        number_of_loaded_nodes = load_region_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        # applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude 
        applied_displacement_components[:, 0] = 1.0 * self.applied_displacement_magnitude

        applied_displacement_1 = AppliedDisplacement(nodal_indices=load_region_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=top_and_bottom_edge_node_indices,
                                                 fixed_y_nodal_indices=top_and_bottom_edge_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

    def is_regular_grid_implemented(self) -> bool:
        return True

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        number_of_center_square_elements_along_edge = int(np.ceil(approximate_number_of_elements**0.5 / 2.0))
        if (number_of_center_square_elements_along_edge % 2) == 1:
            number_of_center_square_elements_along_edge -= 1
        x = np.linspace(0.0, 1.0, number_of_center_square_elements_along_edge + 1)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(-1.5, 1.5, 3 * number_of_center_square_elements_along_edge + 1)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
        #
        x = np.linspace(1.0, 2.0, number_of_center_square_elements_along_edge + 1)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(-0.5, 0.5, number_of_center_square_elements_along_edge + 1)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        new_centroids = np.column_stack((X.ravel(), Y.ravel()))
        regular_grid_element_centroids = np.row_stack((regular_grid_element_centroids, new_centroids))
        #
        regular_grid_element_centroids *= self.side_length
        return regular_grid_element_centroids
    
    def get_passive_region_indices(self,
                                   nodal_coordinates: np.ndarray,
                                   element_connectivity_arrays: Dict[int, np.ndarray],
                                   largest_element_edge_length: float
                                   ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        padding_width = 8.0*largest_element_edge_length + nodal_tolerance
        load_region_thickness = 0.08 * self.side_length
        load_region_x_mask = x_coordinates > (2.0 * self.side_length - padding_width)
        load_region_y_mask = np.abs(y_coordinates) < (0.5 * load_region_thickness + largest_element_edge_length )
        load_region_mask = load_region_x_mask & load_region_y_mask
        passive_design_variable_indices = np.argwhere(load_region_mask).ravel()
        
        passive_local_element_indices = {}
        for number_of_vertices, element_connectivity in element_connectivity_arrays.items():
            # determine which elements have all of their nodes in the passive region
            mask = np.isin(element_connectivity, passive_design_variable_indices)
            mask = np.all(mask, axis=1)
            passive_local_element_indices[number_of_vertices] = np.argwhere(mask).ravel()

        return passive_design_variable_indices, passive_local_element_indices


#######################################################################################################################
#######################################################################################################################
class CrackDomain(BoundaryValueProblem):
    """
    Class for defining a crack domain which demonstrates Mode-I opening of an edge crack \
    with symmetry boundary conditions.
    """
    def __init__(self, side_length: float = 2.0, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        if not isinstance(side_length, (float, int)):
            raise TypeError("The 'side_length' parameter must be a number. "
                            f"User specified side_length = '{side_length}'")
        if side_length <= 0.0:
            raise ValueError("The 'side_length' parameter must be greater than zero. "
                             f"User specified side_length = '{side_length}'")
        self.side_length = float(side_length)

        self.rectangle = Rectangle(x1= 0.0,
                                   x2= 0.5 * side_length,
                                   y1=-0.5 * side_length,
                                   y2= 0.5 * side_length)


    def get_bounding_box(self) -> np.ndarray:
        return self.rectangle.get_bounding_box()

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return self.rectangle.distance_function(point_coordinates)

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        left_edge_mask = x_coordinates < nodal_tolerance
        right_edge_mask = x_coordinates > (0.5 * self.side_length - nodal_tolerance)
        bottom_edge_mask = y_coordinates < (-0.5 * self.side_length + nodal_tolerance)
        bottom_half_mask = y_coordinates < nodal_tolerance
        upper_edge_mask = y_coordinates > (0.9 * self.side_length / 2.0 - nodal_tolerance)

        fixed_x_node_indices = np.argwhere(left_edge_mask & bottom_half_mask).ravel()
        fixed_y_node_indices = np.argwhere(left_edge_mask & bottom_edge_mask).ravel()
        load_region_node_indices = np.argwhere(right_edge_mask & upper_edge_mask).ravel()

        number_of_loaded_nodes = load_region_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 0] = self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=load_region_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=fixed_x_node_indices,
                                                 fixed_y_nodal_indices=fixed_y_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

    def is_regular_grid_implemented(self) -> bool:
        return True

    def get_regular_grid_element_centroids(self, approximate_number_of_elements: int) -> np.ndarray:
        number_of_center_square_elements_along_edge = int(np.ceil(approximate_number_of_elements**0.5 / 2.0))
        if (number_of_center_square_elements_along_edge % 2) == 1:
            number_of_center_square_elements_along_edge -= 1
        x = np.linspace(0.0, 0.5, number_of_center_square_elements_along_edge + 1)
        x = 0.5 * np.diff(x) + x[:-1]
        y = np.linspace(-0.5, 0.5, 2 * number_of_center_square_elements_along_edge + 1)
        y = 0.5 * np.diff(y) + y[:-1]
        X, Y = np.meshgrid(x, y)
        regular_grid_element_centroids = np.column_stack((X.ravel(), Y.ravel()))
        regular_grid_element_centroids *= self.side_length
        return regular_grid_element_centroids


#######################################################################################################################
#######################################################################################################################
class PortalFrame(BoundaryValueProblem):
    """
    Class for defining a portal frame domain.
    """
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        polygon_vertex_list_1 = np.array([[0.0, 3.5],
                                          [6.0 - 0.55, 0.0],
                                          [6.0, 0.0],
                                          [6.0, 6.0],
                                          [0.0, 6.0]])
        polygon_vertex_list_2 = np.array([[ 0.0, 3.5],
                                          [ 0.0, 6.0],
                                          [-6.0, 6.0],
                                          [-6.0, 0.0],
                                          [-6.0 + 0.55, 0.0]])
        self.polygon_1 = Polygon(polygon_vertex_list_1)
        self.polygon_2 = Polygon(polygon_vertex_list_2)


    def get_bounding_box(self) -> np.ndarray:
        return np.array([-6.0, 6.0, 0.0, 6.0])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return union(self.polygon_1.distance_function(point_coordinates),
                     self.polygon_2.distance_function(point_coordinates))

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        left_edge_mask = x_coordinates < (-6.0 + nodal_tolerance)
        bottom_edge_mask = y_coordinates < nodal_tolerance
        top_edge_mask = y_coordinates > (6.0 - nodal_tolerance)
        center_mask = np.abs(x_coordinates) < (0.5 + nodal_tolerance)

        fixed_x_node_indices = np.argwhere(left_edge_mask & bottom_edge_mask).ravel()
        fixed_y_node_indices = np.argwhere(bottom_edge_mask).ravel()
        load_region_node_indices = np.argwhere(top_edge_mask & center_mask).ravel()

        number_of_loaded_nodes = load_region_node_indices.size
        applied_displacement_components = np.zeros((number_of_loaded_nodes, 2), dtype=float)
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=load_region_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=fixed_x_node_indices,
                                                 fixed_y_nodal_indices=fixed_y_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions

    def get_passive_region_indices(self,
                                   nodal_coordinates: np.ndarray,
                                   element_connectivity_arrays: Dict[int, np.ndarray],
                                   largest_element_edge_length: float
                                   ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)
        top_edge_mask = y_coordinates > (6.0 - 2*largest_element_edge_length - nodal_tolerance)
        center_mask = np.abs(x_coordinates) < (0.5 + 2*largest_element_edge_length + nodal_tolerance)
        passive_design_variable_indices = np.argwhere(top_edge_mask & center_mask).ravel()
        passive_local_element_indices = {}

        for number_of_vertices, element_connectivity in element_connectivity_arrays.items():
            # determine which elements have all of their nodes in the passive region
            mask = np.isin(element_connectivity, passive_design_variable_indices)
            mask = np.all(mask, axis=1)
            passive_local_element_indices[number_of_vertices] = np.argwhere(mask).ravel()

        return passive_design_variable_indices, passive_local_element_indices


#######################################################################################################################
#######################################################################################################################
class EyeBar(BoundaryValueProblem):
    """
    Class for defining an eye bar domain.
    """
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        self.rectangle = Rectangle(x1=0.0, x2=1.6, y1=-0.4, y2=0.4)
        self.circular_hole = Circle(center=(0.4, 0.0), radius=0.15)

    def get_bounding_box(self) -> np.ndarray:
        return self.rectangle.get_bounding_box()

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return subtract(self.rectangle.distance_function(point_coordinates),
                        self.circular_hole.distance_function(point_coordinates))

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        right_edge_mask = x_coordinates > (1.6 - nodal_tolerance)
        circular_edge_mask = (np.sqrt((x_coordinates - 0.4)**2 + y_coordinates**2) < (0.15 + nodal_tolerance)) & \
                             (x_coordinates < (0.4 + nodal_tolerance))
        center_mask = np.abs(y_coordinates) < (0.15 + nodal_tolerance)

        fixed_node_indices = np.argwhere(right_edge_mask & center_mask).ravel()
        load_region_node_indices = np.argwhere(circular_edge_mask).ravel()

        applied_displacement_components = np.zeros((load_region_node_indices.size, 2))
        applied_displacement_components[:, 0] = -1.0 * self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=load_region_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=fixed_node_indices,
                                                 fixed_y_nodal_indices=fixed_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions


#######################################################################################################################
#######################################################################################################################
class AntennaBracket(BoundaryValueProblem):
    """
    Class for defining an antenna bracket domain.
    """
    def __init__(self, applied_displacement_magnitude: float = 1.0):
        super().__init__(applied_displacement_magnitude=applied_displacement_magnitude)
        polygon_vertex_list_1 = np.array([[0.0, 0.0],
                                          [1.0, 0.0],
                                          [0.7, 1.0],
                                          [0.0, 1.0]])
        polygon_vertex_list_2 = np.array([[1.0, 0.0],
                                          [3.0, 1.75],
                                          [2.8, 2.0],
                                          [0.7, 1.0]])
        self.polygon_1 = Polygon(polygon_vertex_list_1)
        self.polygon_2 = Polygon(polygon_vertex_list_2)

    def get_bounding_box(self) -> np.ndarray:
        return np.array([0.0, 3.0, 0.0, 2.0])

    def distance_function(self, point_coordinates: np.ndarray) -> np.ndarray:
        return union(self.polygon_1.distance_function(point_coordinates),
                     self.polygon_2.distance_function(point_coordinates))

    def get_boundary_conditions(self, nodal_coordinates: np.ndarray) -> BoundaryConditions:
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        bounding_box = self.get_bounding_box()
        bounding_box_area = (bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])
        number_of_nodes = float(nodal_coordinates.shape[0])
        nodal_tolerance = 0.1 * (bounding_box_area**0.5) / (number_of_nodes**0.5)

        d = 0.15 * np.sqrt((2.8 - 3.0)**2 + (2.0 - 1.75)**2)
        loaded_nodes_mask = (np.abs(1.25 * (x_coordinates - 2.8) + y_coordinates - 2.0) < nodal_tolerance) & \
                            (np.sqrt((x_coordinates - 2.9)**2 + (y_coordinates - 1.875)**2) < (d + nodal_tolerance))

        left_edge_mask = x_coordinates < nodal_tolerance

        fixed_node_indices = np.argwhere(left_edge_mask).ravel()
        load_region_node_indices = np.argwhere(loaded_nodes_mask).ravel()

        number_of_loaded_nodes = load_region_node_indices.size
        applied_displacement_components = np.zeros((load_region_node_indices.size, 2))
        applied_displacement_components[:, 1] = -1.0 * self.applied_displacement_magnitude 

        applied_displacement_1 = AppliedDisplacement(nodal_indices=load_region_node_indices,
                                     applied_displacement_components=applied_displacement_components)
        boundary_conditions = BoundaryConditions(fixed_x_nodal_indices=fixed_node_indices,
                                                 fixed_y_nodal_indices=fixed_node_indices,
                                                 applied_displacements=[applied_displacement_1])
        return boundary_conditions
