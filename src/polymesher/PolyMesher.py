import itertools
import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import src.polymesher.PolyGeometryAndBCs as polygeometry
from matplotlib import collections as mc
from typing import Dict, List, Optional, Tuple

#######################################################################################################################
#######################################################################################################################
class PolyMesher:
    """
    This class is used to generate a polygonal mesh for a given boundary value problem.
    """
    def __init__(self,
                 boundary_value_problem: polygeometry.BoundaryValueProblem,
                 number_of_elements: Optional[int] = None,
                 element_center_points: Optional[np.ndarray] = None,
                 maximum_iterations: int = 20,
                 use_regular_grid_if_implemented: bool = False):
        """
        Constructor for the PyPolyMesher class.

        Args:
            boundary_value_problem (polygeometry.BoundaryValueProblem): The boundary value problem.
            number_of_elements (Optional[int]): The number of elements in the mesh. If this parameter is specified,
                then the initial seeds for the mesh are generated randomly. If this parameter is not specified,
                then the initial seeds for the mesh are specified by the parameter 'element_center_points'.
                Defaults to None.
            element_center_points (Optional[np.ndarray]): The initial seeds for the mesh. If this parameter is
                specified, then the parameter 'number_of_elements' is ignored. If this parameter is not specified,
                then the initial seeds for the mesh are generated randomly. Defaults to None.
            maximum_iterations (int): The maximum number of iterations to use in the mesh generation algorithm.
                Defaults to 20.
            use_regular_grid_if_implemented (bool): If True, then the regular grid implementation of the boundary value
                problem is used to generate the initial seeds for the mesh. If False, then the initial seeds for the
                mesh are generated randomly. Defaults to False.
        """
        self.logger = logging.getLogger("PyPoly.Mesher")
        self.boundary_value_problem = boundary_value_problem
        self.bounding_box = self.boundary_value_problem.get_bounding_box()

        if number_of_elements is not None:
            if not isinstance(number_of_elements, int):
                raise TypeError("The parameter 'number_of_elements' must be an integer. User provided "
                                f"a variable of type = '{type(number_of_elements)}'.")
            if number_of_elements <= 0:
                raise ValueError("The parameter 'number_of_elements' must be positive. "
                                 f"User provided a value of '{number_of_elements}'.")

        if use_regular_grid_if_implemented:
            if boundary_value_problem.is_regular_grid_implemented():
                if number_of_elements is not None:
                    element_center_points = \
                        boundary_value_problem.get_regular_grid_element_centroids(number_of_elements)
                    number_of_elements = None
                    maximum_iterations = 0

        if number_of_elements is None:
            if element_center_points is None:
                raise ValueError("User must specify either the 'number_of_elements' or "
                                 "the initial seeds as 'element_center_points'.")
            if not isinstance(element_center_points, np.ndarray):
                raise TypeError(f"The parameter 'element_center_points' must be a numpy array. "
                                f"User provided a variable of type = '{type(element_center_points)}'.")
            if element_center_points.shape[1] != 2:
                raise ValueError("The second dimension of 'element_center_points' must be of size 2 in 2D.")
            self.number_of_elements = element_center_points.shape[0]
            self.element_center_points = element_center_points
        else:
            self.number_of_elements = number_of_elements
            self.element_center_points = self.generate_random_initial_element_centers()

        if not isinstance(maximum_iterations, int):
            raise TypeError("The parameter 'maximum_iterations' must be an integer. "
                            f"User provided a variable of type = '{type(maximum_iterations)}'.")
        if maximum_iterations < 0:
            raise ValueError("The parameter 'maximum_iterations' cannot be negative. "
                             f"User provided a value of '{maximum_iterations}'.")
        self.maximum_iterations = maximum_iterations

        self.element_connectivity_arrays = None
        self.nodal_coordinates_array = None
        self.boundary_conditions = None
        self.unique_edges = None
        self.unique_edge_lengths = None
        self.matplotlib_edges = None


    def generate_random_initial_element_centers(self) -> np.ndarray:
        number_of_elements = self.number_of_elements
        element_centers = np.zeros((number_of_elements, 2))
        temporary_centers = np.zeros((number_of_elements, 2))
        rng = np.random.default_rng(seed=777)
        number_of_elements_added = 0
        while number_of_elements_added < number_of_elements:
            temporary_centers[:, 0] = \
                rng.uniform(low=self.bounding_box[0], high=self.bounding_box[1], size=(number_of_elements,))
            temporary_centers[:, 1] = \
                rng.uniform(low=self.bounding_box[2], high=self.bounding_box[3], size=(number_of_elements,))
            distances_to_boundary = self.boundary_value_problem.distance_function(temporary_centers)
            valid_elements = np.argwhere(distances_to_boundary[:, -1] < 0.0).ravel()
            number_to_add = min(number_of_elements - number_of_elements_added, valid_elements.size)
            last_index = number_of_elements_added + number_to_add
            element_centers[number_of_elements_added:last_index, :] = \
                temporary_centers[valid_elements[:number_to_add], :]
            number_of_elements_added += number_to_add
        return element_centers


    def reflect_point_set(self, point_set: np.ndarray, alpha: float) -> np.ndarray:
        number_of_elements = self.number_of_elements
        self.logger.debug('Reflect Begin')
        finite_difference_step_size = 1.0e-8
        eta = 0.9
        my_distance = self.boundary_value_problem.distance_function(point_set)
        number_of_boundary_segments = my_distance.shape[1] - 1
        temp1 = np.array([[finite_difference_step_size, 0.0]])
        temp2 = np.array([[0.0, finite_difference_step_size]])
        n1 = (self.boundary_value_problem.distance_function(point_set + temp1) -
              self.boundary_value_problem.distance_function(point_set - temp1)) / (2.0 * finite_difference_step_size)
        n2 = (self.boundary_value_problem.distance_function(point_set + temp2) -
              self.boundary_value_problem.distance_function(point_set - temp2)) / (2.0 * finite_difference_step_size)
        seed_shape = (number_of_elements, number_of_boundary_segments)
        seeds_near_boundary = np.abs(my_distance[:, :number_of_boundary_segments].reshape(seed_shape)) < alpha
        P1 = np.tile(point_set[:, 0].reshape((number_of_elements, 1)), (1, number_of_boundary_segments))
        P2 = np.tile(point_set[:, 1].reshape((number_of_elements, 1)), (1, number_of_boundary_segments))

        extension = np.zeros((seeds_near_boundary.shape[0], 1), dtype=bool)
        seeds_near_boundary_extended = np.hstack((seeds_near_boundary, extension))
        reflected_point_set = np.zeros(seeds_near_boundary.shape)
        reflected_point_set_0 = P1[seeds_near_boundary] - \
            2.0 * n1[seeds_near_boundary_extended] * my_distance[seeds_near_boundary_extended]
        reflected_point_set_1 = P2[seeds_near_boundary] - \
            2.0 * n2[seeds_near_boundary_extended] * my_distance[seeds_near_boundary_extended]
        reflected_point_set = np.zeros((reflected_point_set_0.size, 2))
        reflected_point_set[:, 0] = reflected_point_set_0.ravel()
        reflected_point_set[:, 1] = reflected_point_set_1.ravel()
        distance_to_reflected_points = self.boundary_value_problem.distance_function(reflected_point_set)
        a = np.abs(distance_to_reflected_points[:, -1]) >= (eta * np.abs(my_distance[seeds_near_boundary_extended]))
        b = distance_to_reflected_points[:, -1] > 0.0
        J = a & b
        reflected_point_set = reflected_point_set[J, :]
        reflected_point_set = np.unique(reflected_point_set, axis=0)
        self.logger.debug('Reflect End')
        return reflected_point_set


    def compute_polygon_centroids(self,
                                  element_connectivity: List[List[int]],
                                  nodal_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.logger.debug('Centroid Begin')
        number_of_elements = self.number_of_elements
        number_of_nodes_per_element = np.array(
            [len(element_connectivity[i]) for i in range(number_of_elements)],
            dtype=int
        )
        unique_numbers_of_nodes = np.unique(number_of_nodes_per_element)
        element_centroids = np.zeros((number_of_elements, 2))
        element_areas = np.zeros((number_of_elements))
        for N in unique_numbers_of_nodes:
            element_indices = np.argwhere(number_of_nodes_per_element == N).ravel()
            element_connectivity_array = np.array(
                [element_connectivity[i]
                 for i in range(number_of_elements) if number_of_nodes_per_element[i] == N],
                dtype=int
            )
            vx = nodal_coordinates[element_connectivity_array, 0]
            vy = nodal_coordinates[element_connectivity_array, 1]
            shifted_indices = np.concatenate((np.arange(1, N, dtype=int), np.array([0], dtype=int)))
            vx_shifted = vx[:, shifted_indices]
            vy_shifted = vy[:, shifted_indices]
            temp = vx * vy_shifted - vy * vx_shifted
            element_areas[element_indices] = 0.5 * np.sum(temp, axis=1)
            temp1 = np.sum((vx + vx_shifted) * temp, axis=1)
            temp2 = np.sum((vy + vy_shifted) * temp, axis=1)
            denominator = 6.0 * element_areas[element_indices].reshape((element_indices.size, 1))
            element_centroids[element_indices, :] = np.column_stack((temp1, temp2)) / denominator
        self.logger.debug('Centroid End')
        return element_centroids, np.abs(element_areas)


    def rebuild_arrays(self,
                       nodal_coordinates_in: np.ndarray,
                       element_connectivity_in: List[List[int]],
                       contiguous_node_indices: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        self.logger.debug('Rebuild Begin')
        _, ix, jx = np.unique(contiguous_node_indices, return_index=True, return_inverse=True)
        number_of_elements = len(element_connectivity_in)
        element_connectivity = []
        if nodal_coordinates_in.shape[0] > ix.size:
            ix[-1] = np.amax(contiguous_node_indices)
        nodal_coordinates = nodal_coordinates_in[ix, :]
        for element_index in range(number_of_elements):
            temp_element_connectivity = np.unique(jx[element_connectivity_in[element_index]])
            node_x_coordinates = nodal_coordinates[temp_element_connectivity, 0]
            node_y_coordinates = nodal_coordinates[temp_element_connectivity, 1]
            iix = np.argsort(np.arctan2(node_y_coordinates - np.mean(node_y_coordinates),
                                        node_x_coordinates - np.mean(node_x_coordinates)))
            element_connectivity.append(temp_element_connectivity[iix])
        self.logger.debug('Rebuild End')
        return nodal_coordinates, element_connectivity


    def extract_nodes(self,
                      nodal_coordinates_in: np.ndarray,
                      element_connectivity_in: List[List[int]]) -> Tuple[np.ndarray, List[List[int]]]:
        number_of_elements = self.number_of_elements
        self.logger.debug('Extract Begin')
        element_connectivity = [element_connectivity_in[i] for i in range(number_of_elements)]
        all_element_nodes = np.array(list(itertools.chain(*element_connectivity)), dtype=int)
        unique_element_nodes = np.unique(all_element_nodes)
        contiguous_node_indices = np.arange(nodal_coordinates_in.shape[0], dtype=int)
        my_diff = np.setdiff1d(contiguous_node_indices, unique_element_nodes)
        contiguous_node_indices[my_diff] = np.amax(unique_element_nodes)
        self.logger.debug('Extract End')
        nodal_coordinates, element_connectivity = self.rebuild_arrays(nodal_coordinates_in,
                                                                      element_connectivity,
                                                                      contiguous_node_indices)
        return nodal_coordinates, element_connectivity


    def collapse_edges(self,
                       nodal_coordinates_in: np.ndarray,
                       element_connectivity_in: List[List[int]],
                       tolerance: float) -> Tuple[np.ndarray, List[List[int]]]:
        self.logger.debug('Collapse Begin')
        cEdge = None
        for element_index in range(len(element_connectivity_in)):
            if len(element_connectivity_in[element_index]) < 4:
                continue
            node_indices = np.array(element_connectivity_in[element_index], dtype=int)
            shifted = np.concatenate((np.arange(1, node_indices.size, dtype=int), np.array([0], dtype=int)))
            node_indices_shifted = node_indices[shifted]
            vx = nodal_coordinates_in[node_indices, 0]
            vy = nodal_coordinates_in[node_indices, 1]
            beta = np.pi + np.arctan2(vy-np.mean(vy), vx-np.mean(vx))
            beta = np.mod(beta[shifted] - beta, 2.0*np.pi)
            betaIdeal = 2.0 * np.pi / node_indices.size
            mask = beta < (tolerance * betaIdeal)
            if np.any(mask):
                Edge = np.column_stack((node_indices, node_indices_shifted))
                temp = Edge[mask, :]
                if cEdge is None:
                    cEdge = temp
                else:
                    cEdge = np.vstack((cEdge, temp))
        if cEdge is None:
            return nodal_coordinates_in, element_connectivity_in
        cEdge = np.unique(np.sort(cEdge, axis=1), axis=0)
        cNode = np.arange(nodal_coordinates_in.shape[0], dtype=int)
        for i in range(cEdge.shape[0]):
            cNode[cEdge[i, 1]] = cNode[cEdge[i, 0]]
        self.logger.debug('Collapse End')
        nodal_coordinates_in, element_connectivity_in = self.rebuild_arrays(nodal_coordinates_in,
                                                                            element_connectivity_in,
                                                                            cNode)
        return nodal_coordinates_in, element_connectivity_in


    def get_boundary_edges_without_boundary_conditions(self) -> np.ndarray:
        '''Return the boundary edges without boundary conditions applied.
        This is useful when creating the PDE filter with the boundary penalization. Edges with
        boundary conditions should not have their densities penalized by the filter.

        Returns:
            np.ndarray: The boundary edges (pairs of nodal indices) without applied boundary conditions.
        '''
        smallest_edge_length, _ = self.get_smallest_and_largest_edge_length()
        distances_to_boundary = self.boundary_value_problem.distance_function(self.nodal_coordinates_array)
        geometric_tolerance = smallest_edge_length / 10.0
        mask = np.abs(distances_to_boundary[:, -1]) < geometric_tolerance
        boundary_node_indices = np.argwhere(mask).ravel()

        edges_with_at_least_one_boundary_node_mask = np.isin(self.unique_edges, boundary_node_indices)
        boundary_edges_mask = edges_with_at_least_one_boundary_node_mask[0, :] & \
                              edges_with_at_least_one_boundary_node_mask[1, :]
        unique_boundary_edges = self.unique_edges[:, boundary_edges_mask]

        # remove edges with boundary conditions applied
        first_node_in_edges  = unique_boundary_edges[0, :].ravel()
        second_node_in_edges = unique_boundary_edges[1, :].ravel()
        boundary_condition_nodes = \
            [applied_displacement.nodal_indices for applied_displacement in self.boundary_conditions.applied_displacements]
        boundary_condition_nodes.append(self.boundary_conditions.fixed_x_nodal_indices)
        boundary_condition_nodes.append(self.boundary_conditions.fixed_y_nodal_indices)
        unique_boundary_condition_nodes = np.unique(np.concatenate(boundary_condition_nodes))
        mask = np.isin(first_node_in_edges,  unique_boundary_condition_nodes) | \
               np.isin(second_node_in_edges, unique_boundary_condition_nodes)
        boundary_edges_without_boundary_conditions = unique_boundary_edges[:, ~mask]
        return boundary_edges_without_boundary_conditions


    def get_smallest_and_largest_edge_length(self) -> Tuple[float, float]:
        '''Return the smallest and largest edge lengths in the mesh.
        If the mesh has not been generated, then run() will be called.

        Returns:
            Tuple[float, float]: The smallest and largest edge lengths in the mesh.'''
        if self.unique_edge_lengths is not None:
            smallest_edge_length = np.amin(self.unique_edge_lengths)
            largest_edge_length = np.amax(self.unique_edge_lengths)
            return smallest_edge_length, largest_edge_length
        if self.element_connectivity_arrays is None:
            self.run()
        self.logger.debug('Get Edge Length Begin')
        edges = None
        for N, element_connectivity_array in self.element_connectivity_arrays.items():
            shifted = np.concatenate((np.arange(1, N, dtype=int), np.array([0], dtype=int)))
            shifted_connectivity_array = element_connectivity_array[:, shifted]
            temp = np.row_stack((element_connectivity_array.ravel(), shifted_connectivity_array.ravel()))
            if edges is None:
                edges = temp.copy()
            else:
                edges = np.hstack((edges, temp))
        sorted_edges = np.sort(edges, axis=0)
        self.unique_edges = np.unique(sorted_edges, axis=1)
        first_node  = self.nodal_coordinates_array[self.unique_edges[0, :], :]
        second_node = self.nodal_coordinates_array[self.unique_edges[1, :], :]
        self.unique_edge_lengths = np.linalg.norm(second_node - first_node, axis=1, ord=2)

        self.matplotlib_edges = [[(first_node[edge_index, 0],  first_node[edge_index, 1]),
                                  (second_node[edge_index, 0], second_node[edge_index, 1])]
                                  for edge_index in range(self.unique_edge_lengths.size)]

        smallest_edge_length = np.amin(self.unique_edge_lengths)
        largest_edge_length = np.amax(self.unique_edge_lengths)
        self.logger.debug('Get Edge Length End')
        ratio = smallest_edge_length / largest_edge_length
        message = f"Smallest edge length = {smallest_edge_length:0.3e}, " + \
                  f"Largest edge length = {largest_edge_length:0.3e}, " + \
                  f"Ratio = {ratio:0.5f}"
        self.logger.info(message)
        return smallest_edge_length, largest_edge_length


    def plot(self):
        '''Plot the mesh and the boundary conditions.'''
        if self.matplotlib_edges is None:
            self.get_smallest_and_largest_edge_length()
        self.logger.debug('Plot Begin')
        fig, ax = plt.subplots()
        ax.axis('equal')
        lc = mc.LineCollection(self.matplotlib_edges, colors='k')
        ax.add_collection(lc)
        if self.boundary_conditions is not None:
            my_colors = ['g', 'm', 'b', 'r', 'c', 'y'] * max(2, len(self.boundary_conditions.applied_displacements) // 6 + 1)
            fixed_x = self.nodal_coordinates_array[self.boundary_conditions.fixed_x_nodal_indices, :]
            fixed_y = self.nodal_coordinates_array[self.boundary_conditions.fixed_y_nodal_indices, :]
            ax.plot(fixed_x[:, 0], fixed_x[:, 1], 'bo', label="Fixed X")
            for load_index, applied_displacements in enumerate(self.boundary_conditions.applied_displacements):
                load_x_vectors = applied_displacements.applied_displacement_components[:, 0]
                load_y_vectors = applied_displacements.applied_displacement_components[:, 1]
                applied_load_node_x_coordinates = self.nodal_coordinates_array[applied_displacements.nodal_indices, 0].ravel()
                applied_load_node_y_coordinates = self.nodal_coordinates_array[applied_displacements.nodal_indices, 1].ravel()
                my_label = f"Applied Displacement"
                # ax.plot(applied_load_node_x_coordinates, applied_load_node_y_coordinates, 'go', label=my_label)
                ax.quiver(applied_load_node_x_coordinates,
                          applied_load_node_y_coordinates,
                          load_x_vectors,
                          load_y_vectors,
                          linewidths=1.0,
                          edgecolors=my_colors[load_index],
                          color=my_colors[load_index],
                          label=my_label,
                          pivot='tip',
                          width=0.005)
            ax.plot(fixed_y[:, 0], fixed_y[:, 1], 'rx', label="Fixed Y")

        ax.autoscale()
        ax.legend()
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        fig.tight_layout()
        self.logger.debug('Plot End')


    def run(self):
        '''Generate the mesh.'''
        convergence_tolerance = 1.0e-5
        iteration_number = 0
        convergence_metric = 1.0
        c = 1.5

        bounding_box_area = (self.bounding_box[1] - self.bounding_box[0]) * \
                            (self.bounding_box[3] - self.bounding_box[2])
        element_centers = self.element_center_points.copy()
        alpha = c * np.sqrt(bounding_box_area / self.number_of_elements)
        nodal_coordinates = None
        element_connectivity = None
        while iteration_number <= self.maximum_iterations and convergence_metric > convergence_tolerance:
            element_centers_copy = element_centers.copy()
            reflected_element_centers = self.reflect_point_set(element_centers_copy, alpha)

            stacked_element_centers = np.row_stack((element_centers_copy, reflected_element_centers))
            voronoi_object = sp.spatial.Voronoi(stacked_element_centers, qhull_options="Qbb")
            nodal_coordinates = voronoi_object.vertices
            point_region_indices = voronoi_object.point_region

            element_connectivity = [voronoi_object.regions[point_region_indices[element_index]]
                                    for element_index in range(point_region_indices.size)]
            element_centers, element_areas = self.compute_polygon_centroids(element_connectivity, nodal_coordinates)
            bounding_box_area = np.sum(element_areas)
            area_weighted_element_center_difference_squared = \
                np.sum((element_areas**2) * np.sum((element_centers - element_centers_copy)**2, axis=1))
            convergence_metric = np.sqrt(area_weighted_element_center_difference_squared) * \
                self.number_of_elements / (bounding_box_area**1.5)
            data_string = f"Iteration = {iteration_number:3d}, " + \
                          f"Area = {bounding_box_area:0.3e}, " + \
                          f"Error = {convergence_metric:0.2e}"
            self.logger.info(data_string)
            iteration_number += 1

        nodal_coordinates, element_connectivity = self.extract_nodes(nodal_coordinates, element_connectivity)
        nodal_coordinates, element_connectivity = self.collapse_edges(nodal_coordinates, element_connectivity, 0.1)

        number_of_nodes_per_element = np.array(list(map(len, element_connectivity)), dtype=int)
        unique_numbers_of_nodes, my_counts = np.unique(number_of_nodes_per_element, return_counts=True)
        mesh_string = f"Mesh = {self.number_of_elements} elements"
        for i in range(unique_numbers_of_nodes.size):
            N = unique_numbers_of_nodes[i]
            number_of_N_gons = my_counts[i]
            mesh_string += f", {number_of_N_gons} ({N}-gons)"
        self.logger.info(mesh_string)
        element_connectivity_arrays = {}
        for N in unique_numbers_of_nodes:
            element_connectivity_arrays[N] = np.array(
                [element_connectivity[i]
                 for i in range(self.number_of_elements) if number_of_nodes_per_element[i] == N],
                dtype=int
            )

        self.element_connectivity_arrays = element_connectivity_arrays
        self.nodal_coordinates_array = nodal_coordinates
        self.boundary_conditions = self.boundary_value_problem.get_boundary_conditions(nodal_coordinates)


    def get_mesh_and_boundary_conditions(self) -> Tuple[np.ndarray,
                                                        Dict[int, np.ndarray],
                                                        polygeometry.BoundaryConditions]:
        '''Return the mesh and boundary conditions. Calls run() if the mesh has not been generated.

        Returns:
            Tuple[np.ndarray, Dict[int, np.ndarray], polygeometry.BoundaryConditions]:
                The nodal coordinates array, element connectivity arrays, and boundary conditions.
        '''
        if  self.element_connectivity_arrays is None or \
            self.nodal_coordinates_array is None or \
            self.boundary_conditions is None:
            self.run()
        return self.nodal_coordinates_array, self.element_connectivity_arrays, self.boundary_conditions


    def get_delaunay_simplices(self) -> np.ndarray:
        '''Return the simplices of the Delaunay triangulation of the mesh for making contour plots.

        Returns:
            np.ndarray: The simplices of the Delaunay triangulation of the mesh.
        '''
        delaunay_triangulation = sp.spatial.Delaunay(self.nodal_coordinates_array, qhull_options='Qbb QJ')
        element_centroids = np.mean(self.nodal_coordinates_array[delaunay_triangulation.simplices, :], axis=1)
        distance_to_boundary = self.boundary_value_problem.distance_function(element_centroids)
        indices = np.argwhere(distance_to_boundary[:, -1] <= 0.0).ravel()
        triangles_in_the_domain = delaunay_triangulation.simplices[indices, :]
        return triangles_in_the_domain
