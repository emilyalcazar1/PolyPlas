import logging
from typing import Optional
import numpy as np
import scipy as sp
import src.polymesher.PolyMesher as polymesher
import src.polyplas.PolyPlasUtilities as polyplas_utils
from abc import ABC, abstractmethod
from time import perf_counter
try:
    from sksparse.cholmod import cholesky as factorized
except ImportError:
    from scipy.sparse.linalg import factorized


def get_design_variable_filter(polymesh_object: polymesher.PolyMesher,
                               precomputed_data: polyplas_utils.PolyPlasPrecomputedData,
                               filter_radius: float = 1.0,
                               use_relative_filter_radius: bool = False,
                               filter_type: str = "identity",
                               **kwargs):
    """
    Returns a filter object for use in topology optimization based on user-specified parameters.

    Args:
        polymesh_object: A PolyMesher object representing the geometry and finite element mesh.
        precomputed_data: A named tuple of precomputed data used by the filter object.
        filter_radius: A float specifying the filter radius
        use_relative_filter_radius: A boolean specifying whether the filter radius is a multiplier on the \
                                    largest element edge length in the mesh.
        filter_type: A string specifying the type of filter to use \
                     (e.g., identity, polynomial).

    Returns:
        A PyPolyFilter.FilterBase object that can be used to filter topology optimization design variables.

    """
    logger = logging.getLogger("PyPoly.Filter")

    filter_type = filter_type.lower()

    if filter_type == "identity":
        logger.info("Identity filter selected.")
        return IdentityFilter()

    if not isinstance(polymesh_object, polymesher.PolyMesher):
        raise TypeError(f"The polymesh_object must be a PolyMesher object. User supplied '{type(polymesh_object)}'")

    if not isinstance(filter_radius, (float, int)):
        raise TypeError("The filter_radius must be a number. "
                        f"User supplied '{filter_radius}' of type '{type(filter_radius)}'.")
    if filter_radius < 0.0:
        raise ValueError(f"The filter_radius must be >= 0. User supplied '{filter_radius}'.")

    if not isinstance(use_relative_filter_radius, bool):
        raise TypeError("The use_relative_filter_radius must be a boolean (i.e., True or False). "
                        f"User supplied '{use_relative_filter_radius}' of type '{type(use_relative_filter_radius)}'.")

    if use_relative_filter_radius:
        if filter_radius < 1.0:
            raise ValueError(f"The filter radius must not be less than 1 when using relative filter radius. User supplied '{filter_radius}.")
            logger.warn("User supplied relative filter radius less than 1. This may result in no filtering.")
        _, largest_edge_length = polymesh_object.get_smallest_and_largest_edge_length()
        filter_radius *= largest_edge_length

    if not isinstance(precomputed_data, (polyplas_utils.PolyPlasPrecomputedData)):
        raise TypeError(f"The precomputed_data must be a precomputed data object. "
                        f"User supplied '{type(precomputed_data)}'.")

    if filter_type == "polynomial":
        message = f"Polynomial Filter selected with filter radius = {filter_radius:0.3e}"
        logger.info(message)
        return PolynomialFilter(precomputed_data, filter_radius=filter_radius, **kwargs)
    
    
    raise ValueError(f"The user supplied filter_type '{filter_type}' is not implemented. "
                     "Valid types are 'identity' or 'polynomial'.")


#######################################################################################################################
#######################################################################################################################
class FilterBase(ABC):
    """Base class for design variable filters.

    This class defines the basic interface for all design variable filters.
    """

    def __init__(self, filter_radius: float = 1.0):
        self.filter_radius = filter_radius

    @abstractmethod
    def apply_filter(self, design_variables: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def apply_chain_rule(self, sensitivity_vector: np.ndarray) -> np.ndarray:
        pass

    @property
    def filter_radius(self) -> float:
        return self._filter_radius

    @filter_radius.setter
    def filter_radius(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError(f"The 'filter_radius' must be a number. User supplied '{value}'"
                            f" which is a {type(value)}.")
        if value <= 0.0:
            raise ValueError(f"The 'filter_radius' must be positive. User supplied '{value}'")
        self._filter_radius = value


#######################################################################################################################
#######################################################################################################################
class PolynomialFilter(FilterBase):
    """A standard polynomial design variable filter.
    """
    def __init__(self,
                 precomputed_data: polyplas_utils.PolyPlasPrecomputedData,
                 filter_radius: float = 1.0,
                 filter_exponent: float = 1,
                 axis_of_symmetry: str = 'none'):
        """Constructor for a standard polynomial design variable filter.

        Args:
            precomputed_data (PolyTopPrecomputedData | PolyStressPrecomputedData | PolyPlasPrecomputedData):  \
                a precomputed data object containing the precomputed numerical quantities for the \
                finite element problem.
            filter_radius (float): the filter radius, defaults to 1.0.
            filter_exponent (float): the exponent of the filter polynomial, defaults to 1.

        Raises:
            TypeError: If filter_radius is not a number.
            ValueError: If filter_radius is not positive.
        """
        super().__init__(filter_radius=filter_radius)
        self.type = 'polynomial'
        self.logger = logging.getLogger("PyPoly.Filter")
        self._filter_matrix = None
        self._filter_exponent = filter_exponent
        self.build_filter_matrix(precomputed_data,
                                 axis_of_symmetry=axis_of_symmetry.strip().lower())

    def build_filter_matrix(self,
                            precomputed_data: polyplas_utils.PolyPlasPrecomputedData,
                            axis_of_symmetry: str = 'none'):
        start_time = perf_counter()
        
        nodal_coordinates = precomputed_data.nodal_coordinates.copy()
        if axis_of_symmetry == 'x':
            nodal_coordinates[:, 1] = np.abs(nodal_coordinates[:, 1])
        elif axis_of_symmetry == 'y':
            nodal_coordinates[:, 0] = np.abs(nodal_coordinates[:, 0])
        elif axis_of_symmetry == 'xy':
            nodal_coordinates[:, :] = np.abs(nodal_coordinates[:, :])
        elif axis_of_symmetry != 'none':
            raise ValueError(f"User specified symmetry about '{axis_of_symmetry}' axis "
                             "but only 'x', 'y', 'xy', and 'none' are valid options.")
        my_kd_tree = sp.spatial.KDTree(nodal_coordinates)
        csr_distance_matrix = my_kd_tree.sparse_distance_matrix(my_kd_tree,
                                                                self.filter_radius,
                                                                p=2.0,
                                                                output_type='coo_matrix').tocsr()
        csr_distance_matrix.data[:] *= (-1.0 / self.filter_radius)
        csr_distance_matrix.data[:] += 1.0
        csr_distance_matrix.data[:] **= self._filter_exponent
        row_sum = csr_distance_matrix.sum(axis=1).getA1() + 1.0e-10
        row_sum_inverse_matrix = sp.sparse.diags(1.0 / row_sum, offsets=0, format='csr')
        filter_matrix = row_sum_inverse_matrix @ csr_distance_matrix
        filter_matrix.eliminate_zeros()
        filter_matrix.sort_indices()
        elapsed_time = perf_counter() - start_time
        message = f"Polynomial filter matrix construction required {elapsed_time:0.2f} seconds."
        self.logger.info(message)
        self._filter_matrix = filter_matrix

    def apply_filter(self, design_variables: np.ndarray) -> np.ndarray:
        """
        Apply the polynomial filter to the design variables.

        Args:
            design_variables (np.ndarray): Array containing the design variables to be filtered.

        Returns:
            filtered_design_variables (np.ndarray): Array containing the filtered design variables.
        """
        return self._filter_matrix @ design_variables

    def apply_chain_rule(self, sensitivity_vector: np.ndarray) -> np.ndarray:
        """
        Apply the chain rule to the sensitivity vector.

        Args:
            sensitivity_vector (np.ndarray): Array containing the sensitivity w.r.t. the filtered design variables.

        Returns:
            np.ndarray: Array containing the sensitivity w.r.t. the unfiltered design variables.
        """
        return self._filter_matrix.T @ sensitivity_vector


#######################################################################################################################
#######################################################################################################################
class IdentityFilter(FilterBase):
    """
    An identity design variable filter which simply returns the unmodified design variables.
    """
    def __init__(self):
        super().__init__(filter_radius=1.0)
        self.type = 'identity'

    def apply_filter(self, design_variables: np.ndarray) -> np.ndarray:
        return design_variables.copy()

    def apply_chain_rule(self, sensitivity_vector: np.ndarray) -> np.ndarray:
        return sensitivity_vector.copy()