import numpy as np
from abc import ABC, abstractmethod


#######################################################################################################################
#######################################################################################################################
class ProjectionBase(ABC):
    '''
    Abstract base class for projection functions. These functions are used to potentially project the filtered
    design variables to densities that are closer to 0 or 1.
    '''
    def __init__(self, projection_strength: float = 1.0, projection_threshold: float = 0.5, name: str = ""):
        self.projection_strength  = projection_strength
        self.projection_threshold = projection_threshold
        self.name = name

    @abstractmethod
    def apply_projection(self, filtered_densities: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def apply_chain_rule(self, sensitivity_vector: np.ndarray) -> np.ndarray:
        pass

    def update_logged_values(self, logged_values: dict) -> dict:
        projection_parameters = {f"{self.name} Projection Strength": self.projection_strength,
                                 f"{self.name} Projection Threshold": self.projection_threshold}
        logged_values.update(projection_parameters)
        return logged_values

    @property
    def projection_strength(self) -> float:
        return self._projection_strength

    @projection_strength.setter
    def projection_strength(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError(f"The projection strength must be a number. User supplied '{value}'.")
        if not value >= 0.0:
            raise ValueError(f"The projection strength must be >= 0. User supplied '{value}'.")
        self._projection_strength = value

    @property
    def projection_threshold(self) -> float:
        return self._projection_threshold

    @projection_threshold.setter
    def projection_threshold(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError(f"The projection threshold must be a number. User supplied '{value}'.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"The projection threshold must be between 0 and 1. User supplied '{value}'.")
        self._projection_threshold = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"The name must be a string. Value supplied was '{value}' of type '{type(value)}'.")
        self._name = value


#######################################################################################################################
#######################################################################################################################
class TanhHeavisideProjection(ProjectionBase):
    '''The hyperbolic tangent projection function.

    This class represents the smooth Heaviside hyperbolic tangent projection. The projection function is given by

    .. math::
        \\rho(\\rho^*) = \\frac{\\tanh(\\beta(\\rho^* - \\eta)) + \\tanh(\\beta\\eta)}
                               {\\tanh(\\beta(\\rho^* - \\eta)) + \\tanh(\\beta(1 - \\eta))}

    where :math:`\\rho^*` is the filtered design variable, :math:`\\eta` is the projection threshold,
    and :math:`\\beta` is the projection strength.
    '''
    def __init__(self, projection_strength: float = 1.0, projection_threshold: float = 0.5):
        super().__init__(projection_strength=projection_strength,
                         projection_threshold=projection_threshold,
                         name="Tanh Heaviside")
        self.projection_chain_rule = np.zeros((1,))

    def apply_projection(self, filtered_densities: np.ndarray) -> np.ndarray:
        denominator = np.tanh(self.projection_strength * self.projection_threshold) + \
                      np.tanh(self.projection_strength * (1.0 - self.projection_threshold))
        temp = np.tanh(self.projection_strength * (filtered_densities - self.projection_threshold)).ravel()
        numerator = np.tanh(self.projection_strength * self.projection_threshold) + temp
        projected_filtered_densities = numerator / denominator
        self.projection_chain_rule = (self.projection_strength / denominator) * ( 1.0 - temp**2 )
        return projected_filtered_densities

    def apply_chain_rule(self, sensitivity_vector: np.ndarray) -> np.ndarray:
        return sensitivity_vector.ravel() * self.projection_chain_rule.ravel()


#######################################################################################################################
#######################################################################################################################
class IdentityProjection(ProjectionBase):
    '''The identity projection function.

    This class represents the identity projection function. The projection function is given by

    .. math::
        \\rho = \\rho^*

    where :math:`\\rho^*` is the filtered design variable.
    '''
    def __init__(self):
        super().__init__(projection_strength=0.0,
                         projection_threshold=0.0,
                         name="Identity")

    def apply_projection(self, filtered_densities: np.ndarray) -> np.ndarray:
        return filtered_densities.copy()

    def apply_chain_rule(self, sensitivity_vector: np.ndarray) -> np.ndarray:
        return sensitivity_vector.copy()


#######################################################################################################################
#######################################################################################################################
def get_projection_function(projection_function_type: str = "",
                            projection_strength: float = 1.0,
                            projection_threshold: float = 0.5) -> ProjectionBase:
    '''
    Returns a projection function object.

    Args:
        projection_function_type (str): The type of the projection function to use. Options are 'tanh' and 'identity'.
        projection_strength (float): The strength of the projection function. Default is 1.0.
        projection_threshold (float): The threshold of the projection function. Default is 0.5.

    Returns:
        ProjectionBase: The projection function object.
    '''
    function_type = projection_function_type.lower()

    available_functions = {"tanh": TanhHeavisideProjection(projection_strength=projection_strength,
                                                           projection_threshold=projection_threshold),
                           "identity": IdentityProjection()}
    if function_type not in available_functions:
        raise ValueError(f"Unknown projection function type '{projection_function_type}'."
                         f"Available options are {list(available_functions.keys())}.")

    return available_functions[function_type]
