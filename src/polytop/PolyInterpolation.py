import logging
import numpy as np
from abc import ABC, abstractmethod


#######################################################################################################################
#######################################################################################################################
class MaterialInterpolationFunction(ABC):
    """Abstract base class for material interpolation functions.
    """

    def __init__(self, ersatz_parameter: float = 0.0, penalization_parameter: float = 0.0, name: str = ""):
        self.logger = logging.getLogger("Poly.Interpolation")
        self.name = name
        self.ersatz_parameter = ersatz_parameter
        self.penalization_parameter = penalization_parameter

    @abstractmethod
    def get_value(self, density_vector: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_derivative(self, density_vector: np.ndarray) -> np.ndarray:
        pass

    def update_logged_values(self, logged_values: dict) -> dict:
        logged_values.update({f"{self.name} Penalization Parameter": self.penalization_parameter,
                              f"{self.name} Ersatz Parameter": self.ersatz_parameter})
        return logged_values
    
    def update_logged_values_plastic(self, logged_values: dict) -> dict:
        logged_values.update({f"{self.name} Plastic Penalization Parameter": self.penalization_parameter,
                              f"{self.name} Plastic Ersatz Parameter": self.ersatz_parameter})
        return logged_values

    @property
    def ersatz_parameter(self) -> float:
        return self._ersatz_parameter

    @ersatz_parameter.setter
    def ersatz_parameter(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError("Ersatz parameter must be a number. "
                            f"User specified '{value}' which is of type '{type(value)}'.")
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Ersatz parameter cannot be < 0 or >= 1. User specified '{value}'.")
        if value == 0.0:
            message = f"{self.name} ersatz parameter is 0.0. This may cause numerical problems."
            self.logger.warning(message)
        self._ersatz_parameter = value

    @property
    def penalization_parameter(self) -> float:
        return self._penalization_parameter

    @penalization_parameter.setter
    def penalization_parameter(self, value: float):
        if not isinstance(value, (float, int)):
            raise TypeError("Penalization parameter must be a number. "
                            f"User specified '{value}' which is of type '{type(value)}'.")
        if not -1.0 < value <= 10.0:
            raise ValueError(f"Penalization parameter cannot be <= -1 or > 10. User specified '{value}'.")
        self._penalization_parameter = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"Name must be a string. User specified '{value}' which is of type '{type(value)}'.")
        self._name = value


#######################################################################################################################
#######################################################################################################################
class SIMP(MaterialInterpolationFunction):
    """SIMP material interpolation function.

    The SIMP interpolation function is defined as:
        :math:`f(\\rho) = \\epsilon + (1 - \\epsilon) \\rho^{p}`
    where :math:`\\epsilon` is the ersatz parameter and :math:`p` is the penalization parameter.
    """

    def __init__(self, ersatz_parameter: float = 1.0e-6, penalization_parameter: float = 1.0):
        super().__init__(ersatz_parameter=ersatz_parameter,
                         penalization_parameter=penalization_parameter,
                         name="SIMP")

    def get_value(self, density_vector: np.ndarray) -> np.ndarray:
        return self.ersatz_parameter + (1.0 - self.ersatz_parameter) * density_vector**self.penalization_parameter

    def get_derivative(self, density_vector: np.ndarray) -> np.ndarray:
        temp_scalar = self.penalization_parameter * (1.0 - self.ersatz_parameter)
        return temp_scalar * density_vector**(self.penalization_parameter - 1.0)


#######################################################################################################################
#######################################################################################################################
class RAMP(MaterialInterpolationFunction):
    """
    RAMP material interpolation function.

    The RAMP interpolation function is defined as:
        :math:`f(\\rho) = \\epsilon + (1 - \\epsilon) \\frac{\\rho}{1 + \\beta (1 - \\rho)}`
        where :math:`\\epsilon` is the ersatz parameter and :math:`\\beta` is the penalization parameter.
    """
    def __init__(self, ersatz_parameter: float = 1.0e-6, penalization_parameter: float = 0.0):
        super().__init__(ersatz_parameter=ersatz_parameter,
                         penalization_parameter=penalization_parameter,
                         name="RAMP")

    def get_value(self, density_vector: np.ndarray) -> np.ndarray:
        denominator = 1.0 + self.penalization_parameter * (1.0 - density_vector)
        interpolation_function = density_vector / denominator
        return self.ersatz_parameter + (1.0 - self.ersatz_parameter) * interpolation_function

    def get_derivative(self, density_vector: np.ndarray) -> np.ndarray:
        denominator_squared = (1.0 + self.penalization_parameter * (1.0 - density_vector))**2
        temp_scalar = (1.0 - self.ersatz_parameter) * (1.0 + self.penalization_parameter)
        interpolation_function_derivative = temp_scalar / denominator_squared
        return interpolation_function_derivative


#######################################################################################################################
#######################################################################################################################
def get_material_interpolation_function(material_interpolation_function_name: str = "SIMP",
                                        ersatz_parameter: float = 1.0e-6,
                                        penalization_parameter: float = 0.0) -> MaterialInterpolationFunction:
    """
    Returns a material interpolation function object.

    Args:
        material_interpolation_function_name (str): Name of the material interpolation function (e.g., SIMP, RAMP).
        ersatz_parameter (float): Ersatz parameter.
        penalization_parameter (float): Penalization parameter.

    Returns:
        MaterialInterpolationFunction: Material interpolation function object.
    """
    # upper case name
    function_name = material_interpolation_function_name.upper()
    available_functions = {"SIMP": SIMP(ersatz_parameter=ersatz_parameter,
                                        penalization_parameter=penalization_parameter),
                           "RAMP": RAMP(ersatz_parameter=ersatz_parameter,
                                        penalization_parameter=penalization_parameter)}
    if function_name not in available_functions:
        raise ValueError(f"Material interpolation function name '{material_interpolation_function_name}' is not valid."
                         f" Available function names are: {list(available_functions.keys())}")
    return available_functions[function_name]
