import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import src.polymesher.PolyGeometryAndBCs as polygeometry
import src.polymesher.PolyMesher as polymesher
import src.polyplas.PolyPlasUtilities as polyplas_utils
import src.polytop.PolyFilter as polyfilter
import src.polytop.PolyInterpolation as polyinterpolation
import src.polytop.PolyProjection as polyprojection
import src.polyplas.PolyPlas as polyplas
from src.polytop.PolyProjection import IdentityProjection, TanhHeavisideProjection

polyplas_utils.setup_logger(logging_level=logging.INFO)

volume_fraction_upper_bound = 0.4
number_of_elements = 20500
maximum_polymesher_iterations = 30

available_boundary_value_problems = {
    "HookDomain": polygeometry.HookDomain(),
    "SerpentineDomain": polygeometry.SerpentineDomain(),
    "WrenchDomain": polygeometry.WrenchDomain(),
    "MichellDomain": polygeometry.MichellDomain(),
    "MbbBeam": polygeometry.MbbBeam(),
    "CantileverBeam": polygeometry.CantileverBeam(),
    "CurvedBeam": polygeometry.CurvedBeam(),
    "Corbel": polygeometry.Corbel(),
    "Lbracket": polygeometry.Lbracket(),
    "CooksMembrane": polygeometry.CooksMembrane(),
    "UnitSquare": polygeometry.UnitSquare(),
    "PortalFrame": polygeometry.PortalFrame(),
    "HalfPortalFrame": polygeometry.HalfPortalFrame(),
    "HalfClampedBeam": polygeometry.HalfClampedBeam()
}

boundary_value_problem = available_boundary_value_problems["HalfPortalFrame"]
boundary_value_problem.applied_displacement_magnitude = 1.0

polymesh_object = polymesher.PolyMesher(
    boundary_value_problem,
    number_of_elements=number_of_elements,
    maximum_iterations=maximum_polymesher_iterations,
    use_regular_grid_if_implemented=False,
)
polymesh_object.plot()
plt.show()

material_parameters = dict(elastic_modulus=74633.0,
                        poissons_ratio = 0.3,
                        hardening_modulus = 2000.0, 
                        initial_yield_stress = 344.0) 
    
precomputed_data = polyplas_utils.get_precomputed_data_filter_and_heaviside_projection(polymesh_object, material_parameters)

design_variable_filter = polyfilter.get_design_variable_filter(
    polymesh_object,
    precomputed_data,
    filter_radius=1.0, 
    use_relative_filter_radius=False, # actual_filter_radius = filter_radius * largest_element_edge_length
    filter_type="polynomial", # Options: pde, pde_with_boundary_penalty, polynomial, and identity
    axis_of_symmetry='none' # symmetry about the specified axis. Options: x, y, xy, none (only implemented for polynomial filter)
)

# Determine the minimum and maximum projection strength based on the filter radius, filter type, and mesh size
_, largest_edge_length = polymesh_object.get_smallest_and_largest_edge_length()
maximum_projection_strength = 2.0 * design_variable_filter.filter_radius / largest_edge_length
using_pde_filter = "pde" in design_variable_filter.type
if using_pde_filter: maximum_projection_strength /= (3.0**0.5)
minimum_projection_strength = min(1.0, maximum_projection_strength)

projection_function = polyprojection.get_projection_function(
    projection_function_type="tanh", # Options: tanh, identity
    projection_strength=minimum_projection_strength,
    projection_threshold=0.5
)

elastic_material_interpolation_function = polyinterpolation.get_material_interpolation_function(
    material_interpolation_function_name="SIMP", # Options: SIMP, RAMP
    ersatz_parameter=1.0e-8,
    penalization_parameter=1.0
)

plastic_material_interpolation_function = polyinterpolation.get_material_interpolation_function(
    material_interpolation_function_name="SIMP", # Options: SIMP, RAMP
    ersatz_parameter=1.0e-4,
    penalization_parameter=0.5
)

polyplas_object = polyplas.PolyPlas(
    design_variable_filter=design_variable_filter,
    elastic_material_interpolation=elastic_material_interpolation_function,
    plastic_material_interpolation=plastic_material_interpolation_function,
    projection=projection_function,
    maximum_projection_strength=maximum_projection_strength,
    precomputed_data=precomputed_data,
    volume_fraction_upper_bound=volume_fraction_upper_bound,
    number_of_time_steps=14, 
    directory_name="HalfPortalFrame_Results"
)

design_variables = np.zeros((precomputed_data.number_of_nodes,), dtype=float)
design_variables[:] = volume_fraction_upper_bound
mma_move_limit = 0.5
number_of_design_variables = design_variables.size
MMA_state = polyplas_utils.MMA_Optimizer_State()
passive_design_variable_indices = precomputed_data.passive_design_variable_indices
MMA_state.set_number_of_design_variables(number_of_design_variables, volume_fraction_upper_bound, mma_move_limit, passive_design_variable_indices)
objective_function_value, applied_displacements_x, applied_displacements_y, reaction_forces  = polyplas_object.run_forward_analysis()
print(reaction_forces)

polyplas_object.run_top_opt_problem(MMA_state, maximum_optimization_iterations=300, convergence_tolerance=1.0e-8, check_grad=False)
polyplas_object.plot_density_field()

polyplas_object.plot_optimization_history(save_figure=False, save_data_filepath="logged_data.csv")
plt.show()





