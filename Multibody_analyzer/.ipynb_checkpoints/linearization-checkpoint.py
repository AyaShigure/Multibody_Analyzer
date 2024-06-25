from sympy import symbols, lambdify, Matrix, zeros
import numpy as np
from .basic_math import *
from .EMO_matrices import symbolical_to_numerical


# For linearization of underactuated systems
# Partial linearization with respect to generalized coordinates with control
def Partial_linearization_around_known_fixed_point(multibody_system, fixpoint_position_x_vec, fixpoint_control_u_vec, ground_reaction_lambda_vector_list, g_n=9.81):
    # IMPORTANT: Linearization but only about the actuated DOFs
    # calculate dx/dt = A_lin*x + B_lin*u
    # IMPORTANT: State vector in above equation is only [q_actuated_vec, q_dot_actuated_vec]

    system_dimension = multibody_system.system_dimension
    control_dimension = len(fixpoint_control_u_vec)
    uncontrol_dimension = system_dimension - control_dimension

    # Creat lists for indexing
    n_plus_m = list(range(0,system_dimension)) # Total DOF of the multibody system
    n = list(range(0,system_dimension - control_dimension)) # Unactuated DOFs
    m = list(range(control_dimension)) # Actuated DOFs
    
    # Coordinate vectors
    generalized_coordinates = multibody_system.system_generalized_coordinates
    actuated_generalized_coordinates = generalized_coordinates[len(n):] # Extract directly actuated generalized coordinates

    ######################################################### Part 1, Extract and prepare the symbolic matrices
    # Get manipulator equation components
    Mass_matrix = multibody_system.system_Mass_matrix_numerical
    Coriolis_vector = multibody_system.system_Coriolis_vector_numerical
    Gravitational_Force_vector = multibody_system.system_Gravitational_Force_vector_numerical
    Constraint_equation_object_list = multibody_system.system_constraint_equation_objects
    
    # Extract unactuated and actuated part from the manipulator equation
    Mass_matrix_unactuated_part = Mass_matrix.row(n)
    Mass_matrix_actuated_part = Mass_matrix.row(n_plus_m[len(n):]) # n_plus_m[len(n):] returns a list which first len(n) elements are excluded
    M_11 = Mass_matrix_unactuated_part.col(n)
    M_12 = Mass_matrix_unactuated_part.col(n_plus_m[len(n):])
    M_21 = Mass_matrix_actuated_part.col(n)
    M_22 = Mass_matrix_actuated_part.col(n_plus_m[len(n):])

    C_u = Coriolis_vector.row(n)
    C_a = Coriolis_vector.row(n_plus_m[len(n):])

    tau_g_u = Gravitational_Force_vector.row(n)
    tau_g_a = Gravitational_Force_vector.row(n_plus_m[len(n):])

    # Rank3 tensor and rank1 tensor calculation patch
    # ∂H_mat.T/∂q_vec * lambda_matrix
    # Note, here we convert the numpy matrix to sympy matrix, then extract the rows with .row() method, then convert result matrix back to numpy matrix.
    del_H_T_del_q_lambda_matrix = Matrix(del_H_del_q_lambda_tensor_calculation_patch(multibody_system, fixpoint_position_x_vec, fixpoint_control_u_vec, ground_reaction_lambda_vector_list, g_n=9.81))
    # Above function calculates jacobian of H matrices with respect to the full generalized coordinates.
    # We need to extract columns corresponding to actuated generalized coordinates.
    del_H_T_del_q_lambda_matrix = del_H_T_del_q_lambda_matrix.col(n_plus_m[len(n):]) 
    del_H_T_del_q_lambda_matrix_unactuated_part = np.matrix(del_H_T_del_q_lambda_matrix.row(n))
    del_H_T_del_q_lambda_matrix_actuated_part = np.matrix(del_H_T_del_q_lambda_matrix.row(n_plus_m[len(n):]))

    ######################################################### Part 2, Lambdify and insert constants to get np matrices
    t,g = symbols('t,g')
    # lambda function takes :((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
    # Note: fixpoint_position_x_vec contains both q and q_dot
    variables_to_insert = tuple( [0] + [g_n] + fixpoint_position_x_vec + fixpoint_control_u_vec)
    symbols_for_lambdify = ((t) , (g) , *tuple(multibody_system.system_state_vector) , *tuple(multibody_system.system_Control_force_vector))
    
    M_11_lambda = lambdify((symbols_for_lambdify), M_11)
    M_12_lambda = lambdify((symbols_for_lambdify), M_12)
    M_21_lambda = lambdify((symbols_for_lambdify), M_21)
    M_22_lambda = lambdify((symbols_for_lambdify), M_22)
    # Here we take jacobian of tau_g with respect to actuated DOF coordinates
    tau_g_u_jacobian_lambda = lambdify((symbols_for_lambdify), tau_g_u.jacobian(actuated_generalized_coordinates))
    tau_g_a_jacobian_lambda = lambdify((symbols_for_lambdify), tau_g_a.jacobian(actuated_generalized_coordinates))

    # Insert variables
    M_11 = np.matrix(M_11_lambda(*variables_to_insert))
    M_12 = np.matrix(M_12_lambda(*variables_to_insert))
    M_21 = np.matrix(M_21_lambda(*variables_to_insert))
    M_22 = np.matrix(M_22_lambda(*variables_to_insert))
    tau_g_u_jacobian = np.matrix(tau_g_u_jacobian_lambda(*variables_to_insert))
    tau_g_a_jacobian = np.matrix(tau_g_a_jacobian_lambda(*variables_to_insert))

    # Just a reminder that we need this result to construct the final linearization
    del_H_T_del_q_lambda_matrix_unactuated_part = del_H_T_del_q_lambda_matrix_unactuated_part 
    del_H_T_del_q_lambda_matrix_actuated_part = del_H_T_del_q_lambda_matrix_actuated_part
    # print(del_H_T_del_q_lambda_matrix_unactuated_part)
    
    # Calculate mass matrix and its inverse
    M_11_inv = np.linalg.inv(M_11)

    B_21_element_matrix = np.linalg.inv(M_22 - M_21 * M_11_inv * M_12)
    A_21_element_matrix = B_21_element_matrix * (tau_g_a_jacobian
                                                 + del_H_T_del_q_lambda_matrix_actuated_part
                                                 - M_21*M_11_inv * tau_g_u_jacobian
                                                 - M_21*M_11_inv * del_H_T_del_q_lambda_matrix_unactuated_part)
    
    A_lin = np.block([
                    [np.zeros([control_dimension, control_dimension])            ,    np.eye(control_dimension)],
                    [              A_21_element_matrix                           ,    np.zeros([control_dimension, control_dimension])]
                    ]).astype('float64')
    B_lin = np.block([
                    [np.zeros([control_dimension, control_dimension])],
                    [              B_21_element_matrix ]
                    ]).astype('float64')
    # return
    return A_lin, B_lin

# For linearization of underactuated systems
# For partial linearization calculations
def del_H_del_q_lambda_tensor_calculation_patch(multibody_system, fixpoint_position_x_vec, fixpoint_control_u_vec, ground_reaction_lambda_vector_list, g_n=9.81):
    # This function is only for a step during linearization of the manipulator equation
    # in which the product of a rank2, a rank3 and a rank1 tensor is needed
    system_dimension = multibody_system.system_dimension
    constraint_equation_list = multibody_system.system_constraint_equation_objects
    generalized_coordinates = multibody_system.system_generalized_coordinates
    
    ####################### Part 1, actual calculation of del_H_del_q_lambda matrices
    # Loop through constraint equations:
    del_H_del_q_lambda_matrix_list = []
    for i in range(len(constraint_equation_list)):
        # Extract jacobian of constraint equation, or H matrix
        dummy_H_matrix = constraint_equation_list[i].positional_constraint_equation_jacobian
        dummy_lambda_vector = Matrix(ground_reaction_lambda_vector_list[i])
        dummy_del_H_del_q_lambda_matrix = zeros(system_dimension)
        # Loop through row and column of the result matrix (del_H_del_q_lambda_product)
        for j in range(system_dimension):
            for k in range(system_dimension):
                matrix_element = dummy_H_matrix.col(j).T.diff(generalized_coordinates[k]) * dummy_lambda_vector
                dummy_del_H_del_q_lambda_matrix = change_matrix_element(dummy_del_H_del_q_lambda_matrix, j, k, matrix_element)
        del_H_del_q_lambda_matrix_list.append(dummy_del_H_del_q_lambda_matrix)
        
    
    ####################### Part 2, subs in constants, convert del_H_del_q_lambda_matrix to numpy matrix
    for i in range(len(del_H_del_q_lambda_matrix_list)):
        del_H_del_q_lambda_matrix_list[i] = symbolical_to_numerical(del_H_del_q_lambda_matrix_list[i], multibody_system.system_link_objects)
        for j in range(len(generalized_coordinates)):
            del_H_del_q_lambda_matrix_list[i] = del_H_del_q_lambda_matrix_list[i].subs(generalized_coordinates[j], fixpoint_position_x_vec[j])
        del_H_del_q_lambda_matrix_list[i] = np.matrix(del_H_del_q_lambda_matrix_list[i]).astype('float64')


    del_H_del_q_lambda_matrix = np.zeros([system_dimension, system_dimension])
    for i in range(len(del_H_del_q_lambda_matrix_list)):
        del_H_T_del_q_lambda_matrix = del_H_del_q_lambda_matrix + del_H_del_q_lambda_matrix_list[i]

    return del_H_T_del_q_lambda_matrix
    



# For fully actuated systems
def Linearization_around_known_fixed_point(multibody_system, fixpoint_position_x_vec, fixpoint_control_u_vec, ground_reaction_lambda_vector_list, g_n=9.81):
    # calculate dx/dt = A_lin*x + B_lin*u

    system_dimension = multibody_system.system_dimension
    control_dimension = len(fixpoint_control_u_vec)
    Bm = np.matrix(multibody_system.system_Control_matrix)
    # lambda function takes :((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
    # Note: fixpoint_position_x_vec contains both q and q_dot
    variables_to_insert = tuple( [0] + [g_n] + fixpoint_position_x_vec + fixpoint_control_u_vec)
    
    # Calculate mass matrix and its inverse
    M_lambda = multibody_system.system_Mass_Matrix_lambda
    M = np.matrix(M_lambda(*variables_to_insert))
    M_inv = np.linalg.inv(M)

    # Calculate jacobian of tau_g, lambdify and insert x_0 and u_0
    tau_g = multibody_system.system_Gravitational_Force_vector_numerical
    generalized_coordinates = multibody_system.system_generalized_coordinates
    tau_g_jacobian = tau_g.jacobian(generalized_coordinates)

    t,g = symbols('t,g')
    symbols_for_lambdify = ((t) , (g) , *tuple(multibody_system.system_state_vector) , *tuple(multibody_system.system_Control_force_vector))
    tau_g_jacobian_lambda = lambdify((symbols_for_lambdify), tau_g_jacobian)
    tau_g_jacobian = np.matrix(tau_g_jacobian_lambda(*variables_to_insert))
    
    M_inv_del_H_del_q_lambda_matrix = M_inv_del_H_del_q_lambda_calculation_patch(multibody_system, fixpoint_position_x_vec, fixpoint_control_u_vec, ground_reaction_lambda_vector_list, g_n=9.81)
    A_lin = np.block([
                    [np.zeros([system_dimension, system_dimension])            ,    np.eye(system_dimension)],
                    [M_inv * tau_g_jacobian + M_inv_del_H_del_q_lambda_matrix  ,    np.zeros([system_dimension, system_dimension])]
                    ])
    B_lin = np.block([
                    [np.zeros([system_dimension, control_dimension])],
                    [M_inv * Bm]
                    ])
    return A_lin, B_lin

# For linearization of fully actuated systems
def M_inv_del_H_del_q_lambda_calculation_patch(multibody_system, fixpoint_position_x_vec, fixpoint_control_u_vec, ground_reaction_lambda_vector_list, g_n=9.81):
    # This function is only for a step during linearization of the manipulator equation
    # in which the product of a rank2, a rank3 and a rank1 tensor is needed
    system_dimension = multibody_system.system_dimension
    constraint_equation_list = multibody_system.system_constraint_equation_objects
    generalized_coordinates = multibody_system.system_generalized_coordinates

    ####################### Part 1, actual calculation of del_H_del_q_lambda matrices
    # Loop through constraint equations:
    del_H_del_q_lambda_matrix_list = []
    for i in range(len(constraint_equation_list)):
        # Extract jacobian of constraint equation, or H matrix
        dummy_H_matrix = constraint_equation_list[i].positional_constraint_equation_jacobian
        dummy_lambda_vector = Matrix(ground_reaction_lambda_vector_list[i])
        dummy_del_H_del_q_lambda_matrix = zeros(system_dimension)
        # Loop through row and column of the result matrix (del_H_del_q_lambda_product)
        for j in range(system_dimension):
            for k in range(system_dimension):
                matrix_element = dummy_H_matrix.col(j).T.diff(generalized_coordinates[k]) * dummy_lambda_vector
                dummy_del_H_del_q_lambda_matrix = change_matrix_element(dummy_del_H_del_q_lambda_matrix, j, k, matrix_element)
        del_H_del_q_lambda_matrix_list.append(dummy_del_H_del_q_lambda_matrix)
        
    
    ####################### Part 2, subs in constants, convert del_H_del_q_lambda_matrix to numpy matrix
    for i in range(len(del_H_del_q_lambda_matrix_list)):
        del_H_del_q_lambda_matrix_list[i] = symbolical_to_numerical(del_H_del_q_lambda_matrix_list[i], multibody_system.system_link_objects)
        for j in range(len(generalized_coordinates)):
            del_H_del_q_lambda_matrix_list[i] = del_H_del_q_lambda_matrix_list[i].subs(generalized_coordinates[j], fixpoint_position_x_vec[j])
        del_H_del_q_lambda_matrix_list[i] = np.matrix(del_H_del_q_lambda_matrix_list[i]).astype('float64')

    
    ####################### Part 3, calculate full M_inv * Σ(del_Hi_del_q_lambdai_matrices)
    # lambda function takes :((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
    Mass_Matrix_lambda = multibody_system.system_Mass_Matrix_lambda
    variables_to_insert = tuple( [0] + [g_n] + fixpoint_position_x_vec + fixpoint_control_u_vec)
    M = np.matrix(Mass_Matrix_lambda(*variables_to_insert))
    M_inv = np.linalg.inv(M)

    del_H_del_q_lambda_matrix = np.zeros([system_dimension, system_dimension])
    for i in range(len(del_H_del_q_lambda_matrix_list)):
        del_H_del_q_lambda_matrix = del_H_del_q_lambda_matrix + del_H_del_q_lambda_matrix_list[i]
        
    M_inv_del_H_del_q_lambda_matrix = M_inv * del_H_del_q_lambda_matrix
    return M_inv_del_H_del_q_lambda_matrix
    
