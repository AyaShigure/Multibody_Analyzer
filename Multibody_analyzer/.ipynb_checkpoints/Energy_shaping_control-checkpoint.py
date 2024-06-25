from sympy import lambdify
from .basic_math import *
from .EMO_matrices import *

# Kinetic energy shaping control
class energy_shaping_controller():
    def __init__(self, 
                 multibody_system, 
                 target_state_vector, 
                 stable_u_vec_from_lqr_control,
                 g_n = 9.81):
        
        self.multibody_system = multibody_system
        self.generalized_coordinates = multibody_system.system_generalized_coordinates
        self.g = multibody_system.g
        self.t = multibody_system.t
        self.g_n = g_n
        self.stable_u_vec = stable_u_vec_from_lqr_control
        self.controller_gain = 1000
        # self.u_max = 50 # Motor max/saturation output
        # self.energy_shaping_control_gain
        
        ############ Get numerical gravitational acceleration vector
        self.system_gravitational_acc_vector = multibody_system.system_gravitational_acc_vector
        self.system_gravitational_acc_vector_numerical = self.system_gravitational_acc_vector.subs(self.g, g_n)
    
        ############# Get lambda CoG_vector_list
        self.system_link_object_list = multibody_system.system_link_objects
        self.system_state_vector = multibody_system.system_state_vector
        self.system_link_CoG_vector_lambda_list = [] # CoG position in world frame
        self.system_link_CoG_velocity_lambda_list = [] # CoG velocity in world frame
        for link in self.system_link_object_list:
            # CoG vector
            dummy_CoG_vector_lambda = symbolical_to_numerical(link.link_CoG_vector, self.system_link_object_list) # Insert constants 
            dummy_CoG_vector_lambda = lambdify(tuple(self.system_state_vector), dummy_CoG_vector_lambda) # Lambdify
            self.system_link_CoG_vector_lambda_list.append(dummy_CoG_vector_lambda)
            # CoG velocity
            dummy_CoG_velocity_lambda = diff(link.link_CoG_vector, self.t) # Get velocity vector
            dummy_CoG_velocity_lambda = symbolical_to_numerical(dummy_CoG_velocity_lambda, self.system_link_object_list) # Insert constants 
            dummy_CoG_velocity_lambda = lambdify(tuple(self.system_state_vector), dummy_CoG_velocity_lambda) # Lambdify
            self.system_link_CoG_velocity_lambda_list.append(dummy_CoG_velocity_lambda)
            

        ############# Get lambda CoG_vector_list
        self.target_state_vector = target_state_vector
        
        (self.target_mechanical_energy, 
         self.target_kinetic_energy, 
         self.target_potential_energy) = calculate_mechanical_energy(self.multibody_system, 
                                                                     self.target_state_vector, 
                                                                     self.system_link_CoG_vector_lambda_list, 
                                                                     self.system_gravitational_acc_vector_numerical)
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(bcolors.BOLD + '                                 Energy shaping controller is created.' + bcolors.ENDC)
        print(bcolors.BOLD + '                                 Controller used: u = -k_gain * q_2_dot' + bcolors.ENDC)
        print(bcolors.BOLD + '                                 Controller gain: k_gain = {}'.format(self.controller_gain) + bcolors.ENDC)
        print(bcolors.FAIL + '=================================================================================================\n' + bcolors.ENDC)

    # Change max/saturation output of each motor
    def set_saturation_value(self, saturation_value):
        self.u_max = saturation_value
        print(bcolors.BOLD + 'Motor saturation output is set to {} Nm'.format(self.u_max) + bcolors.ENDC)

    def set_controller_gain(self, new_gain):
        self.controller_gain = new_gain
        print(bcolors.BOLD + 'Controller gain is set to k_gain = {}'.format(self.controller_gain) + bcolors.ENDC)


    def get_target_energy(self):
        return self.target_mechanical_energy, self.target_kinetic_energy, self.target_potential_energy

         
    def get_energy_shaping_control_vector_KE_control(self, current_state_vector):

        ############## Extract current generalized coordinates and generalized velocities
        generalized_coordinates_indexer = list(range(self.multibody_system.system_dimension))
        generalized_velocities_indexer = list(range(self.multibody_system.system_dimension * 2))[self.multibody_system.system_dimension :]
        # Extract vectors with indexers
        current_generalized_coordinate_vector = np.array(current_state_vector)[generalized_coordinates_indexer]
        current_generalized_vecloity_vector = np.array(current_state_vector)[generalized_velocities_indexer]
        # Convert to column vectors
        current_generalized_coordinate_vector = np.transpose(np.matrix(current_generalized_coordinate_vector).astype('float64'))
        current_generalized_vecloity_vector = np.transpose(np.matrix(current_generalized_vecloity_vector).astype('float64'))
        
        ### Generalized velocity vector indexer (from state_vector)
        unactuated_dimension = self.multibody_system.system_dimension - self.multibody_system.system_Control_matrix.rank()
        actuated_dimension = self.multibody_system.system_Control_matrix.rank()
        system_dimension = self.multibody_system.system_dimension
        actuated_generalized_velocity_vector_indexer = list(range(system_dimension * 2)[(system_dimension + unactuated_dimension):])
        ############################################# 
        ############################################# Prepare to calculate Scaler S 
        # Get lambda functions
        Mass_Matrix_lambda = self.multibody_system.system_Mass_Matrix_lambda
        Coriolis_Vector_lambda = self.multibody_system.system_Coriolis_Vector_lambda
        Gravtational_Force_Vector_lambda = self.multibody_system.system_Gravitational_Force_Vector_lambda
        constraint_equation_lambda = self.multibody_system.constraint_equation_lambda # h vector
        constraint_jacobian_lambda = self.multibody_system.constraint_jacobian_lambda # H matrix (∂h_vec/∂q_vec, Jacobian of h_vec to generalized coordinates)
        constraint_jacobian_time_derivative_lambda = self.multibody_system.constraint_jacobian_time_derivative_lambda # dH/dt Matrix

        # Variables to insert into lambda functions
        t = 0
        g = self.g_n
        Control_force_vector =  np.zeros(self.multibody_system.system_Control_matrix.rank()) # dummy 0 control
        state_vector_dummy = current_state_vector
        # Insert state vector into lambda functions, (Calculate current)

        Mass_Matrix = np.matrix(Mass_Matrix_lambda(t, g, *state_vector_dummy, *Control_force_vector))
        Mass_Matrix_inv = np.linalg.inv(Mass_Matrix)        
        Coriolis_Vector = np.matrix(Coriolis_Vector_lambda(t, g, *state_vector_dummy,  *Control_force_vector))
        Gravtational_Force_Vector = np.matrix(Gravtational_Force_Vector_lambda(t, g, *state_vector_dummy,  *Control_force_vector))
        BSP_alpha = self.multibody_system.BSP_alpha
        q_dot_vec_current = current_generalized_vecloity_vector
        
        # clear the constraint force in generalized coordinate
        constraint_force_in_q_coord = []
        for i in range(len(self.multibody_system.system_generalized_coordinates)):
            constraint_force_in_q_coord.append(0)
        constraint_force_in_q_coord = np.transpose(np.matrix(constraint_force_in_q_coord))

        # Calculate i-th constraint force and sum them together in the end
        # Notice that here constraint_force_in_q_coord is the completed ΣHi^T*lambda_i_vec term
        for i in range(len(self.multibody_system.system_constraint_equation_objects)):
            h_Vec =  np.matrix(constraint_equation_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
            H_Mat = np.matrix(constraint_jacobian_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
            H_dot_Mat = np.matrix(constraint_jacobian_time_derivative_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
            
            # constraint_force_in_xyz is constraint force lambda
            # constraint_force_in_xyz = -np.linalg.pinv(H_Mat * Mass_Matrix_inv *np.transpose(H_Mat)) * (H_Mat * Mass_Matrix_inv * (Gravtational_Force_Vector + Control_Matrix * Control_vector.T - Coriolis_Vector) + (2*BSP_alpha*H_Mat + H_dot_Mat)*q_dot_vec_current + BSP_alpha**2 * h_Vec)
            # constraint_force_in_xyz is constraint force lambda (Workaround, assume thate constraint force does not depend on control input)
            constraint_force_in_xyz = -np.linalg.pinv(H_Mat * Mass_Matrix_inv *np.transpose(H_Mat)) * (H_Mat * Mass_Matrix_inv * (Gravtational_Force_Vector - Coriolis_Vector) + (2*BSP_alpha*H_Mat + H_dot_Mat)*q_dot_vec_current + BSP_alpha**2 * h_Vec)
            # constraint_force_in_q_coord is (H_mat^T * lambda_vec)
            constraint_force_in_q_coord = constraint_force_in_q_coord + np.transpose(H_Mat) * constraint_force_in_xyz



        ############################################## 
        ############################################## Calculate S scaler
        # S = -(current potential energy) - q_dot_vec.T * (tau_g - C + Σ H.T * lambda_vec)
        # S = S1 + S2
        # S1 = -(current potential energy)
        # S2 = - q_dot_vec.T * (tau_g - C + Σ H.T * lambda_vec)
            
        S = - np.transpose(current_generalized_vecloity_vector) * (Gravtational_Force_Vector - Coriolis_Vector + constraint_force_in_q_coord)

        ############################# Controller 
        # Calculate current kinetic energy
        current_kinetic_energy = calculate_mechanical_energy(self.multibody_system, 
                                                             current_state_vector, 
                                                             self.system_link_CoG_vector_lambda_list, 
                                                             self.system_gravitational_acc_vector_numerical)         
            
        current_actuated_generalized_velocity_vector = np.matrix(np.array(state_vector_dummy)[actuated_generalized_velocity_vector_indexer]).astype('float64')
        # compair energy to choose controller
        Control_matrix = np.matrix(self.multibody_system.system_Control_matrix)


        #######################################################################################################################
        # Here we set controller as u_vel =  -(gene_vel - target_gene_vel) * k_gain = -gene_vel * k_gain
        # k_gain = (-S) / (current_actuated_generalized_velocity_vector * np.transpose(current_actuated_generalized_velocity_vector))
        # k_gain = 1.1 * k_gain # This line is to lyapunov stable condition 
        
        # Constant k_gain: !! Gain should be a constant
        k_gain = self.controller_gain

        u_vec = - k_gain * current_actuated_generalized_velocity_vector
        #######################################################################################################################


        # #######################################################################################################################
        # # print('Energy shaper')                                                        ## u = -k*u_stable 
        # # print(np.transpose(u_vec))
        # # print(shape(current_actuated_generalized_velocity_vector))
        # # print(shape(np.transpose(self.stable_u_vec)))
        Lyapunov_stable_condition = S / (-(current_actuated_generalized_velocity_vector * np.transpose(current_actuated_generalized_velocity_vector)))
        # if Lyapunov_stable_condition > 1e+5:
        #     Lyapunov_stable_condition = 1e+5
        # if q2_T_u_stable > 0:
        #     k_gain = (S - 10. * abs(S)) / (q2_T_u_stable)
        # else:
        #     k_gain = (S + 10. * abs(S)) / (q2_T_u_stable)
        # u_vec = - k_gain * self.stable_u_vec
        # #######################################################################################################################

        # Problem with using u_stable_vec: actuator act to cancel some of the motion of q_dot with different sign,
        # But helping to pump energy into some other q_dot which has the same sine with u

        # Things to try next: 1. Really add sum(Hi * lambda_i) term to the lyapunov stable condition
                            # 2. Not energy shaping, but just a velocity difference {-(q_2_dot) * kgain} controller which meets Lyapunov stable condition
        

        return np.transpose(u_vec), current_kinetic_energy, Lyapunov_stable_condition

    def get_energy_shaping_control_vector(self, current_state_vector):

        ############## Extract current generalized coordinates and generalized velocities
        generalized_coordinates_indexer = list(range(self.multibody_system.system_dimension))
        generalized_velocities_indexer = list(range(self.multibody_system.system_dimension * 2))[self.multibody_system.system_dimension :]
        # Extract vectors with indexers
        current_generalized_coordinate_vector = np.array(current_state_vector)[generalized_coordinates_indexer]
        current_generalized_vecloity_vector = np.array(current_state_vector)[generalized_velocities_indexer]
        # Convert to column vectors
        current_generalized_coordinate_vector = np.transpose(np.matrix(current_generalized_coordinate_vector).astype('float64'))
        current_generalized_vecloity_vector = np.transpose(np.matrix(current_generalized_vecloity_vector).astype('float64'))

        ############################################# 
        ############################################# Prepare to calculate Scaler S 
        # Get lambda functions
        Mass_Matrix_lambda = self.multibody_system.system_Mass_Matrix_lambda
        Coriolis_Vector_lambda = self.multibody_system.system_Coriolis_Vector_lambda
        Gravtational_Force_Vector_lambda = self.multibody_system.system_Gravitational_Force_Vector_lambda
        constraint_equation_lambda = self.multibody_system.constraint_equation_lambda # h vector
        constraint_jacobian_lambda = self.multibody_system.constraint_jacobian_lambda # H matrix (∂h_vec/∂q_vec, Jacobian of h_vec to generalized coordinates)
        constraint_jacobian_time_derivative_lambda = self.multibody_system.constraint_jacobian_time_derivative_lambda # dH/dt Matrix

        # Variables to insert into lambda functions
        t = 0
        g = self.g_n
        Control_force_vector =  np.zeros(self.multibody_system.system_Control_matrix.rank()) # dummy 0 control
        state_vector_dummy = current_state_vector
        # Insert state vector into lambda functions, (Calculate current)

        Mass_Matrix = np.matrix(Mass_Matrix_lambda(t, g, *state_vector_dummy, *Control_force_vector))
        Mass_Matrix_inv = np.linalg.inv(Mass_Matrix)        
        Coriolis_Vector = np.matrix(Coriolis_Vector_lambda(t, g, *state_vector_dummy,  *Control_force_vector))
        Gravtational_Force_Vector = np.matrix(Gravtational_Force_Vector_lambda(t, g, *state_vector_dummy,  *Control_force_vector))
        BSP_alpha = self.multibody_system.BSP_alpha
        q_dot_vec_current = current_generalized_vecloity_vector
        
        # clear the constraint force in generalized coordinate
        constraint_force_in_q_coord = []
        for i in range(len(self.multibody_system.system_generalized_coordinates)):
            constraint_force_in_q_coord.append(0)
        constraint_force_in_q_coord = np.transpose(np.matrix(constraint_force_in_q_coord))

        # Calculate i-th constraint force and sum them together in the end
        # Notice that here constraint_force_in_q_coord is the completed ΣHi^T*lambda_i_vec term
        for i in range(len(self.multibody_system.system_constraint_equation_objects)):
            h_Vec =  np.matrix(constraint_equation_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
            H_Mat = np.matrix(constraint_jacobian_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
            H_dot_Mat = np.matrix(constraint_jacobian_time_derivative_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
            
            # constraint_force_in_xyz is constraint force lambda
            # constraint_force_in_xyz = -np.linalg.pinv(H_Mat * Mass_Matrix_inv *np.transpose(H_Mat)) * (H_Mat * Mass_Matrix_inv * (Gravtational_Force_Vector + Control_Matrix * Control_vector.T - Coriolis_Vector) + (2*BSP_alpha*H_Mat + H_dot_Mat)*q_dot_vec_current + BSP_alpha**2 * h_Vec)
            # constraint_force_in_xyz is constraint force lambda (Workaround, assume thate constraint force does not depend on control input)
            constraint_force_in_xyz = -np.linalg.pinv(H_Mat * Mass_Matrix_inv *np.transpose(H_Mat)) * (H_Mat * Mass_Matrix_inv * (Gravtational_Force_Vector - Coriolis_Vector) + (2*BSP_alpha*H_Mat + H_dot_Mat)*q_dot_vec_current + BSP_alpha**2 * h_Vec)
            # constraint_force_in_q_coord is (H_mat^T * lambda_vec)
            constraint_force_in_q_coord = constraint_force_in_q_coord + np.transpose(H_Mat) * constraint_force_in_xyz



        ############################################## 
        ############################################## Calculate S scaler
        # S = -(current potential energy) - q_dot_vec.T * (tau_g - C + Σ H.T * lambda_vec)
        # S = S1 + S2
        # S1 = -(current potential energy)
        # S2 = - q_dot_vec.T * (tau_g - C + Σ H.T * lambda_vec)
            
        current_potential_energy_dot = 0
        for i in range(len(self.system_link_CoG_velocity_lambda_list)):
            link_potential_energy_dot =  np.transpose(self.system_link_CoG_velocity_lambda_list[i](*tuple(state_vector_dummy))) * np.matrix(self.system_gravitational_acc_vector_numerical) * self.multibody_system.system_link_objects[i].link_Mass_numerical
            current_potential_energy_dot += link_potential_energy_dot
        S1 = current_potential_energy_dot
        S2 = - np.transpose(current_generalized_vecloity_vector) * (Gravtational_Force_Vector - Coriolis_Vector + constraint_force_in_q_coord)
        S = S1 + S2

        ############################# Controller 
        # Calculate current energy
        (current_mechanical_energy, 
            current_kinetic_energy, 
            current_potential_energy) = calculate_mechanical_energy(self.multibody_system, 
                                                                    current_state_vector, 
                                                                    self.system_link_CoG_vector_lambda_list, 
                                                                    self.system_gravitational_acc_vector_numerical)         


        u_vec = None
        # print('Energy shaper')
        # print(np.transpose(u_vec))

        current_energy_error = current_mechanical_energy - self.target_mechanical_energy
        return np.transpose(u_vec), current_mechanical_energy, current_kinetic_energy, current_potential_energy, current_energy_error










# def calculate_kinetic_energy(multibody_system, 
#                             current_state_vector, 
#                             system_link_CoG_vector_lambda_list,
#                             system_gravitational_acc_vector_numerical,
#                             g_n = 9.81):
    
#     Mass_Matrix_lambda = multibody_system.system_Mass_Matrix_lambda
#     system_link_object = multibody_system.system_link_objects
#     ############## Extract current generalized coordinates and generalized velocities
#     generalized_coordinates_indexer = list(range(multibody_system.system_dimension))
#     generalized_velocities_indexer = list(range(multibody_system.system_dimension * 2))[multibody_system.system_dimension :]
#     # Extract vectors with indexers
#     current_generalized_coordinate_vector = np.array(current_state_vector)[generalized_coordinates_indexer]
#     current_generalized_vecloity_vector = np.array(current_state_vector)[generalized_velocities_indexer]
#     # Convert to column vectors
#     current_generalized_coordinate_vector = np.transpose(np.matrix(current_generalized_coordinate_vector).astype('float64'))
#     current_generalized_vecloity_vector = np.transpose(np.matrix(current_generalized_vecloity_vector).astype('float64'))
#     # print('current_generalized_coordinate_vector')
#     # print(np.transpose(current_generalized_coordinate_vector).tolist()[0])
#     # print('\ncurrent_generalized_vecloity_vector')
#     # print(np.transpose(current_generalized_vecloity_vector).tolist()[0])

#     #######################################################
#     ####################################################### Kinetic energy
#     # Calculate current mass matrix
#     system_dimension = multibody_system.system_dimension
#     # lambda function takes :((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
#     dummy_velocites = np.zeros(system_dimension)
#     # All matrices are lambdified with control, matrices below are control independent, Remove control force from all lambda function later!!!!!
#     dummy_control = np.zeros(multibody_system.system_Control_matrix.rank())
#     variables_to_insert = tuple( [0] + [g_n] + list(np.transpose(current_generalized_coordinate_vector).tolist()[0]) + list(dummy_velocites) + list(dummy_control))
#     M = np.matrix(Mass_Matrix_lambda(*variables_to_insert))

#     ############## Calculate kinetic energy
#     current_kinetic_energy = 0.5 * np.transpose(current_generalized_vecloity_vector) * M * current_generalized_vecloity_vector

#     return current_kinetic_energy
    


def calculate_mechanical_energy(multibody_system, 
                                current_state_vector, 
                                system_link_CoG_vector_lambda_list,
                                system_gravitational_acc_vector_numerical,
                                g_n = 9.81):
    
    Mass_Matrix_lambda = multibody_system.system_Mass_Matrix_lambda
    system_link_object = multibody_system.system_link_objects
    ############## Extract current generalized coordinates and generalized velocities
    generalized_coordinates_indexer = list(range(multibody_system.system_dimension))
    generalized_velocities_indexer = list(range(multibody_system.system_dimension * 2))[multibody_system.system_dimension :]
    # Extract vectors with indexers
    current_generalized_coordinate_vector = np.array(current_state_vector)[generalized_coordinates_indexer]
    current_generalized_vecloity_vector = np.array(current_state_vector)[generalized_velocities_indexer]
    # Convert to column vectors
    current_generalized_coordinate_vector = np.transpose(np.matrix(current_generalized_coordinate_vector).astype('float64'))
    current_generalized_vecloity_vector = np.transpose(np.matrix(current_generalized_vecloity_vector).astype('float64'))
    # print('current_generalized_coordinate_vector')
    # print(np.transpose(current_generalized_coordinate_vector).tolist()[0])
    # print('\ncurrent_generalized_vecloity_vector')
    # print(np.transpose(current_generalized_vecloity_vector).tolist()[0])

    #######################################################
    ####################################################### Kinetic energy
    # Calculate current mass matrix
    system_dimension = multibody_system.system_dimension
    # lambda function takes :((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
    dummy_velocites = np.zeros(system_dimension)
    # All matrices are lambdified with control, matrices below are control independent, Remove control force from all lambda function later!!!!!
    dummy_control = np.zeros(multibody_system.system_Control_matrix.rank())
    variables_to_insert = tuple( [0] + [g_n] + list(np.transpose(current_generalized_coordinate_vector).tolist()[0]) + list(dummy_velocites) + list(dummy_control))
    M = np.matrix(Mass_Matrix_lambda(*variables_to_insert))

    ############## Calculate kinetic energy
    current_kinetic_energy = 0.5 * np.transpose(current_generalized_vecloity_vector) * M * current_generalized_vecloity_vector


    
    #######################################################
    ####################################################### Potential energy
    # Calculate numeric CoG vectors 
    link_CoG_vector_numerical_list = []
    for link_CoG_lambda in system_link_CoG_vector_lambda_list:
        dummy_cog_vector = np.transpose(np.matrix(link_CoG_lambda(*tuple(current_state_vector)))) # Column vector
        link_CoG_vector_numerical_list.append(dummy_cog_vector)

    # Potential energy
    current_potential_energy = 0
    # for i in range(len(link_CoG_vector_numerical_list)):
    #     link_potential_energy = link_CoG_vector_numerical_list[i] * np.matrix(system_gravitational_acc_vector_numerical) * system_link_objects[i].link_Mass_numerical
    #     current_potential_energy += link_potential_energy

    for i in range(len(system_link_CoG_vector_lambda_list)):
        link_potential_energy = -np.transpose(system_link_CoG_vector_lambda_list[i](*tuple(current_state_vector))) * np.matrix(system_gravitational_acc_vector_numerical) * multibody_system.system_link_objects[i].link_Mass_numerical
        current_potential_energy += link_potential_energy

    
    #######################################################
    ####################################################### Total mechanical energy
    current_energy = current_kinetic_energy  + current_potential_energy
    return current_energy, current_kinetic_energy, current_potential_energy


