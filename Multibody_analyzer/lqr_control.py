import control as ct
from .basic_math import *
from .EMO_matrices import symbolical_to_numerical
from .linearization import *

# Pass in two_d_or_three_d = '2D' or '3D'
class lqr_controller():
    def __init__(self, 
                 multibody_system,
                 target_pos_in_generalized_coordinates,
                 Q_matrix, 
                 R_matrix, 
                 two_d_or_three_d,
                 contact_coordinate_list):
        if two_d_or_three_d != '2D' and two_d_or_three_d != '3D':
            return print(bcolors.FAIL + 'Error: Please pass in two_d_or_three_d as \'2D\' or \'3D\'.' + bcolors.ENDC)
        
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(bcolors.BOLD + '                          Creating LQR controller with given pose.' + bcolors.ENDC)
        print(bcolors.BOLD + '                                Calculation mode is {} mode.'.format(two_d_or_three_d) + bcolors.ENDC)
        print(bcolors.FAIL + '=================================================================================================\n' + bcolors.ENDC)
        self.multibody_system = multibody_system

        self.system_dimension = multibody_system.system_dimension
        self.target_pose_in_q_coord = target_pos_in_generalized_coordinates
        self.target_velocities_in_q_coord = np.zeros(self.system_dimension)
        self.contact_coordinate_list = contact_coordinate_list

        # Initialization step 1: Calculate equilibrium control output u_0, given multibody system and target pos q_0
        # Copy matrices to local scope
        Mass_Matrix_lambda = multibody_system.system_Mass_Matrix_lambda
        Coriolis_Vector_lambda = multibody_system.system_Coriolis_Vector_lambda
        Gravitational_force_vector_lambda = multibody_system.system_Gravitational_Force_Vector_lambda
        Control_matrix = multibody_system.system_Control_matrix
        constraint_equation_lambda = multibody_system.constraint_equation_lambda
        constraint_jacobian_lambda = multibody_system.constraint_jacobian_lambda
        constraint_jacobian_time_derivative_lambda = multibody_system.constraint_jacobian_time_derivative_lambda

        # Calculate system center of gravity and total mass
        self.system_total_mass, self.system_CoG_vector = Calculate_system_center_of_gravity(multibody_system, target_pos_in_generalized_coordinates)
        # Solve linear matrix equations from static force equilibruim to obtain ground reaction force vectors (lambda_vec)
        if two_d_or_three_d == '2D':
            self.ground_reaction_lambda_vector_list = Calculate_static_contact_reaction_force_lambda_vector_2D(self.system_total_mass, self.system_CoG_vector, self.contact_coordinate_list)
        else:
            self.ground_reaction_lambda_vector_list = Calculate_static_contact_reaction_force_lambda_vector_3D(self.system_total_mass, self.system_CoG_vector, self.contact_coordinate_list)

        # Stable control vector at given pose x_0
        self.u_0_vector = Calculate_equilibrium_control_output(Mass_Matrix_lambda,
                                                                Coriolis_Vector_lambda,
                                                                Gravitational_force_vector_lambda,
                                                                Control_matrix,
                                                                constraint_equation_lambda,
                                                                constraint_jacobian_lambda,
                                                                constraint_jacobian_time_derivative_lambda,
                                                                self.target_pose_in_q_coord,
                                                                self.ground_reaction_lambda_vector_list)

        # Calculate linearization
        # !!!!!!!!! Here x and u are from linearized state space model, diff(x,t) = Ax + Bu
        # x is state vector and u is control vector
        target_velocites = np.zeros(self.system_dimension)
        self.fixpoint_position_x_vec = list(self.target_pose_in_q_coord) + list(target_velocites)
        self.fixpoint_control_u_vec = self.u_0_vector.T.tolist()[0]
        # Do a partial linearization
        self.A_lin, self.B_lin = Partial_linearization_around_known_fixed_point(self.multibody_system, self.fixpoint_position_x_vec, self.fixpoint_control_u_vec, self.ground_reaction_lambda_vector_list, g_n=9.81)

        if is_controllable(self.A_lin, self.B_lin) == False:
            return print(bcolors.FAIL + 'ERROR: SYSTEM IS UNCONTROLLABLE' + bcolors.ENDC)
        else:
            print(bcolors.OKGREEN + 'Linearized system is controllable.' + bcolors.ENDC)
        
        # LQR K matrix
        self.K_matrix,_,_= ct.lqr(self.A_lin, self.B_lin, Q_matrix, R_matrix)

        # print(self.u_0_vector)
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(bcolors.BOLD + '                                 LQR controller is created.' + bcolors.ENDC)
        print(bcolors.BOLD + '                                 Stable control output u_0 given pose is as below.' + bcolors.ENDC)
        print(bcolors.BOLD + ' u_0 = {}'.format(self.u_0_vector.T.tolist()[0]) + bcolors.ENDC)
        print(bcolors.FAIL + '=================================================================================================\n' + bcolors.ENDC)
    
    def get_ground_reaction_lambda_list(self):
        return self.ground_reaction_lambda_vector_list
    
    def get_target_pose_in_q_coord(self):
        return self.target_pose_in_q_coord
    
    def get_stable_u0(self):
        print('Returning u_0 vector')
        return self.u_0_vector.T.tolist()[0]

    def get_A_B_matrices(self):
        print('Returning A_lin and B_lin matrices')
        return self.A_lin, self.B_lin

    def get_K_matrix(self):
        return self.K_matrix
    
    # LQR control vector is u = -Kx or sometimes u = Kx depending on the definition
    # dx/dt = A_lin * x + B_lin * u = A_lin * x - B_lin * K * x
    # dx/dt = (A_lin - B_lin * K) * x
    def get_stabilization_control_vector(self, current_full_state_vector):
        # Full state vector is [u1,u2,...,un,a1,a2,...,am,] combined with d[u1,u2,...,un,a1,a2,...,am,]/dt
        # Linearization is done with actuated DOFs
        self.system_dimension
        self.controlled_dimension = self.multibody_system.system_Control_matrix.rank()

        # We create a indexer to extract actuated generalized coordinates and velocities
        # from current_full_state_vector

        # list(range(number)) creates a list, use [num:] we can extract elements behind num th column
        # Create 2 lists which contains position of actuated generalized coordinates and actuated generalized velocities in the full state vector
        # Combine these 2 lists to create a indexer which could be used to extract actuated generalized coord/veloc from the full state vector
        actuated_generalized_coordinates_indexer = list(range(self.system_dimension))[self.controlled_dimension - 1 :]
        actuated_generalized_velocities_indexer = list(range(self.system_dimension * 2))[self.controlled_dimension + self.system_dimension - 1 :]
        indexer = actuated_generalized_coordinates_indexer + actuated_generalized_velocities_indexer

        target_full_state_vector = list(self.target_pose_in_q_coord) + list(np.zeros(self.system_dimension))
        full_state_vector_difference = np.matrix(current_full_state_vector) - np.matrix(target_full_state_vector)

        current_actuated_state_vector = np.array(full_state_vector_difference.tolist()[0])[indexer]
        # Note: For some reason if we dont add previously calculated statially stable controll vector u_0_vector,
        # the initial control LQR output will be 0, and the bot will fall
        u_vec = - self.K_matrix * np.matrix(current_actuated_state_vector).T + self.u_0_vector
        # print(u_vec.T)
        return u_vec
    



def Calculate_system_center_of_gravity(multibody_system, target_pose):
    link_objects_list = multibody_system.system_link_objects
    link_objects_list_numerical = []
    ########### Part1. Convert sympy link center of gravity vectors to pure numerical ones
    for i in range(len(link_objects_list)):
        link_object_dummy = link_objects_list[i].link_CoG_vector
        link_object_dummy = symbolical_to_numerical(link_object_dummy, link_objects_list)
        for j in range(len(multibody_system.system_generalized_coordinates)):
            q_coordinate = multibody_system.system_generalized_coordinates[j]
            link_object_dummy = link_object_dummy.subs(q_coordinate, target_pose[j])
        link_objects_list_numerical.append(link_object_dummy)

    ########### Part2. Calculate total mass of the system and center of gravity vector of the system
    system_total_mass = 0.
    system_CoG_vector = np.zeros([3,1], dtype=np.float64)
    print(shape(system_CoG_vector))
    for i in range(len(link_objects_list_numerical)):
        system_total_mass += float(link_objects_list[i].link_Mass_numerical)
        system_CoG_vector = system_CoG_vector + link_objects_list[i].link_Mass_numerical * np.matrix(link_objects_list_numerical[i])
        # system_CoG_vector = 3*  np.matrix(link_objects_list_numerical[i])
    system_CoG_vector = system_CoG_vector / system_total_mass
    
    print(bcolors.BOLD + 'Total mass of the system is {}kg.'.format(system_total_mass) + bcolors.ENDC)
    print('System center of gravity is at:')
    print(system_CoG_vector)

    return system_total_mass, system_CoG_vector



# Given any pose of the robot, the control output which will perfectly balance out 
# the gravity force can be calculated by inserting generalized accelerations and 
# generalized velocities as 0 vector into euqation of motion, and solve for u_vector
# Pass in:
# 1. Lambda matrices of the multibody system 
# 2. Target pose in generalized coordinates
# 3. ground_reaction_lambda_vector_list as [lambda_vec1, lambda_vec2] for 2D bot and [lambda_vec1, lambda_vec2, lambda_vec3, lambda_vec4] for 3D botS
def Calculate_equilibrium_control_output(   Mass_Matrix_lambda,
                                            Coriolis_Vector_lambda,
                                            Gravitational_force_vector_lambda,
                                            Control_matrix,
                                            constraint_equation_lambda,
                                            constraint_jacobian_lambda,
                                            constraint_jacobian_time_derivative_lambda,
                                            target_pos, 
                                            ground_reaction_lambda_vector_list,
                                            g_n = 9.81):
    
    # Step1. Insert target_pos vector into all matrices
    # Step2. Calculate the equilibrium control vector
    
    ######################### Step1. #########################  Insert target_pos vector into all matrices
    system_dimension = shape(target_pos)[0]
    # lambda function takes :((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
    target_velocites = np.zeros(system_dimension)
    # All matrices are lambdified with control, matrices below are control independent, Remove control force from all lambda function later!!!!!
    dummy_control = np.zeros(Control_matrix.rank())
    variables_to_insert = tuple( [0] + [g_n] + list(target_pos) + list(target_velocites) + list(dummy_control))
    M = np.matrix(Mass_Matrix_lambda(*variables_to_insert))
    M_inv = np.linalg.inv(M)
    C = np.matrix(Coriolis_Vector_lambda(*variables_to_insert))
    tau_g = np.matrix(Gravitational_force_vector_lambda(*variables_to_insert))
    Bm = np.matrix(Control_matrix).astype('float64')

    h_list = []
    H_list = []
    dHdt_list = []
    for i in range(len(constraint_equation_lambda)):
        h_list.append(np.matrix(constraint_equation_lambda[i](*variables_to_insert)))
        H_list.append(np.matrix(constraint_jacobian_lambda[i](*variables_to_insert)))
        dHdt_list.append(np.matrix(constraint_jacobian_time_derivative_lambda[i](*variables_to_insert)))
        
    ######################### Step2. ######################### Calculate the equilibrium control vector
    # Map ground reaction force to generalized coordinates
    H_lambda = np.transpose(np.matrix(np.zeros(system_dimension)))
    for i in range(len(h_list)):
        H_lambda = H_lambda + np.transpose(H_list[i]) * ground_reaction_lambda_vector_list[i]

    # Calculate u vector with below equation
    # Manipulator equation but acceleration and velocity is 0: 0 = tau_g + Bm*u + sum(H.T * lambda)
    # Solve for control vector: u = -pinv(Bm) * (tau_g + H.T*lambda)
    u_0 = - np.linalg.pinv(Bm)  * (tau_g + H_lambda)

    print(bcolors.OKGREEN + 'Equilibrium control vector at given pose is solved!\n' + bcolors.ENDC)
    return u_0


def Calculate_static_contact_reaction_force_lambda_vector_2D(system_total_mass, system_CoG_vector, contact_coordinate_list, g_n = 9.81):
    print('Solving reaction force lambda vectors.')
    print(bcolors.FAIL + 'Current calculatio mode is 2D!' + bcolors.ENDC)
    
    hx1 = contact_coordinate_list[0][0].item()
    hx2 = contact_coordinate_list[1][0].item()
    print('Left contact point is at x = {}m.'.format(hx1))
    print('Right contact point is at x = {}m.\n'.format(hx2))
    
    px = system_CoG_vector[0].item()
    py = system_CoG_vector[1].item()
    print(bcolors.FAIL + 'System center of gravity is at x={}, y={}'.format(round(px,5), round(py,5)) + bcolors.ENDC)

    ############################ Solve linear matrix equation Ax = b --> x = inv(A) * b  
    #                            x as reaction froce from the ground
    #                            x = [lambda_x_1,
    #                                 lambda_y_1,
    #                                 lambda_x_2,
    #                                 lambda_y_2]
    
    A_mat = np.matrix([[1,0,-1,0],
                       [0,1,0,1],
                       [0,-abs(hx1),0,hx2],
                       [py,-(abs(hx1)+px), py, (hx2-px)]], dtype=float)
    # print('Rans of A matrix is {}'.format(np.linalg.matrix_rank(A_mat)))

    b_vec = np.matrix([[0],
                   [g_n * system_total_mass],
                   [g_n * system_CoG_vector[0].item() * system_total_mass],
                   [0]])
    
    A_inv = np.linalg.pinv(A_mat)
    
    x = A_inv * b_vec

    lambda_1 = np.transpose(np.matrix([x[0].item(), x[1].item(), 0.]).astype('float64'))
    lambda_2 = np.transpose(np.matrix([x[2].item(), x[3].item(), 0.]).astype('float64'))

    lambda_vector_list = [lambda_1, lambda_2]
    return lambda_vector_list


def Calculate_static_contact_reaction_force_lambda_vector_3D(system_total_mass, system_CoG_vector, contact_coordinate_list, g_n = 9.81):
    pass



def is_controllable(A_matrix, B_matrix):
    A_matrix_rank = np.linalg.matrix_rank(A_matrix) 
    Controllability_matrix = ct.ctrb(A_matrix, B_matrix)
    C_matrix_rank = np.linalg.matrix_rank(Controllability_matrix)

    if A_matrix_rank == C_matrix_rank:
        controllable = True
    else:
        controllable = False
    return controllable

