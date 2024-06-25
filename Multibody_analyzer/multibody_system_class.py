from sympy import lambdify
from .basic_math import *
from .EMO_matrices import *
from .lqr_control import *
from .Energy_shaping_control import *
from .data_logger_class import *

# Full nonlinear dynamics object
# external force and contact force will be added later
class system_full_nonlinear_dynamics():
    def __init__(self, 
                 link_object_list, 
                 generalized_coordinates, 
                 t, # sympy symbol t
                 gravitational_acc_vector, # vector like sympy.Matrix([0,-g,0])
                 g, # sympy symbol g
                 system_Control_matrix, 
                 system_Control_force_vector, 
                 constraint_equation_objects_list,
                 BSP_alpha, 
                 external_force = None, 
                 contact_force = None):
        
        self.raw_integration_result = None # Store raw result
        self.t = t # symbol t

        # Link and constraint equation objects
        self.system_link_objects = link_object_list
        self.system_constraint_equation_objects = constraint_equation_objects_list
        self.BSP_alpha = BSP_alpha

        self.system_gravitational_acc_vector = gravitational_acc_vector
        self.g = g # symbol g
        self.system_Control_matrix = system_Control_matrix
        self.system_Control_force_vector = system_Control_force_vector

        # Generalized coordinates
        self.system_generalized_coordinates = generalized_coordinates
        self.system_generalized_velocities = diff(generalized_coordinates, t)
        self.system_generalized_accelerations = diff(generalized_coordinates, t, 2)
        self.system_dimension = shape(self.system_generalized_coordinates)[0]
        
        # Calculate M, C, tau_vec symbolically.
        self.system_Mass_matrix = Calculate_Complete_Mass_matrix(self.system_link_objects)
        self.system_Coriolis_vector = Calculate_Coriolis_Vector(self.system_Mass_matrix,
                                                                self.system_generalized_coordinates,
                                                                self.system_generalized_velocities) 
        self.system_Gravitational_Force_vector = Calculate_Complete_Gravtational_Force_vector(self.system_link_objects,
                                                                                              self.system_gravitational_acc_vector)
        # Insert constants into M_mat, C_mat, tau_vec
        self.system_Mass_matrix_numerical = symbolical_to_numerical(self.system_Mass_matrix, 
                                                                    self.system_link_objects)
        self.system_Coriolis_vector_numerical = symbolical_to_numerical(self.system_Coriolis_vector,
                                                                        self.system_link_objects)
        self.system_Gravitational_Force_vector_numerical = symbolical_to_numerical(self.system_Gravitational_Force_vector,
                                                                                 self.system_link_objects)
        
        #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Lambdify() ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # Lambdify() all matrices needed to compute state space with t,g,state_vector,control_vector
        self.system_state_vector = Matrix([self.system_generalized_coordinates, self.system_generalized_velocities])
        # In case of closed Energy-Shaping and LQR control: the control vector will be calculated within the model
        # But in that case the control force vector should be incorporated into the state_vector 
        symbols_for_lambdify = ((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
        print(symbols_for_lambdify)
    
        # Multibody system
        self.system_Mass_Matrix_lambda = lambdify((symbols_for_lambdify), self.system_Mass_matrix_numerical)
        self.system_Coriolis_Vector_lambda = lambdify((symbols_for_lambdify), self.system_Coriolis_vector_numerical)
        self.system_Gravitational_Force_Vector_lambda = lambdify((symbols_for_lambdify), self.system_Gravitational_Force_vector_numerical)

        # Constraint conditions boxes
        self.constraint_equation_lambda =[] # h vector
        self.constraint_jacobian_lambda = [] # H matrix (∂h_vec/∂q_vec, Jacobian of h_vec to generalized coordinates)
        self.constraint_jacobian_time_derivative_lambda = [] # ∂H/∂t Matrix
        # Create constraint conditions
        # constraint_equation_lambda = [h1, h2, ... , hn]
        # constraint_jacobian_lambda = [∂h1_vec/∂q_vec, ∂h2_vec/∂q_vec, ... , ∂hn_vec/∂q_vec] = [H1, H2, ... , Hn]
        # constraint_jacobian_time_derivative_lambda = [∂H1/∂t, ∂H2/∂t, ... , ∂Hn/∂t]
        for item in self.system_constraint_equation_objects:
            # Store 1 set of h, H, H_dot
            h_dummy = item.positional_constraint_equation
            H_dummy = item.positional_constraint_equation_jacobian
            H_dot_dummy = diff(item.positional_constraint_equation_jacobian, t)

            # Subs in all constants into h, H, H_dot
            h_dummy = symbolical_to_numerical(h_dummy, self.system_link_objects)
            H_dummy = symbolical_to_numerical(H_dummy, self.system_link_objects)
            H_dot_dummy = symbolical_to_numerical(H_dot_dummy, self.system_link_objects)

            # Lambdify() and store back into the list
            self.constraint_equation_lambda.append(lambdify((symbols_for_lambdify), h_dummy))
            self.constraint_jacobian_lambda.append(lambdify((symbols_for_lambdify), H_dummy))
            self.constraint_jacobian_time_derivative_lambda.append(lambdify((symbols_for_lambdify), H_dot_dummy))

        #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Lambdify() ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Flags for control initialization and activation
        self.Data_logger_is_created_flag = False
        self.LQR_is_created_flag = False
        self.Energy_shaping_controller_is_created_flag = False
        self.Fullauto_shock_absorbing_controller_is_created_flag = False
        self.Use_LQR = False
        self.Use_Energy_shaper = False
        self.Use_Automatic_switch_controller = False
        self.Active_data_logger = False
        
        # Controller switch threshold, 
        # Logic: If kinetic energy is smaller some value, switch to LQR controller
        self.controller_switching_threshold = 10
        self.u_max = 50 # Default max motor torque output

        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(bcolors.BOLD + '                                  Multibody system is created.' + bcolors.ENDC)
        print(bcolors.FAIL + '=================================================================================================\n' + bcolors.ENDC)

        # Create LQR controller for stabilization
    def lqr_controller_initialization(self, target_pose, Q, R, two_d_or_three_d, contact_coordinate_list):
        self.target_pos = target_pose
        self.lqr_controller = lqr_controller(self, np.array(target_pose) ,Q, R, two_d_or_three_d, contact_coordinate_list)
        u_initial_control_vector = self.lqr_controller.get_stable_u0()
        self.stable_u_vec_from_lqr_control = u_initial_control_vector
        
        self.LQR_is_created_flag = True
        return u_initial_control_vector


    def Energy_shaping_controller_initialization(self, target_pose):
        # Target pose is a list, [0] * n creates n length zero list 
        self.target_state_vector = target_pose + [0] * self.system_dimension
        self.Energy_shaping_controller = energy_shaping_controller(self, self.target_state_vector, self.stable_u_vec_from_lqr_control)
        self.Energy_shaping_controller_is_created_flag = True

    def data_logger_initialization(self):
        self.data_logger = data_logger()
        self.Data_logger_is_created_flag = True

    # There is no need to initialize this
    def Automatic_switch_controller_initialization(self):
        print(bcolors.BOLD + 'If Use_Automatic_switch_controller flag is set to true, controller will be automatically switched' + bcolors.ENDC)

    
    def print_controller_status(self):
        print('LQR controller is created: ==> ' + '{}'.format(self.LQR_is_created_flag))
        print('Energy shaping controller is created: ==> ' + '{}'.format(self.Energy_shaping_controller_is_created_flag))
        print('Data logger is created: ==> ' + '{}'.format(self.Data_logger_is_created_flag))
        print('Use LQR controller? ==> ' + '{}'.format(self.Use_LQR))
        print('Use Energy shaper? ==> ' + '{}'.format(self.Use_Energy_shaper))
        print('Use Automatic control? ==> ' + '{}'.format(self.Use_Automatic_switch_controller))
        print('Use Data logger? ==> ' + '{}'.format(self.Active_data_logger))
        # XOR of self.Use_LQR , self.Use_Energy_shaper , self.Use_Automatic_switch_controller
        if not ((self.Use_LQR ^ self.Use_Energy_shaper ^ self.Use_Automatic_switch_controller) and not (self.Use_LQR and self.Use_Energy_shaper and self.Use_Automatic_switch_controller)):
            print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
            print(               '                   WARNING! More than one controller is active!!!!')
            print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        

    
    def calculate_StateSpace_Lambdify_ConstraintForce_OdeForSolver(self):
        # Check controller status
        self.print_controller_status()
        
        t = self.t
        g = self.g
        BSP_alpha = self.BSP_alpha
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(               '                 Baumgarte stabilization parameter alpha = {} is passed in.'.format(BSP_alpha))
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)

        self.system_state_vector = Matrix([self.system_generalized_coordinates, self.system_generalized_velocities])
        # In case of closed Energy-Shaping and LQR control: the control vector will be calculated within the model
        # But in that case the control force vector should be incorporated into the state_vector 
        symbols_for_lambdify = ((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
        print(symbols_for_lambdify)
        
        # Multibody system lambda matrices
        Mass_Matrix_lambda = self.system_Mass_Matrix_lambda
        Coriolis_Vector_lambda = self.system_Coriolis_Vector_lambda
        Gravtational_Force_Vector_lambda = self.system_Gravitational_Force_Vector_lambda

        # Constraint conditions    
        constraint_equation_lambda = self.constraint_equation_lambda # h vector
        constraint_jacobian_lambda = self.constraint_jacobian_lambda # H matrix (∂h_vec/∂q_vec, Jacobian of h_vec to generalized coordinates)
        constraint_jacobian_time_derivative_lambda = self.constraint_jacobian_time_derivative_lambda # dH/dt Matrix
        
        # q_dot lambda
        q_dot_lambda = lambdify((symbols_for_lambdify), self.system_generalized_velocities)

        state_vector_dummy = self.system_state_vector
        # constant_and_control = (g) + tuple(self.system_Control_force_vector)
        def ODE(t, X, pbar, state, g, *Control_force_vector):
            ################################################## Progress bar
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n      
            ################################################## Progress bar

            # Extract current state from X
            state_vector_dummy = X

            ################################################## Control scheme ##################
            ################################################## Control scheme ##################
            ################################################## Control scheme ##################
            # PUT UR DAMN BADASS CONTROL logic here im so damn tired after 4 months of pure math derivations and 2 months 
            # of pythoning matrices manipulation
            if self.Use_LQR == True:
                Control_force_vector = self.lqr_controller.get_stabilization_control_vector(state_vector_dummy)
                # Below function is just there for obtaining current energies
                (_, 
                 current_mechanical_energy, 
                 current_kinetic_energy, 
                 current_potential_energy, 
                 current_energy_error) = self.Energy_shaping_controller.get_energy_shaping_control_vector(state_vector_dummy)
                Lyapunov_stable_condition = 0 # Set to 0 for we dont need to calculate energy shaper anymore but still want to plot easily
                
            if self.Use_Energy_shaper == True:
                (_, 
                 current_mechanical_energy, 
                 current_kinetic_energy, 
                 current_potential_energy, 
                 current_energy_error) = self.Energy_shaping_controller.get_energy_shaping_control_vector(state_vector_dummy)
                Control_force_vector,_,Lyapunov_stable_condition = self.Energy_shaping_controller.get_energy_shaping_control_vector_KE_control(state_vector_dummy)

            if self.Use_Automatic_switch_controller == True:
                (_, 
                 current_mechanical_energy, 
                 current_kinetic_energy, 
                 current_potential_energy, 
                 current_energy_error) = self.Energy_shaping_controller.get_energy_shaping_control_vector(state_vector_dummy)
                Control_force_vector,_,Lyapunov_stable_condition = self.Energy_shaping_controller.get_energy_shaping_control_vector_KE_control(state_vector_dummy)

                # Controller switching condition
                if current_kinetic_energy.item() < self.controller_switching_threshold:
                    self.Use_LQR = True
                    self.Use_Automatic_switch_controller = False
                    print('Controller is switched to LQR')
            # print(current_kinetic_energy.item())
            # print('Use_LQR = {}'.format(self.Use_LQR))
            # Check for controller satuation
            Control_force_vector = self.check_controller_satuation(Control_force_vector)

            # Log the data
            if self.Active_data_logger == True:
                self.data_logger.data_append('t_n', t)
                self.data_logger.data_append('current_state_vector_n', state_vector_dummy)
                self.data_logger.data_append('target_state_vector_n', self.target_state_vector)
                if self.Use_LQR == True:
                    self.data_logger.data_append('current_control_mode', 'LQR')
                else:
                    self.data_logger.data_append('current_control_mode', 'Energy Shaping')
                    
                self.data_logger.data_append('current_control_vector_n', Control_force_vector)
                
                self.data_logger.data_append('current_mechanical_energy', current_mechanical_energy)
                self.data_logger.data_append('current_kinetic_energy', current_kinetic_energy)
                self.data_logger.data_append('current_potential_energy', current_potential_energy)
                
                self.data_logger.data_append('target_mechanical_energy', self.Energy_shaping_controller.target_mechanical_energy)
                self.data_logger.data_append('target_kinetic_energy', self.Energy_shaping_controller.target_kinetic_energy)
                self.data_logger.data_append('target_potential_energy', self.Energy_shaping_controller.target_potential_energy)
                
                self.data_logger.data_append('current_energy_error', current_energy_error)
                self.data_logger.data_append('Lyapunov_stable_condition', Lyapunov_stable_condition)

            ################################################## Control scheme ##################
            ################################################## Control scheme ##################
                
            ################################################### Numerical integration ############
            ################################################### Numerical integration ############
            # sub in initial conditions/ last state to calculate the q double dot term (complete numerical state space model)

            # Extract current state from X
            q_vec_current = []
            q_dot_vec_current = []
            for i in range(self.system_dimension):
                q_vec_current.append(X[i])
                q_dot_vec_current.append(X[self.system_dimension+i])
            q_vec_current =  np.transpose(np.matrix(q_vec_current))
            q_dot_vec_current =  np.transpose(np.matrix(q_dot_vec_current))

            # Calculate numerical matrices
            Mass_Matrix = np.matrix(Mass_Matrix_lambda(t, g, *state_vector_dummy, *Control_force_vector))
            Mass_Matrix_inv = np.linalg.inv(Mass_Matrix)
            Coriolis_Vector = np.matrix(Coriolis_Vector_lambda(t, g, *state_vector_dummy,  *Control_force_vector))
            Gravtational_Force_Vector = np.matrix(Gravtational_Force_Vector_lambda(t, g, *state_vector_dummy,  *Control_force_vector))
            Control_Matrix = np.matrix(self.system_Control_matrix)
            Control_vector =  np.transpose(np.matrix(Control_force_vector))

            # clear the constraint force in generalized coordinate
            constraint_force_in_q_coord = []
            for i in range(len(self.system_generalized_coordinates)):
                constraint_force_in_q_coord.append(0)

            constraint_force_in_q_coord = np.transpose(np.matrix(constraint_force_in_q_coord))

            ground_reaction_y_component = np.zeros([len(self.system_constraint_equation_objects),1]) 
            # Calculate i-th constraint force and sum them together in the end
            for i in range(len(self.system_constraint_equation_objects)):
                
                h_Vec =  np.matrix(constraint_equation_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
                H_Mat = np.matrix(constraint_jacobian_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
                H_dot_Mat = np.matrix(constraint_jacobian_time_derivative_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )

                # constraint_force_in_xyz is constraint force lambda
                constraint_force_in_xyz = -np.linalg.pinv(H_Mat * Mass_Matrix_inv *np.transpose(H_Mat)) * (H_Mat * Mass_Matrix_inv * (Gravtational_Force_Vector + Control_Matrix * Control_vector.T - Coriolis_Vector) + (2*BSP_alpha*H_Mat + H_dot_Mat)*q_dot_vec_current + BSP_alpha**2 * h_Vec)
                # constraint_force_in_q_coord is (H_mat^T * lambda_vec)
                constraint_force_in_q_coord = constraint_force_in_q_coord + np.transpose(H_Mat) * constraint_force_in_xyz
                ground_reaction_y_component[i] = constraint_force_in_xyz[1] # record ground reaction y component
            
            # Log ground reaction of each foot to nx1 array
            # Ex: ground_reaction_y_component[i] is ith foot's y component
            if self.Active_data_logger == True:
                self.data_logger.ground_reaction_force.append(ground_reaction_y_component)


            q_double_dot = Mass_Matrix_inv * (Gravtational_Force_Vector + 
                                            Control_Matrix * Control_vector.T + 
                                            constraint_force_in_q_coord - 
                                            Coriolis_Vector)

            # nonlinear_state_space_model_RHS is [q_dot, q_double_dot]
            nonlinear_state_space_model_RHS = np.transpose(np.matrix(q_dot_lambda(t, g, *state_vector_dummy, *Control_force_vector))).tolist()[0] + np.transpose(np.matrix(q_double_dot)).tolist()[0]

            return nonlinear_state_space_model_RHS
        return ODE

    # Change max/saturation output of each motor
    def set_saturation_value(self, saturation_value_list):
        self.u_max = saturation_value_list
        print(bcolors.BOLD + 'Motor saturation output is set to {} Nm'.format(self.u_max) + bcolors.ENDC)

    def check_controller_satuation(self, u_vec):
        # Check for saturation
        # u_vec = np.transpose(np.matrix(u_vec))
        for i in range(len(u_vec)): 
            if abs(u_vec[i].item()) > self.u_max[i]:
                if u_vec[i].item() > 0:
                    u_vec[i] = np.matrix(self.u_max[i]) # saturates and over 0
                elif u_vec[i] < 0:
                    u_vec[i] = - np.matrix(self.u_max[i]) # saturates and below 0
        # u_vec = np.transpose(u_vec)
        return u_vec
    

    def print_EoM(self):
        Print_out_the_full_EoM(self.system_Mass_matrix,
                               self.system_Coriolis_vector,
                               self.system_Gravitational_Force_vector,
                               self.system_generalized_velocities, 
                               self.system_generalized_accelerations,
                               self.system_Control_matrix,
                               self.system_Control_force_vector,
                               self.system_constraint_equation_objects)

    def print_EoM_numerical(self):
        Print_out_the_full_EoM(self.system_Mass_matrix_numerical,
                               self.system_Coriolis_vector_numerical,
                               self.system_Gravitational_Force_vector_numerical,
                               self.system_generalized_velocities, 
                               self.system_generalized_accelerations,
                               self.system_Control_matrix,
                               self.system_Control_force_vector,
                               self.system_constraint_equation_objects)
        
    def get_lambda_functions_out(self):
        print(bcolors.FAIL + "Returning lambdified equation of motion components in following order." + bcolors.ENDC)
        print(bcolors.FAIL + "Mass_matrix, Coriolis_vector, Gravitational_force_vector, constraint_equation, constraint_jacobian, constraint_jacobian_time_derivative" + bcolors.ENDC)
        return (self.system_Mass_Matrix_lambda, # M matrix
                self.system_Coriolis_Vector_lambda, # C vector
                self.system_Gravitational_Force_Vector_lambda, # tau_g vector
                self.constraint_equation_lambda, # h vector list, [h1, h2, ... , hn]
                self.constraint_jacobian_lambda, # H matrix list, [H1, H2, ... , Hn]
                self.constraint_jacobian_time_derivative_lambda # H_dot list, [∂H1/∂t, ∂H2/∂t, ... , ∂Hn/∂t]
                )