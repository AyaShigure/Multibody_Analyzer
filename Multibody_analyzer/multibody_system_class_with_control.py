from sympy import lambdify
from .basic_math import *
from .EMO_matrices import *


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
        self.constraint_equation_lambda = [] # h vector
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
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(bcolors.BOLD + '                                  Multibody system is created.' + bcolors.ENDC)
        print(bcolors.FAIL + '=================================================================================================\n' + bcolors.ENDC)


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

    # Pass in: 
    #   t: Sympy symbol, t
    #   g: Sympy symbol, g
    #   BSP_alpha: Baumgarte's Stabilization Parameter alpha, (Pass in 0 if there is no constraint condition)
    def calculate_StateSpace_Lambdify_ConstraintForce_OdeForSolver(self):
        t = self.t
        g = self.g
        BSP_alpha = self.BSP_alpha
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(               '                 Baumgarte stabilization parameter alpha = {} is passed in.'.format(BSP_alpha))
        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)

        ##############################################################  Part 1  
        # Create a 0 vector box with dim(q) dimensions


        ##############################################################  Part 2
        # Lambdify() all matrices needed to compute state space with t,g,state_vector,control_vector
        self.system_state_vector = Matrix([self.system_generalized_coordinates, self.system_generalized_velocities])
        # In case of closed Energy-Shaping and LQR control: the control vector will be calculated within the model
        # But in that case the control force vector should be incorporated into the state_vector 
        symbols_for_lambdify = ((t) , (g) , *tuple(self.system_state_vector) , *tuple(self.system_Control_force_vector))
        print(symbols_for_lambdify)
        
        # Multibody system
        Mass_Matrix_lambda = lambdify((symbols_for_lambdify), self.system_Mass_matrix_numerical)
        Coriolis_Vector_lambda = lambdify((symbols_for_lambdify), self.system_Coriolis_vector_numerical)
        Gravtational_Force_Vector_lambda = lambdify((symbols_for_lambdify), self.system_Gravitational_Force_vector_numerical)

        # Constraint conditions    
        constraint_equation_lambda =[] # h vector
        constraint_jacobian_lambda = [] # H matrix (∂h_vec/∂q_vec, Jacobian of h_vec to generalized coordinates)
        constraint_jacobian_time_derivative_lambda = [] # dH/dt Matrix
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
            constraint_equation_lambda.append(lambdify((symbols_for_lambdify), h_dummy))
            constraint_jacobian_lambda.append(lambdify((symbols_for_lambdify), H_dummy))
            constraint_jacobian_time_derivative_lambda.append(lambdify((symbols_for_lambdify), H_dot_dummy))
        
        q_dot_lambda = lambdify((symbols_for_lambdify), self.system_generalized_velocities)

        state_vector_dummy = self.system_state_vector
        # constant_and_control = (g) + tuple(self.system_Control_force_vector)
        Controller_initialization_flag = False
        def ODE(t, X, pbar, state, g, *Control_force_vector):
            # sub in initial conditions/ last state to calculate the q double dot term (complete numerical state space model)
            if Controller_initialization_flag == False:
                # calculate target energy
                target_energy = calculate_target_energy()
                # calculate lyapunov energy shaping controller lambda function
                energy_shaper_u_vec_lambda = energy_shaper()
                # calculate linearization and lqr controler
                A_lin, B_lin = linearization()
                K_matrix = lqr(A,B,Q,R)

                Controller_initialization_flag = True

            # Extract current state from X
            state_vector_dummy = X
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

            # Control vector calculation
            # Control logic: 
            #                   1. If touch ground, active Lyapunov Energy shaping
            #                   2. If close to target energy level, switch to LQR
            Current_energy = calculate_current_energy(xxx)
            if energy_shaping_control:
                # Do energy shaping control here
            else:
                # switch to LQR here


            











            # clear the constraint force in generalized coordinate
            constraint_force_in_q_coord = []
            for i in range(len(self.system_generalized_coordinates)):
                constraint_force_in_q_coord.append(0)

            constraint_force_in_q_coord = np.transpose(np.matrix(constraint_force_in_q_coord))

            # Calculate i-th constraint force and sum them together in the end
            for i in range(len(self.system_constraint_equation_objects)):
                
                h_Vec =  np.matrix(constraint_equation_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
                H_Mat = np.matrix(constraint_jacobian_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )
                H_dot_Mat = np.matrix(constraint_jacobian_time_derivative_lambda[i](t, g, *state_vector_dummy, *Control_force_vector) )

                # constraint_force_in_xyz is constraint force lambda
                constraint_force_in_xyz = -np.linalg.pinv(H_Mat * Mass_Matrix_inv *np.transpose(H_Mat)) * (H_Mat * Mass_Matrix_inv * (Gravtational_Force_Vector + Control_Matrix * Control_vector - Coriolis_Vector) + (2*BSP_alpha*H_Mat + H_dot_Mat)*q_dot_vec_current + BSP_alpha**2 * h_Vec)
                # constraint_force_in_q_coord is (H_mat^T * lambda_vec)
                constraint_force_in_q_coord = constraint_force_in_q_coord + np.transpose(H_Mat) * constraint_force_in_xyz

            q_double_dot = Mass_Matrix_inv * (Gravtational_Force_Vector + 
                                            Control_Matrix * Control_vector + 
                                            constraint_force_in_q_coord - 
                                            Coriolis_Vector)

            # nonlinear_state_space_model_RHS is [q_dot, q_double_dot]
            nonlinear_state_space_model_RHS = np.transpose(np.matrix(q_dot_lambda(t, g, *state_vector_dummy, *Control_force_vector))).tolist()[0] + np.transpose(np.matrix(q_double_dot)).tolist()[0]

            # Progress bar
            last_t, dt = state
            n = int((t - last_t)/dt)
            pbar.update(n)
            state[0] = last_t + dt * n      

            return nonlinear_state_space_model_RHS
        return ODE



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
        