from sympy import Matrix, zeros, shape, diff, Rational, symbols, Function, latex
from IPython.display import Math
from .basic_math import *



def Calculate_Coriolis_Vector(Inertia_matrix, generalized_coordinates, generalized_velocities):
    gene_space_dim = len(generalized_coordinates)
    C_i = zeros(gene_space_dim, gene_space_dim) # ith element of Christoffel symbols box (for mid-calculations)
    Coriolis_Vector = zeros(gene_space_dim,1) # Coriolis Vector box

    # Calculate Christoffel symbol 
    for i in range(gene_space_dim):
        # ith
        for j in range(gene_space_dim):
            for k in range(gene_space_dim):
                tensor_element = diff(access_matrix_element(Inertia_matrix, i+1, j+1), generalized_coordinates[k]) - Rational(1/2) * diff(access_matrix_element(Inertia_matrix, j+1, k+1), generalized_coordinates[i])
                C_i = change_matrix_element(C_i,j+1,k+1,tensor_element)

        # Calculate Coriolis dq.T*C*dq
        C_element = generalized_velocities.T * C_i * generalized_velocities
        C_element.simplify()
        Coriolis_Vector = change_matrix_element(Coriolis_Vector, i+1, 1, C_element)
    return Coriolis_Vector
    

def Create_generalized_coordinate_system(coordinate_list, t):
    # !!! Also pass time variable in for calculate derivatives.!
    # This function takes in list of indivadually defined generalized coordinates and form then into matrix form of generalized_coord, vel, and accele.
    generalized_coordinate = Matrix(coordinate_list)
    generalized_velocity = diff(generalized_coordinate, t)
    generalized_accelerations = diff(generalized_velocity, t)
    generalized_coordinates_dimension = len(coordinate_list)
    
    # Return dimension of the generalized coordinate space, and generalized coordinate, velocity, acceleraction
    return generalized_coordinates_dimension, generalized_coordinate, generalized_velocity, generalized_accelerations


# Pass in all the link object as a list, like [link_obj_1, link_obj_2]. The order does not matter here.
def Calculate_Complete_Mass_matrix(link_object_list):
    # Access the dimension of the generalized coordinates
    # The column number is the same as the dimension of generalized coordinates.
    generalized_coordinates_dimension = shape(link_object_list[0].Jv)[1]

    # Create a 0 matrix for mass matrix
    Complete_Mass_matrix = zeros(generalized_coordinates_dimension, generalized_coordinates_dimension)
    for link in link_object_list:
        Complete_Mass_matrix += link.Calculate_mass_matrix_components()

    return Complete_Mass_matrix


# Also pass in gravtational acceleration vector defined at beginning.
def Calculate_Complete_Gravtational_Force_vector(link_object_list, gravitational_acc_vector):
    # Access the dimension of the generalized coordinates
    # The column number is the same as the dimension of generalized coordinates.
    generalized_coordinates_dimension = shape(link_object_list[0].Jv)[1]
    
    # Create a 0 vector for grav force
    Complete_Gravtational_Froce_vector = zeros(generalized_coordinates_dimension, 1)
    for link in link_object_list:
        Complete_Gravtational_Froce_vector += link.Calculate_generalized_gravitational_force_vector_component(gravitational_acc_vector) 
        
    return Complete_Gravtational_Froce_vector


# This function is uses to sub all constant variables with numerical value in a sympy matrix
def symbolical_to_numerical(matrix_symbolical, link_object_list):
    # Substitude all symbols with numerical values to speed up the state space model generation.    
    matrix_numerical = matrix_symbolical
    for link_object_item in link_object_list:
        for i in range(len(link_object_item.link_Constants_symbolical)):
            matrix_numerical = matrix_numerical.subs(link_object_item.link_Constants_symbolical[i],
                                                     link_object_item.link_Constants_numerical[i])
    return matrix_numerical


# passed in matrix should have row3 and col3 all be 0
def xy_planer_constraint_matrix_pseudo_inversion_patch(matrix):
    matrix.row_del(2)
    matrix.col_del(2)
    inverted_2_by_2 = fast_pinv(matrix)
    inverted_matrix = inverted_2_by_2.row_insert(2, Matrix([[0, 0]]))
    inverted_matrix = inverted_matrix.col_insert(2, Matrix([0, 0, 0]))

    return inverted_matrix


# A equation of motion printer
def Print_out_the_full_EoM(Mass_matrix, Coriolis_vector, Grav_force_vector, generalized_velocities, generalized_accelerations, Control_matrix = None, Control_force_vector = None, constraint_equation_objects_list= None):
    # Dimension mismatch checker
    dim_generalized_space = len(generalized_accelerations) # Use this as reference for dimension mismatch checker (Because its definition is too simple to get wrong)
    dim_mass_mat = shape(Mass_matrix)[0]
    dim_coriolis_vec = shape(Coriolis_vector)[0]
    dim_grav_vec = shape(Grav_force_vector)[0]
    
    if Control_matrix != None and Control_force_vector != None:
        dim_control_mat_row = shape(Control_matrix)[0]
        # If control is passed in, we also need to check if the column matches with row dimension of control force vector        
        dim_control_mat_column = shape(Control_matrix)[1] 
        dim_control_force_vec = shape(Control_force_vector)[0]
        
        # Check control matrix column and control force vector row
        if dim_control_mat_column != dim_control_force_vec:
            print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
            print(bcolors.FAIL + 'Error: Dimension mismatch (Control). Control matrix column does not match with Control force row.' + bcolors.ENDC)
            print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
            
            return
        # Check control matrix row with dimension of generalized coordinates
        if dim_generalized_space != dim_control_mat_row:
            print(bcolors.FAIL + '=================================================================================================================' + bcolors.ENDC)
            print(bcolors.FAIL + 'Error: Dimension mismatch (Control). Control matrix row does not match with dimension of generalized coordinates.' + bcolors.ENDC)
            print(bcolors.FAIL + '=================================================================================================================' + bcolors.ENDC)
            return 
    
    # Check dimension mismatch Mass_mat, Corilis_vec, Grav_force_vec
    if dim_generalized_space != dim_mass_mat or dim_generalized_space != dim_coriolis_vec or dim_generalized_space != dim_grav_vec:
        print(bcolors.FAIL + '==================================================================================================================================================' + bcolors.ENDC)
        print(bcolors.FAIL + 'Error: Dimension mismatch (Mass matrix, Coriolis vector, Gravitational force vector). Double check the row dimensions of all matrices and vectors.' + bcolors.ENDC)
        print(bcolors.FAIL + '==================================================================================================================================================' + bcolors.ENDC)
        return 

    # I was going to write a check on dimension of constraint equations' jacobian, gonna do it later, or maybe never.
    # if constraint_equation_objects_list != None and :
        
    
    # Check is passed, print the EoM
    print(bcolors.WARNING + '======The full nonlinear dynamics is formulated in below from.======' + bcolors.ENDC)
    tau_g = symbols('tau_g')
    # constraint force is lambda_vec
    t = symbols('t')
    lambda_x = Function('lambda_x')(t)
    lambda_y = Function('lambda_y')(t)
    lambda_z = Function('lambda_z')(t)
    lambda_vec = Matrix([lambda_x, lambda_y, lambda_z])
    display(Math('%s%s%s + %s^TC%s = %s + Bu + H^T%s' % ('\hspace{2.2cm}',
                                                 'M',
                                                 latex(generalized_accelerations), 
                                                 latex(generalized_velocities), 
                                                 latex(generalized_velocities), 
                                                 latex(tau_g),
                                                 latex(lambda_vec))))
    print(bcolors.WARNING + '====================================================================\n\n' + bcolors.ENDC)    

    if(Control_matrix == None or Control_force_vector == None):
        print(bcolors.OKCYAN +'======== The full nonlinear dynamics of given system ' + bcolors.FAIL + 'WITHOUT' + bcolors.OKCYAN + ' control inputs is shown below.'+ bcolors.ENDC)
        display(Math('%s%s%s + %s = %s' % ('\hspace{0cm}',
                                           latex(Mass_matrix), 
                                           latex(generalized_accelerations), 
                                           latex(Coriolis_vector), 
                                           latex(Grav_force_vector))))
        print('\n\n')
    else:
        print(bcolors.OKCYAN + '=================== The full nonlinear dynamics of given system ' + bcolors.FAIL + 'WITH' + bcolors.OKCYAN + ' control and positional constraints is shown below.'+ bcolors.ENDC)
        display(Math('%s%s%s + %s = %s + %s%s + H^T%s' % ('\hspace{0cm}',
                                                latex(Mass_matrix), 
                                                latex(generalized_accelerations), 
                                                latex(Coriolis_vector), 
                                                latex(Grav_force_vector), 
                                                latex(Control_matrix),
                                                latex(Control_force_vector),
                                                latex(lambda_vec))))
        print('\n\n')
