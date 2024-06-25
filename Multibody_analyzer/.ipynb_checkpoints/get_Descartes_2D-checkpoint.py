#l_body, l_left_1, l_left_2, l_right_1, l_right_2
# link_constant_numerical = [4, 1,1,1,1]
def get_Descartes_2D(l_body, l_left_1, l_left_2, l_right_1, l_right_2, generalized_coordinates, sol, link_constant_numerical):
    x_sol = sol.T[0]
    y_sol = sol.T[1]
    theta_bot_sol = sol.T[2]
    theta_left_1_sol = sol.T[3]
    theta_left_2_sol = sol.T[4] 
    theta_right_1_sol = sol.T[5]
    theta_right_2_sol = sol.T[6]
    
    # To calculate:
    # 1. Bot left, right edge position,
    # 2. Bot left knee joint position,
    # 3. Bot left foot position
    
    # 4. Bot right knee joint position,
    # 5. Bot right foot position.
    x = Function('x')(t)
    y = Function('y')(t)
    theta_bot = Function('theta_bot')(t)
    theta_left_1 = Function('theta_left_1')(t)
    theta_left_2 = Function('theta_left_2')(t)
    theta_right_1 = Function('theta_right_1')(t)
    theta_right_2 = Function('theta_right_2')(t)

    
    ################# HTM Matrices ###################
    # Origin to body
    HTM_x = construct_HTM(eye(3), [x,0,0])
    HTM_y = construct_HTM(eye(3), [0,y,0])
    HTM_Rz = construct_HTM(Rz(theta_bot), [0,0,0])
    HTM1 = HTM_y * HTM_x * HTM_Rz
    
    # Bot left, right edge position,
    HTM_bot_left_lege = HTM1 * construct_HTM(eye(3),[-0.5*l_body, 0 , 0])
    HTM_bot_right_lege = HTM1 * construct_HTM(eye(3),[0.5*l_body, 0 , 0])

    # Bot left knee joint position, left foot position
    HTM_bot_left_knee = HTM1 * construct_HTM(eye(3),[-0.5*l_body, 0 , 0]) * construct_HTM(Rz(theta_left_1),[0,0,0]) * construct_HTM(eye(3), [l_left_1, 0, 0])
    HTM_bot_left_foot = HTM1 * construct_HTM(eye(3),[-0.5*l_body, 0 , 0]) * construct_HTM(Rz(theta_left_1),[0,0,0]) * construct_HTM(eye(3), [l_left_1, 0, 0]) * construct_HTM(Rz(theta_left_2),[0,0,0]) * construct_HTM(eye(3), [l_left_2, 0, 0])
    
    # Bot right knee joint position, right foot position
    HTM_bot_right_knee = HTM1 * construct_HTM(eye(3),[0.5*l_body, 0 , 0]) * construct_HTM(Rz(theta_right_1),[0,0,0]) * construct_HTM(eye(3), [l_right_1, 0, 0])
    HTM_bot_right_foot = HTM1 * construct_HTM(eye(3),[0.5*l_body, 0 , 0]) * construct_HTM(Rz(theta_right_1),[0,0,0]) * construct_HTM(eye(3), [l_right_1, 0, 0]) * construct_HTM(Rz(theta_right_2),[0,0,0]) * construct_HTM(eye(3), [l_right_2, 0, 0])
    
    
    ################# Extract postion vector from world coordinate ###################
    _, HTM_bot_left_lege_Vec = decompose_HTM(HTM_bot_left_lege)
    _, HTM_bot_right_lege_Vec = decompose_HTM(HTM_bot_right_lege)
    
    ##
    _, HTM_bot_left_knee_Vec = decompose_HTM(HTM_bot_left_knee)
    _, HTM_bot_left_foot_Vec = decompose_HTM(HTM_bot_left_foot)
    
    ##
    _, HTM_bot_right_knee_Vec = decompose_HTM(HTM_bot_right_knee)
    _, HTM_bot_right_foot_Vec = decompose_HTM(HTM_bot_right_foot)
    
    
    ################# Lambdify() all the position vectors ###################
    HTM_bot_left_lege_Vec_x = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_left_lege_Vec[0])
    HTM_bot_left_lege_Vec_y = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_left_lege_Vec[1])

    HTM_bot_right_lege_Vec_x = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_right_lege_Vec[0])
    HTM_bot_right_lege_Vec_y = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_right_lege_Vec[1])
    
    ##
    HTM_bot_left_knee_Vec_x = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_left_knee_Vec[0])
    HTM_bot_left_knee_Vec_y = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_left_knee_Vec[1])

    HTM_bot_left_foot_Vec_x = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_left_foot_Vec[0])
    HTM_bot_left_foot_Vec_y = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_left_foot_Vec[1])
    
    ##
    HTM_bot_right_knee_Vec_x = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_right_knee_Vec[0])
    HTM_bot_right_knee_Vec_y = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_right_knee_Vec[1])
    
    HTM_bot_right_foot_Vec_x = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_right_foot_Vec[0])
    HTM_bot_right_foot_Vec_y = lambdify([l_body, l_left_1, l_left_2, l_right_1, l_right_2, *tuple(generalized_coordinates)], 
                                      HTM_bot_right_foot_Vec[1])
    
    ################# Solve for xy coordinates ###################
    HTM_bot_left_lege_Vec_x_sol = HTM_bot_left_lege_Vec_x(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    HTM_bot_left_lege_Vec_y_sol = HTM_bot_left_lege_Vec_y(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
                            
    HTM_bot_right_lege_Vec_x_sol = HTM_bot_right_lege_Vec_x(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    HTM_bot_right_lege_Vec_y_sol = HTM_bot_right_lege_Vec_y(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    
    ##
    HTM_bot_left_knee_Vec_x_sol = HTM_bot_left_knee_Vec_x(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    HTM_bot_left_knee_Vec_y_sol = HTM_bot_left_knee_Vec_y(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
                            
    HTM_bot_left_foot_Vec_x_sol = HTM_bot_left_foot_Vec_x(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    HTM_bot_left_foot_Vec_y_sol = HTM_bot_left_foot_Vec_y(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
                            
    
    ##
    HTM_bot_right_knee_Vec_x_sol = HTM_bot_right_knee_Vec_x(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    HTM_bot_right_knee_Vec_y_sol = HTM_bot_right_knee_Vec_y(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    
    HTM_bot_right_foot_Vec_x_sol = HTM_bot_right_foot_Vec_x(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    HTM_bot_right_foot_Vec_y_sol = HTM_bot_right_foot_Vec_y(*tuple(link_constant_numerical),sol.T[0], sol.T[1], sol.T[2], sol.T[3], sol.T[4], sol.T[5], sol.T[6])
    
    
    
    return (HTM_bot_left_lege_Vec_x_sol, 
            HTM_bot_left_lege_Vec_y_sol,

            HTM_bot_right_lege_Vec_x_sol,
            HTM_bot_right_lege_Vec_y_sol,

            ##
            HTM_bot_left_knee_Vec_x_sol,
            HTM_bot_left_knee_Vec_y_sol,

            HTM_bot_left_foot_Vec_x_sol,
            HTM_bot_left_foot_Vec_y_sol,
            
            ##
            HTM_bot_right_knee_Vec_x_sol,
            HTM_bot_right_knee_Vec_y_sol,

            HTM_bot_right_foot_Vec_x_sol,
            HTM_bot_right_foot_Vec_y_sol)


#     x1 = np.zeros(len(y_sol))
#     y1 = y_sol
    
#     x2 = l2 * np.cos(theta1_sol)
#     y2 = l2 * np.sin(theta1_sol) + y_sol
    
#     x3 = l2 * np.cos(theta1_sol) + l3 * (-np.sin(theta1_sol)*np.sin(theta2_sol) + np.cos(theta1_sol)*np.cos(theta2_sol))
#     y3 = l2 * np.sin(theta1_sol) + l3 * (np.sin(theta1_sol)*np.cos(theta2_sol) + np.sin(theta2_sol)*np.cos(theta1_sol)) + y_sol
    
    # return (x1,y1, x2,y2, x3,y3)