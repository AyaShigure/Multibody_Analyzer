# Code is build with Sympy version: 1.11.1
from sympy import Matrix, cos, sin, eye ,zeros, shape, transpose
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Colorful print
# print(bcolors.FAIL + "Error: print stuff!" + bcolors.ENDC)
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def Create_Moment_of_Inertia_tensor(Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
    MoI_tensor = Matrix([[Ixx,Ixy,Ixz],
                         [Ixy,Iyy,Iyz],
                         [Ixz,Iyz,Izz]])
    return MoI_tensor

def Rx(a):
    R = Matrix([[1,0,0],
                [0,cos(a),-sin(a)],
                [0,sin(a),cos(a)]])
    return R

def Ry(a):
    R = Matrix([[cos(a),0,sin(a)],
                [0,1,0],
                [-sin(a),0,cos(a)]])
    return R

def Rz(a):
    R = Matrix([[cos(a),-sin(a),0],
                [sin(a),cos(a),0],
                [0,0,1]])
    return R

# Skew symmetrix form of a vector
def skew_symmetric_form(list):
    a = []
    for i in range(3):
        a.append(list[i])
    M = Matrix([[0,-a[2],a[1]],
                [a[2],0,-a[0]],
                [-a[1],a[0],0]])
    return M

# Skew matrix form to vector form
# !!!!!! pass 'M.tolist()'
def vector_form(matrix):
    a = []
    a.append(matrix[2][1])
    a.append(matrix[0][2])
    a.append(matrix[1][0])
    M = Matrix([a[0],a[1],a[2]])
    return M

# Pass in rotation matrix with dim of 3x3, translational vector with dim of 3x1
#
# ex:    Rot_mat = Matrix(3x3)
#        Trans_vec = Matrix(3x1)
#        HTM = construct_HTM(Rot_mat.tolist(),Trans_vec)

def construct_HTM(Rot_mat,Trans_vec):
    HTM = Matrix(Rot_mat)
    HTM = HTM.col_insert(999,Matrix(Trans_vec))
    HTM = HTM.row_insert(999,Matrix([0,0,0,1]).T)
    return HTM

# Homogeneous transformation matrix decomposition
# Pass in HTM.tolist()
def decompose_HTM(Homogeneous_transformation_matrix):
    Rotation_matrix = Matrix(Homogeneous_transformation_matrix)
    Translation_vector = Matrix(Homogeneous_transformation_matrix)
    Rotation_matrix.col_del(3)
    Rotation_matrix.row_del(3)
    
    for i in range(3):
        Translation_vector.col_del(0)
    Translation_vector.row_del(3)
    return Rotation_matrix, Translation_vector

def DH_HTM(theta,d,a,alpha):
    # Homogeneous Transformation Matrix under DH parameter convention
    ##
    # DH HTM:
    # 1. theta: rotation about z
    # 2. d: translation along z
    # 3. a: translation along x
    # 4. alpha: rotation about x
    ##
    DH_HTM_1 = construct_HTM(Rz(theta), Matrix([0,0,0]))
    DH_HTM_2 = construct_HTM(eye(3), Matrix([0,0,d]))
    DH_HTM_3 = construct_HTM(eye(3),Matrix([a,0,0]))
    DH_HTM_4 = construct_HTM(Rx(alpha),Matrix([0,0,0]))
    
    DH_HTM = DH_HTM_1 * DH_HTM_2 * DH_HTM_3 * DH_HTM_4
    return DH_HTM

##############################################################################################################################
##############################################################################################################################
def Rotational_jacobian(Rotation_matrix,generalized_coordinates):
    Jw = Matrix([])
    R = Matrix(Rotation_matrix)
    col_counter = 0
    for i in generalized_coordinates:
        M = zeros(3)
        M = R.diff(i)*R.T
        M.simplify()
        Jw_column = vector_form(M.tolist())
        Jw = Jw.col_insert(999,Jw_column) # insert to 999th or just the last column
    return Jw
##############################################################################################################################
##############################################################################################################################

# This function is created for Coriolis Tensor calculations
# This function starts from 1!S
def change_matrix_element(Mat, row, col, new_val):
    # Sympy matrix is also a long 1D array, here we find the target position of the element, then substitute the value with new_val
    
    # Dimension mismatch check:
    Mat_row_size = shape(Mat)[0]
    Mat_col_size = shape(Mat)[1]
    if(row < 0 or col < 0 or row > Mat_row_size or col > Mat_col_size):
        return print(bcolors.FAIL + "Error: Input row or col exceeds the dimension of the matrix!" + bcolors.ENDC)

    # Element_position:
    # (row-1) * shape(Mat)[1] : calculate total elements before the target row, 
    # + col : add position in last row,
    # -1 : sinse lists starts from 0
    element_position = (row-1) * shape(Mat)[1] + col - 1  
    Mat[element_position] = new_val
    # print('element position is {}'.format(element_position))
    return Mat

# This function is created for Coriolis Tensor calculations
# This function starts from 1!
def access_matrix_element(Mat, row, col):
    # Dimension mismatch check:
    Mat_row_size = shape(Mat)[0]
    Mat_col_size = shape(Mat)[1]
    if(row <= 0 or col <= 0 or row > Mat_row_size or col > Mat_col_size):
        return print(bcolors.FAIL + "Error: Input row or col exceeds the dimension of the matrix!" + bcolors.ENDC)
    
    element_position = (row-1) * shape(Mat)[1] + col - 1  
    return Mat[element_position] 


# Here are 2 functions for faster computation of inverse matrices
# These are for sympy matrices,
# Since all matrices later on are converted into numpy matrices,
# Thses functions are not utilized at all. 
def fast_inv(A): # Fast square inverse
    A_fast_piv = A._rep.to_field().inv().to_Matrix()
    return A_fast_piv

def fast_pinv(A): # Fase Moore-Penrose pseudo-inverse
    # A_pinv = (A^T * A)^-1 * A^T
    A_transpose = transpose(A)
    A_trans_times_A = A_transpose * A
    A_trans_times_A_inverse = A_trans_times_A._rep.to_field().inv().to_Matrix()
    A_fast_pinv = A_trans_times_A_inverse * A_transpose
    return A_fast_pinv
