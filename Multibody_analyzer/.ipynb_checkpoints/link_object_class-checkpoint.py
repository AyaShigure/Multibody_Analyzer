from sympy import latex
from .basic_math import *
from IPython.display import Math


# Here we test some of object oriented programming
class Create_link_object():
    # Initiation of the link object
    # Passing in: 
    # 0, link name (as a string like: 'name of the link')
    # 1. link Mass
    # 2. link Moment of Inertia tensor 
    # 3. link Centor of Gravity vector (From world frame), Derive this with DH HTM
    # 4. link Orientation Matrix (From world frame), Derive this with DH HTM    
    # 5. generalized coordinates.
    # 6. link Constants symbols, This is a list of constants symbols which the link is defined upon. (Example: link_Constant = [m1, lc1, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    def __init__(self, link_Name, link_notes, link_Mass, link_MoI_tensor, link_CoG_vector, link_Orientation_matrix, generalized_coordinates, link_Constants_symbolical):
        self.link_Name = link_Name
        self.link_Notes = link_notes
        # print(bcolors.OKBLUE + "Creating link object '{}'.".format(self.link_Name) + bcolors.ENDC)
        self.link_Mass = link_Mass
        self.link_MoI_tensor = link_MoI_tensor
        self.link_CoG_vector = link_CoG_vector
        self.link_Orientation_matrix = link_Orientation_matrix
        self.link_Constants_symbolical = link_Constants_symbolical

        # IMPORTANT NOTE: These definitions are not used in numerical evaluations later on,
        # These are for print and checking if input values are wrong.
        self.link_Constants_numerical = None
        self.link_Mass_numerical = None
        self.link_MoI_tensor_numerical = None
        print(bcolors.OKGREEN + "Link object '{}' is created.".format(self.link_Name) + bcolors.ENDC)

        # Calculate translational jacobian:
        self.Jv = self.link_CoG_vector.jacobian(generalized_coordinates)
        # Calculate rotational jacobian:
        self.Jw = Rotational_jacobian(self.link_Orientation_matrix, generalized_coordinates)

    def print_link_info(self):
        if(self.link_Constants_numerical == None):
            print(bcolors.WARNING + '========================================================================================' + bcolors.ENDC)
            print(bcolors.WARNING + '=============================== Link Object Informations ===============================' + bcolors.ENDC)
            print(bcolors.FAIL + 'Numerical values of link constants are not provided yet!! Printing with symbols.' + bcolors.ENDC)
            print(bcolors.UNDERLINE + 'Call "provide_link_constant_numerical_values()" function to input numerical values.'+ bcolors.ENDC)
            print('Link Name: {}'.format(self.link_Name))
            print('Link Descriptions: {}'.format(self.link_Notes))
            print('Link Mass: {}'.format(self.link_Mass))
            print('Link Moment of Inertia tensor: ')
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.link_MoI_tensor))))
    
            print('Link Center of Gravity vector: ')
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.link_CoG_vector))))
            
            print('Link Orientation matrix: ')        
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.link_Orientation_matrix))))
    
            print('Link Linear Velocity Jacobian: ')        
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.Jv))))
    
            print('Link Angular Velocity Jacobian: ')        
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.Jw))))
            print(bcolors.WARNING + '=============================== Link Object Informations ===============================' + bcolors.ENDC)
            print(bcolors.WARNING + '========================================================================================' + bcolors.ENDC)

        else:
            print(bcolors.WARNING + '========================================================================================' + bcolors.ENDC)
            print(bcolors.WARNING + '=============================== Link Object Informations ===============================' + bcolors.ENDC)
            print(bcolors.OKCYAN + 'Numerical values of link constants are provided. Printing with values in SI units.' + bcolors.ENDC)
            print('Link Name: {}'.format(self.link_Name))
            print('Link Descriptions: {}'.format(self.link_Notes))
            print('Link Mass: {}kg'.format(self.link_Mass_numerical))
            print('Link Moment of Inertia tensor(kg*m^2): ')
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.link_MoI_tensor_numerical))))
    
            print('Link Center of Gravity vector: ')
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.link_CoG_vector))))
            
            print('Link Orientation matrix: ')        
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.link_Orientation_matrix))))
    
            print('Link Linear Velocity Jacobian: ')        
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.Jv))))
    
            print('Link Angular Velocity Jacobian: ')        
            display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.Jw))))
            print(bcolors.WARNING + '=============================== Link Object Informations ===============================' + bcolors.ENDC)
            print(bcolors.WARNING + '========================================================================================' + bcolors.ENDC)

    def provide_link_constant_numerical_values(self, link_Constants_numerical):
        # link_Constant_symbol, link_Constant_numerical length check:
        if len(self.link_Constants_symbolical) != len(link_Constants_numerical):
            print(bcolors.FAIL + 'Input numerical values list has different dimension with link_Constant_symbols!!!' + bcolors.ENDC)
            print(bcolors.FAIL + 'Please double check the length of link_Constant_symbol & link_Constant_numerical!!' + bcolors.ENDC)
            return 
        
        print(bcolors.FAIL + 'Please be sure that link_Constant is proveded in form of: {} !!'.format(self.link_Constants_symbolical)  )
        self.link_Constants_numerical = link_Constants_numerical
        self.link_Mass_numerical = self.link_Constants_numerical[0]
        self.link_MoI_tensor_numerical = Create_Moment_of_Inertia_tensor(*tuple(self.link_Constants_numerical[2:8]))
        
        print(bcolors.BOLD + bcolors.OKCYAN + 'Printing the numerical values. Please double check if they are correct.' + bcolors.ENDC)
        print('Link mass {} = {}kg'.format(self.link_Mass, self.link_Mass_numerical))
        print('Link length from joint with its parent link lc_i = {}m'.format(self.link_Constants_numerical[1]))
        print('Moment of inertia tensor (kg*m^2): ')        
        display(Math('%s %s' % ('\hspace{4.5cm}', latex(self.link_MoI_tensor_numerical))))
            
    def delete_link_Constant_numerical_values(self):
        if (self.link_Constants_numerical == None):
            print(bcolors.FAIL + 'Numerical value are not defined yet.' + bcolors.ENDC)
            print(bcolors.FAIL + 'Please call "provide_link_constant_numerical_values()" function to redefine the values.' + bcolors.ENDC)
            return
            
        self.link_Constants_numerical = None
        self.link_Mass_numerical = None
        self.link_MoI_tensor_numerical = None
        print(bcolors.OKCYAN + 'Numerical values of link constants are deleted!' + bcolors.ENDC)
        print(bcolors.OKCYAN + 'Please call "provide_link_constant_numerical_values()" function to redefine the values.' + bcolors.ENDC)

    def Calculate_mass_matrix_components(self):
        # generalized_coordinate_space_dimension:
        dim_gene_space = shape(self.Jv)[1]
        # mass matrix component from translational inertia
        Mass_mat_1 = self.Jv.T  * self.Jv * self.link_Mass
        
        # mass matrix component from rotational inertia
        Mass_mat_2 = self.Jw.T * self.link_MoI_tensor * self.Jw

        Mass_matrix_component = Mass_mat_1 + Mass_mat_2
        Mass_matrix_component.simplify()
        return Mass_matrix_component

    # Pass in the gravitational acceleration vector in xyz coordinate. Example: y direction -g accele: grav_acc_vec=Matrix([0,-g,0])
    def Calculate_generalized_gravitational_force_vector_component(self,gravitational_acc_vector):
        # Gravitational force acts on the center of the link,
        # It can be calculated with link mass center 'cartesian position vector -> generalized coordinate' jacobian
        # F_grav = Jv.T * g_acc_vector * mass
        Grav_force_vector_componet = self.Jv.T * gravitational_acc_vector * self.link_Mass
        Grav_force_vector_componet.simplify()
        return Grav_force_vector_componet

