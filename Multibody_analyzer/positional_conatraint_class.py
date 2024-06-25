from sympy import diff
from .basic_math import *

class create_positional_constraint_equation_object():
    def __init__(self, positional_constraint_equation_xyz, t, generalized_coordinates):
        self.positional_constraint_equation = positional_constraint_equation_xyz
        self.t = t
        self.generalized_coordinates = generalized_coordinates
        self.generalized_velocities = diff(self.generalized_coordinates, self.t)
        
        # This constraint equation jacobian is H(q) = dh_vec/dq_vec
        self.positional_constraint_equation_jacobian = self.positional_constraint_equation.jacobian(self.generalized_coordinates)
        
    
        