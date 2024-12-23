from .basic_math import *
import os

class data_logger():
    def __init__(self):
        self.raw_integration_result = None
        
        self.t_n = []
        self.current_state_vector_n = []
        self.target_state_vector_n = []
        self.current_control_mode = []
        self.current_control_vector_n = []
        
        self.current_mechanical_energy = []        
        self.current_kinetic_energy = []
        self.current_potential_energy = []

        self.target_mechanical_energy = []
        self.target_kinetic_energy = []
        self.target_potential_energy = []
        
        self.current_energy_error = []

        self.ground_reaction_force = []
        self.Lyapunov_stable_condition = []
        print('Data logger is created.')

        print(bcolors.FAIL + '=================================================================================================' + bcolors.ENDC)
        print(bcolors.BOLD + '                                 Data logger is created.' + bcolors.ENDC)
        print('')
        print(bcolors.BOLD + '                             Use below index words to log data.' + bcolors.ENDC)
        print('')
        print(bcolors.BOLD + "                                't_n'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'current_state_vector_n'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'target_state_vector_n'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'current_control_mode'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'current_control_vector_n'" + bcolors.ENDC)

        print(bcolors.BOLD + "                                'current_mechanical_energy'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'current_kinetic_energy'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'current_potential_energy'" + bcolors.ENDC)

        print(bcolors.BOLD + "                                'target_mechanical_energy'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'target_kinetic_energy'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'target_potential_energy'" + bcolors.ENDC)

        print(bcolors.BOLD + "                                'current_energy_error'" + bcolors.ENDC)

        print(bcolors.BOLD + "                                'ground_reaction_force'" + bcolors.ENDC)
        print(bcolors.BOLD + "                                'Lyapunov_stable_condition'" + bcolors.ENDC)

        print(bcolors.FAIL + '=================================================================================================\n' + bcolors.ENDC)


    def clear_log(self):
        self.t_n = []
        self.current_state_vector_n = []
        self.target_state_vector_n = []
        self.current_control_mode = []
        self.current_control_vector_n = []
        
        self.current_mechanical_energy = []        
        self.current_kinetic_energy = []
        self.current_potential_energy = []

        self.target_mechanical_energy = []
        self.target_kinetic_energy = []
        self.target_potential_energy = []
        
        self.current_energy_error = []

        self.ground_reaction_force = []
        self.Lyapunov_stable_condition = []

        print('Stored data is cleared!')

    ############################################### Index words for appending data
    # self.t_n = []             
    # self.current_state_vector_n = []
    # self.target_state_vector_n = []
    # self.current_control_mode = []
    # self.current_control_vector_n = []
    
    # self.current_mechanical_energy = []        
    # self.current_kinetic_energy = []
    # self.current_potential_energy = []

    # self.target_mechanical_energy = []
    # self.target_kinetic_energy = []
    # self.target_potential_energy = []
    
    # self.current_energy_error = []

    # self.ground_reaction_force = []
    # self.Lyapunov_stable_condition = []
    ##############################################################################


    
    ###############################################
    def data_append(self, index_word, value):
        if index_word == 't_n':
            self.t_n.append(value)
        if index_word == 'current_state_vector_n':
            self.current_state_vector_n.append(value)
        if index_word == 'target_state_vector_n':
            self.target_state_vector_n.append(value)
        if index_word == 'current_control_mode':
            self.current_control_mode.append(value)
        if index_word == 'current_control_vector_n':
            self.current_control_vector_n.append(value)
            
        if index_word == 'current_mechanical_energy':
            self.current_mechanical_energy.append(value)
        if index_word == 'current_kinetic_energy':
            self.current_kinetic_energy.append(value)
        if index_word == 'current_potential_energy':
            self.current_potential_energy.append(value)

        if index_word == 'target_mechanical_energy':
            self.target_mechanical_energy.append(value)
        if index_word == 'target_kinetic_energy':
            self.target_kinetic_energy.append(value)
        if index_word == 'target_potential_energy':
            self.target_potential_energy.append(value)

        if index_word == 'current_energy_error':
            self.current_energy_error.append(value)

        if index_word == 'ground_reaction_force':
            self.ground_reaction_force.append(value)
        if index_word == 'Lyapunov_stable_condition':
            self.Lyapunov_stable_condition.append(value)
    
    def data_save(self, simulation_number):
        self.save_dir_name = 'Simulation No{}'.format(simulation_number)
        os.mkdir('./visualized/{}'.format(self.save_dir_name))
        save_path = './visualized/{}/Simulation No{}'.format(self.save_dir_name, simulation_number)
        # Save raw data during integration
        np.save(save_path, np.array([self.t_n ,
                                    self.current_state_vector_n ,
                                    self.target_state_vector_n ,
                                    self.current_control_mode ,
                                    self.current_control_vector_n ,
                                    self.current_mechanical_energy ,
                                    self.current_kinetic_energy ,
                                    self.current_potential_energy ,
                                    self.target_mechanical_energy ,
                                    self.target_kinetic_energy ,
                                    self.target_potential_energy ,
                                    self.current_energy_error,
                                    self.ground_reaction_force,
                                    self.Lyapunov_stable_condition]))
        # Save raw result
        save_path = './visualized/{}/Simulation Raw data sol No{}'.format(self.save_dir_name, simulation_number)
        if self.raw_integration_result.any() == None:
            print(bcolors.FAIL + 'ERROR. Raw data is not sotred to raw_integration_result !' + bcolors.ENDC)
        np.save(save_path, np.array([self.raw_integration_result]))
        print('Raw data is saved to {}!'.format(save_path))

    def data_load(self, npy_file_path):
        (self.t_n ,
        self.current_state_vector_n ,
        self.target_state_vector_n ,
        self.current_control_mode ,
        self.current_control_vector_n ,
        self.current_mechanical_energy ,
        self.current_kinetic_energy ,
        self.current_potential_energy ,
        self.target_mechanical_energy ,
        self.target_kinetic_energy ,
        self.target_potential_energy ,
        self.current_energy_error,
        self.ground_reaction_force,
        self.Lyapunov_stable_condition) = np.load(npy_file_path,allow_pickle=True)
        print('Data is loaded from {}!'.format(npy_file_path))

    def data_load_raw_sol(self, sol_npy_path):
        self.raw_integration_result = np.load(sol_npy_path, allow_pickle=True)
        print('Raw sol data is loaded to data logger!')