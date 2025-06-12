import yaml

class TimeCoordinationConfig:
    def __init__(self, config_file='config/time_coordination_config_acados.yaml'):
        self.config_file = config_file
        config = self.read_yaml(self.config_file)

        self.rate = config.get('rate', 20.0) 
        self.flight_duration = config.get('flight_duration', 5.0) 
        self.number_of_drones = config.get('number_of_drones', 3) 
        self.log_data = config.get('log_data', True)
        
        self.takeoff_target_height = config.get('takeoff_target_height', 1.0)
        self.takeoff_duration = config.get('takeoff_duration', 2.5)
        self.hover_duration = config.get('hover_duration', 5.0)
        self.land_duration = config.get('land_duration', 2.5)
        self.land_target_height = config.get('land_target_height', 0.02)
        self.sleep_duration_before_shutdown = config.get('sleep_duration_before_shutdown', 4.0)
        

        self.a = config.get('a', 1.5)
        self.b = config.get('b', 3.6)
        self.delta = config.get('delta', 3.0)
        self.step = config.get('step', 0.02)
        self.d_min = config.get('d_min', 2.0)
        self.d_min_2 = config.get('d_min_2', 1.8)

        # Time coordination parameters
        self.gamma_init = config.get('gamma_init',  [0.0, 5.0, 10.0])
        self.gamma_dot_init = config.get('gamma_dot_init', [0.0, 0.0, 0.0])
        self.l_init = config.get('l_init', [1.0, 0.0, 0.0])
        self.b_LSQ = config.get('b_LSQ', [1.0, 1.0, 1.0])

        # Optimal control problem parameters
        #self.x0 = config.get('x0', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Tf = config.get('Tf', 0.1)  # prediction horizon
        self.N_horizon = config.get('N_horizon', 10)  # number of control intervals
        self.Fmax = config.get('Fmax', 25)  # used only for plotting the results
        self.compile = config.get('compile', False)

        # TODO: given default values for the following parameters does not have the same size as the original code
        # Circular trajectories
        self.radius = config.get('radius', 1.0)
        self.circle_center_coordiantes = config.get('circle_center_coordiantes', [0.0, 0.0, 1.0])
        self.init_angle = config.get('init_angle', 0.0)
        self.angular_velocity = config.get('angular_velocity', 0.5)
        self.rotation_direction = config.get('rotation_direction', 1)

    # Function to read YAML file
    def read_yaml(self,file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data


# Test the class
if __name__ == "__main__":
    config = TimeCoordinationConfig()
    print(config.rate)
    print(config.flight_duration)
    print(config.number_of_drones)
