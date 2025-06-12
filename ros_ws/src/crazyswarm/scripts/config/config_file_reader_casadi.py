import yaml

class TimeCoordinationConfig:
    def __init__(self, config_file='config/time_coordination_config_casadi.yaml'):
        self.config_file = config_file
        config = self.read_yaml(self.config_file)

        self.rate = config.get('rate', 20.0) 
        self.flight_duration = config.get('flight_duration', 5.0) 
        self.num_agents = config.get('num_agents', 3) 
        self.log_data = config.get('log_data', True)
        
        self.takeoff_target_height = config.get('takeoff_target_height', 1.0)
        self.takeoff_duration = config.get('takeoff_duration', 2.5)
        self.hover_duration = config.get('hover_duration', 5.0)
        self.land_duration = config.get('land_duration', 2.5)
        self.land_target_height = config.get('land_target_height', 0.02)
        self.sleep_duration_before_shutdown = config.get('sleep_duration_before_shutdown', 4.0)


        # TODO: given default values for the following parameters does not have the same size as the original code
        # Circular trajectories
        self.radius = config.get('radius', 1.0)
        self.circle_center_coordiantes = config.get('circle_center_coordiantes', [0.0, 0.0, 1.0])
        self.init_angle = config.get('init_angle', 0.0)
        self.angular_velocity = config.get('angular_velocity', 0.5)
        self.rotation_direction = config.get('rotation_direction', 1)

        # Casadi parameters
        self.nx = config.get('nx', 2)  # Number of states - \gamma and \dot{\gamma}
        self.nu = config.get('nu', 1)  # Number of control inputs - \ddot{\gamma}
        self.K = config.get('K', 20)  # Prediction horizon
        self.T = config.get('T', 1000)  # Simulation time
        self.h = config.get('h', 0.1)  # Time step
        self.u_min = config.get('u_min', [-15])
        self.u_max = config.get('u_max', [15])
        self.x_min = config.get('x_min', [0, 0])
        self.x_max_up = config.get('x_max_up', 2)
        self.du = config.get('du', 5)  # Distance between agents for smoothing function


        # Communication term
        self.du11 = config.get('du11', 5.0)
        self.du12 = config.get('du12', 2.5)

        # Collision Avoidance
        self.du21 = config.get('du21', 0.25)
        self.du22 = config.get('du22', 0.125)
        self.du31 = config.get('du31', 0.25)
        self.du32 = config.get('du32', 0.125)

        # Pace keeping term
        self.dupc = config.get('dupc', 2.0)


        self.gamma_init = config.get('gamma_init', [0.0, 0.0, 0.0])



        # Trajectories with phaseshift - Yin-Yang
        self.radius_1 = config.get('radius_1', 1.0)
        self.circle_center_coordiantes_1 = config.get('circle_center_coordiantes_1', [0.0, 0.0, 1.0])
        self.init_angle_1 = config.get('init_angle_1', 0.0)
        self.angular_velocity_1 = config.get('angular_velocity_1', 0.5)
        self.rotation_direction_1 = config.get('rotation_direction_1', 1)
        self.duration_1 = config.get('duration_1', 5.0)

        self.radius_2 = config.get('radius_2', 1.0)
        self.circle_center_coordiantes_2 = config.get('circle_center_coordiantes_2', [0.0, 0.0, 1.0])
        self.init_angle_2 = config.get('init_angle_2', 0.0)
        self.angular_velocity_2 = config.get('angular_velocity_2', 0.5)
        self.rotation_direction_2 = config.get('rotation_direction_2', 1)
        self.duration_2 = config.get('duration_2', 5.0)


        self.radius_21 = config.get('radius_21', 1.0)
        self.circle_center_coordiantes_21 = config.get('circle_center_coordiantes_21', [0.0, 0.0, 1.0])
        self.init_angle_21 = config.get('init_angle_21', 0.0)
        self.angular_velocity_21 = config.get('angular_velocity_21', 0.5)
        self.rotation_direction_21 = config.get('rotation_direction_21', 1)
        self.duration_21 = config.get('duration_21', 5.0)

        self.radius_22 = config.get('radius_22', 1.0)
        self.circle_center_coordiantes_22 = config.get('circle_center_coordiantes_22', [0.0, 0.0, 1.0])
        self.init_angle_22 = config.get('init_angle_22', 0.0)
        self.angular_velocity_22 = config.get('angular_velocity_22', 0.5)
        self.rotation_direction_22 = config.get('rotation_direction_22', 1)
        self.duration_22 = config.get('duration_22', 5.0)

        self.radius_23 = config.get('radius_23', 1.0)
        self.circle_center_coordiantes_23 = config.get('circle_center_coordiantes_23', [0.0, 0.0, 1.0])
        self.init_angle_23 = config.get('init_angle_23', 0.0)
        self.angular_velocity_23 = config.get('angular_velocity_23', 0.5)
        self.rotation_direction_23 = config.get('rotation_direction_23', 1)
        self.duration_23 = config.get('duration_23', 5.0)

        self.radius_24 = config.get('radius_24', 1.0)
        self.circle_center_coordiantes_24 = config.get('circle_center_coordiantes_24', [0.0, 0.0, 1.0])
        self.init_angle_24 = config.get('init_angle_24', 0.0)
        self.angular_velocity_24 = config.get('angular_velocity_24', 0.5)
        self.rotation_direction_24 = config.get('rotation_direction_24', 1)
        self.duration_24 = config.get('duration_24', 5.0)
        

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
