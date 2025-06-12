from config.config_file_reader_acados import TimeCoordinationConfig
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trajectories.circular_trajectories import circ_traj


def main():
    config = TimeCoordinationConfig()
    for i in range(config.number_of_drones):
        traj = circ_traj(config.circle_center_coordiantes[i],config.radius[i],config.angular_velocity[i] ,config.init_angle[i], config.rotation_direction[i])
        initial_state = traj.full_state_at_initial_time()
        initial_state_float = [float(f"{x:.6e}") for x in initial_state]
        print(initial_state_float)


if __name__ == "__main__":
    main()