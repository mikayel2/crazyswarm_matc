import casadi as ca
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from  trajectories.circular_traj_cas2 import cir_traj_2
from config.config_file_reader_casadi import TimeCoordinationConfig


class seq_traj():
    def __init__(self, config):
        self.config = config
        self.trajectories = []

        for i in range(len(config.duration_1)):
             traj = cir_traj_2(self.config.circle_center_coordiantes_1[i],self.config.radius_1[i],self.config.angular_velocity_1[i] ,self.config.init_angle_1[i], self.config.rotation_direction_1[i])
             self.trajectories.append(traj)

    def full_state(self, t):
        
        x, v, acc, yaw, omega = self.trajectories[0].full_state(t)
        
        x, v, acc, yaw, omega = self.trajectories[1].full_state(t) 

       
                   

        # Evaluate the duration array to get numerical values
        duration_values = self.config.duration_1
        i = next((i for i, duration in enumerate(duration_values) if duration > t), -1)
        #if t < self.config.duration_1[0]:
        x, v, acc, yaw, omega = self.trajectories[0].full_state(t)
        #x, v, acc, yaw, omega = self.trajectories[i].full_state(t) 
       
        
        #if t < self.config.duration_1[0]:
        #    x, v, acc, yaw, omega = self.trajectories[1].full_state(t)
        #elif t < self.config.duration_1[0] + self.config.duration_1[1]:
        #    x, v, acc, yaw, omega = self.trajectories[1].full_state(t - self.config.duration_1[0])
        

        

        return x, v, acc, yaw, omega 

        



if __name__ == "__main__":
     
    config = TimeCoordinationConfig()
    seq_traj = seq_traj(config)

    x, v, acc, yaw, omega  = seq_traj.full_state(14)
    print("Position:", x)
    print("Velocity:", v)
    print("Acceleration:", acc)
    print("Yaw:", yaw)
    print("Omega:", omega)