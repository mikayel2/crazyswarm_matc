import numpy as np
import matplotlib.pyplot as plt
import math

class circ_traj():
    def __init__(self,coord = [0.0, 2.0, 1.0],r = 1,rot_rate = 0.314159,init_angle = -1.57, rot_dir = 1):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        self.r = r
        self.rot_rate = rot_rate
        self.init_angle = init_angle
        self.rot_dir = rot_dir

    def full_state_at_initial_time(self):
        x = -self.r * np.cos(self.init_angle) + self.x
        y = self.r * np.sin(self.init_angle) + self.y
        z = self.z
        return np.array([x,y,z])

    def normalize(self,v):
        norm = np.linalg.norm(v)
        assert norm > 0
        return v / norm

    def full_state(self,t):
        x = -self.r * np.cos(self.rot_dir*self.rot_rate*t + self.init_angle) + self.x
        y = self.r * np.sin(self.rot_dir*self.rot_rate* t + self.init_angle) + self.y
        z = self.z
        x = np.array([x,y,z])

        vx = self.r * self.rot_rate * np.sin(self.rot_dir*self.rot_rate*t + self.init_angle)
        vy = self.r * self.rot_rate * np.cos(self.rot_dir*self.rot_rate*t + self.init_angle)
        vz = 0.0
        v = np.array([vx ,vy, vz])

        ax = self.r * (self.rot_dir*self.rot_rate**2) * np.cos(self.rot_dir*self.rot_rate*t + self.init_angle)
        ay = - self.r * (self.rot_dir*self.rot_rate**2) * np.sin(self.rot_dir*self.rot_rate*t + self.init_angle)
        az = 0.0
        acc = np.array([ax ,ay, az])

        jerk_x = -self.r * (self.rot_dir*self.rot_rate**3) * np.sin(self.rot_dir*self.rot_rate*t + self.init_angle)
        jerk_y = -self.r * (self.rot_dir*self.rot_rate**3) * np.cos(self.rot_dir*self.rot_rate*t + self.init_angle)
        jerk_z = 0.0
        jerk = np.array([jerk_x, jerk_y, jerk_z])

        yaw = 0.0
        dyaw = 0.0

        thrust = acc + np.array([0, 0, 9.81])

        z_body= self.normalize(thrust)
        x_world = np.array([np.cos(yaw), np.sin(yaw), 0])
        y_body = self.normalize(np.cross(z_body, x_world))
        x_body = np.cross(y_body, z_body)

        jerk_orth_zbody = jerk - (np.dot(jerk, z_body) * z_body)
        h_w = jerk_orth_zbody / np.linalg.norm(thrust)

        omega = np.array([-np.dot(h_w, y_body), np.dot(h_w, x_body), z_body[2] * dyaw])

        return x , v, acc, yaw, omega
        

if __name__ == "__main__":
    test_traj= circ_traj()
    x,v,acc,yaw,omega = test_traj.full_state(5)
    print(x,v,acc,yaw,omega)