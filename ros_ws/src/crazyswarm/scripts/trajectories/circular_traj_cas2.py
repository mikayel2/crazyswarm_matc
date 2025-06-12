import casadi as ca
import numpy as np

class cir_traj_2():
    def __init__(self,coord = [0,0,1],r = 1,rot_rate = 3.14,init_angle = 0.0, rot_dir = 1):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        self.r = r
        self.rot_rate = rot_rate
        self.init_angle = init_angle
        self.rot_dir = rot_dir

    def normalize(self, v):
        norm = ca.norm_2(v)
        #assert norm > 0
        return v / norm

    def full_state(self, t):
        # Define symbolic variables for time and constants
        #t = ca.SX(t)
        r = self.r
        rot_dir = self.rot_dir
        rot_rate = self.rot_rate
        init_angle = self.init_angle
        x = self.x
        y = self.y

        # Position
        x = r * ca.cos(rot_dir*rot_rate*t + init_angle) + x
        y = -r * ca.sin(rot_dir*rot_rate*t + init_angle) + y
        z = self.z
        position = ca.vertcat(x, y, z)

        # Velocity
        vx = r * rot_rate * ca.sin(rot_dir*rot_rate*t + init_angle)        
        vy = r * rot_rate * ca.cos(rot_dir*rot_rate*t + init_angle)
        vz = 0.0
        velocity = ca.vertcat(vx, vy, vz)

        # Acceleration
        ax = r * (rot_rate ** 2) * ca.cos(rot_dir*rot_rate*t + init_angle)
        ay = -r * (rot_rate ** 2) * ca.sin(rot_dir*rot_rate*t + init_angle)
        az = 0.0
        acceleration = ca.vertcat(ax, ay, az)

        # Jerk
        jerk_x = -r * (rot_rate ** 3) * ca.sin(rot_dir*rot_rate*t + init_angle)
        jerk_y = -r * (rot_rate ** 3) * ca.cos(rot_dir*rot_rate*t + init_angle)
        jerk_z = 0.0
        jerk = ca.vertcat(jerk_x, jerk_y, jerk_z)

        # Yaw and its rate
        yaw = 0.0
        dyaw = 0.0

        # Thrust
        thrust = acceleration + ca.vertcat(0, 0, 9.81)

        # Body axes
        z_body = self.normalize(thrust)
        x_world = ca.vertcat(ca.cos(yaw), ca.sin(yaw), 0)
        y_body = self.normalize(ca.cross(ca.vertcat(0, 0, 1), x_world))
        x_body = ca.cross(y_body, z_body)

        # Orthogonal jerk to z_body
        jerk_orth_zbody = jerk - ca.dot(jerk, z_body) * z_body
        h_w = jerk_orth_zbody / ca.norm_2(thrust)

        # Omega
        omega = ca.vertcat(-ca.dot(h_w, y_body), ca.dot(h_w, x_body), z_body[2] * dyaw)

        return position, velocity, acceleration, yaw, omega

if __name__ == "__main__":
    #coord = [0,0,1],r = 1,rot_rate = 3.14,init_angle = 0.0, rot_dir = 1):
    #tray = cir_traj_2([-0.2, 0.0 , 1.0],0.7, 0.1570796, 1.570795, 1)
    #tray = cir_traj_2([-0.4, 0.0, 1.0],0.9, 0.1570796, 1.570795, 1)
    #tray = cir_traj_2([-0.6, 0.0, 1.0],1.1, 0.1570796, 1.570795, 1)
    #tray = cir_traj_2([0.0, 0.0, 1.0],0.6, 0.314159, 1.570795, 1)
    tray = cir_traj_2([0.0, 0.0, 1.0],1.2, 0.314159, 1.570795, 1)
    x, v, acc, yaw, omega = tray.full_state(5.0)
    print("Position:", x)
    print("Velocity:", v)
    print("Acceleration:", acc)
    print("Yaw:", yaw)
    print("Omega:", omega)

