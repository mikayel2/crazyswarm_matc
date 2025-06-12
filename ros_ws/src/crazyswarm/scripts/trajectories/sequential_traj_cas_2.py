import casadi as ca
import numpy as np


class seq_traj_cas_2():
    def __init__(self, circle_center_coordiantes, radius, rot_rate ,init_angle, rot_dir, duration):
        self.x = circle_center_coordiantes[0][0]
        self.y = circle_center_coordiantes[0][1]
        self.z = circle_center_coordiantes[0][2]
        self.r = radius[0]
        self.rot_rate = rot_rate[0]
        self.init_angle = init_angle[0]
        self.rot_dir = rot_dir[0]

        self.xyz_all = circle_center_coordiantes
        self.r_all = radius
        self.rot_rate_all = rot_rate
        self.init_angle_all = init_angle
        self.rot_dir_all= rot_dir
        self.duration = duration

    def normalize(self, v):
        norm = ca.norm_2(v)
        #assert norm > 0
        return v / norm

    def full_state(self, t):
        # Define symbolic variables for time and constants
        #t = ca.SX(t)
        
        r = ca.if_else(t<=self.duration[0], self.r_all[0], 
                ca.if_else(t<=self.duration[1], self.r_all[1],
                           ca.if_else(t<=self.duration[2], self.r_all[2],
                                      ca.if_else(t<=self.duration[3], self.r_all[3], self.r_all[4]))))
                       
        rot_dir = ca.if_else(t<=self.duration[0], self.rot_dir_all[0],
                             ca.if_else(t<=self.duration[1], self.rot_dir_all[1],
                                        ca.if_else(t<=self.duration[2], self.rot_dir_all[2], 
                                                    ca.if_else(t<=self.duration[3], self.rot_dir_all[3], self.rot_dir_all[4]))))

        rot_rate = ca.if_else(t<=self.duration[0], self.rot_rate_all[0],
                              ca.if_else(t<=self.duration[1], self.rot_rate_all[1],
                                            ca.if_else(t<=self.duration[2], self.rot_rate_all[2],
                                                        ca.if_else(t<=self.duration[3], self.rot_rate_all[3], self.rot_rate_all[4]))))
         
        init_angle = ca.if_else(t<=self.duration[0], self.init_angle_all[0],
                                ca.if_else(t<=self.duration[1], self.init_angle_all[1],
                                             ca.if_else(t<=self.duration[2], self.init_angle_all[2], 
                                                        ca.if_else(t<=self.duration[3], self.init_angle_all[3], self.init_angle_all[4]))))
        
        x_init =  ca.if_else(t<=self.duration[0], self.xyz_all[0][0],
                             ca.if_else(t<=self.duration[1], self.xyz_all[1][0],
                                        ca.if_else(t<=self.duration[2], self.xyz_all[2][0],
                                                    ca.if_else(t<=self.duration[3], self.xyz_all[3][0],self.xyz_all[4][0]))))

        y_init =  ca.if_else(t<=self.duration[0], self.xyz_all[0][1],
                             ca.if_else(t<=self.duration[1], self.xyz_all[1][1],
                                        ca.if_else(t<=self.duration[2], self.xyz_all[2][1],
                                                    ca.if_else(t<=self.duration[3], self.xyz_all[3][1], self.xyz_all[4][1]))))
        
        t = ca.if_else(t<=self.duration[0], t,
                          ca.if_else(t<=self.duration[1], t - self.duration[0],
                                        ca.if_else(t<=self.duration[2], t - self.duration[1], 
                                                    ca.if_else(t<=self.duration[3], t - self.duration[2], t - self.duration[3]))))
                       

        # Position
        x = r * ca.cos(rot_dir*rot_rate*t + init_angle) + x_init
        y = -r * ca.sin(rot_dir*rot_rate*t + init_angle) + y_init
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
    tray1 = cir_traj_2(1.0, 1.0, 3.14 / 5)
    tray = cir_traj_2(1, 0, 1.0)
    x, v, acc, yaw, omega = tray.full_state(0)
    print("Position:", x)
    print("Velocity:", v)
    print("Acceleration:", acc)
    print("Yaw:", yaw)
    print("Omega:", omega)

