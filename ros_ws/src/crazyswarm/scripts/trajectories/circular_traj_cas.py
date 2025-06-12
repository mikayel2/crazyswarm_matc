import casadi as ca
import numpy as np

class cir_traj():
    def __init__(self, r, z, freq, phase=0.0):
        # Constructor
        self.r = r
        self.z = z
        self.freq = freq
        self.phase = phase

    def normalize(self, v):
        norm = ca.norm_2(v)
        #assert norm > 0
        return v / norm

    def full_state(self, t):
        # Define symbolic variables for time and constants
        #t = ca.SX(t)
        r = self.r
        freq = self.freq
        phase = self.phase

        # Position
        x = -r * ca.cos(freq * (t - phase))
        y = r * ca.sin(freq * (t - phase))
        z = self.z
        position = ca.vertcat(x, y, z)

        # Velocity
        vx = r * freq * ca.sin(freq * (t - phase))
        vy = r * freq * ca.cos(freq * (t - phase))
        vz = 0.0
        velocity = ca.vertcat(vx, vy, vz)

        # Acceleration
        ax = r * (freq ** 2) * ca.cos(freq * (t - phase))
        ay = -r * (freq ** 2) * ca.sin(freq * (t - phase))
        az = 0.0
        acceleration = ca.vertcat(ax, ay, az)

        # Jerk
        jerk_x = -r * (freq ** 3) * ca.sin(freq * (t - phase))
        jerk_y = -r * (freq ** 3) * ca.cos(freq * (t - phase))
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
    tray1 = cir_traj(1.0, 1.0, 3.14 / 5)
    tray = cir_traj(1, 0, 1.0)
    x, v, acc, yaw, omega = tray.full_state(0)
    print("Position:", x)
    print("Velocity:", v)
    print("Acceleration:", acc)
    print("Yaw:", yaw)
    print("Omega:", omega)

