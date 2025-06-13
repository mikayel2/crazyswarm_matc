# trajectories/wave_traj_cas.py
# -------------------------------------------------------------
import casadi as ca
import numpy as np

g_vec = ca.vertcat(0, 0, 9.81)        # gravity in +z  (m s-2)


class WaveTrajCas:
    """
    Smooth sinusoidal lane (x forward, y oscillates, z constant).

    ─ State equations ────────────────────────────────────────────
        x(t) = v_x · t
        y(t) =  A · sin(2π x / λ) + y_offset
        z(t) =  z0
    ───────────────────────────────────────────────────────────────

    full_state(t) returns:
        pos, vel, acc, yaw, omega      (same order/type as cir_traj_2)
    """

    def __init__(self, *,
                 y_offset      = 0.0,
                 amplitude     = 2.0,
                 wavelength    = 12.0,
                 forward_speed = 1.5,
                 z0            = 1.0):

        self.A   = amplitude
        self.lam = wavelength
        self.vx  = forward_speed
        self.y0  = y_offset
        self.z0  = z0

        self.k   = 2*np.pi / self.lam           # wave-number
        self.Ak  = self.A * self.k
        self.Ak2 = self.Ak * self.k
        self.Ak3 = self.Ak2 * self.k

    # ----------------------------------------------------------
    @staticmethod
    def _norm(v):       # fast helper
        return v / ca.norm_2(v)

    # ----------------------------------------------------------
    def full_state(self, t):
        # ------------ position
        x = self.vx * t
        y = self.A * ca.sin(self.k * x) + self.y0
        z = self.z0
        pos = ca.vertcat(x, y, z)

        # ------------ velocity
        vx = self.vx
        vy = self.Ak * self.vx * ca.cos(self.k * x)
        vel = ca.vertcat(vx, vy, 0)

        # ------------ acceleration
        ax = 0.0
        ay = -self.Ak2 * self.vx**2 * ca.sin(self.k * x)
        acc = ca.vertcat(ax, ay, 0)

        # ------------ jerk
        jx = 0.0
        jy = -self.Ak3 * self.vx**3 * ca.cos(self.k * x)
        jerk = ca.vertcat(jx, jy, 0)

        # ------------ attitude & body rates  (keep heading +x)
        yaw  = 0.0
        dyaw = 0.0

        thrust = acc + g_vec
        z_b = self._norm(thrust)

        x_w = ca.vertcat(1, 0, 0)              # world x-axis (yaw=0)
        y_b = self._norm(ca.cross(ca.vertcat(0, 0, 1), x_w))
        x_b = ca.cross(y_b, z_b)

        jerk_orth = jerk - ca.dot(jerk, z_b) * z_b
        h_w = jerk_orth / ca.norm_2(thrust)

        omega = ca.vertcat(-ca.dot(h_w, y_b),
                            ca.dot(h_w, x_b),
                            z_b[2]*dyaw)

        return pos, vel, acc, yaw, omega


# -----------------------------------------------------------------
def build_parallel_waves(n_curves=8, spacing=4.0, **wave_kw):
    """Return a list of WaveTrajCas objects, each with a y-offset."""
    offsets = spacing * (np.arange(n_curves) - (n_curves-1)/2)
    return [WaveTrajCas(y_offset=o, **wave_kw) for o in offsets]
