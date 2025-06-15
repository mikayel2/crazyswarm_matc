import casadi as ca
import numpy as np                     # ✅ single import (duplicate removed)
from scipy import interpolate
from pathlib import Path

g_vec = ca.vertcat(0, 0, 9.81)

# ----------------------------------------------------------------------
class BsplineTrajCas:
    """
    CSV → cubic B-spline → CasADi flat-output trajectory.

    full_state(t) returns (pos, vel, acc, yaw, omega).
    """

    # 🔸  Added `samples` so you control spline resolution
    def __init__(self, csv_path, duration, samples=300):
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)  # N × 3
        if data.shape[1] != 3:
            raise ValueError("CSV must have three columns x,y,z")

        # 🔸  Fit SciPy spline *once* and sample it uniformly
        tck, _   = interpolate.splprep(data.T, s=0, k=3)
        grid_u   = np.linspace(0.0, 1.0, samples)
        samples_xyz = np.vstack(interpolate.splev(grid_u, tck))  # 3 × N

        # 🔸  Build CasADi interpolant (works with DM)
        self._p = ca.interpolant(
            "p", "bspline",
            [grid_u.tolist()],           # grid must be list of floats
            ca.DM(samples_xyz)           # values as CasADi DM 3×N
        )

        self.T = float(duration)

        # ---------- symbolic derivatives --------------------------
        t = ca.MX.sym("t")
        u = t / self.T
        p = self._p(u)

        # 🔸  Compute vector derivatives component-wise
        vx = ca.gradient(p[0], t)
        vy = ca.gradient(p[1], t)
        vz = ca.gradient(p[2], t)
        v  = ca.vertcat(vx, vy, vz)

        ax = ca.gradient(v[0], t)
        ay = ca.gradient(v[1], t)
        az = ca.gradient(v[2], t)
        a  = ca.vertcat(ax, ay, az)

        jx = ca.gradient(a[0], t)
        jy = ca.gradient(a[1], t)
        jz = ca.gradient(a[2], t)
        j  = ca.vertcat(jx, jy, jz)

        # store compiled functions
        self._pos  = ca.Function("pos",  [t], [p])
        self._vel  = ca.Function("vel",  [t], [v])
        self._acc  = ca.Function("acc",  [t], [a])
        self._jerk = ca.Function("jerk", [t], [j])

    # --------------------------------------------------------------
    @staticmethod
    def _norm(v):
        return v / ca.norm_2(v)

    # --------------------------------------------------------------
    def full_state(self, t):
        p, v, a, j = (f(t) for f in
                      (self._pos, self._vel, self._acc, self._jerk))

        yaw, dyaw = 0.0, 0.0

        thrust = a + g_vec
        z_b = self._norm(thrust)

        x_w = ca.vertcat(1, 0, 0)              # yaw = 0 → +x
        y_b = self._norm(ca.cross(ca.vertcat(0, 0, 1), x_w))
        x_b = ca.cross(y_b, z_b)

        j_orth = j - ca.dot(j, z_b) * z_b
        h_w    = j_orth / ca.norm_2(thrust)

        omega = ca.vertcat(-ca.dot(h_w, y_b),
                            ca.dot(h_w, x_b),
                            z_b[2] * dyaw)

        return p, v, a, yaw, omega
