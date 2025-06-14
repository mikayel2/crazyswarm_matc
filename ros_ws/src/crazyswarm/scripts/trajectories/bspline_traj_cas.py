import casadi as ca
import numpy as np
from scipy import interpolate
import numpy as np

g_vec = ca.vertcat(0, 0, 9.81)

def spline_from_csv(path, s=0.0):
    data = np.loadtxt(path, delimiter=',', skiprows=1)   # x,y,z
    tck, _ = interpolate.splprep(data.T, s=s, k=3)       # cubic
    return tck                                           # knots, coeffs, k


class BsplineTrajCas:
    """Differential-flat B-spline trajectory driven by CSV way-points."""
    def __init__(self, csv_path, duration):
        knots, coeffs, k = spline_from_csv(csv_path)
        coeffs = np.array(coeffs).T                      # N×3
        self.T = duration

        # CasADi spline interpolant: p(u) with u ∈ [0,1]
        self._p = ca.bspline(knots, coeffs)

        t = ca.MX.sym('t')
        p   = self._p(t/self.T)
        v   = ca.gradient(p, t)
        a   = ca.gradient(v, t)
        j   = ca.gradient(a, t)

        self._pos  = ca.Function('p', [t], [p])
        self._vel  = ca.Function('v', [t], [v])
        self._acc  = ca.Function('a', [t], [a])
        self._jerk = ca.Function('j', [t], [j])

    @staticmethod
    def _norm(v): return v/ca.norm_2(v)

    def full_state(self, t):
        p, v, a, j = [f(t) for f in (self._pos, self._vel,
                                      self._acc, self._jerk)]
        yaw, dyaw = 0.0, 0.0
        thrust = a + g_vec
        z_b = self._norm(thrust)
        x_w = ca.vertcat(1,0,0)
        y_b = self._norm(ca.cross(ca.vertcat(0,0,1), x_w))
        x_b = ca.cross(y_b, z_b)
        j_orth = j - ca.dot(j, z_b)*z_b
        h_w = j_orth / ca.norm_2(thrust)
        omega = ca.vertcat(-ca.dot(h_w, y_b),
                            ca.dot(h_w, x_b),
                            z_b[2]*dyaw)
        return p, v, a, yaw, omega
