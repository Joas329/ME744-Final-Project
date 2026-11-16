import numpy as np

from dataclasses import dataclass

@dataclass
class ServoArmModel:
    J: float = 8.4e-4 # kg m^2 --> arm inertia about the joint (URDF: i_cm + m com^2)
    b: float = 0.0 # N m s --> viscous friction (URDF damping)
    mgl: float = 0.090 # N m --> gravity torque at horizontal (URDF: m g com)
    kp: float = 1.2 # N m/rad --> nominal servo stiffness
    kd: float = 0.05 # N m s --> nominal servo damping

    def f(self, x, r):
        th, om = x
        return np.array([om, (self.kp * (r - th) - (self.kd + self.b) * om - self.mgl * np.sin(th)) / self.J])

    def A(self, x):
        # Continuous-time Jacobian df/dx.
        return np.array([[0.0, 1.0],
                         [-(self.kp + self.mgl * np.cos(x[0])) / self.J, -(self.kd + self.b) / self.J]])

    def step(self, x, r, dt):
        # Advance dt: len(r) equal RK4 sub-steps, one per entry of r (a scalar = one step). Returns x, F.
        r = np.atleast_1d(r)
        h = dt / len(r)
        F = np.eye(len(x))
        for rj in r:
            x, Fj = self.rk4(x, float(rj), h)
            F = Fj @ F
        return x, F

    def rk4(self, x, r, dt):
        # One RK4 step and its exact Jacobian F = d x_next / d x.
        I = np.eye(len(x))
        k1 = self.f(x, r);              d1 = self.A(x)
        x2 = x + 0.5 * dt * k1;         D2 = I + 0.5 * dt * d1
        k2 = self.f(x2, r);             d2 = self.A(x2) @ D2
        x3 = x + 0.5 * dt * k2;         D3 = I + 0.5 * dt * d2
        k3 = self.f(x3, r);             d3 = self.A(x3) @ D3
        x4 = x + dt * k3;               D4 = I + dt * d3
        k4 = self.f(x4, r);             d4 = self.A(x4) @ D4
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), I + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)

@dataclass
class AugServoArmModel(ServoArmModel):
    # B1-aug: x = [theta, omega, d]; d is an extra angular acceleration (random walk) for everything the model misses.

    def f(self, x, r):
        th, om, d = x
        return np.array([om, (self.kp * (r - th) - (self.kd + self.b) * om - self.mgl * np.sin(th)) / self.J + d, 0.0])

    def A(self, x):
        return np.array([[0.0, 1.0, 0.0],
                         [-(self.kp + self.mgl * np.cos(x[0])) / self.J, -(self.kd + self.b) / self.J, 1.0],
                         [0.0, 0.0, 0.0]])

@dataclass
class EKFResult:
    x: np.ndarray # (T, n) posterior mean, n = 2 (B1) or 3 (B1-aug)
    P: np.ndarray # (T, n, n) posterior covariance
    innov: np.ndarray # (T,) innovation (NaN without a measurement)
    S: np.ndarray # (T,) innovation variance

class EKF:
    """Step-by-step B1 / B1-aug: predict(r) once per dt, update(z) when a measurement exists.
    The live estimator node and the offline run_ekf both use this, so they agree."""

    def __init__(self, model, dt, q, R, x0=(0.0, 0.0), P0=None):
        # q = [q_theta, q_omega] for B1, [q_theta, q_omega, q_d] for B1-aug; the state size follows x0.
        self.model, self.dt, self.R = model, dt, R
        self.x = np.asarray(x0, float).copy()
        n = len(self.x)
        self.Q = np.diag(np.broadcast_to(np.asarray(q, float), (n,))) * dt
        self.P = np.diag([0.1**2, 1.0**2, 50.0**2][:n]) if P0 is None else np.asarray(P0, float).copy()

    def predict(self, r):
        self.x, F = self.model.step(self.x, r, self.dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = z - self.x[0]
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S
        self.x = self.x + K * y
        IKH = np.eye(len(self.x)) - np.outer(K, np.eye(len(self.x))[0])
        self.P = IKH @ self.P @ IKH.T + self.R * np.outer(K, K)   # Joseph form
        return y, S

def run_ekf(model, z, r, dt, q, R, x0=None, P0=None) -> EKFResult:
    # Filter a whole run. x[k] is the estimate at t_k after z[k]; step k-1 -> k uses r[k-1]. 
    T = len(z)
    f = EKF(model, dt, q, R, (0.0, 0.0) if x0 is None else x0, P0)
    n = len(f.x)
    xs, Ps = np.zeros((T, n)), np.zeros((T, n, n))
    innov, S_all = np.full(T, np.nan), np.full(T, np.nan)
    for k in range(T):
        if k > 0:
            f.predict(r[k - 1])
        if np.isfinite(z[k]):
            innov[k], S_all[k] = f.update(z[k])
        xs[k], Ps[k] = f.x, f.P
    return EKFResult(xs, Ps, innov, S_all)