from __future__ import annotations

import bisect

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from .ekf import EKF, AugServoArmModel, ServoArmModel

def make_b1(p, x0):
    model = ServoArmModel(p["J"], p["b"], p["mgl"], p["kp"], p["kd"])
    return EKF(model, p["dt"], [0.0, p["q_omega"]], p["sigma"] ** 2, x0=x0)

def make_b1aug(p, x0):
    model = AugServoArmModel(p["J"], p["b"], p["mgl"], p["kp"], p["kd"])
    return EKF(model, p["dt"], [0.0, p["q_omega_aug"], p["q_d"]], p["sigma"] ** 2, x0=(*x0, 0.0))

KINDS = {"b1": make_b1, "b1aug": make_b1aug}

class Estimator(Node):
    def __init__(self):
        super().__init__("estimator")
        d = ServoArmModel()
        defaults = {"name": "b1", "kind": "b1", "joint": "servo_joint", "dt": 0.01, "n_sub": 10, "lag": 0.03, "q_omega": 5.0, "q_omega_aug": 100.0, "q_d": 1e6, "sigma": 0.01, "J": d.J, "b": d.b, "mgl": d.mgl, "kp": d.kp, "kd": d.kd}
        self.p = {k: self.declare_parameter(k, v).value for k, v in defaults.items()}
        if self.p["kind"] not in KINDS:
            raise ValueError(f"unknown kind {self.p['kind']!r}; known: {sorted(KINDS)}")
        self.dt, self.n_sub, self.lag = self.p["dt"], int(self.p["n_sub"]), self.p["lag"]
        self.filter = None # created at the first measurement
        self.k = 0 # last processed step
        self.tg, self.rg = [], [] # target receive times (sim) and values
        self.zs = {} # step index -> measurement

        # Publishers
        self.pub = self.create_publisher(JointState, f"/estimators/{self.p['name']}/state", 50)

        # Subscriptions
        self.create_subscription(Float64, "/servo/target", self.on_target, 10)
        self.create_subscription(JointState, "/servo/angle_meas", self.on_meas, 10)
        self.create_subscription(Clock, "/clock", self.on_clock, 100)
        self.get_logger().info(f"estimator {self.p['name']} ({self.p['kind']}): dt {self.dt} s x {self.n_sub} sub-steps, q_omega {self.p['q_omega']}, sigma {self.p['sigma']}")

    def on_target(self, m):
        self.tg.append(self.get_clock().now().nanoseconds * 1e-9)
        self.rg.append(m.data)

    def r_at(self, t):
        i = bisect.bisect_right(self.tg, t) - 1
        return self.rg[max(i, 0)]

    def on_meas(self, m):
        if not m.position:
            return
        k = int(round((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9) / self.dt))
        if self.filter is None:
            if not self.tg:
                return # wait until the servo has been commanded
            self.filter = KINDS[self.p["kind"]](self.p, (m.position[0], 0.0))
            self.k = k
            self.publish(k)
        elif k > self.k:
            self.zs[k] = m.position[0]

    def on_clock(self, m):
        if self.filter is None:
            return
        now = m.clock.sec + m.clock.nanosec * 1e-9
        while (self.k + 1) * self.dt + self.lag <= now:
            t0 = self.k * self.dt # predict k -> k+1 with the commands actually sent
            self.filter.predict([self.r_at(t0 + j * self.dt / self.n_sub) for j in range(self.n_sub)])
            self.k += 1
            z = self.zs.pop(self.k, None)
            if z is not None:
                self.filter.update(z)
            self.publish(self.k)
        while len(self.tg) > 2 and self.tg[1] < now - 2.0: # keep 2 s of command history
            self.tg.pop(0)
            self.rg.pop(0)

    def publish(self, k):
        x, P = self.filter.x, self.filter.P
        t = k * self.dt
        msg = JointState()
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int(round((t - int(t)) * 1e9))
        msg.name = [self.p["joint"]]
        msg.position, msg.velocity = [float(x[0])], [float(x[1])]
        msg.effort = [float(P[0, 0]), float(P[0, 1]), float(P[1, 1])]
        self.pub.publish(msg)

def main():
    rclpy.init()
    try:
        rclpy.spin(Estimator())
    except KeyboardInterrupt:
        pass