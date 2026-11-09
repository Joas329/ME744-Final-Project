# True servo: the hobby servo's hidden electronics, model and ROS node in one file.
# ros2 run ml_ekf true_servo --ros-args -p use_sim_time:=true -p deadband:=0.01

import math
import rclpy
import bisect

from .servo import ServoNode
from dataclasses import dataclass, field

@dataclass
class TrueServo:
    kp: float = 1.2 # N m / rad
    kd: float = 0.05 # N m s / rad
    deadband: float = 0.01 # rad, half-width
    tau_stall: float = 1.0 # N m
    omega_noload: float = 6.0 # rad / s
    act_tau: float = 0.008 # s, motor lag
    delay: float = 0.01 # s, command delay
    tau_a: float = 0.0 # lagged torque (state)
    _t: list = field(default_factory=list)
    _r: list = field(default_factory=list)
    _t_prev: float | None = None

    def set_target(self, t: float, r: float) -> None:
        # Record a new target received at time t (keeps 2 s of history for the delay).
        self._t.append(t)
        self._r.append(r)
        while len(self._t) > 2 and self._t[1] < t - 2.0:
            self._t.pop(0)
            self._r.pop(0)

    def target_at(self, t: float) -> float | None:
        i = bisect.bisect_right(self._t, t) - 1
        return self._r[i] if i >= 0 else None

    def command(self, r: float, theta: float, omega: float) -> float:
        e = r - theta
        e_db = math.copysign(max(abs(e) - self.deadband, 0.0), e)
        cmd = self.kp * e_db - self.kd * omega
        w = omega / self.omega_noload
        hi = self.tau_stall * min(max(1.0 - w, 0.0), 1.0)
        lo = -self.tau_stall * min(max(1.0 + w, 0.0), 1.0)
        return min(max(cmd, lo), hi)

    def torque(self, t: float, theta: float, omega: float) -> float:
        """Torque to apply at time t. Call once per joint-state sample."""
        h = 0.0 if self._t_prev is None else t - self._t_prev
        self._t_prev = t
        r = self.target_at(t - self.delay)
        if r is None:
            return 0.0 # no command yet: motor off
        if h > 0:
            g = 1.0 - math.exp(-h / self.act_tau) if self.act_tau > 0 else 1.0
            self.tau_a += (self.command(r, theta, omega) - self.tau_a) * g
        return self.tau_a

class TrueServoNode(ServoNode):
    def __init__(self) -> None:
        super().__init__("true_servo")
        d = TrueServo()
        p = {k: self.declare_parameter(k, getattr(d, k)).value
             for k in ("kp", "kd", "deadband", "tau_stall", "omega_noload", "act_tau", "delay")}
        self.servo = TrueServo(**p)
        self.get_logger().info(
            f"true servo: kp {p['kp']} kd {p['kd']} deadband {math.degrees(p['deadband']):.2f} deg, "
            f"stall {p['tau_stall']} N m, no-load {p['omega_noload']} rad/s, lag {p['act_tau'] * 1e3:.1f} ms, "
            f"delay {p['delay'] * 1e3:.0f} ms")

    def on_target(self, t: float, r: float) -> None:
        self.servo.set_target(t, r)

    def torque(self, t: float, th: float, om: float) -> float:
        return self.servo.torque(t, th, om)

def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(TrueServoNode())
    except KeyboardInterrupt:
        pass
