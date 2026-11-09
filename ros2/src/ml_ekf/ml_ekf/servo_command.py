# mode:=steps --> random steps every `hold` s, uniform in [r_lo, r_hi]
# mode:=sine --> r_mid + amp sin(2 pi freq t)
# mode:=chirp --> sine whose frequency sweeps f0 -> f1 over `sweep` s, then repeats
# ros2 run ml_ekf servo_command --ros-args -p use_sim_time:=true -p mode:=steps

import math
import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64

class ServoCommand(Node):
    def __init__(self) -> None:
        super().__init__("servo_command")
        P = lambda k, v: self.declare_parameter(k, v).value
        self.mode = P("mode", "steps")
        self.r_lo, self.r_hi = P("r_lo", 0.3), P("r_hi", 2.8)
        self.hold, self.amp, self.freq = P("hold", 1.0), P("amp", 1.0), P("freq", 0.5)
        self.f0, self.f1, self.sweep = P("f0", 0.1), P("f1", 3.0), P("sweep", 20.0)
        self.rng = np.random.default_rng(P("seed", 0))

        # Publishers
        self.pub = self.create_publisher(Float64, "/servo/target", 10)
        self.t0: float | None = None
        self.next_step, self.r = 0.0, 0.5 * (self.r_lo + self.r_hi)

        self.create_timer(1.0 / P("rate", 50.0), self.tick) # Callback

    def tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        if now == 0.0:
            return # sim clock not running yet

        if self.t0 is None:
            self.t0 = now

        t = now - self.t0
        mid = 0.5 * (self.r_lo + self.r_hi)

        if self.mode == "steps":
            if t >= self.next_step:
                self.r = float(self.rng.uniform(self.r_lo, self.r_hi))
                self.next_step = t + self.hold

        elif self.mode == "sine":
            self.r = mid + self.amp * math.sin(2 * math.pi * self.freq * t)

        elif self.mode == "chirp":
            tc = t % self.sweep
            k = (self.f1 - self.f0) / self.sweep
            self.r = mid + self.amp * math.sin(2 * math.pi * (self.f0 * tc + 0.5 * k * tc * tc))

        self.pub.publish(Float64(data=self.r))

def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(ServoCommand())
    except KeyboardInterrupt:
        pass
