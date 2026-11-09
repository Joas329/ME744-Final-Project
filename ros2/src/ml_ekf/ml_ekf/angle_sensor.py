# Angle sensor: a noisy, low-rate copy of the true joint angle (what the EKF is allowed to see).
# ros2 run ml_ekf angle_sensor --ros-args -p use_sim_time:=true -p sigma:=0.01 -p p_drop:=0.05

import math
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import JointState

class AngleSensor(Node):
    def __init__(self) -> None:
        super().__init__("angle_sensor")
        self.joint = self.declare_parameter("joint", "servo_joint").value
        rate = self.declare_parameter("rate", 20.0).value # Hz
        self.sigma = self.declare_parameter("sigma", 0.01).value # rad (0.57 deg)
        self.p_drop = self.declare_parameter("p_drop", 0.05).value # fraction of samples lost
        self.rng = np.random.default_rng(self.declare_parameter("seed", 0).value)
        self.period = 1.0 / rate
        self.next_t: float | None = None

        # Publishers
        self.pub = self.create_publisher(JointState, "/servo/angle_meas", 10)

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self.on_state, 100)

        self.get_logger().info(f"angle sensor: {rate} Hz, sigma {self.sigma} rad, drop {self.p_drop}")

    def on_state(self, msg: JointState) -> None:
        if self.joint not in msg.name:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.next_t is None:
            self.next_t = math.ceil(t / self.period) * self.period # sample on multiples of the period
        if t + 1e-9 < self.next_t:
            return
        self.next_t += self.period * max(1, math.floor((t - self.next_t) / self.period + 1e-9) + 1)
        if self.rng.random() < self.p_drop:
            return # lost sample
        z = msg.position[list(msg.name).index(self.joint)] + self.sigma * self.rng.standard_normal()
        out = JointState()
        out.header.stamp = msg.header.stamp # time the true angle was sampled
        out.name = [self.joint]
        out.position = [float(z)]
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(AngleSensor())
    except KeyboardInterrupt:
        pass
