# Angle sensor: a noisy, low-rate copy of the true joint angle (what the EKF is allowed to see).
# ros2 run ml_ekf angle_sensor --ros-args -p use_sim_time:=true -p sigma:=0.01 -p p_drop:=0.05

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
        self.last: JointState | None = None

        # Publishers
        self.pub = self.create_publisher(JointState, "/servo/angle_meas", 10)

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self.on_state, 50)

        # Timers
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"angle sensor: {rate} Hz, sigma {self.sigma} rad, drop {self.p_drop}")

    def on_state(self, msg: JointState) -> None:
        if self.joint in msg.name:
            self.last = msg

    def tick(self) -> None:
        if self.last is None or self.rng.random() < self.p_drop:
            return
        i = list(self.last.name).index(self.joint)
        z = self.last.position[i] + self.sigma * self.rng.standard_normal()
        out = JointState()
        out.header.stamp = self.last.header.stamp # time the true angle was sampled
        out.name = [self.joint]
        out.position = [float(z)]
        self.pub.publish(out)

def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(AngleSensor())
    except KeyboardInterrupt:
        pass
