# Ideal servo: the ideal PD servo the EKF assumes (tau = kp (r - theta) - kd omega, nothing else).
# ros2 run ml_ekf nominal_servo --ros-args -p use_sim_time:=true -p kp:=1.2 -p kd:=0.05
# Sub: /servo/target (std_msgs/Float64, rad), /joint_states    Pub: /servo/torque (std_msgs/Float64, N m)

import rclpy

from .servo import ServoNode

class NominalServoNode(ServoNode):
    def __init__(self) -> None:
        super().__init__("nominal_servo")
        self.kp = self.declare_parameter("kp", 1.2).value
        self.kd = self.declare_parameter("kd", 0.05).value
        self.r: float | None = None
        self.get_logger().info(f"nominal servo: kp {self.kp} kd {self.kd}")

    @staticmethod
    def torque_model(r, th, om, kp, kd):
        return kp * (r - th) - kd * om

    @staticmethod
    def jacobian(kp, kd): # d tau/d theta, d tau/d omega
        return -kp, -kd

    def torque(self, t, th, om):
        return 0.0 if self.r is None else self.torque_model(self.r, th, om, self.kp, self.kd)

    def on_target(self, t: float, r: float) -> None:
        self.r = r

def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(NominalServoNode())
    except KeyboardInterrupt:
        pass
