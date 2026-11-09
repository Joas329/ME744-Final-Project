# Parent Class
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class ServoNode(Node):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.declare_parameter("joint", "servo_joint")
        self.declare_parameter("r_min", 0.0) # a hobby servo only accepts 0..pi
        self.declare_parameter("r_max", 3.14159) # One entire revolution
        self.joint = self.get_parameter("joint").value
        self.r_lim = (self.get_parameter("r_min").value, self.get_parameter("r_max").value)

        # Publishers
        self.pub = self.create_publisher(Float64, "/servo/torque", 10)

        # Subscriptions
        self.create_subscription(Float64, "/servo/target", self._on_target, 10)
        self.create_subscription(JointState, "/joint_states", self._on_state, 50)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_target(self, msg: Float64) -> None:
        self.on_target(self.now(), min(max(msg.data, self.r_lim[0]), self.r_lim[1]))

    def _on_state(self, msg: JointState) -> None:
        if self.joint not in msg.name:
            return

        i = list(msg.name).index(self.joint)
        th = msg.position[i]

        om = 0.0
        if len(msg.velocity) > i:
            om = msg.velocity[i]

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.pub.publish(Float64(data=float(self.torque(t, th, om))))

    # implemented by each servo
    def on_target(self, t: float, r: float) -> None: ...
    def torque(self, t: float, th: float, om: float) -> float: ...
