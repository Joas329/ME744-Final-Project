# Recorder: logs one run to a single .npz on a fixed sim-time grid, ready for an offline EKF.
# ros2 run ml_ekf recorder --ros-args -p use_sim_time:=true -p out:=runs/sine_01.npz -p duration:=30.0

import math
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

def stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

def hold(t_ev: np.ndarray, v_ev: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Zero-order hold: the last event value at or before each t (NaN before the first event)."""
    i = np.searchsorted(t_ev, t, side="right") - 1
    out = np.where(i >= 0, v_ev[np.clip(i, 0, None)], np.nan) if len(t_ev) else np.full(t.shape, np.nan)
    return out

def to_grid(js, tg, tq, zm, dt: float) -> dict[str, np.ndarray]:
    """Resample the raw event lists onto t_k = t0 + k dt. Pure NumPy, so it is testable without ROS."""
    js, tg, tq, zm = (np.asarray(a, float).reshape(-1, c) for a, c in ((js, 3), (tg, 2), (tq, 2), (zm, 2)))
    for a in (js, tg, tq, zm):
        a[:] = a[np.argsort(a[:, 0], kind="stable")]
    t0, t1 = tg[0, 0], js[-1, 0]
    t = t0 + dt * np.arange(int(math.floor((t1 - t0) / dt)) + 1)
    z = np.full(t.shape, np.nan)
    k = np.rint((zm[:, 0] - t0) / dt).astype(int)
    ok = (k >= 0) & (k < len(t))
    z[k[ok]] = zm[ok, 1]
    return {"t": t, "r": hold(tg[:, 0], tg[:, 1], t), "tau": np.nan_to_num(hold(tq[:, 0], tq[:, 1], t)),
            "theta_true": np.interp(t, js[:, 0], js[:, 1]), "omega_true": np.interp(t, js[:, 0], js[:, 2]), "z": z}

class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("recorder")
        self.joint = self.declare_parameter("joint", "servo_joint").value
        self.dt = self.declare_parameter("dt", 0.01).value
        self.sigma = self.declare_parameter("sigma", float("nan")).value # pass the sensor's sigma to store it
        self.duration = self.declare_parameter("duration", 0.0).value # sim s after the first command; 0 = until Ctrl+C
        out = self.declare_parameter("out", "").value
        self.out = Path(os.path.expanduser(out or "~/ml_ekf_runs/run.npz"))
        self.js, self.tg, self.tq, self.zm = [], [], [], []
        self.done = False

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self.on_js, 100)
        self.create_subscription(JointState, "/servo/angle_meas", self.on_z, 10)
        self.create_subscription(Float64, "/servo/target", lambda m: self.tg.append((self.now(), m.data)), 10)
        self.create_subscription(Float64, "/servo/torque", lambda m: self.tq.append((self.now(), m.data)), 100)

        self.get_logger().info(f"recording to {self.out}" + (f" for {self.duration} s" if self.duration else ""))

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_js(self, m: JointState) -> None:
        if self.joint not in m.name:
            return
        i = list(m.name).index(self.joint)
        t = stamp(m)
        self.js.append((t, m.position[i], m.velocity[i] if len(m.velocity) > i else np.nan))
        if self.duration and self.tg and t - self.tg[0][0] >= self.duration:
            self.done = True

    def on_z(self, m: JointState) -> None:
        if m.position:
            self.zm.append((stamp(m), m.position[0]))

    def save(self) -> None:
        if not self.tg or len(self.js) < 2:
            self.get_logger().warn("nothing to save: no /servo/target or /joint_states received")
            return
        g = to_grid(self.js, self.tg, self.tq or [(0.0, 0.0)], self.zm or [(0.0, np.nan)], self.dt)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out, **g, dt=self.dt, sigma=self.sigma,
                            raw_joint_states=np.array(self.js), raw_target=np.array(self.tg),
                            raw_torque=np.array(self.tq), raw_meas=np.array(self.zm))
        n_z = int(np.isfinite(g["z"]).sum())
        print(f"[recorder] saved {self.out}: {len(g['t'])} steps ({g['t'][-1] - g['t'][0]:.1f} s sim), "
              f"{n_z} measurements", flush=True)

def main() -> None:
    rclpy.init()
    node = Recorder()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.save()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
