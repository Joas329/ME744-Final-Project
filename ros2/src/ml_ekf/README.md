# ML assisted EKF

![Gazebo simulation](ros2/src/ml_ekf/resource/simulation.png)

ME744 final project (UWaterloo, Fall 2026). The idea is to see if we can improve the prediction step of an EKF by attaching a small neural network (a GRU) to the physics model of a servo joint.

## Experiment

One servo joint simulated in Gazebo Fortress with ROS 2 Humble. Gazebo gives the exact angle and speed, which we use as ground truth. The filters only get a noisy angle at 20 Hz (sigma 0.01 rad, 5% of samples dropped) and the target angle sent to the servo.

The simulated servo has a delay, motor lag, deadband and a torque limit that the EKF model doesn't know about (it assumes an ideal PD servo). The estimators are:

- B1: EKF with the ideal servo model (angle, speed).
- B1-aug: B1 plus an extra disturbance state that soaks up whatever the model gets wrong.
- B3: B1-aug plus a GRU that predicts the missing acceleration from recent history (`gru.py`).

## Build

```bash
cd ros2
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
pip install torch   # only needed for gru.py
```

## Run

```bash
# RViz + joint slider, no physics
ros2 launch ml_ekf display.launch.py

# Gazebo + RViz with a command and live filters
ros2 launch ml_ekf display.launch.py gazebo:=true command:=steps estimators:=b1,b1aug

# record one 30 s run to a .npz
ros2 launch ml_ekf display.launch.py gazebo:=true headless:=true command:=sine record:=$HOME/ml_ekf_runs/sine_01.npz
```

`command` can be `steps`, `sine` or `chirp`. Other launch arguments (`seed`, `duration`, `rtf`, `servo_params`, `command_params`) are listed in `launch/display.launch.py`.

## Dataset and training

```bash
ros2 run ml_ekf make_dataset --out ~/ml_ekf_runs/dataset   # 108 runs, about 70 min
ros2 run ml_ekf gru --data ~/ml_ekf_runs/dataset           # trains B3, saves ~/ml_ekf_runs/gru_dyn/gru.pt
```

The dataset has train, val, test and ood splits. The ood runs use a servo with more delay, less torque and a bigger deadband than training. Filter noise settings are tuned on val only.

## Results

Mean RMSE over 18 runs per split (B3 is one training seed for now):

| | test angle | test speed | ood angle | ood speed |
|---|---|---|---|---|
| B1 | 3.77 deg | 2.27 rad/s | 4.26 deg | 2.63 rad/s |
| B1-aug | 2.85 deg | 1.70 rad/s | 3.04 deg | 1.93 rad/s |
| B3 | 2.20 deg | 1.30 rad/s | 2.62 deg | 1.59 rad/s |

B3 improves on B1-aug by about 23% on test and 14-18% on ood.

## TODO

- More GRU seeds, then S4D and pruning
- Run B3 live in ROS next to B1 and B1-aug
