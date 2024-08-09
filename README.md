# Cartesian Control and Inverse Kinematics for Robotic Arms

## Objective
This project involves developing a node capable of performing Cartesian Control and Numerical Inverse Kinematics (IK) for robotic arms. The project required working with the UR5 and Franka Panda robotic arms in a simulated environment using ROS2. The main objectives were to achieve precise Cartesian control of the robot's end-effector and implement an effective IK solution.

| UR5 Robot | Franka Robot |
|---|---|
| [![YouTube Video 1](https://img.youtube.com/vi/pFHAs-Gojnc/0.jpg)](https://www.youtube.com/watch?v=pFHAs-Gojnc) | [![YouTube Video 2](https://img.youtube.com/vi/uELyix2vK3o/0.jpg)](https://www.youtube.com/watch?v=uELyix2vK3o) |

## Key Features

1. **Cartesian Control:**
   - System to receive Cartesian control commands.
   - Computed necessary joint velocities to achieve the desired end-effector position and orientation.

2. **Inverse Kinematics:**
   - IK solver to determine the joint positions required to place the end-effector at a specific pose relative to the robot's base.

## Project Structure
- `cartesian_control_IK.py`: Main script for implementing Cartesian control and inverse kinematics.
