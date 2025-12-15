#!/usr/bin/env python3
"""
Teleoperation demonstration for surgical robotics.

Demonstrates:
- Master-slave teleoperation
- Motion scaling
- Workspace mapping
- Force feedback
"""

import numpy as np
import time

from surgical_robotics.control import (
    TeleoperationController,
    BilateralController,
    MotionScaler,
)
from surgical_robotics.control.teleoperation import TeleoperationConfig, TeleoperationMode
from surgical_robotics.core.base import Pose
from surgical_robotics.simulation import RobotSimulator


def simulate_master_motion(t: float) -> Pose:
    """Simulate master device motion (e.g., surgeon's hand movement)."""
    # Circular motion with varying z
    x = 0.05 * np.sin(t * 0.5)
    y = 0.05 * np.cos(t * 0.5)
    z = 0.02 * np.sin(t * 0.3)

    return Pose(
        position=np.array([x, y, z]),
        orientation=np.array([1, 0, 0, 0])
    )


def main():
    """Run teleoperation demonstration."""
    print("=" * 60)
    print("Surgical Robotics - Teleoperation Demo")
    print("=" * 60)

    # Configuration
    config = TeleoperationConfig(
        mode=TeleoperationMode.POSITION,
        position_scale=0.5,  # 2:1 motion scaling
        orientation_scale=0.5,
        position_filter_alpha=0.3,
        max_linear_velocity=0.2,  # m/s
    )

    print("\n1. Configuration:")
    print(f"   Mode: {config.mode.name}")
    print(f"   Position scale: {config.position_scale} (master:slave)")
    print(f"   Max velocity: {config.max_linear_velocity} m/s")

    # Create controllers
    print("\n2. Initializing controllers...")
    teleop = TeleoperationController(config)
    teleop.enable()

    # Create simulated slave robot
    slave_robot = RobotSimulator(name="slave_arm", num_joints=7)
    slave_robot.control_mode = "position"

    # Initial poses
    master_initial = Pose(
        position=np.array([0, 0, 0]),
        orientation=np.array([1, 0, 0, 0])
    )
    slave_initial = Pose(
        position=np.array([0, 0, 0.3]),  # Slave workspace center
        orientation=np.array([1, 0, 0, 0])
    )

    # Engage clutch to set reference
    print("   Engaging clutch (setting reference)...")
    teleop.engage_clutch(master_initial, slave_initial)
    time.sleep(0.1)
    teleop.release_clutch()
    print("   Clutch released - teleoperation active")

    # Simulation parameters
    dt = 0.01  # 100 Hz
    duration = 10.0
    steps = int(duration / dt)

    print("\n3. Running teleoperation simulation...")
    print("-" * 50)

    # Data recording
    master_positions = []
    slave_positions = []
    timestamps = []

    start_time = time.time()

    for step in range(steps):
        t = step * dt

        # Simulate master motion
        master_pose = simulate_master_motion(t)
        master_velocity = np.zeros(6)  # Simplified

        # Get current slave state
        slave_pose = slave_robot.get_end_effector_pose()
        slave_force = np.zeros(6)

        # Update teleoperation
        slave_command, master_feedback = teleop.update(
            master_pose=master_pose,
            master_velocity=master_velocity,
            slave_pose=slave_pose,
            slave_force=slave_force,
            dt=dt
        )

        # Command slave robot
        if slave_command:
            # Use IK to compute joint positions
            target_joints, success = slave_robot.inverse_kinematics(
                slave_command,
                max_iterations=20
            )
            if success:
                slave_robot.set_joint_positions(target_joints)

        # Step slave robot
        slave_robot.step(dt)

        # Record data
        master_positions.append(master_pose.position.copy())
        slave_positions.append(slave_robot.get_end_effector_pose().position.copy())
        timestamps.append(t)

        # Print progress
        if step % 100 == 0:
            print(f"   t={t:.1f}s | Master: [{master_pose.position[0]:.3f}, "
                  f"{master_pose.position[1]:.3f}, {master_pose.position[2]:.3f}] | "
                  f"Slave: [{slave_pose.position[0]:.3f}, {slave_pose.position[1]:.3f}, "
                  f"{slave_pose.position[2]:.3f}]")

    elapsed = time.time() - start_time

    print("-" * 50)
    print("\n4. Results:")

    # Analyze tracking performance
    master_positions = np.array(master_positions)
    slave_positions = np.array(slave_positions)

    # Expected slave positions (accounting for scaling and offset)
    expected_slave = slave_initial.position + master_positions * config.position_scale
    tracking_error = slave_positions - expected_slave
    rms_error = np.sqrt(np.mean(np.sum(tracking_error**2, axis=1)))

    print(f"   Simulation duration: {duration:.1f}s")
    print(f"   Real-time factor: {duration/elapsed:.2f}x")
    print(f"   RMS tracking error: {rms_error*1000:.2f}mm")

    # Motion range analysis
    master_range = np.ptp(master_positions, axis=0)
    slave_range = np.ptp(slave_positions, axis=0)

    print(f"\n   Master motion range:")
    print(f"      X: {master_range[0]*1000:.1f}mm")
    print(f"      Y: {master_range[1]*1000:.1f}mm")
    print(f"      Z: {master_range[2]*1000:.1f}mm")

    print(f"\n   Slave motion range (scaled by {config.position_scale}):")
    print(f"      X: {slave_range[0]*1000:.1f}mm (expected: {master_range[0]*config.position_scale*1000:.1f}mm)")
    print(f"      Y: {slave_range[1]*1000:.1f}mm (expected: {master_range[1]*config.position_scale*1000:.1f}mm)")
    print(f"      Z: {slave_range[2]*1000:.1f}mm (expected: {master_range[2]*config.position_scale*1000:.1f}mm)")

    # Check communication delay
    delay = teleop.get_delay()
    print(f"\n   Estimated communication delay: {delay*1000:.2f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
