#!/usr/bin/env python3
"""
Basic simulation example for surgical robotics.

Demonstrates:
- Setting up a simulation scene
- Adding a robot and patient
- Running a simple control loop
"""

import numpy as np
import time

from surgical_robotics.simulation import (
    SurgicalScene,
    RobotSimulator,
    TissueSimulator,
    TissueProperties,
)
from surgical_robotics.simulation.scene import SurgicalProcedureScene
from surgical_robotics.simulation.physics import PhysicsConfig, PhysicsBackend
from surgical_robotics.core.base import Pose


def main():
    """Run basic simulation example."""
    print("=" * 60)
    print("Surgical Robotics - Basic Simulation Example")
    print("=" * 60)

    # Create physics configuration
    physics_config = PhysicsConfig(
        backend=PhysicsBackend.INTERNAL,
        time_step=0.001,  # 1ms physics step
        enable_rendering=False  # Disable for headless operation
    )

    # Create laparoscopic surgical scene
    print("\n1. Creating surgical scene...")
    scene = SurgicalProcedureScene.create_laparoscopic_scene()
    print("   - Operating room set up")
    print("   - Patient positioned")
    print("   - Trocar ports placed")

    # Create robot
    print("\n2. Initializing robot...")
    robot = RobotSimulator(name="surgical_arm", num_joints=7)
    robot.base_pose = Pose(
        position=np.array([0.0, -0.5, 0.0]),
        orientation=np.array([1, 0, 0, 0])
    )
    scene.add_robot(robot, "surgical_arm", robot.base_pose.position)
    print(f"   - Robot initialized with {robot.num_joints} joints")

    # Set up tissue simulation
    print("\n3. Creating tissue model...")
    tissue_sim = TissueSimulator()
    liver = tissue_sim.create_organ(
        "liver",
        "liver",
        center=np.array([0.0, 0.0, 1.1]),
        scale=0.8
    )
    print("   - Liver model created")

    # Set initial joint positions
    initial_joints = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
    robot.set_joint_positions(initial_joints)

    # Define target position (simple reaching task)
    target_position = np.array([0.1, 0.1, 1.15])

    print("\n4. Running simulation...")
    print("-" * 40)

    # Simulation loop
    dt = 0.01  # 100 Hz control loop
    sim_duration = 5.0  # 5 seconds
    steps = int(sim_duration / dt)

    start_time = time.time()

    for step in range(steps):
        # Get current end-effector pose
        ee_pose = robot.get_end_effector_pose()

        # Simple proportional control to target
        position_error = target_position - ee_pose.position
        error_norm = np.linalg.norm(position_error)

        # Compute joint velocities using Jacobian transpose
        J = robot.get_jacobian()
        position_error_6d = np.concatenate([position_error, np.zeros(3)])
        joint_velocities = 0.5 * J.T @ position_error_6d

        # Limit joint velocities
        max_vel = 1.0
        vel_norm = np.linalg.norm(joint_velocities)
        if vel_norm > max_vel:
            joint_velocities = joint_velocities / vel_norm * max_vel

        # Apply velocities
        robot.control_mode = "velocity"
        robot.set_joint_velocities(joint_velocities)

        # Step simulation
        robot.step(dt)
        scene.step(dt)
        tissue_sim.step(dt)

        # Print progress every second
        if step % 100 == 0:
            sim_time = step * dt
            print(f"   t={sim_time:.1f}s | Error: {error_norm:.4f}m | "
                  f"EE: [{ee_pose.position[0]:.3f}, {ee_pose.position[1]:.3f}, {ee_pose.position[2]:.3f}]")

    elapsed = time.time() - start_time

    # Final state
    print("-" * 40)
    print("\n5. Simulation complete!")

    final_ee = robot.get_end_effector_pose()
    final_error = np.linalg.norm(target_position - final_ee.position)

    print(f"\nResults:")
    print(f"   Simulation time: {sim_duration:.1f}s")
    print(f"   Real time: {elapsed:.2f}s")
    print(f"   Real-time factor: {sim_duration/elapsed:.2f}x")
    print(f"   Final position error: {final_error*1000:.2f}mm")
    print(f"   Target reached: {'Yes' if final_error < 0.01 else 'No'}")

    # Cleanup
    scene.shutdown()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
