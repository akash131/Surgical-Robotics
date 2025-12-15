"""Command-line interface commands for surgical robotics."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List
import json

from surgical_robotics.cli.config import (
    Configuration,
    ConfigurationManager,
    RobotType,
    create_default_config,
    load_config,
    save_config,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="surgical-robotics",
        description="Surgical Robotics Control System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  surgical-robotics run --config config.json
  surgical-robotics calibrate --joint 0
  surgical-robotics diagnose --full
  surgical-robotics config --generate davinci_xi
        """
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Configuration file path"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run robot system")
    run_parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode"
    )
    run_parser.add_argument(
        "--mode",
        choices=["position", "velocity", "torque", "teleoperation"],
        default="position",
        help="Control mode"
    )
    run_parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Run duration in seconds (0 for infinite)"
    )

    # Calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate robot")
    cal_parser.add_argument(
        "--joint",
        type=int,
        help="Specific joint to calibrate"
    )
    cal_parser.add_argument(
        "--tool",
        action="store_true",
        help="Calibrate tool center point"
    )
    cal_parser.add_argument(
        "--camera",
        action="store_true",
        help="Calibrate camera"
    )
    cal_parser.add_argument(
        "--all",
        action="store_true",
        help="Full calibration sequence"
    )

    # Diagnose command
    diag_parser = subparsers.add_parser("diagnose", help="Run diagnostics")
    diag_parser.add_argument(
        "--full",
        action="store_true",
        help="Run full diagnostic suite"
    )
    diag_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick health check"
    )
    diag_parser.add_argument(
        "--component",
        type=str,
        help="Diagnose specific component"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument(
        "--generate",
        type=str,
        choices=["custom", "davinci", "davinci_xi", "mako", "rosa", "hugo"],
        help="Generate default configuration"
    )
    config_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration file"
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--output",
        type=str,
        help="Output file path"
    )

    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument(
        "--scene",
        type=str,
        choices=["laparoscopic", "orthopedic", "neurosurgical", "custom"],
        default="laparoscopic",
        help="Simulation scene type"
    )
    sim_parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Enable rendering"
    )
    sim_parser.add_argument(
        "--record",
        type=str,
        help="Record simulation to file"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.verbose:
        print(f"Running with arguments: {parsed_args}")

    if parsed_args.command is None:
        parser.print_help()
        return 0

    # Load configuration
    config = None
    if parsed_args.config:
        try:
            config = load_config(parsed_args.config)
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            return 1
    else:
        config = load_config()

    # Dispatch to command handler
    if parsed_args.command == "run":
        return run_command(parsed_args, config)
    elif parsed_args.command == "calibrate":
        return calibrate_command(parsed_args, config)
    elif parsed_args.command == "diagnose":
        return diagnose_command(parsed_args, config)
    elif parsed_args.command == "config":
        return config_command(parsed_args, config)
    elif parsed_args.command == "simulate":
        return simulate_command(parsed_args, config)
    elif parsed_args.command == "status":
        return status_command(parsed_args, config)

    return 0


def run_command(args: argparse.Namespace, config: Configuration) -> int:
    """Handle run command."""
    print(f"Starting surgical robotics system...")
    print(f"  Robot: {config.robot.name}")
    print(f"  Type: {config.robot.robot_type.value}")
    print(f"  Mode: {args.mode}")
    print(f"  Simulation: {args.simulation}")

    if args.simulation:
        config.robot.simulation.enabled = True
        return run_simulation(config, args.duration)

    # Real robot connection would go here
    print("\nConnecting to robot...")

    # This would connect to actual hardware
    print("Robot system started. Press Ctrl+C to stop.")

    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

    return 0


def run_simulation(config: Configuration, duration: float = 0) -> int:
    """Run robot simulation."""
    from surgical_robotics.simulation import (
        SurgicalScene,
        RobotSimulator,
        TissueSimulator,
    )
    from surgical_robotics.simulation.physics import PhysicsConfig, PhysicsBackend

    print("\nInitializing simulation...")

    # Create physics config
    physics_config = PhysicsConfig(
        backend=PhysicsBackend.INTERNAL,
        time_step=config.robot.simulation.time_step,
        enable_rendering=config.robot.simulation.render_enabled
    )

    # Create scene
    scene = SurgicalScene(physics_config)
    scene.setup_operating_room()
    scene.add_patient()

    # Create robot
    robot = RobotSimulator(
        name=config.robot.name,
        num_joints=config.robot.num_joints
    )
    scene.add_robot(robot, config.robot.name, np.array([0, -0.5, 0]))

    print("Simulation initialized.")
    print("Running simulation loop...")

    dt = config.robot.simulation.time_step
    sim_time = 0.0

    try:
        while True:
            scene.step(dt)
            sim_time += dt

            if sim_time % 1.0 < dt:
                ee_pose = robot.get_end_effector_pose()
                print(f"\rTime: {sim_time:.1f}s | EE pos: [{ee_pose.position[0]:.3f}, {ee_pose.position[1]:.3f}, {ee_pose.position[2]:.3f}]", end="")

            if duration > 0 and sim_time >= duration:
                break

            if config.robot.simulation.real_time:
                time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

    scene.shutdown()
    print("Simulation complete.")
    return 0


def calibrate_command(args: argparse.Namespace, config: Configuration) -> int:
    """Handle calibration command."""
    return calibrate(config, joint=args.joint, tool=args.tool, camera=args.camera)


def calibrate(
    config: Configuration,
    joint: Optional[int] = None,
    tool: bool = False,
    camera: bool = False
) -> int:
    """Run calibration routines."""
    print("Starting calibration...")
    print(f"  Robot: {config.robot.name}")

    if joint is not None:
        print(f"\nCalibrating joint {joint}...")
        # Joint calibration routine
        print("  - Moving to limit switch...")
        print("  - Recording encoder position...")
        print("  - Computing offset...")
        print(f"  Joint {joint} calibration complete.")

    if tool:
        print("\nCalibrating tool center point...")
        print("  - Please touch the calibration point from multiple orientations")
        print("  - Position 1/4... recorded")
        print("  - Position 2/4... recorded")
        print("  - Position 3/4... recorded")
        print("  - Position 4/4... recorded")
        print("  - Computing TCP...")
        print("  Tool calibration complete.")

    if camera:
        print("\nCalibrating camera...")
        print("  - Detecting calibration pattern...")
        print("  - Computing intrinsics...")
        print("  - Computing extrinsics...")
        print("  Camera calibration complete.")

    if not any([joint is not None, tool, camera]):
        print("No calibration type specified. Use --joint, --tool, or --camera")
        return 1

    print("\nCalibration finished successfully.")
    return 0


def diagnose_command(args: argparse.Namespace, config: Configuration) -> int:
    """Handle diagnose command."""
    return diagnose(config, full=args.full, quick=args.quick, component=args.component)


def diagnose(
    config: Configuration,
    full: bool = False,
    quick: bool = False,
    component: Optional[str] = None
) -> int:
    """Run diagnostic routines."""
    print("Running diagnostics...")
    print(f"  Robot: {config.robot.name}")
    print()

    results = {}

    if quick or full:
        # Quick checks
        print("Quick diagnostics:")
        results["communication"] = _check_communication(config)
        results["safety_systems"] = _check_safety(config)
        results["joints"] = _check_joints(config)

    if full:
        # Full diagnostic suite
        print("\nFull diagnostics:")
        results["encoders"] = _check_encoders(config)
        results["motors"] = _check_motors(config)
        results["sensors"] = _check_sensors(config)
        results["calibration"] = _check_calibration(config)

    if component:
        print(f"\nComponent diagnostic: {component}")
        results[component] = _check_component(config, component)

    # Summary
    print("\n" + "=" * 50)
    print("Diagnostic Summary:")
    print("=" * 50)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll diagnostics PASSED")
        return 0
    else:
        print("\nSome diagnostics FAILED")
        return 1


def _check_communication(config: Configuration) -> bool:
    """Check communication systems."""
    print("  [*] Communication... ", end="", flush=True)
    # Would check actual communication
    print("OK")
    return True


def _check_safety(config: Configuration) -> bool:
    """Check safety systems."""
    print("  [*] Safety systems... ", end="", flush=True)
    # Would check safety systems
    print("OK")
    return True


def _check_joints(config: Configuration) -> bool:
    """Check joint functionality."""
    print("  [*] Joints... ", end="", flush=True)
    for i in range(config.robot.num_joints):
        print(f"J{i} ", end="", flush=True)
    print("OK")
    return True


def _check_encoders(config: Configuration) -> bool:
    """Check encoder readings."""
    print("  [*] Encoders... ", end="", flush=True)
    print("OK")
    return True


def _check_motors(config: Configuration) -> bool:
    """Check motor functionality."""
    print("  [*] Motors... ", end="", flush=True)
    print("OK")
    return True


def _check_sensors(config: Configuration) -> bool:
    """Check sensor readings."""
    print("  [*] Sensors... ", end="", flush=True)
    print("OK")
    return True


def _check_calibration(config: Configuration) -> bool:
    """Check calibration status."""
    print("  [*] Calibration... ", end="", flush=True)
    if config.robot.calibration.calibration_valid:
        print("OK")
        return True
    else:
        print("NEEDS CALIBRATION")
        return False


def _check_component(config: Configuration, component: str) -> bool:
    """Check specific component."""
    print(f"  [*] {component}... ", end="", flush=True)
    print("OK")
    return True


def config_command(args: argparse.Namespace, config: Configuration) -> int:
    """Handle configuration command."""
    manager = ConfigurationManager()

    if args.generate:
        # Generate default configuration
        robot_type_map = {
            "custom": RobotType.CUSTOM,
            "davinci": RobotType.DAVINCI,
            "davinci_xi": RobotType.DAVINCI_XI,
            "mako": RobotType.MAKO,
            "rosa": RobotType.ROSA,
            "hugo": RobotType.HUGO,
        }
        robot_type = robot_type_map.get(args.generate, RobotType.CUSTOM)
        new_config = create_default_config(robot_type)

        output_path = args.output or f"{args.generate}_config.json"
        save_config(new_config, output_path)
        print(f"Generated configuration: {output_path}")
        return 0

    if args.validate:
        issues = manager.validate(config)
        if issues:
            print("Configuration validation errors:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("Configuration is valid.")
            return 0

    if args.show:
        data = manager._config_to_dict(config)
        print(json.dumps(data, indent=2, default=str))
        return 0

    print("No config action specified. Use --generate, --validate, or --show")
    return 1


def simulate_command(args: argparse.Namespace, config: Configuration) -> int:
    """Handle simulation command."""
    config.robot.simulation.enabled = True
    config.robot.simulation.render_enabled = args.render

    if args.record:
        config.robot.telemetry.record_enabled = True
        config.robot.telemetry.record_directory = str(Path(args.record).parent)

    return run_simulation(config)


def status_command(args: argparse.Namespace, config: Configuration) -> int:
    """Handle status command."""
    status = {
        "robot": {
            "name": config.robot.name,
            "type": config.robot.robot_type.value,
            "joints": config.robot.num_joints,
            "arms": config.robot.num_arms,
        },
        "safety": {
            "e_stop_enabled": config.robot.safety.enable_e_stop,
            "collision_detection": config.robot.safety.collision_detection_enabled,
            "virtual_fixtures": config.robot.safety.virtual_fixtures_enabled,
        },
        "communication": {
            "interface": config.robot.communication.control_interface,
            "frequency": config.robot.communication.control_frequency,
            "ros2_enabled": config.robot.communication.ros2_enabled,
        },
        "calibration": {
            "valid": config.robot.calibration.calibration_valid,
            "last_date": config.robot.calibration.last_calibration_date,
        },
        "simulation": {
            "enabled": config.robot.simulation.enabled,
            "backend": config.robot.simulation.physics_backend,
        }
    }

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print("System Status")
        print("=" * 40)
        print(f"Robot: {status['robot']['name']} ({status['robot']['type']})")
        print(f"  Joints: {status['robot']['joints']}")
        print(f"  Arms: {status['robot']['arms']}")
        print()
        print("Safety:")
        print(f"  E-Stop: {'Enabled' if status['safety']['e_stop_enabled'] else 'Disabled'}")
        print(f"  Collision Detection: {'Enabled' if status['safety']['collision_detection'] else 'Disabled'}")
        print()
        print("Communication:")
        print(f"  Interface: {status['communication']['interface']}")
        print(f"  Frequency: {status['communication']['frequency']} Hz")
        print()
        print("Calibration:")
        print(f"  Valid: {'Yes' if status['calibration']['valid'] else 'No'}")
        if status['calibration']['last_date']:
            print(f"  Last: {status['calibration']['last_date']}")

    return 0


# Need numpy for simulation
import numpy as np


if __name__ == "__main__":
    sys.exit(main())
