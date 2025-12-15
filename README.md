# Surgical Robotics Control Systems

A comprehensive Python framework for surgical robot control systems, including Da Vinci-class multi-arm platforms, neurosurgical robots, orthopedic surgical systems, and extensive vendor integrations.

## Features

### Robot Systems

#### Da Vinci-Class Systems
- **Multi-arm robotic platforms** with master-slave teleoperation
- **Haptic feedback systems** for force sensing and rendering
- **3D stereoscopic vision** with depth perception
- **Instrument tracking** and collision avoidance
- Remote center of motion (RCM) constraint enforcement

#### Neurosurgical Robots
- **Sub-millimeter positioning accuracy** (±0.1mm)
- **Integration with intraoperative MRI/CT** imaging
- **Tremor cancellation** with adaptive filtering
- Stereotactic frame registration
- Image-guided navigation

#### Orthopedic Robots
- **Bone cutting and milling systems** with various tool types
- **Force feedback** for tissue differentiation
- **Patient-specific surgical planning** from CT/MRI
- Haptic boundaries for safe cutting
- Adaptive cutting control

### Vendor Integrations

| Vendor | Robot Systems | Features |
|--------|---------------|----------|
| **Intuitive Surgical** | Da Vinci Xi/X/SP/Si | Full teleoperation, instrument control |
| **Medtronic** | Hugo RAS, Mazor X | Multi-port surgery, spine navigation |
| **Stryker** | Mako | AccuStop haptic boundaries, TKA/THA/PKA |
| **Zimmer Biomet** | ROSA Knee/Hip/Spine/Brain | Procedure-specific workflows |
| **Smith+Nephew** | CORI | Handheld robotic burr, real-time feedback |

### Communication Protocols
- **ROS2 Integration** - Full ROS2 bridge with joint states, TF, and action servers
- **DICOM** - Medical imaging import/export with registration
- **HL7/FHIR** - Healthcare interoperability for procedure records
- **Real-time UDP/TCP** - Low-latency communication for control loops

### Advanced Control Systems
- **Model Predictive Control (MPC)** - Constraint-aware trajectory optimization
- **Impedance/Admittance Control** - Compliant interaction with environment
- **Hybrid Force-Position Control** - Simultaneous force and position tracking
- **Reinforcement Learning** - Gymnasium-compatible environment for training
- **Bilateral Teleoperation** - Time delay compensation with wave variables

### Simulation Environment
- **Physics Simulation** - PyBullet/MuJoCo backends
- **Tissue Modeling** - Deformable and cuttable tissue with bleeding
- **Surgical Scenes** - Pre-configured OR setups
- **Robot Simulation** - Full kinematic and dynamic simulation

### Workflow Management
- **State Machines** - Robot operational states and procedure phases
- **Task Sequencing** - Surgical task management with dependencies
- **Safety States** - IEC 62443 compliant safety monitoring
- **Emergency Handling** - Coordinated emergency response

### Telemetry & Monitoring
- **Structured Logging** - JSON logs with medical audit trails
- **Metrics Collection** - Prometheus-compatible metrics
- **Data Recording** - High-frequency data capture and replay
- **Audit Logging** - Tamper-evident medical device logging

### Compliance Helpers
- **IEC 62304** - Software lifecycle management
- **IEC 60601** - Electrical safety and risk analysis
- **FDA 21 CFR 820** - Quality system records and design controls

## Installation

```bash
# Clone the repository
git clone https://github.com/surgical-robotics/surgical-robotics.git
cd surgical-robotics

# Install in development mode
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[all]"

# Or install specific extras
pip install -e ".[simulation,vision]"
```

## Quick Start

### Basic Simulation

```python
from surgical_robotics.simulation import SurgicalScene, RobotSimulator
from surgical_robotics.simulation.scene import SurgicalProcedureScene
import numpy as np

# Create laparoscopic scene
scene = SurgicalProcedureScene.create_laparoscopic_scene()

# Add robot
robot = RobotSimulator(name="arm", num_joints=7)
scene.add_robot(robot, "arm", np.array([0, -0.5, 0]))

# Run simulation
for _ in range(1000):
    robot.step(0.01)
    scene.step(0.01)
```

### Teleoperation

```python
from surgical_robotics.control import TeleoperationController
from surgical_robotics.control.teleoperation import TeleoperationConfig, TeleoperationMode

config = TeleoperationConfig(
    mode=TeleoperationMode.POSITION,
    position_scale=0.5,  # 2:1 motion reduction
)

controller = TeleoperationController(config)
controller.enable()

# Set clutch reference
controller.engage_clutch(master_pose, slave_pose)
controller.release_clutch()

# Update loop
slave_command, feedback = controller.update(
    master_pose, master_velocity,
    slave_pose, slave_force, dt=0.01
)
```

### Vendor Integration

```python
from surgical_robotics.vendors import IntuitiveDaVinciInterface

# Connect to Da Vinci Xi
robot = IntuitiveDaVinciInterface(model="xi")
robot.connect()
robot.initialize()

# Start teleoperation
robot.engage_clutch("arm_0")
robot.release_clutch("arm_0")

# Get telemetry
state = robot.get_system_state()
```

### CLI Usage

```bash
# Run simulation
surgical-robotics simulate --scene laparoscopic --render

# Generate configuration
surgical-robotics config --generate davinci_xi --output config.json

# Run diagnostics
surgical-robotics diagnose --full

# System status
surgical-robotics status --json
```

## Project Structure

```
surgical-robotics/
├── src/surgical_robotics/
│   ├── core/                 # Core components
│   ├── davinci/              # Da Vinci systems
│   ├── neurosurgical/        # Neurosurgical robots
│   ├── orthopedic/           # Orthopedic robots
│   ├── vendors/              # Vendor integrations
│   │   ├── intuitive.py      # Intuitive Surgical
│   │   ├── medtronic.py      # Medtronic Hugo/Mazor
│   │   ├── stryker.py        # Stryker Mako
│   │   ├── zimmer.py         # Zimmer ROSA
│   │   └── smith_nephew.py   # Smith+Nephew CORI
│   ├── communication/        # Communication protocols
│   │   ├── ros2.py           # ROS2 bridge
│   │   ├── dicom.py          # DICOM integration
│   │   ├── hl7.py            # HL7/FHIR
│   │   └── realtime.py       # UDP/TCP transport
│   ├── control/              # Advanced control
│   │   ├── mpc.py            # Model Predictive Control
│   │   ├── impedance.py      # Impedance control
│   │   ├── rl_interface.py   # RL environment
│   │   └── teleoperation.py  # Bilateral teleoperation
│   ├── simulation/           # Simulation environment
│   │   ├── physics.py        # Physics backends
│   │   ├── scene.py          # Surgical scenes
│   │   ├── tissue.py         # Tissue simulation
│   │   └── robot_sim.py      # Robot simulation
│   ├── workflow/             # Workflow management
│   │   ├── state_machine.py  # State machines
│   │   ├── procedure.py      # Procedure workflows
│   │   └── safety_states.py  # Safety monitoring
│   ├── telemetry/            # Telemetry & logging
│   │   ├── logging.py        # Structured logging
│   │   ├── metrics.py        # Metrics collection
│   │   └── recording.py      # Data recording
│   ├── compliance/           # Compliance helpers
│   │   ├── iec62304.py       # Software lifecycle
│   │   ├── iec60601.py       # Electrical safety
│   │   └── fda.py            # FDA QSR
│   └── cli/                  # Command-line interface
├── examples/                 # Example scripts
├── tests/                    # Test suite
└── pyproject.toml
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=surgical_robotics --cov-report=html

# Run specific test modules
pytest tests/test_simulation.py tests/test_control.py -v
```

## Safety Considerations

This library is intended for **research and educational purposes only**.

- **NOT for clinical use** without proper regulatory approval
- All safety-critical features require extensive validation
- Force limits and collision detection must be verified
- Vendor integrations are reference implementations only

## Key Concepts

### Control Architecture
- Hierarchical control with safety supervisor
- 1kHz control loops with real-time support
- Force/torque limiting at multiple levels

### Safety Systems
- IEC 62443 compliant safety states
- Virtual fixtures and haptic boundaries
- Watchdog monitoring and emergency stop
- Audit logging for all safety events

### Simulation
- Modular physics with multiple backends
- Real-time capable tissue deformation
- Bleeding and cautery simulation
- Recording and playback for analysis

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Citation

```bibtex
@software{surgical_robotics,
  title = {Surgical Robotics Control Systems},
  year = {2024},
  version = {0.2.0},
  url = {https://github.com/surgical-robotics/surgical-robotics}
}
```
