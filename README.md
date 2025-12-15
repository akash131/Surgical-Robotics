# Surgical Robotics Control Systems

A comprehensive Python framework for surgical robot control systems, including Da Vinci-class multi-arm platforms, neurosurgical robots, and orthopedic surgical systems.

## Features

### Da Vinci-Class Systems
- **Multi-arm robotic platforms** with master-slave teleoperation
- **Haptic feedback systems** for force sensing and rendering
- **3D stereoscopic vision** with depth perception
- **Instrument tracking** and collision avoidance
- Remote center of motion (RCM) constraint enforcement

### Neurosurgical Robots
- **Sub-millimeter positioning accuracy** (±0.1mm)
- **Integration with intraoperative MRI/CT** imaging
- **Tremor cancellation** with adaptive filtering
- Stereotactic frame registration
- Image-guided navigation

### Orthopedic Robots
- **Bone cutting and milling systems** with various tool types
- **Force feedback** for tissue differentiation
- **Patient-specific surgical planning** from CT/MRI
- Haptic boundaries for safe cutting
- Adaptive cutting control

## Installation

```bash
# Clone the repository
git clone https://github.com/surgical-robotics/surgical-robotics.git
cd surgical-robotics

# Install in development mode
pip install -e ".[dev]"

# Or install with optional dependencies
pip install -e ".[dev,vision,simulation]"
```

## Quick Start

### Da Vinci Robot

```python
from surgical_robotics.davinci import DaVinciRobot
from surgical_robotics.core.base import Pose
import numpy as np

# Initialize the robot
robot = DaVinciRobot(num_patient_arms=3, has_endoscope=True)
robot.initialize()
robot.calibrate()

# Set remote center of motion for an arm
robot.set_rcm(0, np.array([0.0, 0.0, 0.3]))

# Start teleoperation
robot.start_teleoperation()

# Move arm to a target pose
target = Pose(
    position=np.array([0.1, 0.05, 0.25]),
    orientation=np.array([1, 0, 0, 0])
)
robot.move_to_pose(0, target)
```

### Neurosurgical Robot

```python
from surgical_robotics.neurosurgical import (
    NeurosurgicalRobot,
    StereotacticFrame,
    NeurosurgicalTarget,
)
import numpy as np

# Initialize the robot
robot = NeurosurgicalRobot()
robot.initialize()
robot.calibrate()

# Set up stereotactic frame
frame = StereotacticFrame(frame_id="leksell_frame")
frame.add_fiducial(np.array([0, 0, 0]))
frame.add_fiducial(np.array([0.1, 0, 0]))
frame.add_fiducial(np.array([0, 0.1, 0]))

# Register frame to image coordinates
image_points = [...]  # From preoperative imaging
frame.compute_registration(image_points)
robot.set_stereotactic_frame(frame)

# Define and set surgical target
target = NeurosurgicalTarget(
    name="tumor_biopsy",
    position=np.array([0.05, 0.02, -0.04]),
    entry_point=np.array([0.05, 0.02, 0.0])
)
robot.set_target(target)

# Approach and advance to target
robot.approach_target(approach_distance=0.02)
robot.advance_to_target(step_size=0.001)
```

### Orthopedic Robot

```python
from surgical_robotics.orthopedic import (
    OrthopedicRobot,
    CuttingMode,
    BoneCutPlanner,
    PatientModel,
    BoneType,
)
import numpy as np

# Initialize the robot
robot = OrthopedicRobot()
robot.initialize()
robot.calibrate()

# Register bone anatomy
ct_landmarks = [...]  # From CT scan
robot_landmarks = [...]  # Digitized with robot
robot.register_bone("femur", ct_landmarks, robot_landmarks)

# Set cutting mode and start spindle
robot.set_cutting_mode(CuttingMode.BURR)
robot.start_spindle()
robot.set_coolant(True)

# Execute cutting path
path_points = [np.array([0.01, 0, -0.005]), np.array([0.02, 0, -0.005])]
robot.execute_cutting_path(path_points, feed_rate=0.003)

# Stop when done
robot.stop_spindle()
```

## Project Structure

```
surgical-robotics/
├── src/surgical_robotics/
│   ├── core/                 # Core components
│   │   ├── base.py          # Base classes (SurgicalRobot, Pose, etc.)
│   │   ├── kinematics.py    # Forward/inverse kinematics
│   │   └── safety.py        # Safety controllers, collision detection
│   ├── davinci/             # Da Vinci-class systems
│   │   ├── robot.py         # Main robot class
│   │   ├── haptics.py       # Haptic feedback
│   │   ├── vision.py        # Stereoscopic vision
│   │   └── instruments.py   # Surgical instruments
│   ├── neurosurgical/       # Neurosurgical robots
│   │   ├── robot.py         # Main robot class
│   │   ├── imaging.py       # MRI/CT integration
│   │   └── tremor.py        # Tremor cancellation
│   └── orthopedic/          # Orthopedic robots
│       ├── robot.py         # Main robot class
│       ├── cutting.py       # Bone cutting systems
│       ├── force_feedback.py # Force sensing
│       └── planning.py      # Surgical planning
├── tests/                   # Test suite
├── pyproject.toml          # Project configuration
└── README.md
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=surgical_robotics --cov-report=html

# Run specific test file
pytest tests/test_davinci.py -v
```

## Safety Considerations

This library is intended for **research and educational purposes only**.

- **NOT for clinical use** without proper regulatory approval
- All safety-critical features require extensive validation
- Force limits and collision detection must be verified for specific applications
- Tremor cancellation parameters should be tuned for individual patients

## Key Concepts

### Kinematics
- Uses Denavit-Hartenberg (DH) convention for robot modeling
- Forward kinematics via homogeneous transformations
- Inverse kinematics using damped least squares

### Safety Systems
- Multi-level safety classification (Normal, Caution, Warning, Critical)
- Virtual fixtures for motion guidance
- Force/torque limiting and monitoring
- Collision detection between instruments

### Haptic Feedback
- Force scaling for teleoperation
- Tissue interaction models
- Vibration rendering for texture feedback

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{surgical_robotics,
  title = {Surgical Robotics Control Systems},
  year = {2024},
  url = {https://github.com/surgical-robotics/surgical-robotics}
}
```
