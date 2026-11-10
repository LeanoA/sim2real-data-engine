"""Local analytical impedance reference adapters for Genesis and NVIDIA Isaac Lab.

Implements the hexagonal SimulatorBackend port with deterministic analytical tracking models
(alpha=0.88 for Genesis, alpha=0.82 for Isaac Lab) to verify cross-backend contracts and
quantify baseline tracking drift on macOS ARM64 / Linux CPU when cloud GPU containers are
not attached.
"""

from __future__ import annotations

from sim2real_engine.simulators.mujoco_adapter import MuJoCoKinematicAdapter


class GenesisAdapter(MuJoCoKinematicAdapter):
    """Local analytical impedance reference adapter for the Genesis backend port.

    Emulates target solver tracking dynamics with relaxed joint compliance
    (kp=44.0 N*m/rad, kd=3.9 N*m*s/rad, alpha=0.88) and 1,000 Hz nominal stepping
    for offline contract and CI drift verification without requiring a GPU container.
    """

    def __init__(self, control_dt: float = 0.02) -> None:
        super().__init__(control_dt=control_dt, kp=44.0, kd=3.9, tracking_alpha=0.88)
        self.backend_name = "genesis"
        self.nominal_physics_hz = 1000

    def __repr__(self) -> str:
        return f"GenesisAdapter(dof={self.dof}, hz={self.nominal_physics_hz})"


class IsaacLabAdapter(MuJoCoKinematicAdapter):
    """Local analytical impedance reference adapter for the NVIDIA Isaac Lab backend port.

    Emulates target solver tracking dynamics at 500 Hz nominal stepping with stiff
    joint impedance (kp=48.0 N*m/rad, kd=4.2 N*m*s/rad, alpha=0.82) for offline
    contract and CI drift verification when headless RTX containers are not attached.
    """

    def __init__(self, control_dt: float = 0.02) -> None:
        super().__init__(control_dt=control_dt, kp=48.0, kd=4.2, tracking_alpha=0.82)
        self.backend_name = "isaac_lab"
        self.nominal_physics_hz = 500

    def __repr__(self) -> str:
        return f"IsaacLabAdapter(dof={self.dof}, hz={self.nominal_physics_hz})"
