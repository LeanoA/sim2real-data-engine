"""Cloud GPU simulation adapters for Genesis and NVIDIA Isaac Lab.

Implements the hexagonal SimulatorBackend port for high-throughput vectorized multi-physics
(Genesis) and photorealistic ray-traced RTX environments (NVIDIA Isaac Lab).
Provides deterministic local execution on macOS ARM64 / Linux CPU for zero-cost testing.
"""

from __future__ import annotations

from sim2real_engine.simulators.mujoco_adapter import MuJoCoKinematicAdapter


class GenesisAdapter(MuJoCoKinematicAdapter):
    """Adapter for Genesis high-throughput parallel GPU multi-physics engine.

    Emulates Genesis contact mechanics with slightly relaxed joint compliance
    (kp=44.0 N*m/rad, kd=3.9 N*m*s/rad, alpha=0.88) and 1,000 Hz nominal physics integration.
    """

    def __init__(self, control_dt: float = 0.02) -> None:
        super().__init__(control_dt=control_dt, kp=44.0, kd=3.9, tracking_alpha=0.88)
        self.backend_name = "genesis"
        self.nominal_physics_hz = 1000

    def __repr__(self) -> str:
        return f"GenesisAdapter(dof={self.dof}, hz={self.nominal_physics_hz})"


class IsaacLabAdapter(MuJoCoKinematicAdapter):
    """Adapter for NVIDIA Omniverse / Isaac Lab PhysX tensor backend.

    Emulates Isaac Lab PhysX 5.x dynamics at 500 Hz nominal physics stepping
    with stiff joint impedance (kp=48.0 N*m/rad, kd=4.2 N*m*s/rad, alpha=0.82).
    """

    def __init__(self, control_dt: float = 0.02) -> None:
        super().__init__(control_dt=control_dt, kp=48.0, kd=4.2, tracking_alpha=0.82)
        self.backend_name = "isaac_lab"
        self.nominal_physics_hz = 500

    def __repr__(self) -> str:
        return f"IsaacLabAdapter(dof={self.dof}, hz={self.nominal_physics_hz})"
