# Architecture Specification — Sim2Real Data Engine (Phases 1–6)

This document is the authoritative contract specification for `sim2real-data-engine`. Every module, class signature, mathematical formulation, and acceptance gate defined below is frozen across phases so any engineer or autonomous coding agent can implement Phases 2 through 6 with zero interface drift.

---

## 1. Complete Target Directory Tree (Phases 1–6)

```text
sim2real-data-engine/
├── CLAUDE.md                                     # Operational manifest & fast terminal invocations
├── AGENTS.md                                     # 6-phase SDD protocol & model handover checklist
├── README.md                                     # Public showcase, diagrams & quickstart
├── ARTICLE_DRAFT.md                              # [Phase 6] Sim-to-Sim & synthetic data technical article
├── pyproject.toml                                # Python 3.13 + uv + pytest + ruff + mypy (--strict)
├── docs/
│   ├── ARCHITECTURE_SPECIFICATION.md             # This authoritative contract specification
│   └── ARCHITECTURE_DECISIONS.md                 # ADR-01 through ADR-06
├── docker/
│   ├── Dockerfile.genesis                        # [Phase 6] Headless GPU container for Genesis batch sweeps
│   ├── Dockerfile.isaaclab                       # [Phase 6] Headless RTX container for NVIDIA Isaac Lab
│   └── docker-compose.cloud.yml                  # [Phase 6] Multi-container GPU burst orchestration
├── src/
│   └── sim2real_engine/
│       ├── __init__.py                           # [Phase 1] Package version
│       ├── cli.py                                # [Phase 1-6] `sim2real` CLI entrypoint
│       ├── core/
│       │   ├── __init__.py                       # [Phase 1] Core exports
│       │   ├── contracts.py                      # [Phase 1] SE3Pose, JointState, DR config, EpisodeStepRecord
│       │   └── kinematics.py                     # [Phase 2] SO(3)/SE(3) Log/Exp, SLERP, Damped Least-Squares IK
│       ├── simulators/
│       │   ├── __init__.py                       # [Phase 2] Simulator adapter registry
│       │   ├── base.py                           # [Phase 2] SimulatorBackend Protocol & PrivilegedPickPlaceExpert
│       │   ├── mujoco_adapter.py                 # [Phase 2] Local ARM64/CPU deterministic physics adapter
│       │   └── cloud_adapters.py                 # [Phase 6] GenesisAdapter & IsaacLabAdapter
│       ├── randomization/
│       │   ├── __init__.py                       # [Phase 3] Randomization exports
│       │   └── samplers.py                       # [Phase 3] EpisodePerturbation, DomainRandomizer (PCG64)
│       ├── dataset/
│       │   ├── __init__.py                       # [Phase 4] Dataset exports
│       │   ├── manifest.py                       # [Phase 4] DatasetFeatureStats, DatasetManifest (SHA-256)
│       │   └── exporter.py                       # [Phase 4] LeRobotEpisodeExporter (chunked tabular + RGB-D)
│       ├── server/
│       │   ├── __init__.py                       # [Phase 5] Inference bridge exports
│       │   ├── chunk_buffer.py                   # [Phase 5] TemporalEnsembleBuffer (exp(-m*k) action chunking)
│       │   └── bridge.py                         # [Phase 5] PolicyInferenceBridge & LatencyTracker (P50/P95/P99)
│       └── benchmarks/
│           ├── __init__.py                       # [Phase 6] Benchmark exports
│           └── sim2sim_eval.py                   # [Phase 6] Cross-engine trajectory divergence & OOD evaluator
└── tests/
    ├── test_contracts_and_cli.py                 # [Phase 1] Core schemas & CLI verification
    ├── test_kinematics_and_mujoco.py             # [Phase 2] DLS IK, SLERP & MuJoCo rollout convergence
    ├── test_randomization.py                     # [Phase 3] Bit-exact seeded sampling & distribution bounds
    ├── test_dataset_exporter.py                  # [Phase 4] LeRobotDataset v2 round-trip & SHA-256 integrity
    ├── test_inference_server.py                  # [Phase 5] Action chunking temporal ensemble & P99 telemetry
    └── test_sim2sim_benchmarks.py                # [Phase 6] Cross-engine Sim-to-Sim evaluation suite
```

---

## 2. Phase-by-Phase Module & Mathematical Contracts

### Phase 2: Kinematics & Local MuJoCo Adapter (`core/kinematics.py`, `simulators/`)

#### Mathematical Formulation — Damped Least-Squares (Levenberg-Marquardt) Differential IK
Given current end-effector pose $\mathbf{T}_{\text{cur}} = (\mathbf{R}_{\text{cur}}, \mathbf{p}_{\text{cur}}) \in SE(3)$ and target pose $\mathbf{T}_{\text{des}} = (\mathbf{R}_{\text{des}}, \mathbf{p}_{\text{des}}) \in SE(3)$:
1. Spatial twist error $\mathbf{e} = [\Delta \mathbf{p}^T, \mathbf{e}_{\text{rot}}^T]^T \in \mathbb{R}^6$, where $\Delta \mathbf{p} = \mathbf{p}_{\text{des}} - \mathbf{p}_{\text{cur}}$ and $\mathbf{e}_{\text{rot}} = \text{Log}(\mathbf{R}_{\text{des}} \mathbf{R}_{\text{cur}}^T) \in \mathbb{R}^3$.
2. With geometric Jacobian $\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{6 \times n}$, damped pseudoinverse $\mathbf{J}^{\dagger}_{\lambda} = \mathbf{J}^T (\mathbf{J}\mathbf{J}^T + \lambda^2 \mathbf{I}_6)^{-1}$:
   $$\Delta \mathbf{q} = \mathbf{J}^{\dagger}_{\lambda} \mathbf{e} + \left(\mathbf{I}_n - \mathbf{J}^{\dagger}_{\lambda} \mathbf{J}\right) k_{\text{null}} (\mathbf{q}_{\text{home}} - \mathbf{q})$$
   where $\lambda(\mathbf{q}) = \lambda_{\max} \sqrt{1 - (\sigma_{\min}(\mathbf{J}) / \sigma_{\text{thresh}})^2}$ if $\sigma_{\min}(\mathbf{J}) < \sigma_{\text{thresh}}$, else $\lambda_{\min}$.

#### Required Interfaces
- `src/sim2real_engine/core/kinematics.py`:
  - `so3_log_map(rotation_error: NDArray[np.float64]) -> NDArray[np.float64]`
  - `quat_slerp(q0_wxyz: tuple[float, float, float, float], q1_wxyz: tuple[float, float, float, float], alpha: float) -> tuple[float, float, float, float]` (includes antipodal check $\mathbf{q}_0 \cdot \mathbf{q}_1 < 0$).
  - `compute_se3_error(current: SE3Pose, target: SE3Pose) -> NDArray[np.float64]` (returns 6D vector).
  - `class DampedLeastSquaresIK`:
    - `__init__(self, dof: int, lambda_min: float = 1e-3, lambda_max: float = 1e-1, sigma_thresh: float = 5e-2, k_null: float = 0.5) -> None`
    - `step(self, q_current: NDArray[np.float64], q_home: NDArray[np.float64], current_pose: SE3Pose, target_pose: SE3Pose, jacobian: NDArray[np.float64], max_step_rad: float = 0.1) -> NDArray[np.float64]`
- `src/sim2real_engine/simulators/base.py`:
  - `class SimulatorBackend(Protocol)`: `reset(seed: int) -> tuple[JointState, SE3Pose, SE3Pose]`, `step(joint_targets_rad: tuple[float, ...]) -> EpisodeStepRecord`, `get_jacobian() -> NDArray[np.float64]`.
  - `class PrivilegedPickPlaceExpert`: generates closed-loop reach, grasp, and lift trajectories using `DampedLeastSquaresIK`.
- `src/sim2real_engine/simulators/mujoco_adapter.py`:
  - `class MuJoCoKinematicAdapter`: deterministic local ARM64/CPU simulator adapter implementing `SimulatorBackend`.

---

### Phase 3: Seeded Domain Randomization (`randomization/samplers.py`)

#### Mathematical Formulation — Isolated PCG64 Perturbation Sampling
For episode index $e \in \mathbb{N}$ and base seed $s_0$, derive a deterministic child stream via `np.random.SeedSequence([s_0, e])` driving `np.random.PCG64`:
- Physical scales: $m_{\text{scale}} \sim \mathcal{U}(m_{\min}, m_{\max})$, $\mu_{\text{scale}} \sim \mathcal{U}(\mu_{\min}, \mu_{\max})$, $d_{\text{scale}} \sim \mathcal{U}(d_{\min}, d_{\max})$.
- Camera extrinsic jitter: $\Delta \mathbf{p}_{\text{cam}} \sim \mathcal{U}(-\delta_c, +\delta_c)^3$, orientation perturbation $\delta \boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{rot}}^2 \mathbf{I}_3)$ converted via exponential map to unit quaternion.

#### Required Interfaces
- `src/sim2real_engine/randomization/samplers.py`:
  - `class EpisodePerturbation(BaseModel)`: frozen Pydantic model holding `episode_id: int`, `derived_seed: int`, `mass_scale: float`, `friction_scale: float`, `damping_scale: float`, `lighting_intensity: float`, `camera_pose_offset: SE3Pose`.
  - `class DomainRandomizer`:
    - `__init__(self, config: DomainRandomizationConfig) -> None`
    - `sample_episode(self, episode_id: int) -> EpisodePerturbation`
    - `sample_batch(self, num_episodes: int) -> list[EpisodePerturbation]`

---

### Phase 4: LeRobotDataset v2 Exporter (`dataset/manifest.py`, `dataset/exporter.py`)

#### Mathematical Formulation — Feature Normalization & SHA-256 Envelope
Across $N$ recorded timesteps, compute per-dimension statistics (`mean`, `std`, `min`, `max`) for `observation.state` (`q`, `dq`) and `action` (`q_des`), and seal each episode file with a SHA-256 digest in `meta/manifest.json`.

#### Required Interfaces
- `src/sim2real_engine/dataset/manifest.py`:
  - `class FeatureStats(BaseModel)`: `mean: tuple[float, ...]`, `std: tuple[float, ...]`, `min: tuple[float, ...]`, `max: tuple[float, ...]`.
  - `class DatasetManifest(BaseModel)`: `dataset_name: str`, `format_version: Literal["lerobot_v2"]`, `num_episodes: int`, `total_steps: int`, `episode_sha256: dict[str, str]`, `feature_stats: dict[str, FeatureStats]`.
- `src/sim2real_engine/dataset/exporter.py`:
  - `class LeRobotEpisodeExporter`:
    - `__init__(self, root_dir: Path, dataset_name: str = "sim2real_pick_place") -> None`
    - `write_episode(self, episode_id: int, steps: Sequence[EpisodeStepRecord]) -> Path`
    - `finalize_manifest(self) -> DatasetManifest`
    - `verify_integrity(self) -> bool`

---

### Phase 5: Decoupled Policy Inference Server (`server/chunk_buffer.py`, `server/bridge.py`)

#### Mathematical Formulation — Receding-Horizon Action Chunking & Temporal Ensembling
At control step $t$, the policy outputs an action chunk $\hat{\mathbf{A}}_t \in \mathbb{R}^{H \times n}$ of horizon $H$. For overlapping chunks predicted at steps $t - k$ for $k \in \{0, \dots, K-1\}$, the executed command $\mathbf{u}_t \in \mathbb{R}^n$ is the exponentially weighted average:
$$\mathbf{u}_t = \frac{\sum_{k=0}^{K-1} \exp(-m \cdot k) \, \hat{\mathbf{a}}_{t | t-k}}{\sum_{k=0}^{K-1} \exp(-m \cdot k)}$$

#### Required Interfaces
- `src/sim2real_engine/server/chunk_buffer.py`:
  - `class TemporalEnsembleBuffer`:
    - `__init__(self, chunk_horizon: int, dof: int, decay_m: float = 0.01) -> None`
    - `push_chunk(self, action_chunk: NDArray[np.float64]) -> None`
    - `pop_ensembled_action(self) -> NDArray[np.float64]`
    - `reset(self) -> None`
- `src/sim2real_engine/server/bridge.py`:
  - `class LatencySummary(BaseModel)`: `count: int`, `p50_ms: float`, `p95_ms: float`, `p99_ms: float`, `max_ms: float`.
  - `class PolicyInferenceBridge`: deterministic Actor-Policy bridge with deadband filtering and P50/P95/P99 latency tracking.

---

### Phase 6: Cloud GPU Burst & Sim-to-Sim Benchmarks (`simulators/cloud_adapters.py`, `benchmarks/sim2sim_eval.py`)

#### Mathematical Formulation — Sim-to-Sim Trajectory Divergence
Given nominal trajectory $\{\mathbf{q}_t^{(A)}, \mathbf{p}_t^{(A)}\}_{t=1}^T$ on Backend $A$ (`mujoco`) and $\{\mathbf{q}_t^{(B)}, \mathbf{p}_t^{(B)}\}_{t=1}^T$ on Backend $B$ (`genesis` or `isaac_lab`) under identical action sequences and seeds:
$$\text{RMSE}_{\text{joint}}(A, B) = \sqrt{\frac{1}{T \cdot n} \sum_{t=1}^T \|\mathbf{q}_t^{(A)} - \mathbf{q}_t^{(B)}\|_2^2}, \qquad \text{RMSE}_{\text{EE}}(A, B) = \sqrt{\frac{1}{T} \sum_{t=1}^T \|\mathbf{p}_t^{(A)} - \mathbf{p}_t^{(B)}\|_2^2}$$

#### Required Interfaces
- `src/sim2real_engine/simulators/cloud_adapters.py`: `GenesisAdapter` and `IsaacLabAdapter` implementing `SimulatorBackend` with contact stiffness/damping fidelity profiles.
- `src/sim2real_engine/benchmarks/sim2sim_eval.py`:
  - `class Sim2SimMetrics(BaseModel)`: `reference_backend: BackendName`, `target_backend: BackendName`, `joint_rmse_rad: float`, `ee_position_rmse_m: float`, `grasp_success_rate_delta: float`, `throughput_steps_per_sec: float`.
  - `evaluate_sim2sim_gap(reference: BackendName, target: BackendName, num_episodes: int = 10, seed: int = 42) -> Sim2SimMetrics`

---

## 3. Phase-by-Phase Acceptance Gates

| Phase | Pytest Verification Target | CLI Subcommand Added |
| :--- | :--- | :--- |
| **Phase 1** | `uv run pytest tests/test_contracts_and_cli.py` | `sim2real info`, `sim2real backends`, `sim2real validate-config` |
| **Phase 2** | `uv run pytest tests/test_kinematics_and_mujoco.py` | `sim2real rollout --backend mujoco --episodes 5 --json` |
| **Phase 3** | `uv run pytest tests/test_randomization.py` | `sim2real sample-dr --seed 42 --episodes 3 --json` |
| **Phase 4** | `uv run pytest tests/test_dataset_exporter.py` | `sim2real export-dataset --episodes 5 --output-dir <dir> --json` |
| **Phase 5** | `uv run pytest tests/test_inference_server.py` | `sim2real benchmark-bridge --steps 200 --chunk-horizon 16 --json` |
| **Phase 6** | `uv run pytest tests/test_sim2sim_benchmarks.py` | `sim2real sim2sim-eval --ref mujoco --target isaac_lab --json` |
