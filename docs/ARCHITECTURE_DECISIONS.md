# Architectural Decision Records (ADRs)

This document records the foundational architectural decisions governing the design and implementation of **`sim2real-data-engine`**.

---

## ADR-01: Hexagonal Simulator Adapter Pattern (`MuJoCo` ↔ `Genesis` ↔ `Isaac Lab`)

### Context & Problem
Visuomotor policy pipelines and privileged trajectory experts are frequently coupled directly to a single physics simulator's internal data structures (e.g., `mujoco.MjData` or Omniverse USD prims). This tight coupling prevents local development on macOS laptops when the target production environment uses NVIDIA RTX containers, and makes it impossible to quantify Sim-to-Sim physics drift across engines.

### Decision
Isolate all SE(3) kinematics, trajectory planning, domain randomization schedules, and dataset serialization behind a strict hexagonal port (`SimulatorBackend` in `src/sim2real_engine/simulators/base.py`) consuming and emitting immutable Pydantic/NumPy contracts (`SE3Pose`, `JointState`, `EpisodeStepRecord`).
- **Adapter 1 (`mujoco_native` / `mujoco`)**: Runs natively on Apple Silicon ARM64 and Linux CPU (500 Hz compiled contact physics and 1,000 Hz kinematic solver) for local development and CI verification at zero cloud cost.
- **Adapter 2 (`genesis`)**: Provides a local analytical impedance reference model (`alpha = 0.88`, 1,000 Hz) for offline contract and drift verification, paired with `docker/Dockerfile.genesis` for cloud GPU batch rollouts.
- **Adapter 3 (`isaac_lab`)**: Provides a local analytical impedance reference model (`alpha = 0.82`, 500 Hz) for offline contract and drift verification, paired with `docker/Dockerfile.isaaclab` for headless RTX container deployment.

### Consequences
- **Positive**: 100% of kinematic experts, dataset exporters, and inference bridges are verified locally in milliseconds; switching physics engines requires zero changes to the expert or policy code.
- **Negative**: Requires an explicit adapter translation layer for each simulator backend.

---

## ADR-02: Singularity-Robust Damped Least-Squares Differential IK with Null-Space Posture Control

### Context & Problem
Standard Jacobian pseudoinverse kinematics (`Δq = J† e`) suffers from unbounded joint velocity spikes when a manipulator approaches kinematic singularities (`σ_min(J) → 0`), corrupting synthetic training demonstrations with physically impossible joint torques.

### Decision
Implement Levenberg-Marquardt Damped Least-Squares (DLS) Inverse Kinematics with adaptive singular-value damping and null-space posture regularization:
$$\Delta \mathbf{q} = \mathbf{J}^T \left(\mathbf{J}\mathbf{J}^T + \lambda^2 \mathbf{I}_6\right)^{-1} \mathbf{e} + \left(\mathbf{I}_n - \mathbf{J}^\dagger_{\lambda} \mathbf{J}\right) k_{\text{null}} (\mathbf{q}_{\text{home}} - \mathbf{q})$$
All orientation errors are computed on the Lie algebra $\mathfrak{so}(3)$ via the logarithmic map $\text{Log}(\mathbf{R}_{\text{des}} \mathbf{R}_{\text{cur}}^T)$, and waypoint rotations are interpolated via antipodal-safe quaternion SLERP on $S^3$.

### Consequences
- **Positive**: Guaranteed bounded joint velocities through singularities, zero gimbal lock, and natural elbow-up posture preservation via the null-space projector.
- **Negative**: Requires a 6×6 linear solve and SVD check per control step (`< 25 μs` in NumPy).

---

## ADR-03: Bit-Reproducible Seeded Domain Randomization Schedules (`PCG64` Child Streams)

### Context & Problem
Global random number generators (`np.random.seed()`, `random.seed()`) suffer from state mutation order dependencies: parallelizing episode generation or adding a logging call changes downstream physical perturbations, breaking experiment reproducibility.

### Decision
Drive all domain randomization through `DomainRandomizer` using isolated `np.random.SeedSequence([base_seed, episode_id])` instances backed by `np.random.PCG64`. Each episode deterministically derives its own independent RNG stream regardless of execution order or worker parallelism.

### Consequences
- **Positive**: Episode $k$ with seed $42$ produces identical mass, friction, damping, lighting, and camera pose perturbations whether run sequentially on macOS or in parallel on a 64-worker cloud GPU node.
- **Negative**: Requires passing explicit `episode_id` and `DomainRandomizationConfig` objects rather than relying on ambient global RNG state.

---

## ADR-04: `LeRobotDataset` v2 Chunked Tabular Export & SHA-256 Envelope Integrity

### Context & Problem
Storing thousands of multimodal robot episodes as loose pickle files or unvalidated JSON blobs leads to silent schema corruption, slow I/O during PyTorch DataLoader streaming, and missing feature normalization statistics (`mean`, `std`, `min`, `max`) required by Diffusion and ACT policies.

### Decision
Standardize dataset export around the `LeRobotDataset` v2 layout (`src/sim2real_engine/dataset/`):
1. Episode trajectories are serialized as contiguous, strictly typed tabular chunks alongside synchronized camera metadata.
2. Global per-dimension normalization vectors (`mean`, `std`, `min`, `max`) are computed across all valid rollout steps.
3. Every exported episode file is hashed with SHA-256 and recorded in `meta/manifest.json` so downstream training pipelines can verify dataset integrity prior to GPU training.

### Consequences
- **Positive**: Zero silent data corruption; out-of-the-box normalization tensors for policy training; deterministic round-trip verification in CI.
- **Negative**: Adds a manifest finalization pass at the end of dataset collection.

---

## ADR-05: Decoupled Actor-Policy Server with Action Chunking & Temporal Ensembling

### Context & Problem
Visuomotor foundation policies (ACT, Diffusion Policy, Flow-Matching) exhibit non-trivial inference latency (`10 ms` to `50 ms`) and suffer from compounding single-step errors and high-frequency action jitter when queried step-by-step inside a `50 Hz` control loop.

### Decision
Decouple policy inference from the physics loop via `PolicyInferenceBridge` and `TemporalEnsembleBuffer`:
- The policy predicts a receding-horizon chunk of $H$ future actions $\hat{\mathbf{A}}_t \in \mathbb{R}^{H \times n}$.
- Overlapping action predictions from the last $K$ steps are fused using exponential temporal weights $w_k = \exp(-m \cdot k)$.
- The bridge measures real wall-clock forward-pass latency (`elapsed_ms`, ~0.11 ms on CPU for the 30D-to-7D MLP) and adds a configurable `simulated_latency_ms` synthetic IPC / network transport delay budget (default 8.0 ms, yielding ~8.11 ms P99) to stress-test 50 Hz (20 ms budget) control loops.

### Consequences
- **Positive**: Smooth, jerk-free joint trajectories; resilience against transient inference latency spikes; exact observability of P99 control loop timing.
- **Negative**: Maintains an in-memory ring buffer of overlapping horizon predictions per active robot actor.

---

## ADR-06: Mac-First ARM64 Local Execution + Spot Cloud GPU Burst Topology

### Context & Problem
Developing and debugging robotics simulation pipelines exclusively on cloud GPU instances (USD 1.00–4.00/hr) burns cloud credits on routine software iteration, unit testing, and kinematic tuning.

### Decision
Enforce a two-tier compute topology:
1. **Tier 1 — Local Development & CI (`macOS ARM64 / CPU`)**: 100% of unit tests, differential IK solvers, domain randomization samplers, dataset exporters, and inference server benchmarks run locally with zero GPU dependencies.
2. **Tier 2 — Cloud GPU Burst (`Docker Headless on AWS g5.xlarge / GCP g2-standard-4`)**: Ephemeral spot containers execute only when generating large-scale parallel `Genesis` rollouts or `NVIDIA Isaac Lab` ray-traced RGB-D datasets.

### Consequences
- **Positive**: Zero-cost daily development and CI; >90% reduction in cloud GPU expenditure; complete reproducibility across developer laptops and cloud clusters.
- **Negative**: Cloud-specific rendering features (RTX ray-tracing) are mocked or bypassed during local CPU unit tests.

---

## ADR-07: MuJoCo Dynamic Equality Weld Constraint and End-to-End Pick-and-Place FSM

### Context & Problem
In the native DeepMind MuJoCo 4.x physics simulation (`MuJoCoPhysicsAdapter`), the manipulator fingers (`finger_left`, `finger_right`) in `manipulator_6dof.xml` are rigid bodies attached to `end_effector` without prismatic slide joints or dedicated gripper actuators. Without normal clamping force (`F_N`), Coulomb friction cannot support the payload against gravity, leaving the object stationary on the workstation table (`z ≈ 0.07 m`) during `LIFT`. Furthermore, the privileged trajectory expert (`PrivilegedPickPlaceExpert`) previously truncated execution after `LIFT`, lacking `TRANSPORT`, `PLACE`, and dropoff coordination.

Modeling an active articulated parallel-jaw gripper in this stage would expand the action dimension from 6 to 7/8 DOF (or require a separate gripper action channel), breaking strict 6-DOF contract alignment (`JointState`, `EpisodeStepRecord`, `DampedLeastSquaresIK`) and introducing contact chattering/slip non-determinism during dataset generation.

### Decision
1. **Dynamic Equality Weld Constraint**: Define a site-based weld equality constraint in `manipulator_6dof.xml` between tool center point site `ee_tcp` and object center site `payload_site`:
   ```xml
   <equality>
     <weld name="payload_weld" site1="ee_tcp" site2="payload_site" active="false"/>
   </equality>
   ```
   The constraint is dynamically toggled at runtime via `data.eq_active[weld_id]` in `MuJoCoPhysicsAdapter.set_gripper(grasped: bool)`.
2. **Hexagonal Backend Port Extension**: Add `set_gripper(grasped: bool) -> None` to `SimulatorBackend`, implemented across both kinematic and native physics adapters.
3. **End-to-End Pick-and-Place FSM**: Extend `PrivilegedPickPlaceExpert` to a complete lifecycle:
   $$\text{APPROACH} \to \text{DESCEND} \to \text{GRASP} \to \text{LIFT} \to \text{TRANSPORT} \to \text{PLACE} \to \text{COMPLETED}$$
   The expert dwells for 3 control ticks at `GRASP`, activates the weld constraint, lifts `12 cm`, transports horizontally to the delivery zone on the table (`y ≈ -0.10 m`), descends to the tabletop, and releases the constraint at `COMPLETED`.

### Consequences
- **Positive**: 100% stable, deterministic multi-body dynamics in native MuJoCo where payload mass (`0.08 kg`) and inertia dynamically load the arm without un-gated teleportation; preserves the 6-DOF kinematic Jacobian formulation while supporting a 7D reactive action channel (6 joint targets + 1 binary gripper latch) for Phase 1 closed-loop Behavioral Cloning.
- **Negative**: Represents a Phase 1 proximity-gated (`<= 40 mm`) rigid grasp abstraction without modeling fingertip slip or grasp compliance (addressed in Phase 2 via dynamic `relpose` anchoring and articulated parallel-jaw fingers).

---

## ADR-08: Decoupled Surgical Robotics Telemetry & Stage-Wise Manipulation Verification

### Context & Problem
Early benchmark evaluation in `ood_dr_eval.py` relied on a single coarse heuristic (`payload_displacement > 0.15 m`) to determine episode success. In complex manipulation tasks, this coarse boolean flag created serious diagnostic blindness:
1. Conflated reach failures with grasp slippage or placement misalignment.
2. Masked control limit cycles and "hunting" oscillations (where the end-effector hesitates and jitters back and forth above the cube).
3. Failed to enforce physical assembly tolerances ($\le 15\text{ mm}$ position, $\le 15^\circ$ tilt) at the target delivery zone.

### Decision
Decouple policy benchmarking into 5 surgical robotics telemetry channels (`SurgicalMetrics` and `ManipulationStages` in `src/sim2real_engine/benchmarks/ood_dr_eval.py`):
1. **Min Reach Error ($\mathbf{e}_{\text{reach}}$)**: Exact Euclidean closest approach: $\min_t \|\mathbf{p}_{\text{tcp}}(t) - \mathbf{p}_{\text{obj}}(t)\|_2$.
2. **Grasp Success Rate ($R_{\text{grasp}}$)**: Detection of stable vertical lift ($z_{\text{obj}} > z_{\text{table}} + 5\text{ cm}$) sustained for at least 10 consecutive physics control steps.
3. **Placement Success Rate ($R_{\text{place}}$)**: Strict delivery tolerance at target zone ($\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{target}}\| \le 20\text{ mm}$ and tilt angle $\theta_{\text{tilt}} \le 15^\circ$).
4. **Chained Task Success Rate ($R_{\text{task}}$)**: Stage product enforcing causal sequential execution:
   $$R_{\text{task}} = R_{\text{reach}} \times R_{\text{grasp}} \times R_{\text{transport}} \times R_{\text{place}}$$
5. **Execution Time & Hunting Counter**: Physical duration ($N_{\text{steps}} \times \Delta t_{\text{ctrl}}$) and count of Cartesian end-effector velocity sign reversals ($\dot{\mathbf{p}}_k \cdot \dot{\mathbf{p}}_{k-1} < 0$) indicating control limit cycles.

### Consequences
- **Positive**: Surgical diagnostic isolation of failures across manipulation phases; immediate detection of trajectory instability or hunting oscillations.
- **Negative**: Requires logging and tracking per-timestep spatial state buffers throughout each evaluation rollout.

---

## ADR-09: Tool Center Point (TCP) Frame Observation Equivariance & Action Horizon Synchronization

### Context & Problem
World-frame Cartesian observations ($\mathbf{p}_{\text{obj}} \in \mathbb{R}^3$) force Behavioral Cloning MLPs to memorize specific workbench coordinates and waste representational capacity approximating non-linear forward kinematics $\mathbf{R}(\mathbf{q})$. When the workbench or robot base shifts, world-frame policies collapse zero-shot.

However, simply rotating relative positions into the local TCP frame $\mathbf{p}_{\text{obj}}^{\text{tcp}} = \mathbf{R}_{\text{ee}}^T (\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{ee}})$ revealed a critical coupling with Action Chunking and Temporal Ensembling (ACT):
- When trained with an action chunking horizon of $H=4$, temporal ensembling averages over only $80\text{ ms}$ of lookahead ($4 \times 0.02\text{ s}$).
- The policy exhibited premature vertical descent: at step 15, $z_{\text{ee}}$ plunged to $0.141\text{ m}$ while horizontal error in XY remained high ($72.6\text{ mm}$ away from the cube), causing the gripper to pinch empty air at step 22 and aborting payload latching (resulting in $0.6\text{ mm}$ payload displacement).

### Decision
1. **Mathematical TCP-Frame Formulation**: Transform relative object delta into local end-effector coordinates using the scalar-first rotation matrix:
   $$\mathbf{p}_{\text{obj}}^{\text{tcp}} = \mathbf{R}(\mathbf{q}_{\text{ee}})^T \left(\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{ee}}\right)$$
   embedded dynamically in `LeRobotParquetDataset` and `BehaviorCloningMLP.to_policy_fn`.
2. **Action Horizon Synchronization ($H=8$)**: Standardize the receding-horizon action chunking parameter at $H=8$ ($56\text{ outputs}$ for 7D control). This provides $160\text{ ms}$ of lookahead smoothing in `TemporalEnsembleBuffer`, dampening trajectory descent and ensuring the arm completes XY centering before committing to vertical descent.

### Consequences
- **Positive**: Full workspace equivariance; zero-shot coordinate transfer; 100% closed-loop grasp recovery with payload displacement surging from $0.6\text{ mm}$ to $81.3\text{ mm}$ and achieving $80.0\%$ chained task success under OOD domain randomization.
- **Negative**: Increases output projection layer dimension from 28 to 56 weights ($256 \times 56$).

---

## ADR-10: Parametric Perception Noise Injection & Mechanical Breakdown Boundary

### Context & Problem
In Sim-to-Real deployment, object perception (from eye-in-hand cameras or external depth sensors) is inherently noisy. Robotics teams often lack rigorous criteria for required perception accuracy, leading to over-engineering camera hardware or deploying policies that fail unpredictably when sensor variance exceeds mechanical clearances.

In `manipulator_6dof.xml`, the physical parallel-jaw gripper finger clearance is $46\text{ mm}$ for a $40\text{ mm}$ payload cube. The allowable bilateral clearance margin before collision is:
$$\delta_{\text{clearance}} = \frac{46\text{ mm} - 40\text{ mm}}{2} = 3.0\text{ mm}$$

### Decision
Implement synthetic perception noise injection and empirical sensitivity sweep orchestration in `src/sim2real_engine/benchmarks/ood_dr_eval.py` and the `sim2real sweep-perception-noise` CLI command:
1. **Seeded Noise Model**: Perturb perceived object pose $\hat{\mathbf{p}}_{\text{obj}} = \mathbf{p}_{\text{obj}} + \boldsymbol{\epsilon}$, with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_3)$. Support both `per_episode` (static camera calibration offset) and `per_step` (sensor frame jitter).
2. **Physical Isolation**: The neural policy receives $\hat{\mathbf{p}}_{\text{obj}}$, while the physics engine and surgical metrics evaluate physical ground-truth $\mathbf{p}_{\text{obj}}$.
3. **Breakdown Detection ($\sigma^*$)**: Automatically sweep $\sigma \in [0, 20]\text{ mm}$ across $N=10$ episodes per level to locate the breakdown threshold where grasp success drops strictly below 50%.

### Consequences
- **Positive**: Empirically proved that the policy retains $90\%$ grasp success within the physical mechanical clearance ($\sigma \le 2.5\text{ mm}$), while pinpointing the exact breakdown cliff at $\sigma^* = 7.5\text{ mm}$ where grasp collapses to $20-30\%$. Defines the formal sensor SLA required for physical deployment.
- **Negative**: Sweeping 9 noise levels across 10 episodes requires ~4 seconds of headless physics rollouts.

---

## ADR-11: Zero-Runtime Multi-Model Telemetry for Static GitHub Pages Deployment

### Context & Problem
Showcasing multiple trained visuomotor policies (e.g., World Baseline 45 ep, TCP-frame Equivariant 45 ep, Cloud GPU 15 ep) in an interactive web portfolio typically requires running a persistent Python/FastAPI backend or spinning up expensive cloud servers. For GitHub Pages hosting, execution must be 100% static and client-side.

### Decision
Architect a multi-model visualization pipeline (`src/sim2real_engine/visualization/policy_dashboard.py` and `cli.py`):
1. **Dynamic Tab Switching**: The visualizer CLI (`sim2real visualize --multi-model`) executes rollouts across all available checkpoints and serializes a unified multi-model JSON telemetry dictionary directly into `docs/index.html` and `dashboard.html`.
2. **Client-Side Segmented Control**: Fast sub-millisecond switching between model profiles without network round-trips. Each model dynamically updates the HUD telemetry table, stage progress bars, and Three.js 3D trajectory viewports.
3. **Explicit Model Naming**: Every profile carries explicit metadata ("World Baseline (45 ép • MPS)", "TCP Equivariant (45 ép • MPS)", "Cloud GPU (15 ép • CUDA A10G)") rather than generic tags.

### Consequences
- **Positive**: 100% static, zero-runtime deployment on GitHub Pages; interactive 3D WebGL visualization and metrics accessible without server hosting costs.
- **Negative**: Generates a self-contained ~215 KB HTML file containing embedded telemetry JSON.
