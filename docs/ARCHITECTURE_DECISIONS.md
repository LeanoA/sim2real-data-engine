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
