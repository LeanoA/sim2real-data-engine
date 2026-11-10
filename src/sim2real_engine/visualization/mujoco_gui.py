"""Native desktop 3D OpenGL viewer integration for DeepMind MuJoCo 4.x."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from sim2real_engine.core.kinematics import DampedLeastSquaresIK
from sim2real_engine.simulators.base import PrivilegedPickPlaceExpert
from sim2real_engine.simulators.mujoco_native import MuJoCoPhysicsAdapter


def run_interactive_mujoco_viewer(
    seed: int = 42,
    episodes: int = 3,
    model_path: str | Path | None = None,
) -> None:
    """Launch the official native MuJoCo 3D desktop viewer running closed-loop pick-and-place.

    Supports running both privileged IK expert demonstrations and trained 7D reactive policies.
    Renders physical contact manifolds, Coulomb friction, gravity dynamics, and articulated
    motion in real time inside a high-performance OpenGL/GLFW desktop window.
    """
    # macOS AppKit/Cocoa main-thread requirement: re-exec under mjpython if not already inside it
    if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        mjpython_path = Path(sys.executable).parent / "mjpython"
        if mjpython_path.is_file():
            os.execv(str(mjpython_path), [str(mjpython_path), *sys.argv])

    adapter = MuJoCoPhysicsAdapter()
    expert: PrivilegedPickPlaceExpert | None = None
    bridge = None
    policy_model = None

    if model_path is not None:
        from sim2real_engine.learning.policy import BehaviorCloningMLP
        from sim2real_engine.server.bridge import PolicyInferenceBridge
        from sim2real_engine.server.chunk_buffer import TemporalEnsembleBuffer

        ckpt = Path(model_path)
        if not ckpt.is_file():
            msg = f"Policy checkpoint not found: {ckpt}"
            raise FileNotFoundError(msg)

        policy_model, metadata = BehaviorCloningMLP.load_checkpoint(ckpt)
        buffer = TemporalEnsembleBuffer(
            action_dim=policy_model.action_dim, horizon=policy_model.horizon
        )
        policy_fn = policy_model.to_policy_fn(metadata)
        bridge = PolicyInferenceBridge(buffer=buffer, policy_fn=policy_fn)
        print(f"Loaded trained policy: {ckpt} (action_dim={policy_model.action_dim})")
    else:
        ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
        expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    joint_state, ee_pose, obj_pose = adapter.reset(seed=seed)
    current_gripper = 0.0
    if expert is not None:
        expert.reset(initial_object_pose=obj_pose)

    print(f"Launching MuJoCo {mujoco.__version__} Native 3D Viewer...")
    print("Press Spacebar to pause/unpause, double-click to select geoms, drag to perturb.")

    is_paused = False

    def on_key(keycode: int) -> None:
        nonlocal is_paused
        if keycode == 32:  # GLFW_KEY_SPACE
            is_paused = not is_paused
            status = "PAUSED" if is_paused else "RESUMED"
            print(f"[Viewer] Simulation {status}")

    with mujoco.viewer.launch_passive(
        adapter.model, adapter.data, key_callback=on_key
    ) as viewer:
        ep_count = 0
        current_seed = seed
        step_idx = 0

        while viewer.is_running() and ep_count < episodes:
            if is_paused:
                viewer.sync()
                time.sleep(0.02)
                continue

            step_start = time.time()

            if bridge is not None and policy_model is not None:
                act_chunk = bridge.infer_step(
                    joint_state,
                    step_idx,
                    ee_pose=ee_pose,
                    object_pose=obj_pose,
                    gripper_state=current_gripper,
                )
                action_targets = tuple(float(x) for x in act_chunk)
                step_record = adapter.step(action_targets)
                current_gripper = (
                    1.0
                    if step_record.gripper_action and step_record.gripper_action >= 0.45
                    else 0.0
                )
            elif expert is not None:
                q_cur = np.asarray(joint_state.positions_rad, dtype=np.float64)
                jacobian = adapter.get_jacobian()
                action_targets = expert.compute_action(
                    q_current=q_cur,
                    q_home=adapter.home_configuration,
                    ee_pose=ee_pose,
                    jacobian=jacobian,
                    emit_gripper=True,
                )
                step_record = adapter.step(action_targets)

            joint_state = step_record.joint_state
            ee_pose = step_record.end_effector_pose
            obj_pose = step_record.privileged_object_pose
            step_idx += 1

            viewer.sync()

            is_finished = (
                (expert is not None and expert.stage == "COMPLETED")
                or step_record.is_terminal
                or step_idx >= 120
            )
            if is_finished:
                time.sleep(0.8)
                ep_count += 1
                current_seed += 1
                step_idx = 0
                joint_state, ee_pose, obj_pose = adapter.reset(seed=current_seed)
                current_gripper = 0.0
                if bridge is not None:
                    bridge.reset()
                elif expert is not None:
                    expert.reset(initial_object_pose=obj_pose)

            # Match physics simulation time to real time
            dt_step = 0.02
            elapsed = time.time() - step_start
            if dt_step > elapsed:
                time.sleep(dt_step - elapsed)
