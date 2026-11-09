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


def run_interactive_mujoco_viewer(seed: int = 42, episodes: int = 3) -> None:
    """Launch the official native MuJoCo 3D desktop viewer running closed-loop pick-and-place.

    Renders physical contact manifolds, Coulomb friction, gravity dynamics, and 6-DOF
    manipulator motion in real time inside a high-performance OpenGL/GLFW desktop window.
    """
    # macOS AppKit/Cocoa main-thread requirement: re-exec under mjpython if not already inside it
    if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        mjpython_path = Path(sys.executable).parent / "mjpython"
        if mjpython_path.is_file():
            os.execv(str(mjpython_path), [str(mjpython_path), *sys.argv])

    adapter = MuJoCoPhysicsAdapter()
    ik_solver = DampedLeastSquaresIK(dof=adapter.dof)
    expert = PrivilegedPickPlaceExpert(ik_solver=ik_solver)

    joint_state, ee_pose, obj_pose = adapter.reset(seed=seed)
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

        while viewer.is_running() and ep_count < episodes:
            if is_paused:
                viewer.sync()
                time.sleep(0.02)
                continue

            step_start = time.time()

            q_cur = np.asarray(joint_state.positions_rad, dtype=np.float64)
            jacobian = adapter.get_jacobian()

            action_targets = expert.compute_action(
                q_current=q_cur,
                q_home=adapter.home_configuration,
                ee_pose=ee_pose,
                jacobian=jacobian,
            )

            step_record = adapter.step(action_targets)
            joint_state = step_record.joint_state
            ee_pose = step_record.end_effector_pose

            viewer.sync()

            if expert.stage == "COMPLETED" or step_record.step_index >= 120:
                time.sleep(0.8)
                ep_count += 1
                current_seed += 1
                joint_state, ee_pose, obj_pose = adapter.reset(seed=current_seed)
                expert.reset(initial_object_pose=obj_pose)

            # Match physics simulation time to real time
            dt_step = 0.02
            elapsed = time.time() - step_start
            if dt_step > elapsed:
                time.sleep(dt_step - elapsed)
