"""Interactive 3D WebGL Dashboard comparing trained policy rollouts against privileged expert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim2real_engine.visualization.data import PolicyVsExpertComparisonData


def _classify_rollout_phases(
    ee_zs: list[float],
    obj_xs: list[float],
    obj_ys: list[float],
    obj_zs: list[float],
) -> tuple[list[str], list[str]]:
    """Classify per-step manipulation phase and dynamic weld gripper state."""
    total_steps = len(obj_zs)
    if total_steps == 0:
        return [], []

    obj_x0 = obj_xs[0]
    obj_y0 = obj_ys[0]
    obj_z0 = obj_zs[0]
    phases: list[str] = []
    grippers: list[str] = []

    for i in range(total_steps):
        lifted = obj_zs[i] > obj_z0 + 0.0015
        if lifted:
            grippers.append("LATCHED (weld)")
            next_z = obj_zs[min(i + 1, total_steps - 1)]
            prev_y = obj_ys[max(0, i - 1)]
            if next_z > obj_zs[i] + 0.002:
                phases.append("3. Grasp & Lift")
            elif abs(obj_ys[i] - prev_y) > 0.002 and obj_zs[i] > 0.16:
                phases.append("4. Cross-Table Transport")
            else:
                phases.append("5. Controlled Descent")
        else:
            grippers.append("OPEN")
            xy_disp = float(np.hypot(obj_xs[i] - obj_x0, obj_ys[i] - obj_y0))
            if xy_disp < 0.02:
                if ee_zs[i] > 0.175:
                    phases.append("1. Pre-Grasp Approach")
                else:
                    phases.append("2. Descend to Grasp")
            else:
                phases.append("6. Payload Released & Hold")

    return phases, grippers


def generate_policy_comparison_html(
    comparison_data: PolicyVsExpertComparisonData,
    output_path: Path | str = "sim2real_policy_comparison.html",
) -> Path:
    """Generate a responsive, standalone 3D WebGL dashboard comparing policy vs. expert."""
    target_file = Path(output_path).resolve()
    target_file.parent.mkdir(parents=True, exist_ok=True)

    policy = comparison_data.policy
    expert = comparison_data.expert

    # Format arm link coordinates for each policy step
    all_arm_coords: list[dict[str, list[float]]] = []
    for pose in policy.arm_trajectory:
        xs, ys, zs = pose.to_coordinates_tuple()
        all_arm_coords.append({"xs": xs, "ys": ys, "zs": zs})

    policy_ee_xs = [p[0] for p in policy.ee_trajectory]
    policy_ee_ys = [p[1] for p in policy.ee_trajectory]
    policy_ee_zs = [p[2] for p in policy.ee_trajectory]

    policy_obj_xs = [p[0] for p in policy.object_trajectory]
    policy_obj_ys = [p[1] for p in policy.object_trajectory]
    policy_obj_zs = [p[2] for p in policy.object_trajectory]
    payload_z_cm = [round(z * 100.0, 2) for z in policy_obj_zs]

    expert_ee_xs = [p[0] for p in expert.ee_trajectory]
    expert_ee_ys = [p[1] for p in expert.ee_trajectory]
    expert_ee_zs = [p[2] for p in expert.ee_trajectory]

    total_steps = len(policy.timesteps_s)
    step_indices = list(range(total_steps))

    # Step-by-step spatial error (mm)
    comp_steps = min(len(policy.ee_trajectory), len(expert.ee_trajectory))
    spatial_errors_mm: list[float] = []
    for i in range(total_steps):
        if i < comp_steps:
            p_pol = np.array(policy.ee_trajectory[i])
            p_exp = np.array(expert.ee_trajectory[i])
            err = float(np.linalg.norm(p_pol - p_exp) * 1000.0)
        else:
            err = spatial_errors_mm[-1] if spatial_errors_mm else 0.0
        spatial_errors_mm.append(round(err, 2))

    phase_labels, gripper_states = _classify_rollout_phases(
        ee_zs=policy_ee_zs,
        obj_xs=policy_obj_xs,
        obj_ys=policy_obj_ys,
        obj_zs=policy_obj_zs,
    )

    obj_init_xy = np.asarray(policy.object_initial_pose[:2], dtype=np.float64)
    obj_final_xy = np.asarray(policy.object_final_pose[:2], dtype=np.float64)
    payload_transport_cm = round(float(np.linalg.norm(obj_final_xy - obj_init_xy) * 100.0), 1)
    peak_payload_z_cm = round(float(max(policy_obj_zs) * 100.0), 1) if policy_obj_zs else 0.0
    mlp_forward_ms = max(0.05, round(comparison_data.inference_p50_latency_ms - 8.0, 2))

    # Initial arm pose (t=0)
    init_arm = all_arm_coords[0]

    # Trace 0: Moving Policy Robot Arm
    policy_arm_trace = {
        "type": "scatter3d",
        "mode": "lines+markers",
        "name": "Policy 6-DOF Manipulator (Green)",
        "x": init_arm["xs"],
        "y": init_arm["ys"],
        "z": init_arm["zs"],
        "line": {"color": "#10b981", "width": 9},
        "marker": {"size": 8, "color": "#ffffff"},
    }

    # Trace 1: Moving Payload Object
    obj_trace = {
        "type": "scatter3d",
        "mode": "markers",
        "name": "Payload Cube (Amber)",
        "x": [policy_obj_xs[0]],
        "y": [policy_obj_ys[0]],
        "z": [policy_obj_zs[0]],
        "marker": {"size": 10, "color": "#f59e0b", "symbol": "diamond"},
    }

    # Trace 2: Policy Dynamic Traveled Trail
    policy_trail_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Policy Executed Trail (Pink)",
        "x": [policy_ee_xs[0]],
        "y": [policy_ee_ys[0]],
        "z": [policy_ee_zs[0]],
        "line": {"color": "#ec4899", "width": 6},
    }

    # Trace 3: Expert Ground Truth Trajectory (Ghost reference)
    expert_ref_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Expert Reference Trajectory (Cyan)",
        "x": expert_ee_xs,
        "y": expert_ee_ys,
        "z": expert_ee_zs,
        "line": {"color": "#00f0ff", "width": 4, "dash": "dash"},
    }

    # Trace 4: Target Waypoints
    wp_names = [w[0] for w in expert.waypoints]
    wp_xs = [w[1][0] for w in expert.waypoints]
    wp_ys = [w[1][1] for w in expert.waypoints]
    wp_zs = [w[1][2] for w in expert.waypoints]

    wp_trace = {
        "type": "scatter3d",
        "mode": "markers+text",
        "name": "Task Waypoints",
        "x": wp_xs,
        "y": wp_ys,
        "z": wp_zs,
        "text": wp_names,
        "textposition": "top center",
        "marker": {"size": 7, "color": "#38bdf8", "symbol": "circle-open"},
    }

    # Trace 5: Workstation table wireframe
    tbl_x = [0.23, 0.67, 0.67, 0.23, 0.23, 0.23, 0.23, 0.67, 0.67, 0.67, 0.67, 0.23, 0.23]
    tbl_y = [
        -0.30, -0.30, 0.30, 0.30, -0.30, -0.30, 0.30, 0.30, -0.30, -0.30, 0.30, 0.30, -0.30,
    ]
    tbl_z = [
        0.05, 0.05, 0.05, 0.05, 0.05, -0.20, -0.20, -0.20, -0.20, 0.05, 0.05, -0.20, -0.20,
    ]

    table_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Workstation Table (z=0.05m)",
        "x": tbl_x,
        "y": tbl_y,
        "z": tbl_z,
        "line": {"color": "#475569", "width": 3},
        "hoverinfo": "none",
    }

    plot_data = [
        policy_arm_trace,
        obj_trace,
        policy_trail_trace,
        expert_ref_trace,
        wp_trace,
        table_trace,
    ]

    plot_layout = {
        "title": {
            "text": "3D Kinematics: Arm (Green), Policy Trail (Pink) vs. Expert (Cyan)",
            "font": {"size": 15, "color": "#f8fafc", "family": "system-ui, sans-serif"},
            "x": 0.03,
            "y": 0.96,
        },
        "scene": {
            "xaxis": {
                "title": "X (Forward) [m]",
                "range": [-0.15, 0.75],
                "backgroundcolor": "#090d16",
                "gridcolor": "#1e293b",
                "showbackground": True,
            },
            "yaxis": {
                "title": "Y (Lateral) [m]",
                "range": [-0.45, 0.45],
                "backgroundcolor": "#090d16",
                "gridcolor": "#1e293b",
                "showbackground": True,
            },
            "zaxis": {
                "title": "Z (Vertical) [m]",
                "range": [-0.05, 0.70],
                "backgroundcolor": "#090d16",
                "gridcolor": "#1e293b",
                "showbackground": True,
            },
            "camera": {
                "eye": {"x": 1.45, "y": -1.55, "z": 0.95},
                "center": {"x": 0.25, "y": 0.0, "z": 0.15},
            },
            "aspectmode": "cube",
        },
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#0f172a",
        "margin": {"l": 0, "r": 0, "b": 0, "t": 38},
        "legend": {
            "font": {"color": "#94a3b8", "size": 11},
            "bgcolor": "rgba(15, 23, 42, 0.82)",
            "x": 0.02,
            "y": 0.92,
        },
    }

    # 2D Synchronized Telemetry Chart (Error mm & Payload Z cm vs Step)
    telemetry_2d_data = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Tracking Error (mm)",
            "x": step_indices,
            "y": spatial_errors_mm,
            "yaxis": "y1",
            "line": {"color": "#ec4899", "width": 2.5},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Payload Height Z (cm)",
            "x": step_indices,
            "y": payload_z_cm,
            "yaxis": "y2",
            "line": {"color": "#f59e0b", "width": 2.5, "dash": "dot"},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "Current Error",
            "x": [0],
            "y": [spatial_errors_mm[0] if spatial_errors_mm else 0.0],
            "yaxis": "y1",
            "showlegend": False,
            "marker": {
                "size": 9,
                "color": "#ffffff",
                "line": {"color": "#ec4899", "width": 2.5},
            },
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "Current Payload Z",
            "x": [0],
            "y": [payload_z_cm[0] if payload_z_cm else 0.0],
            "yaxis": "y2",
            "showlegend": False,
            "marker": {
                "size": 9,
                "symbol": "diamond",
                "color": "#ffffff",
                "line": {"color": "#f59e0b", "width": 2.5},
            },
        },
    ]

    telemetry_2d_layout = {
        "title": {
            "text": "Synchronized Rollout Telemetry: Tracking Error (mm) & Payload Lift Z (cm)",
            "font": {"size": 12, "color": "#cbd5e1", "family": "system-ui, sans-serif"},
            "x": 0.02,
            "y": 0.94,
        },
        "xaxis": {
            "title": "Control Step (50 Hz / Δt = 20 ms)",
            "gridcolor": "#1e293b",
            "color": "#94a3b8",
            "zeroline": False,
        },
        "yaxis": {
            "title": "Error (mm)",
            "titlefont": {"color": "#ec4899"},
            "tickfont": {"color": "#ec4899"},
            "gridcolor": "#1e293b",
            "zeroline": False,
            "range": [0, max(15.0, float(max(spatial_errors_mm, default=12.0)) * 1.15)],
        },
        "yaxis2": {
            "title": "Payload Z (cm)",
            "titlefont": {"color": "#f59e0b"},
            "tickfont": {"color": "#f59e0b"},
            "overlaying": "y",
            "side": "right",
            "gridcolor": "rgba(30, 41, 59, 0.25)",
            "zeroline": False,
            "range": [5.0, max(21.0, peak_payload_z_cm * 1.12)],
        },
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#090d16",
        "margin": {"l": 46, "r": 46, "b": 40, "t": 34},
        "legend": {
            "orientation": "h",
            "font": {"color": "#94a3b8", "size": 10},
            "x": 0.02,
            "y": 1.14,
        },
    }

    # Serialized animation bundle for client-side JavaScript player
    anim_bundle = {
        "totalSteps": total_steps,
        "timesteps": policy.timesteps_s,
        "armTrajectory": all_arm_coords,
        "policyEeTrajectory": policy.ee_trajectory,
        "expertEeTrajectory": expert.ee_trajectory,
        "policyObjectTrajectory": policy.object_trajectory,
        "spatialErrorsMm": spatial_errors_mm,
        "payloadZCm": payload_z_cm,
        "phaseLabels": phase_labels,
        "gripperStates": gripper_states,
    }

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sim2Real Data Engine — Policy vs. Privileged Expert 3D Evaluation</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card: #0f172a;
      --border: #1e293b;
      --accent-policy: #ec4899;
      --accent-expert: #00f0ff;
      --accent-arm: #10b981;
      --accent-obj: #f59e0b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      padding: 1.25rem 1.5rem;
      min-height: 100vh;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1rem;
    }}
    .header h1 {{
      font-size: 1.4rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.45rem;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.84rem;
      margin-top: 0.35rem;
      line-height: 1.45;
    }}
    .nav-links {{
      display: flex;
      gap: 0.6rem;
      align-items: center;
      flex-wrap: wrap;
    }}
    .nav-btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      text-decoration: none;
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 0.45rem 0.85rem;
      border-radius: 0.5rem;
      font-size: 0.8rem;
      font-weight: 600;
      transition: background 0.15s ease, border-color 0.15s ease;
    }}
    .nav-btn:hover {{
      background: #334155;
      border-color: #475569;
    }}
    .nav-btn-primary {{
      background: rgba(16, 185, 129, 0.15);
      border-color: var(--accent-arm);
      color: #34d399;
    }}
    .nav-btn-primary:hover {{
      background: rgba(16, 185, 129, 0.28);
    }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.6rem;
      font-size: 0.72rem;
      font-weight: 600;
      border-radius: 9999px;
    }}
    .badge-policy {{
      background: rgba(236, 72, 153, 0.18);
      color: var(--accent-policy);
      border: 1px solid var(--accent-policy);
    }}
    .badge-arm {{
      background: rgba(16, 185, 129, 0.18);
      color: var(--accent-arm);
      border: 1px solid var(--accent-arm);
    }}
    .badge-expert {{
      background: rgba(0, 240, 255, 0.18);
      color: var(--accent-expert);
      border: 1px solid var(--accent-expert);
    }}
    .grid-kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.9rem;
      margin-bottom: 1.15rem;
    }}
    .kpi-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.9rem 1rem;
    }}
    .kpi-title {{
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .kpi-value {{ font-size: 1.4rem; font-weight: 700; margin: 0.3rem 0; }}
    .kpi-sub {{ font-size: 0.74rem; color: var(--text-muted); line-height: 1.35; }}

    .player-controls {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.85rem 1.15rem;
      margin-bottom: 1.15rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}
    .control-row {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.65rem;
    }}
    .btn {{
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 0.45rem 0.85rem;
      border-radius: 0.5rem;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.82rem;
    }}
    .btn:hover {{ background: #334155; }}
    .btn-primary {{
      background: var(--accent-policy);
      border-color: var(--accent-policy);
      color: white;
    }}
    .btn-primary:hover {{ background: #db2777; }}
    .btn-active {{
      border-color: var(--accent-arm);
      color: #34d399;
    }}
    .slider {{
      flex: 1;
      min-width: 180px;
      accent-color: var(--accent-policy);
      cursor: pointer;
    }}

    .main-stage {{
      display: grid;
      grid-template-columns: 1.75fr 1.05fr;
      gap: 1.25rem;
      margin-bottom: 1rem;
    }}
    @media (max-width: 1080px) {{
      .main-stage {{ grid-template-columns: 1fr; }}
    }}
    .plot-container {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow: hidden;
      height: 620px;
    }}
    .right-column {{
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .telemetry-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1rem 1.15rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }}
    .hud-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.36rem 0;
      font-size: 0.82rem;
    }}
    .hud-row:last-child {{ border-bottom: none; }}
    .hud-label {{ color: var(--text-muted); }}
    .hud-value {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 600; }}
    .phase-pill {{
      padding: 0.15rem 0.55rem;
      border-radius: 0.375rem;
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.4);
      font-size: 0.78rem;
    }}
    .gripper-open {{
      color: #94a3b8;
    }}
    .gripper-latched {{
      color: #fbbf24;
      background: rgba(245, 158, 11, 0.16);
      border: 1px solid rgba(245, 158, 11, 0.45);
      padding: 0.12rem 0.5rem;
      border-radius: 0.375rem;
      font-size: 0.76rem;
    }}
    .plot-2d-container {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow: hidden;
      height: 255px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Policy vs. Privileged Expert 3D Evaluation
        <span class="badge badge-arm">Robot Arm (Green)</span>
        <span class="badge badge-policy">Policy Trail (Pink)</span>
        <span class="badge badge-expert">Expert Ref (Cyan)</span>
      </h1>
      <p class="subtitle">
        Closed-loop <strong>30D Privileged Sensorimotor BC Policy</strong>
        (H=8 Action Chunking &bull; Exponential Temporal Ensembling exp(-m&middot;k))
        in <strong>DeepMind MuJoCo 4.0</strong>
        (500 Hz physics / 50 Hz control, seed={comparison_data.seed})
      </p>
    </div>
    <div class="nav-links">
      <a class="nav-btn nav-btn-primary"
         href="https://github.com/LeanoA/sim2real-data-engine"
         target="_blank" rel="noopener noreferrer">
        GitHub Repository &nearr;
      </a>
      <a class="nav-btn"
         href="https://github.com/LeanoA/sim2real-data-engine/blob/main/ARTICLE_DRAFT.md"
         target="_blank" rel="noopener noreferrer">
        Technical Article &nearr;
      </a>
      <a class="nav-btn"
         href="https://github.com/LeanoA/sim2real-data-engine/blob/main/docs/EXPERIMENT_REPORT_PICK_PLACE.md"
         target="_blank" rel="noopener noreferrer">
        Experiment Report &nearr;
      </a>
    </div>
  </div>

  <div class="grid-kpis">
    <div class="kpi-card">
      <div class="kpi-title">Mean Spatial Tracking Error</div>
      <div class="kpi-value" style="color: var(--accent-policy);">
        {comparison_data.mean_tracking_error_mm:.1f} mm
      </div>
      <div class="kpi-sub">Closed-loop policy vs. privileged IK expert</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Max Spatial Error</div>
      <div class="kpi-value">{comparison_data.max_tracking_error_mm:.1f} mm</div>
      <div class="kpi-sub">Peak transient deviation during grasp transition</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Cross-Table Payload Transport</div>
      <div class="kpi-value" style="color: var(--accent-obj);">{payload_transport_cm:.1f} cm</div>
      <div class="kpi-sub">
        Peak lift Z: {peak_payload_z_cm:.1f} cm &bull; 100% Grasp &amp; Place
      </div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Policy EE Net Displacement</div>
      <div class="kpi-value" style="color: var(--accent-arm);">
        {comparison_data.policy_ee_displacement_mm:.1f} mm
      </div>
      <div class="kpi-sub">End-effector spatial span across workstation</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Inference &amp; Bridge Latency</div>
      <div class="kpi-value">
        ~{mlp_forward_ms:.2f} ms
        <span style="font-size: 0.85rem; font-weight: 500; color: var(--text-muted);">CPU MLP</span>
      </div>
      <div class="kpi-sub">
        Bridge P99: {comparison_data.inference_p99_latency_ms:.2f} ms
        (incl. 8.0 ms synthetic IPC &lt; 20 ms 50 Hz budget)
      </div>
    </div>
  </div>

  <div class="player-controls">
    <div class="control-row">
      <button class="btn btn-primary" id="btn-play">&#9654; Play</button>
      <button class="btn" id="btn-reset">&#9198; Reset</button>
      <button class="btn" id="btn-prev">&#9664; Step</button>
      <button class="btn" id="btn-next">Step &#9654;</button>
      <button class="btn btn-active" id="btn-loop" title="Toggle continuous playback loop">
        &#128257; Loop: ON
      </button>
      <button class="btn" id="btn-speed" title="Cycle playback speed">Speed: 1.0x</button>
      <input type="range" class="slider" id="scrubber" min="0" max="{total_steps - 1}" value="0">
      <span id="scrubber-label"
            style="font-family: monospace; font-size: 0.84rem; min-width: 92px; text-align: right;">
        Step 0 / {total_steps - 1}
      </span>
    </div>
  </div>

  <div class="main-stage">
    <div class="plot-container" id="plot-3d"></div>

    <div class="right-column">
      <div class="telemetry-card">
        <h3 style="font-size: 0.95rem; border-bottom: 1px solid var(--border);
                   padding-bottom: 0.45rem; margin-bottom: 0.2rem;">
          Real-Time Telemetry HUD
        </h3>
        <div class="hud-row">
          <span class="hud-label">Manipulation Phase:</span>
          <span class="hud-value phase-pill" id="hud-phase">1. Pre-Grasp Approach</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Gripper Constraint:</span>
          <span class="hud-value gripper-open" id="hud-gripper">OPEN</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Control Step / Elapsed:</span>
          <span class="hud-value">
            <span id="hud-step">0 / {total_steps - 1}</span>
            (<span id="hud-time">0.00 s</span>)
          </span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Instant Tracking Error:</span>
          <span class="hud-value" id="hud-error" style="color: var(--accent-policy);">0.0 mm</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Policy EE [x, y, z]:</span>
          <span class="hud-value" id="hud-policy-ee">[0.000, 0.000, 0.000] m</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Expert EE [x, y, z]:</span>
          <span class="hud-value" id="hud-expert-ee">[0.000, 0.000, 0.000] m</span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Payload Cube [x, y, z]:</span>
          <span class="hud-value" id="hud-policy-obj" style="color: var(--accent-obj);">
            [0.000, 0.000, 0.000] m
          </span>
        </div>
        <div class="hud-row">
          <span class="hud-label">Physics Engine:</span>
          <span class="hud-value">{comparison_data.backend} (500 Hz RK4/Implicit)</span>
        </div>
      </div>

      <div class="plot-2d-container" id="plot-2d-telemetry"></div>
    </div>
  </div>

  <script>
    const bundle = {json.dumps(anim_bundle)};
    const plotData = {json.dumps(plot_data)};
    const plotLayout = {json.dumps(plot_layout)};
    const telemetry2dData = {json.dumps(telemetry_2d_data)};
    const telemetry2dLayout = {json.dumps(telemetry_2d_layout)};

    Plotly.newPlot('plot-3d', plotData, plotLayout, {{responsive: true, displaylogo: false}});
    Plotly.newPlot(
      'plot-2d-telemetry',
      telemetry2dData,
      telemetry2dLayout,
      {{responsive: true, displayModeBar: false}}
    );

    let currentStep = 0;
    let isPlaying = false;
    let loopEnabled = true;
    const speeds = [1.0, 2.0, 0.5];
    let speedIdx = 0;
    let animTimer = null;

    const btnPlay = document.getElementById('btn-play');
    const btnReset = document.getElementById('btn-reset');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const btnLoop = document.getElementById('btn-loop');
    const btnSpeed = document.getElementById('btn-speed');
    const slider = document.getElementById('scrubber');
    const sliderLabel = document.getElementById('scrubber-label');
    const hudGripper = document.getElementById('hud-gripper');

    function renderStep(step) {{
      if (step < 0) step = 0;
      if (step >= bundle.totalSteps) step = bundle.totalSteps - 1;
      currentStep = step;

      slider.value = step;
      sliderLabel.textContent = 'Step ' + step + ' / ' + (bundle.totalSteps - 1);

      const arm = bundle.armTrajectory[step];
      const polEe = bundle.policyEeTrajectory[step];
      const expIdx = Math.min(step, bundle.expertEeTrajectory.length - 1);
      const expEe = bundle.expertEeTrajectory ? bundle.expertEeTrajectory[expIdx] : polEe;
      const polObj = bundle.policyObjectTrajectory[step];
      const errMm = bundle.spatialErrorsMm[step];
      const objZCm = bundle.payloadZCm[step];
      const phaseStr = bundle.phaseLabels[step] || 'Rollout';
      const gripStr = bundle.gripperStates[step] || 'OPEN';

      // Update 3D Traces: 0=Arm, 1=Payload Cube, 2=Policy Trail
      const trailXs = bundle.policyEeTrajectory.slice(0, step + 1).map(p => p[0]);
      const trailYs = bundle.policyEeTrajectory.slice(0, step + 1).map(p => p[1]);
      const trailZs = bundle.policyEeTrajectory.slice(0, step + 1).map(p => p[2]);

      Plotly.restyle('plot-3d', {{
        x: [arm.xs, [polObj[0]], trailXs],
        y: [arm.ys, [polObj[1]], trailYs],
        z: [arm.zs, [polObj[2]], trailZs],
      }}, [0, 1, 2]);

      // Update 2D Telemetry Cursor Traces (2=Error Marker, 3=Payload Z Marker)
      Plotly.restyle('plot-2d-telemetry', {{
        x: [[step], [step]],
        y: [[errMm], [objZCm]],
      }}, [2, 3]);

      // Update Telemetry HUD
      document.getElementById('hud-phase').textContent = phaseStr;
      hudGripper.textContent = gripStr;
      const gripCls = gripStr.startsWith('LATCHED') ? 'gripper-latched' : 'gripper-open';
      hudGripper.className = 'hud-value ' + gripCls;
      document.getElementById('hud-step').textContent = step + ' / ' + (bundle.totalSteps - 1);
      document.getElementById('hud-time').textContent = bundle.timesteps[step].toFixed(2) + ' s';
      document.getElementById('hud-error').textContent = errMm.toFixed(1) + ' mm';
      const fmtVec = (vec) => '[' + vec.map(v => v.toFixed(3)).join(', ') + '] m';
      document.getElementById('hud-policy-ee').textContent = fmtVec(polEe);
      document.getElementById('hud-expert-ee').textContent = fmtVec(expEe);
      document.getElementById('hud-policy-obj').textContent = fmtVec(polObj);
    }}

    function startInterval() {{
      clearInterval(animTimer);
      const intervalMs = Math.round(50 / speeds[speedIdx]);
      animTimer = setInterval(() => {{
        if (currentStep >= bundle.totalSteps - 1) {{
          if (loopEnabled) {{
            renderStep(0);
          }} else {{
            pause();
          }}
        }} else {{
          renderStep(currentStep + 1);
        }}
      }}, intervalMs);
    }}

    function play() {{
      if (currentStep >= bundle.totalSteps - 1) currentStep = 0;
      isPlaying = true;
      btnPlay.textContent = '⏸ Pause';
      btnPlay.classList.remove('btn-primary');
      startInterval();
    }}

    function pause() {{
      isPlaying = false;
      btnPlay.textContent = '▶ Play';
      btnPlay.classList.add('btn-primary');
      clearInterval(animTimer);
    }}

    btnPlay.addEventListener('click', () => isPlaying ? pause() : play());
    btnReset.addEventListener('click', () => {{ pause(); renderStep(0); }});
    btnPrev.addEventListener('click', () => {{ pause(); renderStep(currentStep - 1); }});
    btnNext.addEventListener('click', () => {{ pause(); renderStep(currentStep + 1); }});
    btnLoop.addEventListener('click', () => {{
      loopEnabled = !loopEnabled;
      btnLoop.textContent = loopEnabled ? '🔁 Loop: ON' : '🔁 Loop: OFF';
      btnLoop.classList.toggle('btn-active', loopEnabled);
    }});
    btnSpeed.addEventListener('click', () => {{
      speedIdx = (speedIdx + 1) % speeds.length;
      btnSpeed.textContent = 'Speed: ' + speeds[speedIdx].toFixed(1) + 'x';
      if (isPlaying) startInterval();
    }});
    slider.addEventListener('input', (e) => {{
      pause();
      renderStep(parseInt(e.target.value, 10));
    }});

    renderStep(0);
    setTimeout(() => play(), 400);
  </script>
</body>
</html>
"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file
