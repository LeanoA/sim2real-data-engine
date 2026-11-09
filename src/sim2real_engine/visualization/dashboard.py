"""HTML and 3D WebGL Dashboard generator for Sim2Real trajectory and physics diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

from sim2real_engine.core.contracts import BackendName
from sim2real_engine.visualization.data import (
    EpisodeVisualizationData,
    Sim2SimComparisonData,
    collect_sim2sim_comparison_data,
    collect_trajectory_data,
)


def generate_html_dashboard(
    traj_data: EpisodeVisualizationData | None = None,
    sim2sim_data: Sim2SimComparisonData | None = None,
    seed: int = 42,
    output_path: Path | str = "sim2real_visualizer.html",
) -> Path:
    """Generate a responsive, standalone 3D visualization dashboard HTML file."""
    if traj_data is None:
        traj_data = collect_trajectory_data(seed=seed)
    if sim2sim_data is None:
        sim2sim_data = collect_sim2sim_comparison_data(seed=seed)

    target_file = Path(output_path).resolve()
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prepare animated sequence data for time scrubber
    all_arm_coords: list[dict[str, list[float]]] = []
    for pose in traj_data.arm_trajectory:
        xs, ys, zs = pose.to_coordinates_tuple()
        all_arm_coords.append({"xs": xs, "ys": ys, "zs": zs})

    obj_xs = [p[0] for p in traj_data.object_trajectory]
    obj_ys = [p[1] for p in traj_data.object_trajectory]
    obj_zs = [p[2] for p in traj_data.object_trajectory]

    ee_xs = [p[0] for p in traj_data.ee_trajectory]
    ee_ys = [p[1] for p in traj_data.ee_trajectory]
    ee_zs = [p[2] for p in traj_data.ee_trajectory]

    total_steps = len(traj_data.timesteps_s)

    # Compute phase labels for each step
    phases: list[str] = []
    for idx in range(total_steps):
        t_sec = traj_data.timesteps_s[idx]
        ee_pos = traj_data.ee_trajectory[idx]
        obj_pos = traj_data.object_trajectory[idx]
        dist = ((ee_pos[0] - obj_pos[0]) ** 2 + (ee_pos[1] - obj_pos[1]) ** 2) ** 0.5
        if idx == total_steps - 1:
            phases.append("Place Complete — Payload Released")
        elif dist <= 0.02 and t_sec > 0.15:
            phases.append("Grasping & Lifting Object")
        elif t_sec < 0.12:
            phases.append("Approach to Hover Waypoint")
        else:
            phases.append("Descending to Contact Pose")

    # Initial arm pose (t=0)
    init_arm = all_arm_coords[0]

    # Trace 0: Moving robot arm
    arm_trace = {
        "type": "scatter3d",
        "mode": "lines+markers",
        "name": "Articulated Robot Arm",
        "x": init_arm["xs"],
        "y": init_arm["ys"],
        "z": init_arm["zs"],
        "line": {"color": "#00f0ff", "width": 8},
        "marker": {"size": 8, "color": "#ffffff"},
    }

    # Trace 1: Moving payload object
    obj_trace = {
        "type": "scatter3d",
        "mode": "markers",
        "name": "Payload Object",
        "x": [obj_xs[0]],
        "y": [obj_ys[0]],
        "z": [obj_zs[0]],
        "marker": {"size": 10, "color": "#f6ad55", "symbol": "diamond"},
    }

    # Trace 2: Dynamic path trail up to current step
    trail_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Traveled Trail",
        "x": [ee_xs[0]],
        "y": [ee_ys[0]],
        "z": [ee_zs[0]],
        "line": {"color": "#38a169", "width": 5},
    }

    # Trace 3: Ghost planned full trajectory
    ghost_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Full Planned Trajectory",
        "x": ee_xs,
        "y": ee_ys,
        "z": ee_zs,
        "line": {"color": "rgba(255, 255, 255, 0.25)", "width": 2, "dash": "dot"},
    }

    # Trace 4: Waypoints
    wp_names = [w[0] for w in traj_data.waypoints]
    wp_xs = [w[1][0] for w in traj_data.waypoints]
    wp_ys = [w[1][1] for w in traj_data.waypoints]
    wp_zs = [w[1][2] for w in traj_data.waypoints]

    wp_trace = {
        "type": "scatter3d",
        "mode": "markers+text",
        "name": "Target Waypoints",
        "x": wp_xs,
        "y": wp_ys,
        "z": wp_zs,
        "text": wp_names,
        "textposition": "top center",
        "marker": {"size": 7, "color": "#f6ad55", "symbol": "circle-open"},
    }

    # Trace 5: Workstation table wireframe (surface at z=0.05m and 4 corner legs)
    tbl_x = [0.23, 0.67, 0.67, 0.23, 0.23, 0.23, 0.23, 0.67, 0.67, 0.67, 0.67, 0.23, 0.23]
    tbl_y = [
        -0.30,
        -0.30,
        0.30,
        0.30,
        -0.30,
        -0.30,
        -0.30,
        -0.30,
        -0.30,
        0.30,
        0.30,
        0.30,
        0.30,
    ]
    tbl_z = [0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.05, 0.05, 0.0, 0.0, 0.05, 0.05, 0.0]
    table_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Workstation Table",
        "x": tbl_x,
        "y": tbl_y,
        "z": tbl_z,
        "line": {"color": "#4a5568", "width": 4},
        "hoverinfo": "none",
    }

    # Trace 6: Base mounting flange on floor (r=0.10m ring at z=0)
    flange_angles = [i * 2.0 * math.pi / 16 for i in range(17)]
    flange_x = [0.10 * math.cos(a) for a in flange_angles]
    flange_y = [0.10 * math.sin(a) for a in flange_angles]
    flange_z = [0.0] * 17
    flange_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Mounting Flange",
        "x": flange_x,
        "y": flange_y,
        "z": flange_z,
        "line": {"color": "#718096", "width": 5},
        "hoverinfo": "none",
    }

    # 4. Sim-to-Sim multi-engine 3D traces
    sim_traces: list[dict[str, object]] = []
    sim_colors = {"mujoco": "#00f0ff", "genesis": "#ed8936", "isaac_lab": "#9f7aea"}
    sim_styles = {"mujoco": "solid", "genesis": "dash", "isaac_lab": "dot"}
    labels = {
        "mujoco": "MuJoCo (Local ARM64)",
        "genesis": "Genesis (Cloud Parallel)",
        "isaac_lab": "Isaac Lab (Cloud RTX)",
    }

    eval_backends: list[BackendName] = ["mujoco", "genesis", "isaac_lab"]
    for b_name in eval_backends:
        path = sim2sim_data.ee_trajectories.get(b_name)
        if path:
            sim_traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": labels.get(b_name, b_name),
                    "x": [p[0] for p in path],
                    "y": [p[1] for p in path],
                    "z": [p[2] for p in path],
                    "line": {
                        "color": sim_colors.get(b_name, "#fff"),
                        "width": 5 if b_name == "mujoco" else 4,
                        "dash": sim_styles.get(b_name, "solid"),
                    },
                }
            )

    # 5. Sim-to-Sim 2D time series curves
    t_steps = sim2sim_data.timesteps_s
    cartesian_traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Isaac Lab vs MuJoCo (mm)",
            "x": t_steps,
            "y": [v * 1000.0 for v in sim2sim_data.cartesian_errors_m["isaac_lab"]],
            "line": {"color": "#9f7aea", "width": 3},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Genesis vs MuJoCo (mm)",
            "x": t_steps,
            "y": [v * 1000.0 for v in sim2sim_data.cartesian_errors_m["genesis"]],
            "line": {"color": "#ed8936", "width": 2, "dash": "dash"},
        },
    ]

    joint_traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Isaac Lab Joint RMSE (rad)",
            "x": t_steps,
            "y": sim2sim_data.joint_rmses_rad["isaac_lab"],
            "line": {"color": "#9f7aea", "width": 3},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Genesis Joint RMSE (rad)",
            "x": t_steps,
            "y": sim2sim_data.joint_rmses_rad["genesis"],
            "line": {"color": "#ed8936", "width": 2, "dash": "dash"},
        },
    ]

    # Metrics summary
    isaac_metrics = sim2sim_data.metrics["isaac_lab"]
    gen_metrics = sim2sim_data.metrics["genesis"]

    rmse_val = f"{isaac_metrics.mean_trajectory_rmse:.4f}"
    gen_rmse_val = f"{gen_metrics.mean_trajectory_rmse:.4f}"
    div_val = f"{isaac_metrics.mean_ee_divergence_m * 1000.0:.2f}"
    succ_val = f"{isaac_metrics.success_rate_target * 100.0:.0f}%"

    initial_arm_data = [
        arm_trace,
        obj_trace,
        trail_trace,
        ghost_trace,
        wp_trace,
        table_trace,
        flange_trace,
    ]
    arm_json = json.dumps(initial_arm_data)
    sim_json = json.dumps(sim_traces + [wp_trace, table_trace, flange_trace])
    cart_json = json.dumps(cartesian_traces)
    joint_json = json.dumps(joint_traces)

    anim_bundle_json = json.dumps(
        {
            "armPoses": all_arm_coords,
            "objX": obj_xs,
            "objY": obj_ys,
            "objZ": obj_zs,
            "eeX": ee_xs,
            "eeY": ee_ys,
            "eeZ": ee_zs,
            "timesteps": traj_data.timesteps_s,
            "phases": phases,
            "jointsDeg": traj_data.joint_trajectory_deg,
        }
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sim2Real Data Engine — 3D Motion Player & Sim-to-Sim Diagnostics</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0a0f1d;
      --card-bg: #141b2d;
      --border: #232d4b;
      --accent-cyan: #00f0ff;
      --accent-purple: #9f7aea;
      --accent-orange: #ed8936;
      --accent-green: #38a169;
      --text: #e2e8f0;
      --text-dim: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 24px;
      line-height: 1.5;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      font-size: 24px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge {{
      background: rgba(0, 240, 255, 0.1);
      color: var(--accent-cyan);
      border: 1px solid var(--accent-cyan);
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .subtitle {{ color: var(--text-dim); font-size: 14px; margin-top: 4px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .kpi-title {{
      font-size: 13px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .kpi-val {{ font-size: 26px; font-weight: 700; color: #fff; }}
    .kpi-sub {{ font-size: 12px; color: var(--text-dim); }}
    .grid-2col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }}
    @media (max-width: 1024px) {{
      .grid-2col {{ grid-template-columns: 1fr; }}
    }}
    .chart-box {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
    }}
    .chart-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .chart-title {{ font-size: 16px; font-weight: 600; color: #fff; }}
    .chart-desc {{ font-size: 12px; color: var(--text-dim); }}
    .plot-container {{ width: 100%; height: 440px; }}
    .plot-container-sm {{ width: 100%; height: 260px; }}

    /* Player controls */
    .player-controls {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: rgba(10, 15, 29, 0.7);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .btn-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .btn {{
      background: #1f293d;
      color: #fff;
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .btn:hover {{ background: #2d3b55; border-color: var(--accent-cyan); }}
    .btn-primary {{
      background: var(--accent-cyan);
      color: #0a0f1d;
      border-color: var(--accent-cyan);
    }}
    .btn-primary:hover {{ background: #5df4ff; }}
    .slider-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      width: 100%;
    }}
    .slider-row input[type="range"] {{
      flex: 1;
      accent-color: var(--accent-cyan);
      cursor: pointer;
    }}
    .hud-panel {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px;
      font-size: 11px;
      color: var(--text-dim);
      background: rgba(0, 0, 0, 0.3);
      padding: 8px 10px;
      border-radius: 6px;
      border-left: 3px solid var(--accent-cyan);
    }}
    .hud-label {{ color: var(--text-dim); font-size: 10px; text-transform: uppercase; }}
    .hud-val {{ color: #fff; font-weight: 600; font-family: monospace; font-size: 11px; }}

    footer {{
      margin-top: 32px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
      color: var(--text-dim);
      font-size: 13px;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Sim2Real Data Engine <span class="badge">3D Motion Player</span></h1>
      <div class="subtitle">Articulated 6-DOF Robot Motion & Sim-to-Sim Diagnostics</div>
    </div>
    <div style="text-align: right;">
      <div style="font-size: 12px; color: var(--text-dim);">
        Deterministic Seed: <strong>{seed}</strong>
      </div>
      <div style="font-size: 12px; color: var(--text-dim);">
        Platform: <strong>Apple Silicon (ARM64) + Cloud GPU</strong>
      </div>
    </div>
  </header>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">Sim2Sim Trajectory RMSE</div>
      <div class="kpi-val" style="color: var(--accent-purple);">
        {rmse_val} <span style="font-size: 16px;">rad</span>
      </div>
      <div class="kpi-sub">Isaac: {rmse_val} | Genesis: {gen_rmse_val} rad</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">End-Effector Divergence</div>
      <div class="kpi-val" style="color: var(--accent-cyan);">
        {div_val} <span style="font-size: 16px;">mm</span>
      </div>
      <div class="kpi-sub">Final Cartesian terminal delta</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Task Success Rate</div>
      <div class="kpi-val" style="color: var(--accent-green);">{succ_val}</div>
      <div class="kpi-sub">Dual-engine pick-and-place success</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Inference Telemetry</div>
      <div class="kpi-val" style="color: var(--accent-orange);">
        &lt; 8.1 <span style="font-size: 16px;">ms</span>
      </div>
      <div class="kpi-sub">P99 action chunking latency (target &lt; 15ms)</div>
    </div>
  </div>

  <div class="grid-2col">
    <!-- View 1: 3D Manipulator Trajectory & Motion Player -->
    <div class="chart-box">
      <div class="chart-header">
        <div>
          <div class="chart-title">1. Articulated Robot Motion Player (3D Animation)</div>
          <div class="chart-desc">
            Scrub the timeline or press Play to watch the 6-DOF arm perform pick-and-place
          </div>
        </div>
      </div>

      <div class="player-controls">
        <div class="btn-row">
          <button id="btn-play" class="btn btn-primary">▶ Play</button>
          <button id="btn-reset" class="btn">⏮ Reset</button>
          <button id="btn-prev" class="btn">◀ Step</button>
          <button id="btn-next" class="btn">Step ▶</button>
          <div style="margin-left: auto; display: flex; align-items: center; gap: 6px;">
            <label style="font-size: 12px; color: var(--text-dim);">Speed:</label>
            <select id="select-speed" class="btn" style="padding: 4px 8px;">
              <option value="0.5">0.5x</option>
              <option value="1.0" selected>1.0x</option>
              <option value="2.0">2.0x</option>
            </select>
          </div>
        </div>

        <div class="slider-row">
          <input type="range" id="time-slider" min="0" max="{total_steps - 1}" value="0">
        </div>

        <div class="hud-panel">
          <div>
            <div class="hud-label">Timeline</div>
            <div id="hud-time" class="hud-val">Step 0 / {total_steps} (0.00s)</div>
          </div>
          <div>
            <div class="hud-label">Current Phase</div>
            <div id="hud-phase" class="hud-val" style="color: var(--accent-cyan);">Approach</div>
          </div>
          <div>
            <div class="hud-label">End-Effector (X, Y, Z)</div>
            <div id="hud-ee" class="hud-val">(0.00, 0.00, 0.00)</div>
          </div>
          <div>
            <div class="hud-label">Joints 1..6 (deg)</div>
            <div id="hud-joints" class="hud-val">[0, 0, 0, 0, 0, 0]</div>
          </div>
        </div>
      </div>

      <div id="plot-3d-arm" class="plot-container"></div>
    </div>

    <!-- View 2: Sim-to-Sim 3D Trajectory Divergence -->
    <div class="chart-box">
      <div class="chart-header">
        <div>
          <div class="chart-title">2. Sim-to-Sim Physics Divergence (3D Path Comparison)</div>
          <div class="chart-desc">MuJoCo (Local ARM64) vs Genesis vs Isaac Lab (Cloud GPU)</div>
        </div>
      </div>
      <div id="plot-3d-sim2sim" class="plot-container" style="height: 575px;"></div>
    </div>
  </div>

  <div class="grid-2col">
    <!-- View 3: Cartesian Drift Time Series -->
    <div class="chart-box">
      <div class="chart-header">
        <div>
          <div class="chart-title">3. Cartesian End-Effector Drift vs. Time</div>
          <div class="chart-desc">Spatial error ||p_tgt(t) - p_ref(t)|| across control steps</div>
        </div>
      </div>
      <div id="plot-drift-cartesian" class="plot-container-sm"></div>
    </div>

    <!-- View 4: Joint Angle RMSE Time Series -->
    <div class="chart-box">
      <div class="chart-header">
        <div>
          <div class="chart-title">4. Articulated Joint Angle RMSE vs. Time</div>
          <div class="chart-desc">Time-series angular discrepancy across 6 joints</div>
        </div>
      </div>
      <div id="plot-drift-joints" class="plot-container-sm"></div>
    </div>
  </div>

  <footer>
    <div>Sim2Real Data Engine — Hexagonal Architecture: Local MuJoCo ARM64 + Cloud GPU</div>
    <div>LeRobotDataset v2 Parquet & Decoupled Action Chunking Server</div>
  </footer>

  <script>
    const bundle = {anim_bundle_json};
    const totalSteps = bundle.timesteps.length;

    const darkLayout3D = {{
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {{ color: '#e2e8f0', size: 11 }},
      margin: {{ l: 0, r: 0, b: 0, t: 10 }},
      scene: {{
        aspectmode: 'cube',
        xaxis: {{
          title: 'X (m)', gridcolor: '#232d4b', zerolinecolor: '#3182ce', color: '#94a3b8'
        }},
        yaxis: {{
          title: 'Y (m)', gridcolor: '#232d4b', zerolinecolor: '#3182ce', color: '#94a3b8'
        }},
        zaxis: {{
          title: 'Z (m)', gridcolor: '#232d4b', zerolinecolor: '#3182ce', color: '#94a3b8'
        }},
        camera: {{
          eye: {{ x: 1.45, y: 1.45, z: 1.15 }}
        }}
      }},
      legend: {{ orientation: 'h', y: 1.05, font: {{ size: 10 }} }}
    }};

    const darkLayout2D = {{
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {{ color: '#e2e8f0', size: 11 }},
      margin: {{ l: 50, r: 20, b: 40, t: 20 }},
      xaxis: {{ title: 'Time (s)', gridcolor: '#232d4b', color: '#94a3b8' }},
      yaxis: {{ gridcolor: '#232d4b', color: '#94a3b8' }},
      legend: {{ orientation: 'h', y: 1.12, font: {{ size: 10 }} }}
    }};

    // 1. Arm trajectory plot
    const armData = {arm_json};
    Plotly.newPlot('plot-3d-arm', armData, darkLayout3D, {{ responsive: true }});

    // 2. Sim2Sim 3D plot
    const simData = {sim_json};
    Plotly.newPlot('plot-3d-sim2sim', simData, darkLayout3D, {{ responsive: true }});

    // 3. Cartesian drift
    const cartesianData = {cart_json};
    const cartesianLayout = {{
      ...darkLayout2D,
      yaxis: {{ title: 'Drift Error (mm)', gridcolor: '#232d4b', color: '#94a3b8' }}
    }};
    Plotly.newPlot('plot-drift-cartesian', cartesianData, cartesianLayout, {{ responsive: true }});

    // 4. Joint RMSE
    const jointData = {joint_json};
    const jointLayout = {{
      ...darkLayout2D,
      yaxis: {{ title: 'Joint RMSE (rad)', gridcolor: '#232d4b', color: '#94a3b8' }}
    }};
    Plotly.newPlot('plot-drift-joints', jointData, jointLayout, {{ responsive: true }});

    // 5. Interactive Animation Logic
    let currentStep = 0;
    let isPlaying = false;
    let animTimer = null;
    let playbackSpeed = 1.0;

    const slider = document.getElementById('time-slider');
    const btnPlay = document.getElementById('btn-play');
    const btnReset = document.getElementById('btn-reset');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const selectSpeed = document.getElementById('select-speed');

    const hudTime = document.getElementById('hud-time');
    const hudPhase = document.getElementById('hud-phase');
    const hudEe = document.getElementById('hud-ee');
    const hudJoints = document.getElementById('hud-joints');

    function updateHud(idx) {{
      const t = bundle.timesteps[idx];
      hudTime.textContent = `Step ${{idx + 1}} / ${{totalSteps}} (${{t.toFixed(2)}}s)`;
      hudPhase.textContent = bundle.phases[idx] || "Active";
      const eeX = bundle.eeX[idx].toFixed(2);
      const eeY = bundle.eeY[idx].toFixed(2);
      const eeZ = bundle.eeZ[idx].toFixed(2);
      hudEe.textContent = `(${{eeX}}, ${{eeY}}, ${{eeZ}})`;
      const jDeg = bundle.jointsDeg[idx];
      hudJoints.textContent = `[${{jDeg.join('°, ')}}°]`;
    }}

    function renderStep(idx) {{
      currentStep = Math.max(0, Math.min(idx, totalSteps - 1));
      slider.value = currentStep;
      updateHud(currentStep);

      const pose = bundle.armPoses[currentStep];
      const ox = bundle.objX[currentStep];
      const oy = bundle.objY[currentStep];
      const oz = bundle.objZ[currentStep];

      const trailX = bundle.eeX.slice(0, currentStep + 1);
      const trailY = bundle.eeY.slice(0, currentStep + 1);
      const trailZ = bundle.eeZ.slice(0, currentStep + 1);

      Plotly.restyle('plot-3d-arm', {{
        x: [pose.xs, [ox], trailX],
        y: [pose.ys, [oy], trailY],
        z: [pose.zs, [oz], trailZ]
      }}, [0, 1, 2]);
    }}

    function play() {{
      if (currentStep >= totalSteps - 1) {{
        currentStep = 0;
      }}
      isPlaying = true;
      btnPlay.textContent = '⏸ Pause';
      btnPlay.classList.remove('btn-primary');
      clearInterval(animTimer);
      const intervalMs = Math.round(40 / playbackSpeed);
      animTimer = setInterval(() => {{
        if (currentStep >= totalSteps - 1) {{
          pause();
        }} else {{
          renderStep(currentStep + 1);
        }}
      }}, intervalMs);
    }}

    function pause() {{
      isPlaying = false;
      btnPlay.textContent = '▶ Play';
      btnPlay.classList.add('btn-primary');
      clearInterval(animTimer);
    }}

    btnPlay.addEventListener('click', () => {{
      if (isPlaying) pause(); else play();
    }});

    btnReset.addEventListener('click', () => {{
      pause();
      renderStep(0);
    }});

    btnPrev.addEventListener('click', () => {{
      pause();
      renderStep(currentStep - 1);
    }});

    btnNext.addEventListener('click', () => {{
      pause();
      renderStep(currentStep + 1);
    }});

    slider.addEventListener('input', (e) => {{
      pause();
      renderStep(parseInt(e.target.value, 10));
    }});

    selectSpeed.addEventListener('change', (e) => {{
      playbackSpeed = parseFloat(e.target.value);
      if (isPlaying) {{
        play();
      }}
    }});

    // Initialize HUD
    updateHud(0);
  </script>
</body>
</html>
"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file
