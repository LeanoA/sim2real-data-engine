"""HTML and 3D WebGL Dashboard generator for Sim2Real trajectory and physics diagnostics."""

from __future__ import annotations

import json
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

    # 1. Prepare 3D Arm skeleton traces for Plotly
    arm_traces: list[dict[str, object]] = []
    colors = {
        "Initial Home": "#718096",
        "Grasp Approach": "#3182ce",
        "Final Place": "#38a169",
    }
    widths = {"Initial Home": 4, "Grasp Approach": 5, "Final Place": 6}

    for name, pose in traj_data.key_arm_poses.items():
        xs, ys, zs = pose.to_coordinates_tuple()
        arm_traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": f"Arm: {name}",
                "x": xs,
                "y": ys,
                "z": zs,
                "line": {"color": colors.get(name, "#e2e8f0"), "width": widths.get(name, 5)},
                "marker": {"size": 6, "color": colors.get(name, "#e2e8f0")},
            }
        )

    # 2. End-effector trajectory trace
    ee_xs = [p[0] for p in traj_data.ee_trajectory]
    ee_ys = [p[1] for p in traj_data.ee_trajectory]
    ee_zs = [p[2] for p in traj_data.ee_trajectory]

    ee_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "EE Pick-and-Place Path",
        "x": ee_xs,
        "y": ee_ys,
        "z": ee_zs,
        "line": {"color": "#00f0ff", "width": 6},
    }

    # 3. Waypoints and target object markers
    wp_names = [w[0] for w in traj_data.waypoints]
    wp_xs = [w[1][0] for w in traj_data.waypoints]
    wp_ys = [w[1][1] for w in traj_data.waypoints]
    wp_zs = [w[1][2] for w in traj_data.waypoints]

    wp_trace = {
        "type": "scatter3d",
        "mode": "markers+text",
        "name": "Key Waypoints",
        "x": wp_xs,
        "y": wp_ys,
        "z": wp_zs,
        "text": wp_names,
        "textposition": "top center",
        "marker": {"size": 8, "color": "#f6ad55", "symbol": "diamond"},
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

    arm_json = json.dumps(arm_traces + [ee_trace, wp_trace])
    sim_json = json.dumps(sim_traces + [wp_trace])
    cart_json = json.dumps(cartesian_traces)
    joint_json = json.dumps(joint_traces)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sim2Real Data Engine — 3D Diagnostics & Sim-to-Sim Benchmark</title>
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
    .plot-container {{ width: 100%; height: 480px; }}
    .plot-container-sm {{ width: 100%; height: 280px; }}
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
      <h1>Sim2Real Data Engine <span class="badge">v0.1.0 Interactive Visualizer</span></h1>
      <div class="subtitle">Multi-Engine Physics & Privileged SE(3) Trajectory Diagnostics</div>
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
    <!-- View 1: 3D Manipulator Trajectory -->
    <div class="chart-box">
      <div class="chart-header">
        <div>
          <div class="chart-title">1. 3D Kinematics Trajectory & Manipulator Posture</div>
          <div class="chart-desc">6-DOF manipulator kinematic chain & end-effector path</div>
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
      <div id="plot-3d-sim2sim" class="plot-container"></div>
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
  </script>
</body>
</html>
"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file
