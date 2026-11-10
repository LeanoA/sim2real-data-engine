"""Interactive 3D WebGL Dashboard comparing trained policy rollouts against privileged expert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim2real_engine.visualization.data import PolicyVsExpertComparisonData


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

    expert_ee_xs = [p[0] for p in expert.ee_trajectory]
    expert_ee_ys = [p[1] for p in expert.ee_trajectory]
    expert_ee_zs = [p[2] for p in expert.ee_trajectory]

    total_steps = len(policy.timesteps_s)

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

    # Initial arm pose (t=0)
    init_arm = all_arm_coords[0]

    # Trace 0: Moving Policy Robot Arm
    policy_arm_trace = {
        "type": "scatter3d",
        "mode": "lines+markers",
        "name": "Policy 6-DOF Manipulator",
        "x": init_arm["xs"],
        "y": init_arm["ys"],
        "z": init_arm["zs"],
        "line": {"color": "#ec4899", "width": 8},
        "marker": {"size": 8, "color": "#ffffff"},
    }

    # Trace 1: Moving Payload Object
    obj_trace = {
        "type": "scatter3d",
        "mode": "markers",
        "name": "Payload Object (Policy)",
        "x": [policy_obj_xs[0]],
        "y": [policy_obj_ys[0]],
        "z": [policy_obj_zs[0]],
        "marker": {"size": 10, "color": "#f59e0b", "symbol": "diamond"},
    }

    # Trace 2: Policy Dynamic Traveled Trail
    policy_trail_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Policy Executed Trail",
        "x": [policy_ee_xs[0]],
        "y": [policy_ee_ys[0]],
        "z": [policy_ee_zs[0]],
        "line": {"color": "#ec4899", "width": 6},
    }

    # Trace 3: Expert Ground Truth Trajectory (Ghost reference)
    expert_ref_trace = {
        "type": "scatter3d",
        "mode": "lines",
        "name": "Expert Reference Trajectory",
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
        "name": "Expert Target Waypoints",
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
            "text": "3D Kinematic Space: Policy (Pink) vs. Expert Ground Truth (Cyan)",
            "font": {"size": 16, "color": "#f8fafc", "family": "system-ui, sans-serif"},
            "x": 0.05,
            "y": 0.95,
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
        "margin": {"l": 0, "r": 0, "b": 0, "t": 40},
        "legend": {
            "font": {"color": "#94a3b8", "size": 11},
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "x": 0.02,
            "y": 0.92,
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
    }

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sim2Real Policy vs. Expert 3D Evaluation Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card: #0f172a;
      --border: #1e293b;
      --accent-policy: #ec4899;
      --accent-expert: #00f0ff;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      padding: 1.5rem;
      min-height: 100vh;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1rem;
    }}
    .header h1 {{ font-size: 1.5rem; font-weight: 700; }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.65rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 9999px;
      margin-left: 0.5rem;
    }}
    .badge-policy {{
      background: rgba(236, 72, 153, 0.2);
      color: var(--accent-policy);
      border: 1px solid var(--accent-policy);
    }}
    .badge-expert {{
      background: rgba(0, 240, 255, 0.2);
      color: var(--accent-expert);
      border: 1px solid var(--accent-expert);
    }}
    
    .grid-kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .kpi-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1rem;
    }}
    .kpi-title {{
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .kpi-value {{ font-size: 1.5rem; font-weight: 700; margin: 0.35rem 0; }}
    .kpi-sub {{ font-size: 0.75rem; color: var(--text-muted); }}

    .main-stage {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    @media (max-width: 1024px) {{
      .main-stage {{ grid-template-columns: 1fr; }}
    }}
    .plot-container {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      overflow: hidden;
      height: 600px;
    }}
    .telemetry-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .hud-row {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.5rem 0;
      font-size: 0.85rem;
    }}
    .hud-label {{ color: var(--text-muted); }}
    .hud-value {{ font-family: monospace; font-weight: 600; }}

    .player-controls {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .control-row {{
      display: flex;
      align-items: center;
      gap: 1rem;
    }}
    .btn {{
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.85rem;
    }}
    .btn:hover {{ background: #334155; }}
    .btn-primary {{
      background: var(--accent-policy);
      border-color: var(--accent-policy);
      color: white;
    }}
    .btn-primary:hover {{ background: #db2777; }}
    .slider {{
      flex: 1;
      accent-color: var(--accent-policy);
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Policy vs. Privileged Expert 3D Evaluation
        <span class="badge badge-policy">Policy (BC-MLP)</span>
        <span class="badge badge-expert">Expert (IK-FSM)</span>
      </h1>
      <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.25rem;">
        Synchronized closed-loop execution in native physics (seed={comparison_data.seed})
      </p>
    </div>
  </div>

  <div class="grid-kpis">
    <div class="kpi-card">
      <div class="kpi-title">Mean Spatial Tracking Error</div>
      <div class="kpi-value" style="color: var(--accent-policy);">
        {comparison_data.mean_tracking_error_mm:.1f} mm
      </div>
      <div class="kpi-sub">Mean Cartesian error across rollout steps</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Max Spatial Error</div>
      <div class="kpi-value">{comparison_data.max_tracking_error_mm:.1f} mm</div>
      <div class="kpi-sub">Peak deviation during trajectory</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Policy EE Net Displacement</div>
      <div class="kpi-value">{comparison_data.policy_ee_displacement_mm:.1f} mm</div>
      <div class="kpi-sub">Total physical arm movement in workspace</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Inference Latency (P50 / P99)</div>
      <div class="kpi-value">{comparison_data.inference_p50_latency_ms:.2f} ms</div>
      <div class="kpi-sub">
        P99: {comparison_data.inference_p99_latency_ms:.2f} ms (Target &lt; 15 ms)
      </div>
    </div>
  </div>

  <div class="main-stage">
    <div class="plot-container" id="plot-3d"></div>

    <div class="telemetry-card">
      <h3 style="font-size: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem;">
        Real-Time Telemetry HUD
      </h3>
      <div class="hud-row">
        <span class="hud-label">Current Step / Total:</span>
        <span class="hud-value" id="hud-step">0 / {total_steps}</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Time Elapsed:</span>
        <span class="hud-value" id="hud-time">0.00 s</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Instant Spatial Error:</span>
        <span class="hud-value" id="hud-error" style="color: var(--accent-policy);">0.0 mm</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Policy EE Position:</span>
        <span class="hud-value" id="hud-policy-ee">[0.0, 0.0, 0.0]</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Expert EE Position:</span>
        <span class="hud-value" id="hud-expert-ee">[0.0, 0.0, 0.0]</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Policy Object Position:</span>
        <span class="hud-value" id="hud-policy-obj">[0.0, 0.0, 0.0]</span>
      </div>
      <div class="hud-row">
        <span class="hud-label">Simulator Backend:</span>
        <span class="hud-value">{comparison_data.backend}</span>
      </div>
    </div>
  </div>

  <div class="player-controls">
    <div class="control-row">
      <button class="btn btn-primary" id="btn-play">▶ Play</button>
      <button class="btn" id="btn-reset">⏮ Reset</button>
      <button class="btn" id="btn-prev">◀ Step</button>
      <button class="btn" id="btn-next">Step ▶</button>
      <input type="range" class="slider" id="scrubber" min="0" max="{total_steps - 1}" value="0">
      <span style="font-family: monospace; font-size: 0.85rem;" id="scrubber-label">Step 0</span>
    </div>
  </div>

  <script>
    const bundle = {json.dumps(anim_bundle)};
    const plotData = {json.dumps(plot_data)};
    const plotLayout = {json.dumps(plot_layout)};

    Plotly.newPlot('plot-3d', plotData, plotLayout, {{responsive: true}});

    let currentStep = 0;
    let isPlaying = false;
    let animTimer = null;

    const btnPlay = document.getElementById('btn-play');
    const btnReset = document.getElementById('btn-reset');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const slider = document.getElementById('scrubber');
    const sliderLabel = document.getElementById('scrubber-label');

    function renderStep(step) {{
      if (step < 0) step = 0;
      if (step >= bundle.totalSteps) step = bundle.totalSteps - 1;
      currentStep = step;

      slider.value = step;
      sliderLabel.textContent = 'Step ' + step;

      const arm = bundle.armTrajectory[step];
      const polEe = bundle.policyEeTrajectory[step];
      const expIdx = Math.min(step, bundle.expertEeTrajectory.length - 1);
      const expEe = bundle.expertEeTrajectory ? bundle.expertEeTrajectory[expIdx] : polEe;
      const polObj = bundle.policyObjectTrajectory[step];

      // Update Plotly Traces:
      // 0: Policy Arm
      // 1: Policy Object
      // 2: Policy Traveled Trail
      const trailXs = bundle.policyEeTrajectory.slice(0, step + 1).map(p => p[0]);
      const trailYs = bundle.policyEeTrajectory.slice(0, step + 1).map(p => p[1]);
      const trailZs = bundle.policyEeTrajectory.slice(0, step + 1).map(p => p[2]);

      Plotly.restyle('plot-3d', {{
        x: [arm.xs, [polObj[0]], trailXs],
        y: [arm.ys, [polObj[1]], trailYs],
        z: [arm.zs, [polObj[2]], trailZs],
      }}, [0, 1, 2]);

      // Update Telemetry HUD
      document.getElementById('hud-step').textContent = step + ' / ' + bundle.totalSteps;
      document.getElementById('hud-time').textContent = bundle.timesteps[step].toFixed(2) + ' s';
      const errStr = bundle.spatialErrorsMm[step].toFixed(1) + ' mm';
      document.getElementById('hud-error').textContent = errStr;
      const polEeStr = '[' + polEe.map(v => v.toFixed(3)).join(', ') + ']';
      document.getElementById('hud-policy-ee').textContent = polEeStr;
      const expEeStr = '[' + expEe.map(v => v.toFixed(3)).join(', ') + ']';
      document.getElementById('hud-expert-ee').textContent = expEeStr;
      const polObjStr = '[' + polObj.map(v => v.toFixed(3)).join(', ') + ']';
      document.getElementById('hud-policy-obj').textContent = polObjStr;
    }}

    function play() {{
      if (currentStep >= bundle.totalSteps - 1) currentStep = 0;
      isPlaying = true;
      btnPlay.textContent = '⏸ Pause';
      btnPlay.classList.remove('btn-primary');
      clearInterval(animTimer);
      animTimer = setInterval(() => {{
        if (currentStep >= bundle.totalSteps - 1) {{
          pause();
        }} else {{
          renderStep(currentStep + 1);
        }}
      }}, 50);
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
    slider.addEventListener('input', (e) => {{
      pause();
      renderStep(parseInt(e.target.value, 10));
    }});

    renderStep(0);
  </script>
</body>
</html>
"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file
