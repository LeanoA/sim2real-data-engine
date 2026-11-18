"""Empirical benchmark section builder (Wilson CIs, 2^4 ANOVA, psychometric MLE, FFT/jerk)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def tip_html(label: str, text: str) -> str:
    """Render an accessible hover/focus educational tooltip badge."""
    return (
        f'<span class="tip-wrap" tabindex="0" aria-label="{label}: {text}">'
        '<span class="tip-icon" aria-hidden="true">?</span>'
        f'<span class="tip-popover" role="tooltip"><strong>{label}</strong><br/>{text}</span>'
        "</span>"
    )


def _extract_ci_trace(
    name: str,
    color: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    regimes = ["nominal", "id_dr_ensembled", "ood_dr_ensembled"]
    labels = ["Nominal", "ID-DR", "True OOD"]
    ys: list[float] = []
    err_plus: list[float] = []
    err_minus: list[float] = []
    custom: list[str] = []

    for key in regimes:
        ci = payload.get(key, {}).get("surgical", {}).get("stages", {}).get("chained_ci", {})
        rate = float(ci.get("rate", 0.0)) * 100.0
        lo = float(ci.get("ci_low", rate / 100.0)) * 100.0
        hi = float(ci.get("ci_high", rate / 100.0)) * 100.0
        ys.append(round(rate, 1))
        err_plus.append(round(max(0.0, hi - rate), 1))
        err_minus.append(round(max(0.0, rate - lo), 1))
        custom.append(f"[{lo:.1f}%, {hi:.1f}%] (N={ci.get('trials', 20)})")

    return {
        "type": "bar",
        "name": name,
        "x": labels,
        "y": ys,
        "customdata": custom,
        "error_y": {
            "type": "data",
            "symmetric": False,
            "array": err_plus,
            "arrayminus": err_minus,
            "color": "#cbd5e1",
            "thickness": 1.6,
            "width": 5,
        },
        "marker": {"color": color},
        "hovertemplate": (
            "%{x} — %{fullData.name}: <b>%{y:.1f}%</b>"
            "<br>95% Wilson CI: %{customdata}<extra></extra>"
        ),
    }


def _build_anova_traces(sensitivity: dict[str, Any]) -> list[dict[str, Any]]:
    pretty_names = {
        "spatial_shift_m": "Spatial Shift (XY)",
        "mass_scale": "Payload Mass",
        "friction_scale": "Surface Friction",
        "damping_scale": "Joint Damping",
    }
    rows: list[tuple[str, float, float]] = []
    for me in sensitivity.get("main_effects", []):
        fname = pretty_names.get(me["factor_name"], me["factor_name"])
        rows.append(
            (
                fname,
                round(float(me.get("grasp_eta_sq", 0.0)) * 100.0, 1),
                round(float(me.get("chained_eta_sq", 0.0)) * 100.0, 1),
            )
        )
    for ix in sensitivity.get("interactions_2way", []):
        g_eta = float(ix.get("grasp_eta_sq", 0.0)) * 100.0
        c_eta = float(ix.get("chained_eta_sq", 0.0)) * 100.0
        if g_eta > 0.5 or c_eta > 0.5:
            fa = pretty_names.get(ix["factor_a"], ix["factor_a"])
            fb = pretty_names.get(ix["factor_b"], ix["factor_b"])
            rows.append((f"{fa} × {fb} (2-Way)", round(g_eta, 1), round(c_eta, 1)))

    rows.sort(key=lambda r: max(r[1], r[2]))
    labels = [r[0] for r in rows]
    grasp_vals = [r[1] for r in rows]
    task_vals = [r[2] for r in rows]

    return [
        {
            "type": "bar",
            "orientation": "h",
            "name": "Grasp Clamping η² (%)",
            "y": labels,
            "x": grasp_vals,
            "marker": {"color": "#06b6d4"},
            "hovertemplate": "%{y}<br>Grasp variance η²: <b>%{x:.1f}%</b><extra></extra>",
        },
        {
            "type": "bar",
            "orientation": "h",
            "name": "Chained Task η² (%)",
            "y": labels,
            "x": task_vals,
            "marker": {"color": "#8b5cf6"},
            "hovertemplate": "%{y}<br>Task variance η²: <b>%{x:.1f}%</b><extra></extra>",
        },
    ]


def _build_psychometric_traces(sweep: dict[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
    points = sweep.get("noise_points", [])
    sigmas = [float(p["sigma_mm"]) for p in points]
    rates = [round(float(p["chained_success_rate"]) * 100.0, 1) for p in points]
    err_plus: list[float] = []
    err_minus: list[float] = []
    for p, r in zip(points, rates, strict=True):
        ci = p.get("chained_ci", {})
        lo = float(ci.get("ci_low", r / 100.0)) * 100.0
        hi = float(ci.get("ci_high", r / 100.0)) * 100.0
        err_plus.append(round(max(0.0, hi - r), 1))
        err_minus.append(round(max(0.0, r - lo), 1))

    traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "markers+lines",
            "name": "Observed Success (N=20/σ)",
            "x": sigmas,
            "y": rates,
            "error_y": {
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "color": "#64748b",
                "thickness": 1.3,
            },
            "line": {"color": "#38bdf8", "width": 2.2},
            "marker": {"size": 7, "color": "#06b6d4"},
            "hovertemplate": "Noise σ=%{x:.1f} mm<br>Success: <b>%{y:.1f}%</b><extra></extra>",
        }
    ]

    fit = sweep.get("psychometric_fit", {})
    sigma_50: float | None = None
    if fit.get("converged") and fit.get("slope_per_mm") is not None:
        slope = float(fit["slope_per_mm"])
        intercept = float(fit["intercept"])
        raw_s50 = fit.get("sigma_50_mm")
        sigma_50 = round(float(raw_s50), 2) if raw_s50 is not None else None
        xs_fit = [i * 0.5 for i in range(41)]
        ys_fit = [round(100.0 / (1.0 + math.exp(-(intercept - slope * x))), 1) for x in xs_fit]
        fit_name = f"Binomial MLE Fit (σ₅₀={sigma_50:.2f} mm)" if sigma_50 else "Binomial MLE Fit"
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": fit_name,
                "x": xs_fit,
                "y": ys_fit,
                "line": {"color": "#f59e0b", "width": 2.4, "dash": "dash"},
                "hovertemplate": "σ=%{x:.1f} mm<br>MLE P(success): <b>%{y:.1f}%</b><extra></extra>",
            }
        )
    return traces, sigma_50


def _build_control_traces(
    models: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    names = [label for label, _ in models]
    hf_raw: list[float] = []
    hf_ens: list[float] = []
    hf_red_text: list[str] = []
    jerk_red: list[float] = []

    for _, payload in models:
        cd_raw = payload.get("ood_dr_no_ensemble", {}).get("control_diagnostics", {})
        cd_ens = payload.get("ood_dr_ensembled", {}).get("control_diagnostics", {})
        raw_pct = round(float(cd_raw.get("mean_hf_power_ratio", 0.0)) * 100.0, 1)
        ens_pct = round(float(cd_ens.get("mean_hf_power_ratio", 0.0)) * 100.0, 1)
        hf_red = float(payload.get("hf_power_reduction_pct", 0.0))
        j_red = float(payload.get("physical_jerk_reduction_pct", 0.0))
        hf_raw.append(raw_pct)
        hf_ens.append(ens_pct)
        hf_red_text.append(f"-{hf_red:.1f}% HF / -{j_red:.1f}% Jerk")
        jerk_red.append(round(j_red, 1))

    return [
        {
            "type": "bar",
            "name": "Raw Single-Step HF ≥5 Hz (%)",
            "x": names,
            "y": hf_raw,
            "marker": {"color": "#ef4444"},
            "hovertemplate": "%{x}<br>Raw HF Power ≥5Hz: <b>%{y:.1f}%</b><extra></extra>",
        },
        {
            "type": "bar",
            "name": "Ensembled HF ≥5 Hz (%)",
            "x": names,
            "y": hf_ens,
            "customdata": hf_red_text,
            "marker": {"color": "#10b981"},
            "hovertemplate": (
                "%{x}<br>Ensembled HF Power: <b>%{y:.1f}%</b>"
                "<br>Reduction: <b>%{customdata}</b><extra></extra>"
            ),
        },
        {
            "type": "scatter",
            "mode": "lines+markers+text",
            "name": "Physical Jerk (q̈̇) Reduction (%)",
            "x": names,
            "y": jerk_red,
            "text": [f"-{v:.1f}%" for v in jerk_red],
            "textposition": "top center",
            "textfont": {"color": "#f59e0b", "size": 11},
            "line": {"color": "#f59e0b", "width": 2.2},
            "marker": {"size": 8, "color": "#f59e0b", "symbol": "diamond"},
            "hovertemplate": "%{x}<br>Physical Joint Jerk cut: <b>-%{y:.1f}%</b><extra></extra>",
        },
    ]


def build_empirical_section_html_and_js(results_dir: Path | None = None) -> tuple[str, str]:
    """Build static Plotly panels + educational tooltips from verified benchmark JSON artifacts."""
    root = results_dir or Path("results")
    eval_world = _load_json_if_exists(root / "eval_world_450_50.json")
    eval_tcp = _load_json_if_exists(root / "eval_tcp_450_50.json")
    eval_ego = _load_json_if_exists(root / "eval_ego_450_50.json")
    sens_tcp = _load_json_if_exists(root / "sensitivity_tcp_m1.json")
    sweep_tcp = _load_json_if_exists(root / "sweep_perception_tcp_m1.json")

    if not (eval_world and eval_tcp and eval_ego and sens_tcp and sweep_tcp):
        return "", ""

    ci_data = [
        _extract_ci_trace("World-Frame 30D", "#ec4899", eval_world),
        _extract_ci_trace("TCP-Hybrid 30D (Winner)", "#06b6d4", eval_tcp),
        _extract_ci_trace("Ego-SE(3) 20D (Ablation)", "#8b5cf6", eval_ego),
    ]
    anova_data = _build_anova_traces(sens_tcp)
    psycho_data, sigma_50 = _build_psychometric_traces(sweep_tcp)
    control_data = _build_control_traces(
        [
            ("World-Frame 30D", eval_world),
            ("TCP-Hybrid 30D", eval_tcp),
            ("Ego-SE(3) 20D", eval_ego),
        ]
    )

    ci_layout = {
        "barmode": "group",
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#090d16",
        "font": {"color": "#cbd5e1", "size": 11},
        "margin": {"l": 46, "r": 14, "t": 18, "b": 38},
        "yaxis": {"title": "Task Success (%)", "range": [0, 105], "gridcolor": "#1e293b"},
        "xaxis": {"gridcolor": "#1e293b"},
        "legend": {"orientation": "h", "y": 1.15, "font": {"size": 10}},
    }
    anova_layout = {
        "barmode": "group",
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#090d16",
        "font": {"color": "#cbd5e1", "size": 11},
        "margin": {"l": 145, "r": 18, "t": 18, "b": 38},
        "xaxis": {"title": "Variance Explained η² (%)", "range": [0, 55], "gridcolor": "#1e293b"},
        "yaxis": {"gridcolor": "#1e293b"},
        "legend": {"orientation": "h", "y": 1.15, "font": {"size": 10}},
    }
    psycho_shapes: list[dict[str, Any]] = []
    psycho_annotations: list[dict[str, Any]] = []
    if sigma_50 is not None:
        psycho_shapes.append(
            {
                "type": "line",
                "x0": sigma_50,
                "x1": sigma_50,
                "y0": 0,
                "y1": 100,
                "line": {"color": "#f59e0b", "width": 1.5, "dash": "dot"},
            }
        )
        psycho_annotations.append(
            {
                "x": sigma_50,
                "y": 88,
                "text": f"σ₅₀ = {sigma_50:.2f} mm",
                "showarrow": False,
                "xanchor": "left",
                "xshift": 6,
                "font": {"color": "#f59e0b", "size": 11},
            }
        )
    psycho_layout = {
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#090d16",
        "font": {"color": "#cbd5e1", "size": 11},
        "margin": {"l": 46, "r": 16, "t": 18, "b": 42},
        "xaxis": {"title": "Injected Perception Noise σ (mm)", "gridcolor": "#1e293b"},
        "yaxis": {"title": "Task Success (%)", "range": [0, 105], "gridcolor": "#1e293b"},
        "shapes": psycho_shapes,
        "annotations": psycho_annotations,
        "legend": {"orientation": "h", "y": 1.15, "font": {"size": 10}},
    }
    control_layout = {
        "barmode": "group",
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#090d16",
        "font": {"color": "#cbd5e1", "size": 11},
        "margin": {"l": 46, "r": 16, "t": 18, "b": 38},
        "yaxis": {
            "title": "HF Chatter (%) / Jerk Cut (%)",
            "range": [0, 112],
            "gridcolor": "#1e293b",
        },
        "xaxis": {"gridcolor": "#1e293b"},
        "legend": {"orientation": "h", "y": 1.15, "font": {"size": 10}},
    }

    tip_ci = tip_html(
        "95% Wilson Score CI",
        "Statistical confidence bounds designed for binary outcomes on small samples (N=20). "
        "Overlapping bars indicate statistical uncertainty; separated bars show real gains.",
    )
    tip_anova = tip_html(
        "2⁴ Factorial ANOVA & η²",
        "48 balanced episodes varying 4 physical parameters at once. Eta-Squared (η²) measures "
        "what % of OOD failure variance each factor drives—here Damping × Spatial Shift drives "
        "44.4% of grasp failures while Mass & Friction contribute ~0%.",
    )
    tip_psycho = tip_html(
        "Psychometric Curve & σ₅₀ (MLE)",
        "Fits a Binomial logistic curve as Gaussian camera/pose noise (σ mm) increases. "
        "σ₅₀ is the noise threshold where task success drops to 50%.",
    )
    tip_control = tip_html(
        "FFT ≥ 5 Hz & Physical Joint Jerk",
        "HF Power ≥ 5 Hz measures fast chatter; Jerk (rad/s³) measures acceleration spikes. "
        "Without temporal ensembling w_k=exp(-0.05k), OOD success is 0%; ensembling cuts chatter "
        "by ~78–88% and physical jerk by >95%.",
    )

    html_block = f"""
  <section class="empirical-section" aria-labelledby="empirical-heading">
    <div class="empirical-header">
      <div>
        <h2 id="empirical-heading">
          Empirical Benchmarks, Factorial Attribution &amp; Control Diagnostics
        </h2>
        <p class="empirical-sub">
          Panels 1 &amp; 4 compare all three homogeneous <strong>450/50</strong> representations
          (<code>N=20</code> episodes per regime). Panels 2 &amp; 3 drill down into the winning
          <strong>TCP-Hybrid 30D</strong> (<code>bc_tcp_m1_45ep.pt</code>) policy. Hover or focus
          any <span class="tip-icon tip-inline">?</span> badge to inspect terminology.
        </p>
      </div>
    </div>

    <div class="empirical-grid">
      <div class="stat-card">
        <div class="plot-card-header">
          <div class="plot-card-title">
            1. Multi-Model Success Rates (95% Wilson CI) {tip_ci}
          </div>
          <span class="badge badge-arm">All 3 Models &bull; 450/50</span>
        </div>
        <div class="stat-plot" id="plot-stat-ci"></div>
      </div>

      <div class="stat-card">
        <div class="plot-card-header">
          <div class="plot-card-title">
            2. Orthogonal 2⁴ Factorial ANOVA (η² Attribution) {tip_anova}
          </div>
          <span class="badge badge-expert">TCP-Hybrid 30D &bull; 48 Ep</span>
        </div>
        <div class="stat-plot" id="plot-stat-anova"></div>
      </div>

      <div class="stat-card">
        <div class="plot-card-header">
          <div class="plot-card-title">
            3. Psychometric Perception Noise Tolerance (σ₅₀) {tip_psycho}
          </div>
          <span class="badge badge-expert">TCP-Hybrid 30D &bull; N=20/σ</span>
        </div>
        <div class="stat-plot" id="plot-stat-psycho"></div>
      </div>

      <div class="stat-card">
        <div class="plot-card-header">
          <div class="plot-card-title">
            4. Temporal Ensembling: FFT ≥ 5 Hz &amp; Physical Jerk {tip_control}
          </div>
          <span class="badge badge-arm">True OOD &bull; All 3 Models</span>
        </div>
        <div class="stat-plot" id="plot-stat-control"></div>
      </div>
    </div>
  </section>
"""

    js_block = f"""
    Plotly.newPlot('plot-stat-ci', {json.dumps(ci_data)}, {json.dumps(ci_layout)}, {{
      responsive: true, displayModeBar: false
    }});
    Plotly.newPlot('plot-stat-anova', {json.dumps(anova_data)}, {json.dumps(anova_layout)}, {{
      responsive: true, displayModeBar: false
    }});
    Plotly.newPlot('plot-stat-psycho', {json.dumps(psycho_data)}, {json.dumps(psycho_layout)}, {{
      responsive: true, displayModeBar: false
    }});
    Plotly.newPlot('plot-stat-control', {json.dumps(control_data)}, {json.dumps(control_layout)}, {{
      responsive: true, displayModeBar: false
    }});
"""
    return html_block, js_block
