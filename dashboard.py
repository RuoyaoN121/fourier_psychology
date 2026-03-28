"""
Fourier Psychology — Spectral Profile
======================================
Single-page Streamlit app.  Enter a GitHub username or select a synthetic
developer to see the four spectral metrics, power spectrum, interpretive
narrative, divergence score, and temporal profile.

Launch:
    streamlit run dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import analyse, SpectralProfile, POPULATION_BASELINE
from ingestion import (
    generate_synthetic_data,
    fetch_github_events,
    build_signal_from_events,
)
from narrative import generate_interpretation

# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fourier Psychology",
    page_icon="\u2609",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.overview-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
}
.overview-card .card-label {
    color: #9ca3af;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.overview-card .card-value {
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 2px;
}
.overview-card .card-sub {
    color: #6b7280;
    font-size: 0.78rem;
}

.metric-block {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 28px 26px;
}
.metric-block .mb-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.metric-block .mb-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.metric-block .mb-z {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-bottom: 14px;
}
.metric-block .mb-text {
    color: #d1d5db;
    font-size: 0.88rem;
    line-height: 1.65;
}

.divergence-box {
    background: rgba(168,85,247,0.06);
    border: 1px solid rgba(168,85,247,0.18);
    border-radius: 14px;
    padding: 28px;
}
.urgency-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.18);
    border-radius: 12px;
    padding: 22px 24px;
    margin-top: 16px;
    color: #fbbf24;
    font-size: 0.88rem;
    line-height: 1.65;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 4px;
}
.section-caption {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 18px;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Plotly defaults
# ═══════════════════════════════════════════════════════════════════════════

_PLT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#d1d5db"),
)
_GRID = "rgba(255,255,255,0.04)"

# Metric accent colours
_C = {
    "entropy": "#818cf8",
    "centroid": "#34d399",
    "hnr": "#fbbf24",
    "slope": "#f472b6",
    "divergence": "#a855f7",
}

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — data source
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<h2 style='background:linear-gradient(135deg,#6366f1,#a855f7);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "margin-bottom:0;'>\u2609 Fourier Psychology</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Spectral self-knowledge through signal processing")
    st.divider()

    mode = st.radio(
        "Data source",
        ["GitHub username", "Synthetic developer"],
        index=1,
        label_visibility="collapsed",
    )

    signal = None
    start_date = ""
    end_date = ""
    start_weekday = 0  # Monday default

    if mode == "Synthetic developer":
        @st.cache_data(show_spinner="Generating synthetic data\u2026")
        def _synth():
            return generate_synthetic_data(n_developers=100, days=180)

        signals = _synth()
        dev = st.selectbox("Select developer", sorted(signals.keys()))
        signal = signals[dev]
        start_date = "2024-01-01"
        end_date = "2024-06-28"
        start_weekday = 0  # synthetic starts Monday

    else:
        username = st.text_input("GitHub username", placeholder="e.g. torvalds")
        if username:
            try:
                with st.spinner(f"Fetching events for **{username}**\u2026"):
                    events = fetch_github_events(username)
                if not events:
                    st.warning("No public events found for this user.")
                else:
                    signal, start_date, end_date, start_weekday = (
                        build_signal_from_events(events)
                    )
                    if len(signal) < 48:
                        st.error(
                            f"Only {len(signal)} hours of data \u2014 need at "
                            f"least 48 for meaningful analysis."
                        )
                        signal = None
                    elif len(signal) < 168:
                        st.warning(
                            f"Only {len(signal)} hours of data. Results may "
                            f"be unreliable with less than a week."
                        )
            except ValueError as exc:
                st.error(str(exc))

    if signal is not None:
        st.divider()
        st.markdown(f"**Events:** `{int(signal.sum()):,}`")
        st.markdown(f"**Span:** `{len(signal) // 24}` days")
        st.markdown(f"**Period:** `{start_date}` \u2014 `{end_date}`")

# ═══════════════════════════════════════════════════════════════════════════
# Welcome screen (no data selected yet)
# ═══════════════════════════════════════════════════════════════════════════

if signal is None:
    st.markdown("")
    st.markdown("## Fourier Psychology")
    st.markdown(
        "A mathematical instrument for self-knowledge. Using Fourier "
        "analysis, we extract four spectral metrics from your GitHub "
        "activity that quantify behavioural traits invisible in the raw data."
    )
    st.markdown("")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**Spectral Entropy**\n\nOrder vs chaos")
    col2.markdown("**Spectral Centroid**\n\nGoverning timescale")
    col3.markdown("**HNR**\n\nPattern vs noise")
    col4.markdown("**Spectral Slope**\n\nStability vs reactivity")
    st.markdown("")
    st.info("Select a data source in the sidebar to begin.", icon="\U0001F50D")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# Run analysis
# ═══════════════════════════════════════════════════════════════════════════

profile: SpectralProfile = analyse(signal, start_date, end_date)
interp = generate_interpretation(profile)

# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Overview cards
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("")

c1, c2, c3, c4, c5 = st.columns(5)


def _overview(col, label, value, sub, color):
    col.markdown(
        f"<div class='overview-card'>"
        f"<div class='card-label'>{label}</div>"
        f"<div class='card-value' style='color:{color}'>{value}</div>"
        f"<div class='card-sub'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


_overview(c1, "Spectral Entropy", f"{profile.spectral_entropy:.2f}",
          "0 ordered \u2014 1 chaotic", _C["entropy"])
_overview(c2, "Governing Timescale", profile.governing_timescale_label,
          f"{profile.spectral_centroid_hz:.5f} cph", _C["centroid"])
_overview(c3, "HNR", f"{profile.harmonic_to_noise_ratio_dB:+.1f} dB",
          "pattern vs noise", _C["hnr"])
_overview(c4, "Spectral Slope", f"{profile.spectral_slope:.2f}",
          "0 reactive \u2014 2 rigid", _C["slope"])
_overview(c5, "Divergence",
          f"{profile.divergence['composite_D']:.1f}\u03c3",
          "from population mean", _C["divergence"])

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Power spectrum (hero visual)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>Power Spectrum</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-caption'>"
    "Your activity\u2019s frequency fingerprint. Peaks reveal the repeating "
    "cycles hidden in your behaviour. X-axis shows period (hours / days), "
    "y-axis shows strength."
    "</div>",
    unsafe_allow_html=True,
)

freqs = np.array(profile.power_spectrum["frequencies"])
power = np.array(profile.power_spectrum["power"])

# Convert to period and filter
valid = freqs > 0
f_valid = freqs[valid]
p_valid = power[valid]
periods = 1.0 / f_valid
max_period = min(profile.metadata["observation_days"] * 24 / 2, 1500)
keep = (periods >= 2) & (periods <= max_period)
periods = periods[keep]
p_valid = p_valid[keep]

# Sort by period ascending for plotting
order = np.argsort(periods)
periods = periods[order]
p_valid = p_valid[order]

fig_spec = go.Figure()
fig_spec.add_trace(go.Scatter(
    x=periods, y=p_valid,
    mode="lines",
    line=dict(color="#a855f7", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(168,85,247,0.08)",
    hovertemplate="Period: %{x:.1f}h<br>Power: %{y:.0f}<extra></extra>",
))

# Annotate dominant peaks
_PEAK_COLORS = ["#6366f1", "#34d399", "#fbbf24", "#f472b6", "#22d3ee"]
for i, peak in enumerate(profile.dominant_periods[:5]):
    ph = peak["period_hours"]
    if ph < 2 or ph > max_period:
        continue
    # Find nearest power value in our plotted data
    idx = int(np.argmin(np.abs(periods - ph)))
    fig_spec.add_annotation(
        x=ph, y=p_valid[idx],
        text=peak["label"],
        showarrow=True,
        arrowhead=2,
        arrowcolor=_PEAK_COLORS[i % len(_PEAK_COLORS)],
        font=dict(color=_PEAK_COLORS[i % len(_PEAK_COLORS)], size=11),
        ax=0, ay=-35 - i * 12,
    )

# Human-readable x-axis ticks
_TICKS = [2, 4, 6, 8, 12, 24, 48, 72, 120, 168, 336, 720]
_LABELS = ["2h", "4h", "6h", "8h", "12h", "1d", "2d", "3d", "5d", "1w", "2w", "30d"]
tick_vals = [t for t in _TICKS if t <= max_period]
tick_text = [_LABELS[i] for i, t in enumerate(_TICKS) if t <= max_period]

fig_spec.update_layout(
    **_PLT,
    height=420,
    margin=dict(l=50, r=20, t=40, b=50),
    yaxis_title="Power  |F(\u03c9)|\u00b2",
    showlegend=False,
)
fig_spec.update_xaxes(
    type="log", title="Period",
    tickvals=tick_vals, ticktext=tick_text, gridcolor=_GRID,
)
fig_spec.update_yaxes(gridcolor=_GRID)
st.plotly_chart(fig_spec, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Detailed metrics (2 x 2 grid)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div class='section-title'>Spectral Metrics \u2014 Detailed</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-caption'>"
    "Each metric quantifies a different dimension of your behavioural structure."
    "</div>",
    unsafe_allow_html=True,
)


def _z_label(z: float) -> str:
    sign = "+" if z > 0 else ""
    return f"{sign}{z:.1f}\u03c3 from population mean"


def _metric_block(name, value_str, z_score, interpretation, color):
    st.markdown(
        f"<div class='metric-block' style='border-left:3px solid {color};'>"
        f"<div class='mb-label' style='color:{color}'>{name}</div>"
        f"<div class='mb-value' style='color:{color}'>{value_str}</div>"
        f"<div class='mb-z'>{_z_label(z_score)}</div>"
        f"<div class='mb-text'>{interpretation}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


row1_l, row1_r = st.columns(2)
with row1_l:
    _metric_block(
        "Spectral Entropy (H\u2099)",
        f"{profile.spectral_entropy:.3f}",
        profile.divergence["z_entropy"],
        interp["entropy"],
        _C["entropy"],
    )
with row1_r:
    _metric_block(
        "Governing Timescale (C\u209b)",
        profile.governing_timescale_label,
        profile.divergence["z_centroid"],
        interp["centroid"],
        _C["centroid"],
    )

st.markdown("")

row2_l, row2_r = st.columns(2)
with row2_l:
    _metric_block(
        "Harmonic-to-Noise Ratio",
        f"{profile.harmonic_to_noise_ratio_dB:+.1f} dB",
        profile.divergence["z_hnr"],
        interp["hnr"],
        _C["hnr"],
    )
with row2_r:
    _metric_block(
        "Spectral Slope (\u03b2)",
        f"{profile.spectral_slope:.3f}",
        profile.divergence["z_slope"],
        interp["slope"],
        _C["slope"],
    )

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — Divergence
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div class='section-title'>Divergence from Population</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-caption'>"
    "How far your behavioural structure is from the population average, "
    "measured in standard deviations across all four metrics."
    "</div>",
    unsafe_allow_html=True,
)

div_l, div_r = st.columns([1, 2])

with div_l:
    D = profile.divergence["composite_D"]
    st.markdown(
        f"<div class='divergence-box' style='text-align:center;'>"
        f"<div style='color:#9ca3af;font-size:0.72rem;font-weight:600;"
        f"letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;'>"
        f"Composite Divergence</div>"
        f"<div style='font-size:3rem;font-weight:700;color:#a855f7;'>"
        f"{D:.1f}\u03c3</div>"
        f"<div style='color:#d1d5db;font-size:0.88rem;line-height:1.65;"
        f"margin-top:14px;'>{interp['divergence']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with div_r:
    z_vals = [
        profile.divergence["z_entropy"],
        profile.divergence["z_centroid"],
        profile.divergence["z_hnr"],
        profile.divergence["z_slope"],
    ]
    z_names = ["Entropy", "Centroid", "HNR", "Slope"]
    z_colors = [_C["entropy"], _C["centroid"], _C["hnr"], _C["slope"]]

    fig_z = go.Figure()
    fig_z.add_trace(go.Bar(
        y=z_names,
        x=z_vals,
        orientation="h",
        marker=dict(color=z_colors),
        hovertemplate="%{y}: %{x:+.2f}\u03c3<extra></extra>",
    ))
    fig_z.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig_z.update_layout(
        **_PLT,
        height=260,
        xaxis_title="z-score (\u03c3)",
        showlegend=False,
        margin=dict(l=80, r=20, t=10, b=40),
    )
    fig_z.update_xaxes(gridcolor=_GRID)
    fig_z.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_z, use_container_width=True)

# Composite narrative
st.markdown(
    f"<div style='background:rgba(255,255,255,0.025);border:1px solid "
    f"rgba(255,255,255,0.06);border-radius:12px;padding:22px 26px;"
    f"color:#d1d5db;font-size:0.92rem;line-height:1.7;'>"
    f"{interp['composite']}</div>",
    unsafe_allow_html=True,
)

# Urgency note
if interp["urgency_note"]:
    st.markdown(
        f"<div class='urgency-box'>{interp['urgency_note']}</div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# Section 5 — Temporal profile
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div class='section-title'>Temporal Profile</div>", unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-caption'>"
    "When you work \u2014 derived from the raw time series, not the Fourier "
    "transform. These patterns are visible in the data; the spectral "
    "metrics above quantify what is not."
    "</div>",
    unsafe_allow_html=True,
)

# Compute hour-of-day and day-of-week profiles
hour_profile = np.array([signal[h::24].sum() for h in range(24)])
n_complete_days = len(signal) // 24
daily_sums = signal[: n_complete_days * 24].reshape(n_complete_days, 24).sum(axis=1)
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
week_profile = np.zeros(7)
for d in range(n_complete_days):
    week_profile[(start_weekday + d) % 7] += daily_sums[d]

tp_l, tp_r = st.columns(2)

with tp_l:
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Bar(
        x=list(range(24)),
        y=hour_profile,
        marker=dict(
            color=hour_profile,
            colorscale=[[0, "#1e1b4b"], [0.5, "#6366f1"], [1, "#c084fc"]],
            showscale=False,
        ),
        hovertemplate="Hour %{x}:00<br>Events: %{y:.0f}<extra></extra>",
    ))
    fig_hour.update_layout(
        **_PLT,
        height=300,
        yaxis_title="Total events",
        showlegend=False,
        margin=dict(l=50, r=10, t=10, b=50),
    )
    fig_hour.update_xaxes(
        title="Hour of day",
        tickvals=list(range(0, 24, 3)),
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
        gridcolor=_GRID,
    )
    fig_hour.update_yaxes(gridcolor=_GRID)
    st.plotly_chart(fig_hour, use_container_width=True)

with tp_r:
    fig_day = go.Figure()
    fig_day.add_trace(go.Bar(
        x=day_names,
        y=week_profile,
        marker=dict(
            color=week_profile,
            colorscale=[[0, "#022c22"], [0.5, "#10b981"], [1, "#6ee7b7"]],
            showscale=False,
        ),
        hovertemplate="%{x}<br>Events: %{y:.0f}<extra></extra>",
    ))
    fig_day.update_layout(
        **_PLT,
        height=300,
        yaxis_title="Total events",
        showlegend=False,
        margin=dict(l=50, r=10, t=10, b=50),
    )
    fig_day.update_xaxes(title="Day of week", gridcolor=_GRID)
    fig_day.update_yaxes(gridcolor=_GRID)
    st.plotly_chart(fig_day, use_container_width=True)
