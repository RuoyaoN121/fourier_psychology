"""
Spectral Sabermetrics — Part 3b: Interactive Streamlit Dashboard
=================================================================
Premium dark-themed dashboard with two tabs:

* **Developer Biometrics** — time-domain commit history + frequency-domain
  periodogram with annotated CVS & FSR scores.
* **Spectral Team Matcher** — clustered 4-person teams + 2-D PCA scatter.

Launch:
    streamlit run dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA

from ingestion import load_data, fetch_macro_data
from engine import (
    analyse_developer,
    build_feature_matrix,
    build_radar_profile,
    rank_developers_by_cvs,
    compute_rolling_macro_volatility,
    DeveloperSpectralProfile,
)
from clustering import spectral_cluster_teams

# ═══════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Spectral Sabermetrics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- glassmorphism card ---------- */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.25);
}

div[data-testid="stMetric"] label {
    color: #9ca3af;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-size: 0.72rem;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ---------- tab styling ---------- */
button[data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ---------- team cards ---------- */
.team-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    transition: transform 0.2s, box-shadow 0.2s;
}

.team-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}

.team-card h4 {
    margin: 0 0 10px 0;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.05rem;
}

.team-card .member {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px;
    padding: 4px 12px;
    margin: 3px 4px;
    font-size: 0.85rem;
    color: #c4b5fd;
}

/* ---------- global tweaks ---------- */
header[data-testid="stHeader"] {
    background: transparent;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Shared Plotly layout defaults (dark theme)
# ═══════════════════════════════════════════════════════════════════════════

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#d1d5db"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
)

# Extended colour palette — 25 perceptually distinct hues so every team gets
# its own colour even after the greedy-balanced clustering produces many groups.
_GRADIENT = [
    "#6366f1", "#818cf8", "#a78bfa", "#c084fc", "#d946ef",
    "#f472b6", "#fb7185", "#f87171", "#facc15", "#4ade80",
    "#2dd4bf", "#22d3ee", "#38bdf8", "#60a5fa", "#a3e635",
    "#e879f9", "#f0abfc", "#fbbf24", "#34d399", "#f97316",
    "#ef4444", "#8b5cf6", "#06b6d4", "#10b981", "#ec4899",
]

# ═══════════════════════════════════════════════════════════════════════════
# Data loading (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="🔬 Generating spectral signals…")
def _load_signals():
    return load_data(use_bigquery=False)


@st.cache_data(show_spinner="⚡ Computing Fourier features…")
def _compute_features(_signals):
    return build_feature_matrix(_signals)


@st.cache_data(show_spinner="🧩 Clustering developer rhythms…")
def _cluster(_X, _names):
    return spectral_cluster_teams(_X, _names, team_size=4)


signals = _load_signals()
X, dev_names, feature_names = _compute_features(signals)
team_result = _cluster(X, dev_names)

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<h2 style='background:linear-gradient(135deg,#6366f1,#a855f7);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "margin-bottom:2px;'>🔬 Spectral Sabermetrics</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Cognitive-rhythm analysis for open-source developers")
    st.divider()
    st.markdown(f"**Developers analysed:** `{len(signals)}`")
    st.markdown(f"**Signal length:** `{len(next(iter(signals.values())))} h`")
    st.markdown(f"**Teams formed:** `{team_result.n_clusters}`")

# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠  Developer Biometrics",
    "🚀  Spectral Team Matcher",
    "🕸️  Team Synchrony",
    "🔥  Burnout Visualizer",
    "📈  Macro-Sabermetrics (Alpha)",
])

# ───────────────────────────────────────────────────────────────────────────
# Tab 1 — Developer Biometrics
# ───────────────────────────────────────────────────────────────────────────

with tab1:
    selected_dev = st.selectbox(
        "Select a developer",
        sorted(signals.keys()),
        index=0,
        key="dev_selector",
    )

    profile: DeveloperSpectralProfile = analyse_developer(
        selected_dev, signals[selected_dev]
    )

    # Metric cards row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cognitive Volatility (CVS)", f"{profile.cvs:.3f}")
    c2.metric("Daily Amplitude (FSR)", f"{profile.fsr.daily_amplitude:.1f}")
    c3.metric("Weekly Amplitude (FSR)", f"{profile.fsr.weekly_amplitude:.1f}")
    c4.metric("Total Commits", f"{int(profile.signal.sum()):,}")

    st.markdown("")

    col_time, col_freq = st.columns(2)

    # ── Time-domain chart ──
    with col_time:
        st.markdown("##### 📈 Commit History (Time Domain)")
        hours = np.arange(len(profile.signal))
        days = hours / 24.0
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=days, y=profile.signal,
            mode="lines",
            line=dict(color="#818cf8", width=1),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            name="Commits/hour",
        ))
        fig_time.update_layout(
            **_PLOTLY_LAYOUT,
            xaxis_title="Day",
            yaxis_title="Commits / hour",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Frequency-domain periodogram ──
    with col_freq:
        st.markdown("##### 🌊 Periodogram (Frequency Domain)")
        # Only plot up to 6 cycles/day (period ≥ 4 h) so the chart is readable
        freq_mask = (profile.freqs > 0) & (profile.freqs <= 6)
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(
            x=profile.freqs[freq_mask],
            y=profile.power[freq_mask],
            mode="lines",
            line=dict(color="#a855f7", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(168,85,247,0.10)",
            name="Power",
        ))

        # Annotate daily peak
        daily_freq = 1.0
        daily_idx = int(np.argmin(np.abs(profile.freqs - daily_freq)))
        fig_freq.add_annotation(
            x=daily_freq,
            y=profile.power[daily_idx],
            text="24 h cycle",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#6366f1",
            font=dict(color="#c4b5fd", size=11),
            ax=40, ay=-40,
        )

        # Annotate weekly peak
        weekly_freq = 1.0 / 7.0
        weekly_idx = int(np.argmin(np.abs(profile.freqs - weekly_freq)))
        fig_freq.add_annotation(
            x=weekly_freq,
            y=profile.power[weekly_idx],
            text="7-day cycle",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#d946ef",
            font=dict(color="#f0abfc", size=11),
            ax=-50, ay=-50,
        )

        # CVS boundary line
        cvs_boundary = 24.0 / 12.0  # 2 cycles/day
        fig_freq.add_vline(
            x=cvs_boundary,
            line_dash="dash",
            line_color="rgba(251,113,133,0.5)",
            annotation_text="CVS threshold (12 h)",
            annotation_font_color="#fb7185",
        )

        fig_freq.update_layout(
            **_PLOTLY_LAYOUT,
            xaxis_title="Frequency (cycles / day)",
            yaxis_title="Power  |F(ω)|²",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig_freq, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# Tab 2 — Spectral Team Matcher
# ───────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("##### 🧩 Optimally Balanced Developer Teams")
    st.caption(
        "Teams are formed by Spectral Clustering on Fourier-derived rhythm "
        "features — magnitude, phase, CVS, and FSR — producing groups with "
        "the most synchronised behavioural frequencies."
    )

    # ── Team cards ──
    n_cols = 4
    sorted_teams = sorted(team_result.teams.items())
    for row_start in range(0, len(sorted_teams), n_cols):
        cols = st.columns(n_cols)
        for i, col in enumerate(cols):
            team_idx = row_start + i
            if team_idx >= len(sorted_teams):
                break
            cluster_id, members = sorted_teams[team_idx]
            members_html = "".join(
                f"<span class='member'>{m}</span>" for m in members
            )
            col.markdown(
                f"<div class='team-card'>"
                f"<h4>Team {cluster_id + 1}</h4>"
                f"{members_html}</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── 2-D PCA Biplot ──
    st.markdown("##### 🗺️ Developer Rhythm Landscape — PCA Biplot")
    st.caption(
        "Scatter shows developer positions in PC space (coloured by team). "
        "Arrows show how each original feature (frequency magnitude, phase, "
        "CVS, FSR) loads onto the principal components — revealing which "
        "behavioural rhythms define each region of the graph."
    )

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    scatter_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Developer": dev_names,
        "Team": [f"Team {l + 1}" for l in team_result.labels],
    })

    n_teams = team_result.n_clusters
    color_map = {
        f"Team {i + 1}": _GRADIENT[i % len(_GRADIENT)]
        for i in range(n_teams)
    }

    fig_biplot = go.Figure()

    # --- Scatter points per team ---
    for team_id in range(n_teams):
        team_label = f"Team {team_id + 1}"
        sub = scatter_df[scatter_df["Team"] == team_label]
        colour = color_map[team_label]
        fig_biplot.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"],
            mode="markers",
            marker=dict(
                size=9,
                color=colour,
                line=dict(width=0.8, color="#1f1f2e"),
                opacity=0.85,
            ),
            name=team_label,
            text=sub["Developer"],
            hovertemplate=(
                "%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}"
                "<extra>" + team_label + "</extra>"
            ),
        ))

    # --- Feature-loading arrows (biplot vectors) ---
    loadings = pca.components_.T  # shape (n_features, 2)
    # Scale arrows so they span roughly the same visual range as the scatter
    scatter_range = max(
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
    )
    loading_max = np.abs(loadings).max()
    arrow_scale = (scatter_range * 0.40) / (loading_max + 1e-9)

    # Highlight the three interpretable metrics with distinct colours
    _ARROW_HIGHLIGHTS = {
        "Cognitive Volatility": "#fb7185",   # rose
        "Flow State (Daily)":  "#34d399",   # emerald
        "Flow State (Weekly)": "#38bdf8",   # sky
    }
    _DEFAULT_ARROW_COLOR = "rgba(209,213,219,0.45)"  # muted grey for mag/phase

    for idx, fname in enumerate(feature_names):
        lx = loadings[idx, 0] * arrow_scale
        ly = loadings[idx, 1] * arrow_scale
        arrow_color = _ARROW_HIGHLIGHTS.get(fname, _DEFAULT_ARROW_COLOR)
        is_key = fname in _ARROW_HIGHLIGHTS

        # Arrow shaft
        fig_biplot.add_trace(go.Scatter(
            x=[0, lx], y=[0, ly],
            mode="lines",
            line=dict(
                color=arrow_color,
                width=2.5 if is_key else 1.2,
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Arrowhead (small triangle via marker)
        fig_biplot.add_trace(go.Scatter(
            x=[lx], y=[ly],
            mode="markers",
            marker=dict(
                symbol="arrow-up",
                size=10 if is_key else 6,
                color=arrow_color,
                angle=np.degrees(np.arctan2(ly, lx)) - 90,
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Text label at arrow tip (only for key features or if few features)
        if is_key or len(feature_names) <= 10:
            fig_biplot.add_annotation(
                x=lx, y=ly,
                text=f"<b>{fname}</b>" if is_key else fname,
                showarrow=False,
                font=dict(
                    color=arrow_color if is_key else "#9ca3af",
                    size=12 if is_key else 9,
                ),
                xshift=12 if lx >= 0 else -12,
                yshift=12 if ly >= 0 else -12,
                xanchor="left" if lx >= 0 else "right",
            )

    ev = pca.explained_variance_ratio_
    fig_biplot.update_layout(
        **_PLOTLY_LAYOUT,
        xaxis_title=f"PC 1  ({ev[0]:.1%} variance)",
        yaxis_title=f"PC 2  ({ev[1]:.1%} variance)",
        height=620,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=10),
            itemsizing="constant",
        ),
    )
    # Equal aspect ratio so arrows aren't distorted
    fig_biplot.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_biplot, use_container_width=True)

    st.caption(
        f"PCA explained variance: PC1 = {ev[0]:.1%}, PC2 = {ev[1]:.1%} "
        f"(total = {ev.sum():.1%}).  "
        "Coloured arrows = key behavioural metrics; grey arrows = frequency magnitudes & phases."
    )

# ───────────────────────────────────────────────────────────────────────────
# Tab 3 — Team Synchrony Radar Chart
# ───────────────────────────────────────────────────────────────────────────

_RADAR_COLORS = [
    ("#6366f1", "rgba(99,102,241,0.25)"),    # indigo
    ("#34d399", "rgba(52,211,153,0.25)"),     # emerald
    ("#f472b6", "rgba(244,114,182,0.25)"),    # pink
    ("#facc15", "rgba(250,204,21,0.25)"),     # amber
]

with tab3:
    st.markdown("##### 🕸️ Team Synchrony — Radar Chart")
    st.caption(
        "Select a team to visualise how the 4 members complement each other "
        "across five behavioural dimensions.  Semi-transparent fills highlight "
        "overlapping strengths and the gaps each teammate fills."
    )

    team_options = {f"Team {cid + 1}": members
                    for cid, members in sorted(team_result.teams.items())}
    selected_team = st.selectbox(
        "Select a team",
        list(team_options.keys()),
        index=0,
        key="radar_team_selector",
    )
    members = team_options[selected_team]

    # Precompute global min/max baselines so features are scaled across all developers
    @st.cache_data(show_spinner="📊 Computing global radar metrics…")
    def _compute_global_radar(_signals):
        data = {dev: build_radar_profile(dev, sig) for dev, sig in _signals.items()}
        df = pd.DataFrame(data).T
        
        # Min/Max scaling to 0-100 range
        min_vals = df.min()
        max_vals = df.max()
        rng = max_vals - min_vals
        rng[rng == 0] = 1.0  # prevent division by zero
        
        scaled_df = (df - min_vals) / rng * 100.0
        return df, scaled_df

    global_raw, global_scaled = _compute_global_radar(signals)
    categories = list(global_raw.columns)

    fig_radar = go.Figure()
    for i, member in enumerate(members):
        vals = global_scaled.loc[member].tolist()
        vals.append(vals[0])  # close the polygon
        line_color, fill_color = _RADAR_COLORS[i % len(_RADAR_COLORS)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            name=member,
            opacity=0.85,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                gridcolor="rgba(255,255,255,0.06)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.08)",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d1d5db"),
        height=520,
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Raw-values table underneath ──
    st.markdown("###### Raw Metric Values")
    raw_df = global_raw.loc[members].copy()
    raw_df.index.name = "Developer"
    raw_df["Commit Volume"] = raw_df["Commit Volume"].astype(int)
    for col in ["Cognitive Volatility", "Weekend Activity", "Night-Owl Index"]:
        raw_df[col] = raw_df[col].map("{:.3f}".format)
    raw_df["Flow State Resonance"] = raw_df["Flow State Resonance"].map("{:.1f}".format)
    st.dataframe(raw_df, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# Tab 4 — Burnout Visualizer (Time-Domain Contrast Plot)
# ───────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("##### 🔥 Burnout Visualizer — Time-Domain Contrast")
    st.caption(
        "Side-by-side comparison of the *most rhythmic* developer (low CVS, "
        "high FSR — the 'Flow State' archetype) vs. the *most volatile* "
        "developer (high CVS — the 'Burned-Out' archetype).  "
        "Zoom into any 4-week window to see the contrast."
    )

    # Identify the two archetypes
    @st.cache_data(show_spinner="🔍 Ranking developers…")
    def _rank(_signals):
        return rank_developers_by_cvs(_signals)

    ranked = _rank(signals)
    flow_dev, flow_cvs = ranked[0]       # lowest CVS = most rhythmic
    burn_dev, burn_cvs = ranked[-1]      # highest CVS = most volatile

    # 4-week window selector
    total_hours = len(signals[flow_dev])
    total_days = total_hours // 24
    window_days = 28  # 4 weeks
    max_start = max(0, total_days - window_days)
    start_day = st.slider(
        "Window start (day)",
        min_value=0,
        max_value=max_start,
        value=0,
        step=7,
        key="burnout_slider",
    )
    h_start = start_day * 24
    h_end = min(h_start + window_days * 24, total_hours)

    flow_signal = signals[flow_dev][h_start:h_end]
    burn_signal = signals[burn_dev][h_start:h_end]
    x_hours = np.arange(len(flow_signal))
    x_days = x_hours / 24.0 + start_day

    # Metric cards
    mc1, mc2 = st.columns(2)
    mc1.markdown(
        f"<div style='background:rgba(52,211,153,0.08);border:1px solid "
        f"rgba(52,211,153,0.25);border-radius:12px;padding:16px;'>"
        f"<span style='color:#9ca3af;font-size:0.75rem;text-transform:uppercase;"
        f"letter-spacing:0.05em;'>Flow State Developer</span><br>"
        f"<span style='color:#34d399;font-size:1.3rem;font-weight:700;'>{flow_dev}</span>"
        f"<br><span style='color:#6b7280;font-size:0.85rem;'>CVS = {flow_cvs:.4f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    mc2.markdown(
        f"<div style='background:rgba(239,68,68,0.08);border:1px solid "
        f"rgba(239,68,68,0.25);border-radius:12px;padding:16px;'>"
        f"<span style='color:#9ca3af;font-size:0.75rem;text-transform:uppercase;"
        f"letter-spacing:0.05em;'>Volatile / Burned-Out Developer</span><br>"
        f"<span style='color:#ef4444;font-size:1.3rem;font-weight:700;'>{burn_dev}</span>"
        f"<br><span style='color:#6b7280;font-size:0.85rem;'>CVS = {burn_cvs:.4f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Stacked subplots ──
    fig_contrast = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"🟢 Flow State — {flow_dev}  (CVS = {flow_cvs:.4f})",
            f"🔴 Volatile — {burn_dev}  (CVS = {burn_cvs:.4f})",
        ),
    )

    # Top: Flow State developer (calm green / blue)
    fig_contrast.add_trace(
        go.Scatter(
            x=x_days, y=flow_signal,
            mode="lines",
            line=dict(color="#34d399", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(52,211,153,0.08)",
            name=flow_dev,
            hovertemplate="Day %{x:.1f}<br>Commits: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Bottom: Volatile developer (alert red / orange)
    fig_contrast.add_trace(
        go.Scatter(
            x=x_days, y=burn_signal,
            mode="lines",
            line=dict(color="#ef4444", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)",
            name=burn_dev,
            hovertemplate="Day %{x:.1f}<br>Commits: %{y}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── Annotations ──
    # Mark 24-hour cycle peaks in the flow developer (first few days)
    for day_offset in range(min(3, window_days)):
        peak_x = start_day + day_offset + 0.5  # approximate midday peak
        if day_offset == 0:
            fig_contrast.add_annotation(
                x=peak_x, y=flow_signal[int(day_offset * 24 + 12)] if len(flow_signal) > day_offset * 24 + 12 else 0,
                text="↓ 24-hour cycle",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#34d399",
                font=dict(color="#34d399", size=11),
                ax=40, ay=-35,
                row=1, col=1,
            )

    # Mark high-frequency noise in the volatile developer
    # Find the largest spike in the first week
    spike_window = burn_signal[:min(168, len(burn_signal))]
    if len(spike_window) > 0:
        spike_idx = int(np.argmax(spike_window))
        fig_contrast.add_annotation(
            x=x_days[spike_idx],
            y=float(spike_window[spike_idx]),
            text="⚡ High-freq noise",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ef4444",
            font=dict(color="#ef4444", size=11),
            ax=-50, ay=-30,
            row=2, col=1,
        )

    # Find a late-night commit in the volatile developer
    for h in range(len(burn_signal)):
        hour_of_day = (h_start + h) % 24
        if hour_of_day >= 2 and hour_of_day <= 4 and burn_signal[h] > 0:
            fig_contrast.add_annotation(
                x=x_days[h],
                y=float(burn_signal[h]),
                text="🌙 Late-night commit",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#fbbf24",
                font=dict(color="#fbbf24", size=10),
                ax=50, ay=-25,
                row=2, col=1,
            )
            break  # only annotate the first one

    fig_contrast.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d1d5db"),
        height=600,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False,
    )
    fig_contrast.update_xaxes(
        title_text="Day", row=2, col=1,
        gridcolor="rgba(255,255,255,0.04)",
    )
    fig_contrast.update_xaxes(gridcolor="rgba(255,255,255,0.04)", row=1, col=1)
    fig_contrast.update_yaxes(
        title_text="Commits / hour",
        gridcolor="rgba(255,255,255,0.04)",
    )
    # Colour the subplot titles
    fig_contrast.layout.annotations[0].font.color = "#34d399"  # top
    fig_contrast.layout.annotations[1].font.color = "#ef4444"  # bottom

    st.plotly_chart(fig_contrast, use_container_width=True)

    st.caption(
        "The steady, wave-like pattern of the Flow State developer reflects a "
        "strong 24-hour circadian rhythm (high FSR, low CVS).  The volatile "
        "developer shows erratic spikes and late-night activity — high-frequency "
        "noise that our Fourier analysis correctly flags as high Cognitive Volatility."
    )

# ───────────────────────────────────────────────────────────────────────────
# Tab 5 — Macro-Sabermetrics (Alpha Generation)
# ───────────────────────────────────────────────────────────────────────────

with tab5:
    st.markdown("##### 📈 Macro-Sabermetrics — Alpha Signal Testing")
    st.caption(
        "Does extreme engineering volatility precede stock price movements? "
        "Here we apply our FFT pipeline to the entire Microsoft organisation's "
        "commit volume via an expanding 30-day rolling window."
    )
    
    # ── Fetch and Process Macro Data ──
    @st.cache_data(show_spinner="⏳ Fetching 3-year Macro Data & computing signals...")
    def _load_macro_alpha():
        # NOTE: Defaulting to use_bigquery=False so this can be demoed without GCP credits.
        df = fetch_macro_data(org_pattern="microsoft", ticker="MSFT", years=3, use_bigquery=False)
        # Compute the 30-day rolling organisational volatility score
        df["Volatility_Score"] = compute_rolling_macro_volatility(df["pushes"], window=30)
        # Compute 7-day lagged correlation between volatility and stock price
        # Lag definition: Volatility today -> Stock price in 7 days.
        # So we correlate Volatility with Close shifted backwards by 7 days.
        corr_series = df["Volatility_Score"].corr(df["Close"].shift(-7))
        return df, corr_series
        
    macro_df, pearson_corr = _load_macro_alpha()

    # Create the dual-axis chart
    fig_macro = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Primary Y-axis: Volatility (Red)
    fig_macro.add_trace(
        go.Scatter(
            x=macro_df.index,
            y=macro_df["Volatility_Score"],
            mode="lines",
            line=dict(color="#ef4444", width=2),
            name="Org Volatility Score (30d)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Volatility: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    
    # Secondary Y-axis: MSFT Stock (Blue)
    fig_macro.add_trace(
        go.Scatter(
            x=macro_df.index,
            y=macro_df["Close"],
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="MSFT Close Price",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    
    # Event Markers: January 2023 and May 2024
    # Approximate dates; we'll pick mid-month for visualization clarity
    fig_macro.add_vline(
        x=pd.Timestamp("2023-01-18").timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="10k Layoffs Announced",
        annotation_position="top left",
        annotation_font=dict(color="#f59e0b", size=10),
    )
    
    fig_macro.add_vline(
        x=pd.Timestamp("2024-05-15").timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="6k Layoffs Announced",
        annotation_position="top left",
        annotation_font=dict(color="#f59e0b", size=10),
    )
    
    fig_macro.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d1d5db"),
        height=550,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
        ),
        hovermode="x unified",
    )
    
    fig_macro.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        showgrid=True,
    )
    fig_macro.update_yaxes(
        title_text="Org Volatility (High-Freq Spectral Energy)", 
        secondary_y=False,
        gridcolor="rgba(255,255,255,0.04)",
    )
    fig_macro.update_yaxes(
        title_text="MSFT Stock Price ($)", 
        secondary_y=True,
        showgrid=False,
    )
    
    st.plotly_chart(fig_macro, use_container_width=True)

    # ── Signal Card ──
    st.markdown("###### Alpha Generation Signal (Pearson Correlation)")
    card_colour = "#10b981" if pearson_corr < 0 else "#ef4444" # Negative correlation implies high volatility predicts low returns
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.03);border:1px solid "
        f"rgba(255,255,255,0.08);border-radius:12px;padding:24px;text-align:center;'>"
        f"<span style='color:#9ca3af;font-size:0.85rem;text-transform:uppercase;"
        f"letter-spacing:0.05em;'>7-Day Lagged Volatility ↔ Stock Correlation</span><br>"
        f"<span style='color:{card_colour};font-size:2.5rem;font-weight:700;'>{pearson_corr:+.3f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )    
