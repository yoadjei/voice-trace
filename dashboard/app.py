# VoiceTrace — Injury Surveillance Research Dashboard
# Run: streamlit run dashboard/app.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VoiceTrace",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_SYN = Path("data/synthetic")

LANGUAGES = {
    "Twi": "pipeline_results.csv",
    "Fante": "pipeline_results_fante.csv",
    "Ewe": "pipeline_results_ewe.csv",
    "Ga": "pipeline_results_ga.csv",
    "Dagbani": "pipeline_results_dagbani.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# Research Observatory Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --void: #13161C;
    --surface-1: #1A1E26;
    --surface-2: #21262F;
    --surface-3: #2A303B;
    --surface-4: #343B48;
    --text-1: #EAEDF3;
    --text-2: #A8B1C0;
    --text-3: #6B7385;
    --cyan: #22D3EE;
    --cyan-dim: #0E7490;
    --amber: #FBBF24;
    --rose: #FB7185;
    --emerald: #34D399;
    --violet: #A78BFA;
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--text-1);
    font-size: 14px;
}

.stApp {
    background: var(--void);
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(34, 211, 238, 0.08), transparent),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(167, 139, 250, 0.05), transparent);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

.block-container {
    padding: 1.5rem 1.5rem 3rem 1.5rem;
    max-width: 1800px;
}

/* === HEADER === */
.observatory-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 0 1.5rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--surface-3);
}
.brand-cluster {
    display: flex;
    align-items: center;
    gap: 0;
}
.brand-text {
    display: flex;
    flex-direction: column;
}
.brand-name {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-1);
    letter-spacing: -0.02em;
}
.brand-tag {
    font-size: 0.7rem;
    color: var(--text-3);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.header-meta {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.live-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: var(--text-3);
}
.live-dot {
    width: 6px;
    height: 6px;
    background: var(--emerald);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--emerald);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.9); }
}
.timestamp {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-3);
}

/* === STATS STRIP === */
.stats-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: var(--surface-3);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
}
@media (max-width: 1024px) {
    .stats-strip { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 640px) {
    .stats-strip { grid-template-columns: repeat(2, 1fr); }
}
.stat-cell {
    background: var(--surface-1);
    padding: 1.25rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.stat-cell:hover { background: var(--surface-2); }
.stat-label {
    font-size: 0.65rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stat-row {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: var(--text-1);
    line-height: 1;
}
.stat-unit {
    font-size: 0.75rem;
    color: var(--text-3);
}
.stat-trend {
    font-size: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}
.stat-trend.up { color: var(--emerald); }
.stat-trend.down { color: var(--rose); }
.stat-cell.accent-cyan .stat-value { color: var(--cyan); }
.stat-cell.accent-amber .stat-value { color: var(--amber); }
.stat-cell.accent-rose .stat-value { color: var(--rose); }
.stat-cell.accent-emerald .stat-value { color: var(--emerald); }

/* === SECTION === */
.section-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.section-action {
    font-size: 0.7rem;
    color: var(--cyan-dim);
    cursor: pointer;
}
.section-action:hover { color: var(--cyan); }

/* === CARDS === */
.card {
    background: var(--surface-1);
    border: 1px solid var(--surface-3);
    border-radius: 12px;
    overflow: hidden;
}
.card-body { padding: 1rem; }
.card-header {
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--surface-3);
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-2);
}

/* === LANGUAGE BAR === */
.lang-visual {
    margin-bottom: 2rem;
}
.lang-bar-track {
    height: 6px;
    background: var(--surface-3);
    border-radius: 3px;
    overflow: hidden;
    display: flex;
    margin-bottom: 0.75rem;
}
.lang-bar-fill {
    height: 100%;
    transition: width 0.5s ease;
}
.lang-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}
.lang-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--text-2);
}
.lang-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
}
.lang-item span:last-child {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-3);
    font-size: 0.7rem;
}

/* === INCIDENT FEED === */
.feed {
    display: flex;
    flex-direction: column;
}
.feed-item {
    padding: 1rem;
    border-bottom: 1px solid var(--surface-3);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    cursor: pointer;
    transition: background 0.1s;
}
.feed-item:hover { background: var(--surface-2); }
.feed-item:last-child { border-bottom: none; }
.feed-main { flex: 1; }
.feed-type {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-1);
    margin-bottom: 0.25rem;
}
.feed-meta {
    font-size: 0.75rem;
    color: var(--text-3);
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.feed-meta span { display: flex; align-items: center; gap: 0.25rem; }
.severity-tag {
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    white-space: nowrap;
}
.severity-tag.severe { background: rgba(251, 113, 133, 0.15); color: var(--rose); }
.severity-tag.moderate { background: rgba(251, 191, 36, 0.15); color: var(--amber); }
.severity-tag.mild { background: rgba(52, 211, 153, 0.15); color: var(--emerald); }

/* === DETAIL PANEL === */
.detail-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0;
}
.detail-cell {
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--surface-3);
    border-right: 1px solid var(--surface-3);
}
.detail-cell:nth-child(2n) { border-right: none; }
.detail-cell:nth-last-child(-n+2) { border-bottom: none; }
.detail-key {
    font-size: 0.6rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.detail-val {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-1);
}

/* === FIRST AID === */
.aid-panel {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.08) 0%, rgba(167, 139, 250, 0.05) 100%);
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}
.aid-title {
    font-size: 0.6rem;
    font-weight: 600;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.aid-text {
    font-size: 0.85rem;
    color: var(--text-2);
    line-height: 1.6;
}

/* === FACILITY === */
.facility-row {
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--surface-3);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.facility-row:last-child { border-bottom: none; }
.facility-name {
    font-size: 0.85rem;
    color: var(--text-1);
}
.facility-count {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--cyan);
}

/* === MAP === */
.map-wrap {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--surface-3);
}

/* === STREAMLIT OVERRIDES === */
.stSelectbox > div > div {
    background: var(--surface-2);
    border-color: var(--surface-3);
    border-radius: 8px;
    font-size: 0.85rem;
}
.stSelectbox label, .stMultiSelect label {
    font-size: 0.65rem !important;
    color: var(--text-3) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stMultiSelect > div > div {
    background: var(--surface-2);
    border-color: var(--surface-3);
    border-radius: 8px;
}
div[data-baseweb="select"] > div {
    background: var(--surface-2);
}

/* === FOOTER === */
.observatory-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--surface-3);
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--text-3);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading + Mock Enhancement
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_all_data():
    all_dfs = []
    for lang, filename in LANGUAGES.items():
        path = DATA_SYN / filename
        if path.exists():
            df = pd.read_csv(path)
            df["language"] = lang
            all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Add mock timestamps for realism
        base_date = datetime.now() - timedelta(days=30)
        combined["timestamp"] = [base_date + timedelta(hours=random.randint(0, 720)) for _ in range(len(combined))]
        return combined
    return None

df = load_all_data()
if df is None:
    st.error("No data found")
    st.stop()

# Mock metrics for dashboard realism
MOCK_METRICS = {
    "calls_today": 47,
    "calls_trend": "+12%",
    "avg_response_ms": 2340,
    "extraction_f1": 0.89,
    "geocode_rate": 0.78,
    "active_regions": 8
}

LANG_COLORS = {
    "Twi": "#22D3EE",
    "Fante": "#A78BFA",
    "Ewe": "#FBBF24",
    "Ga": "#FB7185",
    "Dagbani": "#34D399"
}

INJURY_LABELS = {
    "rta": "Road Traffic Accident",
    "fall": "Fall",
    "assault": "Assault",
    "burn": "Burn",
    "cut": "Cut / Laceration",
    "occupational": "Occupational",
    "sports": "Sports Injury",
    "drowning": "Drowning",
    "other": "Other",
    "unknown": "Unclassified"
}

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

now = datetime.now()

st.markdown(f'''
<div class="observatory-header">
    <div class="brand-cluster">
        <div class="brand-text">
            <span class="brand-name">VoiceTrace</span>
            <span class="brand-tag">Injury Surveillance Research</span>
        </div>
    </div>
    <div class="header-meta">
        <div class="live-indicator">
            <span class="live-dot"></span>
            <span>Live</span>
        </div>
        <span class="timestamp">{now.strftime("%Y-%m-%d")} · {now.strftime("%H:%M")} UTC</span>
    </div>
</div>
''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────

fcol1, fcol2, fcol3, fcol4 = st.columns([1,1,1,2])

with fcol1:
    lang_opts = ["All"] + list(df["language"].unique())
    sel_lang = st.selectbox("Language", lang_opts)

with fcol2:
    type_opts = ["All"] + sorted([t for t in df["injury_type"].dropna().unique() if t != "unknown"])
    sel_type = st.selectbox("Type", type_opts)

with fcol3:
    sev_opts = [s for s in df["severity"].dropna().unique() if s != "unknown"]
    sel_sev = st.multiselect("Severity", sev_opts, default=sev_opts)

filtered = df.copy()
if sel_lang != "All":
    filtered = filtered[filtered["language"] == sel_lang]
if sel_type != "All":
    filtered = filtered[filtered["injury_type"] == sel_type]
if sel_sev:
    filtered = filtered[filtered["severity"].isin(sel_sev)]

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Stats Strip
# ─────────────────────────────────────────────────────────────────────────────

total = len(filtered)
severe = len(filtered[filtered["severity"] == "severe"])
rta = len(filtered[filtered["injury_type"] == "rta"])
geocoded = len(filtered[filtered["lat"].notna()])
geo_pct = int(geocoded / total * 100) if total else 0

st.markdown(f'''
<div class="stats-strip">
    <div class="stat-cell">
        <span class="stat-label">Incidents</span>
        <div class="stat-row">
            <span class="stat-value">{total}</span>
        </div>
        <span class="stat-trend up">↑ {MOCK_METRICS["calls_trend"]} vs last week</span>
    </div>
    <div class="stat-cell accent-rose">
        <span class="stat-label">Severe</span>
        <div class="stat-row">
            <span class="stat-value">{severe}</span>
        </div>
    </div>
    <div class="stat-cell accent-amber">
        <span class="stat-label">Road Traffic</span>
        <div class="stat-row">
            <span class="stat-value">{rta}</span>
        </div>
    </div>
    <div class="stat-cell accent-cyan">
        <span class="stat-label">Geocoded</span>
        <div class="stat-row">
            <span class="stat-value">{geo_pct}</span>
            <span class="stat-unit">%</span>
        </div>
    </div>
    <div class="stat-cell">
        <span class="stat-label">Avg Latency</span>
        <div class="stat-row">
            <span class="stat-value">{MOCK_METRICS["avg_response_ms"]}</span>
            <span class="stat-unit">ms</span>
        </div>
    </div>
    <div class="stat-cell accent-emerald">
        <span class="stat-label">Extraction F1</span>
        <div class="stat-row">
            <span class="stat-value">{MOCK_METRICS["extraction_f1"]}</span>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Language Distribution
# ─────────────────────────────────────────────────────────────────────────────

lang_counts = filtered["language"].value_counts()
lang_total = lang_counts.sum()

bar_html = '<div class="lang-bar-track">'
for lang in LANG_COLORS:
    if lang in lang_counts:
        pct = lang_counts[lang] / lang_total * 100
        bar_html += f'<div class="lang-bar-fill" style="width:{pct}%;background:{LANG_COLORS[lang]}"></div>'
bar_html += '</div>'

legend_html = '<div class="lang-legend">'
for lang, color in LANG_COLORS.items():
    count = lang_counts.get(lang, 0)
    legend_html += f'<div class="lang-item"><div class="lang-dot" style="background:{color}"></div>{lang}<span>{count}</span></div>'
legend_html += '</div>'

st.markdown(f'<div class="lang-visual">{bar_html}{legend_html}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout: Map + Feed + Charts
# ─────────────────────────────────────────────────────────────────────────────

col_map, col_feed = st.columns([2, 1])

with col_map:
    st.markdown('<div class="section-head"><span class="section-title">Incident Map</span></div>', unsafe_allow_html=True)
    
    geo = filtered[filtered["lat"].notna() & filtered["lng"].notna()].copy()
    
    if not geo.empty:
        sev_colors = {"severe": "#FB7185", "moderate": "#FBBF24", "mild": "#34D399", "unknown": "#6B7385"}
        
        fig = px.scatter_mapbox(
            geo, lat="lat", lon="lng",
            color="severity",
            color_discrete_map=sev_colors,
            hover_data={"injury_type": True, "language": True, "lat": False, "lng": False},
            zoom=5.5, center={"lat": 7.5, "lon": -1.5},
            height=420
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#1A1E26",
            legend=dict(
                orientation="h", yanchor="top", y=0.98, xanchor="left", x=0.02,
                bgcolor="rgba(26,30,38,0.9)", font=dict(color="#A8B1C0", size=10),
                borderwidth=0
            )
        )
        fig.update_traces(marker=dict(size=9, opacity=0.85))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geocoded incidents")

with col_feed:
    st.markdown('<div class="section-head"><span class="section-title">Recent Incidents</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><div class="feed">', unsafe_allow_html=True)
    
    recent = filtered.sort_values("timestamp", ascending=False).head(7)
    for _, row in recent.iterrows():
        itype = INJURY_LABELS.get(row.get("injury_type", ""), row.get("injury_type", "Unknown"))
        sev = row.get("severity", "unknown")
        sev_cls = sev if sev in ["severe", "moderate", "mild"] else ""
        loc = row.get("location_description", "—")
        lang = row.get("language", "")
        ts = row.get("timestamp")
        ts_str = ts.strftime("%H:%M") if pd.notna(ts) else ""
        
        st.markdown(f'''
        <div class="feed-item">
            <div class="feed-main">
                <div class="feed-type">{itype}</div>
                <div class="feed-meta">
                    <span>{loc}</span>
                    <span>·</span>
                    <span>{lang}</span>
                    <span>·</span>
                    <span>{ts_str}</span>
                </div>
            </div>
            <span class="severity-tag {sev_cls}">{sev}</span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Charts Row
# ─────────────────────────────────────────────────────────────────────────────

ch1, ch2, ch3 = st.columns(3)

with ch1:
    st.markdown('<div class="card"><div class="card-header">Injury Type</div><div class="card-body">', unsafe_allow_html=True)
    
    tc = filtered["injury_type"].value_counts().reset_index()
    tc.columns = ["Type", "Count"]
    tc["Label"] = tc["Type"].map(INJURY_LABELS).fillna(tc["Type"])
    
    fig_t = px.bar(tc, x="Count", y="Label", orientation="h", color_discrete_sequence=["#22D3EE"])
    fig_t.update_layout(
        plot_bgcolor="#1A1E26", paper_bgcolor="#1A1E26",
        font=dict(family="IBM Plex Sans", size=10, color="#A8B1C0"),
        margin=dict(l=0, r=10, t=0, b=0), showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#2A303B", zeroline=False, tickfont=dict(size=9, color="#6B7385")),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#A8B1C0")),
        height=200
    )
    st.plotly_chart(fig_t, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with ch2:
    st.markdown('<div class="card"><div class="card-header">Severity</div><div class="card-body">', unsafe_allow_html=True)
    
    sc = filtered["severity"].value_counts().reset_index()
    sc.columns = ["Severity", "Count"]
    sev_map = {"severe": "#FB7185", "moderate": "#FBBF24", "mild": "#34D399", "unknown": "#6B7385"}
    
    fig_s = px.pie(sc, values="Count", names="Severity", color="Severity", color_discrete_map=sev_map, hole=0.6)
    fig_s.update_layout(
        plot_bgcolor="#1A1E26", paper_bgcolor="#1A1E26",
        font=dict(family="IBM Plex Sans", size=10, color="#A8B1C0"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=9)),
        height=200
    )
    fig_s.update_traces(textposition='inside', textinfo='percent', textfont=dict(size=10, color="#EAEDF3"))
    st.plotly_chart(fig_s, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with ch3:
    st.markdown('<div class="card"><div class="card-header">Body Region</div><div class="card-body">', unsafe_allow_html=True)
    
    bc = filtered["body_region"].value_counts().head(6).reset_index()
    bc.columns = ["Region", "Count"]
    
    fig_b = px.bar(bc, x="Count", y="Region", orientation="h", color_discrete_sequence=["#A78BFA"])
    fig_b.update_layout(
        plot_bgcolor="#1A1E26", paper_bgcolor="#1A1E26",
        font=dict(family="IBM Plex Sans", size=10, color="#A8B1C0"),
        margin=dict(l=0, r=10, t=0, b=0), showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#2A303B", zeroline=False, tickfont=dict(size=9, color="#6B7385")),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#A8B1C0")),
        height=200
    )
    st.plotly_chart(fig_b, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Detail + Facilities
# ─────────────────────────────────────────────────────────────────────────────

det1, det2 = st.columns([1, 1])

with det1:
    st.markdown('<div class="section-head"><span class="section-title">Incident Detail</span></div>', unsafe_allow_html=True)
    
    ids = filtered["id"].tolist()
    if ids:
        sel_id = st.selectbox("Select", ids, label_visibility="collapsed")
        rec = filtered[filtered["id"] == sel_id].iloc[0]
        
        st.markdown(f'''
        <div class="card">
            <div class="detail-grid">
                <div class="detail-cell">
                    <div class="detail-key">Type</div>
                    <div class="detail-val">{INJURY_LABELS.get(rec.get("injury_type",""), rec.get("injury_type","—"))}</div>
                </div>
                <div class="detail-cell">
                    <div class="detail-key">Severity</div>
                    <div class="detail-val">{rec.get("severity","—").title()}</div>
                </div>
                <div class="detail-cell">
                    <div class="detail-key">Body Region</div>
                    <div class="detail-val">{rec.get("body_region","—").replace("_"," ").title()}</div>
                </div>
                <div class="detail-cell">
                    <div class="detail-key">Victim</div>
                    <div class="detail-val">{rec.get("victim_sex","—").title()}, {rec.get("victim_age_group","—")}</div>
                </div>
                <div class="detail-cell">
                    <div class="detail-key">Location</div>
                    <div class="detail-val">{rec.get("location_description","—")}</div>
                </div>
                <div class="detail-cell">
                    <div class="detail-key">Language</div>
                    <div class="detail-val">{rec.get("language","—")}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        aid = rec.get("first_aid")
        if pd.notna(aid) and aid:
            st.markdown(f'''
            <div class="aid-panel">
                <div class="aid-title">First Aid Guidance</div>
                <div class="aid-text">{aid}</div>
            </div>
            ''', unsafe_allow_html=True)

with det2:
    st.markdown('<div class="section-head"><span class="section-title">Facility Referrals</span></div>', unsafe_allow_html=True)
    
    fac = filtered[filtered["facility_name"].notna()]["facility_name"].value_counts().head(6)
    
    if not fac.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for name, cnt in fac.items():
            st.markdown(f'''
            <div class="facility-row">
                <span class="facility-name">{name}</span>
                <span class="facility-count">{cnt}</span>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f'''
<div class="observatory-footer">
    <span>VoiceTrace · KNUST Research</span>
    <span>v1.0</span>
</div>
''', unsafe_allow_html=True)
