import numpy as np
import streamlit as st
from datetime import date, datetime, timedelta, timezone
from streamlit_folium import st_folium
import folium

from services.analytics.temporal import detect_temporal_anomalies
from services.analytics.fusion import calculate_emission_score
from services.analytics.clustering import cluster_anomalies
from services.gee_sources.sentinel5p import fetch_all_pollutants, build_pollutant_tile_layers
from services.visualization.folium_map import create_map

# ---------------------------------------------------------------------------
# Cached analysis pipeline  (keyed on lat / lon / ISO dates)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _run_analysis(lat, lon, start_str, end_str):
    """Execute the full pipeline once and cache the results.

    Parameters are flat / hashable types so ``st.cache_data`` can
    invalidate cleanly when the user changes the location or time
    window.
    """
    s = date.fromisoformat(start_str)
    e = date.fromisoformat(end_str)

    lst_array, anomaly_indices, z_score_map, tiles, ndvi_mean, ndbi_mean = (
        detect_temporal_anomalies(lat, lon, s, e)
    )
    s5p_data = fetch_all_pollutants(lat, lon, s, e)
    pollutant_tiles = build_pollutant_tile_layers(lat, lon, s, e)
    clusters, _ = cluster_anomalies(lst_array, anomaly_indices, z_score_map, lat, lon)

    return {
        "lst_array": lst_array,
        "anomaly_indices": anomaly_indices,
        "z_score_map": z_score_map,
        "ndvi_mean": ndvi_mean,
        "ndbi_mean": ndbi_mean,
        "s5p_data": s5p_data,
        "tile_layers": tiles + pollutant_tiles,
        "clusters": clusters,
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "select_on_map" not in st.session_state:
    st.session_state.select_on_map = False
if "picker_click" not in st.session_state:
    st.session_state.picker_click = None
if "picker_coords" not in st.session_state:
    st.session_state.picker_coords = {"lat": 12.9235, "lon": 77.4986}


st.set_page_config(
    page_title="Factory Emissions Visualizer",
    page_icon="globe",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - Environmental Intelligence Terminal aesthetic
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    * {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main canvas */
    [data-testid="stAppViewContainer"] {
        background-color: #1a1a18;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f0f0e;
        position: relative;
    }
    
    .stMarkdown, [class*="css"] {
        color: #e8e6e1;
    }
    
    /* Input fields and containers */
    [data-testid="stNumberInput"] input,
    input[type="number"] {
        background-color: #242420 !important;
        color: #e8e6e1 !important;
        border: 1px solid #2e3d2e !important;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    
    [data-testid="stNumberInput"] input::placeholder,
    input[type="number"]::placeholder {
        color: #5a6a5a;
    }
    
    /* Metrics - scientific readout style */
    [data-testid="stMetric"] {
        background: rgba(24, 24, 20, 0.6);
        padding: 24px;
        border-radius: 4px;
        border: 1px solid #2e3d2e;
        box-shadow: none;
        backdrop-filter: blur(2px);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 10px;
        color: #8a9a8a;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        color: #6b8f5e;
        font-weight: 600;
        margin-top: 10px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Buttons - precision instrument style */
    .stButton > button {
        padding: 10px 24px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 4px;
        border: 1px solid #6b8f5e;
        transition: all 0.2s ease;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        background-color: #242420;
        color: #6b8f5e;
    }
    
    .stButton > button:first-child {
        background-color: #6b8f5e;
        color: #0f0f0e;
        border: 1px solid #6b8f5e;
    }
    
    .stButton > button:first-child:hover {
        background-color: #0f0f0e;
        color: #6b8f5e;
        border: 1px solid #6b8f5e;
        box-shadow: none;
    }
    
    .stButton > button:not(:first-child):hover {
        background-color: #6b8f5e;
        color: #0f0f0e;
        border-color: #6b8f5e;
    }
    
    /* Decorative SVG elements */
    .decorative-svg {
        pointer-events: none;
        display: block;
        width: 100%;
    }
    
    .topo-lines {
        opacity: 0.12;
        margin: -200px 0 -100px 0;
        height: 300px;
    }
    
    .dot-grid {
        opacity: 0.08;
        margin: -450px 0 -50px 0;
        height: 500px;
    }
    
    .orbit-arc {
        opacity: 0.1;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 300px;
        pointer-events: none;
    }
    
    /* Custom Loading State - Minimal Terrain-Inspired Loader */
    /* Modal backdrop overlay */
    [data-testid="stSpinner"] {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 10000 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 24px !important;
        background: rgba(15, 15, 14, 0.98) !important;
        border: 1px solid #2e3d2e !important;
        border-radius: 8px !important;
        padding: 40px 24px !important;
        width: auto !important;
        max-width: 340px !important;
        box-shadow: 0 25px 80px rgba(0, 0, 0, 0.9) !important;
        animation: modalSlideIn 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
    }
    
    @keyframes modalSlideIn {
        from {
            opacity: 0;
            transform: translate(-50%, -48%) scale(0.92);
        }
        to {
            opacity: 1;
            transform: translate(-50%, -50%) scale(1);
        }
    }
    
    /* Create dark overlay when spinner is visible */
    html body {
        transition: filter 0.3s ease;
    }
    
    /* Target the main content container when spinner exists */
    main {
        transition: filter 0.3s ease;
    }
    
    /* Use a wrapper div approach - add overlay via CSS grid/absolute positioning parent */
    [data-testid="stAppViewContainer"] {
        position: relative;
    }
    
    /* Shadow overlay that appears with spinner */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(5px);
        z-index: 9999;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease, visibility 0.3s ease;
        pointer-events: auto;
    }
    
    /* Show overlay when spinner is present */
    [data-testid="stAppViewContainer"]:has([data-testid="stSpinner"])::before {
        opacity: 1;
        visibility: visible;
    }


    
    
    /* Hide default spinner elements and text */
    [data-testid="stSpinner"] svg,
    [data-testid="stSpinner"] > div {
        display: none !important;
    }
    
    /* Animated expanding contours loader */
    [data-testid="stSpinner"]::before {
        content: '';
        width: 100px;
        height: 100px;
        background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="8" fill="none" stroke="%236b8f5e" stroke-width="1.2"/><circle cx="50" cy="50" r="28" fill="none" stroke="%236b8f5e" stroke-width="0.8" opacity="0.7"/><circle cx="50" cy="50" r="48" fill="none" stroke="%236b8f5e" stroke-width="0.6" opacity="0.4"/></svg>');
        background-size: 100%;
        background-repeat: no-repeat;
        background-position: center;
        animation: pulseContours 3s ease-in-out infinite;
    }
    
    @keyframes pulseContours {
        0% {
            transform: scale(0.9);
            opacity: 1;
        }
        50% {
            transform: scale(1);
            opacity: 1;
        }
        100% {
            transform: scale(1.3);
            opacity: 0.2;
        }
    }
    
    /* Cycling status text */
    [data-testid="stSpinner"]::after {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #8a9a8a;
        letter-spacing: 0.08em;
        text-align: center;
        height: 36px;
        display: flex;
        align-items: center;
        animation: textCycle 6s steps(3, end) infinite;
        min-width: 240px;
    }
    
    @keyframes textCycle {
        0%, 33% {
            content: '› fetching Landsat 9 imagery...';
        }
        33.1%, 66% {
            content: '› filtering cloud coverage...';
        }
        66.1%, 100% {
            content: '› computing LST anomalies...';
        }
    }
    
    [data-testid="stSpinner"]::after {
        content: '› fetching Landsat 9 imagery...';
    }
</style>
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 48px 0 32px 0; border-bottom: 1px solid #2e3d2e; position: relative; z-index: 1;">
    <h1 style="color: #6b8f5e; margin: 0; font-size: 42px; font-weight: 600; letter-spacing: -0.5px; font-family: 'DM Sans', sans-serif;">Factory Emissions Visualizer</h1>
    <p style="color: #8a9a8a; font-size: 13px; margin-top: 8px; font-weight: 400; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace;">Geospatial Thermal Analysis / Landsat 9 LST Retrieval</p>
</div>

<svg class="decorative-svg topo-lines" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
    <circle cx="300" cy="300" r="80" fill="none" stroke="#6b8f5e" stroke-width="1.5"/>
    <circle cx="300" cy="300" r="120" fill="none" stroke="#6b8f5e" stroke-width="1"/>
    <circle cx="300" cy="300" r="160" fill="none" stroke="#6b8f5e" stroke-width="0.8"/>
    <circle cx="300" cy="300" r="200" fill="none" stroke="#6b8f5e" stroke-width="0.6"/>
    <circle cx="280" cy="320" r="70" fill="none" stroke="#6b8f5e" stroke-width="0.8"/>
    <circle cx="320" cy="280" r="90" fill="none" stroke="#6b8f5e" stroke-width="0.8"/>
    <path d="M 300 150 Q 350 200 300 300 T 300 450" fill="none" stroke="#6b8f5e" stroke-width="0.8"/>
</svg>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <svg class="decorative-svg orbit-arc" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
        <path d="M 50 150 Q 150 50 250 150" fill="none" stroke="#6b8f5e" stroke-width="1" stroke-dasharray="5,3"/>
        <circle cx="50" cy="150" r="2" fill="#6b8f5e"/>
        <circle cx="250" cy="150" r="2" fill="#6b8f5e"/>
    </svg>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 20px 0 8px 0; position: relative; z-index: 1;">
        <h3 style="color: #8a9a8a; margin: 0; font-size: 11px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Location Input</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="small")
    with col1:
        lat = st.number_input("Latitude", value=st.session_state.picker_coords["lat"],
                             format="%.8f", help="Decimal degrees")
    with col2:
        lon = st.number_input("Longitude", value=st.session_state.picker_coords["lon"],
                             format="%.8f", help="Decimal degrees")

    st.checkbox("Select on Map", key="select_on_map",
                help="Open an interactive map to click-set coordinates")

    st.markdown("---")

    st.markdown("""
    <div style="padding: 4px 0 4px 0;">
        <h3 style="color: #8a9a8a; margin: 0; font-size: 11px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Time Window</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 10px; color: #5a6a5a; margin-bottom: 8px; line-height: 1.5; font-family: 'JetBrains Mono', monospace;">
    Current LST is computed over this range and compared against the
    <em>identical calendar dates</em> shifted back year‑by‑year across
    the preceding 5 years.  (e.g. Mar&nbsp;1 – Jun&nbsp;1 → compared
    against Mar&nbsp;1 – Jun&nbsp;1 in each prior year.)
    </div>
    """, unsafe_allow_html=True)

    default_end = datetime.now(timezone.utc).date()
    default_start = default_end - timedelta(days=90)

    col3, col4 = st.columns(2, gap="small")
    with col3:
        sel_start = st.date_input("From", value=default_start)
    with col4:
        sel_end = st.date_input("To", value=default_end)

    range_days = (sel_end - sel_start).days
    range_valid = 1 <= range_days <= 365

    if not range_valid:
        if range_days < 1:
            st.warning("End date must be after start date.")
        else:
            st.warning(f"Range is {range_days} days. Maximum allowed is 365 days (1 year).")

    st.markdown("---")

    if st.button("RUN ANALYSIS", type="primary", use_container_width=True,
                 disabled=not range_valid):
        st.session_state.show_analysis = True
        st.session_state.lat = lat
        st.session_state.lon = lon
        st.session_state.start_date = sel_start
        st.session_state.end_date = sel_end
    
    st.markdown("---")
    
    st.markdown("""
    <div style='margin-top: 32px; padding: 16px; background: rgba(24, 24, 20, 0.4); border-radius: 4px; border: 1px solid #2e3d2e;'>
        <p style='font-size: 10px; color: #8a9a8a; margin: 0; line-height: 1.8; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 600; font-family: "JetBrains Mono", monospace;'>Processing Pipeline</p>
        <ul style='font-size: 11px; color: #6b8f5e; margin: 12px 0 0 0; padding-left: 0; line-height: 1.9; list-style: none; font-family: "JetBrains Mono", monospace;'>
            <li style="color: #8a9a8a; margin: 4px 0;"><span style="color: #6b8f5e;">›</span> Cloud-filtered GEE fetch</li>
            <li style="color: #8a9a8a; margin: 4px 0;"><span style="color: #6b8f5e;">›</span> Thermal anomaly detection</li>
            <li style="color: #8a9a8a; margin: 4px 0;"><span style="color: #6b8f5e;">›</span> Emission scoring</li>
            <li style="color: #8a9a8a; margin: 4px 0;"><span style="color: #6b8f5e;">›</span> Interactive map visualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Coordinate picker map  (overrides the main area when toggled)
# ---------------------------------------------------------------------------
if st.session_state.select_on_map and not st.session_state.show_analysis:
    st.markdown("""
    <div style="padding: 12px 0 4px 0;">
        <h3 style="color: #8a9a8a; font-size: 11px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">
            Click any location on the map
        </h3>
    </div>
    """, unsafe_allow_html=True)

    pc = st.session_state.picker_coords
    m_picker = folium.Map(location=[pc["lat"], pc["lon"]], zoom_start=5, control_scale=True)

    # Show marker only if user has clicked at least once
    if st.session_state.get("picker_click"):
        folium.CircleMarker(
            [pc["lat"], pc["lon"]], radius=6, color="#ff0000", fill=True,
            fill_opacity=0.9, tooltip="Selected Location"
        ).add_to(m_picker)

    folium.LatLngPopup().add_to(m_picker)
    map_data = st_folium(m_picker, width=1400, height=500, key="picker_map")

    if map_data:
        clicked = map_data.get("last_clicked")
        if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
            prev = st.session_state.get("picker_click")
            if prev is None or prev["lat"] != clicked["lat"] or prev["lng"] != clicked["lng"]:
                st.session_state.picker_coords["lat"] = round(clicked["lat"], 6)
                st.session_state.picker_coords["lon"] = round(clicked["lng"], 6)
                st.session_state.picker_click = {"lat": clicked["lat"], "lng": clicked["lng"]}
                st.rerun()

    st.stop()

# Home page content
if not st.session_state.show_analysis:
    st.markdown("""
    <div style="margin-top: 40px; padding: 0 16px; position: relative; z-index: 1;">
        <h2 style="color: #6b8f5e; font-size: 18px; font-weight: 600; margin-bottom: 32px; letter-spacing: -0.2px; font-family: 'DM Sans', sans-serif;">Core Features</h2>
    </div>
    
    <svg class="decorative-svg dot-grid" viewBox="0 0 1200 500" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
        <defs>
            <radialGradient id="dotGridFade">
                <stop offset="0%" stop-color="#6b8f5e" stop-opacity="1"/>
                <stop offset="100%" stop-color="#6b8f5e" stop-opacity="0"/>
            </radialGradient>
        </defs>
        <rect width="1200" height="500" fill="url(#dotGridFade)" opacity="0.03"/>
        <circle cx="100" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="180" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="260" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="340" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="420" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="500" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="580" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="660" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="740" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="820" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="900" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="980" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="1060" cy="100" r="1.5" fill="#6b8f5e"/>
        <circle cx="100" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="180" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="260" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="340" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="420" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="500" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="580" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="660" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="740" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="820" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="900" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="980" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="1060" cy="180" r="1.5" fill="#6b8f5e"/>
        <circle cx="100" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="180" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="260" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="340" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="420" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="500" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="580" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="660" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="740" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="820" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="900" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="980" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="1060" cy="260" r="1.5" fill="#6b8f5e"/>
        <circle cx="100" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="180" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="260" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="340" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="420" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="500" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="580" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="660" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="740" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="820" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="900" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="980" cy="340" r="1.5" fill="#6b8f5e"/>
        <circle cx="1060" cy="340" r="1.5" fill="#6b8f5e"/>
    </svg>
    
    <div style="position: relative; z-index: 1;">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div style="padding: 28px; border: 1px solid #2e3d2e; border-radius: 4px; background: rgba(24, 24, 20, 0.5); backdrop-filter: blur(3px); transition: all 0.3s ease;">
            <h3 style="color: #6b8f5e; font-size: 15px; margin: 0 0 12px 0; font-weight: 600; font-family: 'DM Sans', sans-serif;">Thermal Analysis</h3>
            <p style="color: #8a9a8a; font-size: 13px; margin: 0; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">Multi-spectral LST retrieval with cloud filtering and radiometric calibration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        st.markdown("""
        <div style="padding: 28px; border: 1px solid #2e3d2e; border-radius: 4px; background: rgba(24, 24, 20, 0.5); backdrop-filter: blur(3px); transition: all 0.3s ease;">
            <h3 style="color: #6b8f5e; font-size: 15px; margin: 0 0 12px 0; font-weight: 600; font-family: 'DM Sans', sans-serif;">Geospatial Processing</h3>
            <p style="color: #8a9a8a; font-size: 13px; margin: 0; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">High-resolution spatial analysis at 100m pixel resolution with coordinate-based queries.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 28px; border: 1px solid #2e3d2e; border-radius: 4px; background: rgba(24, 24, 20, 0.5); backdrop-filter: blur(3px); transition: all 0.3s ease;">
            <h3 style="color: #6b8f5e; font-size: 15px; margin: 0 0 12px 0; font-weight: 600; font-family: 'DM Sans', sans-serif;">Anomaly Detection</h3>
            <p style="color: #8a9a8a; font-size: 13px; margin: 0; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">Statistical thermal outlier identification with pixel-level emission scoring.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        st.markdown("""
        <div style="padding: 28px; border: 1px solid #2e3d2e; border-radius: 4px; background: rgba(24, 24, 20, 0.5); backdrop-filter: blur(3px); transition: all 0.3s ease;">
            <h3 style="color: #6b8f5e; font-size: 15px; margin: 0 0 12px 0; font-weight: 600; font-family: 'DM Sans', sans-serif;">Data Visualization</h3>
            <p style="color: #8a9a8a; font-size: 13px; margin: 0; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">Interactive vector mapping with thermal overlays and statistical dashboards.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="padding: 32px; background: rgba(24, 24, 20, 0.5); border: 1px solid #2e3d2e; border-radius: 4px; margin-top: 32px; text-align: center; backdrop-filter: blur(3px);">
        <h3 style="color: #6b8f5e; font-size: 15px; margin: 0 0 8px 0; font-weight: 600; font-family: 'DM Sans', sans-serif;">Begin Analysis</h3>
        <p style="color: #8a9a8a; font-size: 13px; margin: 0; font-family: 'JetBrains Mono', monospace;">Enter coordinates and click RUN ANALYSIS to process satellite data.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# Analysis results
if st.session_state.show_analysis:
    # Check if we need to run analysis
    needs_analysis = (
        st.session_state.analysis_results is None
        or st.session_state.analysis_results.get('lat') != st.session_state.lat
        or st.session_state.analysis_results.get('lon') != st.session_state.lon
        or st.session_state.analysis_results.get('start_date') != st.session_state.start_date
        or st.session_state.analysis_results.get('end_date') != st.session_state.end_date
    )
    
    if needs_analysis:
        try:
            with st.spinner("Running analysis pipeline (GEE + S5P + clustering)..."):
                results = _run_analysis(
                    st.session_state.lat,
                    st.session_state.lon,
                    str(st.session_state.start_date),
                    str(st.session_state.end_date),
                )
            results.update({
                'lat': st.session_state.lat,
                'lon': st.session_state.lon,
                'start_date': st.session_state.start_date,
                'end_date': st.session_state.end_date,
                'error': None,
            })
            st.session_state.analysis_results = results
            st.success("Analysis complete!")
        except Exception as e:
            st.session_state.analysis_results = {
                'lat': st.session_state.lat,
                'lon': st.session_state.lon,
                'start_date': st.session_state.start_date,
                'end_date': st.session_state.end_date,
                'error': str(e),
            }
            st.error(f"**Error:** {e}")
    
    # Display results from cache
    if st.session_state.analysis_results and not st.session_state.analysis_results.get('error'):
        r = st.session_state.analysis_results
        lst_array = r["lst_array"]
        anomaly_indices = r["anomaly_indices"]
        z_score_map = r["z_score_map"]
        s5p_data = r["s5p_data"]
        tile_layers = r["tile_layers"]
        clusters = r["clusters"]
        ndvi_mean = r["ndvi_mean"]
        ndbi_mean = r["ndbi_mean"]

        # ----- Pipeline status -----
        st.markdown(f"""
        <div style="display: flex; gap: 8px; align-items: center; padding: 12px 0; margin-bottom: 8px; border-bottom: 1px solid #2e3d2e;">
            <span style="font-size: 10px; color: #6b8f5e; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.1em; text-transform: uppercase;">Pipeline</span>
            <span style="color: #6b8f5e; font-size: 11px; font-family: 'JetBrains Mono', monospace;">GEE Fetch ✓</span>
            <span style="color: #5a6a5a;">→</span>
            <span style="color: #6b8f5e; font-size: 11px; font-family: 'JetBrains Mono', monospace;">Thermal Anomaly ✓</span>
            <span style="color: #5a6a5a;">→</span>
            <span style="color: #6b8f5e; font-size: 11px; font-family: 'JetBrains Mono', monospace;">S5P Composition ✓</span>
            <span style="color: #5a6a5a;">→</span>
            <span style="color: #6b8f5e; font-size: 11px; font-family: 'JetBrains Mono', monospace;">Clustering ✓</span>
            <span style="color: #5a6a5a;">→</span>
            <span style="color: #6b8f5e; font-size: 11px; font-family: 'JetBrains Mono', monospace;">Emission Score ✓</span>
            <span style="margin-left: auto; font-size: 10px; color: #5a6a5a; font-family: 'JetBrains Mono', monospace;">
                {st.session_state.lat:.4f}, {st.session_state.lon:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        valid_lst = lst_array[np.isfinite(lst_array)]
        lst_max = f"{np.max(valid_lst):.2f}" if valid_lst.size else "N/A"
        lst_min = f"{np.min(valid_lst):.2f}" if valid_lst.size else "N/A"
        lst_mean = f"{np.mean(valid_lst):.2f}" if valid_lst.size else "N/A"
        score, category = calculate_emission_score(
            z_score_map=z_score_map,
            anomaly_indices=anomaly_indices,
            s5p_data=s5p_data,
            clusters=clusters,
            ndvi_mean=ndvi_mean,
            ndbi_mean=ndbi_mean,
        )

        # ----- Metrics row (top) -----
        cols = st.columns(7, gap="small")
        cols[0].metric("Anomaly pixels", len(anomaly_indices))
        cols[1].metric("Max LST (°C)", lst_max)
        cols[2].metric("Min LST (°C)", lst_min)
        cols[3].metric("Mean LST (°C)", lst_mean)
        cols[4].metric("Emission Score", f"{score:.1f}", delta=category)
        cols[5].metric("Hotspots", len(clusters) if clusters else 0)
        if clusters:
            total_area = sum(c['area_km2'] for c in clusters)
            cols[6].metric("Combined area", f"{total_area:.2f} km²")
        else:
            cols[6].metric("Combined area", "—")

        # ----- Large map -----
        m = create_map(
            st.session_state.lat, st.session_state.lon,
            tile_layers=tile_layers,
            clusters=clusters,
        )
        st_folium(m, width=1400, height=650)

        # ----- Secondary panels -----
        col_bottom_left, col_bottom_right = st.columns(2, gap="large")

        with col_bottom_left:
            if clusters:
                st.markdown("""
                <div style="padding-bottom: 8px;">
                    <h3 style='color: #8a9a8a; font-size: 11px; margin: 0 0 8px 0; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; font-family: "JetBrains Mono", monospace;'>Hotspot Clusters</h3>
                </div>
                """, unsafe_allow_html=True)

                for c in clusters:
                    st.markdown(f"""
                    <div style="padding: 6px 0; display: flex; justify-content: space-between; border-bottom: 1px solid #1e2e1e; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                        <span style="color: #8a9a8a;">Cluster #{c['cluster_id']}</span>
                        <span style="color: #e8e6e1;">{c['size']} px · Z̅={c['mean_z_score']:.1f} · {c['area_km2']} km²</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No hotspot clusters identified.")

        with col_bottom_right:
            if s5p_data:
                st.markdown("""
                <div style="padding-bottom: 8px;">
                    <h3 style='color: #8a9a8a; font-size: 11px; margin: 0 0 8px 0; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; font-family: "JetBrains Mono", monospace;'>Atmospheric Composition</h3>
                </div>
                """, unsafe_allow_html=True)

                for poll, data in s5p_data.items():
                    st.markdown(f"""
                    <div style="padding: 6px 0; display: flex; justify-content: space-between; border-bottom: 1px solid #1e2e1e; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                        <span style="color: #8a9a8a;">{data['label']}</span>
                        <span style="color: #e8e6e1;">{data['mean']:.3e} {data['unit']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style="margin-top: 8px; font-size: 10px; color: #5a6a5a; font-family: 'JetBrains Mono', monospace;">
                    Sentinel‑5P TROPOMI · ~7 km native resolution
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No atmospheric composition data available.")
