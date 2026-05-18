import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from services.analytics.temporal import detect_temporal_anomalies, calculate_emission_score
from services.gee_sources.sentinel5p import fetch_all_pollutants
from services.visualization.folium_map import create_map

# Initialize session state
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None


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
        lat = st.number_input("Latitude", value=20.95150000, format="%.8f", 
                             help="Decimal degrees")
    with col2:
        lon = st.number_input("Longitude", value=85.2157, format="%.8f",
                             help="Decimal degrees")
    
    st.markdown("---")
    
    if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
        st.session_state.show_analysis = True
        st.session_state.lat = lat
        st.session_state.lon = lon
    
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
        st.session_state.analysis_results is None or
        st.session_state.analysis_results.get('lat') != st.session_state.lat or
        st.session_state.analysis_results.get('lon') != st.session_state.lon
    )
    
    if needs_analysis:
        try:
            with st.spinner("Computing temporal anomalies (GEE climatology)..."):
                lst_array, anomaly_indices, z_score_map = detect_temporal_anomalies(
                    st.session_state.lat, st.session_state.lon
                )

            with st.spinner("Fetching Sentinel-5P atmospheric data..."):
                s5p_data = fetch_all_pollutants(
                    st.session_state.lat, st.session_state.lon
                )

            # Cache results
            st.session_state.analysis_results = {
                'lat': st.session_state.lat,
                'lon': st.session_state.lon,
                'lst_array': lst_array,
                'anomaly_indices': anomaly_indices,
                'z_score_map': z_score_map,
                's5p_data': s5p_data,
                'error': None
            }
            st.success("Analysis complete!")
        except Exception as e:
            st.session_state.analysis_results = {
                'lat': st.session_state.lat,
                'lon': st.session_state.lon,
                'error': str(e)
            }
            st.error(f"**Error:** {e}")
    
    # Display results from cache
    if st.session_state.analysis_results and not st.session_state.analysis_results.get('error'):
        results = st.session_state.analysis_results
        lst_array = results['lst_array']
        anomaly_indices = results['anomaly_indices']
        z_score_map = results.get('z_score_map')
        s5p_data = results.get('s5p_data')

        col_map, col_metrics = st.columns([3, 1], gap="large")

        with col_map:
            m = create_map(
                st.session_state.lat, st.session_state.lon,
                lst_array, anomaly_indices,
                pollutants=s5p_data,
            )
            st_folium(m, width=1100, height=600)

        with col_metrics:
            st.markdown("""
            <div style="padding-bottom: 8px; margin-bottom: 16px;">
                <h3 style='color: #6b8f5e; font-size: 16px; margin: 0; font-weight: 600; letter-spacing: -0.2px; font-family: "DM Sans", sans-serif;'>Results</h3>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Anomaly pixels", len(anomaly_indices))
            st.metric("Max LST (C)", f"{np.nanmax(lst_array):.2f}")
            st.metric("Min LST (C)", f"{np.nanmin(lst_array):.2f}")
            st.metric("Mean LST (C)", f"{np.nanmean(lst_array):.2f}")
            st.metric("Emission Score", f"{calculate_emission_score(lst_array, anomaly_indices, z_score_map):.2f}")

            if s5p_data:
                st.markdown("""
                <div style="padding-top: 24px; padding-bottom: 8px;">
                    <h3 style='color: #8a9a8a; font-size: 11px; margin: 0; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; font-family: "JetBrains Mono", monospace;'>Atmospheric Composition</h3>
                </div>
                """, unsafe_allow_html=True)

                for poll, data in s5p_data.items():
                    mean_val = data['mean']
                    st.metric(
                        data['label'],
                        f"{mean_val:.3e}",
                        help=f"Mean column density · {data['unit']}",
                    )

                st.markdown("""
                <div style="margin-top: 8px; font-size: 10px; color: #5a6a5a; font-family: 'JetBrains Mono', monospace;">
                    Sentinel‑5P TROPOMI · ~7 km native resolution
                </div>
                """, unsafe_allow_html=True)
