# AGENTS.md

## Quickstart

```bash
# Activate venv (Python 3.12)
source .venv/bin/activate

# Install deps (UTF-16 LE encoding! convert first if needed)
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Encoding Gotcha

`requirements.txt` is **UTF-16 LE encoded**. Standard `pip install -r requirements.txt` will fail with an encoding error. If you need to re-save it, convert to UTF-8 first:

```bash
python -c "open('requirements2.txt','w').write(open('requirements.txt',encoding='utf-16-le').read())" && mv requirements2.txt requirements.txt
```

## Architecture

Single-page Streamlit app with four service packages:

```
app.py
services/
├── utils/gee_auth.py        → `st.cache_resource`-guarded `ee.Initialize()`
├── gee_sources/
│   ├── landsat.py            → Landsat 9 LST retrieval via GEE
│   └── sentinel5p.py         → NO₂ / SO₂ / CO column density (≈7 km res.)
├── analytics/
│   ├── temporal.py            → Z‑score climatology anomaly detection
│   └── fusion.py              → Multi‑signal weighted confidence scoring
└── visualization/
    ├── raster_layers.py      → GEE tile URL helpers + vis param presets
    └── folium_map.py         → Folium map with TileLayer + LayerControl
```

Flow: user enters lat/lon → `temporal.detect_temporal_anomalies` queries GEE for current LST (last 90 days, Landsat 9) and historical monthly composites (5 previous years, Landsat 8+9), computes per-pixel Z = (current − hist_mean) / hist_std, flags Z > 2 as anomalies, also returns mean NDVI/NDBI over industrial footprint → `sentinel5p.fetch_all_pollutants` fetches NO₂/SO₂/CO rasters at native S5P resolution (~7 km) → `clustering.cluster_anomalies` groups anomalous pixels with DBSCAN → `fusion.calculate_emission_score` combines thermal, atmospheric, spectral, and cluster signals into weighted 0–100 confidence score + Low/Medium/High category → `folium_map.create_map` uses GEE native `TileLayer` tiles (via `raster_layers.build_layer_entry`) with `LayerControl` toggles → `app.py` displays map + metrics.

## Key Constraints

- **No test, lint, build, or CI commands exist.** The only verification is running the app.
- **Earth Engine auth** uses `key.json` (service account credentials) plus `GEE_SERVICE_ACCOUNT` and `GEE_PROJECT_ID` environment variables. The key file is in `.gitignore` and not tracked — if it's missing, the app won't start.
- Default coordinates hardcoded in `app.py`: lat=12.9235, lon=77.4986 (RVCE, Bengaluru).
- Sidebar date inputs let users set the analysis time window (defaults to last 90 days). Changing dates triggers a fresh GEE pipeline via `st.cache_data` keyed on `(lat, lon, start_iso, end_iso)`.
- `st.cache_data(ttl=3600)` wraps the full pipeline (`detect_temporal_anomalies` + S5P + clustering) in `app._run_analysis`. The cache auto-invalidates when lat/lon or date changes, avoiding redundant GEE calls on Streamlit reruns.
- `earthengine-api` calls `ee.Initialize()` inside `detect_temporal_anomalies()`. In Streamlit's rerun-on-interaction model this runs on every script execution. `gee_auth._initialize_ee` is guarded by `st.cache_resource` so the init runs at most once per session.
- `pandas` and `geemap` are listed in `requirements.txt` but **not used** in any source file — leftover dependencies.
