# Factory Emissions Visualiser

A geospatial analysis tool that detects industrial thermal emissions using satellite
imagery and calculates a multi-signal confidence score — no black-box ML, fully interpretable
weighted rules.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [How It Works](#how-it-works)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Earth Engine Credentials](#earth-engine-credentials)
   - [Installation](#installation)
   - [Running the App](#running-the-app)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Emission Score Breakdown](#emission-score-breakdown)
7. [Codebase Structure](#codebase-structure)
8. [Key Constraints & Caveats](#key-constraints--caveats)
9. [License](#license)

---

## Overview

Given a latitude and longitude (defaulting to the Odisha industrial belt at 20.9515° N,
85.2157° E), the tool:

1. Queries **Landsat 8/9** surface temperature over the user-defined date window.
2. Builds a historical climatology from the same calendar dates across the preceding
   5 years.
3. Computes a per-pixel **Z-score** — `(current − historical_mean) / historical_stddev` —
   flagging pixels where Z > 1.5 as thermal anomalies.
4. Fetches **Sentinel-5P TROPOMI** column densities for NO₂, SO₂, and CO at ~7 km
   native resolution.
5. Groups anomaly pixels into spatial clusters using **DBSCAN**.
6. Fuses thermal, atmospheric, spectral, and cluster signals into a **0–100 emission
   confidence score** (Low / Medium / High).
7. Renders all results on an interactive **Folium** map with layer toggles and hotspot
   overlays.

The entire pipeline is wrapped in Streamlit's `st.cache_data` (1-hour TTL) so repeated
interactions don't re-trigger expensive GEE calls.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) 1.57+ | Sidebar controls, metrics dashboard, markdown rendering |
| Maps | [Folium](https://python-visualization.github.io/folium/) 0.20 + [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) 0.27 | Interactive Leaflet map, tile layers, cluster overlays |
| Satellite Data | [Google Earth Engine](https://earthengine.google.com) (`earthengine-api` 1.7) | Landsat 9 C2/T1_L2 (LST, NDVI, NDBI), Sentinel-5P OFF/L3 (NO₂, SO₂, CO) |
| Array Compute | [NumPy](https://numpy.org) 2.4 | Z-score maps, anomaly detection, raster arithmetic |
| Clustering | [scikit-learn](https://scikit-learn.org) 1.8 (DBSCAN) | Spatial grouping of thermal anomaly pixels |
| Auth | GEE Service Account (`key.json`) | Authenticated access to Earth Engine API |
| Python | 3.12 | Runtime |

### Notable dependencies not used

`pandas` and `geemap` are listed in `requirements.txt` but are **not imported** by any
source file — leftover from earlier prototypes.

---

## How It Works

### 1. Land Surface Temperature (LST) Retrieval

- Queries `LANDSAT/LC09/C02/T1_L2` and `LANDSAT/LC08/C02/T1_L2` (Collection 2,
  Tier 1, atmospherically corrected).
- Uses band `ST_B10` — thermal infrared (10.6–11.19 µm).
- Applies the Collection 2 scaling: `LST (°C) = ST_B10 × 0.00341802 + 149 − 273.15`.
- Filters scenes to `CLOUD_COVER < 20%` and computes the median composite over the
  user's date window.
- Computes **NDVI** `(NIR − Red) / (NIR + Red)` from bands `SR_B5` and `SR_B4`.
- Computes **NDBI** `(SWIR1 − NIR) / (SWIR1 + NIR)` from bands `SR_B6` and `SR_B5`.
- Builds an **industrial area mask**: pixels where `NDVI < 0.3` **AND** `NDBI > 0.0`
  (low vegetation + built-up reflectance). Only masked pixels are analysed.

### 2. Z-Score Climatology

For every industrial pixel, the tool:

1. Takes the **current** LST from the user's selected window (e.g., Mar 1 – Jun 1,
   2026).
2. Builds a **historical stack** from the *same calendar dates* shifted back year-by-year
   over 5 prior years (2021–2025), using the same Landsat 8+9 merge.
3. Computes `hist_mean` and `hist_stddev` across those 5 annual composites.
4. Calculates `Z = (current − hist_mean) / max(hist_stddev, 0.05 °C)` — the 0.05 floor
   prevents division-by-zero in stable areas.
5. Pixels with **Z > 1.5** are flagged as thermal anomalies.

### 3. Sentinel-5P Atmospheric Composition

- Fetches from `COPERNICUS/S5P/OFFL/L3_*` (offline Level-3 products).
- Three pollutants: **NO₂** (`NO2_column_number_density`), **SO₂**
  (`SO2_column_number_density`), **CO** (`CO_column_number_density`).
- Computes the mean image over the same date window at native ~7 km resolution.
- Products with no data for the period are silently omitted.

### 4. DBSCAN Clustering

- Converts raveled 1-D anomaly indices back to 2-D pixel coordinates, then to
  approximate geographic coordinates.
- Runs DBSCAN with `eps = 0.3 km` (~3 pixels at 100 m resolution) and `min_samples = 3`.
- Returns clusters sorted by size, each with centroid coordinates, mean Z-score,
  pixel count, and approximate area in km².
- Noise points (DBSCAN label −1) are excluded.

### 5. Emission Confidence Scoring

The **0–100 score** is a weighted sum of six sub-scores (see [Emission Score
Breakdown](#emission-score-breakdown)). The final value maps to one of three
categories:

| Score Range | Category |
|---|---|
| 0 – 33 | **Low** |
| 34 – 66 | **Medium** |
| 67 – 100 | **High** |

---

## Getting Started

### Prerequisites

- **Python 3.12** (required — the venv is built for 3.12)
- A **Google Cloud project** with the Earth Engine API enabled
- A **GEE service account** with the Earth Engine Resource Viewer role

### Earth Engine Credentials

The app authenticates via a service account JSON key file. Here's how to get one:

#### Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com).
2. Create a new project or select an existing one.
3. Note the **Project ID**.

#### Step 2: Enable the Earth Engine API

1. Navigate to **APIs & Services → Library**.
2. Search for **"Earth Engine API"** (`earthengine.googleapis.com`).
3. Click **Enable**.

#### Step 3: Create a Service Account

1. Go to **IAM & Admin → Service Accounts**.
2. Click **Create Service Account**.
3. Name it (e.g., `gee-streamlit`) and click **Create and Continue**.
4. Grant the role: **Earth Engine → Earth Engine Resource Viewer**.
5. Click **Done**.

#### Step 4: Generate a JSON Key

1. From the Service Accounts list, click the three-dot menu on your new account and
   select **Manage keys**.
2. Click **Add Key → Create New Key**.
3. Choose **JSON** format and download the file.
4. Rename the downloaded file to `key.json`.
5. Place `key.json` in the **project root** (same directory as `app.py`).

#### Step 5: Register the Service Account with Earth Engine

1. Visit [Earth Engine's service account registration](https://signup.earthengine.google.com/#!/service_accounts).
2. Enter your service account email (it looks like `name@project-id.iam.gserviceaccount.com`).
3. Complete the registration — you'll be redirected to configure your Cloud project.

#### Step 6: Set Environment Variables

The app reads credentials from environment variables. Set them before running:

```bash
export GEE_SERVICE_ACCOUNT="your-account@your-project.iam.gserviceaccount.com"
export GEE_PROJECT_ID="your-project-id"
```

For convenience, create a `.env` file (already in `.gitignore`):

```
GEE_SERVICE_ACCOUNT=your-account@your-project.iam.gserviceaccount.com
GEE_PROJECT_ID=your-project-id
```

Then load it before starting the app:

```bash
set -a && source .env && set +a
streamlit run app.py
```

> ⚠️ **Never commit `key.json` to version control.** It is already listed in
> `.gitignore`.

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Factory-Emissions-Visualiser

# Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
# NOTE: requirements.txt is UTF-16 LE encoded.
# Convert to UTF-8 first, then install:
python -c "open('_reqs.txt','w').write(open('requirements.txt',encoding='utf-16-le').read())" \
  && mv _reqs.txt requirements.txt \
  && pip install -r requirements.txt

# Place your service account key.json in the project root
```

### Running the App

```bash
source .venv/bin/activate
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

---

## Pipeline Architecture

```
┌────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)             │
│  ┌──────────┐  ┌────────────┐  ┌────────────────┐ │
│  │ Sidebar  │  │ Map Picker │  │ Results Panel  │ │
│  │ Lat/Lon  │  │ (Leaflet)  │  │ Metrics + Map  │ │
│  │ Dates    │  │            │  │ Clusters + S5P │ │
│  └────┬─────┘  └────────────┘  └───────┬────────┘ │
│       │                                 │          │
│       └────── _run_analysis() ──────────┘          │
│                  @st.cache_data                     │
└───────────────────────┬────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│   temporal   │ │  sentinel5p │ │  clustering  │
│ .detect_temp │ │ .fetch_all_ │ │ .cluster_    │
│ _anomalies() │ │ pollutants()│ │ anomalies()  │
└──────┬───────┘ └──────┬──────┘ └──────┬───────┘
       │                │               │
       ▼                ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│   landsat    │ │  GEE S5P    │ │   DBSCAN     │
│   LC08/LC09  │ │  OFF/L3     │ │  (sklearn)   │
│   LST + NDVI │ │  NO₂/SO₂/CO │ │              │
│   + NDBI     │ │             │ │              │
└──────────────┘ └─────────────┘ └──────────────┘
       │                │               │
       └────────────────┼───────────────┘
                        ▼
                ┌──────────────┐
                │    fusion    │
                │ .calculate_  │
                │ _emission_   │
                │ score()      │
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │  folium_map  │
                │ .create_map()│
                │ TileLayers + │
                │ CircleMkrs   │
                └──────────────┘
```

### Caching Strategy

- **`gee_auth._initialize_ee()`** → `@st.cache_resource` — runs at most once per
  Streamlit session (Earth Engine init is expensive and side-effect-heavy).
- **`app._run_analysis()`** → `@st.cache_data(ttl=3600)` — caches the full pipeline
  output keyed on `(lat, lon, start_date, end_date)`. Cache auto-invalidates when any
  input changes. Refresh occurs hourly even if inputs don't change.

### Data Flow per Request

1. **`temporal.detect_temporal_anomalies(lat, lon, start, end)`**
   - Calls `_compute_anomaly_images()` which:
     1. Builds a 10 km buffer around the point.
     2. Fetches Landsat 8+9 scenes for the current window.
     3. Fetches Landsat 8+9 scenes for the same calendar window across 5 prior years.
     4. Computes per-pixel Z-score and anomaly flag.
     5. Returns `current_lst`, `z_score`, `anomaly_flag` (all `ee.Image`), plus
        `ndvi_mean` and `ndbi_mean` (floats).
   - Downloads a 3-band NPY (LST, Z_Score, Anomaly) at 100 m scale.
   - Builds 3 tile layer descriptors: LST, Z-Score, Anomaly Flag.
   - Returns `(lst_array, anomaly_indices, z_score_map, tile_layers, ndvi_mean, ndbi_mean)`.

2. **`sentinel5p.fetch_all_pollutants(lat, lon, start, end)`**
   - Iterates over NO₂, SO₂, CO.
   - For each: fetches the mean image over the window, downloads the raster at 7 km
     scale, computes `np.nanmean`.
   - Silently skips pollutants with no data.

3. **`clustering.cluster_anomalies(lst_array, anomaly_indices, z_score_map, lat, lon)`**
   - Converts anomaly ravel indices to geographic coordinates.
   - Runs DBSCAN (eps=0.3 km, min_samples=3).
   - Returns cluster records sorted by size.

4. **`fusion.calculate_emission_score(...)`**
   - Computes 6 sub-scores from thermal, atmospheric, spectral, and cluster signals.
   - Weighted sum → final 0–100 score + category.

5. **`folium_map.create_map(...)`**
   - Creates a Folium map centered on the query point.
   - Adds GEE TileLayers (LST, Z-score, anomaly flag, pollutant overlays).
   - Draws cluster circles (radius proportional to area, color by mean Z-score).
   - Adds centroid markers and a factory location pin.
   - Enables `LayerControl` for toggling individual layers.

---

## Emission Score Breakdown

The **Emission Confidence Score** (0–100) fuses six independent signals. Each sub-score
is a piecewise-linear function mapped to \[0, 100\], then weighted and summed.

| Signal | Weight | What it measures | Scoring logic |
|---|---|---|---|
| **Thermal** | 30% | Mean Z-score of anomaly pixels | Z ≤ 2.0 → 0; Z ≥ 4.0 → 100; linear ramp in between |
| **NO₂** | 20% | Tropospheric NO₂ column density (mol/m²) | ≤ 2e⁻⁶ → 0; ≥ 50e⁻⁶ → 100; two-segment linear ramp |
| **Cluster persistence** | 15% | Number and size of DBSCAN clusters | 1 cluster → 25; 2 → 50; 3+ → 50+(n−2)×12 (capped at 75) + bonus for large clusters |
| **NDBI** | 15% | Built-up index over industrial footprint | ≤ 0.2 → 10; ≥ 0.4 → 100; linear ramp |
| **NDVI suppression** | 10% | Vegetation stress near industry | ≥ 0.25 → 0; ≤ 0.05 → 100; linear ramp (lower NDVI = higher score) |
| **SO₂** | 10% | SO₂ column density (mol/m²) | ≤ 1e⁻⁶ → 0; ≥ 20e⁻⁶ → 100; two-segment linear ramp |

**Weights rationale:** Thermal anomalies are the most direct indicator of industrial
heat output, and NO₂ is a well-known combustion tracer, so they receive the highest
weights. Cluster persistence separates transient thermal noise from sustained industrial
activity. NDBI confirms built-up surfaces (reducing false positives from bare soil).
NDVI suppression captures vegetation stress around facilities. SO₂ is assigned a lower
weight because modern plants often scrub it.

All weights are configurable constants in `services/analytics/fusion.py:26-33`.

---

## Codebase Structure

```
.
├── app.py                          # Streamlit UI + cached pipeline entry point
├── requirements.txt                # Python dependencies (UTF-16 LE encoded!)
├── key.json                        # GEE service account credentials (gitignored)
├── AGENTS.md                       # Developer reference (architecture, constraints)
├── .gitignore
└── services/
    ├── utils/
    │   └── gee_auth.py             # ee.Initialize() with @st.cache_resource guard
    ├── gee_sources/
    │   ├── landsat.py              # Landsat 9 LST fetch (90-day, cloud-filtered)
    │   ├── sentinel5p.py           # Sentinel-5P NO₂/SO₂/CO mean column densities
    │   ├── indices.py              # NDVI, NDBI computation + industrial area mask
    │   └── baselines.py            # (empty — placeholder for future baselines)
    ├── analytics/
    │   ├── temporal.py             # Z-score climatology anomaly detection
    │   ├── clustering.py           # DBSCAN spatial grouping of anomaly pixels
    │   ├── fusion.py               # Multi-signal weighted emission confidence scoring
    │   └── masking.py              # (empty — placeholder for future masking)
    └── visualization/
        ├── raster_layers.py        # GEE TileLayer URL helpers + vis param presets
        ├── folium_map.py           # Folium map builder (TileLayer, clusters, LayerControl)
        └── timeslider.py           # (empty — placeholder for future time-slider)
```

---

## Key Constraints & Caveats

- **No test suite, linters, or CI.** The only verification is running the app.
- **Earth Engine is rate-limited.** The `@st.cache_data(ttl=3600)` decorator prevents
  redundant GEE calls, but rapid coordinate changes will still trigger fresh downloads.
- **Landsat 9's temporal depth is limited.** The pipeline merges Landsat 8 for
  historical queries (pre-2022), but very early years may have sparse coverage.
- **The industrial mask is conservative.** It requires BOTH `NDVI < 0.3` AND
  `NDBI > 0.0`. Genuine industrial sites in vegetated corridors may be partially
  masked out.
- **Sentinel-5P resolution is ~7 km.** Column density values represent broad area
  averages, not point-source measurements. Use them as corroborating signals, not
  definitive indicators.
- **Clustering is approximate.** Geographic coordinates are derived from pixel positions
  assuming a uniform 0.05° span — sufficient for a 20 km analysis tile, but not
  survey-grade.
- **Default coordinates** are hardcoded to 20.9515° N, 85.2157° E (industrial belt near
  Angul/Talcher, Odisha, India). Change the defaults in `app.py:62-63` if needed.
- **`requirements.txt` is UTF-16 LE encoded.** Standard `pip install -r requirements.txt`
  will fail. Convert to UTF-8 first, or use the install command above.
- **Several files are empty placeholders:** `baselines.py`, `masking.py`, `timeslider.py`.
  They exist for future feature development (historical baselines, advanced masking,
  time-slider visualization).

---

## Contributors

- **Prarthana Upadhyaya**
- **P R Hari Hara Sai Pratham**

## License

MIT License — see [LICENSE](LICENSE) for full text.

Copyright (c) 2026
