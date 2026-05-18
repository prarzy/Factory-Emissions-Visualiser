import io
from datetime import date, datetime, timedelta

import ee
import numpy as np
import requests

from services.gee_sources.indices import compute_ndvi, compute_ndbi, create_industrial_mask
from services.utils.gee_auth import ensure_ee_initialized


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------

def _add_lst_celsius(img: ee.Image) -> ee.Image:
    """Add a 'LST_Celsius' band from ST_B10 via the Collection 2 scaling."""
    return img.addBands(
        img.select("ST_B10")
        .multiply(0.00341802)
        .add(149)
        .subtract(273.15)
        .rename("LST_Celsius")
    ).copyProperties(img, img.propertyNames())


def _build_collection(start, end, region, include_l8=False):
    """Return a Landsat collection with LST and SR bands.

    Uses Landsat 9 by default.  When *include_l8* is True, merges in
    Landsat 8 as well — needed for historical queries where LC09 may not
    have enough temporal depth (<~2022).
    """
    coll = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    if include_l8:
        coll = coll.merge("LANDSAT/LC08/C02/T1_L2")
    return (
        coll.filterDate(str(start), str(end))
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .select(["ST_B10", "SR_B4", "SR_B5", "SR_B6"])
        .map(_add_lst_celsius)
    )


def _apply_industrial_mask(image: ee.Image) -> ee.Image:
    """Compute NDVI/NDBI, build the mask, and return masked LST_Celsius."""
    image = compute_ndbi(compute_ndvi(image))
    mask = create_industrial_mask(image)
    return image.select("LST_Celsius").updateMask(mask)


def _get_month_range(year: int, month: int):
    """Return ``(start_date, end_date)`` covering the whole month."""
    start = date(year, month, 1)
    if month == 12:
        return start, date(year, 12, 31)
    return start, date(year, month + 1, 1) - timedelta(days=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_temporal_anomalies(lat: float, lon: float):
    """Detect thermal anomalies via Z-scores against a 5‑year climatology.

    For every pixel:
      1. Retrieve the **current** LST composite (median of the last 90
         days of Landsat 9, masked to industrial areas).
      2. Retrieve **historical** annual composites (same calendar month,
         previous 5 years, Landsat 8+9 merged, independently masked).
      3. Compute the per‑pixel historical mean and standard deviation.
      4. Z = (current – historical_mean) / historical_std.
      5. Anomalies are pixels where *Z > 2*.

    Returns
    -------
    lst_array : np.ndarray  —  current LST (°C) of industrial pixels
    anomaly_indices : np.ndarray  —  raveled indices where Z > 2
    z_score_map : np.ndarray  —  Z‑score for every industrial pixel
    """
    ensure_ee_initialized()

    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(10_000).bounds()
    today = datetime.utcnow().date()

    # ----- current composite (last 90 days) -----
    cur_coll = _build_collection(today - timedelta(days=90), today, region)
    if cur_coll.size().getInfo() == 0:
        raise RuntimeError("No cloud‑free Landsat 9 scenes in the last 90 days.")
    current_lst = _apply_industrial_mask(cur_coll.median())

    # ----- historical composites (same month, previous 5 years) -----
    historical_lst = []
    for offset in range(1, 6):
        yr = today.year - offset
        start, end = _get_month_range(yr, today.month)
        # include_l8=True so earlier years (pre‑L9) are covered
        coll = _build_collection(start, end, region, include_l8=True)
        if coll.size().getInfo() > 0:
            historical_lst.append(coll.median())
        # years without any cloud‑free scenes are silently skipped

    if len(historical_lst) < 2:
        raise RuntimeError(
            "Insufficient historical data: need ≥2 years of cloud‑free "
            f"composites for month {today.month}, got {len(historical_lst)}."
        )

    # Apply the same industrial mask independently to each historical year
    # so we compare industrial-to-industrial temperatures over time.
    masked_hist = [_apply_industrial_mask(img) for img in historical_lst]

    # Per‑pixel mean & standard deviation across the available years
    hist_stack = ee.Image.cat(
        [img.rename(f"y{i}") for i, img in enumerate(masked_hist)]
    )
    hist_mean = hist_stack.reduce(ee.Reducer.mean())
    hist_std = hist_stack.reduce(ee.Reducer.stdDev())

    # ----- Z‑score with division‑by‑zero guard -----
    z_score = current_lst.subtract(hist_mean).divide(hist_std)
    z_score = z_score.where(hist_std.eq(0), 0)

    anomaly_flag = z_score.gt(2)

    # Bundle into a single 3‑band NPY for one network round‑trip
    output = ee.Image.cat([current_lst, z_score, anomaly_flag])
    output = output.rename(["LST", "Z_Score", "Anomaly"])

    url = output.getDownloadURL({
        "scale": 100,
        "region": region,
        "format": "NPY",
    })

    # ----- download & unpack -----
    # Performance: the GEE server does all aggregation server‑side.
    # The biggest cost is waiting for *six* image collections to be
    # evaluated (1 current + up to 5 historical).  For a given (lat, lon,
    # month) the historical mean / std change only once per year and could
    # be cached with st.cache_data in the caller along with the location.
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = np.load(io.BytesIO(resp.content))

    lst_array = data[:, :, 0]
    z_score_map = data[:, :, 1]
    anomaly_mask = data[:, :, 2] > 0.5
    anomaly_indices = np.where(anomaly_mask.ravel())[0]

    return lst_array, anomaly_indices, z_score_map


def calculate_emission_score(lst_array, anomaly_indices, z_score_map=None):
    """Emission score (0–100) based on Z‑score severity.

    Uses Z‑scores when available (new flow); falls back to pure
    temperature‑based severity when called with legacy cached results.

    With Z‑scores:
        severity  = clamp((mean(Z[anomalies]) - 2) / 3,  0, 1)
    Fallback severity:
        severity  = clamp((T_anomaly - T_mean) / T_range,  0, 1)

    density   = |anomalies| / |valid industrial pixels|
    score     = (0.7 · severity  +  0.3 · density) × 100
    """
    flat = lst_array.ravel()

    if len(anomaly_indices) == 0:
        return 0.0

    anomaly_values = flat[anomaly_indices]
    anomaly_values = anomaly_values[~np.isnan(anomaly_values)]

    if anomaly_values.size == 0:
        return 0.0

    all_valid = flat[~np.isnan(flat)]
    density = len(anomaly_values) / len(all_valid) if len(all_valid) > 0 else 0

    if z_score_map is not None:
        flat_z = z_score_map.ravel()
        anomaly_z = flat_z[anomaly_indices]
        anomaly_z = anomaly_z[~np.isnan(anomaly_z)]
        if anomaly_z.size > 0:
            # Z=2 is threshold → severity 0 at Z=2, 1 at Z=5
            severity = np.clip((anomaly_z.mean() - 2.0) / 3.0, 0, 1)
        else:
            severity = 0.0
    else:
        # Fallback: temperature-based severity from LST (°C)
        temp_range = all_valid.max() - all_valid.min()
        if temp_range == 0:
            return 0.0
        mean_temp = all_valid.mean()
        severity = np.clip((anomaly_values.mean() - mean_temp) / temp_range, 0, 1)

    return round((0.7 * severity + 0.3 * density) * 100, 2)
