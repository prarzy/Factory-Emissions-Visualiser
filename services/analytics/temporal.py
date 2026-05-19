import io
from datetime import date, datetime, timedelta, timezone

import ee
import numpy as np
import requests

from services.gee_sources.indices import compute_ndvi, compute_ndbi, create_industrial_mask
from services.utils.gee_auth import ensure_ee_initialized
from services.visualization.raster_layers import (
    LST_VIS, ZSCORE_VIS, ANOMALY_VIS, build_layer_entry,
)


# ---------------------------------------------------------------------------
# GEE helpers (shared between download & tile paths)
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


def _industrial_index_stats(image_with_indices, mask, region):
    """Return ``(ndvi_mean, ndbi_mean)`` over the masked industrial area
    for the confidence scoring system."""
    combined = ee.Image.cat([
        image_with_indices.select("NDVI").updateMask(mask),
        image_with_indices.select("NDBI").updateMask(mask),
    ])
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=100,
        maxPixels=1e9,
    )
    return stats.get("NDVI").getInfo(), stats.get("NDBI").getInfo()


def _default_date_range():
    """Return ``(start_date, end_date)`` defaulting to the last 90 days."""
    end = datetime.now(timezone.utc).date()
    return end - timedelta(days=90), end


def _compute_anomaly_images(lat: float, lon: float,
                            start_date: date | None = None,
                            end_date: date | None = None):
    """Core GEE computation — return ee.Images for current LST, Z‑score,
    and anomaly flag.  Shared by the download path (metrics) and the
    tile path (map rendering).

    *start_date* / *end_date* set the **current** analysis window
    (default last 90 days).  The historical climatology is always
    computed from the same calendar month as *end_date* across the
    preceding 5 years.

    Also returns mean NDVI / NDBI over the industrial footprint for
    the confidence scoring system.
    """
    ensure_ee_initialized()

    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(10_000).bounds()

    if start_date is None or end_date is None:
        start_date, end_date = _default_date_range()

    ref_date = end_date  # reference point for historical month

    # ----- current composite (selected window) -----
    cur_coll = _build_collection(start_date, end_date, region)
    if cur_coll.size().getInfo() == 0:
        raise RuntimeError(
            f"No cloud‑free Landsat scenes for {start_date} – {end_date}."
        )

    current_median = cur_coll.median()
    current_median = compute_ndbi(compute_ndvi(current_median))
    ind_mask = create_industrial_mask(current_median)
    current_lst = current_median.select("LST_Celsius").updateMask(ind_mask)

    ndvi_mean, ndbi_mean = _industrial_index_stats(current_median, ind_mask, region)

    # ----- historical composites (same month as ref_date, previous 5 years) -----
    historical_lst = []
    for offset in range(1, 6):
        yr = ref_date.year - offset
        start, end = _get_month_range(yr, ref_date.month)
        coll = _build_collection(start, end, region, include_l8=True)
        if coll.size().getInfo() > 0:
            historical_lst.append(coll.median())

    if len(historical_lst) < 2:
        raise RuntimeError(
            "Insufficient historical data: need ≥2 years of cloud‑free "
            f"composites for month {ref_date.month}, got {len(historical_lst)}."
        )

    masked_hist = [_apply_industrial_mask(img) for img in historical_lst]

    hist_stack = ee.Image.cat(
        [img.rename(f"y{i}") for i, img in enumerate(masked_hist)]
    )
    hist_mean = hist_stack.reduce(ee.Reducer.mean())
    hist_std = hist_stack.reduce(ee.Reducer.stdDev())

    # ----- Z‑score with division‑by‑zero guard -----
    z_score = current_lst.subtract(hist_mean).divide(hist_std)
    z_score = z_score.where(hist_std.eq(0), 0)

    anomaly_flag = z_score.gt(2)

    return current_lst, z_score, anomaly_flag, region, ndvi_mean, ndbi_mean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_temporal_anomalies(lat: float, lon: float,
                              start_date: date | None = None,
                              end_date: date | None = None):
    """Detect thermal anomalies via Z-scores for a configurable time window.

    Parameters
    ----------
    start_date, end_date :
        Date range for the **current** analysis window.  Defaults to
        the last 90 days when omitted.

    Returns
    -------
    lst_array : np.ndarray  —  current LST (°C) of industrial pixels
    anomaly_indices : np.ndarray  —  raveled indices where Z > 2
    z_score_map : np.ndarray  —  Z‑score for every industrial pixel
    tile_layers : list[dict]  —  Folium tile layer descriptors
    ndvi_mean : float  —  mean NDVI over the industrial footprint
    ndbi_mean : float  —  mean NDBI over the industrial footprint
    """
    r = _compute_anomaly_images(lat, lon, start_date, end_date)
    current_lst, z_score, anomaly_flag, region, ndvi_mean, ndbi_mean = r

    # ---- tile layers ----
    tile_layers = [
        build_layer_entry(current_lst, LST_VIS,
                          "Land Surface Temperature", opacity=0.55),
        build_layer_entry(z_score, ZSCORE_VIS,
                          "Z-Score (Thermal Anomaly)", opacity=0.50),
        build_layer_entry(anomaly_flag, ANOMALY_VIS,
                          "Anomaly Flag (Z>2)", opacity=0.45, show=False),
    ]

    # ---- array download for metrics ----
    # Cast all bands to float32 so GEE returns a uniform‑type 3‑D NPY
    # (avoiding the structured‑array issue from mixed dtypes).
    output = ee.Image.cat([
        current_lst.toFloat(),
        z_score.toFloat(),
        anomaly_flag.toFloat(),
    ]).rename(["LST", "Z_Score", "Anomaly"])

    url = output.getDownloadURL({
        "scale": 100,
        "region": region,
        "format": "NPY",
    })

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = np.load(io.BytesIO(resp.content))

    # GEE can return multi-band NPY in several shapes depending on the
    # server-side band types:
    #   3-D (bands, rows, cols)  – when all bands share the same dtype.
    #   2-D structured (rows, cols) with named fields – when dtypes differ.
    if data.ndim == 3 and data.shape[0] == 3:
        lst_array = np.asarray(data[0], dtype=np.float32)
        z_score_map = np.asarray(data[1], dtype=np.float32)
        anomaly_map = np.asarray(data[2], dtype=np.float32)
    elif data.ndim == 3:
        lst_array = np.asarray(data[:, :, 0], dtype=np.float32)
        z_score_map = np.asarray(data[:, :, 1], dtype=np.float32)
        anomaly_map = np.asarray(data[:, :, 2], dtype=np.float32)
    elif data.dtype.names:
        # Structured 2-D array:  (rows, cols) with named bands
        lst_array = np.asarray(data["LST"], dtype=np.float32)
        z_score_map = np.asarray(data["Z_Score"], dtype=np.float32)
        anomaly_map = np.asarray(data["Anomaly"], dtype=np.float32)
    else:
        lst_array = np.asarray(data, dtype=np.float32)
        z_score_map = np.zeros_like(lst_array)
        anomaly_map = np.zeros_like(lst_array)

    anomaly_indices = np.where((anomaly_map > 0.5).ravel())[0]

    return lst_array, anomaly_indices, z_score_map, tile_layers, ndvi_mean, ndbi_mean


def build_temporal_tile_layers(lat: float, lon: float,
                               start_date: date | None = None,
                               end_date: date | None = None):
    """Return tile layer descriptors (no array download — for preview use)."""
    r = _compute_anomaly_images(lat, lon, start_date, end_date)
    current_lst, z_score, anomaly_flag = r[0], r[1], r[2]

    return [
        build_layer_entry(current_lst, LST_VIS,
                          "Land Surface Temperature", opacity=0.55),
        build_layer_entry(z_score, ZSCORE_VIS,
                          "Z-Score (Thermal Anomaly)", opacity=0.50),
        build_layer_entry(anomaly_flag, ANOMALY_VIS,
                          "Anomaly Flag (Z>2)", opacity=0.45, show=False),
    ]
