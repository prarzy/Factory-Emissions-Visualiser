import io
from datetime import date, datetime, timedelta

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


def _compute_anomaly_images(lat: float, lon: float):
    """Core GEE computation — return ee.Images for current LST, Z‑score,
    and anomaly flag.  Shared by the download path (metrics) and the
    tile path (map rendering).

    Also returns mean NDVI / NDBI over the industrial footprint for
    the confidence scoring system."""
    ensure_ee_initialized()

    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(10_000).bounds()
    today = datetime.utcnow().date()

    # ----- current composite (last 90 days) -----
    cur_coll = _build_collection(today - timedelta(days=90), today, region)
    if cur_coll.size().getInfo() == 0:
        raise RuntimeError("No cloud‑free Landsat 9 scenes in the last 90 days.")

    # Compute industrially‑masked LST plus mean NDVI / NDBI over those
    # same industrial pixels (used downstream by the fusion confidence scorer).
    current_median = cur_coll.median()
    current_median = compute_ndbi(compute_ndvi(current_median))
    ind_mask = create_industrial_mask(current_median)
    current_lst = current_median.select("LST_Celsius").updateMask(ind_mask)

    ndvi_mean, ndbi_mean = _industrial_index_stats(current_median, ind_mask, region)

    # ----- historical composites (same month, previous 5 years) -----
    historical_lst = []
    for offset in range(1, 6):
        yr = today.year - offset
        start, end = _get_month_range(yr, today.month)
        coll = _build_collection(start, end, region, include_l8=True)
        if coll.size().getInfo() > 0:
            historical_lst.append(coll.median())

    if len(historical_lst) < 2:
        raise RuntimeError(
            "Insufficient historical data: need ≥2 years of cloud‑free "
            f"composites for month {today.month}, got {len(historical_lst)}."
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

def detect_temporal_anomalies(lat: float, lon: float):
    """Detect thermal anomalies via Z-scores.

    Returns
    -------
    lst_array : np.ndarray  —  current LST (°C) of industrial pixels
    anomaly_indices : np.ndarray  —  raveled indices where Z > 2
    z_score_map : np.ndarray  —  Z‑score for every industrial pixel
    tile_layers : list[dict]  —  Folium tile layer descriptors
    ndvi_mean : float  —  mean NDVI over the industrial footprint
    ndbi_mean : float  —  mean NDBI over the industrial footprint
    """
    # pylint: disable=unbalanced-tuple-unpacking
    r = _compute_anomaly_images(lat, lon)
    current_lst, z_score, anomaly_flag, region, ndvi_mean, ndbi_mean = r

    # ---- tile layers (native GEE map tiles, no matplotlib) ----
    tile_layers = [
        build_layer_entry(current_lst, LST_VIS,
                          "Land Surface Temperature", opacity=0.55),
        build_layer_entry(z_score, ZSCORE_VIS,
                          "Z-Score (Thermal Anomaly)", opacity=0.50),
        build_layer_entry(anomaly_flag, ANOMALY_VIS,
                          "Anomaly Flag (Z>2)", opacity=0.45, show=False),
    ]

    # ---- array download for metrics ----
    output = ee.Image.cat([current_lst, z_score, anomaly_flag])
    output = output.rename(["LST", "Z_Score", "Anomaly"])

    url = output.getDownloadURL({
        "scale": 100,
        "region": region,
        "format": "NPY",
    })

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = np.load(io.BytesIO(resp.content))

    lst_array = data[:, :, 0]
    z_score_map = data[:, :, 1]
    anomaly_mask = data[:, :, 2] > 0.5
    anomaly_indices = np.where(anomaly_mask.ravel())[0]

    return lst_array, anomaly_indices, z_score_map, tile_layers, ndvi_mean, ndbi_mean


def build_temporal_tile_layers(lat: float, lon: float):
    """Return Folium tile layer descriptors for the LST heatmap, Z‑score,
    and anomaly overlay.

    Each entry follows the shape produced by
    ``raster_layers.build_layer_entry``.
    """
    current_lst, z_score, anomaly_flag, _region, *_ = _compute_anomaly_images(lat, lon)

    return [
        build_layer_entry(current_lst, LST_VIS,
                          "Land Surface Temperature", opacity=0.55),
        build_layer_entry(z_score, ZSCORE_VIS,
                          "Z-Score (Thermal Anomaly)", opacity=0.50),
        build_layer_entry(anomaly_flag, ANOMALY_VIS,
                          "Anomaly Flag (Z>2)", opacity=0.45, show=False),
    ]



