import io
from datetime import date

import ee
import numpy as np
import requests

from services.utils.gee_auth import ensure_ee_initialized


# ---------------------------------------------------------------------------
# Sentinel‑5P product metadata
# ---------------------------------------------------------------------------

S5P_PRODUCTS = {
    "NO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "NO2_column_number_density",
        "unit": "mol/m²",
        "cmap": "viridis",
        "label": "NO\u2082 Tropospheric Column",
    },
    "SO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
        "band": "SO2_column_number_density",
        "unit": "mol/m²",
        "cmap": "RdYlGn_r",
        "label": "SO\u2082 Column Density",
    },
    "CO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "unit": "mol/m²",
        "cmap": "plasma",
        "label": "CO Column Density",
    },
}

# Default time window for atmospheric queries (last 90 days, matching Landsat)
_DEFAULT_DAYS = 90


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_sentinel5p(
    lat: float,
    lon: float,
    start_date: date | None = None,
    end_date: date | None = None,
    pollutant: str = "NO2",
):
    """Fetch the mean column density raster for a single S5P pollutant.

    Parameters
    ----------
    lat, lon :
        Centre of the 20×20 km bounding box (matches the Landsat region).
    start_date, end_date :
        Date range.  Defaults to the last 90 days when omitted.
    pollutant :
        One of ``"NO2"``, ``"SO2"``, ``"CO"``.

    Returns
    -------
    dict
        ``{pollutant: {array, unit, cmap, label, mean}}`` — a single‑key
        dict in the same shape as :func:`fetch_all_pollutants` so callers
        can use a uniform display path.
    """
    ensure_ee_initialized()

    if start_date is None or end_date is None:
        from datetime import datetime, timedelta
        end = datetime.utcnow().date()
        start = end - timedelta(days=_DEFAULT_DAYS)
    else:
        start, end = start_date, end_date

    info = S5P_PRODUCTS[pollutant]
    point = ee.Geometry.Point(lon, lat)
    # 10 km radius → 20 km bounding box, same as the Landsat pipeline
    region = point.buffer(10_000).bounds()

    coll = (
        ee.ImageCollection(info["collection"])
        .filterDate(str(start), str(end))
        .filterBounds(region)
        .select(info["band"])
    )

    if coll.size().getInfo() == 0:
        raise RuntimeError(
            f"No S5P {pollutant} data available for the given period and location."
        )

    mean_image = coll.mean()

    # ------------------------------------------------------------------
    # Resolution note:
    # Sentinel‑5P has a native pixel footprint of ≈3.5–7 km (TROPOMI).
    # Setting scale=7000 keeps us at that native resolution so we never
    # falsely imply 30 m atmospheric precision.  The resulting raster
    # will be ≈3×3 pixels over a 20 km box — intentionally blocky.
    # ------------------------------------------------------------------
    url = mean_image.getDownloadURL({
        "scale": 7000,
        "region": region,
        "format": "NPY",
    })

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = np.load(io.BytesIO(resp.content))
    array = data.astype(np.float32, copy=False)

    return {
        pollutant: {
            "array": array,
            "unit": info["unit"],
            "cmap": info["cmap"],
            "label": info["label"],
            "mean": float(np.nanmean(array)),
        }
    }


def fetch_all_pollutants(
    lat: float,
    lon: float,
    start_date: date | None = None,
    end_date: date | None = None,
):
    """Fetch mean column density rasters for NO₂, SO₂, and CO.

    Returns a dict keyed by pollutant name, each value containing:

    .. code-block:: python

        {
            "array": np.ndarray,   # 2‑D array, native S5P resolution
            "unit": "mol/m²",
            "cmap": "viridis",
            "label": "NO₂ Tropospheric Column",
            "mean": 1.23e-6,       # scalar mean of valid pixels
        }

    The arrays are intentionally coarse (~3–7 km / pixel) and rendered
    with nearest‑neighbour interpolation to communicate the true sensor
    resolution.
    """
    result = {}
    for pollutant in S5P_PRODUCTS:
        try:
            result.update(fetch_sentinel5p(lat, lon, start_date, end_date, pollutant))
        except RuntimeError:
            # One product may be empty while others have data (cloud gaps,
            # polar coverage limits, etc.) — skip silently.
            pass
    return result
