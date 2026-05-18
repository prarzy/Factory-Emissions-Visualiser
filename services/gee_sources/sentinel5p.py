import io
from datetime import date

import ee
import numpy as np
import requests

from services.utils.gee_auth import ensure_ee_initialized
from services.visualization.raster_layers import POLLUTANT_VIS, build_layer_entry


# ---------------------------------------------------------------------------
# Sentinel‑5P product metadata
# ---------------------------------------------------------------------------

S5P_PRODUCTS = {
    "NO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "NO2_column_number_density",
        "unit": "mol/m²",
        "label": "NO\u2082 Tropospheric Column",
    },
    "SO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
        "band": "SO2_column_number_density",
        "unit": "mol/m²",
        "label": "SO\u2082 Column Density",
    },
    "CO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "unit": "mol/m²",
        "label": "CO Column Density",
    },
}

_DEFAULT_DAYS = 90


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_dates(start_date, end_date):
    if start_date is None or end_date is None:
        from datetime import datetime, timedelta
        end = datetime.utcnow().date()
        start = end - timedelta(days=_DEFAULT_DAYS)
        return start, end
    return start_date, end_date


def _mean_image(lat, lon, start, end, pollutant):
    """Return the mean ee.Image for a single S5P pollutant over the
    given time window clipped to the 20 km bounding box."""
    info = S5P_PRODUCTS[pollutant]
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(10_000).bounds()

    coll = (
        ee.ImageCollection(info["collection"])
        .filterDate(str(start), str(end))
        .filterBounds(region)
        .select(info["band"])
    )
    if coll.size().getInfo() == 0:
        raise RuntimeError(
            f"No S5P {pollutant} data for the given period and location."
        )
    return coll.mean(), region


def _download_raster(image, region, scale=7000):
    """Download a single‑band GEE image and return the 2‑D numpy array."""
    url = image.getDownloadURL({
        "scale": scale,
        "region": region,
        "format": "NPY",
    })
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return np.load(io.BytesIO(resp.content)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sentinel5p(lat, lon, start_date=None, end_date=None, pollutant="NO2"):
    """Fetch the mean column density array for a single S5P pollutant.

    Returns a dict keyed by *pollutant*:

    .. code-block:: python

        {pollutant: {array, unit, label, mean}}
    """
    start, end = _resolve_dates(start_date, end_date)
    mean_image, region = _mean_image(lat, lon, start, end, pollutant)
    array = _download_raster(mean_image, region)
    info = S5P_PRODUCTS[pollutant]

    return {
        pollutant: {
            "array": array,
            "unit": info["unit"],
            "label": info["label"],
            "mean": float(np.nanmean(array)),
        }
    }


def fetch_all_pollutants(lat, lon, start_date=None, end_date=None):
    """Fetch mean column density rasters for NO₂, SO₂, and CO.

    Returns a dict ``{pollutant: {array, unit, label, mean}}``.
    Products with no data are silently omitted.
    """
    result = {}
    for pollutant in S5P_PRODUCTS:
        try:
            result.update(fetch_sentinel5p(lat, lon, start_date, end_date, pollutant))
        except RuntimeError:
            pass
    return result


def build_pollutant_tile_layers(lat, lon, start_date=None, end_date=None):
    """Return Folium tile layer descriptors for each available S5P
    pollutant.

    Each entry follows the shape produced by
    ``raster_layers.build_layer_entry``.  Pollutants without data for
    the requested period are silently omitted.
    """
    start, end = _resolve_dates(start_date, end_date)
    layers = []

    for pollutant in S5P_PRODUCTS:
        try:
            mean_image, _region = _mean_image(lat, lon, start, end, pollutant)
            info = S5P_PRODUCTS[pollutant]
            layers.append(
                build_layer_entry(
                    mean_image,
                    POLLUTANT_VIS[pollutant],
                    info["label"],
                    opacity=0.35,
                    show=False,     # off by default (too many layers)
                )
            )
        except RuntimeError:
            pass

    return layers
