"""Reusable helpers for GEE tile URL generation and visualisation presets.

Replaces the former matplotlib‑based PNG overlay system with native
Earth Engine map tiles, providing proper geographic alignment.
"""


# ---------------------------------------------------------------------------
# Visualisation parameter presets for each layer type
# ---------------------------------------------------------------------------

LST_VIS = {
    "min": 20,
    "max": 45,
    "palette": ["#000000", "#8b0000", "#ff4500", "#ffd700", "#ffffff"],
}

# Z-score (signed diverging palette — blue=cold, red=hot)
ZSCORE_VIS = {
    "min": -3,
    "max": 3,
    "palette": ["#0000ff", "#ffffff", "#ff0000"],
}

# Anomaly flag (binary highlight)
ANOMALY_VIS = {
    "min": 0,
    "max": 3,
    "palette": ["#000000", "#00ffff"],
}

NO2_VIS = {
    "min": 0,
    "max": 0.00015,
    "palette": ["#0000ff", "#00ff00", "#ffff00", "#ff0000"],
}

SO2_VIS = {
    "min": 0,
    "max": 0.01,
    "palette": ["#0000ff", "#00ffff", "#ffff00", "#ff0000"],
}

CO_VIS = {
    "min": 0,
    "max": 0.04,
    "palette": ["#000000", "#0000ff", "#ff00ff", "#ffffff"],
}

POLLUTANT_VIS = {
    "NO2": NO2_VIS,
    "SO2": SO2_VIS,
    "CO": CO_VIS,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_ee_tile_url(image, vis_params):
    """Get a GEE tile URL template for *image* with given *vis_params*.

    The returned URL uses ``{z}/{x}/{y}`` placeholders suitable for
    ``folium.TileLayer``.
    """
    map_id = image.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


def build_layer_entry(image, vis_params, name, *, opacity=0.6, show=True):
    """Create a tile layer descriptor from a GEE image.

    Returns a dict::

        {
            "url":   "https://earthengine.../{z}/{x}/{y}",
            "name":  "Land Surface Temperature",
            "opacity": 0.6,
            "show":  True,
        }

    Suitable for passing to ``folium_map.create_map``.
    """
    return {
        "url": get_ee_tile_url(image, vis_params),
        "name": name,
        "opacity": opacity,
        "show": show,
    }
