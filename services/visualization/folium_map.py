import folium
import numpy as np

from services.visualization.raster_layers import render_lst_heatmap, render_pollutant_layer


def create_map(lat: float, lon: float,
               lst_array: np.ndarray,
               anomaly_indices: np.ndarray | list[int],
               pollutants: dict | None = None):
    """Build a Folium map with LST heatmap, anomaly markers, and optional
    Sentinel‑5P pollutant overlays.

    Parameters
    ----------
    pollutants : dict, optional
        Output of ``sentinel5p.fetch_all_pollutants()`` — a dict of
        ``{name: {array, cmap, label, ...}}``.  Each pollutant is
        rendered as a separate ``ImageOverlay`` beneath the LST layer
        at a lower opacity.
    """
    m = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True)
    span_deg = 0.05
    bounds = [[lat - span_deg, lon - span_deg],
              [lat + span_deg, lon + span_deg]]

    # ------------------------------------------------------------------
    # Pollutant overlays (bottom layers — added first so they draw
    # beneath the LST heatmap and markers).
    # ------------------------------------------------------------------
    if pollutants:
        for _name, data in pollutants.items():
            img_url = render_pollutant_layer(
                data["array"], cmap=data["cmap"],
            )
            folium.raster_layers.ImageOverlay(
                image=img_url,
                bounds=bounds,
                opacity=0.35,
                interactive=False,
            ).add_to(m)

    # ------------------------------------------------------------------
    # LST heatmap (middle layer)
    # ------------------------------------------------------------------
    img_url = render_lst_heatmap(lst_array, anomaly_indices)

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=0.60,
        interactive=False,
    ).add_to(m)

    # ------------------------------------------------------------------
    # Factory marker (top layer)
    # ------------------------------------------------------------------
    folium.Marker([lat, lon],
                  tooltip="Factory Location",
                  icon=folium.Icon(color="red", icon="industry", prefix="fa")
                  ).add_to(m)

    return m
