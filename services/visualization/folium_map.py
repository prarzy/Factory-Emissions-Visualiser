import folium
import numpy as np

from services.visualization.raster_layers import render_lst_heatmap


def create_map(lat: float, lon: float,
               lst_array: np.ndarray,
               anomaly_indices: np.ndarray | list[int]):
    m = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True)

    img_url = render_lst_heatmap(lst_array, anomaly_indices)
    span_deg = 0.05
    bounds = [[lat - span_deg, lon - span_deg],
              [lat + span_deg, lon + span_deg]]

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=bounds,
        opacity=0.60,
        interactive=False
    ).add_to(m)

    folium.Marker([lat, lon],
                  tooltip="Factory Location",
                  icon=folium.Icon(color="red", icon="industry", prefix="fa")
                  ).add_to(m)

    return m
