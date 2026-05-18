import folium
import numpy as np


def _cluster_color(mean_z: float):
    """Map a mean Z‑score to a CSS colour string."""
    if mean_z >= 3.0:
        return "#dc2626"  # red
    if mean_z >= 2.5:
        return "#ea580c"  # orange
    return "#ca8a04"  # yellow


def create_map(
    lat: float,
    lon: float,
    tile_layers: list | None = None,
    clusters: list | None = None,
):
    """Build a Folium map with GEE tile layers and cluster hotspots.

    Parameters
    ----------
    tile_layers : list of dict, optional
        Each dict must contain ``url`` (GEE tile URL), ``name`` (display
        label), ``opacity`` (0–1), and ``show`` (bool).  Designed to
        accept the output of ``raster_layers.build_layer_entry``.
    clusters : list of dict, optional
        Output of ``clustering.cluster_anomalies``.  Each cluster is
        drawn as a coloured circle whose radius is proportional to its
        area, and a centroid marker.
    """
    m = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True)

    # ----- Raster tile layers -----
    if tile_layers:
        for entry in tile_layers:
            folium.TileLayer(
                tiles=entry["url"],
                attr="Google Earth Engine",
                name=entry["name"],
                overlay=True,
                opacity=entry.get("opacity", 0.6),
                show=entry.get("show", True),
                control=True,
            ).add_to(m)

    # ----- Thermal cluster overlays -----
    if clusters:
        # Group cluster circles under a single toggleable FeatureGroup
        fg = folium.FeatureGroup(name="Hotspot Clusters", show=True, control=True).add_to(m)

        for c in clusters:
            color = _cluster_color(c["mean_z_score"])

            # Approximate circle radius (metres) from area_km²
            radius_m = max(80, int(np.sqrt(c["area_km2"] / np.pi) * 1000))

            tooltip = (
                f"Cluster #{c['cluster_id']}  |  "
                f"{c['size']} px  |  "
                f"Z\u0304 = {c['mean_z_score']:.1f}  |  "
                f"{c['area_km2']:.2f} km\u00b2"
            )

            # Fill circle (semi‑transparent, proportional to area)
            folium.Circle(
                location=[c["centroid_lat"], c["centroid_lon"]],
                radius=radius_m,
                color=color,
                fill=True,
                fill_opacity=0.25,
                weight=1.5,
                tooltip=tooltip,
            ).add_to(fg)

            # Centroid dot
            folium.CircleMarker(
                location=[c["centroid_lat"], c["centroid_lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.9,
                weight=1,
            ).add_to(fg)

    # ----- Factory location marker (always on top) -----
    folium.Marker(
        [lat, lon],
        tooltip="Factory Location",
        icon=folium.Icon(color="red", icon="industry", prefix="fa"),
    ).add_to(m)

    # ----- Layer toggle control -----
    folium.LayerControl(collapsed=True).add_to(m)

    return m
