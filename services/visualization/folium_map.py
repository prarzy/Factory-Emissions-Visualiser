import folium


def create_map(
    lat: float,
    lon: float,
    tile_layers: list | None = None,
):
    """Build a Folium map with toggleable GEE raster tile layers.

    Parameters
    ----------
    tile_layers : list of dict, optional
        Each dict must contain ``url`` (GEE tile URL), ``name`` (display
        label), ``opacity`` (0–1), and ``show`` (bool).  Designed to
        accept the output of ``raster_layers.build_layer_entry``.
    """
    m = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True)

    # ----- Raster tile layers -----
    # Layers are added in order so the last-added draws on top.
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

    # ----- Factory marker (always on top) -----
    folium.Marker(
        [lat, lon],
        tooltip="Factory Location",
        icon=folium.Icon(color="red", icon="industry", prefix="fa"),
    ).add_to(m)

    # ----- Layer toggle control -----
    folium.LayerControl(collapsed=True).add_to(m)

    return m
