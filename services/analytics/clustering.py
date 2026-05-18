import numpy as np
from sklearn.cluster import DBSCAN


# Approximate conversion: 1° latitude ≈ 111 000 m
_DEG_TO_M = 111_000


def cluster_anomalies(
    lst_array: np.ndarray,
    anomaly_indices: np.ndarray,
    z_score_map: np.ndarray,
    lat: float,
    lon: float,
    eps_km: float = 0.3,
    min_samples: int = 3,
):
    """Group anomalous pixels into contiguous hotspot regions using DBSCAN.

    Parameters
    ----------
    lst_array : np.ndarray
        2‑D array of LST (°C); used to infer raster shape.
    anomaly_indices : np.ndarray
        Raveled 1‑D indices where *Z > 2*.
    z_score_map : np.ndarray
        2‑D array of Z‑scores (same shape as *lst_array*).
    lat, lon : float
        Centre of the analysed region.
    eps_km : float
        DBSCAN neighbourhood radius in kilometres.  Default 0.3 km
        (≈3 pixels at 100 m resolution).
    min_samples : int
        Minimum points to form a dense cluster.  Default 3.

    Returns
    -------
    clusters : list[dict]
        Each entry::

            {
                "cluster_id": int,
                "size": int,               # number of pixels
                "mean_z_score": float,      # mean Z‑score of the cluster
                "centroid_lat": float,
                "centroid_lon": float,
                "area_km2": float,          # approximate (pixels × 0.01)
            }

        Sorted by *size* descending.  Noise points (label == −1) are
        excluded.
    labels : np.ndarray
        Cluster label for every entry in *anomaly_indices* (−1 = noise).
    """
    rows, cols = lst_array.shape

    # ---- convert raveled indices to (row, col) pixel coords ----
    yy, xx = np.unravel_index(anomaly_indices, (rows, cols))

    # ---- pixel → geographic coords ----
    # The GEE download box is point.buffer(10 000).bounds(), which we
    # approximate as ±0.05° around the centre.  This is accurate enough
    # for clustering within a 20 km tile.
    span = 0.05
    pixel_lats = lat + span * (1 - 2 * yy / (rows - 1)) if rows > 1 else np.full_like(yy, lat)
    pixel_lons = lon + span * (2 * xx / (cols - 1) - 1) if cols > 1 else np.full_like(xx, lon)

    coords = np.column_stack([pixel_lats, pixel_lons])

    # ---- DBSCAN ----
    # Convert epsilon to degrees (good enough for a 20 km region).
    eps_deg = eps_km / 111.0
    db = DBSCAN(eps=eps_deg, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords)

    if len(set(labels)) <= 1 and -1 in set(labels):
        return [], labels   # all noise

    # ---- build cluster records ----
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        n = int(mask.sum())
        clat = float(np.mean(pixel_lats[mask]))
        clon = float(np.mean(pixel_lons[mask]))
        z_vals = z_score_map.ravel()[anomaly_indices[mask]]
        mean_z = float(np.nanmean(z_vals))

        # Approx area: each pixel ≈ 100 m × 100 m = 0.01 km²
        area = round(n * 0.01, 3)

        clusters.append({
            "cluster_id": int(label),
            "size": n,
            "mean_z_score": round(mean_z, 2),
            "centroid_lat": clat,
            "centroid_lon": clon,
            "area_km2": area,
        })

    clusters.sort(key=lambda c: c["size"], reverse=True)

    # Re-number clusters so the biggest is #1
    for i, c in enumerate(clusters, 1):
        c["cluster_id"] = i

    return clusters, labels
