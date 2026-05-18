"""Multi‑signal fusion for industrial emission confidence scoring.

Replaces the old single‑signal (Z‑score‑only) emission score with a
weighted rule‑based system that combines thermal, atmospheric, and
spectral indicators.  Every weight is exposed as a configurable
constant so the system remains fully interpretable — no ML black box.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Configurable component weights  (sum ≈ 1.0)
# ---------------------------------------------------------------------------
# Rationale:
#   * Thermal anomalies are the most direct indicator of industrial heat
#     output → highest weight.
#   * NO₂ is a well‑known tracer of combustion → second.
#   * Cluster persistence separates transient thermal noise from sustained
#     industrial activity.
#   * NDBI confirms the surface is built‑up (reduces false positives from
#     bare soil / water thermal anomalies).
#   * NDVI suppression captures vegetation stress around facilities.
#   * SO₂ is often scrubbed in modern plants → lower weight.
# ---------------------------------------------------------------------------

WEIGHTS = {
    "thermal": 0.30,
    "no2": 0.20,
    "cluster_persistence": 0.15,
    "ndbi": 0.15,
    "ndvi_suppression": 0.10,
    "so2": 0.10,
}


# ---------------------------------------------------------------------------
# Component sub‑score helpers
# ---------------------------------------------------------------------------
# Each returns a value in [0, 100] using piecewise‑linear rules.
# Thresholds are documented alongside the formula.
# ---------------------------------------------------------------------------


def _score_thermal(mean_z: float) -> float:
    """Thermal anomaly intensity from the Z‑score climatology.

    * Z ≤ 2.0 → 0   (below anomaly threshold)
    * 2.0 < Z < 4.0 → linear ramp
    * Z ≥ 4.0 → 100 (extreme anomaly)
    """
    if mean_z <= 2.0:
        return 0.0
    if mean_z >= 4.0:
        return 100.0
    return (mean_z - 2.0) / 2.0 * 100.0


def _score_no2(mean_val: float) -> float:
    """Tropospheric NO₂ column density (mol / m²).

    Background air is ≈1e‑6; urban plumes run 5–30e‑6; heavy industrial
    areas can exceed 50e‑6.
    """
    bkg, mod, high = 2e-6, 15e-6, 50e-6
    if mean_val <= bkg:
        return 0.0
    if mean_val >= high:
        return 100.0
    if mean_val <= mod:
        return (mean_val - bkg) / (mod - bkg) * 50.0
    return 50.0 + (mean_val - mod) / (high - mod) * 50.0


def _score_so2(mean_val: float) -> float:
    """SO₂ column density (mol / m²).

    Background ≈0.5e‑6; volcanic / industrial hotspots can reach 20e‑6.
    Most modern plants keep SO₂ low thanks to scrubbers.
    """
    bkg, mod, high = 1e-6, 5e-6, 20e-6
    if mean_val <= bkg:
        return 0.0
    if mean_val >= high:
        return 100.0
    if mean_val <= mod:
        return (mean_val - bkg) / (mod - bkg) * 50.0
    return 50.0 + (mean_val - mod) / (high - mod) * 50.0


def _score_clusters(clusters: list | None) -> float:
    """Cluster persistence — more / larger clusters → higher confidence.

    A single thermal pixel could be noise; multiple spatially‑coherent
    clusters indicate a sustained emission source.
    """
    if not clusters:
        return 0.0
    n = len(clusters)
    largest = max(c["size"] for c in clusters)
    # Base score from cluster count
    if n == 1:
        score = 25.0
    elif n == 2:
        score = 50.0
    else:
        score = min(75.0, 50.0 + (n - 2) * 12.0)
    # Bonus for a dominant hotspot
    if largest >= 50:
        score += 25.0
    elif largest >= 20:
        score += 15.0
    return min(score, 100.0)


def _score_ndbi(mean_val: float) -> float:
    """Normalised Difference Built‑up Index — higher is more built‑up.

    Bare soil / vegetation are <0; urban / industrial surfaces are >0.
    The industrial mask already enforces NDBI > 0.2, so we score from
    that threshold upward.
    """
    if mean_val <= 0.2:
        return 10.0
    if mean_val >= 0.4:
        return 100.0
    return 10.0 + (mean_val - 0.2) / 0.2 * 90.0


def _score_ndvi_suppression(mean_val: float) -> float:
    """NDVI suppression — very low NDVI near industry indicates stress.

    The industrial mask caps NDVI at 0.3, so values near 0 imply bare
    ground / stressed vegetation rather than natural surfaces.
    """
    if mean_val >= 0.25:
        return 0.0
    if mean_val <= 0.05:
        return 100.0
    # Linear ramp: 0.25→0, 0.05→100
    return (0.25 - mean_val) / 0.2 * 100.0


# ---------------------------------------------------------------------------
# Category thresholds
# ---------------------------------------------------------------------------

_CATEGORY_BINS = [
    (34, "Low"),      # 0  ≤ score < 34
    (67, "Medium"),   # 34 ≤ score < 67
]
# 67 ≤ score        → "High"


def _categorise(score: float) -> str:
    for threshold, label in _CATEGORY_BINS:
        if score < threshold:
            return label
    return "High"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_emission_score(
    *,
    z_score_map: np.ndarray | None = None,
    anomaly_indices: np.ndarray | None = None,
    s5p_data: dict | None = None,
    clusters: list | None = None,
    ndvi_mean: float | None = None,
    ndbi_mean: float | None = None,
    weights: dict | None = None,
):
    """Multi‑signal industrial emission confidence score.

    Parameters
    ----------
    z_score_map, anomaly_indices :
        Output of ``temporal.detect_temporal_anomalies``.  Used to
        derive *mean Z‑score* of anomalous pixels.
    s5p_data :
        Output of ``sentinel5p.fetch_all_pollutants``.  Uses the
        ``"mean"`` key for NO₂ and SO₂.
    clusters :
        Output of ``clustering.cluster_anomalies``.
    ndvi_mean, ndbi_mean :
        Mean spectral indices over the industrial footprint (returned by
        ``temporal.detect_temporal_anomalies``).
    weights :
        Override the default ``WEIGHTS`` dict.

    Returns
    -------
    score : float
        Confidence score in **[0, 100]**.
    category : str
        ``"Low"``, ``"Medium"``, or ``"High"``.
    """
    w = weights if weights is not None else WEIGHTS

    # -- thermal --
    thermal_score = 0.0
    if z_score_map is not None and anomaly_indices is not None and len(anomaly_indices):
        z_vals = z_score_map.ravel()[anomaly_indices]
        z_vals = z_vals[~np.isnan(z_vals)]
        if len(z_vals):
            thermal_score = _score_thermal(float(np.mean(z_vals)))

    # -- NO₂, SO₂ --
    no2_score = so2_score = 0.0
    if s5p_data:
        if "NO2" in s5p_data:
            no2_score = _score_no2(s5p_data["NO2"]["mean"])
        if "SO2" in s5p_data:
            so2_score = _score_so2(s5p_data["SO2"]["mean"])

    # -- cluster persistence --
    cluster_score = _score_clusters(clusters)

    # -- NDBI --
    ndbi_score = _score_ndbi(ndbi_mean) if ndbi_mean is not None else 0.0

    # -- NDVI suppression --
    ndvi_score = (
        _score_ndvi_suppression(ndvi_mean) if ndvi_mean is not None else 0.0
    )

    # -- weighted fusion --
    score = (
        w["thermal"] * thermal_score
        + w["no2"] * no2_score
        + w["so2"] * so2_score
        + w["cluster_persistence"] * cluster_score
        + w["ndbi"] * ndbi_score
        + w["ndvi_suppression"] * ndvi_score
    )

    return round(score, 1), _categorise(score)
