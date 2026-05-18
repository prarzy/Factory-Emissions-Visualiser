import base64
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


def render_lst_heatmap(lst_array: np.ndarray,
                       anomaly_indices: np.ndarray | list[int] | None = None):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.axis('off')
    ax.imshow(lst_array, cmap='hot', vmin=np.nanmin(lst_array),
              vmax=np.nanmax(lst_array))

    if anomaly_indices is not None and len(anomaly_indices):
        yy, xx = np.unravel_index(anomaly_indices, lst_array.shape)
        ax.scatter(xx, yy, s=6, alpha=0.6, linewidths=0,
                   marker='o', edgecolors='none', c='cyan')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_pollutant_layer(array: np.ndarray,
                           cmap: str = 'viridis',
                           vmin: float | None = None,
                           vmax: float | None = None):
    """Render a 2-D array as a colour‑mapped PNG with nearest‑neighbour
    interpolation.

    Using ``interpolation='nearest'`` preserves the blocky pixel
    appearance of coarse‑resolution sensors (e.g. Sentinel‑5P at
    ~7 km / pixel) instead of smoothly interpolating to a higher
    resolution that the sensor cannot actually provide.
    """
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.axis('off')
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation='nearest')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
