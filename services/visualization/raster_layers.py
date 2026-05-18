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
