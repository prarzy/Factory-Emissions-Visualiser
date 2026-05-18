import io
import requests
import numpy as np
from sklearn.ensemble import IsolationForest


def analyze_anomalies(npy_url: str,
                      contamination: float = 0.05
                      ):
    response = requests.get(npy_url, timeout=60)
    response.raise_for_status()

    lst_array = np.load(io.BytesIO(response.content))
    lst_array = lst_array.astype(np.float32, copy=False)

    flat = lst_array.ravel()
    valid_mask = ~np.isnan(flat)
    flat_valid = flat[valid_mask].reshape(-1, 1)

    if flat_valid.size == 0:
        return lst_array, np.array([], dtype=int)

    clf = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ).fit(flat_valid)

    is_outlier = clf.predict(flat_valid) == -1
    anomaly_indices = np.where(valid_mask)[0][is_outlier]

    return lst_array, anomaly_indices


def calculate_emission_score(lst_array: np.ndarray, anomaly_indices: np.ndarray):
    flat = lst_array.ravel()

    if len(anomaly_indices) == 0:
        return 0.0

    anomaly_values = flat[anomaly_indices]
    anomaly_values = anomaly_values[~np.isnan(anomaly_values)]

    if anomaly_values.size == 0:
        return 0.0

    all_valid = flat[~np.isnan(flat)]
    temp_range = all_valid.max() - all_valid.min()

    if temp_range == 0:
        return 0.0

    mean_temp = all_valid.mean()
    severity = np.clip((anomaly_values.mean() - mean_temp) / temp_range, 0, 1)

    density = len(anomaly_values) / len(all_valid)

    score = (0.7 * severity + 0.3 * density) * 100
    return round(score, 2)
