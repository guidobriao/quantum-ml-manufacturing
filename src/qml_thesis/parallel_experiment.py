"""Classical segmentation and energy-state discovery downstream of frozen fusion.

This module reads the approved SQLite dataset without modifying it. It does not
import or execute any quantum-computing package.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.spatial.distance import jensenshannon
import sklearn
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
import yaml


BOOLEAN_PLC = [
    "Busy", "RFIDTagPresent", "Done", "StationEntryxBG5",
    "ReadyAtStationxBG1", "DoneWorkingxBG9", "StationExitxBG6",
]
EVENT_FIELDS = [
    "Busy", "Done", "StationEntryxBG5", "ReadyAtStationxBG1", "StationExitxBG6",
]
MES_FIELDS = ["OperationNo", "WorkPlanNo", "OrderNo", "StepNo", "CarrierID", "iResourceID", "OrderPosition", "PartNumber"]
SENSOR_COLUMNS = ["ActivePowerL1", "Flow", "Pressure"]
DISCOVERY_FEATURES = [
    "duration_seconds", "sample_count", "sampling_density_hz",
    "power_mean", "power_min", "power_max", "power_std", "power_median",
    "power_range", "power_delta", "power_slope", "energy_joule",
    "flow_mean", "flow_min", "flow_max", "flow_std", "flow_delta", "flow_slope",
    "pressure_mean", "pressure_min", "pressure_max", "pressure_std",
    "pressure_delta", "pressure_slope",
]
IDENTIFIERS = [
    "segment_id", "experiment_id", "station_id", "station_name", "session_id",
    "segment_start_epoch_ms", "segment_end_epoch_ms", "segment_start_utc",
    "segment_end_utc", "segmentation_type", "segmentation_config",
]


def prepare_scaled_matrices(
    x_frame: pd.DataFrame, feature_columns: list[str], group_columns: list[str], scalers: list[str]
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    raw = x_frame[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    imputed = pd.DataFrame(index=raw.index, columns=feature_columns, dtype=float)
    scaling_rows: list[dict[str, object]] = []
    for key, indices in x_frame.groupby(group_columns, sort=True).groups.items():
        values = raw.loc[indices]
        medians = values.median()
        imputed.loc[indices] = values.fillna(medians)
    global_medians = raw.median().fillna(0.0)
    imputed = imputed.fillna(global_medians).astype(float)
    matrices = {name: np.zeros_like(imputed.to_numpy(dtype=float)) for name in scalers}
    for key, indices in x_frame.groupby(group_columns, sort=True).groups.items():
        positions = x_frame.index.get_indexer(indices)
        values = imputed.loc[indices].to_numpy(dtype=float)
        for name in scalers:
            scaler = StandardScaler() if name == "standard" else RobustScaler(quantile_range=(25, 75))
            transformed = scaler.fit_transform(values)
            transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
            matrices[name][positions] = transformed
            center = scaler.mean_ if name == "standard" else scaler.center_
            scaling_rows.append({
                **dict(zip(group_columns, key if isinstance(key, tuple) else (key,), strict=True)),
                "scaler": name,
                "row_count": len(indices),
                "center": json.dumps(dict(zip(feature_columns, np.asarray(center, dtype=float), strict=True)), separators=(",", ":")),
                "scale": json.dumps(dict(zip(feature_columns, np.asarray(scaler.scale_, dtype=float), strict=True)), separators=(",", ":")),
                "outliers_removed": False,
            })
    return matrices, imputed, pd.DataFrame(scaling_rows)


def _internal_metrics(matrix: np.ndarray, labels: np.ndarray, metric_limit: int, seed: int) -> dict[str, float | int | None]:
    keep = labels >= 0
    matrix = matrix[keep]
    labels = labels[keep]
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None, "evaluated_rows": int(len(labels))}
    if len(labels) > metric_limit:
        rng = np.random.default_rng(seed)
        selection = np.sort(rng.choice(len(labels), metric_limit, replace=False))
        matrix = matrix[selection]
        labels = labels[selection]
    return {
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "evaluated_rows": int(len(labels)),
    }


def _model_row(
    family: str, config: str, scaler: str, labels: np.ndarray, matrix: np.ndarray,
    metric_limit: int, seed: int, stability: float | None = None,
) -> dict[str, object]:
    non_noise = labels[labels >= 0]
    unique, counts = np.unique(non_noise, return_counts=True)
    metrics = _internal_metrics(matrix, labels, metric_limit, seed)
    return {
        "algorithm": family, "configuration": config, "scaler": scaler,
        "fit_rows": len(labels), "cluster_count_excluding_noise": len(unique),
        "noise_count": int((labels < 0).sum()), "noise_fraction": float((labels < 0).mean()),
        "smallest_cluster_count": int(counts.min()) if len(counts) else 0,
        "smallest_cluster_fraction": float(counts.min() / len(labels)) if len(counts) else 0.0,
        "largest_cluster_fraction": float(counts.max() / len(labels)) if len(counts) else 0.0,
        "stability_ari": stability, **metrics,
    }


def compare_models(
    matrices: dict[str, np.ndarray], config: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    n = len(next(iter(matrices.values())))
    rng = np.random.default_rng(seed)
    sample_size = min(n, int(config["comparison_sample_max"]))
    sample_indices = np.sort(rng.choice(n, sample_size, replace=False))
    rows: list[dict[str, object]] = []
    labels_store: dict[str, np.ndarray] = {}
    metric_limit = int(config["metric_sample_max"])
    seeds = [int(value) for value in config["seeds"]]
    for scaler_name, full_matrix in matrices.items():
        matrix = full_matrix[sample_indices]
        for k in config["k_range"]:
            seed_labels = []
            for model_seed in seeds:
                seed_labels.append(KMeans(n_clusters=int(k), n_init=20, random_state=model_seed).fit_predict(matrix))
            stability = float(np.mean([adjusted_rand_score(seed_labels[0], values) for values in seed_labels[1:]]))
            key = f"kmeans|{scaler_name}|k={k}"
            labels_store[key] = seed_labels[0]
            rows.append(_model_row("kmeans", f"k={k}", scaler_name, seed_labels[0], matrix, metric_limit, seed, stability))

            gmm_labels = []
            for model_seed in seeds:
                gmm = GaussianMixture(n_components=int(k), covariance_type=config["gmm_covariance_type"], random_state=model_seed, n_init=2, max_iter=300)
                gmm_labels.append(gmm.fit_predict(matrix))
            stability = float(np.mean([adjusted_rand_score(gmm_labels[0], values) for values in gmm_labels[1:]]))
            key = f"gmm|{scaler_name}|k={k}"
            labels_store[key] = gmm_labels[0]
            rows.append(_model_row("gmm", f"k={k};covariance={config['gmm_covariance_type']}", scaler_name, gmm_labels[0], matrix, metric_limit, seed, stability))

        max_min_samples = max(int(v) for v in config["dbscan"]["min_samples"])
        distances = NearestNeighbors(n_neighbors=max_min_samples).fit(matrix).kneighbors(matrix, return_distance=True)[0]
        for min_samples in config["dbscan"]["min_samples"]:
            kth = distances[:, int(min_samples) - 1]
            for quantile in config["dbscan"]["eps_quantiles"]:
                eps = float(np.quantile(kth, float(quantile)))
                labels = DBSCAN(eps=eps, min_samples=int(min_samples), n_jobs=-1).fit_predict(matrix)
                key = f"dbscan|{scaler_name}|min_samples={min_samples};eps={eps:.6g}"
                labels_store[key] = labels
                rows.append(_model_row("dbscan", f"min_samples={min_samples};eps={eps:.6g};kdist_q={quantile}", scaler_name, labels, matrix, metric_limit, seed))

        for min_cluster_size in config["hdbscan"]["min_cluster_size"]:
            for min_samples in config["hdbscan"]["min_samples"]:
                labels = HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples), n_jobs=-1, copy=True).fit_predict(matrix)
                key = f"hdbscan|{scaler_name}|min_cluster_size={min_cluster_size};min_samples={min_samples}"
                labels_store[key] = labels
                rows.append(_model_row("hdbscan", f"min_cluster_size={min_cluster_size};min_samples={min_samples}", scaler_name, labels, matrix, metric_limit, seed))
    return pd.DataFrame(rows), labels_store, sample_indices



from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from .qml_common import evaluate_predictions  # noqa: E402

def classical_models(config: dict[str, Any], seed: int, y_train: np.ndarray) -> dict[str, Any]:
    cfg = config["classical"]
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))
    return {
        "LogisticRegression": LogisticRegression(**cfg["logistic_regression"], random_state=seed),
        "SVM_RBF": SVC(**cfg["svm"], random_state=seed),
        "RandomForest": RandomForestClassifier(**cfg["random_forest"], random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(**cfg["xgboost"], random_state=seed, n_jobs=-1, scale_pos_weight=negative / positive, eval_metric="logloss"),
    }

def run_classical(
    config: dict[str, Any], seed: int, arrays: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, confusions, predictions = [], [], []
    x_train, y_train = arrays["train"]
    for name, model in classical_models(config, seed, y_train).items():
        start = perf_counter(); model.fit(x_train, y_train); training = perf_counter() - start
        for split in ("validation", "test"):
            x, y = arrays[split]
            start = perf_counter(); predicted = model.predict(x); inference = perf_counter() - start
            row, confusion = evaluate_predictions(name, "fair_6_feature", split, y, predicted, training, inference)
            rows.append(row); confusions.append(confusion)
            predictions.extend({"model": name, "condition": "fair_6_feature", "split": split, "row": i, "truth": int(t), "prediction": int(p)} for i, (t, p) in enumerate(zip(y, predicted, strict=True)))
    return pd.DataFrame(rows), pd.concat(confusions, ignore_index=True), pd.DataFrame(predictions)


# TODO(phase-2): quantum kernel k-means, QSVC/VQC vs classical on
# energy-only features, evaluation vs PLC ground truth (ARI/NMI/purity).
