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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_segment_id(config_name: str, experiment: str, station: int, session: str, number: int) -> str:
    return f"{config_name}:{experiment}:S{int(station):02d}:{session}:{number:06d}"


def _source_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_canonical(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    power_columns = [
        "canonical_row_id", "experiment_id", "station_id", "station_name", "session_id",
        "timestamp_epoch_ms_original", "timestamp_utc", "ActivePowerL1", "Flow", "Pressure",
        "join_matched", *BOOLEAN_PLC, *MES_FIELDS,
    ]
    operation_columns = [
        "canonical_row_id", "experiment_id", "station_id", "station_name", "session_id",
        "timestamp_epoch_ms_original", "timestamp_utc", *BOOLEAN_PLC, *MES_FIELDS,
    ]
    with _source_connection(source) as connection:
        power = pd.read_sql_query(
            f"SELECT {','.join(power_columns)} FROM power_operation_fused "
            "WHERE session_id IS NOT NULL ORDER BY experiment_id,station_id,session_id,timestamp_epoch_ms_original,canonical_row_id",
            connection,
        )
        operation = pd.read_sql_query(
            f"SELECT {','.join(operation_columns)} FROM operation_log_canonical "
            "ORDER BY experiment_id,station_id,session_id,timestamp_epoch_ms_original,canonical_row_id",
            connection,
        )
    return power, operation


def fixed_assignments(power: pd.DataFrame, duration_seconds: int, config_name: str) -> pd.Series:
    labels = pd.Series(index=power.index, dtype="string")
    duration_ms = int(duration_seconds * 1000)
    for key, indices in power.groupby(["experiment_id", "station_id", "session_id"], sort=True).groups.items():
        timestamps = power.loc[indices, "timestamp_epoch_ms_original"].to_numpy(dtype=np.int64)
        numbers = ((timestamps - timestamps.min()) // duration_ms).astype(int) + 1
        labels.loc[indices] = [
            _stable_segment_id(config_name, str(key[0]), int(key[1]), str(key[2]), number)
            for number in numbers
        ]
    return labels


def _robust_change_score(frame: pd.DataFrame, channels: list[str], smoothing_bins: int) -> np.ndarray:
    smoothed = frame[channels].rolling(smoothing_bins, center=True, min_periods=1).median()
    differences = smoothed.diff().fillna(0.0)
    z_columns: list[np.ndarray] = []
    for column in channels:
        values = differences[column].to_numpy(dtype=float)
        center = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.nanstd(values))
        if not np.isfinite(scale) or scale <= 1e-12:
            z_columns.append(np.zeros_like(values))
        else:
            z_columns.append((values - center) / scale)
    return np.sqrt(np.sum(np.square(np.column_stack(z_columns)), axis=1))


def change_point_assignments(
    power: pd.DataFrame,
    config_name: str,
    channels: list[str],
    resample_seconds: int,
    smoothing_bins: int,
    robust_score_threshold: float,
    minimum_separation_seconds: int,
) -> tuple[pd.Series, pd.DataFrame]:
    labels = pd.Series(index=power.index, dtype="string")
    diagnostics: list[dict[str, object]] = []
    bin_ms = int(resample_seconds * 1000)
    minimum_bins = max(1, math.ceil(minimum_separation_seconds / resample_seconds))
    for key, indices in power.groupby(["experiment_id", "station_id", "session_id"], sort=True).groups.items():
        group = power.loc[indices]
        origin = int(group["timestamp_epoch_ms_original"].min())
        buckets = ((group["timestamp_epoch_ms_original"] - origin) // bin_ms).astype(int)
        binned = group.assign(_bucket=buckets).groupby("_bucket", sort=True)[channels].mean()
        score = _robust_change_score(binned, channels, smoothing_bins)
        peak_positions, properties = find_peaks(
            score,
            height=robust_score_threshold,
            distance=minimum_bins,
        )
        boundary_times = origin + binned.index.to_numpy(dtype=np.int64)[peak_positions] * bin_ms
        timestamps = group["timestamp_epoch_ms_original"].to_numpy(dtype=np.int64)
        numbers = np.searchsorted(boundary_times, timestamps, side="right") + 1
        labels.loc[indices] = [
            _stable_segment_id(config_name, str(key[0]), int(key[1]), str(key[2]), number)
            for number in numbers
        ]
        for position, boundary in zip(peak_positions, boundary_times, strict=True):
            diagnostics.append(
                {
                    "segmentation_config": config_name,
                    "experiment_id": key[0],
                    "station_id": int(key[1]),
                    "session_id": key[2],
                    "change_point_epoch_ms": int(boundary),
                    "change_point_score": float(score[position]),
                }
            )
    return labels, pd.DataFrame(diagnostics)


def plc_transition_times(operation: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in operation.groupby(["experiment_id", "station_id", "session_id"], sort=True):
        group = group.sort_values(["timestamp_epoch_ms_original", "canonical_row_id"]).copy()
        changed = pd.Series(False, index=group.index)
        for column in fields:
            values = group[column].astype("object").where(group[column].notna(), "__NULL__")
            changed |= values.ne(values.shift()) & values.shift().notna()
        rows.append(group.loc[changed, ["experiment_id", "station_id", "session_id", "timestamp_epoch_ms_original"]])
    if not rows:
        return pd.DataFrame(columns=["experiment_id", "station_id", "session_id", "timestamp_epoch_ms_original"])
    return pd.concat(rows, ignore_index=True)


def event_assignments(
    power: pd.DataFrame, operation: pd.DataFrame, fields: list[str], config_name: str
) -> tuple[pd.Series, pd.DataFrame]:
    transitions = plc_transition_times(operation, fields)
    by_group = {
        key: group["timestamp_epoch_ms_original"].drop_duplicates().sort_values().to_numpy(dtype=np.int64)
        for key, group in transitions.groupby(["experiment_id", "station_id", "session_id"], sort=True)
    }
    labels = pd.Series(index=power.index, dtype="string")
    for key, indices in power.groupby(["experiment_id", "station_id", "session_id"], sort=True).groups.items():
        boundaries = by_group.get(key, np.array([], dtype=np.int64))
        timestamps = power.loc[indices, "timestamp_epoch_ms_original"].to_numpy(dtype=np.int64)
        numbers = np.searchsorted(boundaries, timestamps, side="right") + 1
        labels.loc[indices] = [
            _stable_segment_id(config_name, str(key[0]), int(key[1]), str(key[2]), number)
            for number in numbers
        ]
    transitions = transitions.rename(columns={"timestamp_epoch_ms_original": "transition_epoch_ms"})
    return labels, transitions


def _group_slope(frame: pd.DataFrame, value_column: str) -> pd.Series:
    valid = frame[["segment_id", "timestamp_epoch_ms_original", value_column]].dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    starts = valid.groupby("segment_id")["timestamp_epoch_ms_original"].transform("min")
    x = (valid["timestamp_epoch_ms_original"] - starts) / 1000.0
    y = valid[value_column].astype(float)
    work = pd.DataFrame({"segment_id": valid["segment_id"], "x": x, "y": y})
    work["xx"] = work["x"] * work["x"]
    work["xy"] = work["x"] * work["y"]
    agg = work.groupby("segment_id").agg(n=("y", "size"), sx=("x", "sum"), sy=("y", "sum"), sxx=("xx", "sum"), sxy=("xy", "sum"))
    denominator = agg["n"] * agg["sxx"] - agg["sx"] ** 2
    slope = (agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]) / denominator.replace(0, np.nan)
    return slope.fillna(0.0)


def extract_features(segmented: pd.DataFrame, segmentation_type: str, config_name: str) -> pd.DataFrame:
    ordered = segmented.sort_values(["segment_id", "timestamp_epoch_ms_original", "canonical_row_id"]).copy()
    grouped = ordered.groupby("segment_id", sort=True, observed=True)
    features = grouped.agg(
        experiment_id=("experiment_id", "first"),
        station_id=("station_id", "first"),
        station_name=("station_name", "first"),
        session_id=("session_id", "first"),
        segment_start_epoch_ms=("timestamp_epoch_ms_original", "min"),
        segment_end_epoch_ms=("timestamp_epoch_ms_original", "max"),
        sample_count=("canonical_row_id", "size"),
        power_mean=("ActivePowerL1", "mean"),
        power_min=("ActivePowerL1", "min"),
        power_max=("ActivePowerL1", "max"),
        power_std=("ActivePowerL1", "std"),
        power_median=("ActivePowerL1", "median"),
        power_first=("ActivePowerL1", "first"),
        power_last=("ActivePowerL1", "last"),
        flow_mean=("Flow", "mean"), flow_min=("Flow", "min"), flow_max=("Flow", "max"),
        flow_std=("Flow", "std"), flow_first=("Flow", "first"), flow_last=("Flow", "last"),
        pressure_mean=("Pressure", "mean"), pressure_min=("Pressure", "min"), pressure_max=("Pressure", "max"),
        pressure_std=("Pressure", "std"), pressure_first=("Pressure", "first"), pressure_last=("Pressure", "last"),
    )
    features["duration_seconds"] = (features["segment_end_epoch_ms"] - features["segment_start_epoch_ms"]) / 1000.0
    features["sampling_density_hz"] = features["sample_count"] / features["duration_seconds"].replace(0, np.nan)
    features["power_range"] = features["power_max"] - features["power_min"]
    features["power_delta"] = features["power_last"] - features["power_first"]
    features["flow_delta"] = features["flow_last"] - features["flow_first"]
    features["pressure_delta"] = features["pressure_last"] - features["pressure_first"]
    for source, target in (("ActivePowerL1", "power_slope"), ("Flow", "flow_slope"), ("Pressure", "pressure_slope")):
        features[target] = _group_slope(ordered, source)
    same_segment = ordered["segment_id"].eq(ordered["segment_id"].shift())
    dt = ordered["timestamp_epoch_ms_original"].diff().where(same_segment, 0.0) / 1000.0
    previous_power = ordered["ActivePowerL1"].shift()
    ordered["_energy"] = ((ordered["ActivePowerL1"] + previous_power) / 2.0 * dt).where(same_segment, 0.0)
    features["energy_joule"] = ordered.groupby("segment_id")["_energy"].sum()
    features["segment_start_utc"] = pd.to_datetime(features["segment_start_epoch_ms"], unit="ms", utc=True)
    features["segment_end_utc"] = pd.to_datetime(features["segment_end_epoch_ms"], unit="ms", utc=True)
    features["segmentation_type"] = segmentation_type
    features["segmentation_config"] = config_name
    features = features.drop(columns=["power_first", "power_last", "flow_first", "flow_last", "pressure_first", "pressure_last"])
    for column in ("power_std", "flow_std", "pressure_std"):
        features[column] = features[column].fillna(0.0)
    return features.reset_index()[[*IDENTIFIERS, *DISCOVERY_FEATURES]]


def build_validation(segmented: pd.DataFrame) -> pd.DataFrame:
    ordered = segmented.sort_values(["segment_id", "timestamp_epoch_ms_original", "canonical_row_id"]).copy()
    grouped = ordered.groupby("segment_id", sort=True, observed=True)
    result = pd.DataFrame(index=grouped.size().index)
    result["validation_sample_count"] = grouped.size()
    result["matched_sample_fraction"] = grouped["join_matched"].mean()
    for column in BOOLEAN_PLC:
        result[f"{column}_true_fraction"] = grouped[column].mean()
        valid = ordered[column].notna() & ordered[column].shift().notna() & ordered["segment_id"].eq(ordered["segment_id"].shift())
        changes = (ordered[column] != ordered[column].shift()) & valid
        result[f"{column}_transition_count"] = changes.groupby(ordered["segment_id"]).sum().reindex(result.index, fill_value=0)
    for column in MES_FIELDS:
        result[f"{column}_first"] = grouped[column].first()
        result[f"{column}_last"] = grouped[column].last()
        result[f"{column}_distinct_count"] = grouped[column].nunique(dropna=True)
    return result.reset_index()


def segmentation_summary(features: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    duration = features["duration_seconds"]
    samples = features["sample_count"]
    merged = features[["segment_id"]].merge(validation[["segment_id", *[f"{x}_transition_count" for x in EVENT_FIELDS]]], on="segment_id")
    return {
        "segmentation_type": features["segmentation_type"].iloc[0],
        "segmentation_config": features["segmentation_config"].iloc[0],
        "segment_count": int(len(features)),
        "sample_count_min": int(samples.min()),
        "sample_count_p25": float(samples.quantile(.25)),
        "sample_count_median": float(samples.median()),
        "sample_count_p75": float(samples.quantile(.75)),
        "sample_count_max": int(samples.max()),
        "duration_min_seconds": float(duration.min()),
        "duration_p25_seconds": float(duration.quantile(.25)),
        "duration_median_seconds": float(duration.median()),
        "duration_p75_seconds": float(duration.quantile(.75)),
        "duration_p95_seconds": float(duration.quantile(.95)),
        "duration_max_seconds": float(duration.max()),
        "segments_under_1_second": int((duration < 1).sum()),
        "segments_over_60_seconds": int((duration > 60).sum()),
        "median_power_std": float(features["power_std"].median()),
        "median_power_range": float(features["power_range"].median()),
        "segments_with_plc_transition_fraction": float((merged.drop(columns="segment_id").sum(axis=1) > 0).mean()),
    }


def boundary_overlap(
    features: pd.DataFrame, transitions: pd.DataFrame, tolerance_seconds: float
) -> float:
    tolerance_ms = tolerance_seconds * 1000
    transition_groups = {
        key: np.sort(group["transition_epoch_ms"].to_numpy(dtype=np.int64))
        for key, group in transitions.groupby(["experiment_id", "station_id", "session_id"])
    }
    matched = 0
    considered = 0
    for key, group in features.groupby(["experiment_id", "station_id", "session_id"]):
        starts = np.sort(group["segment_start_epoch_ms"].unique())[1:]
        events = transition_groups.get(key, np.array([], dtype=np.int64))
        for start in starts:
            considered += 1
            position = np.searchsorted(events, start)
            distances = []
            if position < len(events): distances.append(abs(int(events[position]) - int(start)))
            if position > 0: distances.append(abs(int(events[position - 1]) - int(start)))
            matched += bool(distances and min(distances) <= tolerance_ms)
    return matched / considered if considered else float("nan")


def change_point_stability(
    left: pd.DataFrame, right: pd.DataFrame, tolerance_seconds: float
) -> dict[str, float]:
    tolerance_ms = tolerance_seconds * 1000
    left_groups = {key: np.sort(g["change_point_epoch_ms"].to_numpy(dtype=np.int64)) for key, g in left.groupby(["experiment_id", "station_id", "session_id"])}
    right_groups = {key: np.sort(g["change_point_epoch_ms"].to_numpy(dtype=np.int64)) for key, g in right.groupby(["experiment_id", "station_id", "session_id"])}
    matched = 0
    total_left = sum(map(len, left_groups.values()))
    total_right = sum(map(len, right_groups.values()))
    for key, values in left_groups.items():
        candidates = right_groups.get(key, np.array([], dtype=np.int64))
        available = set(range(len(candidates)))
        for value in values:
            if not available: continue
            index = min(available, key=lambda i: abs(int(candidates[i]) - int(value)))
            if abs(int(candidates[index]) - int(value)) <= tolerance_ms:
                matched += 1
                available.remove(index)
    precision = matched / total_left if total_left else 0.0
    recall = matched / total_right if total_right else 0.0
    return {"matched_change_points": matched, "left_count": total_left, "right_count": total_right, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


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


def choose_kmeans(model_comparison: pd.DataFrame, scaler: str, minimum_fraction: float) -> tuple[int, pd.DataFrame]:
    all_candidates = model_comparison[(model_comparison["algorithm"] == "kmeans") & (model_comparison["scaler"] == scaler)].copy()
    all_candidates["meets_minimum_cluster_fraction"] = all_candidates["smallest_cluster_fraction"] >= minimum_fraction
    candidates = all_candidates[all_candidates["meets_minimum_cluster_fraction"]].copy()
    if candidates.empty:
        candidates = all_candidates.copy()
    for column, ascending in (("silhouette", False), ("davies_bouldin", True), ("calinski_harabasz", False), ("stability_ari", False), ("largest_cluster_fraction", True)):
        candidates[f"rank_{column}"] = candidates[column].rank(ascending=ascending, method="min")
    candidates["multi_metric_rank_sum"] = candidates.filter(like="rank_").sum(axis=1)
    candidates["k"] = candidates["configuration"].str.extract(r"k=(\d+)").astype(int)
    chosen = candidates.sort_values(["multi_metric_rank_sum", "k"]).iloc[0]
    return int(chosen["k"]), candidates.sort_values("k")


def parameter_stability(labels_store: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for family in ("dbscan", "hdbscan"):
        for scaler in ("standard", "robust"):
            keys = sorted(key for key in labels_store if key.startswith(f"{family}|{scaler}|"))
            for left, right in zip(keys[:-1], keys[1:]):
                rows.append({
                    "stability_test": "parameter_change",
                    "configuration": f"{left} -> {right}",
                    "algorithm": family,
                    "scaler": scaler,
                    "adjusted_rand_index": adjusted_rand_score(labels_store[left], labels_store[right]),
                })
    return pd.DataFrame(rows)


def bootstrap_stability(matrix: np.ndarray, k: int, seeds: list[int], repetitions: int, fraction: float) -> pd.DataFrame:
    base = KMeans(n_clusters=k, n_init=20, random_state=seeds[0]).fit(matrix)
    base_labels = base.labels_
    rows = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(seeds[repetition % len(seeds)] + repetition * 1009)
        indices = rng.choice(len(matrix), max(k * 10, int(len(matrix) * fraction)), replace=True)
        model = KMeans(n_clusters=k, n_init=20, random_state=seeds[repetition % len(seeds)]).fit(matrix[indices])
        predicted = model.predict(matrix)
        rows.append({"stability_test": "bootstrap", "configuration": f"rep={repetition + 1};fraction={fraction}", "adjusted_rand_index": adjusted_rand_score(base_labels, predicted)})
    for model_seed in seeds[1:]:
        predicted = KMeans(n_clusters=k, n_init=20, random_state=model_seed).fit_predict(matrix)
        rows.append({"stability_test": "seed", "configuration": f"seed={model_seed}", "adjusted_rand_index": adjusted_rand_score(base_labels, predicted)})
    return pd.DataFrame(rows)


def experiment_stability(matrix: np.ndarray, metadata: pd.DataFrame, global_labels: np.ndarray, k: int, seed: int) -> pd.DataFrame:
    global_model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(matrix)
    rows = []
    for experiment, indices in metadata.groupby("experiment_id").groups.items():
        positions = metadata.index.get_indexer(indices)
        if len(positions) < k * 10:
            continue
        local = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(matrix[positions])
        cost = np.linalg.norm(local.cluster_centers_[:, None, :] - global_model.cluster_centers_[None, :, :], axis=2)
        local_idx, global_idx = linear_sum_assignment(cost)
        mapping = dict(zip(local_idx, global_idx, strict=True))
        aligned = np.array([mapping[label] for label in local.labels_])
        rows.append({
            "stability_test": "experiment_refit",
            "configuration": str(experiment),
            "adjusted_rand_index": adjusted_rand_score(global_labels[positions], aligned),
            "mean_matched_centroid_distance": float(cost[local_idx, global_idx].mean()),
            "row_count": len(positions),
        })
    return pd.DataFrame(rows)


def session_distribution_stability(metadata: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    work = metadata[["experiment_id", "station_id", "session_id"]].copy()
    work["cluster"] = labels
    global_distribution = np.bincount(labels, minlength=k) / len(labels)
    rows = []
    for key, group in work.groupby(["experiment_id", "station_id", "session_id"]):
        distribution = np.bincount(group["cluster"], minlength=k) / len(group)
        rows.append({"stability_test": "session_distribution", "configuration": "|".join(map(str, key)), "jensen_shannon_distance": float(jensenshannon(distribution, global_distribution)), "row_count": len(group)})
    return pd.DataFrame(rows)


def cluster_profiles(features: pd.DataFrame, validation: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    work = features.merge(validation, on="segment_id", validate="one_to_one").copy()
    work["cluster"] = labels
    rows = []
    for cluster, group in work.groupby("cluster", sort=True):
        row: dict[str, object] = {"cluster": int(cluster), "segment_count": len(group), "segment_fraction": len(group) / len(work)}
        for column in ["power_mean", "power_min", "power_max", "power_std", "energy_joule", "power_slope", "duration_seconds", "flow_mean", "pressure_mean"]:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_p10"] = float(group[column].quantile(.10))
            row[f"{column}_p90"] = float(group[column].quantile(.90))
        for column in BOOLEAN_PLC:
            row[f"{column}_true_fraction_mean"] = float(group[f"{column}_true_fraction"].mean())
            row[f"{column}_segments_with_transition_fraction"] = float((group[f"{column}_transition_count"] > 0).mean())
        for column in MES_FIELDS:
            row[f"{column}_segments_with_change_fraction"] = float((group[f"{column}_distinct_count"] > 1).mean())
        row["experiment_count"] = int(group["experiment_id"].nunique())
        row["station_count"] = int(group["station_id"].nunique())
        row["session_count"] = int(group["session_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def propose_states(profiles: pd.DataFrame) -> pd.DataFrame:
    result = profiles[["cluster", "segment_count", "segment_fraction"]].copy()
    power = profiles.set_index("cluster")["power_mean_median"]
    slope = profiles.set_index("cluster")["power_slope_median"]
    delta_scale = max(float(slope.abs().quantile(.75)), 1e-9)
    low_power = float(power.quantile(.25))
    high_power = float(power.quantile(.75))
    rows = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["cluster"])
        p = float(profile["power_mean_median"])
        s = float(profile["power_slope_median"])
        busy = float(profile["Busy_true_fraction_mean"])
        entry = float(profile["StationEntryxBG5_segments_with_transition_fraction"])
        exit_rate = float(profile["StationExitxBG6_segments_with_transition_fraction"])
        done = float(profile["Done_segments_with_transition_fraction"])
        evidence_energy = f"power median={p:.3f} W; slope median={s:.4f} W/s; energy median={profile['energy_joule_median']:.3f} J"
        evidence_plc = f"Busy true={busy:.1%}; entry transitions={entry:.1%}; exit transitions={exit_rate:.1%}; Done transitions={done:.1%}"
        state = "ambiguous"
        confidence = "low"
        reason = "No single physical interpretation is sufficiently separated."
        if p <= low_power and busy < .35:
            state, confidence, reason = "idle_ready", "medium", "Low relative power and low PLC Busy occupancy agree."
        elif (entry + exit_rate + done) >= .20 and busy < .60:
            state, confidence, reason = "loading_unloading_transfer", "medium", "PLC transition evidence dominates without sustained Busy."
        elif s > delta_scale and p >= low_power:
            state, confidence, reason = "ramp_up", "medium", "Positive power slope is a dominant cluster characteristic."
        elif s < -delta_scale and p >= low_power:
            state, confidence, reason = "ramp_down", "medium", "Negative power slope is a dominant cluster characteristic."
        elif p >= high_power and busy >= .35:
            state, confidence, reason = "processing", "medium", "High relative power and PLC Busy occupancy agree."
        rows.append({"cluster": cluster, "energy_evidence": evidence_energy, "plc_mes_evidence": evidence_plc, "proposed_state": state, "confidence": confidence, "decision_reason": reason, "pseudo_label_not_ground_truth": True})
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%S.%fZ")


def run(config_path: Path) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = config_path.parents[1]
    source = root / config["source"]["sqlite"]
    output_db = root / config["outputs"]["sqlite"]
    processed_dir = root / config["outputs"]["processed"]
    results_dir = root / config["outputs"]["results"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(source)
    raw_manifest_hash_before = sha256_file(root / "data/raw_manifest.csv")
    power, operation = load_canonical(source)

    assignments: dict[str, pd.Series] = {}
    metadata: dict[str, tuple[str, str]] = {}
    change_points: dict[str, pd.DataFrame] = {}
    for duration in config["segmentation"]["fixed"]["durations_seconds"]:
        name = f"fixed_{duration}s"
        assignments[name] = fixed_assignments(power, int(duration), name)
        metadata[name] = ("fixed", name)
    cp_config = config["segmentation"]["change_point"]
    for sensitivity, parameters in cp_config["configurations"].items():
        name = f"change_point_{sensitivity}"
        assignments[name], change_points[name] = change_point_assignments(
            power, name, list(cp_config["channels"]), int(cp_config["resample_seconds"]),
            int(parameters["smoothing_bins"]), float(parameters["robust_score_threshold"]),
            int(parameters["minimum_separation_seconds"]),
        )
        metadata[name] = ("change_point", name)
    event_name = "event_driven_plc_transitions"
    assignments[event_name], transitions = event_assignments(
        power, operation, list(config["segmentation"]["event_driven"]["transition_fields"]), event_name
    )
    metadata[event_name] = ("event_driven", event_name)

    feature_frames = []
    validation_frames = []
    summaries = []
    primary_name = config["segmentation"]["primary"]
    primary_segmented: pd.DataFrame | None = None
    for name, labels in assignments.items():
        segmented = power.copy()
        segmented["segment_id"] = labels
        features = extract_features(segmented, metadata[name][0], name)
        validation = build_validation(segmented)
        feature_frames.append(features)
        validation["segmentation_config"] = name
        validation_frames.append(validation)
        summary = segmentation_summary(features, validation)
        summary["plc_transition_boundary_overlap_fraction_1s"] = boundary_overlap(
            features, transitions, float(config["stability"]["plc_boundary_comparison_tolerance_seconds"])
        )
        summaries.append(summary)
        if name == primary_name:
            primary_segmented = segmented
    all_features = pd.concat(feature_frames, ignore_index=True)
    all_validation = pd.concat(validation_frames, ignore_index=True)
    segmentation_comparison = pd.DataFrame(summaries)
    cp_stability = change_point_stability(
        change_points["change_point_conservative"], change_points["change_point_sensitive"],
        float(config["stability"]["boundary_match_tolerance_seconds"]),
    )
    if primary_segmented is None:
        raise ValueError(f"Unknown primary segmentation: {primary_name}")
    primary_features = all_features[all_features["segmentation_config"] == primary_name].reset_index(drop=True)
    primary_validation = all_validation[all_validation["segmentation_config"] == primary_name].drop(columns="segmentation_config").reset_index(drop=True)
    eligible = primary_features["sample_count"] >= int(config["features"]["minimum_samples_for_clustering"])
    x_discovery = primary_features.loc[eligible, [*IDENTIFIERS, *DISCOVERY_FEATURES]].reset_index(drop=True)
    z_validation = x_discovery[["segment_id"]].merge(primary_validation, on="segment_id", how="left", validate="one_to_one")
    matrices, imputed, scaling_summary = prepare_scaled_matrices(
        x_discovery, DISCOVERY_FEATURES, list(config["scaling"]["group_by"]), list(config["scaling"]["candidates"])
    )
    model_comparison, labels_store, comparison_indices = compare_models(matrices, config["clustering"], int(config["seed"]))
    selected_scaler = config["scaling"]["selected"]
    configured_k = config["clustering"]["selection"].get("selected_k")
    if configured_k is None:
        selected_k, k_selection = choose_kmeans(model_comparison, selected_scaler, float(config["clustering"]["selection"]["minimum_cluster_fraction"]))
    else:
        selected_k = int(configured_k)
        _, k_selection = choose_kmeans(model_comparison, selected_scaler, float(config["clustering"]["selection"]["minimum_cluster_fraction"]))
    chosen_matrix = matrices[selected_scaler]
    selected_model = KMeans(n_clusters=selected_k, n_init=50, random_state=int(config["seed"])).fit(chosen_matrix)
    cluster_labels = selected_model.labels_
    seeds = [int(value) for value in config["clustering"]["seeds"]]
    stability = pd.concat([
        bootstrap_stability(chosen_matrix, selected_k, seeds, int(config["stability"]["bootstrap_repetitions"]), float(config["stability"]["bootstrap_fraction"])),
        experiment_stability(chosen_matrix, x_discovery, cluster_labels, selected_k, int(config["seed"])),
        session_distribution_stability(x_discovery, cluster_labels, selected_k),
        parameter_stability(labels_store),
    ], ignore_index=True, sort=False)
    profiles = cluster_profiles(x_discovery, z_validation, cluster_labels)
    mapping = propose_states(profiles)
    final_segments = primary_features.merge(
        pd.DataFrame({"segment_id": x_discovery["segment_id"], "cluster": cluster_labels}),
        on="segment_id", how="left", validate="one_to_one",
    ).merge(mapping[["cluster", "proposed_state", "confidence"]], on="cluster", how="left")
    final_segments = final_segments.rename(columns={"proposed_state": "inferred_state", "confidence": "inferred_state_confidence"})
    final_segments["inferred_state"] = final_segments["inferred_state"].fillna("unknown")
    final_segments["inferred_state_confidence"] = final_segments["inferred_state_confidence"].fillna("low")
    final_segments["inferred_state_is_pseudo_label"] = True
    cluster_distribution = (
        final_segments.groupby(["experiment_id", "station_id", "session_id", "cluster", "inferred_state"], dropna=False)
        .size().rename("segment_count").reset_index()
    )
    sample_labels = final_segments[["segment_id", "cluster", "inferred_state", "inferred_state_confidence"]]
    segmented_samples = primary_segmented[[
        "canonical_row_id", "experiment_id", "station_id", "station_name", "session_id",
        "timestamp_epoch_ms_original", "timestamp_utc", "ActivePowerL1", "Flow", "Pressure", "segment_id",
    ]].merge(sample_labels, on="segment_id", how="left", validate="many_to_one")
    scaled_frame = x_discovery[IDENTIFIERS].copy()
    scaled_frame[[f"scaled_{column}" for column in DISCOVERY_FEATURES]] = chosen_matrix

    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    with sqlite3.connect(output_db) as connection:
        tables = {
            "segmented_power_samples": segmented_samples,
            "segment_features_all": all_features,
            "X_discovery_energy_sensor": x_discovery,
            "X_discovery_energy_sensor_scaled": scaled_frame,
            "Z_plc_mes_validation": z_validation,
            "segmentation_comparison": segmentation_comparison,
            "change_points": pd.concat(change_points.values(), ignore_index=True),
            "plc_event_transitions": transitions,
            "clustering_comparison": model_comparison,
            "cluster_stability": stability,
            "cluster_profiles": profiles,
            "cluster_state_mapping": mapping,
            "cluster_distribution": cluster_distribution,
            "segments_with_inferred_state": final_segments,
        }
        for table, frame in tables.items():
            frame.to_sql(table, connection, index=False, if_exists="replace")
        connection.execute("CREATE UNIQUE INDEX idx_segment_final ON segments_with_inferred_state(segment_id)")
        connection.execute("CREATE INDEX idx_sample_segment ON segmented_power_samples(segment_id)")

    csv_outputs = {
        "segmented_power_samples.csv.gz": segmented_samples,
        "segment_features_all.csv.gz": all_features,
        "X_discovery_energy_sensor.csv.gz": x_discovery,
        "Z_plc_mes_validation.csv.gz": z_validation,
        "segments_with_inferred_state.csv.gz": final_segments,
    }
    for filename, frame in csv_outputs.items():
        _write_csv(frame, processed_dir / filename)
    report_outputs = {
        "segmentation_comparison.csv": segmentation_comparison,
        "change_point_stability.csv": pd.DataFrame([cp_stability]),
        "scaling_summary.csv.gz": scaling_summary,
        "clustering_comparison.csv": model_comparison,
        "kmeans_k_selection.csv": k_selection,
        "cluster_stability.csv": stability,
        "cluster_profiles.csv": profiles,
        "cluster_state_mapping.csv": mapping,
        "cluster_distribution.csv": cluster_distribution,
    }
    for filename, frame in report_outputs.items():
        compression = "gzip" if filename.endswith(".gz") else None
        frame.to_csv(results_dir / filename, index=False, compression=compression)

    source_hash_after = sha256_file(source)
    raw_manifest_hash_after = sha256_file(root / "data/raw_manifest.csv")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "classical_energy_state_discovery",
        "version": config["version"],
        "seed": config["seed"],
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "source_sqlite": str(source.relative_to(root)),
        "source_sqlite_sha256": source_hash_before,
        "source_sqlite_unchanged": source_hash_before == source_hash_after,
        "raw_manifest_sha256": raw_manifest_hash_before,
        "raw_manifest_unchanged": raw_manifest_hash_before == raw_manifest_hash_after,
        "source_power_rows_with_session": len(power),
        "segmentation_configs": list(assignments),
        "primary_segmentation": primary_name,
        "primary_segment_count": len(primary_features),
        "clustering_eligible_segments": len(x_discovery),
        "discovery_features": DISCOVERY_FEATURES,
        "plc_mes_excluded_from_discovery": [*BOOLEAN_PLC, *MES_FIELDS],
        "selected_scaler": selected_scaler,
        "selected_clustering_algorithm": "kmeans",
        "selected_cluster_count": selected_k,
        "change_point_stability": cp_stability,
        "inferred_state_counts": final_segments["inferred_state"].fillna("not_clustered").value_counts().to_dict(),
        "ambiguous_cluster_count": int((mapping["proposed_state"] == "ambiguous").sum()),
        "sqlite_output": str(output_db.relative_to(root)),
        "sqlite_output_sha256": sha256_file(output_db),
        "canonical_output_format": "SQLite",
        "companion_exports": "gzip CSV",
        "forbidden_steps_performed": [],
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
