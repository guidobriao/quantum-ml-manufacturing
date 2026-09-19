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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%S.%fZ")


def verify_frozen_inputs(root: Path, config: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    pairs = [
        ("discovery_sqlite", "discovery_sqlite_sha256"),
        ("common_sqlite", "common_sqlite_sha256"),
        ("segmentation_config", "segmentation_config_sha256"),
        ("inference_logic", "inference_logic_sha256"),
    ]
    for path_key, hash_key in pairs:
        path = root / config["frozen_inputs"][path_key]
        actual = sha256_file(path)
        expected = config["frozen_inputs"][hash_key]
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {path_key}: {actual} != {expected}")
        resolved[path_key] = actual
    return resolved


def load_segments(path: Path, features: list[str]) -> pd.DataFrame:
    columns = [
        "segment_id", "experiment_id", "station_id", "session_id",
        "segment_start_epoch_ms", "segment_end_epoch_ms", "segment_start_utc",
        "segment_end_utc", "inferred_state", *features,
    ]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(
            f"SELECT {','.join(columns)} FROM segments_with_inferred_state ORDER BY segment_start_epoch_ms,segment_id",
            connection,
        )
    return frame


def assign_temporal_split(frame: pd.DataFrame, split_config: dict[str, Any]) -> pd.DataFrame:
    """Assign train/validation/test by whole experiment (chronological holdout).

    Adds a `split` column; does NOT touch any target column. Target handling
    is the caller's responsibility (regression or classification).
    """
    sets = {
        experiment: "train" for experiment in split_config["train_experiments"]
    }
    sets.update({experiment: "validation"
                 for experiment in split_config["validation_experiments"]})
    sets.update({experiment: "test" for experiment in split_config["test_experiments"]})
    result = frame.copy()
    result["split"] = result["experiment_id"].map(sets).astype("string")
    unassigned = result["split"].isna().sum()
    if unassigned:
        raise ValueError(f"{unassigned} rows with unknown experiment_id; check split config")
    return result

def _temporal_quantiles(group: pd.DataFrame, count: int) -> pd.Index:
    ordered = group.sort_values(["segment_start_epoch_ms", "segment_id"])
    if count >= len(ordered):
        return ordered.index
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered.iloc[np.unique(positions)].index


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in frame.groupby("split", sort=False):
        counts = group["inferred_state"].value_counts()
        rows.append({
            "split": split, "segment_count": len(group),
            "eligible_count": int(group["supervised_eligible"].sum()),
            "idle_ready_count": int(counts.get("idle_ready", 0)),
            "loading_unloading_transfer_count": int(counts.get("loading_unloading_transfer", 0)),
            "unknown_count": int(counts.get("unknown", 0)),
            "experiments": json.dumps(sorted(group["experiment_id"].unique().tolist())),
            "start_utc": group["segment_start_utc"].min(), "end_utc": group["segment_end_utc"].max(),
            "station_count": int(group["station_id"].nunique()), "session_count": int(group["session_id"].nunique()),
        })
    return pd.DataFrame(rows)


