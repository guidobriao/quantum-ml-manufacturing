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



# TODO(phase-2): label projection onto fixed windows with purity threshold,
# using plc_state_machine.classify_frame; duration distributions per state.
