"""Reproducible local pipeline through canonical power/operation fusion.

The module deliberately does not create windows, features, inferred states,
splits, clusters, classifiers, or quantum artefacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sqlite3
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .schemas import OPERATION_COLUMNS, POWER_COLUMNS, STATIONS, TABLE_SCHEMAS
from .sql_dump import inspect_dump, iter_copy_rows, row_signature


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    raw: Path
    manifest: Path
    archive_changes: Path
    processed: Path
    preprocessing: Path
    fusion: Path


def utc_iso(epoch_ms: int | float | None) -> str | None:
    if epoch_ms is None or pd.isna(epoch_ms):
        return None
    return datetime.fromtimestamp(float(epoch_ms) / 1000, tz=timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(config_path: Path) -> tuple[dict[str, Any], PipelinePaths]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = config_path.resolve().parents[1]
    paths = PipelinePaths(
        root=root,
        raw=root / config["paths"]["raw"],
        manifest=root / config["paths"]["manifest"],
        archive_changes=root / config["paths"]["archive_changes"],
        processed=root / config["paths"]["processed"],
        preprocessing=root / config["paths"]["preprocessing_results"],
        fusion=root / config["paths"]["fusion_results"],
    )
    return config, paths


def raw_state(paths: PipelinePaths) -> pd.DataFrame:
    with paths.manifest.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    manifest_by_path = {row["relative_path"]: row for row in manifest}
    actual_files = sorted(path for path in paths.raw.rglob("*") if path.is_file())
    actual_rel = {path.relative_to(paths.raw).as_posix() for path in actual_files}
    manifest_rel = set(manifest_by_path)
    if actual_rel != manifest_rel:
        raise RuntimeError(
            "raw/manifest mismatch: "
            f"missing={sorted(manifest_rel - actual_rel)}, untracked={sorted(actual_rel - manifest_rel)}"
        )
    rows: list[dict[str, object]] = []
    for path in actual_files:
        relative = path.relative_to(paths.raw).as_posix()
        digest = sha256_file(path)
        expected = manifest_by_path[relative]["sha256"]
        if digest != expected:
            raise RuntimeError(f"raw integrity failure for {relative}")
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative,
                "experiment_id": Path(relative).parts[0],
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "sha256": digest,
            }
        )
    return pd.DataFrame(rows)


def classify_source(name: str, extension: str) -> tuple[str, str]:
    lower = name.lower()
    if extension == ".sql":
        return "primary_sql_snapshot", "canonical_candidate"
    if lower.startswith("query_"):
        return "sparse_query_export", "validation_only"
    if lower.startswith("probe") or (lower.startswith("dump_") and extension == ".txt"):
        return "probe_opcua_log", "auxiliary_only"
    if lower.startswith("1_") or lower.startswith("2_fuse"):
        return "probe_annotated_capture", "auxiliary_only"
    if extension == ".xlsx":
        return "workbook", "documentation_or_validation_only"
    if extension == ".png":
        return "screenshot", "documentation_only"
    return "other", "inventory_only"


def counterpart_for(relative: str, existing: set[str]) -> str | None:
    if not relative.endswith("_new.sql"):
        return None
    ordinary = relative.removesuffix("_new.sql") + ".sql"
    return ordinary if ordinary in existing else None


def build_inventory(
    raw: pd.DataFrame, paths: PipelinePaths
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    inventory = raw.copy()
    roles = inventory.apply(lambda row: classify_source(row["name"], row["extension"]), axis=1)
    inventory["source_type"] = [item[0] for item in roles]
    inventory["pipeline_role"] = [item[1] for item in roles]

    sql_profiles: list[dict[str, object]] = []
    sql_files = inventory.loc[inventory["extension"] == ".sql", "relative_path"].tolist()
    existing = set(sql_files)
    selected: list[str] = []
    selection_rows: list[dict[str, object]] = []
    for relative in sorted(sql_files):
        if relative.endswith("_new.sql"):
            ordinary = counterpart_for(relative, existing)
            selected.append(relative)
            selection_rows.append(
                {
                    "experiment_id": Path(relative).parts[0],
                    "canonical_source_file": relative,
                    "ordinary_counterpart": ordinary,
                    "selection_reason": "_new preferred" if ordinary else "only _new available",
                }
            )
        elif relative.removesuffix(".sql") + "_new.sql" in existing:
            continue
        else:
            selected.append(relative)
            selection_rows.append(
                {
                    "experiment_id": Path(relative).parts[0],
                    "canonical_source_file": relative,
                    "ordinary_counterpart": None,
                    "selection_reason": "only ordinary dump available",
                }
            )

    selected_set = set(selected)
    for relative in sorted(sql_files):
        profile, _ = inspect_dump(paths.raw / relative)
        for table in profile:
            sql_profiles.append(
                {
                    "relative_path": relative,
                    "experiment_id": Path(relative).parts[0],
                    "is_canonical_snapshot": relative in selected_set,
                    **table,
                }
            )
    return inventory, pd.DataFrame(sql_profiles), pd.DataFrame(selection_rows), selected


def compare_dump_pairs(selection: pd.DataFrame, paths: PipelinePaths) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paired = selection[selection["ordinary_counterpart"].notna()]
    for item in paired.to_dict("records"):
        new_file = str(item["canonical_source_file"])
        ordinary_file = str(item["ordinary_counterpart"])
        new_profile, new_signatures = inspect_dump(paths.raw / new_file, collect_signatures=True)
        ordinary_profile, ordinary_signatures = inspect_dump(paths.raw / ordinary_file, collect_signatures=True)
        profiles = {
            "new": {row["table"]: row for row in new_profile},
            "ordinary": {row["table"]: row for row in ordinary_profile},
        }
        for table in sorted(set(profiles["new"]) | set(profiles["ordinary"])):
            new_counter = new_signatures.get(table, Counter())
            ordinary_counter = ordinary_signatures.get(table, Counter())
            schemas_equal = json.loads(profiles["new"].get(table, {}).get("columns", "[]")) == json.loads(
                profiles["ordinary"].get(table, {}).get("columns", "[]")
            )
            payload_equal = new_counter == ordinary_counter
            rows.append(
                {
                    "experiment_id": item["experiment_id"],
                    "table": table,
                    "ordinary_file": ordinary_file,
                    "new_file": new_file,
                    "ordinary_schema_names": profiles["ordinary"].get(table, {}).get("schema_names"),
                    "new_schema_names": profiles["new"].get(table, {}).get("schema_names"),
                    "columns_equal": schemas_equal,
                    "ordinary_rows": sum(ordinary_counter.values()),
                    "new_rows": sum(new_counter.values()),
                    "logical_payload_equal": payload_equal,
                    "only_in_ordinary": sum((ordinary_counter - new_counter).values()),
                    "only_in_new": sum((new_counter - ordinary_counter).values()),
                    "difference_class": "schema_name_only" if schemas_equal and payload_equal else "content_or_schema_difference",
                }
            )
    return pd.DataFrame(rows)


def stable_canonical_id(experiment: str, table: str, signature: str) -> str:
    return hashlib.sha256(f"{experiment}\0{table}\0{signature}".encode()).hexdigest()


def canonicalize_snapshots(
    selected: list[str], selection: pd.DataFrame, paths: PipelinePaths
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counterpart = dict(
        zip(selection["canonical_source_file"], selection["ordinary_counterpart"], strict=True)
    )
    canonical: dict[str, dict[str, dict[str, object]]] = {
        "tblPowerLog": {},
        "tblOperationLog": {},
        "tblMachineReport": {},
        "tblSensorsLog": {},
    }
    lineage_sources: dict[str, set[str]] = defaultdict(set)
    occurrence_count: Counter[str] = Counter()
    duplicate_summary: Counter[tuple[str, str]] = Counter()
    original_counts: Counter[tuple[str, str]] = Counter()
    lineage_path = paths.preprocessing / "row_lineage_occurrences.csv.gz"
    paths.preprocessing.mkdir(parents=True, exist_ok=True)

    with gzip.open(lineage_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "canonical_row_id",
                "experiment_id",
                "table",
                "source_file",
                "source_line_number",
                "ordinary_counterpart",
                "occurrence_status",
            ],
        )
        writer.writeheader()
        for relative in sorted(selected):
            experiment = Path(relative).parts[0]
            for row in iter_copy_rows(paths.raw / relative):
                if row.table not in canonical:
                    continue
                signature = row_signature(row.table, row.columns, row.values)
                canonical_id = stable_canonical_id(experiment, row.table, signature)
                original_counts[(experiment, row.table)] += 1
                occurrence_count[canonical_id] += 1
                status = "kept" if canonical_id not in canonical[row.table] else "duplicate_occurrence"
                if status == "duplicate_occurrence":
                    duplicate_summary[(experiment, row.table)] += 1
                else:
                    record = dict(zip(row.columns, row.values, strict=True))
                    record.update(
                        {
                            "experiment_id": experiment,
                            "canonical_row_id": canonical_id,
                            "canonical_source_file": relative,
                            "canonical_source_line_number": row.raw_line_number,
                            "ordinary_counterpart": counterpart.get(relative),
                        }
                    )
                    canonical[row.table][canonical_id] = record
                lineage_sources[canonical_id].add(relative)
                writer.writerow(
                    {
                        "canonical_row_id": canonical_id,
                        "experiment_id": experiment,
                        "table": row.table,
                        "source_file": relative,
                        "source_line_number": row.raw_line_number,
                        "ordinary_counterpart": counterpart.get(relative),
                        "occurrence_status": status,
                    }
                )

    frames: dict[str, pd.DataFrame] = {}
    for table, records in canonical.items():
        frame = pd.DataFrame(records.values())
        if not frame.empty:
            frame["lineage_source_files"] = frame["canonical_row_id"].map(
                lambda row_id: json.dumps(sorted(lineage_sources[row_id]), separators=(",", ":"))
            )
            frame["source_occurrence_count"] = frame["canonical_row_id"].map(occurrence_count).astype("Int64")
        frames[table] = frame

    count_rows: list[dict[str, object]] = []
    for experiment, table in sorted(original_counts):
        original = original_counts[(experiment, table)]
        canonical_count = int(
            (frames[table]["experiment_id"] == experiment).sum()
        ) if not frames[table].empty else 0
        count_rows.append(
            {
                "experiment_id": experiment,
                "table": table,
                "selected_snapshot_rows": original,
                "canonical_rows": canonical_count,
                "deduplicated_occurrences": original - canonical_count,
            }
        )

    conflicts: list[dict[str, object]] = []
    for table in ("tblPowerLog", "tblOperationLog"):
        frame = frames[table]
        if frame.empty:
            continue
        grouped = frame.groupby(["experiment_id", "ResourceID", "timeStamp"], dropna=False, sort=True)
        for key, group in grouped:
            if len(group) > 1:
                conflicts.append(
                    {
                        "experiment_id": key[0],
                        "table": table,
                        "ResourceID": key[1],
                        "timeStamp": key[2],
                        "distinct_row_count": len(group),
                        "canonical_row_ids": json.dumps(sorted(group["canonical_row_id"]), separators=(",", ":")),
                        "source_files": json.dumps(
                            sorted(set(group["canonical_source_file"])), separators=(",", ":")
                        ),
                    }
                )
    conflict_columns = [
        "experiment_id", "table", "ResourceID", "timeStamp", "distinct_row_count",
        "canonical_row_ids", "source_files",
    ]
    duplicate_columns = ["experiment_id", "table", "deduplicated_occurrences"]
    return frames, pd.DataFrame(count_rows), pd.DataFrame(conflicts, columns=conflict_columns), pd.DataFrame(
        [
            {"experiment_id": exp, "table": table, "deduplicated_occurrences": count}
            for (exp, table), count in sorted(duplicate_summary.items())
        ], columns=duplicate_columns
    )


def normalize_resource_columns(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    result = frame.copy()
    for column, kind, _ in TABLE_SCHEMAS[table]:
        if kind == "boolean":
            result[column] = result[column].astype("boolean")
        elif kind in {"integer", "int64"}:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
        elif kind == "numeric":
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Float64")
        elif kind == "string":
            result[column] = result[column].astype("string")
    result["resource_id_normalized"] = result["ResourceID"].astype("string").str.strip()
    power_match = result["resource_id_normalized"].str.extract(r"^(ene[1-8])(?:_|:|$)", expand=False)
    mes_match = result["resource_id_normalized"].str.extract(r"^(mes(?:10|20|30|40|50|60|70|80))(?:_|:|$)", expand=False)
    result["energy_resource_id"] = power_match.astype("string")
    result["mes_resource_id"] = mes_match.astype("string")
    prefix_to_station = {
        prefix: station
        for station, (energy, mes, _) in STATIONS.items()
        for prefix in (energy, mes)
    }
    prefix = power_match.fillna(mes_match)
    result["station_id"] = prefix.map(prefix_to_station).astype("Int64")
    result["station_name"] = result["station_id"].map(
        {station: values[2] for station, values in STATIONS.items()}
    ).astype("string")
    result["timestamp_epoch_ms_original"] = pd.to_numeric(result["timeStamp"], errors="coerce").astype("Int64")
    result["timestamp_utc"] = pd.to_datetime(
        result["timestamp_epoch_ms_original"], unit="ms", utc=True, errors="coerce"
    )
    result["timestamp_europe_rome"] = result["timestamp_utc"].dt.tz_convert("Europe/Rome")
    if table == "tblPowerLog":
        result["energy_resource_id"] = result["energy_resource_id"].astype("string")
    else:
        result["mes_resource_id"] = result["mes_resource_id"].astype("string")
    return result


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {name: None for name in ("min", "p01", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max")}
    probabilities = {"min": 0, "p01": .01, "p10": .10, "p25": .25, "p50": .50, "p75": .75, "p90": .90, "p95": .95, "p99": .99, "max": 1}
    return {name: float(clean.quantile(probability)) for name, probability in probabilities.items()}


def derive_session_boundaries(
    operation: pd.DataFrame, upper_tail_quantile: float, minimum_separation_ratio: float
) -> tuple[dict[tuple[str, int], dict[str, object]], pd.DataFrame]:
    """Derive a separate PLC gap threshold from each group's empirical gaps.

    The largest multiplicative jump among unique gaps at or above the group's
    empirical upper-tail quantile is located. If its separation ratio reaches
    the explicit configured minimum, the threshold is the geometric mean of
    the gaps around that jump. There is no global threshold in seconds.
    """
    events = operation[
        ["experiment_id", "station_id", "timestamp_epoch_ms_original"]
    ].dropna(subset=["station_id", "timestamp_epoch_ms_original"])
    events = events.drop_duplicates().sort_values(
        ["experiment_id", "station_id", "timestamp_epoch_ms_original"]
    )
    definitions: dict[tuple[str, int], dict[str, object]] = {}
    gap_rows: list[dict[str, object]] = []
    for (experiment, station), group in events.groupby(["experiment_id", "station_id"], sort=True):
        timestamps = group["timestamp_epoch_ms_original"].astype("int64").to_numpy()
        gaps = np.diff(timestamps)
        positive = gaps[gaps > 0]
        threshold: float | None = None
        tail_floor: float | None = None
        separation_ratio: float | None = None
        gap_below: float | None = None
        gap_above: float | None = None
        if positive.size >= 2:
            tail_floor = float(np.quantile(positive, upper_tail_quantile))
            unique_tail = np.unique(positive[positive >= tail_floor]).astype(float)
            if unique_tail.size >= 2:
                ratios = unique_tail[1:] / unique_tail[:-1]
                index = int(np.argmax(ratios))
                separation_ratio = float(ratios[index])
                gap_below = float(unique_tail[index])
                gap_above = float(unique_tail[index + 1])
                if separation_ratio >= minimum_separation_ratio:
                    threshold = math.sqrt(gap_below * gap_above)
        boundary_mask = np.zeros_like(gaps, dtype=bool) if threshold is None else gaps > threshold
        boundary_gaps = gaps[boundary_mask].astype(int).tolist()
        boundary_after = timestamps[:-1][boundary_mask].astype(int).tolist()
        definition = {
            "experiment_id": experiment,
            "station_id": int(station),
            "station_name": STATIONS[int(station)][2],
            "boundary_source": "tblOperationLog",
            "unique_plc_timestamps": len(timestamps),
            "positive_gap_count": len(positive),
            "rule": "largest_multiplicative_jump_in_upper_tail",
            "upper_tail_quantile": upper_tail_quantile,
            "upper_tail_floor_ms": tail_floor,
            "minimum_separation_ratio": minimum_separation_ratio,
            "largest_tail_separation_ratio": separation_ratio,
            "gap_below_jump_ms": gap_below,
            "gap_above_jump_ms": gap_above,
            "session_gap_threshold_ms": threshold,
            "boundary_count": len(boundary_after),
            "boundary_after_epoch_ms": boundary_after,
            "boundary_gap_ms": boundary_gaps,
        }
        definitions[(str(experiment), int(station))] = definition
        gap_rows.append({**definition, **{f"gap_{key}_ms": value for key, value in _quantiles(pd.Series(positive)).items()}})
    return definitions, pd.DataFrame(gap_rows)


def detect_synchronized_plc_silences(
    operation: pd.DataFrame,
    candidate_top_gaps_per_station: int,
    minimum_station_fraction: float,
    minimum_station_count: int,
    minimum_relative_duration: float,
) -> tuple[pd.DataFrame, dict[tuple[str, int], set[int]]]:
    """Detect long PLC silences supported by multiple stations.

    Only ``tblOperationLog`` timestamps are used. Each station contributes a
    small, explicit number of its longest empirical gaps. Atomic overlaps with
    the configured station consensus are merged. To avoid creating sessions
    from every coincident pause, only synchronized durations above a
    configured fraction of that experiment's longest synchronized duration
    are retained. The criterion is dimensionless; no duration is hard-coded.
    """
    diagnostics: list[dict[str, object]] = []
    synchronized_boundaries: dict[tuple[str, int], set[int]] = defaultdict(set)
    valid = operation.dropna(subset=["station_id", "timestamp_epoch_ms_original"])
    for experiment, experiment_group in valid.groupby("experiment_id", sort=True):
        active_stations = sorted(int(value) for value in experiment_group["station_id"].unique())
        required = max(
            int(minimum_station_count),
            int(math.ceil(len(active_stations) * minimum_station_fraction)),
        )
        candidates: list[dict[str, int]] = []
        for station, station_group in experiment_group.groupby("station_id", sort=True):
            timestamps = np.unique(
                station_group["timestamp_epoch_ms_original"].astype("int64").to_numpy()
            )
            gaps = np.diff(timestamps)
            if not len(gaps):
                continue
            count = min(int(candidate_top_gaps_per_station), len(gaps))
            indices = np.argsort(gaps, kind="stable")[-count:]
            for index in indices:
                candidates.append(
                    {
                        "station_id": int(station),
                        "gap_start_epoch_ms": int(timestamps[index]),
                        "gap_end_epoch_ms": int(timestamps[index + 1]),
                        "gap_duration_ms": int(gaps[index]),
                    }
                )
        points = sorted(
            {value for row in candidates for value in (row["gap_start_epoch_ms"], row["gap_end_epoch_ms"])}
        )
        qualifying: list[tuple[int, int, set[int]]] = []
        for start, end in zip(points[:-1], points[1:]):
            support = {
                row["station_id"]
                for row in candidates
                if row["gap_start_epoch_ms"] <= start and row["gap_end_epoch_ms"] >= end
            }
            if len(support) >= required:
                qualifying.append((start, end, support))
        merged: list[tuple[int, int, set[int]]] = []
        for start, end, support in qualifying:
            if merged and start == merged[-1][1]:
                old_start, _, old_support = merged[-1]
                merged[-1] = (old_start, end, old_support | support)
            else:
                merged.append((start, end, set(support)))
        if not merged:
            continue
        durations = np.array([end - start for start, end, _ in merged], dtype=float)
        duration_floor = float(durations.max() * minimum_relative_duration)
        retained_index = 0
        for start, end, _ in merged:
            duration = end - start
            if duration <= duration_floor:
                continue
            covering = [
                row for row in candidates
                if row["gap_start_epoch_ms"] < end and row["gap_end_epoch_ms"] > start
            ]
            support_stations = sorted({row["station_id"] for row in covering})
            if len(support_stations) < required:
                continue
            retained_index += 1
            for row in covering:
                synchronized_boundaries[(str(experiment), row["station_id"])].add(
                    row["gap_start_epoch_ms"]
                )
            diagnostics.append(
                {
                    "experiment_id": experiment,
                    "synchronized_silence_id": f"{experiment}-SYNC-{retained_index:03d}",
                    "core_start_epoch_ms": start,
                    "core_end_epoch_ms": end,
                    "core_start_timestamp_utc": pd.to_datetime(start, unit="ms", utc=True),
                    "core_end_timestamp_utc": pd.to_datetime(end, unit="ms", utc=True),
                    "core_duration_seconds": duration / 1000,
                    "active_station_count": len(active_stations),
                    "required_station_count": required,
                    "support_station_count": len(support_stations),
                    "support_station_fraction": len(support_stations) / len(active_stations),
                    "support_station_ids": json.dumps(support_stations, separators=(",", ":")),
                    "active_station_ids": json.dumps(active_stations, separators=(",", ":")),
                    "candidate_top_gaps_per_station": candidate_top_gaps_per_station,
                    "minimum_station_fraction": minimum_station_fraction,
                    "minimum_relative_duration_to_experiment_max": minimum_relative_duration,
                    "experiment_candidate_segment_count": len(merged),
                    "experiment_duration_floor_seconds": duration_floor / 1000,
                    "supporting_station_gaps": json.dumps(covering, separators=(",", ":")),
                }
            )
    columns = [
        "experiment_id", "synchronized_silence_id", "core_start_epoch_ms", "core_end_epoch_ms",
        "core_start_timestamp_utc", "core_end_timestamp_utc", "core_duration_seconds",
        "active_station_count", "required_station_count", "support_station_count",
        "support_station_fraction", "support_station_ids", "active_station_ids",
        "candidate_top_gaps_per_station", "minimum_station_fraction",
        "minimum_relative_duration_to_experiment_max", "experiment_candidate_segment_count",
        "experiment_duration_floor_seconds", "supporting_station_gaps",
    ]
    return pd.DataFrame(diagnostics, columns=columns), synchronized_boundaries


def add_synchronized_boundaries(
    definitions: dict[tuple[str, int], dict[str, object]],
    gap_report: pd.DataFrame,
    synchronized: dict[tuple[str, int], set[int]],
) -> tuple[dict[tuple[str, int], dict[str, object]], pd.DataFrame]:
    report = gap_report.copy()
    report["local_boundary_count"] = 0
    report["synchronized_boundary_count"] = 0
    for key, definition in definitions.items():
        local = set(int(value) for value in definition["boundary_after_epoch_ms"])
        extra = set(int(value) for value in synchronized.get(key, set()))
        definition["local_boundary_count"] = len(local)
        definition["synchronized_boundary_count"] = len(extra - local)
        definition["boundary_after_epoch_ms"] = sorted(local | extra)
        definition["boundary_count"] = len(local | extra)
        definition["boundary_gap_ms"] = sorted(
            set(int(value) for value in definition["boundary_gap_ms"])
        )
    for index, row in report.iterrows():
        definition = definitions[(str(row["experiment_id"]), int(row["station_id"]))]
        for column in (
            "local_boundary_count", "synchronized_boundary_count", "boundary_count",
            "boundary_after_epoch_ms",
        ):
            report.at[index, column] = definition[column]
    return definitions, report


def assign_sessions(frame: pd.DataFrame, definitions: dict[tuple[str, int], dict[str, object]]) -> pd.DataFrame:
    result = frame.copy()
    session_values = pd.Series(pd.NA, index=result.index, dtype="string")
    for (experiment, station), indices in result.dropna(subset=["station_id"]).groupby(
        ["experiment_id", "station_id"], sort=True
    ).groups.items():
        definition = definitions[(str(experiment), int(station))]
        boundaries = np.array(definition["boundary_after_epoch_ms"], dtype=np.int64)
        timestamps = result.loc[indices, "timestamp_epoch_ms_original"].astype("int64").to_numpy()
        numbers = np.searchsorted(boundaries, timestamps, side="left") + 1
        labels = [f"{experiment}-S{int(station):02d}-{number:03d}" for number in numbers]
        session_values.loc[indices] = labels
    result["session_id"] = session_values
    return result.sort_values(
        ["experiment_id", "station_id", "session_id", "timestamp_epoch_ms_original", "canonical_row_id"],
        na_position="last",
    ).reset_index(drop=True)


def build_plc_session_intervals(operation: pd.DataFrame) -> pd.DataFrame:
    intervals = operation.dropna(subset=["station_id", "session_id"]).groupby(
        ["experiment_id", "station_id", "station_name", "session_id"],
        observed=True,
        sort=True,
    ).agg(
        plc_first_epoch_ms=("timestamp_epoch_ms_original", "min"),
        plc_last_epoch_ms=("timestamp_epoch_ms_original", "max"),
        plc_first_timestamp_utc=("timestamp_utc", "min"),
        plc_last_timestamp_utc=("timestamp_utc", "max"),
        plc_record_count=("canonical_row_id", "size"),
    ).reset_index()
    intervals["interval_duration_seconds"] = (
        intervals["plc_last_epoch_ms"] - intervals["plc_first_epoch_ms"]
    ) / 1000
    return intervals


def assign_power_to_plc_sessions(power: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Assign power only inside inclusive PLC session intervals."""
    result = power.copy()
    session_values = pd.Series(pd.NA, index=result.index, dtype="string")
    for (experiment, station), indices in result.dropna(subset=["station_id"]).groupby(
        ["experiment_id", "station_id"], sort=True
    ).groups.items():
        group_intervals = intervals[
            (intervals["experiment_id"] == experiment) &
            (intervals["station_id"] == station)
        ].sort_values("plc_first_epoch_ms")
        if group_intervals.empty:
            continue
        starts = group_intervals["plc_first_epoch_ms"].astype("int64").to_numpy()
        ends = group_intervals["plc_last_epoch_ms"].astype("int64").to_numpy()
        labels = group_intervals["session_id"].astype("string").to_numpy()
        timestamps = result.loc[indices, "timestamp_epoch_ms_original"].astype("int64").to_numpy()
        positions = np.searchsorted(starts, timestamps, side="right") - 1
        valid = positions >= 0
        valid[valid] &= timestamps[valid] <= ends[positions[valid]]
        assigned = np.full(len(timestamps), None, dtype=object)
        assigned[valid] = labels[positions[valid]]
        session_values.loc[indices] = pd.array(assigned, dtype="string")
    result["session_id"] = session_values
    return result.sort_values(
        ["experiment_id", "station_id", "session_id", "timestamp_epoch_ms_original", "canonical_row_id"],
        na_position="last",
    ).reset_index(drop=True)


def sampling_report(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.dropna(subset=["station_id"]).groupby(
        ["experiment_id", "station_id", "session_id"], sort=True, dropna=False
    ):
        ordered = group.sort_values("timestamp_epoch_ms_original")
        timestamps = ordered["timestamp_epoch_ms_original"].astype("int64")
        gaps = timestamps.diff().dropna()
        positive = gaps[gaps > 0]
        duration_s = (timestamps.max() - timestamps.min()) / 1000 if len(timestamps) > 1 else 0.0
        quantiles = _quantiles(positive)
        rows.append(
            {
                "table": table,
                "experiment_id": key[0],
                "station_id": int(key[1]),
                "station_name": STATIONS[int(key[1])][2],
                "session_id": key[2],
                "observations": len(group),
                "first_timestamp_utc": ordered["timestamp_utc"].min(),
                "last_timestamp_utc": ordered["timestamp_utc"].max(),
                "duration_seconds": duration_s,
                "observations_per_second": len(group) / duration_s if duration_s > 0 else None,
                "delta_mean_ms": float(positive.mean()) if len(positive) else None,
                "delta_std_ms": float(positive.std(ddof=1)) if len(positive) > 1 else None,
                **{f"delta_{name}_ms": value for name, value in quantiles.items()},
                "zero_delta_count": int((gaps == 0).sum()),
                "negative_delta_count_after_canonical_sort": int((gaps < 0).sum()),
                "burst_gap_count_le_100ms": int((positive <= 100).sum()),
            }
        )
    return pd.DataFrame(rows)


def missing_zero_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table, frame in frames.items():
        if frame.empty:
            for column, kind, unit in TABLE_SCHEMAS[table]:
                rows.append(
                    {"table": table, "column": column, "unit": unit, "row_count": 0, "null_count": 0, "null_percent": None, "zero_count": 0, "zero_percent": None, "minimum": None, "maximum": None, "value_types": "[]"}
                )
            continue
        for column, _, unit in TABLE_SCHEMAS[table]:
            series = frame[column]
            count = len(series)
            null_count = int(series.isna().sum())
            non_null = series.dropna()
            zero_count = int((non_null == 0).sum()) if not non_null.empty else 0
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            value_types = sorted({type(value).__name__ for value in non_null.head(10000)})
            rows.append(
                {
                    "table": table,
                    "column": column,
                    "unit": unit,
                    "row_count": count,
                    "null_count": null_count,
                    "null_percent": 100 * null_count / count,
                    "zero_count": zero_count,
                    "zero_percent": 100 * zero_count / count,
                    "minimum": float(numeric.min()) if not numeric.empty else None,
                    "maximum": float(numeric.max()) if not numeric.empty else None,
                    "value_types": json.dumps(value_types, separators=(",", ":")),
                }
            )
    return pd.DataFrame(rows)


def resource_diagnostics(power: pd.DataFrame, operation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for table, frame in (("tblPowerLog", power), ("tblOperationLog", operation)):
        for resource, group in frame.groupby("ResourceID", dropna=False, sort=True):
            stations = sorted(int(value) for value in group["station_id"].dropna().unique())
            rows.append(
                {
                    "table": table,
                    "ResourceID_original": resource,
                    "resource_id_normalized": group["resource_id_normalized"].iloc[0],
                    "station_ids": json.dumps(stations),
                    "recognized": len(stations) == 1,
                    "ambiguous": len(stations) > 1,
                    "row_count": len(group),
                }
            )
    resources = pd.DataFrame(rows)
    presence_rows: list[dict[str, object]] = []
    all_keys = sorted(
        set(power.dropna(subset=["station_id"])[["experiment_id", "station_id"]].itertuples(index=False, name=None))
        | set(operation.dropna(subset=["station_id"])[["experiment_id", "station_id"]].itertuples(index=False, name=None))
    )
    for experiment, station in all_keys:
        power_rows = power[(power["experiment_id"] == experiment) & (power["station_id"] == station)]
        operation_rows = operation[(operation["experiment_id"] == experiment) & (operation["station_id"] == station)]
        presence_rows.append(
            {
                "experiment_id": experiment,
                "station_id": int(station),
                "station_name": STATIONS[int(station)][2],
                "power_rows": len(power_rows),
                "operation_rows": len(operation_rows),
                "power_only": len(power_rows) > 0 and len(operation_rows) == 0,
                "operation_only": len(operation_rows) > 0 and len(power_rows) == 0,
            }
        )
    return resources, pd.DataFrame(presence_rows)


def timestamp_diagnostics(
    power: pd.DataFrame, operation: pd.DataFrame, sql_schema: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table, frame in (("tblPowerLog", power), ("tblOperationLog", operation)):
        for experiment, group in frame.groupby("experiment_id", sort=True):
            expected_date = datetime.strptime(experiment.removeprefix("Exp"), "%Y%m%d").date()
            invalid = group["timestamp_utc"].isna()
            wrong_date = group["timestamp_utc"].dt.date.ne(expected_date) & ~invalid
            granular = group["timestamp_epoch_ms_original"].astype("string").str.fullmatch(r"\d{13}", na=False)
            profile = sql_schema[(sql_schema["experiment_id"] == experiment) & (sql_schema["table"] == table)]
            rows.append(
                {
                    "experiment_id": experiment,
                    "table": table,
                    "rows": len(group),
                    "invalid_timestamp_count": int(invalid.sum()),
                    "not_13_digit_epoch_ms_count": int((~granular).sum()),
                    "date_incompatible_with_experiment_count": int(wrong_date.sum()),
                    "raw_order_regressions_across_selected_dump_profiles": int(
                        profile.loc[profile["is_canonical_snapshot"], "timestamp_order_regressions_raw_order"].sum()
                    ),
                    "first_timestamp_utc": group["timestamp_utc"].min(),
                    "last_timestamp_utc": group["timestamp_utc"].max(),
                }
            )
    return pd.DataFrame(rows)


def snapshot_overlap_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table in ("tblPowerLog", "tblOperationLog"):
        frame = frames[table]
        if frame.empty:
            continue
        for experiment, exp_group in frame.groupby("experiment_id", sort=True):
            lineage_sets = exp_group["lineage_source_files"].map(lambda value: set(json.loads(value)))
            sources = sorted(set().union(*lineage_sets))
            for left_index, left in enumerate(sources):
                left_mask = lineage_sets.map(lambda values: left in values)
                left_rows = exp_group[left_mask]
                left_ids = set(left_rows["canonical_row_id"])
                for right in sources[left_index + 1:]:
                    right_mask = lineage_sets.map(lambda values: right in values)
                    right_rows = exp_group[right_mask]
                    right_ids = set(right_rows["canonical_row_id"])
                    intersection = left_ids & right_ids
                    left_start, left_end = left_rows["timeStamp"].min(), left_rows["timeStamp"].max()
                    right_start, right_end = right_rows["timeStamp"].min(), right_rows["timeStamp"].max()
                    temporal_overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
                    rows.append(
                        {
                            "experiment_id": experiment,
                            "table": table,
                            "left_snapshot": left,
                            "right_snapshot": right,
                            "left_rows": len(left_rows),
                            "right_rows": len(right_rows),
                            "shared_canonical_rows": len(intersection),
                            "temporal_overlap_ms": int(temporal_overlap),
                            "right_is_temporal_superset": bool(right_start <= left_start and right_end >= left_end),
                        }
                    )
    return pd.DataFrame(rows)


def fuse_power_operation(power: pd.DataFrame, operation: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    operation_groups = {
        key: group.copy()
        for key, group in operation.dropna(subset=["station_id", "session_id"]).groupby(
            ["experiment_id", "station_id", "session_id"], sort=False
        )
    }
    operation_payload = [column for column in OPERATION_COLUMNS if column not in {"ResourceID", "timeStamp"}]
    for key, power_group in power.dropna(subset=["station_id", "session_id"]).groupby(
        ["experiment_id", "station_id", "session_id"], sort=True
    ):
        left = power_group.sort_values(["timestamp_epoch_ms_original", "canonical_row_id"]).copy()
        right = operation_groups.get(key)
        if right is None or right.empty:
            merged = left.copy()
            merged["operation_row_id"] = pd.NA
            merged["plc_timestamp_epoch_ms"] = pd.NA
            merged["plc_timestamp_matched"] = pd.NaT
            merged["plc_resource_id_original"] = pd.NA
            merged["plc_source_file"] = pd.NA
            merged["plc_lineage_source_files"] = pd.NA
            merged["mes_resource_id_plc"] = pd.NA
            for column in operation_payload:
                merged[column] = pd.NA
            merged["join_ambiguous"] = False
        else:
            right = right.sort_values(["timestamp_epoch_ms_original", "canonical_row_id"]).copy()
            conflict_mask = right.duplicated("timestamp_epoch_ms_original", keep=False)
            right["join_ambiguous"] = conflict_mask
            right.loc[conflict_mask, "canonical_row_id"] = pd.NA
            rename = {
                "canonical_row_id": "operation_row_id",
                "timestamp_epoch_ms_original": "plc_timestamp_epoch_ms",
                "timestamp_utc": "plc_timestamp_matched",
                "ResourceID": "plc_resource_id_original",
                "canonical_source_file": "plc_source_file",
                "lineage_source_files": "plc_lineage_source_files",
                "mes_resource_id": "mes_resource_id_plc",
            }
            keep = [
                "timestamp_epoch_ms_original", "canonical_row_id", "timestamp_utc", "ResourceID",
                "canonical_source_file", "lineage_source_files", "mes_resource_id", "join_ambiguous",
                *operation_payload,
            ]
            right = right[keep].rename(columns=rename)
            merged = pd.merge_asof(
                left,
                right,
                left_on="timestamp_epoch_ms_original",
                right_on="plc_timestamp_epoch_ms",
                direction="backward",
                allow_exact_matches=True,
            )
        merged["join_ambiguous"] = merged["join_ambiguous"].fillna(False).astype(bool)
        merged["join_matched"] = merged["operation_row_id"].notna() & ~merged["join_ambiguous"]
        merged["state_age_seconds"] = (
            merged["timestamp_epoch_ms_original"] - merged["plc_timestamp_epoch_ms"]
        ) / 1000
        unmatched = ~merged["join_matched"]
        for column in [
            "operation_row_id", "plc_timestamp_epoch_ms", "plc_timestamp_matched",
            "plc_resource_id_original", "plc_source_file", "plc_lineage_source_files", "mes_resource_id_plc",
            *operation_payload,
        ]:
            if column in merged:
                merged.loc[unmatched, column] = pd.NA
        merged.loc[unmatched, "state_age_seconds"] = np.nan
        pieces.append(merged)

    unknown = power[power["station_id"].isna() | power["session_id"].isna()].copy()
    if not unknown.empty:
        unknown["operation_row_id"] = pd.NA
        unknown["plc_timestamp_epoch_ms"] = pd.NA
        unknown["plc_timestamp_matched"] = pd.NaT
        unknown["plc_resource_id_original"] = pd.NA
        unknown["plc_source_file"] = pd.NA
        unknown["plc_lineage_source_files"] = pd.NA
        unknown["mes_resource_id_plc"] = pd.NA
        for column in operation_payload:
            unknown[column] = pd.NA
        unknown["join_ambiguous"] = False
        unknown["join_matched"] = False
        unknown["state_age_seconds"] = np.nan
        pieces.append(unknown)
    fused = pd.concat(pieces, ignore_index=True)
    fused["session_id"] = fused["session_id"].astype("string")
    for column in (
        "operation_row_id", "plc_resource_id_original", "plc_source_file",
        "plc_lineage_source_files", "mes_resource_id_plc", "iResourceID",
    ):
        fused[column] = fused[column].astype("string")
    for column, kind, _ in TABLE_SCHEMAS["tblOperationLog"]:
        if column in {"ResourceID", "timeStamp", "iResourceID"}:
            continue
        if kind == "boolean":
            fused[column] = fused[column].astype("boolean")
        elif kind == "integer":
            fused[column] = pd.to_numeric(fused[column], errors="coerce").astype("Int64")
    fused["plc_timestamp_epoch_ms"] = pd.to_numeric(
        fused["plc_timestamp_epoch_ms"], errors="coerce"
    ).astype("Int64")
    fused["plc_timestamp_matched"] = pd.to_datetime(
        fused["plc_timestamp_matched"], utc=True, errors="coerce"
    )
    fused["timestamp_utc"] = pd.to_datetime(fused["timestamp_utc"], utc=True, errors="coerce")
    fused["timestamp_europe_rome"] = fused["timestamp_utc"].dt.tz_convert("Europe/Rome")
    fused["join_ambiguous"] = fused["join_ambiguous"].astype("boolean")
    fused["join_matched"] = fused["join_matched"].astype("boolean")
    fused["state_age_seconds"] = pd.to_numeric(
        fused["state_age_seconds"], errors="coerce"
    ).astype("Float64")
    return fused.sort_values(
        ["experiment_id", "station_id", "session_id", "timestamp_epoch_ms_original", "canonical_row_id"],
        na_position="last",
    ).reset_index(drop=True)


def fusion_diagnostics(
    fused: pd.DataFrame, operation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict[str, object]] = []
    for key, group in fused.groupby(["experiment_id", "station_id", "session_id"], dropna=False, sort=True):
        ages = group.loc[group["join_matched"], "state_age_seconds"]
        stats = _quantiles(ages)
        diagnostic_rows.append(
            {
                "experiment_id": key[0], "station_id": key[1], "session_id": key[2],
                "station_name": group["station_name"].dropna().iloc[0] if group["station_name"].notna().any() else None,
                "power_rows": len(group), "matched_power_rows": int(group["join_matched"].sum()),
                "unmatched_power_rows": int((~group["join_matched"]).sum()),
                "matched_percent": 100 * group["join_matched"].mean(),
                "state_age_mean_seconds": float(ages.mean()) if len(ages) else None,
                **{f"state_age_{name}_seconds": value for name, value in stats.items()},
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)

    usage = fused.loc[fused["join_matched"], "operation_row_id"].value_counts()
    operation_usage = operation[["canonical_row_id", "experiment_id", "station_id", "session_id", "timestamp_utc"]].copy()
    operation_usage["associated_power_rows"] = operation_usage["canonical_row_id"].map(usage).fillna(0).astype("Int64")
    usage_summary = operation_usage.groupby(["experiment_id", "station_id", "session_id"], dropna=False).agg(
        operation_rows=("canonical_row_id", "size"),
        operation_rows_used_zero_times=("associated_power_rows", lambda x: int((x == 0).sum())),
        operation_rows_used_once=("associated_power_rows", lambda x: int((x == 1).sum())),
        operation_rows_used_multiple_times=("associated_power_rows", lambda x: int((x > 1).sum())),
        mean_power_rows_per_operation=("associated_power_rows", "mean"),
        max_power_rows_per_operation=("associated_power_rows", "max"),
    ).reset_index()

    high_age = fused[fused["join_matched"]].nlargest(100, "state_age_seconds")[
        ["experiment_id", "station_id", "station_name", "session_id", "timestamp_utc", "plc_timestamp_matched", "state_age_seconds", "canonical_row_id", "operation_row_id", "canonical_source_file", "plc_source_file"]
    ]

    overlap_rows: list[dict[str, object]] = []
    for key, power_group in fused.dropna(subset=["station_id", "session_id"]).groupby(["experiment_id", "station_id", "session_id"]):
        op_group = operation[
            (operation["experiment_id"] == key[0]) &
            (operation["station_id"] == key[1]) &
            (operation["session_id"] == key[2])
        ]
        p_start, p_end = power_group["timestamp_epoch_ms_original"].min(), power_group["timestamp_epoch_ms_original"].max()
        if op_group.empty:
            o_start = o_end = None
            overlap_ms = 0
        else:
            o_start, o_end = op_group["timestamp_epoch_ms_original"].min(), op_group["timestamp_epoch_ms_original"].max()
            overlap_ms = max(0, min(p_end, o_end) - max(p_start, o_start))
        overlap_rows.append(
            {"experiment_id": key[0], "station_id": key[1], "session_id": key[2],
             "power_start_utc": utc_iso(p_start), "power_end_utc": utc_iso(p_end),
             "operation_start_utc": utc_iso(o_start), "operation_end_utc": utc_iso(o_end),
             "temporal_overlap_seconds": overlap_ms / 1000,
             "power_before_first_operation_seconds": max(0, (o_start - p_start) / 1000) if o_start is not None else (p_end - p_start) / 1000,
             "power_after_last_operation_seconds": max(0, (p_end - o_end) / 1000) if o_end is not None else (p_end - p_start) / 1000}
        )
    return diagnostics, operation_usage, usage_summary, pd.DataFrame(overlap_rows), high_age


def write_table(frame: pd.DataFrame, base: Path) -> list[str]:
    outputs: list[str] = []
    csv_path = base.with_suffix(".csv.gz")
    frame.to_csv(csv_path, index=False, compression="gzip")
    outputs.append(csv_path.name)
    parquet_path = base.with_suffix(".parquet")
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        frame.to_parquet(parquet_path, index=False)
        outputs.append(parquet_path.name)
    else:
        parquet_path.unlink(missing_ok=True)
    return outputs


def write_schema_dictionary(path: Path) -> pd.DataFrame:
    semantics = {
        "ResourceID": "Original source identifier, preserved unchanged",
        "timeStamp": "Original Unix epoch timestamp in milliseconds",
        "ReadyAtStationxBG1": "PLC field; query export alias WRK maps here",
        "DoneWorkingxBG9": "PLC field observed NULL in examined dumps; not reconstructed",
        "WorkPlanNo": "MES field observed NULL in examined dumps; not reconstructed",
        "StepNo": "MES field observed NULL in examined dumps; not reconstructed",
    }
    rows = []
    for table, columns in TABLE_SCHEMAS.items():
        for name, kind, unit in columns:
            rows.append(
                {"table": table, "column": name, "canonical_type": kind, "unit": unit,
                 "semantic_status": "confirmed", "description": semantics.get(name, "Original SQL field; no semantic transformation")}
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def save_sqlite(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    serializable = {}
    for name, frame in frames.items():
        copy = frame.copy()
        for column in copy.columns:
            non_null = copy[column].dropna()
            contains_timestamp = not non_null.empty and isinstance(non_null.iloc[0], pd.Timestamp)
            if pd.api.types.is_datetime64_any_dtype(copy[column].dtype) or contains_timestamp:
                copy[column] = copy[column].astype("string")
        serializable[name] = copy
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as connection:
        for name, frame in serializable.items():
            frame.to_sql(name, connection, if_exists="replace", index=False, chunksize=5000)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_power_join ON power_log_canonical (experiment_id, station_id, session_id, timestamp_epoch_ms_original)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_operation_join ON operation_log_canonical (experiment_id, station_id, session_id, timestamp_epoch_ms_original)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_fused_join ON power_operation_fused (experiment_id, station_id, session_id, timestamp_epoch_ms_original)")


def create_fusion_plots(fused: pd.DataFrame, output_dir: Path) -> list[str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in plots_dir.glob("fusion_*.png"):
        stale_plot.unlink()
    candidates = fused[(fused["station_id"] == 30) & fused["join_matched"]]
    sessions = candidates.groupby(["experiment_id", "session_id"]).size().sort_values(ascending=False).head(5)
    outputs: list[str] = []
    for (experiment, session), _ in sessions.items():
        sample = candidates[(candidates["experiment_id"] == experiment) & (candidates["session_id"] == session)].copy()
        if len(sample) > 10000:
            sample = sample.iloc[np.linspace(0, len(sample) - 1, 10000).astype(int)]
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(sample["timestamp_utc"], sample["ActivePowerL1"], linewidth=.7, label="ActivePowerL1 (W)")
        axis.set_ylabel("Power (W)")
        axis.set_xlabel("UTC")
        second = axis.twinx()
        for column, offset in [("Busy", 0), ("Done", 1.2), ("StationEntryxBG5", 2.4), ("StationExitxBG6", 3.6)]:
            values = sample[column].astype("Float64") + offset
            second.step(sample["timestamp_utc"], values, where="post", linewidth=.7, label=column)
        second.set_yticks([0, 1.2, 2.4, 3.6], ["Busy", "Done", "Entry", "Exit"])
        lines, labels = axis.get_legend_handles_labels()
        lines2, labels2 = second.get_legend_handles_labels()
        axis.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
        axis.set_title(f"Backward as-of check — {experiment}, station 30, {session}")
        figure.tight_layout()
        target = plots_dir / f"fusion_{experiment}_{session}.png"
        figure.savefig(target, dpi=140)
        plt.close(figure)
        outputs.append(target.relative_to(output_dir).as_posix())
    return outputs


def validate_pipeline(
    raw_before: pd.DataFrame, raw_after: pd.DataFrame, selection: pd.DataFrame,
    comparison: pd.DataFrame, power: pd.DataFrame, operation: pd.DataFrame,
    fused: pd.DataFrame, intervals: pd.DataFrame
) -> dict[str, bool]:
    before = raw_before.sort_values("relative_path").reset_index(drop=True)
    after = raw_after.sort_values("relative_path").reset_index(drop=True)
    matched = fused[fused["join_matched"]]
    operation_keys = operation.set_index("canonical_row_id")[["experiment_id", "station_id", "session_id"]]
    matched_keys = matched["operation_row_id"].map(operation_keys["experiment_id"])
    matched_stations = matched["operation_row_id"].map(operation_keys["station_id"])
    matched_sessions = matched["operation_row_id"].map(operation_keys["session_id"])
    outside = fused["session_id"].isna()
    plc_nullable_columns = [
        "operation_row_id", "plc_timestamp_epoch_ms", "plc_timestamp_matched",
        "plc_resource_id_original", "plc_source_file", "plc_lineage_source_files",
        "mes_resource_id_plc", *[column for column in OPERATION_COLUMNS if column not in {"ResourceID", "timeStamp"}],
    ]
    checks = {
        "raw_unchanged": before.equals(after),
        "new_selected_when_pair_exists": all(
            row.canonical_source_file.endswith("_new.sql")
            for row in selection.itertuples()
            if pd.notna(row.ordinary_counterpart)
        ),
        "ordinary_counterpart_in_lineage": all(
            power.loc[power["canonical_source_file"] == row.canonical_source_file, "ordinary_counterpart"].notna().all()
            and operation.loc[operation["canonical_source_file"] == row.canonical_source_file, "ordinary_counterpart"].notna().all()
            for row in selection.itertuples() if pd.notna(row.ordinary_counterpart)
        ),
        "paired_payloads_equal": bool(comparison["logical_payload_equal"].all()) if len(comparison) else True,
        "fused_row_count_equals_power": len(fused) == len(power),
        "power_rows_unique": fused["canonical_row_id"].is_unique,
        "one_operation_max_per_power": fused["operation_row_id"].notna().groupby(fused["canonical_row_id"]).sum().max() <= 1,
        "no_future_plc": bool((fused.loc[fused["join_matched"], "state_age_seconds"] >= 0).all()),
        "unmatched_age_is_null": bool(fused.loc[~fused["join_matched"], "state_age_seconds"].isna().all()),
        "same_experiment_join": bool((matched["experiment_id"].to_numpy() == matched_keys.to_numpy()).all()),
        "same_station_join": bool((matched["station_id"].to_numpy() == matched_stations.to_numpy()).all()),
        "same_session_join": bool((matched["session_id"].to_numpy() == matched_sessions.to_numpy()).all()),
        "operation_sessions_derived_for_every_plc_row": operation["session_id"].notna().all(),
        "power_outside_plc_intervals_is_unmatched": bool((~fused.loc[outside, "join_matched"]).all()),
        "power_outside_plc_intervals_has_null_plc": bool(fused.loc[outside, plc_nullable_columns].isna().all().all()),
        "plc_intervals_are_ordered": bool((intervals["plc_first_epoch_ms"] <= intervals["plc_last_epoch_ms"]).all()),
        "nullable_boolean_restored": all(str(fused[column].dtype) == "boolean" for column in ["Busy", "RFIDTagPresent", "Done", "StationEntryxBG5", "ReadyAtStationxBG1", "DoneWorkingxBG9", "StationExitxBG6", "join_matched", "join_ambiguous"]),
        "nullable_int64_restored": all(str(fused[column].dtype) == "Int64" for column in ["OperationNo", "WorkPlanNo", "OrderNo", "StepNo", "CarrierID", "OrderPosition", "PartNumber", "plc_timestamp_epoch_ms"]),
        "nullable_string_restored": all(str(fused[column].dtype) == "string" for column in ["operation_row_id", "plc_resource_id_original", "mes_resource_id_plc", "iResourceID"]),
        "plc_timestamp_is_utc": str(fused["plc_timestamp_matched"].dtype).startswith("datetime64[") and str(fused["plc_timestamp_matched"].dtype).endswith(", UTC]"),
        "forbidden_possible_reset_column_absent": "possible_reset_or_sentinel" not in fused.columns,
        "forbidden_inferred_state_absent": "inferred_state" not in fused.columns,
        "forbidden_feature_columns_absent": not any(str(column).startswith("feature_") for column in fused.columns),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"pipeline invariant failure: {failed}")
    return checks


def run(config_path: Path) -> dict[str, object]:
    config, paths = load_config(config_path)
    np.random.seed(int(config["seed"]))
    for directory in (paths.processed, paths.preprocessing, paths.fusion):
        directory.mkdir(parents=True, exist_ok=True)

    raw_before = raw_state(paths)
    deleted_name = "Exp20190320/dump_20190320_4154 (1).sql"
    deletion_state = {
        "deleted_file_absent": deleted_name not in set(raw_before["relative_path"]),
        "manifest_row_absent": deleted_name not in set(raw_before["relative_path"]),
        "archive_changes_exists": paths.archive_changes.exists(),
        "archive_changes_documents_deletion": False,
    }
    if paths.archive_changes.exists():
        deletion_state["archive_changes_documents_deletion"] = deleted_name in paths.archive_changes.read_text(
            encoding="utf-8", errors="replace"
        )

    inventory, sql_schema, selection, selected = build_inventory(raw_before, paths)
    comparison = compare_dump_pairs(selection, paths)
    frames, counts, conflicts, duplicate_summary = canonicalize_snapshots(selected, selection, paths)
    snapshot_overlap = snapshot_overlap_report(frames)
    power = normalize_resource_columns(frames["tblPowerLog"], "tblPowerLog")
    operation = normalize_resource_columns(frames["tblOperationLog"], "tblOperationLog")
    frames["tblPowerLog"] = power
    frames["tblOperationLog"] = operation

    definitions, gap_report = derive_session_boundaries(
        operation,
        float(config["sessions"]["upper_tail_quantile"]),
        float(config["sessions"]["minimum_separation_ratio"]),
    )
    synchronized_config = config["sessions"]["synchronized_silence"]
    synchronized_silences, synchronized_boundaries = detect_synchronized_plc_silences(
        operation,
        int(synchronized_config["candidate_top_gaps_per_station"]),
        float(synchronized_config["minimum_station_fraction"]),
        int(synchronized_config["minimum_station_count"]),
        float(synchronized_config["minimum_relative_duration_to_experiment_max"]),
    )
    definitions, gap_report = add_synchronized_boundaries(
        definitions, gap_report, synchronized_boundaries
    )
    operation = assign_sessions(operation, definitions)
    intervals = build_plc_session_intervals(operation)
    power = assign_power_to_plc_sessions(power, intervals)
    frames["tblPowerLog"] = power
    frames["tblOperationLog"] = operation

    sampling = pd.concat(
        [sampling_report(power, "tblPowerLog"), sampling_report(operation, "tblOperationLog")],
        ignore_index=True,
    )
    missing_zero = missing_zero_report(frames)
    resources, station_presence = resource_diagnostics(power, operation)
    timestamps = timestamp_diagnostics(power, operation, sql_schema)
    session_summary = gap_report[
        ["experiment_id", "station_id", "station_name", "session_gap_threshold_ms", "boundary_count"]
    ].copy()
    session_summary["session_count"] = session_summary["boundary_count"] + 1
    power_outside = power[power["session_id"].isna()].groupby(
        ["experiment_id", "station_id", "station_name"], dropna=False, as_index=False
    ).agg(
        power_rows_outside_plc_intervals=("canonical_row_id", "size"),
        first_power_timestamp_utc=("timestamp_utc", "min"),
        last_power_timestamp_utc=("timestamp_utc", "max"),
    )
    fused = fuse_power_operation(power, operation)
    diagnostics, operation_usage, usage_summary, overlap, high_age = fusion_diagnostics(fused, operation)
    state_age_review_threshold = float(config["diagnostics"]["state_age_review_seconds"])
    old_matches = fused[
        fused["join_matched"] & (fused["state_age_seconds"] > state_age_review_threshold)
    ].copy()
    old_match_summary = old_matches.groupby(
        ["experiment_id", "station_id", "station_name", "session_id"],
        as_index=False,
        dropna=False,
    ).agg(
        match_count=("canonical_row_id", "size"),
        minimum_state_age_seconds=("state_age_seconds", "min"),
        mean_state_age_seconds=("state_age_seconds", "mean"),
        maximum_state_age_seconds=("state_age_seconds", "max"),
        first_power_timestamp_utc=("timestamp_utc", "min"),
        last_power_timestamp_utc=("timestamp_utc", "max"),
    )

    inventory.to_csv(paths.preprocessing / "source_inventory.csv", index=False)
    sql_schema.to_csv(paths.preprocessing / "sql_schema_inventory.csv", index=False)
    sql_schema.groupby(["experiment_id", "table"], as_index=False)["row_count"].sum().to_csv(
        paths.preprocessing / "sql_rows_by_experiment_all_dumps.csv", index=False
    )
    selection.to_csv(paths.preprocessing / "canonical_dump_selection.csv", index=False)
    comparison.to_csv(paths.preprocessing / "ordinary_new_comparison.csv", index=False)
    snapshot_overlap.to_csv(paths.preprocessing / "snapshot_overlap.csv", index=False)
    counts.to_csv(paths.preprocessing / "canonical_row_counts.csv", index=False)
    conflicts.to_csv(paths.preprocessing / "same_resource_timestamp_conflicts.csv", index=False)
    duplicate_summary.to_csv(paths.preprocessing / "duplicate_occurrences.csv", index=False)
    missing_zero.to_csv(paths.preprocessing / "missing_and_zero_statistics.csv", index=False)
    sampling.to_csv(paths.preprocessing / "sampling_frequency.csv", index=False)
    gap_report.to_csv(paths.preprocessing / "gap_distribution_and_session_thresholds.csv", index=False)
    synchronized_silences.to_csv(
        paths.preprocessing / "synchronized_plc_silences.csv", index=False
    )
    session_summary.to_csv(paths.preprocessing / "session_summary.csv", index=False)
    intervals.to_csv(paths.preprocessing / "plc_session_intervals.csv", index=False)
    power_outside.to_csv(paths.fusion / "power_outside_plc_intervals.csv", index=False)
    resources.to_csv(paths.preprocessing / "resource_id_diagnostics.csv", index=False)
    station_presence.to_csv(paths.preprocessing / "station_presence.csv", index=False)
    timestamps.to_csv(paths.preprocessing / "timestamp_diagnostics.csv", index=False)
    write_schema_dictionary(paths.preprocessing / "column_dictionary.csv")
    (paths.preprocessing / "archive_changes_status.json").write_text(
        json.dumps(deletion_state, indent=2), encoding="utf-8"
    )
    diagnostics.to_csv(paths.fusion / "join_diagnostics.csv", index=False)
    operation_usage.to_csv(paths.fusion / "operation_row_usage.csv", index=False)
    usage_summary.to_csv(paths.fusion / "operation_usage_summary.csv", index=False)
    overlap.to_csv(paths.fusion / "temporal_overlap.csv", index=False)
    high_age.to_csv(paths.fusion / "highest_state_age_examples.csv", index=False)
    old_match_summary.to_csv(paths.fusion / "state_age_over_1800_summary.csv", index=False)
    old_matches.nlargest(200, "state_age_seconds")[
        [
            "experiment_id", "station_id", "station_name", "session_id", "timestamp_utc",
            "plc_timestamp_matched", "state_age_seconds", "canonical_row_id", "operation_row_id",
        ]
    ].to_csv(paths.fusion / "state_age_over_1800_examples.csv", index=False)

    produced = {
        "power_log_canonical": write_table(power, paths.processed / "power_log_canonical"),
        "operation_log_canonical": write_table(operation, paths.processed / "operation_log_canonical"),
        "power_operation_fused": write_table(fused, paths.processed / "power_operation_fused"),
    }
    save_sqlite(
        paths.processed / "common_pipeline.sqlite",
        {"power_log_canonical": power, "operation_log_canonical": operation, "power_operation_fused": fused},
    )
    plots = create_fusion_plots(fused, paths.fusion)
    raw_after = raw_state(paths)
    checks = validate_pipeline(
        raw_before, raw_after, selection, comparison, power, operation, fused, intervals
    )

    schema_metadata = {
        name: {column: str(dtype) for column, dtype in frame.dtypes.items()}
        for name, frame in {"power_log_canonical": power, "operation_log_canonical": operation, "power_operation_fused": fused}.items()
    }
    (paths.processed / "dataset_schemas.json").write_text(json.dumps(schema_metadata, indent=2), encoding="utf-8")

    empty_verification = {
        table: int(sql_schema.loc[sql_schema["table"] == table, "row_count"].sum())
        for table in ("tblMachineReport", "tblSensorsLog")
    }
    null_always = {
        column: bool(operation[column].isna().all())
        for column in ("DoneWorkingxBG9", "WorkPlanNo", "StepNo")
    }
    matched_ages = fused.loc[fused["join_matched"], "state_age_seconds"]
    global_age = {
        "mean_seconds": float(matched_ages.mean()) if len(matched_ages) else None,
        "median_seconds": float(matched_ages.median()) if len(matched_ages) else None,
        "p90_seconds": float(matched_ages.quantile(.90)) if len(matched_ages) else None,
        "p95_seconds": float(matched_ages.quantile(.95)) if len(matched_ages) else None,
        "p99_seconds": float(matched_ages.quantile(.99)) if len(matched_ages) else None,
        "max_seconds": float(matched_ages.max()) if len(matched_ages) else None,
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "config": str(config_path.relative_to(paths.root)),
        "seed": config["seed"],
        "raw_files": len(raw_before),
        "raw_bytes": int(raw_before["size_bytes"].sum()),
        "experiments": sorted(raw_before["experiment_id"].unique()),
        "sql_files": int((raw_before["extension"] == ".sql").sum()),
        "selected_canonical_dumps": selected,
        "deletion_state": deletion_state,
        "selected_snapshot_rows": int(counts["selected_snapshot_rows"].sum()),
        "canonical_rows": int(counts["canonical_rows"].sum()),
        "deduplicated_occurrences": int(counts["deduplicated_occurrences"].sum()),
        "conflict_groups": len(conflicts),
        "power_log_canonical_rows": len(power),
        "operation_log_canonical_rows": len(operation),
        "fused_rows": len(fused),
        "matched_power_rows": int(fused["join_matched"].sum()),
        "unmatched_power_rows": int((~fused["join_matched"]).sum()),
        "join_coverage_percent": 100 * float(fused["join_matched"].mean()),
        "state_age_seconds": global_age,
        "state_age_review_threshold_seconds": state_age_review_threshold,
        "matches_over_state_age_review_threshold": len(old_matches),
        "synchronized_plc_silence_count": len(synchronized_silences),
        "session_count": int(pd.concat([power["session_id"], operation["session_id"]]).nunique()),
        "power_rows_outside_plc_intervals": int(power["session_id"].isna().sum()),
        "sessions_by_experiment": {
            experiment: int(
                pd.concat([
                    power.loc[power["experiment_id"] == experiment, "session_id"],
                    operation.loc[operation["experiment_id"] == experiment, "session_id"],
                ]).nunique()
            )
            for experiment in sorted(power["experiment_id"].unique())
        },
        "empty_table_rows_across_all_sql_files": empty_verification,
        "required_always_null_columns": null_always,
        "parquet_written": bool(importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")),
        "canonical_storage": "SQLite",
        "companion_exports": ["gzip CSV"] + (["Parquet"] if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet") else []),
        "plots": plots,
        "outputs": produced,
        "checks": checks,
        "forbidden_steps_performed": [],
    }
    (paths.fusion / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readmes(paths, config, summary, gap_report)
    return summary


def write_readmes(paths: PipelinePaths, config: dict[str, Any], summary: dict[str, object], gap_report: pd.DataFrame) -> None:
    command = ".venv/bin/python scripts/run_common_pipeline.py --config configs/common_pipeline.yaml"
    readme = f"""# Common preprocessing and data-fusion pipeline

Run from the repository root:

```bash
{command}
```

The pipeline verifies every raw SHA-256 before and after execution, selects `_new` when paired,
deduplicates only complete normalized rows, derives PLC sessions exclusively from `tblOperationLog`,
and performs a backward as-of join grouped by experiment, station, and PLC session. Power rows
outside the inclusive first/last PLC timestamp of every session remain unmatched. It performs no windowing,
features, clustering, inferred-state construction, split, classification, or quantum operation.

The typed SQLite database is the sole canonical storage format. Compressed CSV files are companion
exports; Parquet is additionally written only when a local engine is already installed. This run
reports `parquet_written={summary['parquet_written']}`.

Configuration: `configs/common_pipeline.yaml`.

SQLite is canonical. When a CSV export is needed, restore the recorded nullable dtypes with:

```python
from qml_thesis.dataset_loader import load_schema_aware_csv
frame = load_schema_aware_csv(
    "data/processed/power_operation_fused.csv.gz",
    "data/processed/dataset_schemas.json",
)
```
"""
    (paths.processed / "README.md").write_text(readme, encoding="utf-8")
    session_text = """# Session definition

Session boundaries are derived independently for each `experiment_id + station_id` using only
unique timestamps from `tblOperationLog`. For every group, the algorithm examines the unique
positive gaps in the empirical upper tail (from the configured 95th percentile), finds its largest
multiplicative jump and, when the two sides are separated by at least the configured ratio, sets
the group-specific threshold to their geometric mean. The quantile, separation criterion and
formula are explicit in `configs/common_pipeline.yaml`; no fixed global duration threshold is used.

A second PLC-only level detects synchronized silences. Every station contributes its configured
number of longest empirical gaps; their temporal overlaps are retained only when supported by the
configured majority of active stations. Contiguous consensus segments are merged and, to prevent
over-segmentation by brief coincident pauses, a segment must exceed the configured fraction of the
longest synchronized duration observed in that experiment. These boundaries augment, but
never remove, the per-station boundaries. All counts, support stations and source gaps are written
to `results/preprocessing/synchronized_plc_silences.csv`.

Each session is valid only from its first through its last PLC record, inclusively. A power row is
assigned a `session_id` only inside one of those intervals. PLC state is never propagated outside
an interval or across experiment, station, or PLC-session boundaries.
"""
    (paths.preprocessing / "session_definition.md").write_text(session_text, encoding="utf-8")
    report = f"""# Verified common-pipeline report

- Raw archive: {summary['raw_files']} files, {summary['raw_bytes']} bytes; before/after SHA-256 check passed.
- Canonical SQL dumps: {len(summary['selected_canonical_dumps'])}.
- Selected snapshot rows: {summary['selected_snapshot_rows']}.
- Canonical power rows: {summary['power_log_canonical_rows']}.
- Canonical operation rows: {summary['operation_log_canonical_rows']}.
- Deduplicated complete-row occurrences: {summary['deduplicated_occurrences']}.
- Same ResourceID/timestamp conflict groups: {summary['conflict_groups']}.
- Sessions: {summary['session_count']}.
- Power rows outside PLC session intervals: {summary['power_rows_outside_plc_intervals']}.
- Fused rows: {summary['fused_rows']} (exactly one per canonical power row).
- Join coverage: {summary['join_coverage_percent']:.3f}%.
- Synchronized PLC silences: {summary['synchronized_plc_silence_count']}.
- Matches with state age above {summary['state_age_review_threshold_seconds']:.0f} s: {summary['matches_over_state_age_review_threshold']}.
- Empty tables verified: {summary['empty_table_rows_across_all_sql_files']}.
- Always-NULL PLC/MES fields verified: {summary['required_always_null_columns']}.
- Canonical storage format: SQLite. Compressed CSV files are companion exports.
- Parquet available locally: {summary['parquet_written']}.
- Forbidden downstream steps performed: none.
"""
    (paths.fusion / "README.md").write_text(report, encoding="utf-8")
