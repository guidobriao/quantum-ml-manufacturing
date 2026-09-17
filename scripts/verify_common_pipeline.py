#!/usr/bin/env python3
"""Independent read-only audit of common-pipeline outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3

from qml_thesis.common_pipeline import stable_canonical_id
from qml_thesis.sql_dump import iter_copy_rows, row_signature


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "raw_manifest.csv"
SELECTION = ROOT / "results" / "preprocessing" / "canonical_dump_selection.csv"
DATABASE = ROOT / "data" / "processed" / "common_pipeline.sqlite"
SUMMARY = ROOT / "results" / "data_fusion" / "pipeline_summary.json"
SCHEMAS = ROOT / "data" / "processed" / "dataset_schemas.json"
SYNCHRONIZED_SILENCES = ROOT / "results" / "preprocessing" / "synchronized_plc_silences.csv"
OLD_MATCH_SUMMARY = ROOT / "results" / "data_fusion" / "state_age_over_1800_summary.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    actual = sorted(path for path in RAW.rglob("*") if path.is_file())
    expected_paths = {row["relative_path"] for row in manifest}
    actual_paths = {path.relative_to(RAW).as_posix() for path in actual}
    checks["raw_paths_match_manifest"] = expected_paths == actual_paths
    checks["raw_sha256_match_manifest"] = all(
        digest(RAW / row["relative_path"]) == row["sha256"] for row in manifest
    )
    details["raw_files"] = len(actual)
    details["raw_bytes"] = sum(path.stat().st_size for path in actual)

    with SELECTION.open(newline="", encoding="utf-8") as stream:
        selection = list(csv.DictReader(stream))
    selected = [row["canonical_source_file"] for row in selection]
    checks["new_selected_for_every_pair"] = all(
        row["canonical_source_file"].endswith("_new.sql")
        for row in selection
        if row["ordinary_counterpart"]
    )
    checks["deleted_duplicate_not_selected"] = not any("dump_20190320_4154 (1).sql" in value for value in selected)

    with SYNCHRONIZED_SILENCES.open(newline="", encoding="utf-8") as stream:
        synchronized_silences = list(csv.DictReader(stream))

    expected_ids: dict[str, set[str]] = {"tblPowerLog": set(), "tblOperationLog": set()}
    for relative in selected:
        experiment = Path(relative).parts[0]
        for row in iter_copy_rows(RAW / relative):
            if row.table in expected_ids:
                signature = row_signature(row.table, row.columns, row.values)
                expected_ids[row.table].add(stable_canonical_id(experiment, row.table, signature))

    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        database_ids = {
            "tblPowerLog": {row[0] for row in connection.execute("SELECT canonical_row_id FROM power_log_canonical")},
            "tblOperationLog": {row[0] for row in connection.execute("SELECT canonical_row_id FROM operation_log_canonical")},
        }
        checks["all_power_values_preserved_by_row_identity"] = expected_ids["tblPowerLog"] == database_ids["tblPowerLog"]
        checks["all_operation_values_preserved_by_row_identity"] = expected_ids["tblOperationLog"] == database_ids["tblOperationLog"]
        power_count = connection.execute("SELECT COUNT(*) FROM power_log_canonical").fetchone()[0]
        operation_count = connection.execute("SELECT COUNT(*) FROM operation_log_canonical").fetchone()[0]
        fused_count = connection.execute("SELECT COUNT(*) FROM power_operation_fused").fetchone()[0]
        details.update(power_rows=power_count, operation_rows=operation_count, fused_rows=fused_count)
        checks["fused_count_equals_power_count"] = fused_count == power_count
        checks["power_row_has_at_most_one_output"] = connection.execute(
            "SELECT COUNT(*) FROM (SELECT canonical_row_id FROM power_operation_fused GROUP BY canonical_row_id HAVING COUNT(*) > 1)"
        ).fetchone()[0] == 0
        checks["no_cross_experiment_join"] = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused f JOIN operation_log_canonical o ON f.operation_row_id=o.canonical_row_id WHERE f.experiment_id<>o.experiment_id"
        ).fetchone()[0] == 0
        checks["no_cross_station_join"] = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused f JOIN operation_log_canonical o ON f.operation_row_id=o.canonical_row_id WHERE f.station_id<>o.station_id"
        ).fetchone()[0] == 0
        checks["no_cross_session_join"] = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused f JOIN operation_log_canonical o ON f.operation_row_id=o.canonical_row_id WHERE f.session_id<>o.session_id"
        ).fetchone()[0] == 0
        checks["no_future_plc"] = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused WHERE join_matched=1 AND plc_timestamp_epoch_ms>timestamp_epoch_ms_original"
        ).fetchone()[0] == 0
        checks["matched_age_nonnegative"] = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused WHERE join_matched=1 AND (state_age_seconds IS NULL OR state_age_seconds<0)"
        ).fetchone()[0] == 0
        checks["unmatched_age_null"] = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused WHERE join_matched=0 AND state_age_seconds IS NOT NULL"
        ).fetchone()[0] == 0
        checks["power_outside_plc_intervals_has_null_plc"] = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused WHERE session_id IS NULL AND (join_matched<>0 OR operation_row_id IS NOT NULL OR plc_timestamp_epoch_ms IS NOT NULL OR Busy IS NOT NULL OR OperationNo IS NOT NULL)"
        ).fetchone()[0] == 0
        checks["matched_plc_is_inside_same_session_interval"] = connection.execute(
            "WITH intervals AS (SELECT experiment_id, station_id, session_id, MIN(timestamp_epoch_ms_original) first_ms, MAX(timestamp_epoch_ms_original) last_ms FROM operation_log_canonical GROUP BY experiment_id, station_id, session_id) SELECT COUNT(*) FROM power_operation_fused f JOIN intervals i ON f.experiment_id=i.experiment_id AND f.station_id=i.station_id AND f.session_id=i.session_id WHERE f.join_matched=1 AND (f.timestamp_epoch_ms_original<i.first_ms OR f.timestamp_epoch_ms_original>i.last_ms)"
        ).fetchone()[0] == 0

        synchronized_gap_violations = 0
        synchronized_source_violations = 0
        synchronized_gap_power_rows = 0
        april_acceptance_rows = 0
        april_acceptance_violations = 0
        april_acceptance_found = False
        for silence in synchronized_silences:
            experiment = silence["experiment_id"]
            station_ids = [int(value) for value in json.loads(silence["support_station_ids"])]
            for source_gap in json.loads(silence["supporting_station_gaps"]):
                station_id = int(source_gap["station_id"])
                gap_start = int(source_gap["gap_start_epoch_ms"])
                gap_end = int(source_gap["gap_end_epoch_ms"])
                source_parameters = (
                    experiment,
                    station_id,
                    gap_start,
                    gap_end,
                )
                endpoint_count = connection.execute(
                    "SELECT COUNT(DISTINCT timestamp_epoch_ms_original) FROM operation_log_canonical WHERE experiment_id=? AND station_id=? AND timestamp_epoch_ms_original IN (?,?)",
                    source_parameters,
                ).fetchone()[0]
                interior_count = connection.execute(
                    "SELECT COUNT(*) FROM operation_log_canonical WHERE experiment_id=? AND station_id=? AND timestamp_epoch_ms_original>? AND timestamp_epoch_ms_original<?",
                    source_parameters,
                ).fetchone()[0]
                synchronized_source_violations += int(endpoint_count != 2 or interior_count != 0)

                gap_power_rows = connection.execute(
                    "SELECT COUNT(*) FROM power_operation_fused WHERE experiment_id=? AND station_id=? AND timestamp_epoch_ms_original>? AND timestamp_epoch_ms_original<?",
                    source_parameters,
                ).fetchone()[0]
                gap_violations = connection.execute(
                    "SELECT COUNT(*) FROM power_operation_fused WHERE experiment_id=? AND station_id=? AND timestamp_epoch_ms_original>? AND timestamp_epoch_ms_original<? AND (session_id IS NOT NULL OR join_matched<>0 OR operation_row_id IS NOT NULL OR plc_timestamp_epoch_ms IS NOT NULL OR state_age_seconds IS NOT NULL)",
                    source_parameters,
                ).fetchone()[0]
                synchronized_gap_power_rows += gap_power_rows
                synchronized_gap_violations += gap_violations

                required_april_stations = {10, 30, 60, 70, 80}
                duration_seconds = float(silence["core_duration_seconds"])
                is_april_acceptance_gap = (
                    experiment == "Exp20190416"
                    and 43 * 60 <= duration_seconds <= 45 * 60
                    and required_april_stations.issubset(station_ids)
                )
                if is_april_acceptance_gap and station_id in required_april_stations:
                    april_acceptance_rows += gap_power_rows
                    april_acceptance_violations += gap_violations

            required_april_stations = {10, 30, 60, 70, 80}
            duration_seconds = float(silence["core_duration_seconds"])
            if (
                experiment == "Exp20190416"
                and 43 * 60 <= duration_seconds <= 45 * 60
                and required_april_stations.issubset(station_ids)
            ):
                april_acceptance_found = True

        checks["synchronized_silences_derive_from_consecutive_plc_timestamps"] = synchronized_source_violations == 0
        checks["power_inside_synchronized_silences_is_unmatched_and_null"] = synchronized_gap_violations == 0
        checks["april_43_45_minute_silence_detected"] = april_acceptance_found
        checks["april_required_stations_are_null_inside_silence"] = april_acceptance_found and april_acceptance_violations == 0
        details["synchronized_silences"] = len(synchronized_silences)
        details["power_rows_inside_synchronized_silences"] = synchronized_gap_power_rows
        details["april_required_station_power_rows_inside_silence"] = april_acceptance_rows

        matches_over_1800 = connection.execute(
            "SELECT COUNT(*) FROM power_operation_fused WHERE join_matched=1 AND state_age_seconds>1800"
        ).fetchone()[0]
        with OLD_MATCH_SUMMARY.open(newline="", encoding="utf-8") as stream:
            old_match_report_rows = list(csv.DictReader(stream))
        checks["state_age_over_1800_report_matches_database"] = (
            sum(int(row["match_count"]) for row in old_match_report_rows) == matches_over_1800
        )
        details["matches_over_1800_seconds"] = matches_over_1800
        details["power_zero_and_extrema"] = dict(
            zip(
                ["power_zero_count", "flow_zero_count", "pressure_zero_count", "power_min", "power_max", "flow_min", "flow_max", "pressure_min", "pressure_max"],
                connection.execute(
                    "SELECT SUM(ActivePowerL1=0), SUM(Flow=0), SUM(Pressure=0), MIN(ActivePowerL1), MAX(ActivePowerL1), MIN(Flow), MAX(Flow), MIN(Pressure), MAX(Pressure) FROM power_log_canonical"
                ).fetchone(),
                strict=True,
            )
        )
        forbidden_columns = {
            row[1]
            for table in ("power_log_canonical", "operation_log_canonical", "power_operation_fused")
            for row in connection.execute(f"PRAGMA table_info({table})")
            if row[1] == "inferred_state" or row[1] == "possible_reset_or_sentinel" or row[1].startswith("feature_") or row[1].startswith("window_")
        }
        checks["no_windows_features_or_inferred_state"] = not forbidden_columns

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    schemas = json.loads(SCHEMAS.read_text(encoding="utf-8"))["power_operation_fused"]
    checks["nullable_dtypes_documented"] = (
        all(schemas[column] == "boolean" for column in ["Busy", "RFIDTagPresent", "Done", "StationEntryxBG5", "ReadyAtStationxBG1", "DoneWorkingxBG9", "StationExitxBG6", "join_matched", "join_ambiguous"])
        and all(schemas[column] == "Int64" for column in ["OperationNo", "WorkPlanNo", "OrderNo", "StepNo", "CarrierID", "OrderPosition", "PartNumber", "plc_timestamp_epoch_ms"])
        and all(schemas[column] == "string" for column in ["operation_row_id", "plc_resource_id_original", "mes_resource_id_plc", "iResourceID"])
        and schemas["plc_timestamp_matched"].endswith(", UTC]")
    )
    checks["no_downstream_or_qml_steps_reported"] = summary["forbidden_steps_performed"] == []
    checks["all_pipeline_checks_passed"] = all(summary["checks"].values())
    details["selected_canonical_dumps"] = len(selected)
    details["session_count"] = summary["session_count"]
    details["join_coverage_percent"] = summary["join_coverage_percent"]
    result = {"checks": checks, "details": details, "status": "PASS" if all(checks.values()) else "FAIL"}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
