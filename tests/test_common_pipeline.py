from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from qml_thesis.common_pipeline import (
    assign_power_to_plc_sessions,
    add_synchronized_boundaries,
    assign_sessions,
    build_plc_session_intervals,
    detect_synchronized_plc_silences,
    derive_session_boundaries,
    fuse_power_operation,
)
from qml_thesis.sql_dump import inspect_dump, iter_copy_rows
from qml_thesis.dataset_loader import load_schema_aware_csv


def _base_frame(rows: list[dict[str, object]], table: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp_epoch_ms_original"] = pd.array(frame["timeStamp"], dtype="Int64")
    frame["timestamp_utc"] = pd.to_datetime(frame["timeStamp"], unit="ms", utc=True)
    frame["timestamp_europe_rome"] = frame["timestamp_utc"].dt.tz_convert("Europe/Rome")
    frame["station_id"] = pd.array(frame["station_id"], dtype="Int64")
    frame["station_name"] = "DrillingCPS"
    frame["canonical_source_file"] = f"ExpX/{table}.sql"
    frame["canonical_source_line_number"] = 1
    frame["ordinary_counterpart"] = pd.NA
    frame["lineage_source_files"] = '[]'
    frame["source_occurrence_count"] = 1
    frame["resource_id_normalized"] = frame["ResourceID"]
    frame["energy_resource_id"] = "ene3" if table == "power" else pd.NA
    frame["mes_resource_id"] = "mes30" if table == "operation" else pd.NA
    return frame


def test_sql_copy_parser_preserves_null_false_and_zero(tmp_path: Path) -> None:
    dump = tmp_path / "dump.sql"
    dump.write_text(
        'CREATE SCHEMA demo;\n'
        'COPY demo."tblOperationLog" ("ResourceID", "timeStamp", "Busy", "RFIDTagPresent", "Done", "StationEntryxBG5", "ReadyAtStationxBG1", "DoneWorkingxBG9", "StationExitxBG6", "OperationNo", "WorkPlanNo", "OrderNo", "StepNo", "CarrierID", "iResourceID", "OrderPosition", "PartNumber") FROM stdin;\n'
        'mes30_3:plcDrillingCPS\t1550000000000\tf\tt\tf\t\\N\tf\t\\N\tt\t0\t\\N\t0\t\\N\t0\tmes30\t0\t0\n'
        '\\.\n',
        encoding="utf-8",
    )
    rows = list(iter_copy_rows(dump))
    assert len(rows) == 1
    values = dict(zip(rows[0].columns, rows[0].values, strict=True))
    assert values["Busy"] is False
    assert values["StationEntryxBG5"] is None
    assert values["OperationNo"] == 0
    profile, _ = inspect_dump(dump)
    assert profile[0]["row_count"] == 1


def test_session_rule_detects_empirical_tail_jump() -> None:
    common = {
        "experiment_id": "ExpX",
        "station_id": 30,
        "ResourceID": "ene3_2:CECC-LK",
    }
    plc_timestamps = list(range(0, 20_000, 1000)) + list(range(200_000, 220_000, 1000))
    power_timestamps = [-1, 0, 19_000, 50_000, 200_000, 219_000, 220_000]
    power = _base_frame(
        [{**common, "timeStamp": value, "canonical_row_id": f"p{value}"} for value in power_timestamps],
        "power",
    )
    operation = _base_frame(
        [{**common, "ResourceID": "mes30_3:plcDrillingCPS", "timeStamp": value, "canonical_row_id": f"o{value}"} for value in plc_timestamps],
        "operation",
    )
    definitions, report = derive_session_boundaries(operation, .95, 2.0)
    assert definitions[("ExpX", 30)]["boundary_count"] == 1
    assert report.loc[0, "boundary_source"] == "tblOperationLog"
    operation = assign_sessions(operation, definitions)
    intervals = build_plc_session_intervals(operation)
    assigned = assign_power_to_plc_sessions(power, intervals).set_index("timeStamp")
    assert pd.isna(assigned.loc[-1, "session_id"])
    assert assigned.loc[0, "session_id"] == "ExpX-S30-001"
    assert assigned.loc[19_000, "session_id"] == "ExpX-S30-001"
    assert pd.isna(assigned.loc[50_000, "session_id"])
    assert assigned.loc[200_000, "session_id"] == "ExpX-S30-002"
    assert assigned.loc[219_000, "session_id"] == "ExpX-S30-002"
    assert pd.isna(assigned.loc[220_000, "session_id"])


def test_backward_join_never_uses_future_or_cross_session() -> None:
    power_rows = [
        {"experiment_id": "ExpX", "station_id": 30, "ResourceID": "ene3_2:CECC-LK", "timeStamp": 1000, "canonical_row_id": "p1", "session_id": "ExpX-S30-001", "ActivePowerL1": 10.0, "Flow": 0.0, "Pressure": 5.0},
        {"experiment_id": "ExpX", "station_id": 30, "ResourceID": "ene3_2:CECC-LK", "timeStamp": 2000, "canonical_row_id": "p2", "session_id": "ExpX-S30-001", "ActivePowerL1": 11.0, "Flow": 0.0, "Pressure": 5.0},
        {"experiment_id": "ExpX", "station_id": 30, "ResourceID": "ene3_2:CECC-LK", "timeStamp": 5000, "canonical_row_id": "p3", "session_id": "ExpX-S30-002", "ActivePowerL1": 12.0, "Flow": 0.0, "Pressure": 5.0},
    ]
    operation_rows = [
        {"experiment_id": "ExpX", "station_id": 30, "ResourceID": "mes30_3:plcDrillingCPS", "timeStamp": 1500, "canonical_row_id": "o1", "session_id": "ExpX-S30-001", "Busy": True, "RFIDTagPresent": False, "Done": False, "StationEntryxBG5": False, "ReadyAtStationxBG1": False, "DoneWorkingxBG9": None, "StationExitxBG6": False, "OperationNo": 0, "WorkPlanNo": None, "OrderNo": 0, "StepNo": None, "CarrierID": 0, "iResourceID": "mes30", "OrderPosition": 0, "PartNumber": 0},
    ]
    power = _base_frame(power_rows, "power")
    power["session_id"] = [row["session_id"] for row in power_rows]
    operation = _base_frame(operation_rows, "operation")
    operation["session_id"] = [row["session_id"] for row in operation_rows]
    fused = fuse_power_operation(power, operation)
    assert len(fused) == len(power)
    assert not fused.loc[fused["canonical_row_id"] == "p1", "join_matched"].item()
    assert fused.loc[fused["canonical_row_id"] == "p2", "operation_row_id"].item() == "o1"
    assert not fused.loc[fused["canonical_row_id"] == "p3", "join_matched"].item()
    assert (fused.loc[fused["join_matched"], "state_age_seconds"] >= 0).all()
    assert str(fused["Busy"].dtype) == "boolean"
    assert str(fused["OperationNo"].dtype) == "Int64"
    assert str(fused["operation_row_id"].dtype) == "string"
    assert str(fused["plc_timestamp_matched"].dtype).startswith("datetime64[")
    assert str(fused["plc_timestamp_matched"].dtype).endswith(", UTC]")


def test_csv_loader_restores_recorded_nullable_types(tmp_path: Path) -> None:
    csv_path = tmp_path / "demo.csv"
    schema_path = tmp_path / "dataset_schemas.json"
    csv_path.write_text(
        "flag,count,label,timestamp_utc\ntrue,1,a,2019-04-16T07:00:00.123Z\nfalse,2,b,2019-04-16T07:00:01Z\n,,c,\n",
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(
            {"demo": {"flag": "boolean", "count": "Int64", "label": "string", "timestamp_utc": "datetime64[ms, UTC]"}}
        ),
        encoding="utf-8",
    )
    frame = load_schema_aware_csv(csv_path, schema_path)
    assert str(frame["flag"].dtype) == "boolean"
    assert str(frame["count"].dtype) == "Int64"
    assert str(frame["label"].dtype) == "string"
    assert str(frame["timestamp_utc"].dtype).endswith(", UTC]")
    assert frame["timestamp_utc"].notna().sum() == 2


def test_two_level_plc_discontinuities_add_synchronized_boundary() -> None:
    rows: list[dict[str, object]] = []
    unique_gaps = {10: (1000, 1500), 20: (2000, 2500), 30: (3000, 3500)}
    for station, (unique_start, unique_end) in unique_gaps.items():
        timestamps = list(range(0, 4001, 10))
        timestamps = [value for value in timestamps if not (500 < value < 550)]
        timestamps = [value for value in timestamps if not (unique_start < value < unique_end)]
        rows.extend(
            {
                "experiment_id": "ExpSynthetic",
                "station_id": station,
                "timestamp_epoch_ms_original": value,
            }
            for value in timestamps
        )
    operation = pd.DataFrame(rows)
    definitions, gap_report = derive_session_boundaries(operation, .95, 2.0)
    assert all(definition["boundary_count"] == 1 for definition in definitions.values())
    silences, synchronized = detect_synchronized_plc_silences(
        operation,
        candidate_top_gaps_per_station=3,
        minimum_station_fraction=.60,
        minimum_station_count=2,
        minimum_relative_duration=.20,
    )
    assert len(silences) == 1
    assert silences.iloc[0]["support_station_count"] == 3
    assert silences.iloc[0]["core_start_epoch_ms"] == 500
    assert silences.iloc[0]["core_end_epoch_ms"] == 550
    definitions, report = add_synchronized_boundaries(definitions, gap_report, synchronized)
    assert all(definition["local_boundary_count"] == 1 for definition in definitions.values())
    assert all(definition["synchronized_boundary_count"] == 1 for definition in definitions.values())
    assert all(definition["boundary_count"] == 2 for definition in definitions.values())
    assert report["synchronized_boundary_count"].eq(1).all()
