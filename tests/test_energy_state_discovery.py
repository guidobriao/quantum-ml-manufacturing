from __future__ import annotations

import pandas as pd

from qml_thesis.energy_state_discovery import (
    BOOLEAN_PLC,
    DISCOVERY_FEATURES,
    MES_FIELDS,
    event_assignments,
    extract_features,
    fixed_assignments,
)


def _power() -> pd.DataFrame:
    rows = []
    for timestamp, power in [(0, 10.0), (1000, 12.0), (2000, 14.0), (6000, 20.0)]:
        row = {
            "canonical_row_id": f"p{timestamp}", "experiment_id": "ExpX", "station_id": 30,
            "station_name": "DrillingCPS", "session_id": "ExpX-S30-001",
            "timestamp_epoch_ms_original": timestamp, "timestamp_utc": str(timestamp),
            "ActivePowerL1": power, "Flow": 1.0, "Pressure": 5.0, "join_matched": True,
        }
        row.update({column: False for column in BOOLEAN_PLC})
        row.update({column: None for column in MES_FIELDS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_fixed_segments_and_integrated_energy_do_not_cross_session() -> None:
    power = _power()
    power["segment_id"] = fixed_assignments(power, 5, "fixed_5s")
    assert power["segment_id"].nunique() == 2
    features = extract_features(power, "fixed", "fixed_5s")
    first = features.sort_values("segment_start_epoch_ms").iloc[0]
    assert first["duration_seconds"] == 2.0
    assert first["sample_count"] == 3
    assert first["power_mean"] == 12.0
    assert first["energy_joule"] == 24.0


def test_event_segmentation_uses_plc_transition_as_boundary() -> None:
    power = _power()
    operation = pd.DataFrame([
        {"canonical_row_id": "o0", "experiment_id": "ExpX", "station_id": 30, "station_name": "DrillingCPS", "session_id": "ExpX-S30-001", "timestamp_epoch_ms_original": 0, "timestamp_utc": "0", **{column: False for column in BOOLEAN_PLC}, **{column: None for column in MES_FIELDS}},
        {"canonical_row_id": "o1", "experiment_id": "ExpX", "station_id": 30, "station_name": "DrillingCPS", "session_id": "ExpX-S30-001", "timestamp_epoch_ms_original": 1500, "timestamp_utc": "1500", **{column: (True if column == "Busy" else False) for column in BOOLEAN_PLC}, **{column: None for column in MES_FIELDS}},
    ])
    labels, transitions = event_assignments(power, operation, ["Busy"], "event")
    assert len(transitions) == 1
    assert labels.iloc[0] == labels.iloc[1]
    assert labels.iloc[2] != labels.iloc[1]


def test_discovery_feature_list_has_no_plc_or_mes_fields() -> None:
    assert not (set(DISCOVERY_FEATURES) & set(BOOLEAN_PLC + MES_FIELDS))
