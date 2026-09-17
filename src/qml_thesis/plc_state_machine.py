"""PLC boolean-vector -> physical energy state, per station.

Rule-based ground-truth labelling derived from tblOperationLog vectors,
validated against power coherence (sanity check of 2026-09-XX, see thesis).

Vector field order (fixed everywhere in this module):
    (Busy, RFIDTagPresent, Done, StationEntryxBG5, ReadyAtStationxBG1, StationExitxBG6)

Rules are evaluated in order; first match wins. Station overrides are applied
after the generic rules. The machine is PLC-only: ActivePower/Flow/Pressure
never enter the classification. Power was used only to validate the rules.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

VECTOR_FIELDS = [
    "Busy",
    "RFIDTagPresent",
    "Done",
    "StationEntryxBG5",
    "ReadyAtStationxBG1",
    "StationExitxBG6",
]

# Ordered generic rules. Each predicate receives the six boolean values.
GENERIC_RULES: list[tuple[str, tuple]] = [
    ("processing",            lambda b, r, d, e, y, x: b == 1),
    ("waiting_carrier_ready", lambda b, r, d, e, y, x: b == 0 and r == 1 and y == 1),
    ("waiting_carrier",       lambda b, r, d, e, y, x: b == 0 and r == 1 and y == 0),
    ("transfer",              lambda b, r, d, e, y, x: b == 0 and r == 0 and e == 1 and x == 1),
    ("entering",              lambda b, r, d, e, y, x: b == 0 and r == 0 and e == 1),
    ("exiting",               lambda b, r, d, e, y, x: b == 0 and r == 0 and x == 1),
    ("idle_ready",            lambda b, r, d, e, y, x: b == 0 and r == 0 and d == 1),
    ("unknown",               lambda b, r, d, e, y, x: True),
]

# Station-specific overrides on the raw 6-bit vector string.
# Station 70 (MagazineBack): StationExit persists while carriers rest at the
# exit buffer; power at 001001 (43.8 W) equals idle (44.3 W), so the flag does
# not indicate an active exiting phase there.
STATION_OVERRIDES: dict[int, dict[str, str]] = {
    70: {"001001": "idle_ready"},
}

# Station 50 (RobotAssembly): Entry/Ready/Exit are always NULL in the frozen
# data; only (Busy, RFID, Done) are informative.
STATION50_RULES: list[tuple[str, tuple]] = [
    ("processing",      lambda b, r, d, e, y, x: b == 1),
    ("waiting_carrier", lambda b, r, d, e, y, x: b == 0 and r == 1 and d == 1),
    ("idle_ready",      lambda b, r, d, e, y, x: b == 0 and r == 0 and d == 1),
    ("unknown",         lambda b, r, d, e, y, x: True),
]

STATE_ORDER = [
    "idle_ready",
    "waiting_carrier",
    "waiting_carrier_ready",
    "processing",
    "entering",
    "exiting",
    "transfer",
    "unknown",
]

IDLE_STATES = {"idle_ready"}  # trigger states for the idle-duration pipeline


def classify_row(station_id: int, busy, rfid, done, entry, ready, exit_) -> str:
    """Classify one PLC vector. NULL flags are treated as 0."""
    def as_bit(value) -> int:
        return 1 if value == 1 else 0

    bits = [as_bit(v) for v in (busy, rfid, done, entry, ready, exit_)]
    key = "".join(str(b) for b in bits)

    if station_id in STATION_OVERRIDES and key in STATION_OVERRIDES[station_id]:
        return STATION_OVERRIDES[station_id][key]

    rules = STATION50_RULES if station_id == 50 else GENERIC_RULES
    for state, predicate in rules:
        if predicate(*bits):
            return state
    return "unknown"  # unreachable; defensive


def classify_frame(frame: pd.DataFrame) -> pd.Series:
    """Vectorized classification for a fused power_operation_fused frame."""
    return pd.Series(
        [
            classify_row(int(row.station_id), row.Busy, row.RFIDTagPresent, row.Done,
                         row.StationEntryxBG5, row.ReadyAtStationxBG1, row.StationExitxBG6)
            for row in frame.itertuples(index=False)
        ],
        index=frame.index,
        dtype="string",
        name="plc_state",
    )


def state_coverage_report(sqlite_path: str) -> pd.DataFrame:
    """Coverage of classified PLC vectors by station and state.

    Reads the frozen common pipeline read-only and reports, for every
    station and assigned state, the number of matched power rows and the
    mean ActivePowerL1. This is the rule-validation report: power of
    idle/wating states must sit below power of processing, per station.
    """
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            """
            SELECT station_id, Busy, RFIDTagPresent, Done, StationEntryxBG5,
                   ReadyAtStationxBG1, StationExitxBG6, ActivePowerL1
            FROM power_operation_fused WHERE join_matched = 1
            """,
            connection,
        )
    finally:
        connection.close()

    frame["plc_state"] = classify_frame(frame)
    report = (
        frame.groupby(["station_id", "plc_state"])
        .agg(n=("ActivePowerL1", "size"), avg_watts=("ActivePowerL1", "mean"))
        .reset_index()
        .sort_values(["station_id", "avg_watts"])
    )
    report["avg_watts"] = report["avg_watts"].round(2)
    return report


def expected_cycle_check(transition_table: pd.DataFrame) -> pd.DataFrame:
    """Report agreement with the expected dominant operating cycle.

    Given a prev_vec/vec/transitions table (as produced by the audit query),
    verify that the dominant transitions are consistent with the labelled
    state sequence:
        processing -> waiting_carrier_ready -> processing   (same carrier)
        waiting_carrier_ready -> idle_ready                  (carrier leaves)
        idle_ready -> entering/exiting/transfer/processing   (new activity)
    """
    def vec_to_state(station: int, vec: str) -> str:
        b, r, d, e, y, x = (int(c) for c in vec)
        return classify_row(station, b, r, d, e, y, x)

    rows = []
    for record in transition_table.itertuples(index=False):
        prev = vec_to_state(int(record.station_id), str(record.prev_vec))
        curr = vec_to_state(int(record.station_id), str(record.vec))
        in_cycle = (
            (prev == "processing" and curr == "waiting_carrier_ready")
            or (prev == "waiting_carrier_ready" and curr in {"processing", "idle_ready"})
            or (prev == "idle_ready" and curr in {"entering", "exiting", "transfer",
                                                  "processing", "waiting_carrier",
                                                  "waiting_carrier_ready"})
            or prev == curr  # flicker-free persistence
        )
        rows.append({"station_id": record.station_id, "prev_vec": record.prev_vec,
                     "vec": record.vec, "transitions": record.transitions,
                     "prev_state": prev, "curr_state": curr, "in_expected_cycle": in_cycle})
    return pd.DataFrame(rows)