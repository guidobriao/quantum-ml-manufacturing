"""Confirmed schemas and station semantics for the common local pipeline."""

from __future__ import annotations

TABLE_SCHEMAS: dict[str, list[tuple[str, str, str | None]]] = {
    "tblMachineReport": [
        ("ResourceID", "string", None),
        ("timeStamp", "int64", "ms since Unix epoch"),
        ("Busy", "boolean", None),
        ("RFIDTagPresent", "boolean", None),
        ("Done", "boolean", None),
    ],
    "tblOperationLog": [
        ("ResourceID", "string", None),
        ("timeStamp", "int64", "ms since Unix epoch"),
        ("Busy", "boolean", None),
        ("RFIDTagPresent", "boolean", None),
        ("Done", "boolean", None),
        ("StationEntryxBG5", "boolean", None),
        ("ReadyAtStationxBG1", "boolean", None),
        ("DoneWorkingxBG9", "boolean", None),
        ("StationExitxBG6", "boolean", None),
        ("OperationNo", "integer", None),
        ("WorkPlanNo", "integer", None),
        ("OrderNo", "integer", None),
        ("StepNo", "integer", None),
        ("CarrierID", "integer", None),
        ("iResourceID", "string", None),
        ("OrderPosition", "integer", None),
        ("PartNumber", "integer", None),
    ],
    "tblPowerLog": [
        ("ResourceID", "string", None),
        ("timeStamp", "int64", "ms since Unix epoch"),
        ("ActivePowerL1", "numeric", "W"),
        ("Flow", "numeric", "l/min"),
        ("Pressure", "numeric", "atm"),
    ],
    "tblSensorsLog": [
        ("ResourceID", "string", None),
        ("timeStamp", "int64", "ms since Unix epoch"),
        ("StationEntryxBG5", "boolean", None),
        ("ReadyAtStationxBG1", "boolean", None),
        ("DoneWorkingxBG9", "boolean", None),
        ("StationExitxBG6", "boolean", None),
    ],
}

STATIONS = {
    10: ("ene1", "mes10", "Manual"),
    20: ("ene2", "mes20", "MagazineFront"),
    30: ("ene3", "mes30", "DrillingCPS"),
    40: ("ene4", "mes40", "Bridge"),
    50: ("ene5", "mes50", "RobotAssembly"),
    60: ("ene6", "mes60", "CameraInspection"),
    70: ("ene7", "mes70", "MagazineBack"),
    80: ("ene8", "mes80", "Press"),
}

POWER_COLUMNS = [name for name, _, _ in TABLE_SCHEMAS["tblPowerLog"]]
OPERATION_COLUMNS = [name for name, _, _ in TABLE_SCHEMAS["tblOperationLog"]]
BOOLEAN_COLUMNS = {
    name
    for schema in TABLE_SCHEMAS.values()
    for name, kind, _ in schema
    if kind == "boolean"
}
INTEGER_COLUMNS = {
    name
    for schema in TABLE_SCHEMAS.values()
    for name, kind, _ in schema
    if kind in {"integer", "int64"}
}
NUMERIC_COLUMNS = {
    name
    for schema in TABLE_SCHEMAS.values()
    for name, kind, _ in schema
    if kind == "numeric"
}
