"""Schema-aware loaders for human-readable pipeline CSV exports.

SQLite remains the canonical storage format. These helpers restore the pandas
nullable dtypes recorded in ``dataset_schemas.json`` when a CSV export is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def dataset_name_from_csv(path: Path) -> str:
    name = path.name
    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"unsupported CSV filename: {path}")


def load_schema_aware_csv(
    csv_path: str | Path,
    schema_path: str | Path,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    schema_path = Path(schema_path)
    schemas = json.loads(schema_path.read_text(encoding="utf-8"))
    name = dataset_name or dataset_name_from_csv(csv_path)
    if name not in schemas:
        raise KeyError(f"dataset {name!r} is absent from {schema_path}")
    schema: dict[str, str] = schemas[name]
    dtype_map: dict[str, str] = {}
    timestamp_columns: dict[str, str] = {}
    for column, dtype in schema.items():
        if dtype in {"boolean", "Int64", "Float64", "string"}:
            dtype_map[column] = dtype
        elif dtype == "str":
            dtype_map[column] = "string"
        elif dtype.startswith("datetime64["):
            timestamp_columns[column] = dtype
    frame = pd.read_csv(csv_path, dtype=dtype_map, compression="infer", low_memory=False)
    missing = sorted(set(schema) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(schema))
    if missing or extra:
        raise ValueError(f"CSV/schema column mismatch; missing={missing}, extra={extra}")
    for column, dtype in timestamp_columns.items():
        # Pipeline CSVs legitimately mix whole-second and fractional-second
        # ISO timestamps. Parse each value's ISO shape independently.
        converted = pd.to_datetime(frame[column], utc=True, errors="raise", format="mixed")
        if "Europe/Rome" in dtype:
            converted = converted.dt.tz_convert("Europe/Rome")
        frame[column] = converted
    for column, dtype in schema.items():
        if dtype == "int64":
            if frame[column].isna().any():
                raise ValueError(f"non-nullable column {column} contains NULL")
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    return frame[list(schema)]
