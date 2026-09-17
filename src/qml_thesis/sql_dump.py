"""Streaming reader for PostgreSQL plain-text COPY dumps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Iterator

from .schemas import BOOLEAN_COLUMNS, INTEGER_COLUMNS, NUMERIC_COLUMNS

COPY_RE = re.compile(r'^COPY\s+([^.]+)\."([^"]+)"\s+\((.+)\)\s+FROM stdin;$')
CREATE_SCHEMA_RE = re.compile(r"^CREATE SCHEMA\s+([^;]+);")


@dataclass(frozen=True)
class CopyRow:
    schema: str
    table: str
    columns: tuple[str, ...]
    values: tuple[object, ...]
    raw_line_number: int


def _unescape_copy_text(value: str) -> str:
    result: list[str] = []
    index = 0
    escapes = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            result.append(escapes.get(value[index + 1], value[index + 1]))
            index += 2
        else:
            result.append(value[index])
            index += 1
    return "".join(result)


def normalize_value(column: str, value: str) -> object:
    if value == r"\N":
        return None
    decoded = _unescape_copy_text(value)
    if column in BOOLEAN_COLUMNS:
        if decoded == "t":
            return True
        if decoded == "f":
            return False
        raise ValueError(f"invalid PostgreSQL boolean {decoded!r} in {column}")
    if column in INTEGER_COLUMNS:
        return int(decoded)
    if column in NUMERIC_COLUMNS:
        return float(decoded)
    return decoded


def iter_copy_rows(path: Path) -> Iterator[CopyRow]:
    current: tuple[str, str, tuple[str, ...]] | None = None
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            match = COPY_RE.match(line)
            if match:
                schema, table, column_text = match.groups()
                columns = tuple(re.findall(r'"([^"]+)"', column_text))
                current = (schema, table, columns)
                continue
            if current is None:
                continue
            if line == r"\.":
                current = None
                continue
            schema, table, columns = current
            fields = line.split("\t")
            if len(fields) != len(columns):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(columns)} COPY fields, found {len(fields)}"
                )
            yield CopyRow(
                schema=schema,
                table=table,
                columns=columns,
                values=tuple(normalize_value(column, value) for column, value in zip(columns, fields)),
                raw_line_number=line_number,
            )


def row_signature(table: str, columns: tuple[str, ...] | list[str], values: tuple[object, ...] | list[object]) -> str:
    payload = json.dumps(
        {"table": table, "columns": list(columns), "values": list(values)},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inspect_dump(path: Path, collect_signatures: bool = False) -> tuple[list[dict[str, object]], dict[str, Counter[str]]]:
    schemas: set[str] = set()
    table_columns: dict[str, tuple[str, ...]] = {}
    counts: Counter[str] = Counter()
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    signatures: dict[str, Counter[str]] = {}
    seen_rows: dict[str, set[str]] = {}
    exact_duplicates: Counter[str] = Counter()
    timestamp_conflicts: Counter[str] = Counter()
    last_timestamp: dict[tuple[str, str], int] = {}
    regressions: Counter[str] = Counter()
    first_at_timestamp: dict[tuple[str, str, int], str] = {}

    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            match = CREATE_SCHEMA_RE.match(line)
            if match:
                schemas.add(match.group(1))
            copy_match = COPY_RE.match(line)
            if copy_match:
                _, table, column_text = copy_match.groups()
                table_columns[table] = tuple(re.findall(r'"([^"]+)"', column_text))

    for row in iter_copy_rows(path):
        table_columns[row.table] = row.columns
        counts[row.table] += 1
        signature = row_signature(row.table, row.columns, row.values)
        table_seen = seen_rows.setdefault(row.table, set())
        if signature in table_seen:
            exact_duplicates[row.table] += 1
        else:
            table_seen.add(signature)
        if "timeStamp" in row.columns:
            timestamp = int(row.values[row.columns.index("timeStamp")])
            resource = str(row.values[row.columns.index("ResourceID")]) if "ResourceID" in row.columns else ""
            starts[row.table] = min(starts.get(row.table, timestamp), timestamp)
            ends[row.table] = max(ends.get(row.table, timestamp), timestamp)
            previous = last_timestamp.get((row.table, resource))
            if previous is not None and timestamp < previous:
                regressions[row.table] += 1
            last_timestamp[(row.table, resource)] = timestamp
            timestamp_key = (row.table, resource, timestamp)
            previous_signature = first_at_timestamp.get(timestamp_key)
            if previous_signature is None:
                first_at_timestamp[timestamp_key] = signature
            elif previous_signature != signature:
                timestamp_conflicts[row.table] += 1
        if collect_signatures:
            signatures.setdefault(row.table, Counter())[signature] += 1

    tables = sorted(set(table_columns) | set(signatures))
    profile = [
        {
            "table": table,
            "schema_names": json.dumps(sorted(schemas), separators=(",", ":")),
            "columns": json.dumps(list(table_columns.get(table, ())), separators=(",", ":")),
            "row_count": counts[table],
            "exact_duplicate_rows": exact_duplicates[table],
            "different_rows_same_resource_timestamp": timestamp_conflicts[table],
            "timestamp_order_regressions_raw_order": regressions[table],
            "timestamp_start_epoch_ms": starts.get(table),
            "timestamp_end_epoch_ms": ends.get(table),
        }
        for table in tables
    ]
    return profile, signatures
