"""Read-only preliminary inventory for the raw thesis archive.

This script deliberately performs no cleaning, fusion, windowing, feature
engineering, labelling, clustering, classification, or QML.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import struct
import sys
import zipfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "raw_manifest.csv"
OUT = ROOT / "results" / "data_quality" / "preliminary"
SQL_COPY = re.compile(r'^COPY\s+[^.]+\."([^"]+)"\s+\((.+)\)\s+FROM stdin;$')
ISO_IN_TEXT = re.compile(r"2019-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
QUERY_TS = re.compile(r"^\s*(2019-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2}\.\d+)")
TIMESTAMP_MS = re.compile(r"timestampms[\"']?\s*:\s*(\d{13})")


def iso_from_ms(value: int | None) -> str:
    if value is None:
        return ""
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_query_timestamp(line: str) -> int | None:
    match = QUERY_TS.match(line)
    if not match:
        return None
    value = datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H:%M:%S.%f")
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)


def detect_text_encoding(path: Path) -> str:
    sample = path.read_bytes()[: 1024 * 1024]
    for encoding in ("ascii", "utf-8-sig", "cp1252"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    return "binary-or-unknown"


def interval_summary(differences: list[int]) -> str:
    positive = [value for value in differences if value > 0]
    if not positive:
        return ""
    return json.dumps(
        {
            "median_ms": statistics.median(positive),
            "p10_ms": sorted(positive)[int((len(positive) - 1) * 0.10)],
            "p90_ms": sorted(positive)[int((len(positive) - 1) * 0.90)],
        },
        separators=(",", ":"),
    )


def role_for(path: Path) -> tuple[str, str, str]:
    name = path.name.lower()
    if path.suffix.lower() == ".sql":
        return "database snapshot containing power and PLC/MES tables", "confirmed", "PostgreSQL COPY dump"
    if name.startswith("query_"):
        return "sparse union-style export of power and operation/PLC events", "probable", "not a time-fused table; requires original query confirmation"
    if name.startswith("probe") or name.startswith("dump_") and path.suffix.lower() == ".txt":
        return "OPC UA / Node-RED sensor event log", "probable", "sensor/topic semantics require confirmation"
    if name.startswith("1_") or name.startswith("2_fuse"):
        return "annotated Node-RED sensor capture", "probable", "manual headings require interpretation"
    if name.startswith("datalogger"):
        return "OPC UA data logger workbook", "probable", "workbook semantics require confirmation"
    if "elaborazioni" in name:
        return "manually derived/analysis workbook", "probable", "not a primary raw source until lineage is confirmed"
    if "summary" in name:
        return "experiment summary workbook", "probable", "contextual/derived source"
    if "performancemonitor" in name:
        return "system-performance screenshot", "confirmed", "contextual evidence, not a tabular measurement source"
    if name == "ui4155.png" or name.startswith("produzione"):
        return "Node-RED energy-dashboard screenshot", "confirmed", "contextual validation of drill-station power profile"
    return "unclassified", "to_confirm", ""


def analyze_sql(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    tables: list[dict[str, object]] = []
    line_count = 0
    schema_names: set[str] = set()
    current: dict[str, object] | None = None
    columns: list[str] = []
    timestamp_index: int | None = None
    resource_index: int | None = None
    seen_hashes: set[bytes] = set()
    last_by_resource: dict[str, int] = {}
    differences: list[int] = []
    numeric_ranges: dict[str, list[float]] = {}

    def close_table() -> None:
        nonlocal current, columns, timestamp_index, resource_index, seen_hashes, last_by_resource, differences, numeric_ranges
        if current is None:
            return
        current["timestamp_start_utc"] = iso_from_ms(current.pop("min_timestamp_ms"))
        current["timestamp_end_utc"] = iso_from_ms(current.pop("max_timestamp_ms"))
        current["sampling_interval_ms"] = interval_summary(differences)
        current["resource_ids"] = json.dumps(sorted(current.pop("resources")), separators=(",", ":"))
        current["missing_by_column"] = json.dumps(
            dict(zip(columns, current.pop("missing_counts"))), separators=(",", ":")
        )
        current["numeric_ranges"] = json.dumps(
            {key: {"min": values[0], "max": values[1]} for key, values in numeric_ranges.items()},
            separators=(",", ":"),
        )
        tables.append(current)
        current = None
        columns = []
        timestamp_index = None
        resource_index = None
        seen_hashes = set()
        last_by_resource = {}
        differences = []
        numeric_ranges = {}

    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw_line in stream:
            line_count += 1
            line = raw_line.rstrip("\r\n")
            schema_match = re.match(r'CREATE SCHEMA\s+([^;]+);', line)
            if schema_match:
                schema_names.add(schema_match.group(1))
            copy_match = SQL_COPY.match(line)
            if copy_match:
                close_table()
                table_name, column_text = copy_match.groups()
                columns = re.findall(r'"([^"]+)"', column_text)
                timestamp_index = columns.index("timeStamp") if "timeStamp" in columns else None
                resource_index = columns.index("ResourceID") if "ResourceID" in columns else None
                current = {
                    "relative_path": path.relative_to(RAW).as_posix(),
                    "experiment": path.relative_to(RAW).parts[0],
                    "table": table_name,
                    "columns": json.dumps(columns, separators=(",", ":")),
                    "row_count": 0,
                    "duplicate_rows": 0,
                    "duplicate_timestamps_per_resource": 0,
                    "timestamp_order_inversions": 0,
                    "min_timestamp_ms": None,
                    "max_timestamp_ms": None,
                    "resources": set(),
                    "missing_counts": [0] * len(columns),
                }
                continue
            if current is None:
                continue
            if line == r"\.":
                close_table()
                continue

            fields = line.split("\t")
            current["row_count"] += 1
            row_hash = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
            if row_hash in seen_hashes:
                current["duplicate_rows"] += 1
            else:
                seen_hashes.add(row_hash)
            for index, value in enumerate(fields[: len(columns)]):
                if value == r"\N":
                    current["missing_counts"][index] += 1
            resource = fields[resource_index] if resource_index is not None and resource_index < len(fields) else ""
            if resource:
                current["resources"].add(resource)
            if timestamp_index is not None and timestamp_index < len(fields):
                try:
                    timestamp = int(fields[timestamp_index])
                except ValueError:
                    timestamp = None
                if timestamp is not None:
                    previous = last_by_resource.get(resource)
                    if previous is not None:
                        difference = timestamp - previous
                        differences.append(difference)
                        if difference == 0:
                            current["duplicate_timestamps_per_resource"] += 1
                        elif difference < 0:
                            current["timestamp_order_inversions"] += 1
                    last_by_resource[resource] = timestamp
                    old_min = current["min_timestamp_ms"]
                    old_max = current["max_timestamp_ms"]
                    current["min_timestamp_ms"] = timestamp if old_min is None else min(old_min, timestamp)
                    current["max_timestamp_ms"] = timestamp if old_max is None else max(old_max, timestamp)
            for numeric_column in ("ActivePowerL1", "Flow", "Pressure"):
                if numeric_column not in columns:
                    continue
                value = fields[columns.index(numeric_column)]
                if value == r"\N":
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                limits = numeric_ranges.setdefault(numeric_column, [number, number])
                limits[0] = min(limits[0], number)
                limits[1] = max(limits[1], number)
    close_table()

    starts = [row["timestamp_start_utc"] for row in tables if row["timestamp_start_utc"]]
    ends = [row["timestamp_end_utc"] for row in tables if row["timestamp_end_utc"]]
    summary = {
        "line_count": line_count,
        "record_count": sum(int(row["row_count"]) for row in tables),
        "timestamp_start_utc": min(starts, default=""),
        "timestamp_end_utc": max(ends, default=""),
        "delimiter": "PostgreSQL COPY (tab)",
        "headers": "; ".join(f"{row['table']}:{row['columns']}" for row in tables),
        "schema_or_sheets": ",".join(sorted(schema_names)),
        "quality_notes": "tables with zero rows and cumulative snapshots must be handled explicitly",
    }
    return summary, tables


def analyze_query(path: Path) -> dict[str, object]:
    encoding = detect_text_encoding(path)
    line_count = 0
    record_count = 0
    headers: list[str] = []
    minimum: int | None = None
    maximum: int | None = None
    missing: Counter[str] = Counter()
    seen: set[bytes] = set()
    duplicate_rows = 0
    last_by_resource: dict[str, int] = {}
    differences: list[int] = []
    delimiter = "," if path.name.endswith("_lim.txt") and "1302" in path.name else "|"
    with path.open("r", encoding=encoding, newline="") as stream:
        for line in stream:
            line_count += 1
            stripped = line.strip("\r\n")
            if not headers and ("dateandtime" in stripped.lower()):
                headers = [part.strip() for part in stripped.split(delimiter)]
                continue
            timestamp = parse_query_timestamp(stripped)
            if timestamp is None:
                continue
            fields = [part.strip() for part in stripped.split(delimiter)]
            if len(fields) != len(headers):
                continue
            record_count += 1
            digest = hashlib.blake2b(stripped.encode(encoding), digest_size=16).digest()
            if digest in seen:
                duplicate_rows += 1
            else:
                seen.add(digest)
            minimum = timestamp if minimum is None else min(minimum, timestamp)
            maximum = timestamp if maximum is None else max(maximum, timestamp)
            resource = fields[1] if len(fields) > 1 else ""
            previous = last_by_resource.get(resource)
            if previous is not None:
                differences.append(timestamp - previous)
            last_by_resource[resource] = timestamp
            for header, value in zip(headers, fields):
                if value == "":
                    missing[header] += 1
    return {
        "line_count": line_count,
        "record_count": record_count,
        "timestamp_start_utc": iso_from_ms(minimum),
        "timestamp_end_utc": iso_from_ms(maximum),
        "delimiter": delimiter,
        "headers": json.dumps(headers, separators=(",", ":")),
        "schema_or_sheets": "flat export",
        "quality_notes": json.dumps(
            {
                "missing_by_column": missing,
                "duplicate_rows": duplicate_rows,
                "sampling_interval_ms": json.loads(interval_summary(differences) or "{}"),
            },
            separators=(",", ":"),
        ),
    }


def analyze_event_text(path: Path) -> dict[str, object]:
    encoding = detect_text_encoding(path)
    line_count = 0
    event_count = 0
    minimum: int | None = None
    maximum: int | None = None
    topics: Counter[str] = Counter()
    keys: set[str] = set()
    last_by_topic: dict[str, int] = {}
    differences: list[int] = []
    seen: set[tuple[int, str]] = set()
    duplicate_events = 0
    with path.open("r", encoding=encoding, newline="") as stream:
        for line in stream:
            line_count += 1
            timestamp_match = TIMESTAMP_MS.search(line)
            if not timestamp_match:
                continue
            timestamp = int(timestamp_match.group(1))
            event_count += 1
            minimum = timestamp if minimum is None else min(minimum, timestamp)
            maximum = timestamp if maximum is None else max(maximum, timestamp)
            topic_match = re.search(r'topic[\"\']?\s*:\s*[\"\']([^\"\']+)', line)
            topic = topic_match.group(1) if topic_match else "unknown"
            topics[topic] += 1
            previous = last_by_topic.get(topic)
            if previous is not None:
                differences.append(timestamp - previous)
            last_by_topic[topic] = timestamp
            event_key = (timestamp, topic)
            if event_key in seen:
                duplicate_events += 1
            else:
                seen.add(event_key)
            keys.update(re.findall(r'([A-Za-z][A-Za-z0-9_]*)[\"\']?\s*:', line))
    return {
        "line_count": line_count,
        "record_count": event_count,
        "timestamp_start_utc": iso_from_ms(minimum),
        "timestamp_end_utc": iso_from_ms(maximum),
        "delimiter": "JSON Lines" if line_count == event_count else "Node-RED multiline log",
        "headers": json.dumps(sorted(keys), separators=(",", ":")),
        "schema_or_sheets": json.dumps(dict(topics), separators=(",", ":")),
        "quality_notes": json.dumps(
            {"duplicate_events": duplicate_events, "sampling_interval_ms": json.loads(interval_summary(differences) or "{}")},
            separators=(",", ":"),
        ),
    }


def xlsx_metadata(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    rel_ns = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}
    sheet_rows: list[dict[str, object]] = []
    workbook_timestamps: list[int] = []
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iterfind(".//m:t", ns)))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in rels.findall("p:Relationship", rel_ns)}
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            sheet_name = sheet.attrib["name"]
            target = targets[sheet.attrib[f"{{{ns['r']}}}id"]]
            sheet_path = "xl/" + target.lstrip("/").removeprefix("xl/")
            xml = ET.fromstring(archive.read(sheet_path))
            dimension = xml.find("m:dimension", ns)
            formulas = xml.findall(".//m:f", ns)
            rows = xml.findall("m:sheetData/m:row", ns)
            sample_values: list[str] = []
            formula_samples = [node.text or "" for node in formulas[:20]]
            error_values: Counter[str] = Counter()
            timestamp_values: list[int] = []
            values_by_row: dict[int, list[str]] = defaultdict(list)
            for cell in xml.findall(".//m:c", ns):
                value_node = cell.find("m:v", ns)
                inline_node = cell.find("m:is/m:t", ns)
                if inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    index = int(value_node.text or 0)
                    value = shared[index] if index < len(shared) else ""
                else:
                    value = value_node.text or ""
                row_match = re.search(r"(\d+)$", cell.attrib.get("r", ""))
                row_number = int(row_match.group(1)) if row_match else 0
                if value and len(sample_values) < 80:
                    sample_values.append(value[:120])
                if value:
                    values_by_row[row_number].append(value[:120])
                    if value.startswith("#"):
                        error_values[value] += 1
                    iso_match = ISO_IN_TEXT.search(value)
                    if iso_match:
                        parsed = datetime.fromisoformat(iso_match.group(0).replace("Z", "+00:00"))
                        timestamp_values.append(int(parsed.timestamp() * 1000))
                    elif re.fullmatch(r"1[45]\d{11}", value):
                        timestamp_values.append(int(value))
            workbook_timestamps.extend(timestamp_values)
            max_row = max((int(row.attrib.get("r", 0)) for row in rows), default=0)
            first_populated_row = values_by_row[min(values_by_row)] if values_by_row else []
            sheet_rows.append(
                {
                    "relative_path": path.relative_to(RAW).as_posix(),
                    "experiment": path.relative_to(RAW).parts[0],
                    "sheet": sheet_name,
                    "dimension": dimension.attrib.get("ref", "") if dimension is not None else "",
                    "xml_rows": len(rows),
                    "max_row": max_row,
                    "formula_count": len(formulas),
                    "formula_samples": json.dumps(formula_samples, ensure_ascii=False, separators=(",", ":")),
                    "cached_formula_errors": json.dumps(error_values, separators=(",", ":")),
                    "timestamp_start_utc": iso_from_ms(min(timestamp_values)) if timestamp_values else "",
                    "timestamp_end_utc": iso_from_ms(max(timestamp_values)) if timestamp_values else "",
                    "first_populated_row": json.dumps(first_populated_row, ensure_ascii=False, separators=(",", ":")),
                    "sample_values": json.dumps(sample_values, ensure_ascii=False, separators=(",", ":")),
                }
            )
    return {
        "line_count": "",
        "record_count": sum(int(row["xml_rows"]) for row in sheet_rows),
        "timestamp_start_utc": iso_from_ms(min(workbook_timestamps)) if workbook_timestamps else "",
        "timestamp_end_utc": iso_from_ms(max(workbook_timestamps)) if workbook_timestamps else "",
        "delimiter": "OOXML",
        "headers": "inspection limited to OOXML metadata/sample values",
        "schema_or_sheets": json.dumps([row["sheet"] for row in sheet_rows], ensure_ascii=False),
        "quality_notes": "artifact-tool unavailable; formulas counted but not recalculated or visually validated",
    }, sheet_rows


def png_metadata(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        signature = stream.read(24)
    width, height = struct.unpack(">II", signature[16:24])
    return {
        "line_count": "",
        "record_count": 1,
        "timestamp_start_utc": "",
        "timestamp_end_utc": "",
        "delimiter": "binary PNG",
        "headers": "",
        "schema_or_sheets": f"{width}x{height}",
        "quality_notes": "contextual screenshot; values cannot be treated as machine-readable measurements",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    if not manifest_rows:
        raise RuntimeError("raw manifest is empty")

    manifest_paths = {Path(row["relative_path"]) for row in manifest_rows}
    raw_paths = {path.relative_to(RAW) for path in RAW.rglob("*") if path.is_file()}
    if manifest_paths != raw_paths:
        missing = sorted(path.as_posix() for path in manifest_paths - raw_paths)
        untracked = sorted(path.as_posix() for path in raw_paths - manifest_paths)
        raise RuntimeError(
            f"raw/manifest path mismatch; missing={missing}, untracked={untracked}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    sql_tables: list[dict[str, object]] = []
    excel_sheets: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = defaultdict(list)

    for manifest_row in manifest_rows:
        relative = Path(manifest_row["relative_path"])
        path = RAW / relative
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != manifest_row["sha256"]:
            raise RuntimeError(f"raw integrity failure: {relative}")
        hashes[actual_hash].append(relative.as_posix())
        role, confidence, note = role_for(path)
        suffix = path.suffix.lower()
        if suffix == ".sql":
            details, tables = analyze_sql(path)
            sql_tables.extend(tables)
            encoding = detect_text_encoding(path)
            detected_format = "PostgreSQL plain-text dump"
        elif suffix == ".xlsx":
            details, sheets = xlsx_metadata(path)
            excel_sheets.extend(sheets)
            encoding = "binary OOXML"
            detected_format = "Excel OOXML workbook"
        elif suffix == ".png":
            details = png_metadata(path)
            encoding = "binary"
            detected_format = "PNG screenshot"
        elif path.name.startswith("query_"):
            details = analyze_query(path)
            encoding = detect_text_encoding(path)
            detected_format = "delimited text export"
        else:
            details = analyze_event_text(path)
            encoding = detect_text_encoding(path)
            detected_format = "event/debug text log"
        inventory.append(
            {
                "relative_path": relative.as_posix(),
                "experiment": manifest_row["experiment"],
                "name": manifest_row["name"],
                "extension": manifest_row["extension"],
                "size_bytes": int(manifest_row["size_bytes"]),
                "sha256": actual_hash,
                "detected_format": detected_format,
                "encoding": encoding,
                "role": role,
                "interpretation_confidence": confidence,
                "role_note": note,
                **details,
            }
        )

    write_csv(OUT / "file_inventory.csv", inventory)
    write_csv(OUT / "sql_table_inventory.csv", sql_tables)
    write_csv(OUT / "excel_sheet_inventory.csv", excel_sheets)
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "raw_file_count": len(inventory),
        "raw_bytes": sum(int(row["size_bytes"]) for row in inventory),
        "raw_integrity": "PASS",
        "stochastic_seed": None,
        "transformations_performed": [],
        "duplicate_content_groups": duplicates,
        "limitation": "artifact-tool loader unavailable; Excel inspection is preliminary OOXML metadata only",
    }
    (OUT / "inventory_metadata.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(environment, indent=2))
    print(f"inventory={OUT / 'file_inventory.csv'}")
    print(f"sql_tables={OUT / 'sql_table_inventory.csv'}")
    print(f"excel_sheets={OUT / 'excel_sheet_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
