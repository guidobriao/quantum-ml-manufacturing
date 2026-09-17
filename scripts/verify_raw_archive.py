"""Verify the local raw archive and generate its content-free integrity manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "File esperimenti"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST = PROJECT_ROOT / "data" / "raw_manifest.csv"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_from_manifest() -> int:
    """Verify the moved archive when the former source directories are absent."""
    if not MANIFEST.is_file():
        print(f"ERROR: manifest not found: {MANIFEST}", file=sys.stderr)
        return 1

    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    required_columns = {
        "relative_path",
        "experiment",
        "name",
        "extension",
        "size_bytes",
        "sha256",
    }
    if not rows or not required_columns.issubset(rows[0]):
        print("ERROR: manifest is empty or has invalid columns", file=sys.stderr)
        return 1

    errors: list[str] = []
    manifest_paths: set[Path] = set()
    hashes: dict[tuple[int, str], list[str]] = {}
    total_bytes = 0

    for row in rows:
        relative_path = Path(row["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe manifest path: {relative_path}")
            continue

        archive_file = RAW_ROOT / relative_path
        manifest_paths.add(relative_path)
        if not archive_file.is_file():
            errors.append(f"archive file missing: {relative_path}")
            continue

        expected_size = int(row["size_bytes"])
        expected_hash = row["sha256"]
        actual_size = archive_file.stat().st_size
        actual_hash = sha256(archive_file)
        total_bytes += actual_size
        hashes.setdefault((actual_size, actual_hash), []).append(relative_path.as_posix())

        if actual_size != expected_size:
            errors.append(f"size mismatch: {relative_path}")
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative_path}")

    archive_paths = {
        path.relative_to(RAW_ROOT)
        for path in RAW_ROOT.rglob("*")
        if path.is_file()
    }
    for path in sorted(manifest_paths - archive_paths):
        errors.append(f"archive path missing: {path}")
    for path in sorted(archive_paths - manifest_paths):
        errors.append(f"unexpected archive path: {path}")

    if errors:
        print("ARCHIVE VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    print("source_archive=not_present (manifest verification mode)")
    print(f"files={len(rows)}")
    print(f"bytes={total_bytes}")
    print("sha256_verification=PASS")
    print(f"duplicate_content_groups={len(duplicate_groups)}")
    for group in duplicate_groups:
        print("duplicate=" + " | ".join(group))
    print(f"manifest={MANIFEST}")
    return 0


def main() -> int:
    experiments = sorted(
        path for path in SOURCE_ROOT.rglob("Exp*") if path.is_dir()
    )
    if not experiments:
        return verify_from_manifest()

    for root in (SOURCE_ROOT, RAW_ROOT):
        symlinks = [path for path in root.rglob("*") if path.is_symlink()]
        if symlinks:
            print(f"ERROR: symbolic links found below {root}", file=sys.stderr)
            for path in symlinks:
                print(path, file=sys.stderr)
            return 1

    rows: list[dict[str, object]] = []
    source_paths: set[Path] = set()
    destination_paths: set[Path] = set()
    hashes: dict[tuple[int, str], list[str]] = {}
    errors: list[str] = []

    for experiment in experiments:
        destination_experiment = RAW_ROOT / experiment.name
        if not destination_experiment.is_dir():
            errors.append(f"missing destination directory: {destination_experiment}")
            continue

        for source_file in sorted(path for path in experiment.rglob("*") if path.is_file()):
            inside_experiment = source_file.relative_to(experiment)
            relative_path = Path(experiment.name) / inside_experiment
            destination_file = RAW_ROOT / relative_path
            source_paths.add(relative_path)

            if not destination_file.is_file():
                errors.append(f"missing destination file: {relative_path}")
                continue

            destination_paths.add(relative_path)
            source_size = source_file.stat().st_size
            destination_size = destination_file.stat().st_size
            source_hash = sha256(source_file)
            destination_hash = sha256(destination_file)

            if source_size != destination_size:
                errors.append(f"size mismatch: {relative_path}")
            if source_hash != destination_hash:
                errors.append(f"SHA-256 mismatch: {relative_path}")

            hashes.setdefault((source_size, source_hash), []).append(relative_path.as_posix())
            rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "experiment": experiment.name,
                    "name": source_file.name,
                    "extension": source_file.suffix,
                    "size_bytes": source_size,
                    "sha256": source_hash,
                }
            )

    for destination_file in sorted(path for path in RAW_ROOT.rglob("*") if path.is_file()):
        destination_paths.add(destination_file.relative_to(RAW_ROOT))

    if source_paths != destination_paths:
        for path in sorted(source_paths - destination_paths):
            errors.append(f"destination path missing: {path}")
        for path in sorted(destination_paths - source_paths):
            errors.append(f"unexpected destination path: {path}")

    if errors:
        print("ARCHIVE VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = MANIFEST.with_suffix(".csv.tmp")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "relative_path",
                "experiment",
                "name",
                "extension",
                "size_bytes",
                "sha256",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(MANIFEST)

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    print(f"experiments={len(experiments)}")
    print(f"files={len(rows)}")
    print(f"bytes={sum(int(row['size_bytes']) for row in rows)}")
    print("sha256_verification=PASS")
    print(f"duplicate_content_groups={len(duplicate_groups)}")
    for group in duplicate_groups:
        print("duplicate=" + " | ".join(group))
    print(f"manifest={MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
