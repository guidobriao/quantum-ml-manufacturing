# Read-only audit of the definitive common pipeline

Audit date: 2026-09-11  
Scope: synchronized PLC silences and valid long-age fusion matches  
Result: **PASS — no substantive anomaly found**

The audit used the existing SQLite output in read-only mode and verified raw files against
`data/raw_manifest.csv`. It did not regenerate the pipeline and did not perform windowing, feature
extraction, clustering, `inferred_state`, classification, or QML.

## Synchronized PLC silences

Station map: 10 Manual; 20 MagazineFront; 30 DrillingCPS; 50 RobotAssembly; 60
CameraInspection; 70 MagazineBack; 80 Press. Times are UTC.

| Experiment / silence | Interval | Duration (s) | Supporting stations | Required / active | Supporting gaps | Power rows inside supporting gaps | Violations |
|---|---|---:|---|---:|---:|---:|---:|
| Exp20190124 / SYNC-001 | 2019-01-24 09:30:38.489 – 09:50:00.229 | 1161.740 | 10, 20, 30, 50, 60, 70, 80 | 5 / 7 | 9 | 10,708 | 0 |
| Exp20190320 / SYNC-001 | 2019-03-20 14:58:11.250 – 16:06:18.865 | 4087.615 | 10, 20, 30, 60, 70, 80 | 4 / 6 | 12 | 53,622 | 0 |
| Exp20190416 / SYNC-001 | 2019-04-16 07:35:23.149 – 08:18:24.955 | 2581.806 | 10, 20, 30, 60, 70, 80 | 5 / 7 | 6 | 2,119 | 0 |
| Exp20190416 / SYNC-002 | 2019-04-16 09:16:23.655 – 12:26:32.963 | 11409.308 | 10, 20, 30, 60, 70, 80 | 5 / 7 | 6 | 4,714 | 0 |
| Exp20190514 / SYNC-001 | 2019-05-14 08:04:25.024 – 08:18:56.866 | 871.842 | 10, 20, 30, 60, 70, 80 | 5 / 7 | 6 | 11,296 | 0 |
| Exp20190514 / SYNC-002 | 2019-05-14 09:19:42.847 – 09:25:32.226 | 349.379 | 20, 30, 60, 70, 80 | 5 / 7 | 5 | 3,625 | 0 |
| Exp20190617 / SYNC-001 | 2019-06-17 10:04:49.299 – 12:02:07.391 | 7038.092 | 10, 20, 30, 50, 60, 70, 80 | 5 / 7 | 7 | 8,432 | 0 |

For every one of the 49 station-specific supporting gaps:

- both endpoints occur in `operation_log_canonical` and are consecutive distinct
  `tblOperationLog` timestamps for that experiment and station;
- there are zero `tblOperationLog` records strictly between the endpoints;
- across 94,516 unique power rows strictly inside those gaps, `session_id` is NULL,
  `join_matched=false`, `join_ambiguous=false`, and all PLC/join/MES columns are NULL, including
  `operation_row_id`, PLC timestamp/resource/lineage fields, all seven boolean state fields,
  `OperationNo`, `WorkPlanNo`, `OrderNo`, `StepNo`, `CarrierID`, `iResourceID`, `OrderPosition`,
  `PartNumber`, and `state_age_seconds`.

This verifies real source discontinuities rather than algorithmic gaps. The synchronized `core` is
a merged majority-consensus interval. Individual stations can emit within that core while the
required majority remains silent; nullness therefore applies to each station's actual consecutive
supporting gap, which is also the interval used to create that station's session boundary.

The detector parameters are explicit and non-temporal: 3 longest candidate gaps per station,
minimum support of 60% of active stations (plus minimum count 2), and duration longer than 20% of
the longest synchronized candidate in the same experiment. The last quantity is relative and
dimensionless. None is an absolute threshold in seconds. Both sessionization levels use only
`tblOperationLog`.

## Valid matches from 600 s to the observed maximum

There are 13,857 valid matches in the requested range and 13 experiment–station groups. They occur
only in Exp20190416 and Exp20190617. The maximum is 1205.278 s. One representative per group was
chosen deterministically as its maximum-age match; 600 s is only the user-requested audit selection,
not a new pipeline threshold.

| Experiment | Station | Matches | Observed age range (s) | Representative power UTC | Matched PLC UTC | Next PLC UTC | Next PLC after power (s) |
|---|---:|---:|---:|---|---|---|---:|
| Exp20190416 | 10 | 51 | 620.456–818.755 | 13:29:24.461 | 13:15:45.706 | 13:29:24.704 | 0.243 |
| Exp20190416 | 20 | 106 | 614.701–659.735 | 14:35:24.533 | 14:24:24.798 | 14:35:24.769 | 0.236 |
| Exp20190416 | 30 | 425 | 600.676–880.422 | 13:30:40.119 | 13:15:59.697 | 13:30:40.968 | 0.849 |
| Exp20190416 | 50 | 801 | 600.929–877.067 | 16:16:44.440 | 16:02:07.373 | 16:16:44.867 | 0.427 |
| Exp20190416 | 60 | 387 | 684.296–864.278 | 16:18:31.472 | 16:04:07.194 | 16:18:32.184 | 0.712 |
| Exp20190416 | 70 | 393 | 667.494–861.268 | 16:18:45.034 | 16:04:23.766 | 16:18:45.214 | 0.180 |
| Exp20190416 | 80 | 636 | 647.275–857.985 | 16:19:01.967 | 16:04:43.982 | 16:19:02.514 | 0.547 |
| Exp20190617 | 20 | 565 | 600.003–812.016 | 09:49:14.366 | 09:35:42.350 | 09:49:14.830 | 0.464 |
| Exp20190617 | 30 | 510 | 600.007–802.821 | 09:49:24.551 | 09:36:01.730 | 09:49:25.207 | 0.656 |
| Exp20190617 | 50 | 4,338 | 600.214–1205.278 | 09:56:40.843 | 09:36:35.565 | 09:56:41.033 | 0.190 |
| Exp20190617 | 60 | 2,037 | 600.122–972.941 | 09:38:02.224 | 09:21:49.283 | 09:38:02.246 | 0.022 |
| Exp20190617 | 70 | 1,895 | 600.020–968.672 | 09:38:14.653 | 09:22:05.981 | 09:38:15.358 | 0.705 |
| Exp20190617 | 80 | 1,713 | 600.054–920.349 | 09:58:06.272 | 09:42:45.923 | 09:58:06.385 | 0.113 |

### PLC sequences around representatives

Each sequence shows the last three PLC timestamps ending at the matched event, then the first three
PLC timestamps after it. There is no PLC timestamp between the matched event and representative
power sample. Date is the experiment date and times are UTC.

| Experiment | Station | Previous → matched PLC times | Following PLC times |
|---|---:|---|---|
| Exp20190416 | 10 | 13:15:42.222 → 13:15:44.245 → **13:15:45.706** | 13:29:24.704 → 13:29:25.701 → 13:29:27.694 |
| Exp20190416 | 20 | 14:24:21.282 → 14:24:23.290 → **14:24:24.798** | 14:35:24.769 → 14:35:26.266 → 14:35:28.290 |
| Exp20190416 | 30 | 13:15:59.156 → 13:15:59.695 → **13:15:59.697** | 13:30:40.968 → 13:30:42.969 → 13:30:44.959 |
| Exp20190416 | 50 | 16:02:06.385 → 16:02:06.403 → **16:02:07.373** | 16:16:44.867 → 16:16:44.872 → 16:16:44.878 |
| Exp20190416 | 60 | 16:04:03.705 → 16:04:05.697 → **16:04:07.194** | 16:18:32.184 → 16:18:33.667 → 16:18:36.174 |
| Exp20190416 | 70 | 16:04:20.762 → 16:04:23.756 → **16:04:23.766** | 16:18:45.214 → 16:18:46.709 → 16:18:49.213 |
| Exp20190416 | 80 | 16:04:40.502 → 16:04:42.490 → **16:04:43.982** | 16:19:02.514 → 16:19:03.966 → 16:19:05.947 |
| Exp20190617 | 20 | 09:35:37.878 → 09:35:39.858 → **09:35:42.350** | 09:49:14.830 → 09:49:16.324 → 09:49:18.325 |
| Exp20190617 | 30 | 09:35:58.431 → 09:36:00.743 → **09:36:01.730** | 09:49:25.207 → 09:49:26.712 → 09:49:28.708 |
| Exp20190617 | 50 | 09:36:34.573 → 09:36:34.591 → **09:36:35.565** | 09:56:41.033 → 09:56:41.042 → 09:56:41.051 |
| Exp20190617 | 60 | 09:21:45.818 → 09:21:48.279 → **09:21:49.283** | 09:38:02.246 → 09:38:03.732 → 09:38:06.284 |
| Exp20190617 | 70 | 09:22:02.981 → 09:22:05.968 → **09:22:05.981** | 09:38:15.358 → 09:38:16.863 → 09:38:19.356 |
| Exp20190617 | 80 | 09:42:39.414 → 09:42:41.419 → **09:42:45.923** | 09:58:06.385 → 09:58:07.382 → 09:58:09.887 |

### MES state at matched and next PLC events

Vector order is `Busy, RFIDTagPresent, Done, StationEntryxBG5, ReadyAtStationxBG1,
DoneWorkingxBG9, StationExitxBG6, OperationNo, WorkPlanNo, OrderNo, StepNo, CarrierID,
iResourceID, OrderPosition, PartNumber`. `N` means SQL NULL.

| Experiment | Station | Matched PLC vector | Next PLC vector |
|---|---:|---|---|
| Exp20190416 | 10 | 0,0,1,0,0,N,0,0,N,0,N,2,0,0,0 | 0,0,1,1,0,N,0,0,N,0,N,2,0,0,0 |
| Exp20190416 | 20 | 0,0,1,0,0,N,0,122,N,4299,N,2,3,6,210 | 0,0,1,1,0,N,0,122,N,4299,N,2,3,6,210 |
| Exp20190416 | 30 | 1,1,0,0,1,N,1,0,N,0,N,3,0,0,0 | 0,0,1,1,0,N,0,510,N,4297,N,2,1,1,1224 |
| Exp20190416 | 50 | 0,0,1,N,N,N,N,301,N,4302,N,2,5,6,210 | 0,0,0,N,N,N,N,301,N,4302,N,2,5,6,210 |
| Exp20190416 | 60 | 0,0,1,0,0,N,0,200,N,4302,N,2,7,6,210 | 0,0,1,1,0,N,0,200,N,4302,N,2,7,6,210 |
| Exp20190416 | 70 | 0,0,1,0,0,N,0,110,N,4302,N,2,8,6,210 | 0,0,1,1,0,N,0,110,N,4302,N,2,8,6,210 |
| Exp20190416 | 80 | 0,0,1,0,0,N,0,510,N,4302,N,2,1,6,1211 | 0,0,1,1,0,N,0,510,N,4302,N,2,1,6,1211 |
| Exp20190617 | 20 | 0,0,1,0,0,N,0,122,N,4463,N,5,3,3,210 | 0,0,1,1,0,N,0,122,N,4463,N,5,3,3,210 |
| Exp20190617 | 30 | 0,0,1,0,0,N,0,304,N,4463,N,5,5,3,210 | 0,0,1,1,0,N,0,304,N,4463,N,5,5,3,210 |
| Exp20190617 | 50 | 0,0,1,N,N,N,N,304,N,4463,N,5,5,3,210 | 0,0,0,N,N,N,N,304,N,4463,N,5,5,3,210 |
| Exp20190617 | 60 | 0,0,1,0,0,N,0,200,N,4462,N,3,7,4,210 | 0,0,1,1,0,N,0,200,N,4462,N,3,7,4,210 |
| Exp20190617 | 70 | 0,0,1,0,0,N,1,110,N,4462,N,3,8,4,210 | 0,0,1,1,0,N,1,110,N,4462,N,3,8,4,210 |
| Exp20190617 | 80 | 0,0,1,0,0,N,0,510,N,4463,N,5,1,3,1214 | 0,0,1,1,0,N,0,510,N,4463,N,5,1,3,1214 |

For stations 10, 20, 60, 70, and 80 the next event raises `StationEntryxBG5`; for station 50 it
clears `Done`; the Exp20190416 station-30 case begins a new operation/order state. These explicit
next-event changes, occurring just after the representative power record, support event-driven
state persistence until that event rather than indicating a stale join anomaly.

### Power profile in each exact PLC inter-event interval

These are descriptive audit summaries over the observed interval `[matched PLC, next PLC)`, not
fixed or sliding windows and not extracted features. Values are count and min/mean/max.

| Experiment | Station | Rows | ActivePowerL1 | Flow | Pressure |
|---|---:|---:|---|---|---|
| Exp20190416 | 10 | 77 | 44.078 / 49.161 / 65.592 | 0.000 / 0.058 / 0.333 | 5.813 / 5.880 / 5.913 |
| Exp20190416 | 20 | 850 | 42.455 / 42.894 / 52.771 | 0.000 / 0.002 / 0.333 | 5.735 / 5.859 / 5.873 |
| Exp20190416 | 30 | 195 | 46.411 / 46.650 / 47.022 | 0.000 / 0.101 / 0.550 | 5.805 / 5.913 / 5.926 |
| Exp20190416 | 50 | 1,239 | 11.105 / 11.803 / 12.242 | 0.678 / 1.091 / 5.790 | 5.682 / 5.825 / 5.843 |
| Exp20190416 | 60 | 852 | 76.442 / 77.004 / 87.677 | 0.000 / 0.025 / 0.344 | 5.763 / 5.908 / 5.927 |
| Exp20190416 | 70 | 803 | 42.822 / 43.273 / 53.375 | 0.000 / 0.003 / 0.320 | 5.715 / 5.860 / 5.877 |
| Exp20190416 | 80 | 1,185 | 41.045 / 41.535 / 52.820 | 0.000 / 0.302 / 0.442 | 5.740 / 5.886 / 5.903 |
| Exp20190617 | 20 | 1,607 | 42.381 / 42.774 / 52.473 | 0.000 / 0.000 / 0.321 | 5.719 / 5.892 / 5.909 |
| Exp20190617 | 30 | 1,793 | 46.283 / 46.741 / 57.178 | 0.000 / 0.127 / 4.697 | 5.744 / 5.917 / 5.940 |
| Exp20190617 | 50 | 3,562 | 10.982 / 11.746 / 12.314 | 0.341 / 0.753 / 8.169 | 5.648 / 5.838 / 5.857 |
| Exp20190617 | 60 | 2,005 | 76.163 / 76.760 / 87.272 | 0.000 / 0.024 / 0.356 | 5.746 / 5.928 / 5.946 |
| Exp20190617 | 70 | 1,920 | 42.531 / 42.857 / 53.522 | 0.000 / 0.004 / 0.371 | 5.676 / 5.879 / 5.895 |
| Exp20190617 | 80 | 1,985 | 40.681 / 41.062 / 52.937 | 0.000 / 0.069 / 1.562 | 5.695 / 5.893 / 5.918 |

Power sampling remains present throughout these PLC inter-event intervals. Because the join is a
backward event-state join, there is no intervening PLC event to justify a different recorded state,
and the next PLC event explicitly changes a state field, retaining the previous state is plausible.

## Reproducibility record

- Package version: `quantum-ml-manufacturing 0.1.0`.
- Configuration: version 1, seed 20260616, `configs/common_pipeline.yaml`.
- Recorded pipeline generation: 2026-08-25T12:19:40.654166Z.
- Audit runtime: Python 3.11.9; SQLite 3.51.0; pinned dependencies in `pyproject.toml`.
- Audit commands:
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/verify_common_pipeline.py` and
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q`.
- Results: independent verifier PASS; 18 tests passed in 6.23 s.
- Raw integrity: 45 files, 163,719,816 bytes; every path and SHA-256 matches the manifest.
- Forbidden downstream columns/steps: absent; `forbidden_steps_performed=[]`.

SHA-256 identifiers for the accepted version:

| Artifact | SHA-256 |
|---|---|
| `configs/common_pipeline.yaml` | `7b4e9d6fd148cc564b0130e38b55faf28edaa926078a1875926a8d2c16d718f1` |
| `src/qml_thesis/common_pipeline.py` | `6e17eaff627f936a8aadbe02f97b2bfc0d7aaacbe6b45e36940c33afab79f24d` |
| `scripts/verify_common_pipeline.py` | `56e615a4908f8725d4be5f7252823c4d27ce5ab1ee25eb677f282e7373c95a7d` |
| `data/raw_manifest.csv` | `a9251a7c376d591eac3f8b8aac54efdcdbd00cb1304b6c159c76f6560350e0db` |
| `data/processed/common_pipeline.sqlite` | `65c41fd54a1619b4acede6c990b25098ab00460118f80e8dc7586dc056256802` |
| `data/processed/dataset_schemas.json` | `c6ce0ae1a6658a9715470461aa83aa07a2ba993c62d15f5ba15937378af85ab6` |
| `power_log_canonical.csv.gz` | `b5d2513febb803f4cc0f8943db464e4202fe9091dba5d33e9fe2ab0abae84976` |
| `operation_log_canonical.csv.gz` | `ef8c7d9a5abae3a49582fa3a7634497252bc5d0c8b12bafd64b7e264273ba3b4` |
| `power_operation_fused.csv.gz` | `912519c17d34e1300b6b4a749b70a2cab8afbeb7fbeb293e10101e290f570e84` |
