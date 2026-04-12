"""
Summarize failures from the latest parallel batch group.

Looks for `outputs/batch_runs/<batch_id>/run_summary.csv`, groups by the newest
timestamp prefix `YYYYMMDD_HHMMSS`, and reports all rows with `success=False`.

Usage:
  python summarize_latest_failures.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FailureRow:
    batch_dir: str
    iteration_index: str
    field_oid: str
    sn_type: str
    sn_name: str
    template_file: str
    redshift: str
    required_filter: str
    n_det_required_filter: str
    min_det: str
    error: str


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latest_prefix(batch_dirs: List[Path]) -> Optional[str]:
    prefixes = sorted({p.name[:15] for p in batch_dirs})
    return prefixes[-1] if prefixes else None


def main() -> int:
    root = Path("outputs") / "batch_runs"
    if not root.exists():
        print("[ERROR] No existe outputs/batch_runs")
        return 2

    batch_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "run_summary.csv").exists()]
    if not batch_dirs:
        print("[ERROR] No hay run_summary.csv en outputs/batch_runs")
        return 2

    prefix = _latest_prefix(batch_dirs)
    if not prefix:
        print("[ERROR] No pude inferir prefix latest")
        return 2

    group = sorted([p for p in batch_dirs if p.name.startswith(prefix)], key=lambda p: p.name)
    print(f"latest_prefix: {prefix}")
    print(f"batches_in_group: {len(group)}")

    total_rows = 0
    failures: List[FailureRow] = []
    per_batch: Dict[str, Dict[str, int]] = {}

    for p in group:
        rows = _read_csv(p / "run_summary.csv")
        total_rows += len(rows)
        batch_fail = [r for r in rows if str(r.get("success", "")).strip().lower() == "false"]
        per_batch[p.name] = {"rows": len(rows), "failed": len(batch_fail)}

        for r in batch_fail:
            failures.append(
                FailureRow(
                    batch_dir=p.name,
                    iteration_index=str(r.get("iteration_index", "")),
                    field_oid=str(r.get("field_oid", "")),
                    sn_type=str(r.get("sn_type", "")),
                    sn_name=str(r.get("sn_name", "")),
                    template_file=str(r.get("template_file", "")),
                    redshift=str(r.get("redshift", "")),
                    required_filter=str(r.get("required_filter", "")),
                    n_det_required_filter=str(r.get("n_detections_required_filter", "")),
                    min_det=str(r.get("min_detections_required", "")),
                    error=str(r.get("error", "")),
                )
            )

    print(f"total_rows_in_group: {total_rows}")
    print(f"total_failed_in_group: {len(failures)}")

    print("\nper_batch:")
    for k in sorted(per_batch.keys()):
        v = per_batch[k]
        print(f"  {k}: rows={v['rows']} failed={v['failed']}")

    if failures:
        print("\nFAILED_ROWS:")
        for f in failures:
            print(
                f"- oid={f.field_oid} type={f.sn_type} z={f.redshift} "
                f"req={f.required_filter} det={f.n_det_required_filter}/{f.min_det} "
                f"batch={f.batch_dir} err={f.error}"
            )
    else:
        print("\nNo failures in latest group.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

