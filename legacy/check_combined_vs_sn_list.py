"""
Compare combined projected dataset vs sn_list_to_project.csv.

Goal:
- Count unique successful runs per SN type in the combined CSV (unique iteration_id).
- Count requested targets per SN type in sn_list_to_project.csv.
- Subtract failures from the latest batch group (from outputs/batch_runs/*/run_summary.csv).
- Verify counts match and list missing/extra OIDs.

No pandas dependency (stdlib csv only).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def norm_type(x: str) -> str:
    s = str(x).strip().lower()
    if "ia" in s:
        return "Ia"
    if "ibc" in s or s.endswith(" ib") or s.endswith(" ic") or " ib" in s or " ic" in s:
        return "Ibc"
    if "ii" in s:
        return "II"
    return str(x).strip()


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_latest_prefix(batch_runs_dir: Path) -> Optional[str]:
    run_summaries = [p for p in batch_runs_dir.glob("*/run_summary.csv") if p.is_file()]
    if not run_summaries:
        return None
    prefixes = sorted({p.parent.name[:15] for p in run_summaries})
    return prefixes[-1] if prefixes else None


def read_failures(batch_runs_dir: Path, batch_prefix: Optional[str] = None) -> Set[str]:
    """
    Returns failed OIDs (field_oid) for the chosen batch_prefix group.
    """
    if batch_prefix is None:
        batch_prefix = infer_latest_prefix(batch_runs_dir)
    if not batch_prefix:
        return set()

    failed: Set[str] = set()
    for p in batch_runs_dir.glob(f"{batch_prefix}_*/run_summary.csv"):
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                success = str(r.get("success", "")).strip().lower()
                if success == "false":
                    failed.add(str(r.get("field_oid", "")).strip())
    return {x for x in failed if x}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True, help="Combined projected CSV path")
    ap.add_argument("--sn-list", default=str(Path("data") / "sn_list_to_project.csv"), help="sn_list_to_project.csv path")
    ap.add_argument("--batch-prefix", default=None, help="Batch prefix to read failures (e.g. 20260125_231012). Default: infer latest")
    args = ap.parse_args()

    combined_path = Path(args.combined)
    sn_list_path = Path(args.sn_list)
    batch_runs_dir = Path("outputs") / "batch_runs"

    if not combined_path.exists():
        raise FileNotFoundError(combined_path)
    if not sn_list_path.exists():
        raise FileNotFoundError(sn_list_path)

    # Failures
    failed_oids = read_failures(batch_runs_dir, batch_prefix=args.batch_prefix)

    # sn_list counts (requested)
    requested_rows = read_csv_dicts(sn_list_path)
    requested_oids: List[str] = [str(r.get("sn_name", "")).strip() for r in requested_rows]
    requested_oids = [x for x in requested_oids if x]
    requested_unique_oids = set(requested_oids)
    requested_type_by_oid: Dict[str, str] = {}
    requested_counts = Counter()
    for r in requested_rows:
        oid = str(r.get("sn_name", "")).strip()
        if not oid:
            continue
        t = norm_type(r.get("sn_type", ""))
        requested_type_by_oid[oid] = t
        requested_counts[t] += 1

    # Combined counts (successful runs)
    # Use unique iteration_id as run identifier, and sn_type from the file.
    run_type_by_iteration: Dict[str, str] = {}
    run_oid_by_iteration: Dict[str, str] = {}
    combined_oids: Set[str] = set()
    combined_counts = Counter()

    with combined_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            it = str(r.get("iteration_id", "")).strip()
            if not it:
                continue
            if it in run_type_by_iteration:
                continue
            t = norm_type(r.get("sn_type", ""))
            oid = str(r.get("field_oid", "")).strip()
            run_type_by_iteration[it] = t
            run_oid_by_iteration[it] = oid
            combined_counts[t] += 1
            if oid:
                combined_oids.add(oid)

    # Expected successful = requested - failed
    expected_success_oids = requested_unique_oids.difference(failed_oids)
    missing_in_combined = sorted(expected_success_oids.difference(combined_oids))
    extra_in_combined = sorted(combined_oids.difference(requested_unique_oids))

    # Expected counts by type (requested minus failures per type)
    failed_counts = Counter()
    for oid in failed_oids:
        t = requested_type_by_oid.get(oid, "UNKNOWN")
        failed_counts[t] += 1
    expected_counts = Counter(requested_counts)
    for t, n in failed_counts.items():
        expected_counts[t] -= n

    print("=== Requested (sn_list_to_project.csv) ===")
    print(f"rows: {len(requested_rows)}")
    print(f"unique_oids: {len(requested_unique_oids)}")
    print("counts_by_type:", dict(requested_counts))
    print("")

    print("=== Failures (from outputs/batch_runs) ===")
    print(f"batch_prefix: {args.batch_prefix or infer_latest_prefix(batch_runs_dir)}")
    print(f"failed_unique_oids: {len(failed_oids)}")
    print("failed_counts_by_type:", dict(failed_counts))
    if failed_oids:
        print("failed_oids:", sorted(failed_oids))
    print("")

    print("=== Combined (successful runs inferred from combined CSV) ===")
    print(f"rows_photometry: (not computed)")
    print(f"runs_successful (unique iteration_id): {len(run_type_by_iteration)}")
    print(f"unique_oids_in_combined: {len(combined_oids)}")
    print("counts_by_type:", dict(combined_counts))
    print("")

    print("=== Consistency checks ===")
    print(f"expected_success_unique_oids (requested - failed): {len(expected_success_oids)}")
    print(f"missing_in_combined: {len(missing_in_combined)}")
    print(f"extra_in_combined: {len(extra_in_combined)}")
    if missing_in_combined:
        print("missing_oids (first 50):", missing_in_combined[:50])
    if extra_in_combined:
        print("extra_oids (first 50):", extra_in_combined[:50])
    print("")

    print("expected_counts_by_type:", dict(expected_counts))
    print("combined_counts_by_type:", dict(combined_counts))

    ok = (len(missing_in_combined) == 0 and len(extra_in_combined) == 0 and expected_counts == combined_counts)
    print(f"\nOK_MATCH: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

