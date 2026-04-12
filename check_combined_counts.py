"""
Check counts in a combined projected dataset.

- Counts successful SN runs per sn_type using unique iteration_id
  (since each run produces many photometry rows).
- Confirms failed OIDs are absent/present.
- Verifies that target OIDs exist in the observing log (chunked scan).

Usage (PowerShell):
  conda activate ALR_37
  python check_combined_counts.py --combined outputs/multiband_runs/<file>.csv --obslog data/ZTF_observing_log_complete.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


def _norm_type(x: str) -> str:
    s = str(x).strip().lower()
    if "ia" in s:
        return "Ia"
    if "ibc" in s or s.endswith(" ib") or s.endswith(" ic") or " ib" in s or " ic" in s:
        return "Ibc"
    if "ii" in s:
        return "II"
    return str(x)


def _read_failures_from_latest_group(batch_runs_dir: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (oid, sn_type) for failures in latest YYYYMMDD_HHMMSS group.
    """
    run_summaries = [p for p in batch_runs_dir.glob("*/run_summary.csv") if p.is_file()]
    if not run_summaries:
        return []
    prefixes = sorted({p.parent.name[:15] for p in run_summaries})
    if not prefixes:
        return []
    latest_prefix = prefixes[-1]
    failures: List[Tuple[str, str]] = []
    for p in run_summaries:
        if not p.parent.name.startswith(latest_prefix):
            continue
        df = pd.read_csv(p, usecols=["field_oid", "sn_type", "success"])
        df["success"] = df["success"].astype(str).str.lower().eq("true")
        df_fail = df[~df["success"]]
        for _, r in df_fail.iterrows():
            failures.append((str(r["field_oid"]), _norm_type(r["sn_type"])))
    return failures


def _scan_obslog_for_oids(obslog_path: Path, target_oids: Set[str], chunksize: int = 500_000) -> Set[str]:
    """
    Streaming scan: returns the subset of target_oids that are present in obslog.
    """
    found: Set[str] = set()
    remaining = set(target_oids)
    if not remaining:
        return found

    for chunk in pd.read_csv(obslog_path, usecols=["oid"], dtype={"oid": "string"}, chunksize=chunksize):
        oids = set(chunk["oid"].dropna().astype(str).tolist())
        hit = remaining.intersection(oids)
        if hit:
            found.update(hit)
            remaining.difference_update(hit)
            if not remaining:
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=str, required=True, help="Path to combined CSV")
    ap.add_argument("--obslog", type=str, required=True, help="Path to observing log CSV (ZTF_observing_log_complete.csv)")
    ap.add_argument("--chunksize", type=int, default=500_000, help="Chunksize for obslog scan")
    args = ap.parse_args()

    combined_path = Path(args.combined)
    obslog_path = Path(args.obslog)
    if not combined_path.exists():
        raise FileNotFoundError(combined_path)
    if not obslog_path.exists():
        raise FileNotFoundError(obslog_path)

    df = pd.read_csv(combined_path)
    # runs are uniquely identified by iteration_id
    df["sn_type_norm"] = df["sn_type"].map(_norm_type)

    runs = df[["iteration_id", "sn_type_norm", "field_oid"]].drop_duplicates("iteration_id")
    runs_by_type = runs.groupby("sn_type_norm")["iteration_id"].nunique().to_dict()
    total_runs = int(runs["iteration_id"].nunique())
    unique_targets = int(runs["field_oid"].nunique())

    # failures from latest group of batch_runs
    failures = _read_failures_from_latest_group(Path("outputs") / "batch_runs")
    failed_oids = sorted({oid for oid, _t in failures})
    failed_present = sorted(set(failed_oids).intersection(set(df["field_oid"].astype(str).unique().tolist())))

    # verify oids exist in observing log
    target_oids = set(runs["field_oid"].astype(str).unique().tolist())
    found = _scan_obslog_for_oids(obslog_path, target_oids, chunksize=int(args.chunksize))
    missing = sorted(target_oids.difference(found))

    print("=== Combined dataset counts ===")
    print(f"combined_file: {combined_path}")
    print(f"rows_photometry: {len(df)}")
    print(f"runs_successful (unique iteration_id): {total_runs}")
    print(f"unique_targets (field_oid): {unique_targets}")
    print("runs_by_type:", runs_by_type)
    print("")

    print("=== Failures (latest batch group) ===")
    print(f"n_failed_rows_latest_group: {len(failures)}")
    print(f"n_failed_unique_oids_latest_group: {len(failed_oids)}")
    if failed_oids:
        print("failed_oids:", failed_oids)
    print(f"failed_oids_present_in_combined: {failed_present}")
    print("")

    print("=== OID presence in observing log ===")
    print(f"obslog_file: {obslog_path}")
    print(f"targets_checked: {len(target_oids)}")
    print(f"targets_found_in_obslog: {len(found)}")
    print(f"targets_missing_in_obslog: {len(missing)}")
    if missing:
        print("missing_oids:", missing[:50])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

