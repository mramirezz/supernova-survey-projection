"""
CONSTRUYE CATÁLOGO UNIFICADO DE COORDENADAS POR OID
====================================================

Merge de todas las fuentes locales con coords ZTF:
  1. data/ztf_targets_with_coords_multicat_summary.csv (BTS con cross-match)
  2. ../TNS_ZTF_df_new.csv (TNS ZTF + clasificación spectra)
  3. (futuro: Alerce API si se instala)

Extrae el OID ZTF* de cada fila (de 'sn_name' directo o del
campo 'internal_names' de TNS), toma ra/dec preferentemente de la
primera fuente y fallback a las siguientes.

Salida: data/ztf_coords_master.parquet
    oid, ra_deg, dec_deg, source

Este catálogo es consumido por:
  - run_per_field.py (para el lookup ebmv_mw_oid)
  - tools/precompute_sfd98.py (para el cache SFD98 masivo)

Uso:
    python tools/build_coords_catalog.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHD_ROOT = ROOT.parents[1]  # .../paper2_ZTF/

# Fuentes (orden de precedencia: la primera gana si hay conflicto)
SOURCES = [
    {
        "name": "bts_multicat",
        "path": ROOT / "data" / "ztf_targets_with_coords_multicat_summary.csv",
        "oid_col": "sn_name",
        "ra_col": "ra_used_deg",
        "dec_col": "dec_used_deg",
        "needs_extraction": False,
    },
    {
        "name": "tns",
        "path": PHD_ROOT / "TNS_ZTF_df_new.csv",
        "oid_col": "internal_names",  # contiene "ZTF25..., ATLAS..., etc"
        "ra_col": "ra",
        "dec_col": "declination",
        "needs_extraction": True,
    },
]


ZTF_OID_RE = re.compile(r"ZTF\w+")


def extract_ztf_oid(s) -> str | None:
    """Extrae el primer ZTF* de una cadena tipo 'ZTF25abc, ATLAS25xyz, ...'."""
    if pd.isna(s):
        return None
    m = ZTF_OID_RE.search(str(s))
    return m.group(0) if m else None


def load_source(cfg: dict) -> pd.DataFrame:
    """Lee una fuente y retorna DataFrame con [oid, ra_deg, dec_deg, source]."""
    if not cfg["path"].exists():
        print(f"  [skip] {cfg['name']}: archivo no existe en {cfg['path']}")
        return pd.DataFrame(columns=["oid", "ra_deg", "dec_deg", "source"])

    cols = [cfg["oid_col"], cfg["ra_col"], cfg["dec_col"]]
    df = pd.read_csv(cfg["path"], usecols=cols)
    df = df.rename(columns={cfg["ra_col"]: "ra_deg", cfg["dec_col"]: "dec_deg"})

    if cfg["needs_extraction"]:
        df["oid"] = df[cfg["oid_col"]].apply(extract_ztf_oid)
    else:
        df["oid"] = df[cfg["oid_col"]]

    df = df[["oid", "ra_deg", "dec_deg"]].copy()
    df = df.dropna(subset=["oid", "ra_deg", "dec_deg"])
    df["source"] = cfg["name"]
    print(f"  [{cfg['name']}] {len(df):,} OIDs con coords válidas")
    return df


def main():
    print(f"Construyendo catálogo de coords")
    print(f"  PHD_ROOT: {PHD_ROOT}")
    print()

    frames = [load_source(s) for s in SOURCES]
    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal bruto (con duplicados): {len(df):,}")

    # Dedup: mantener primera ocurrencia (orden = precedencia)
    df = df.drop_duplicates(subset=["oid"], keep="first")
    print(f"Únicos después de dedup: {len(df):,}")

    # Cobertura vs observing log
    obslog_path = ROOT / "data" / "ZTF_observing_log_complete.csv"
    if obslog_path.exists():
        df_obs = pd.read_csv(obslog_path, usecols=["oid"])
        obslog_oids = set(df_obs["oid"].unique())
        cov = df["oid"].isin(obslog_oids).sum()
        print(f"\nCobertura vs obslog ({len(obslog_oids):,} OIDs):")
        print(f"  OIDs con coords que aparecen en obslog: {cov:,} "
              f"({100*cov/len(obslog_oids):.1f}%)")

    # Guardar
    out_path = ROOT / "data" / "ztf_coords_master.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nGuardado: {out_path}")
    print(f"  Columnas: {list(df.columns)}")
    print(f"  Por source:")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
