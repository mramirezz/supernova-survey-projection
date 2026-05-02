"""
PRECOMPUTE SFD98 EXTINCTION CACHE
==================================

Consulta IRSA Dust Service para todos los OIDs con coordenadas y guarda
los valores E(B-V)_MW en un cache parquet local. Esto elimina el cuello de
botella de red durante las corridas de producción (30 llamadas redundantes
por OID × 67k OIDs = 2M llamadas HTTP).

Cache: data/sfd98_cache.parquet
Columnas: oid, ra_deg, dec_deg, ebmv_mw, sfd_ok, queried_at

Uso:
    python tools/precompute_sfd98.py                 # Procesa todos los OIDs faltantes
    python tools/precompute_sfd98.py --limit 100     # Primeros 100 OIDs faltantes (test)
    python tools/precompute_sfd98.py --save-every 50 # Guarda cada 50 queries (default 100)
    python tools/precompute_sfd98.py --force         # Re-consulta OIDs ya cacheados

Es resumible: si se interrumpe, al relanzar salta los OIDs ya cacheados.
"""

import os
import sys
import argparse
import time
from datetime import datetime
import pandas as pd

# Añadir root al path para imports relativos al proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DUST_MAPS_QUIET", "1")  # silenciar banner del import
from tools.dust_maps import get_sfd98_extinction_real  # noqa: E402


CACHE_FILENAME = "sfd98_cache.parquet"
COORDS_FILENAME = "ztf_targets_with_coords_multicat_summary.csv"


def load_cache(cache_path):
    """Carga cache existente o retorna DataFrame vacío con el esquema esperado."""
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df
    return pd.DataFrame(columns=["oid", "ra_deg", "dec_deg", "ebmv_mw", "sfd_ok", "queried_at"])


def save_cache(df_cache, cache_path):
    """Guarda cache atómicamente (escribe a tmp y mueve)."""
    tmp = cache_path + ".tmp"
    df_cache.to_parquet(tmp, index=False)
    os.replace(tmp, cache_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute SFD98 E(B-V)_MW cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Máximo de OIDs a procesar en esta corrida (default: todos)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Guardar cache cada N queries (default: 100)")
    parser.add_argument("--force", action="store_true",
                        help="Re-consultar OIDs ya cacheados")
    parser.add_argument("--data-dir", type=str, default=os.path.join(ROOT, "data"),
                        help="Directorio de datos")
    args = parser.parse_args()

    coords_path = os.path.join(args.data_dir, COORDS_FILENAME)
    cache_path = os.path.join(args.data_dir, CACHE_FILENAME)

    if not os.path.exists(coords_path):
        print(f"[ERROR] No se encontró {coords_path}")
        sys.exit(1)

    # Cargar coordenadas
    df_coords = pd.read_csv(coords_path, usecols=["sn_name", "ra_used_deg", "dec_used_deg"])
    df_coords = df_coords.dropna(subset=["ra_used_deg", "dec_used_deg"])
    df_coords = df_coords.rename(columns={"sn_name": "oid",
                                          "ra_used_deg": "ra_deg",
                                          "dec_used_deg": "dec_deg"})
    print(f"Coordenadas disponibles: {len(df_coords):,} OIDs")

    # Cargar cache existente
    df_cache = load_cache(cache_path)
    print(f"Cache existente: {len(df_cache):,} OIDs")

    cached_oids = set(df_cache["oid"].tolist()) if len(df_cache) else set()

    # OIDs pendientes
    if args.force:
        pending = df_coords.copy()
        # al forzar, descartamos entradas previas de los OIDs a reconsultar
        if len(df_cache):
            df_cache = df_cache[~df_cache["oid"].isin(pending["oid"])].copy()
    else:
        pending = df_coords[~df_coords["oid"].isin(cached_oids)].copy()

    print(f"OIDs pendientes: {len(pending):,}")
    if args.limit is not None:
        pending = pending.head(args.limit)
        print(f"  Limitado a los primeros {len(pending):,}")

    if len(pending) == 0:
        print("Nada por hacer.")
        return

    # Estimación de tiempo (asumiendo ~1.5s por query)
    est_s = len(pending) * 1.5
    print(f"Estimación: ~{est_s/60:.0f} min ({est_s/3600:.1f} h)")
    print()

    t0 = time.time()
    new_rows = []
    n_ok = 0
    n_fail = 0

    for i, row in enumerate(pending.itertuples(index=False), start=1):
        oid = row.oid
        ra = float(row.ra_deg)
        dec = float(row.dec_deg)

        try:
            ebmv_mw, sfd_ok = get_sfd98_extinction_real(ra, dec)
        except Exception as e:
            print(f"  [{i}/{len(pending)}] {oid} → EXCEPTION: {e}")
            ebmv_mw, sfd_ok = 0.02, False

        new_rows.append({
            "oid": oid,
            "ra_deg": ra,
            "dec_deg": dec,
            "ebmv_mw": float(ebmv_mw),
            "sfd_ok": bool(sfd_ok),
            "queried_at": datetime.now().isoformat(timespec="seconds"),
        })

        if sfd_ok:
            n_ok += 1
        else:
            n_fail += 1

        # Progreso cada N queries
        if i % args.save_every == 0 or i == len(pending):
            df_new = pd.DataFrame(new_rows)
            df_cache = pd.concat([df_cache, df_new], ignore_index=True)
            save_cache(df_cache, cache_path)
            new_rows = []
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(pending) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(pending)}] {oid} E(B-V)={ebmv_mw:.4f} ok={sfd_ok} "
                  f"| rate={rate:.2f}/s | ETA={eta/60:.1f} min | cache={len(df_cache):,}")

    elapsed = time.time() - t0
    print()
    print(f"Completado en {elapsed/60:.1f} min")
    print(f"  Exitosos: {n_ok:,} | Fallidos: {n_fail:,}")
    print(f"  Cache total: {len(df_cache):,} OIDs → {cache_path}")


if __name__ == "__main__":
    main()
