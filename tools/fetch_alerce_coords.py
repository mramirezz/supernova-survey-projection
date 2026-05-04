"""
FETCH COORDS FROM ALERCE FOR MISSING OIDS
==========================================

Consulta la API pública de Alerce (api.alerce.online) para obtener
meanra/meandec de los OIDs que no están en ztf_coords_master.parquet.
Usa el endpoint batch: GET /objects/?oid=X&oid=Y&... (200 OIDs/request).

Proceso:
  1. Lee obslog → OIDs sin coords en master
  2. Consulta Alerce en batches (resumible)
  3. Guarda parciales en data/alerce_coords_cache.parquet
  4. Al finalizar (o con --merge): fusiona en ztf_coords_master.parquet

Uso:
    python tools/fetch_alerce_coords.py                  # todos los pendientes
    python tools/fetch_alerce_coords.py --limit 1000     # test con 1000 OIDs
    python tools/fetch_alerce_coords.py --merge-only     # sólo fusionar → master
    python tools/fetch_alerce_coords.py --batch-size 200 # OIDs por request (default)
    python tools/fetch_alerce_coords.py --save-every 20  # checkpoints cada 20 batches
"""

import os
import sys
import time
import argparse
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALERCE_BATCH_URL = "https://api.alerce.online/ztf/v1/objects/"
CACHE_FILE = "alerce_coords_cache.parquet"
MASTER_FILE = "ztf_coords_master.parquet"
OBSLOG_FILE = "ZTF_observing_log_complete.csv"


def make_session(retries=3, backoff=1.5):
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_batch(session, oids, timeout=30):
    """
    Consulta un batch de OIDs en una sola request.
    Retorna dict {oid: (ra, dec)} para los encontrados.
    OIDs no en Alerce simplemente no estarán en el dict.
    """
    params = [(("oid", o)) for o in oids]
    params.append(("page_size", len(oids)))
    try:
        r = session.get(ALERCE_BATCH_URL, params=params, timeout=timeout)
        r.raise_for_status()
        items = r.json().get("items", [])
        result = {}
        for it in items:
            oid = it.get("oid")
            ra = it.get("meanra")
            dec = it.get("meandec")
            if oid and ra is not None and dec is not None:
                result[oid] = (float(ra), float(dec))
        return result
    except Exception as e:
        print(f"    [WARN] batch error: {e}")
        return {}


def load_cache(cache_path):
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return pd.DataFrame(columns=["oid", "ra_deg", "dec_deg", "alerce_ok", "queried_at"])


def save_cache(df, cache_path):
    tmp = cache_path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, cache_path)


def merge_into_master(data_dir):
    master_path = os.path.join(data_dir, MASTER_FILE)
    cache_path = os.path.join(data_dir, CACHE_FILE)

    if not os.path.exists(cache_path):
        print("[ERROR] No existe alerce_coords_cache.parquet")
        return

    df_cache = pd.read_parquet(cache_path)
    df_ok = df_cache[df_cache["alerce_ok"]].copy()
    print(f"Cache Alerce: {len(df_cache):,} consultados, {len(df_ok):,} con coords válidas")

    if len(df_ok) == 0:
        print("Nada para fusionar.")
        return

    df_ok = df_ok[["oid", "ra_deg", "dec_deg"]].copy()
    df_ok["source"] = "alerce"

    if os.path.exists(master_path):
        df_master = pd.read_parquet(master_path)
        existing_oids = set(df_master["oid"])
        new_rows = df_ok[~df_ok["oid"].isin(existing_oids)]
        print(f"Master antes: {len(df_master):,} OIDs")
        print(f"Nuevas filas de Alerce: {len(new_rows):,}")
        df_master = pd.concat([df_master, new_rows], ignore_index=True)
    else:
        df_master = df_ok
        print(f"Creando master con {len(df_master):,} OIDs de Alerce")

    out = master_path + ".tmp"
    df_master.to_parquet(out, index=False)
    os.replace(out, master_path)
    print(f"Master actualizado: {len(df_master):,} OIDs → {master_path}")
    print(df_master["source"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=200,
                        help="OIDs por request (default 200; max ~200 antes de URL limit)")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Guardar cache cada N batches (default 20)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Segundos entre batches (default 0.1)")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--data-dir", type=str, default=os.path.join(ROOT, "data"))
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    data_dir = args.data_dir
    cache_path = os.path.join(data_dir, CACHE_FILE)
    master_path = os.path.join(data_dir, MASTER_FILE)
    obslog_path = os.path.join(data_dir, OBSLOG_FILE)

    if args.merge_only:
        merge_into_master(data_dir)
        return

    df_obs = pd.read_csv(obslog_path, usecols=["oid"])
    obslog_oids = set(df_obs["oid"].unique())
    print(f"Obslog: {len(obslog_oids):,} OIDs únicos")

    known_oids = set()
    if os.path.exists(master_path):
        df_master = pd.read_parquet(master_path)
        known_oids = set(df_master["oid"])
        print(f"Master existente: {len(known_oids):,} OIDs con coords")

    df_cache = load_cache(cache_path)
    already_queried = set(df_cache["oid"].tolist()) if len(df_cache) else set()
    print(f"Cache Alerce existente: {len(already_queried):,} OIDs ya consultados")

    pending = sorted(obslog_oids - known_oids - already_queried)
    print(f"OIDs pendientes para Alerce: {len(pending):,}")

    if args.limit:
        pending = pending[: args.limit]
        print(f"  Limitado a {len(pending):,}")

    if not pending:
        print("Nada por consultar.")
        merge_into_master(data_dir)
        return

    bs = args.batch_size
    n_batches = (len(pending) + bs - 1) // bs
    est_s = n_batches * (0.3 + args.delay)
    print(f"Batches: {n_batches:,} × {bs} OIDs | Estimación: ~{est_s/60:.1f} min")
    print()

    session = make_session()
    t0 = time.time()
    new_rows = []
    n_ok = n_fail = 0
    processed = 0

    for b_idx, start in enumerate(range(0, len(pending), bs), start=1):
        batch = pending[start: start + bs]
        found = fetch_batch(session, batch, timeout=args.timeout)

        ts = datetime.now().isoformat(timespec="seconds")
        for oid in batch:
            if oid in found:
                ra, dec = found[oid]
                new_rows.append({"oid": oid, "ra_deg": ra, "dec_deg": dec,
                                 "alerce_ok": True, "queried_at": ts})
                n_ok += 1
            else:
                new_rows.append({"oid": oid, "ra_deg": None, "dec_deg": None,
                                 "alerce_ok": False, "queried_at": ts})
                n_fail += 1

        processed += len(batch)

        if b_idx % args.save_every == 0 or b_idx == n_batches:
            df_new = pd.DataFrame(new_rows)
            df_cache = pd.concat([df_cache, df_new], ignore_index=True)
            save_cache(df_cache, cache_path)
            new_rows = []
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(pending) - processed) / rate if rate > 0 else 0
            print(
                f"  batch {b_idx}/{n_batches} | {processed}/{len(pending)} OIDs "
                f"| ok={n_ok} fail={n_fail} "
                f"| {rate:.0f} OIDs/s | ETA={eta/60:.1f} min"
            )

        time.sleep(args.delay)

    elapsed = time.time() - t0
    print()
    print(f"Completado en {elapsed/60:.1f} min")
    print(f"  Encontrados: {n_ok:,} | Sin datos en Alerce: {n_fail:,}")
    print(f"  Cache Alerce total: {len(df_cache):,}")
    print()
    merge_into_master(data_dir)


if __name__ == "__main__":
    main()
