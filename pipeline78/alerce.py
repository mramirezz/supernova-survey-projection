# pipeline78/alerce.py
"""API de ALeRCE (detecciones y no detecciones por oid) con cache en disco, y emparejamiento de filas con su alerta.
Lo usan obslog_alerce (diffmaglim real del log) y real_to_parquet (restas negativas, isdiffpos < 0). calib_ruido tiene
su propia copia del fetch, con su cache de val ya versionado."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

API = "https://api.alerce.online/ztf/v1/objects/{oid}/{ep}"
DET_COLS = ["oid", "mjd", "fid", "magpsf", "diffmaglim", "isdiffpos"]
ND_COLS = ["oid", "mjd", "fid", "diffmaglim"]
# Precision del cache: mjd a 1e-7 d (9 ms, las exposiciones distan >= 30 s), magnitudes a 1e-4 mag.
_DEC = {"mjd": 7, "magpsf": 4, "diffmaglim": 4}


def _get(oid, ep, tries=5):
    """(oid, lista de dicts) o (oid, None) si falla tras los reintentos (espera creciente)."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(API.format(oid=oid, ep=ep), timeout=90) as r:
                return oid, json.load(r)
        except Exception as e:                                       # red o 5xx: reintento
            if k == tries - 1:
                print(f"  ALeRCE {ep} {oid}: falla tras {tries} intentos ({e})", flush=True)
                return oid, None
            time.sleep(3.0 * (k + 1))


def _frame(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    for c, n in _DEC.items():
        if c in df:
            df[c] = df[c].astype(float).round(n)
    for c in ("fid", "isdiffpos"):
        if c in df:
            df[c] = pd.to_numeric(df[c])
            if df[c].notna().all():
                df[c] = df[c].astype("int64")
    return df


def fetch(oids, cache, ep="detections", workers=4, every=100):
    """Bajar `ep` (detections o non_detections) de cada oid y guardar en `cache` (csv, .gz si el nombre lo dice).
    Reanuda: los oid que ya estan en el cache no se vuelven a pedir. Guarda cada `every` objetos. Un oid sin filas en
    ALeRCE no queda en el cache (se vuelve a pedir si se reanuda). Devuelve (cache completo, oid que fallaron)."""
    cols = DET_COLS if ep == "detections" else ND_COLS
    cache = Path(cache)
    old = pd.read_csv(cache) if cache.exists() else _frame([], cols)
    todo = sorted(set(oids) - set(old.oid))
    parts, failed, rows, t0 = [old] if len(old) else [], [], [], time.time()

    def flush():
        if rows:
            parts.append(_frame(rows, cols))
            rows.clear()
        df = pd.concat(parts, ignore_index=True) if parts else _frame([], cols)
        df = df.sort_values(["oid", "mjd", "fid"], kind="stable")
        df.to_csv(cache, index=False)
        return df

    print(f"ALeRCE {ep}: {len(todo)} oid por bajar ({len(set(oids)) - len(todo)} ya en {cache.name})", flush=True)
    with ThreadPoolExecutor(min(workers, 4)) as ex:                    # 4 conexiones como maximo
        futs = [ex.submit(_get, o, ep) for o in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            oid, js = fu.result()
            if js is None:
                failed.append(oid)
            else:
                rows.extend([oid] + [x.get(c) for c in cols[1:]] for x in js)
            if k % every == 0:
                flush()
                print(f"  {k}/{len(todo)} en {time.time() - t0:.0f} s", flush=True)
    df = flush().reset_index(drop=True)
    print(f"ALeRCE {ep}: {df.oid.nunique()} oid, {len(df)} filas; fallaron {len(failed)} {failed[:10]}", flush=True)
    return df, sorted(failed)


def match(rows, alerts, tol_mjd, tol_mag=None, fid_col="fid", mag_col="magpsf"):
    """Indice posicional en `alerts` de la alerta de cada fila de `rows`, o -1. Misma oid y banda (`fid_col` en las
    dos tablas), |dmjd| <= tol_mjd y, si tol_mag no es None, |dmag| <= tol_mag (rows[mag_col] contra alerts.magpsf).
    Entre varias candidatas gana la de menor dmjd/tol_mjd + dmag/tol_mag: las alertas duplicadas de una misma
    exposicion (campos que se solapan) tienen el mismo mjd y se distinguen por la magnitud."""
    rows = rows.reset_index(drop=True)
    out = np.full(len(rows), -1, dtype=np.int64)
    am, rm = alerts.mjd.to_numpy(float), rows.mjd.to_numpy(float)
    if tol_mag is not None:
        amag, rmag = alerts.magpsf.to_numpy(float), rows[mag_col].to_numpy(float)
    ga = alerts.groupby([alerts.oid.to_numpy(), alerts[fid_col].to_numpy()]).indices
    for key, ir in rows.groupby([rows.oid.to_numpy(), rows[fid_col].to_numpy()]).indices.items():
        ia = ga.get(key)
        if ia is None:
            continue
        ia = ia[np.argsort(am[ia], kind="stable")]
        x = rm[ir]
        lo = np.searchsorted(am[ia], x - tol_mjd, side="left")
        hi = np.searchsorted(am[ia], x + tol_mjd, side="right")
        for j in np.flatnonzero(hi > lo):                          # pocas candidatas por fila (la misma exposicion)
            c = ia[lo[j]:hi[j]]
            cost = np.abs(am[c] - x[j]) / tol_mjd
            if tol_mag is not None:
                dmag = np.abs(amag[c] - rmag[ir[j]])
                ok = dmag <= tol_mag
                if not ok.any():
                    continue
                c, cost = c[ok], cost[ok] + dmag[ok] / tol_mag
            out[ir[j]] = c[np.argmin(cost)]
    return out
