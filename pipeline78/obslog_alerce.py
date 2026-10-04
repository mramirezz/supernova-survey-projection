# pipeline78/obslog_alerce.py
"""Log de observacion ZTF de los campos de la proyeccion con el diffmaglim REAL de ALeRCE en las epocas con deteccion
(decision del controlador 2026-10-04, .superpowers/sdd/2026-10-02-biblioteca-a-tasas/logfix-brief.md).

Problema: en ZTF_observing_log_complete.csv (la base de ztf_obslog_best.parquet) las filas con deteccion del objeto
que da nombre al campo no traen el limite de la imagen: diffmaglim_original vacio, diffmaglim_estimated=True, y el
valor queda recortado a >= m + 0.5. En los 1000 campos de ztf_fields_1000.txt son 625 783 de 1 220 829 filas, el
41 % de ellas en m + 0.5 exacto. Las no detecciones si traen el limite original.
Arreglo: cada fila con deteccion se empareja con su alerta de ALeRCE (misma banda, |dmjd| <= 1e-5 d, |dmag| <= 1e-3,
alerce.match) y toma su diffmaglim, el limite 5 sigma de la imagen diferencia. Una alerta negativa (isdiffpos < 0)
tambien sirve: el limite es de la imagen, no del signo. Una fila sin alerta conserva el valor viejo y queda marcada
(estimado=True). Despues va la misma regla survey._best_per_day (una epoca por dia, campo y banda, maglim maximo).
Salida: RUNS/obslog/ztf_obslog_alerce.parquet, con el esquema de ztf_obslog_best (field, mjd, band, maglim) mas la
columna estimado. survey.load_log no la usa. Tambien deja RUNS/obslog/ztf_obslog_alerce_campos.csv con el resumen
por campo. ztf_obslog_best.parquet no se toca.
Cache de ALeRCE: data/obslog_alerce_cache.csv.gz (versionado; ~6e5 alertas, por eso comprimido).
  python -m pipeline78.obslog_alerce --fetch            # baja lo que falte del cache y construye el log
  python -m pipeline78.obslog_alerce --verify-nd        # ademas compara las no detecciones del log con la API"""
import argparse
import numpy as np
import pandas as pd
from pipeline78 import alerce
from pipeline78.paths import DATA, RUNS, STORE
from pipeline78.survey import _best_per_day

CSV = DATA / "ZTF_observing_log_complete.csv"
FIELDS = STORE / "ztf_fields_1000.txt"
OLD = STORE / "ztf_obslog_best.parquet"
CACHE = DATA / "obslog_alerce_cache.csv.gz"
OUT = RUNS / "obslog/ztf_obslog_alerce.parquet"
ND_CACHE = RUNS / "obslog/alerce_non_detections.csv.gz"
TOL_MJD, TOL_MAG = 1e-5, 1e-3      # el log y la API traen el mjd y magpsf completos: misma exposicion, misma alerta
BAND = {1: "g", 2: "r", 3: "i"}


def read_fields(path=FIELDS):
    return sorted({l.strip() for l in open(path) if l.strip()})


def log_rows(fields, csv=CSV, chunksize=2_000_000):
    """Filas crudas del log de esos campos (por chunks: el csv pesa 355 MB)."""
    cols = ["oid", "mjd", "fid", "magpsf", "diffmaglim", "diffmaglim_estimated", "is_detection"]
    want, out = set(fields), []
    for ch in pd.read_csv(csv, usecols=cols, chunksize=chunksize, dtype={"oid": str}):
        out.append(ch[ch.oid.isin(want)])
    return pd.concat(out, ignore_index=True)


def with_alerce(rows, alerts):
    """maglim nuevo por fila: diffmaglim de la alerta en las detecciones emparejadas, el viejo en el resto.
    Columnas agregadas: maglim, estimado (deteccion sin alerta), ialerta (indice en alerts o -1)."""
    rows = rows.reset_index(drop=True)
    det = rows.is_detection.astype(bool).to_numpy()
    j = np.full(len(rows), -1, dtype=np.int64)
    j[det] = alerce.match(rows[det], alerts, TOL_MJD, TOL_MAG)
    new = rows.diffmaglim.to_numpy(float).copy()
    new[j >= 0] = alerts.diffmaglim.to_numpy(float)[j[j >= 0]]
    return rows.assign(maglim=new, estimado=det & (j < 0), ialerta=j)


def best(rows, col="maglim"):
    """La regla de ztf_obslog_best: survey._best_per_day sobre (field, mjd, band, maglim[, estimado])."""
    extra = ["estimado"] if "estimado" in rows else []
    df = rows.rename(columns={"oid": "field"}).assign(band=rows.fid.map(BAND))
    df = df[["field", "mjd", "band", col] + extra].rename(columns={col: "maglim"})
    return _best_per_day(df)


def compare(old, new):
    """Epoca a epoca (campo, banda, dia): maglim viejo y nuevo. Las dos tablas tienen las mismas epocas."""
    k = ["field", "band", "day"]
    a = old.assign(day=np.floor(old.mjd).astype("int64"))[k + ["maglim"]]
    b = new.assign(day=np.floor(new.mjd).astype("int64"))[k + ["maglim", "estimado"]]
    c = a.merge(b, on=k, how="outer", suffixes=("_viejo", "_nuevo"), indicator=True)
    if (c._merge != "both").any():
        raise RuntimeError(f"epocas distintas entre el log viejo y el nuevo: {c._merge.value_counts().to_dict()}")
    return c.drop(columns="_merge").assign(delta=c.maglim_nuevo - c.maglim_viejo)


def resumen_campos(rows, cmp):
    """Por campo: epocas, epocas que cambian, detecciones, detecciones sin alerta, y sin_alerce (ninguna alerta)."""
    r = rows.assign(det=rows.is_detection.astype(bool))
    g = r.groupby("oid").agg(n_filas=("mjd", "size"), n_det=("det", "sum"), n_det_sin_alerta=("estimado", "sum"))
    e = cmp.assign(cambia=np.abs(cmp.delta) > 1e-6).groupby("field").agg(
        n_epocas=("day", "size"), n_epocas_cambian=("cambia", "sum"), n_epocas_estimadas=("estimado", "sum"))
    out = e.join(g, how="outer").fillna(0).astype(int)
    out["sin_alerce"] = (out.n_det > 0) & (out.n_det_sin_alerta == out.n_det)
    return out.rename_axis("field").reset_index()


def describe(delta):
    d = np.asarray(delta, float)
    q = np.percentile(d, [1, 5, 16, 25, 50, 75, 84, 95, 99])
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    return dict(n=d.size, media=d.mean(), mediana=q[4], std=d.std(), sigma_mad=mad, p01=q[0], p05=q[1], p16=q[2],
                p25=q[3], p75=q[5], p84=q[6], p95=q[7], p99=q[8], frac_abs_gt_0_5=float((np.abs(d) > 0.5).mean()),
                frac_abs_gt_1=float((np.abs(d) > 1.0).mean()), frac_pos=float((d > 0).mean()))


def verify_non_detections(rows, nd):
    """No detecciones del log contra las de la API (misma oid, banda y |dmjd| <= TOL_MJD): cuantas emparejan y cuanto
    difiere el diffmaglim."""
    r = rows[~rows.is_detection.astype(bool)].reset_index(drop=True)
    r = r[r.oid.isin(set(nd.oid))].reset_index(drop=True)
    j = alerce.match(r, nd, TOL_MJD)
    ok = j >= 0
    dl = r.diffmaglim.to_numpy(float)[ok] - nd.diffmaglim.to_numpy(float)[j[ok]]
    used = np.zeros(len(nd), bool)
    used[j[ok]] = True
    nd_in = nd.oid.isin(set(r.oid)).to_numpy()
    return dict(campos=int(r.oid.nunique()), nd_log=len(r), emparejadas=int(ok.sum()),
                api_sin_log=int((nd_in & ~used).sum()), max_abs_dlim=float(np.abs(dl).max()) if ok.any() else np.nan,
                frac_dlim_gt_1e3=float((np.abs(dl) > 1e-3).mean()) if ok.any() else np.nan)


def build(fields=None, fetch=False, verify_nd=False, out=OUT, csv=CSV, cache=CACHE):
    fields = fields or read_fields()
    rows = log_rows(fields, csv)
    print(f"log: {len(rows)} filas de {rows.oid.nunique()} campos; detecciones {int(rows.is_detection.sum())} "
          f"(estimadas {int((rows.is_detection & rows.diffmaglim_estimated).sum())})")
    det = rows[rows.is_detection.astype(bool)]
    print(f"detecciones en m + 0.5 exacto (+-0.002): {float((np.abs(det.diffmaglim - det.magpsf - 0.5) < 0.002).mean()):.3f}")
    if fetch or not cache.exists():
        alerts, failed = alerce.fetch(fields, cache)
    else:
        alerts, failed = pd.read_csv(cache), []
    alerts = alerts[alerts.oid.isin(set(fields))].reset_index(drop=True)
    rows = with_alerce(rows, alerts)
    old_best = best(rows, "diffmaglim").drop(columns="estimado")
    ref = pd.read_parquet(OLD, filters=[("field", "in", fields)]).sort_values(["field", "mjd"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(old_best, ref)          # el log viejo se reconstruye exacto desde el csv
    new_best = best(rows)
    cmp = compare(old_best, new_best)
    camp = resumen_campos(rows, cmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_best.to_parquet(out, index=False)
    camp.to_csv(out.with_name(out.stem + "_campos.csv"), index=False)
    d = rows[rows.is_detection.astype(bool)]
    em = d.ialerta >= 0
    print(f"detecciones emparejadas con su alerta: {int(em.sum())} de {len(d)}; sin alerta {int((~em).sum())} "
          f"(i: {int(((~em) & (d.fid == 3)).sum())} de {int((d.fid == 3).sum())} detecciones en i)")
    neg = alerts.isdiffpos.to_numpy()[d.ialerta[em].to_numpy()] < 0
    print(f"  de ellas con alerta negativa (isdiffpos < 0): {int(neg.sum())}")
    dd = d[em].maglim - d[em].diffmaglim
    print("fila a fila (nuevo - viejo) en las detecciones emparejadas:", {k: round(v, 4) for k, v in describe(dd).items()})
    ch = np.abs(cmp.delta) > 1e-6
    print(f"epocas: {len(cmp)}; cambian {int(ch.sum())} ({ch.mean():.3f}); quedan estimadas {int(cmp.estimado.sum())}")
    print("epocas que cambian, nuevo - viejo:", {k: round(v, 4) for k, v in describe(cmp.delta[ch]).items()})
    print(f"campos sin ninguna alerta en ALeRCE (conservan el valor viejo, sin_alerce=True): {int(camp.sin_alerce.sum())} "
          f"{camp.field[camp.sin_alerce].tolist()[:20]}; campos con alguna deteccion sin alerta: "
          f"{int((camp.n_det_sin_alerta > 0).sum())}; fallaron en la API: {len(failed)} {failed[:20]}")
    res = dict(rows=rows, alerts=alerts, old=old_best, new=new_best, cmp=cmp, campos=camp, failed=failed)
    if verify_nd:
        nd, fnd = alerce.fetch(fields, ND_CACHE, ep="non_detections")
        v = verify_non_detections(rows, nd)
        print("no detecciones log vs API:", v, f"fallaron {len(fnd)}")
        res["verify_nd"] = v
    print(f"-> {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="baja de ALeRCE los campos que falten en el cache")
    ap.add_argument("--verify-nd", action="store_true", help="compara las no detecciones del log con la API")
    a = ap.parse_args()
    build(fetch=a.fetch, verify_nd=a.verify_nd)


if __name__ == "__main__":
    main()
