"""Fotometria real -> el mismo esquema que la proyeccion, para que real y sintetico pasen por el MISMO
run_parquet.py (misma cascada de reintentos, mismos filtros de calidad, mismo MCMC).
Dos listas: el holdout nuevo (origen=holdout, split val/final) y las 675 viejas (origen=viejas,
split val_viejo/final_viejo). Los upper limits se respetan: upperlimit 'T' con magerr NaN.
Limpieza (Fix G, 2026-10-03): ALeRCE junta todo lo de esa posicion del cielo. Cada SN pasa por clean_lc con t_ref =
descubrimiento de TNS (discoverydate, cruce por el ZTF de internal_names como en holdout.py), o la primera deteccion
si no esta en TNS. revisar_reales.csv (motivo): "<7 det r" o "span>300 sin gap" (span de detecciones > 300 d sin ser IIn).
excluir_reales.csv: primera_det_tardia (la curva no tiene la fase principal) y las exclusiones manuales de
data/excluir_manual_reales.csv (oid, motivo: SNe dudosas que Mauricio reviso a mano). Se marcan con excluir=True y su
motivo en meta pero siguen en los parquets: los consumidores filtran por excluir (realism.prepare)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import OC, PHD, DATA, ZLF
from pipeline78.lcclean import clean_lc
from pipeline78.holdout import TNS

sys.path.insert(0, str(ZLF))
PHOT = PHD / "paper2_ZTF/Photometry_ZTF_ST_Alerce"
EXCLUIR_MANUAL = DATA / "excluir_manual_reales.csv"      # versionado: decision de Mauricio SN por SN


def photometry_rows(sn, label, filters_data, bands=("g", "r")):
    out = []
    for b, d in filters_data.items():
        if b not in bands:
            continue
        up = d["Upperlimit"].to_numpy(bool)
        err = np.where(up, np.nan, d["MAGERR"].astype(float))
        out.append(pd.DataFrame({"oid": sn, "part_index": np.int32(0), "sn_type": label, "mjd": d["MJD"].astype(float).to_numpy(),
                                 "filter": b, "magnitud_proyectada": d["MAG"].astype(float).to_numpy(),
                                 "magerr": err, "upperlimit": np.where(up, "T", "F")}))
    return pd.concat(out, ignore_index=True) if out else None


def tns_disc(tns=TNS):
    """ZTF -> MJD de descubrimiento de TNS. Cada ZTF de internal_names apunta a su fila (la primera gana)."""
    from astropy.time import Time
    t = pd.read_csv(tns, usecols=["discoverydate", "internal_names"]).dropna()
    mjd = Time(t.discoverydate.astype(str).tolist(), format="iso", scale="utc").mjd
    idx = {}
    for names, m in zip(t.internal_names, mjd):
        for n in names.split(","):
            n = n.strip()
            if n.startswith("ZTF") and n not in idx:
                idx[n] = float(m)
    return idx


def load_excluir_manual(path=None):
    """oid -> motivo de las SNe reales excluidas a mano (data/excluir_manual_reales.csv, columnas oid,motivo)."""
    d = pd.read_csv(Path(path) if path else EXCLUIR_MANUAL, dtype=str)
    d = d.assign(oid=d.oid.str.strip(), motivo=d.motivo.fillna("").str.strip())
    if d.oid.duplicated().any():
        raise ValueError(f"oid repetidos en {path or EXCLUIR_MANUAL}: {d.oid[d.oid.duplicated()].tolist()}")
    if (d.motivo == "").any():
        raise ValueError(f"exclusion manual sin motivo: {d.oid[d.motivo == ''].tolist()}")
    return dict(zip(d.oid, d.motivo))


def load_meta(holdout=None, oc=OC):
    holdout = Path(holdout) if holdout else DATA / "holdout_ztf_v78.csv"
    h = pd.read_csv(holdout)
    h = pd.DataFrame({"oid": h.oid, "sn_type": h.clase, "subtipo": h.subtipo, "z": h.z, "split": h.split, "origen": "holdout"})
    v = []
    for n, s in (("real_val", "val_viejo"), ("real_final", "final_viejo")):
        d = pd.read_parquet(Path(oc) / f"data/{n}.parquet")
        v.append(pd.DataFrame({"oid": d.sn_name, "sn_type": d.label, "subtipo": d.label, "z": d.z, "split": s, "origen": "viejas"}))
    return pd.concat([h] + v, ignore_index=True)


def ztf_real_to_parquet(out_dir, holdout=None, oc=OC, phot=PHOT, tns=TNS, excluir_manual=None):
    from reader import parse_photometry_file
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(holdout, oc)
    from pipeline78.holdout import CARPETA_CLASE
    carpetas = [c for c, _, _ in CARPETA_CLASE] + sorted(d.name for d in Path(phot).iterdir() if d.is_dir() and d.name not in {c for c, _, _ in CARPETA_CLASE})

    def find(oid):      # stat directo, sin listar carpetas enormes de Drive
        for c in carpetas:
            f = Path(phot) / c / f"{oid}_photometry.dat"
            if f.exists():
                return f
        return None
    disc = tns_disc(tns)
    frames, missing, empty, sin_tns, res = {}, [], [], [], {}
    for r in meta.drop_duplicates("oid").itertuples():
        f = find(r.oid)
        if f is None:
            missing.append(r.oid); continue
        fd, _ = parse_photometry_file(str(f))
        rows = photometry_rows(r.oid, r.sn_type, fd)
        if rows is None:
            empty.append(r.oid); continue
        t_disc = disc.get(r.oid)
        if t_disc is None:                              # sin TNS: la primera deteccion hace de descubrimiento
            sin_tns.append(r.oid)
            t_disc = float(rows.mjd[rows.upperlimit == "F"].min())
        rows, s = clean_lc(rows, t_disc)
        if not (rows.upperlimit == "F").any():          # nada de la SN en la ventana
            empty.append(r.oid); continue
        res[r.oid] = dict(t_disc=t_disc, **s, n_det_r=int(((rows["filter"] == "r") & (rows.upperlimit == "F")).sum()))
        frames.setdefault(r.sn_type, []).append(rows)
    for label, fr in frames.items():
        pd.concat(fr, ignore_index=True).to_parquet(out_dir / f"{label}.parquet", index=False)
    bad = set(missing) | set(empty)
    m = meta[~meta.oid.isin(bad)].assign(part_index=0)
    m = m.join(pd.DataFrame.from_dict(res, orient="index"), on="oid")
    man = load_excluir_manual(excluir_manual)
    tardia = m.primera_det_tardia.fillna(False).astype(bool)
    manual = m.oid.map(man).fillna("")
    m["excluir"] = tardia | (manual != "")
    m["motivo"] = [" + ".join(x for x in (a, b) if x) for a, b in zip(np.where(tardia, "primera_det_tardia", ""), manual)]
    fuera = sorted(set(man) - set(m.oid))
    m.drop(columns="n_det_r").to_csv(out_dir / "meta_real_ztf.csv", index=False)
    m[m.excluir].drop(columns="n_det_r").to_csv(out_dir / "excluir_reales.csv", index=False)
    pocas, largo = m.n_det_r < 7, (m.span_det_despues > 300) & (m.sn_type != "IIn")
    motivo = np.where(pocas & largo, "<7 det r + span>300 sin gap", np.where(pocas, "<7 det r", "span>300 sin gap"))
    rev = m.rename(columns={"motivo": "motivo_excluir"}).assign(motivo=motivo)[pocas | largo]
    rev.to_csv(out_dir / "revisar_reales.csv", index=False)
    print(f"{len(meta)} SNe en las listas, {len(missing)} sin archivo de fotometria {missing[:10]}, {len(empty)} sin g/r o sin "
          f"detecciones en la ventana {empty[:10]}")
    print(f"{len(sin_tns)} sin fecha de TNS (t_disc = primera deteccion): {sin_tns}")
    print(m.groupby(["origen", "split", "sn_type"]).size())
    print("SNe que pierden filas al limpiar (pierden / total):")
    print(m.assign(pierden=m.n_filas_despues < m.n_filas_antes).groupby(["origen", "sn_type"]).pierden.agg(["sum", "size"]))
    print("Grupos eliminados por regla (antes del pico, chicos tras el pico, no siguen bajando):")
    print(m.groupby(["origen", "sn_type"])[["n_grupos_antes_pico", "n_grupos_chicos", "n_grupos_no_bajan"]].sum())
    print(f"{len(rev)} a revisar ({rev.motivo.value_counts().to_dict()}) -> {out_dir / 'revisar_reales.csv'}")
    print(f"{int(m.excluir.sum())} a excluir (siguen en los parquets): {int(tardia.sum())} primera_det_tardia "
          f"{m.oid[tardia].tolist()}, {int((manual != '').sum())} manuales {m.oid[manual != ''].tolist()}")
    if fuera:
        print(f"WARNING: {len(fuera)} exclusiones manuales que no estan en meta (sin fotometria o fuera de las listas): {fuera}")
    return m


if __name__ == "__main__":
    ztf_real_to_parquet(Path.home() / "thesis_runs/real_ztf")
