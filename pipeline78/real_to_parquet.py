"""Fotometria real -> el mismo esquema que la proyeccion, para que real y sintetico pasen por el MISMO
run_parquet.py (misma cascada de reintentos, mismos filtros de calidad, mismo MCMC).
Dos listas: el holdout nuevo (origen=holdout, split val/final) y las 675 viejas (origen=viejas,
split val_viejo/final_viejo). Los upper limits se respetan: upperlimit 'T' con magerr NaN."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import OC, PHD, DATA, ZLF

sys.path.insert(0, str(ZLF))
PHOT = PHD / "paper2_ZTF/Photometry_ZTF_ST_Alerce"


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


def load_meta(holdout=None, oc=OC):
    holdout = Path(holdout) if holdout else DATA / "holdout_ztf_v78.csv"
    h = pd.read_csv(holdout)
    h = pd.DataFrame({"oid": h.oid, "sn_type": h.clase, "subtipo": h.subtipo, "z": h.z, "split": h.split, "origen": "holdout"})
    v = []
    for n, s in (("real_val", "val_viejo"), ("real_final", "final_viejo")):
        d = pd.read_parquet(Path(oc) / f"data/{n}.parquet")
        v.append(pd.DataFrame({"oid": d.sn_name, "sn_type": d.label, "subtipo": d.label, "z": d.z, "split": s, "origen": "viejas"}))
    return pd.concat([h] + v, ignore_index=True)


def ztf_real_to_parquet(out_dir, holdout=None, oc=OC, phot=PHOT):
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
    frames, missing, empty = {}, [], []
    for r in meta.drop_duplicates("oid").itertuples():
        f = find(r.oid)
        if f is None:
            missing.append(r.oid); continue
        fd, _ = parse_photometry_file(str(f))
        rows = photometry_rows(r.oid, r.sn_type, fd)
        if rows is None:
            empty.append(r.oid); continue
        frames.setdefault(r.sn_type, []).append(rows)
    for label, fr in frames.items():
        pd.concat(fr, ignore_index=True).to_parquet(out_dir / f"{label}.parquet", index=False)
    bad = set(missing) | set(empty)
    m = meta[~meta.oid.isin(bad)].assign(part_index=0)
    m.to_csv(out_dir / "meta_real_ztf.csv", index=False)
    print(f"{len(meta)} SNe en las listas, {len(missing)} sin archivo de fotometria {missing[:10]}, {len(empty)} sin g/r {empty[:10]}")
    print(m.groupby(["origen", "split", "sn_type"]).size())
    return m


if __name__ == "__main__":
    ztf_real_to_parquet(Path.home() / "thesis_runs/real_ztf")
