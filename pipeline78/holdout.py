"""Holdout ZTF nuevo desde TNS (decision 2026-10-03), solo con los tipos de las series.

REGLA
1. Fuente: carpetas de paper2_ZTF/Photometry_ZTF_ST_Alerce/<carpeta TNS>/<ZTF>_photometry.dat. Solo entran las
   carpetas de CARPETA_CLASE (Ia, II = II/IIP/IIL, IIb, IIn, Ibc = Ib/Ic/Ic-BL/Ibc). Ninguna otra.
2. z y nombre IAU salen de TNS_ZTF_df_new.csv (name, redshift, internal_names). El ZTF de la fotometria se
   busca dentro de internal_names.
3. Exclusiones, en este orden: (b) las SNe que son plantilla
   (IAU de TNS contra los nombres de catalog.csv sin el prefijo SN), (a) las 675 SNe viejas
   (sn_name de real_val y real_final), (c) z no finita o <= 0,
   (d) sin fila en TNS, (e) duplicados por ZTF (se queda la primera carpeta en orden de CARPETA_CLASE).
4. Tope: a lo mas CAP SNe por clase. Si hay mas, se sortean con numpy.random.default_rng(SEED) sobre la
   lista ordenada por ZTF. Las clases con menos entran completas.
5. Particion 50/50 estratificada por clase en split = val | final, con el mismo rng. La mitad impar va al final.
6. Salida: data/holdout_ztf_v78.csv (oid, tns_name, clase, subtipo, z, split, carpeta).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import PHD, DATA, OC, STORE

CAP = 300
SEED = 20261003
# (carpeta TNS, clase, subtipo). El orden fija la prioridad ante duplicados.
CARPETA_CLASE = [("SN Ia", "Ia", "Ia"),
                 ("SN II", "II", "II"), ("SN IIP", "II", "IIP"), ("SN IIL", "II", "IIL"),
                 ("SN IIb", "IIb", "IIb"), ("SN IIn", "IIn", "IIn"),
                 ("SN Ib", "Ibc", "Ib"), ("SN Ic", "Ibc", "Ic"), ("SN Ic-BL", "Ibc", "Ic-BL"), ("SN Ibc", "Ibc", "Ibc")]
CLASES = ["Ia", "II", "IIb", "IIn", "Ibc"]
# Plantillas que deben quedar fuera (verificacion, 2024ggi no esta en el csv de TNS)
PLANTILLAS_ESPERADAS = ["2021krf", "2021yja", "2023ixf", "2024hpj"]

PHOT = PHD / "paper2_ZTF/Photometry_ZTF_ST_Alerce"
TNS = PHD / "paper2_ZTF/TNS_ZTF_df_new.csv"
OUT = DATA / "holdout_ztf_v78.csv"


def old_names(oc=OC):
    return set(pd.read_parquet(Path(oc) / "data/real_val.parquet").sn_name) | \
           set(pd.read_parquet(Path(oc) / "data/real_final.parquet").sn_name)


def template_iau(store=STORE):
    cat = pd.read_csv(Path(store) / "catalog.csv")
    return {s[2:] if s.startswith("SN") else s for s in cat.sn}


def tns_index(tns=TNS):
    """ZTF -> (IAU, z). Cada ZTF de internal_names apunta a su fila."""
    t = pd.read_csv(tns, usecols=["name", "redshift", "internal_names"])
    idx = {}
    for r in t.itertuples():
        if not isinstance(r.internal_names, str):
            continue
        for n in r.internal_names.split(","):
            n = n.strip()
            if n.startswith("ZTF") and n not in idx:
                idx[n] = (str(r.name), r.redshift)
    return idx


def list_candidates(phot=PHOT):
    """(oid, carpeta, clase, subtipo) de las carpetas mapeadas, sin duplicados (primera carpeta gana)."""
    rows, dup, seen = [], [], {}
    for carpeta, clase, sub in CARPETA_CLASE:
        d = Path(phot) / carpeta
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*_photometry.dat")):
            oid = f.name[:-len("_photometry.dat")]
            if oid in seen:
                dup.append((oid, seen[oid], carpeta))
                continue
            seen[oid] = carpeta
            rows.append((oid, carpeta, clase, sub))
    return pd.DataFrame(rows, columns=["oid", "carpeta", "clase", "subtipo"]), dup


def build_holdout(phot=PHOT, tns=TNS, oc=OC, store=STORE, cap=CAP, seed=SEED):
    cand, dup = list_candidates(phot)
    idx = tns_index(tns)
    viejas, plant = old_names(oc), template_iau(store)
    excl = {"viejas": [], "plantilla": [], "z_invalido": [], "sin_tns": [], "duplicado": dup}
    keep = []
    for r in cand.itertuples():
        if r.oid not in idx:
            excl["sin_tns"].append(r.oid); continue
        iau, z = idx[r.oid]
        if iau in plant:                      # primero la plantilla, asi se reporta aunque tambien sea de las 675
            excl["plantilla"].append((r.oid, iau)); continue
        if r.oid in viejas:
            excl["viejas"].append(r.oid); continue
        if not np.isfinite(z) or z <= 0:
            excl["z_invalido"].append(r.oid); continue
        keep.append((r.oid, iau, r.clase, r.subtipo, float(z), r.carpeta))
    df = pd.DataFrame(keep, columns=["oid", "tns_name", "clase", "subtipo", "z", "carpeta"])
    rng = np.random.default_rng(seed)
    out = []
    for c in CLASES:
        g = df[df.clase == c].sort_values("oid").reset_index(drop=True)
        if len(g) > cap:
            g = g.iloc[np.sort(rng.choice(len(g), cap, replace=False))].reset_index(drop=True)
        n = len(g)
        perm = rng.permutation(n)
        split = np.array(["final"] * n, dtype=object)
        split[perm[:n // 2]] = "val"
        out.append(g.assign(split=split))
    res = pd.concat(out, ignore_index=True)[["oid", "tns_name", "clase", "subtipo", "z", "split", "carpeta"]]
    return res, excl


def report(res, excl):
    print("Por clase y split:\n", pd.crosstab(res.clase, res.split, margins=True))
    print("Por clase y subtipo:\n", pd.crosstab(res.clase, res.subtipo, margins=True))
    for k, v in excl.items():
        print(f"excluidas por {k}: {len(v)}", v if k in ("plantilla", "duplicado") else v[:10])


def main():
    res, excl = build_holdout()
    report(res, excl)
    missing = [p for p in PLANTILLAS_ESPERADAS if p not in {i for _, i in excl["plantilla"]}]
    print("plantillas esperadas no excluidas aqui (ausentes de las carpetas o de TNS):", missing)
    res.to_csv(OUT, index=False)
    print("escrito", OUT, len(res))


if __name__ == "__main__":
    main()
