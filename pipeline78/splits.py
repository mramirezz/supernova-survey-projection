"""Particion anidada de las reales ZTF de validacion: val_sel (elegir) y val_rep (reportar).

POR QUE (decision del 2026-10-04, revision nn-lit H1). La mitad val del holdout (origen == holdout, split == val,
~excluir) se usa para elegir configuraciones: flags, arquitectura, uso de z, temperatura, priors. Una cifra
reportada sobre la misma muestra con la que se eligio queda optimista. Por eso val se parte en dos:
- val_sel: SOLO para elegir y calibrar (temperatura, priors de clase).
- val_rep: SOLO para reportar. Es la estimacion honesta dentro de val.
La mitad final (split == final) no se toca nunca. La cifra de la tesis sale de ella.

REGLA
1. Entran las filas con origen == holdout, split == val y excluir falso ("true" o "1", sin importar mayusculas, es
   verdadero). El resto de las filas se descarta antes de mirar cualquier otra columna y no influye en el resultado.
2. Clase (CLASS_OF): Ia, II (= II + IIb), Ibc e IIn. Otro sn_type queda fuera. La IIn se parte siempre, asi la variante
   de 4 clases y la de 3 comparten exactamente la particion de Ia, II e Ibc.
3. Por clase: oids ordenadas, permutadas con numpy.random.default_rng([seed, indice de la clase en CLASSES]). Las
   primeras n // 2 van a val_sel y las n - n // 2 restantes a val_rep. La particion de una clase no depende de las
   otras ni del orden de las filas.
4. SEED = 20261004.

API (la usan pipeline78.nnclf y pipeline78.clf_villar)
    from pipeline78 import splits
    meta = splits.read_val_meta(RUNS / "real_ztf/meta_real_ztf.csv")   # solo las filas val llegan a pandas
    val_sel, val_rep = splits.val_split(meta)                          # listas ordenadas de oids (str), disjuntas
`val_split` acepta cualquier tabla con las columnas oid, sn_type, origen, split y excluir (un DataFrame, o lo que
pandas.DataFrame acepte, por ejemplo una lista de dicts de csv.DictReader). Si la tabla trae filas de la mitad final,
las descarta sin usarlas.
"""
import csv
import numpy as np
import pandas as pd

SEED = 20261004
CLASSES = ("Ia", "II", "Ibc", "IIn")
CLASS_OF = {"Ia": "Ia", "II": "II", "IIb": "II", "Ibc": "Ibc", "IIn": "IIn"}
SUBSETS = ("val_sel", "val_rep")


def _true(x):
    return str(x).strip().lower() in ("true", "1")


def _is_val(origen, split, excluir):
    return origen == "holdout" and split == "val" and not _true(excluir)


def val_rows(meta):
    """Filas de la mitad val (regla 1), con la columna cls (regla 2). Las filas de otros splits se descartan primero."""
    m = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)
    keep = [_is_val(o, s, e) for o, s, e in zip(m["origen"], m["split"], m["excluir"])]
    v = m[np.asarray(keep, bool)].copy()
    v["oid"] = v["oid"].astype(str)
    v["cls"] = v["sn_type"].map(CLASS_OF)
    return v[v.cls.notna()].reset_index(drop=True)


def val_split(meta, seed=SEED):
    """(val_sel_oids, val_rep_oids): mitad val partida 50/50 estratificada por clase (reglas 1 a 4)."""
    v = val_rows(meta)
    if v.oid.duplicated().any():
        raise ValueError("oid repetida en la mitad val")
    sel, rep = [], []
    for k, c in enumerate(CLASSES):
        oids = np.array(sorted(v.oid[v.cls == c]), dtype=object)
        perm = oids[np.random.default_rng([int(seed), k]).permutation(len(oids))]
        sel += list(perm[:len(oids) // 2])
        rep += list(perm[len(oids) // 2:])
    return sorted(sel), sorted(rep)


def _typed(df):
    """Tipos como los de pandas.read_csv: vacio -> NaN, columnas numericas a numero, True/False a bool."""
    for c in df.columns:
        s = df[c].replace("", np.nan)
        num = pd.to_numeric(s, errors="coerce")
        if s.notna().any() and num.notna().sum() == s.notna().sum():
            s = num
        elif s.notna().all() and set(s) <= {"True", "False"}:
            s = s == "True"
        df[c] = s
    return df


def read_val_meta(path):
    """meta_real_ztf.csv -> DataFrame con SOLO las filas val (regla 1). Se lee con csv: las filas de otros splits no
    se guardan. Una segunda pasada mira solo la columna oid de las otras filas para verificar que ninguna oid val
    aparezca fuera de val."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if _is_val(r.get("origen"), r.get("split"), r.get("excluir"))]
    vo = {r["oid"] for r in rows}
    if len(vo) != len(rows):
        raise ValueError("oid repetida en la mitad val")
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != "val" and r["oid"] in vo:
                raise ValueError(f"la oid val {r['oid']} aparece tambien en otro split")
    return _typed(pd.DataFrame(rows, columns=None if rows else ["oid", "sn_type", "origen", "split", "excluir"]))
