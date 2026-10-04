"""Tests de pipeline78.splits (particion anidada val_sel / val_rep de las reales ZTF, revision nn-lit H1).

Sin torch. Con pytest (/opt/anaconda3/bin/python -m pytest tests/test_p78_splits.py) o como script en cualquier env
con pandas (series o projection):
    /opt/anaconda3/envs/series/bin/python tests/test_p78_splits.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline78 import splits


def _meta_val(n_por_clase=None, seed=0):
    """meta con filas val de las 5 sn_type, excluidas, viejas y la mitad final."""
    n_por_clase = n_por_clase or {"Ia": 11, "II": 9, "IIb": 4, "Ibc": 7, "IIn": 6}
    rows = []
    for st, n in n_por_clase.items():
        for k in range(n):
            rows.append({"oid": f"ZTF{st}{k:03d}v", "sn_type": st, "z": 0.05, "split": "val", "origen": "holdout",
                         "excluir": False})
            rows.append({"oid": f"ZTF{st}{k:03d}f", "sn_type": st, "z": 0.05, "split": "final", "origen": "holdout",
                         "excluir": False})
    rows.append({"oid": "ZTFexcl", "sn_type": "Ia", "z": 0.05, "split": "val", "origen": "holdout", "excluir": True})
    rows.append({"oid": "ZTFviejo", "sn_type": "Ia", "z": 0.05, "split": "val_viejo", "origen": "viejas",
                 "excluir": False})
    rows.append({"oid": "ZTFslsn", "sn_type": "SLSN-I", "z": 0.05, "split": "val", "origen": "holdout",
                 "excluir": False})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_val_split_estratificado_y_sin_final():
    meta = _meta_val()
    sel, rep = splits.val_split(meta)
    val = {o for o, s, st in zip(meta.oid, meta.split, meta.sn_type) if s == "val" and o not in ("ZTFexcl", "ZTFslsn")}
    assert not set(sel) & set(rep) and set(sel) | set(rep) == val                  # particion de val, sin excluidas
    assert not (set(sel) | set(rep)) & set(meta.oid[meta.split != "val"])          # nada de la final ni viejas
    assert sel == sorted(sel) and rep == sorted(rep)
    cls = {o: splits.CLASS_OF[st] for o, st in zip(meta.oid, meta.sn_type) if st in splits.CLASS_OF}
    for c, n in (("Ia", 11), ("II", 13), ("Ibc", 7), ("IIn", 6)):                   # II = II + IIb
        assert sum(cls[o] == c for o in sel) == n // 2 and sum(cls[o] == c for o in rep) == n - n // 2
    # determinista, no depende del orden de las filas ni de la mitad final
    assert splits.val_split(_meta_val(seed=7)) == (sel, rep)
    otra_final = meta[meta.split != "final"]
    otra_final = pd.concat([otra_final, pd.DataFrame([{"oid": "ZTFnueva", "sn_type": "Ia", "z": 0.1, "split": "final",
                                                       "origen": "holdout", "excluir": False}])])
    assert splits.val_split(otra_final) == (sel, rep)
    # la variante de 3 clases comparte la particion de Ia, II e Ibc
    s3, r3 = splits.val_split(meta[meta.sn_type != "IIn"])
    assert s3 == [o for o in sel if cls[o] != "IIn"] and r3 == [o for o in rep if cls[o] != "IIn"]
    # otra semilla, otra particion
    assert splits.val_split(meta, seed=1) != (sel, rep)
    # acepta filas de csv.DictReader (strings)
    assert splits.val_split(meta.astype(str).to_dict("records")) == (sel, rep)


def test_read_val_meta_solo_val():
    tmp = Path(tempfile.mkdtemp())
    meta = _meta_val()
    meta.to_csv(tmp / "meta.csv", index=False)
    v = splits.read_val_meta(tmp / "meta.csv")
    assert set(v.oid) == {o for o, s, e in zip(meta.oid, meta.split, meta.excluir) if s == "val" and not e}
    assert v.z.dtype.kind == "f" and set(v.split) == {"val"}
    assert splits.val_split(v) == splits.val_split(meta)
    malo = pd.concat([meta, pd.DataFrame([{"oid": v.oid.iloc[0], "sn_type": "Ia", "z": 0.05, "split": "final",
                                           "origen": "holdout", "excluir": False}])])
    malo.to_csv(tmp / "malo.csv", index=False)
    try:
        splits.read_val_meta(tmp / "malo.csv")
        raise AssertionError("no detecto la oid val repetida en la final")
    except ValueError:
        pass


if __name__ == "__main__":
    fallas = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {nombre}")
            except Exception as e:  # noqa: BLE001
                fallas += 1
                print(f"FALLA {nombre}: {type(e).__name__}: {e}")
    sys.exit(1 if fallas else 0)
