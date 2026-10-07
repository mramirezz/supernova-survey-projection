"""Tests de pipeline78.evaluar_test: la mitad final solo se lee con el flag explicito, y solo la final.

Sin torch: /opt/anaconda3/bin/python3 -m pytest tests/test_p78_evaluar_test.py
"""
import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from pipeline78 import evaluar_test as ET
from pipeline78 import splits
from pipeline78.nnclf import data as D


def _fake_real(tmp):
    meta, rows = [], []
    for k, (st, split, origen, excl) in enumerate([("Ia", "val", "holdout", False), ("Ia", "final", "holdout", False),
                                                   ("II", "val", "holdout", False), ("IIb", "final", "holdout", False),
                                                   ("Ibc", "final", "holdout", True), ("IIn", "final", "holdout", False),
                                                   ("Ia", "final_viejo", "viejas", False)]):
        oid = f"ZTF{k:02d}{split}"
        meta.append({"oid": oid, "sn_type": st, "z": 0.05, "split": split, "origen": origen, "excluir": excl,
                     "part_index": 0})
        for j in range(6):
            rows.append({"oid": oid, "sn_type": st, "mjd": 100.0 + j, "filter": "gr"[j % 2],
                         "magnitud_proyectada": 18.0, "magerr": 0.05, "upperlimit": "F", "part_index": 0})
    pd.DataFrame(meta).to_csv(tmp / "meta_real_ztf.csv", index=False)
    for st, g in pd.DataFrame(rows).groupby("sn_type"):
        g.to_parquet(tmp / f"{st}.parquet", index=False)
    return pd.DataFrame(meta)


@pytest.mark.parametrize("cmd", ["villar", "red", "plantillas", "resumen"])
def test_sin_flag_no_lee_nada(tmp_path, monkeypatch, cmd):
    _fake_real(tmp_path)
    abiertos, orig = [], builtins.open

    def espia(f, *a, **k):
        abiertos.append(str(f))
        return orig(f, *a, **k)
    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(D.pq, "read_table", lambda *a, **k: abiertos.append("pq.read_table"))
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: abiertos.append("pd.read_csv"))
    out = tmp_path / "test_final"
    with pytest.raises(SystemExit, match="bloqueada"):
        ET.main([cmd, "--real-dir", str(tmp_path), "--out-root", str(out)])
    assert not [f for f in abiertos if str(tmp_path) in f or f.startswith(("pq.", "pd."))]
    assert not out.exists()


def test_lector_final_exige_autorizacion(tmp_path):
    with pytest.raises(PermissionError):                       # no llega a abrir: el archivo ni existe
        ET.read_final_meta(tmp_path / "no_existe.csv")
    with pytest.raises(PermissionError):
        ET.final_meta(tmp_path, autorizada="si")              # solo True autoriza


def test_con_flag_solo_final(tmp_path, monkeypatch):
    meta = _fake_real(tmp_path)
    no_final = set(meta.oid[(meta.split != "final") | meta.excluir])
    vistos, orig = [], D.pq.read_table

    def espia(path, columns=None, filters=None, **kw):
        vistos.extend(next(v for k, op, v in filters if k == "oid"))
        t = orig(path, columns=columns, filters=filters, **kw)
        vistos.extend(t.column("oid").to_pylist())
        return t
    monkeypatch.setattr(D.pq, "read_table", espia)
    monkeypatch.setattr(ET.pq, "read_table", espia)
    v = ET.final_meta(tmp_path, autorizada=True)
    assert set(v.oid) == {"ZTF01final", "ZTF03final"} and set(v.subset) == {"test"}    # sin excluida, IIn ni viejas
    assert dict(zip(v.oid, v.cls))["ZTF03final"] == "II"                               # IIb -> II
    v4 = ET.final_meta(tmp_path, four=True, autorizada=True)
    assert set(v4.oid) == {"ZTF01final", "ZTF03final", "ZTF05final"}
    curves, sin = ET.load_final_curves(v4, tmp_path, four=True)
    assert {c.key for c in curves} == set(v4.oid) and not sin
    assert vistos and not set(vistos) & no_final
    # los lectores de val siguen sin ver la final
    assert not set(splits.read_val_meta(tmp_path / "meta_real_ztf.csv").oid) & set(v4.oid)
    cv, _ = D.load_real_val(tmp_path, four_classes=True)
    assert {c.key for c in cv} == {"ZTF00val", "ZTF02val"}


def test_oid_final_repetida_en_val(tmp_path):
    meta = _fake_real(tmp_path)
    meta.loc[meta.oid == "ZTF00val", "oid"] = "ZTF01final"
    meta.to_csv(tmp_path / "meta_real_ztf.csv", index=False)
    with pytest.raises(ValueError, match="otro split"):
        ET.read_final_meta(tmp_path / "meta_real_ztf.csv", autorizada=True)


def _preds(oids, y, yp, subset):
    cls = ("Ia", "II", "Ibc")
    p = np.array([[0.8 if c == q else 0.1 for c in cls] for q in yp])
    return pd.DataFrame({"subset": subset, "y_true": y, "y_pred": yp, **{f"p_{c}": p[:, i] for i, c in enumerate(cls)}},
                        index=pd.Index(oids, name="oid"))


def test_sistema_test_perfecto_y_respaldo():
    y = ["Ia", "II", "Ibc"] * 10
    V = {k: _preds([f"v{i}" for i in range(30)], y, y, ["val_sel", "val_rep"] * 15) for k in ("red", "villar")}
    red = _preds([f"t{i}" for i in range(30)], y, y, "test")
    villar = red.iloc[:20].copy()
    villar["y_pred"] = "Ia"                                                # villar falla en lo que cubre
    r = ET.sistema_test({"red": red, "villar": villar}, V)
    m = r["metricas"]
    assert m["red"]["bal_acc"] == 1.0 and m["red"]["L1_argmax"] == 0.0 and m["red"]["cobertura_propia"] == 1.0
    assert m["villar"]["n"] == 30 and abs(m["villar"]["cobertura_propia"] - 20 / 30) < 1e-12
    assert abs(m["villar"]["acc"] - (7 + 10) / 30) < 1e-12              # 7 Ia de las 20 + respaldo perfecto en 10
    assert r["corregidas"]["red"]["L1"] < 1e-12 and r["pares"]["red_vs_villar"]["gana_bal"] == "red"


# ----------------------------------------------------------------------------- 5 clases (regla 7)
def test_cinco_clases_final_y_congelados(tmp_path):
    _fake_real(tmp_path)
    v5 = ET.final_meta(tmp_path, four=5, autorizada=True)
    assert dict(zip(v5.oid, v5.cls)) == {"ZTF01final": "Ia", "ZTF03final": "IIb", "ZTF05final": "IIn"}
    curves, sin = ET.load_final_curves(v5, tmp_path, four=5)
    assert {c.key: D.classes(5)[c.y] for c in curves} == dict(zip(v5.oid, v5.cls)) and not sin
    for b, n in ((False, 3), (True, 4)):                       # la bandera de siempre y el numero dan lo mismo
        pd.testing.assert_frame_equal(ET.final_meta(tmp_path, four=b, autorizada=True),
                                      ET.final_meta(tmp_path, four=n, autorizada=True))
    a = ET.parser().parse_args(["villar", ET.FLAG, "--cinco-clases"])
    assert ET.n_pedido(a) == 5 and ET.congelado("plantillas", 5)[0] == "plantillas_5c"
    assert ET.congelado("red", 5) == ("red_5c", ET.CONGELADOS["red_5c"]) and ET.congelado("red", 3)[0] == "red"
    with pytest.raises(SystemExit, match="congelada de 4"):
        ET.congelado("red", 4)


def _preds5(oids, y, yp, subset):
    cls = D.classes(5)
    p = np.array([[0.8 if c == q else 0.05 for c in cls] for q in yp])
    return pd.DataFrame({"subset": subset, "y_true": y, "y_pred": yp, **{f"p_{c}": p[:, i] for i, c in enumerate(cls)}},
                        index=pd.Index(oids, name="oid"))


def test_sistema_test_cinco_clases():
    cls = D.classes(5)
    y = list(cls) * 8
    V = {k: _preds5([f"v{i}" for i in range(40)], y, y, ["val_sel", "val_rep"] * 20) for k in ("red_5c", "villar_5c")}
    red = _preds5([f"t{i}" for i in range(40)], y, y, "test")
    villar = red.iloc[:25].copy()
    villar["y_pred"] = "IIb"                                    # villar dice IIb en todo lo que cubre
    r = ET.sistema_test({"red_5c": red, "villar_5c": villar}, V, "red_5c", cls)
    m = r["metricas"]
    assert m["red_5c"]["bal_acc"] == 1.0 and m["red_5c"]["L1_argmax"] == 0.0 and set(m["red_5c"]["frac_real"]) == set(cls)
    assert abs(m["villar_5c"]["acc"] - (5 + 15) / 40) < 1e-12         # 5 IIb de las 25 + respaldo perfecto en 15
    assert r["corregidas"]["red_5c"]["L1"] < 1e-12 and r["pares"]["red_5c_vs_villar_5c"]["gana_bal"] == "red_5c"
    from pipeline78 import sistema_tasas as ST
    ms = ET.metricas_sistema(ST.sistema(villar, red, cls), 50, cls)
    assert ms["n"] == 40 and ms["n_real"] == 50 and abs(ms["cobertura"] - 0.8) < 1e-12
    assert np.array(ms["confusion"]).shape == (5, 5) and ms["bal_acc_ic95"][0] <= ms["bal_acc"] <= ms["bal_acc_ic95"][1]
