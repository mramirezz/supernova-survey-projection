"""Tests del clasificador NN (pipeline78/nnclf).

Con pytest (env con torch y pytest):  python -m pytest tests/test_p78_nnclf.py
Sin pytest (env series, torch sin pytest), desde la raiz del repo:
    /opt/anaconda3/envs/series/bin/python tests/test_p78_nnclf.py
En el env projection (pytest sin torch) el modulo se salta.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import pytest
    torch = pytest.importorskip("torch")
except ModuleNotFoundError:
    import torch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pipeline78.nnclf import data as D
from pipeline78.nnclf.models import build_model
from pipeline78.nnclf.train import collate


def _curve(n_det_g=4, n_det_r=5, n_ul_pre=3, t_pre=(-100.0, -40.0, -5.0), ul_post=True, key="x", y=0):
    """Curva sintetica ordenada: UL previos, detecciones g/r intercaladas, un UL despues de la ultima deteccion."""
    t, band, mag, err, ul = [], [], [], [], []
    for k in range(n_ul_pre):
        t.append(1000.0 + t_pre[k]); band.append(1); mag.append(20.5); err.append(np.nan); ul.append(True)
    for k in range(max(n_det_g, n_det_r)):
        for b, n in ((0, n_det_g), (1, n_det_r)):
            if k < n:
                t.append(1000.0 + 3 * k + 0.1 * b); band.append(b); mag.append(18.0 + 0.1 * k)
                err.append(0.05); ul.append(False)
    if ul_post:
        t.append(t[-1] + 30); band.append(1); mag.append(20.0); err.append(np.nan); ul.append(True)
    return D.Curve(key=key, y=y, t=np.array(t), band=np.array(band, np.int8), mag=np.array(mag, np.float32),
                   err=np.array(err, np.float32), ul=np.array(ul), z=0.05, w=1.0, template="T", sn_type="Ia")


def test_class_map():
    assert D.classes() == ("Ia", "II", "Ibc")
    assert [D.class_of(t) for t in ("Ia", "II", "IIb", "Ibc", "IIn")] == ["Ia", "II", "II", "Ibc", None]
    assert D.class_of("IIn", four_classes=True) == "IIn"
    assert D.class_of("IIb", four_classes=True) == "II"
    assert D.class_of("SLSN-I") is None


def test_tokenize_ventana_y_features():
    c = _curve()
    x, dt, g = D.tokenize(c, use_z=True)
    # el UL de -100 d cae fuera de la ventana de 60 d y el UL posterior a la ultima deteccion tambien
    assert len(x) == c.n_det() + 2
    assert x.shape[1] == D.N_FEAT and len(dt) == len(x)
    assert np.all(np.diff(dt) >= 0)
    det = x[:, 3] == 0
    assert dt[det].min() == 0.0                                     # dt desde la primera deteccion
    m_ref = np.median(c.mag[~c.ul])
    assert np.isclose(np.median(x[det, 1]), 0.0, atol=1e-6)         # m - m_ref con m_ref = mediana
    assert np.all(x[~det, 2] == 0) and np.allclose(x[det, 2], 0.5)  # 10 magerr, 0 en UL
    assert np.all(x[:, 4] + x[:, 5] == 1)                           # banda one-hot
    assert np.isclose(g[0], (m_ref - 19) / 2) and np.isclose(g[1], 0.5)
    assert np.isclose(g[2], (m_ref - D.distmod(0.05)[0] + 18) / 2)
    x2, _, g2 = D.tokenize(c, use_magerr=False, use_z=False)
    assert np.all(x2[:, 2] == 0) and len(g2) == 1


def test_tokenize_trunca_y_conserva_ul():
    c = _curve(n_det_g=90, n_det_r=90, n_ul_pre=3, t_pre=(-50.0, -30.0, -5.0))
    x, dt, _ = D.tokenize(c, max_len=32)
    assert len(x) == 32
    assert (x[:, 3] == 1).sum() == 3 and (x[:, 3] == 0).sum() == 29
    assert np.all(np.diff(dt) >= 0)


def test_collate_mascara():
    items = [D.tokenize(_curve(n_det_g=a, n_det_r=b)) for a, b in ((3, 0), (6, 7), (1, 2))]
    x, t, mask, g = collate(items)
    assert x.shape == (3, max(len(i[0]) for i in items), D.N_FEAT)
    assert mask.sum(1).tolist() == [len(i[0]) for i in items]
    assert torch.all(x[~mask] == 0) and torch.all(t[~mask] == 0)
    assert g.shape == (3, 1)


def test_aumento_nunca_deja_menos_de_3():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        ng, nr = rng.integers(0, 30), rng.integers(0, 30)
        if ng + nr < 3:
            continue
        c = _curve(n_det_g=int(ng), n_det_r=int(nr))
        a = D.augment(c, rng, p_thin=1.0, p_ronly=0.5)
        assert D.MIN_DET <= a.n_det() <= c.n_det()
    # con p_ronly = 1 y r suficiente, solo queda r
    a = D.augment(_curve(n_det_g=5, n_det_r=5), rng, p_thin=0.0, p_ronly=1.0)
    assert set(a.band.tolist()) == {1}
    # sin r suficiente no se fuerza solo r
    a = D.augment(_curve(n_det_g=5, n_det_r=2), rng, p_thin=0.0, p_ronly=1.0)
    assert a.n_det((0,)) == 5


def test_degradacion():
    rng = np.random.default_rng(1)
    c = _curve(n_det_g=4, n_det_r=6)
    assert D.degrade(c, 7, ("r",), rng) is None
    assert D.degrade(c, 5, ("r",), rng).n_det() == 5
    d = D.degrade(c, None, ("g", "r"), rng)
    assert d.n_det() == 10
    assert D.degrade(_curve(n_det_g=2, n_det_r=0), None, ("g", "r"), rng) is None


def test_split_por_plantilla():
    pairs = [(f"Ia{k}", "Ia") for k in range(15)] + [(f"II{k}", "II") for k in range(13)] + \
            [(f"IIb{k}", "IIb") for k in range(10)] + [(f"Ibc{k}", "Ibc") for k in range(30)] + \
            [(f"IIn{k}", "IIn") for k in range(10)]
    folds = [D.split_templates(pairs, 5, f) for f in range(5)]
    assert set().union(*folds) == {p for p, _ in pairs}
    assert all(not (folds[i] & folds[j]) for i in range(5) for j in range(i + 1, 5))
    for st in ("Ia", "II", "IIb", "Ibc", "IIn"):
        assert any(p.startswith(st) and p[len(st)].isdigit() for p in folds[0])
    # la 3 clases comparte la particion de sus tipos con la 4 clases
    sin_iin = D.split_templates([p for p in pairs if p[1] != "IIn"], 5, 0)
    assert sin_iin == {p for p in folds[0] if not p.startswith("IIn")}


def _fake_real(tmp):
    meta, rows = [], []
    for k, (st, split, origen, excl) in enumerate([("Ia", "val", "holdout", False), ("Ia", "final", "holdout", False),
                                                   ("II", "val", "holdout", False), ("IIb", "val", "holdout", False),
                                                   ("IIb", "final", "holdout", False), ("Ibc", "val", "holdout", True),
                                                   ("IIn", "val", "holdout", False), ("Ia", "val_viejo", "viejas", False)]):
        oid = f"ZTF{k:02d}{split}"
        meta.append({"oid": oid, "sn_type": st, "z": 0.05, "split": split, "origen": origen, "excluir": excl})
        for j in range(6):
            rows.append({"oid": oid, "sn_type": st, "mjd": 100.0 + j, "filter": "gr"[j % 2],
                         "magnitud_proyectada": 18.0, "magerr": 0.05, "upperlimit": "F", "part_index": 0})
    pd.DataFrame(meta).to_csv(tmp / "meta_real_ztf.csv", index=False)
    r = pd.DataFrame(rows)
    for st, g in r.groupby("sn_type"):
        g.to_parquet(tmp / f"{st}.parquet", index=False)
    return pd.DataFrame(meta)


def test_mitad_final_nunca_se_carga():
    tmp = Path(tempfile.mkdtemp())
    meta = _fake_real(tmp)
    final = set(meta.oid[meta.split != "val"])
    vistos, orig = [], D.pq.read_table

    def espia(path, columns=None, filters=None, **kw):
        oids = next(v for k, op, v in filters if k == "oid")
        vistos.extend(oids)
        t = orig(path, columns=columns, filters=filters, **kw)
        vistos.extend(t.column("oid").to_pylist())
        return t
    D.pq.read_table = espia
    try:
        curves, skipped = D.load_real_val(tmp)
        curves4, _ = D.load_real_val(tmp, four_classes=True)
    finally:
        D.pq.read_table = orig
    assert not set(vistos) & final                      # ni en el filtro ni en lo leido
    assert {c.key for c in curves} == {"ZTF00val", "ZTF02val", "ZTF03val"}   # sin final, excluida, IIn ni viejas
    assert {c.key for c in curves4} == {"ZTF00val", "ZTF02val", "ZTF03val", "ZTF06val"}
    assert {c.key: c.y for c in curves}["ZTF03val"] == 1                    # IIb -> II
    assert skipped == []


def test_forward_modelos():
    torch.manual_seed(0)
    items = [D.tokenize(_curve(n_det_g=a, n_det_r=b), use_z=True) for a, b in ((3, 0), (10, 12), (2, 4))]
    x, t, mask, g = collate(items)
    for kind in ("gru", "transformer"):
        for n_cls in (3, 4):
            m = build_model(kind, D.N_FEAT, D.n_glob(True), n_cls)
            out = m(x, t, mask, g)
            assert out.shape == (3, n_cls) and torch.isfinite(out).all()
            out.sum().backward()
            m.eval()
            n0 = int(mask[0].sum())
            with torch.no_grad():   # el padding no cambia la prediccion de una curva
                solo = m(x[:1, :n0], t[:1, :n0], mask[:1, :n0], g[:1])
                assert torch.allclose(solo, m(x, t, mask, g)[:1], atol=1e-5)


def test_distmod():
    assert abs(D.distmod(0.1)[0] - 38.31) < 0.02     # D_L(0.1) = 460 Mpc con H0 70, Om 0.3


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
