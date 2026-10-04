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
    x, dt, g, b = D.tokenize(c, use_z=True)
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
    assert np.array_equal(x[:, 5] == 1, b == 1) and b.dtype == np.int64
    assert np.isclose(g[0], (m_ref - 19) / 2) and np.isclose(g[1], 0.5)
    assert np.isclose(g[2], (m_ref - D.distmod(0.05)[0] + 18) / 2)
    x2, _, g2, _ = D.tokenize(c, use_magerr=False, use_z=False)
    assert np.all(x2[:, 2] == 0) and len(g2) == 1
    # banda por lambda pivote: un solo escalar en la columna 4, el resto igual
    xl, dtl, gl, bl = D.tokenize(c, use_z=True, band_enc="lambda")
    assert xl.shape[1] == D.n_feat("lambda") == 5
    assert np.allclose(xl[:, :4], x[:, :4]) and np.array_equal(bl, b) and np.allclose(gl, g)
    assert np.allclose(xl[:, 4], np.where(b == 0, D.LAMBDA_TOKEN[0], D.LAMBDA_TOKEN[1]))


def test_lambda_pivote_de_las_curvas():
    # lambda pivote de data/filters, no de tabla. El brief anotaba g ~ 4770 y r ~ 6420 A.
    assert 4740 < D.LAMBDA_PIVOT["g"] < 4830 and 6380 < D.LAMBDA_PIVOT["r"] < 6460
    assert D.LAMBDA_TOKEN[0] < 0 < D.LAMBDA_TOKEN[1]
    lam = np.linspace(4000, 6000, 2001)
    T = np.exp(-0.5 * ((lam - 5000) / 200) ** 2)
    f = Path(tempfile.mkdtemp()) / "filtro.dat"
    np.savetxt(f, np.c_[lam, T])
    assert abs(D.pivot_wavelength(f) - 5000) < 10          # gaussiana simetrica: pivote ~ centro


def test_tokenize_trunca_y_conserva_ul():
    c = _curve(n_det_g=90, n_det_r=90, n_ul_pre=3, t_pre=(-50.0, -30.0, -5.0))
    x, dt, _, _ = D.tokenize(c, max_len=32)
    assert len(x) == 32
    assert (x[:, 3] == 1).sum() == 3 and (x[:, 3] == 0).sum() == 29
    assert np.all(np.diff(dt) >= 0)


def test_collate_mascara():
    items = [D.tokenize(_curve(n_det_g=a, n_det_r=b)) for a, b in ((3, 0), (6, 7), (1, 2))]
    x, t, mask, g, b = collate(items)
    assert x.shape == (3, max(len(i[0]) for i in items), D.N_FEAT)
    assert mask.sum(1).tolist() == [len(i[0]) for i in items]
    assert torch.all(x[~mask] == 0) and torch.all(t[~mask] == 0) and torch.all(b[~mask] == 0)
    assert g.shape == (3, 1) and b.dtype == torch.long
    xl = collate([D.tokenize(_curve(), band_enc="lambda")])[0]
    assert xl.shape[-1] == 5


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
    # con truncamiento ORACLE-2 tampoco
    for _ in range(3000):
        ng, nr = rng.integers(0, 30), rng.integers(0, 30)
        if ng + nr < 3:
            continue
        c = _curve(n_det_g=int(ng), n_det_r=int(nr))
        a = D.augment(c, rng, p_thin=0.5, p_ronly=0.5, trunc=["pow2", "frac", "both"][_ % 3], p_trunc=1.0)
        assert D.MIN_DET <= a.n_det() <= c.n_det()


def test_truncamiento_oracle2():
    rng = np.random.default_rng(3)
    c = _curve(n_det_g=40, n_det_r=40, n_ul_pre=3, t_pre=(-50.0, -30.0, -5.0), ul_post=True)
    td = c.t[~c.ul]
    cortes = []
    for _ in range(500):
        a = D.truncate(c, rng, "pow2")
        ta = a.t[~a.ul]
        assert len(ta) >= D.MIN_DET and np.array_equal(ta, td[:len(ta)])     # prefijo: se corta por tiempo
        assert a.ul[a.t < td[0]].all() and (a.t < td[0]).sum() == 3            # los UL previos se conservan
        cortes.append(a.t.max() - td[0])
    cortes = np.array(cortes)
    assert cortes.max() <= 2 ** 10 and np.mean(cortes < 32) > 0.4               # sesgo a fases tempranas
    for _ in range(500):
        a = D.truncate(c, rng, "frac")
        assert 0.1 * len(td) - 1 <= a.n_det() <= len(td) and a.n_det() >= D.MIN_DET
    assert D.truncate(c, rng, "none") is c
    corta = _curve(n_det_g=2, n_det_r=1)                                        # con <= 3 det no se corta
    assert D.truncate(corta, rng, "pow2").n_det() == 3


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
    variantes = [("gru", {}), ("gru", {"gru_pool": "attn"}), ("gru", {"bidir": True}),
                 ("gru", {"bidir": True, "gru_pool": "attn"}), ("gru", {"time_enc": "atat"}),
                 ("transformer", {}), ("transformer", {"time_enc": "atat"})]
    for band_enc in ("onehot", "lambda"):
        items = [D.tokenize(_curve(n_det_g=a, n_det_r=b), use_z=True, band_enc=band_enc)
                 for a, b in ((3, 0), (10, 12), (2, 4))]
        x, t, mask, g, bb = collate(items)
        for kind, opts in variantes:
            for n_cls in (3, 4):
                m = build_model(kind, D.n_feat(band_enc), D.n_glob(True), n_cls, **opts)
                out = m(x, t, mask, g, bb)
                assert out.shape == (3, n_cls) and torch.isfinite(out).all(), (kind, opts)
                out.sum().backward()
                m.eval()
                n0 = int(mask[0].sum())
                with torch.no_grad():   # el padding no cambia la prediccion de una curva
                    solo = m(x[:1, :n0], t[:1, :n0], mask[:1, :n0], g[:1], bb[:1, :n0])
                    assert torch.allclose(solo, m(x, t, mask, g, bb)[:1], atol=1e-4), (kind, opts, band_enc)


def test_time_modulator_atat():
    from pipeline78.nnclf.models import TimeModulator
    torch.manual_seed(1)
    tm = TimeModulator(8, n_bands=2, harmonics=4, t_max=100.0)
    e = torch.randn(1, 3, 8)
    t = torch.tensor([[0.0, 10.0, 10.0]])
    b = torch.tensor([[0, 0, 1]])
    out = tm(e, t, b)
    # ec. 1 y 2 de ATAT a mano para el token 2 (banda 1, t = 10)
    h = torch.arange(4.0)
    s, c = torch.sin(2 * np.pi * h * 10 / 100), torch.cos(2 * np.pi * h * 10 / 100)
    g1 = s @ tm.alpha_sin[1] + c @ tm.alpha_cos[1]
    g2 = s @ tm.beta_sin[1] + c @ tm.beta_cos[1]
    assert torch.allclose(out[0, 2], e[0, 2] * g1 + g2, atol=1e-5)
    # con t = 0 los senos se anulan y todos los cosenos valen 1 (el codigo oficial incluye h = 0)
    assert torch.allclose(out[0, 0], e[0, 0] * tm.alpha_cos[0].sum(0) + tm.beta_cos[0].sum(0), atol=1e-4)
    # misma t, otra banda: otra modulacion
    assert not torch.allclose(out[0, 1], e[0, 1] * g1 + g2)


def _curvas_reales_falsas(n=30):
    rng = np.random.default_rng(5)
    return [_curve(n_det_g=int(rng.integers(0, 10)), n_det_r=int(rng.integers(2, 12)), key=f"ZTF{k:03d}", y=k % 3)
            for k in range(n)]


def test_degradacion_muestra_fija_y_rng_por_curva():
    from pipeline78.nnclf.evaluate import enumerate_cells, FIXED_MIN
    cs = _curvas_reales_falsas()
    meta, dcs = enumerate_cells(cs, n_draws=3, seed=7, fixed=True)
    assert len(meta) == len(dcs)
    for bands, bid in (("r", (1,)), ("g+r", (0, 1))):
        fx = meta[(meta["mode"] == "fixed") & (meta.bands == bands)]
        esperado = {c.key for c in cs if c.n_det(bid) >= FIXED_MIN}
        for N, g in fx.groupby("N"):        # la misma muestra en las cuatro celdas
            for d, gd in g.groupby("draw"):
                assert set(gd.key) == esperado, (bands, N, d)
        nat = meta[(meta["mode"] == "natural") & (meta.bands == bands) & (meta.N == "3")]
        assert set(nat.key) == {c.key for c in cs if c.n_det(bid) >= 3}
    # los sorteos no dependen del orden de la lista ni del modo (rng por curva)
    meta2, dcs2 = enumerate_cells(cs[::-1], n_draws=3, seed=7, fixed=True)
    def _k(r):
        return r["key"], r["mode"], r["bands"], r["N"], r["draw"]
    a = {_k(r): tuple(dc.t) for r, dc in zip(meta.to_dict("records"), dcs)}
    b = {_k(r): tuple(dc.t) for r, dc in zip(meta2.to_dict("records"), dcs2)}
    assert a == b
    for (k, mode, bands, N, d), t in a.items():
        if mode == "fixed":
            assert a[(k, "natural", bands, N, d)] == t


def test_temperature_scaling_y_ece():
    from pipeline78.nnclf import calib
    rng = np.random.default_rng(0)
    n, K = 3000, 3
    y = rng.integers(0, K, n)
    z = rng.normal(0, 1, (n, K))
    z[np.arange(n), y] += 1.0
    p_ok = np.exp(z) / np.exp(z).sum(1, keepdims=True)              # calibrada por construccion
    yy = np.array([rng.choice(K, p=pi) for pi in p_ok])             # etiquetas sorteadas de p: calibracion perfecta
    p_sobre = calib.apply_temperature(p_ok, 0.3)                     # sobreconfiada
    assert np.array_equal(p_sobre.argmax(1), p_ok.argmax(1))         # T no cambia el argmax
    T = calib.fit_temperature(p_sobre, yy)
    assert 2.8 < T < 3.9                                             # deshace el afilado: T ~ 1 / 0.3
    assert calib.ece(p_sobre, yy) > 2 * calib.ece(calib.apply_temperature(p_sobre, T), yy)
    rep = calib.calibration_report(p_sobre, yy)
    assert rep["ece_ts_cv5"] < rep["ece_raw"] and len(rep["reliability_raw"]) == calib.N_BINS
    # ECE a mano: dos predicciones con confianza 0.9 y 0.6, una acierta
    p = np.array([[0.9, 0.05, 0.05], [0.6, 0.3, 0.1]])
    assert np.isclose(calib.ece(p, np.array([0, 1]), n_bins=10), 0.5 * 0.1 + 0.5 * 0.6)


def test_summarize_oids_villar_y_cobertura():
    from pipeline78.nnclf.evaluate import summarize, summary_row
    cls = ("Ia", "II", "Ibc")
    rows = []
    for k in range(9):
        p = np.eye(3)[k % 3] * 0.8 + 0.2 / 3 if k < 6 else np.eye(3)[(k + 1) % 3] * 0.8 + 0.2 / 3
        rows.append({"key": f"o{k}", "dataset": "real", "mode": "natural", "bands": "g+r", "N": "all", "draw": 0,
                     "y": k % 3, "w": 1.0, "n_det": 10, **{f"p_{c}": p[i] for i, c in enumerate(cls)}})
    tab = pd.DataFrame(rows)
    res, agg = summarize(tab, cls, n_real_total=12, villar_oids=["o0", "o1", "o2", "o7", "zz"])
    assert np.isclose(res["coverage"], 9 / 12) and res["main_real_all_gr"]["acc"] == 6 / 9
    vo = res["villar_oids"]
    assert vo["n_comun"] == 4 and np.isclose(vo["metrics"]["acc"], 3 / 4)
    row = summary_row("x", "nn", res, agg, cls)
    assert row["n_villar_oids"] == 4 and np.isclose(row["coverage"], 0.75)


def test_conversion_supernnova():
    from pipeline78.nnclf import snn
    c = _curve(n_det_g=3, n_det_r=3, n_ul_pre=3, t_pre=(-100.0, -40.0, -5.0))
    snid, mjd, f, fe, flt = snn.lc_arrays(c, "S1")
    idx = D.token_idx(c)
    assert len(mjd) == len(idx) and set(flt) <= {"g", "r"}
    ul = c.ul[idx]
    fl = 10 ** (-0.4 * (c.mag[idx] - 27.5))
    assert np.all(f[ul] == 0) and np.allclose(fe[ul], fl[ul] / 5)                  # UL: flujo 0, sigma = F_lim / 5
    assert np.allclose(f[~ul], fl[~ul]) and np.allclose(fe[~ul], 0.4 * np.log(10) * fl[~ul] * 0.05, rtol=1e-5)
    h = snn.head_row(c, "S1")
    assert h["SNTYPE"] == c.y and np.isclose(h["HOSTGAL_SPECZ"], 0.05)
    assert snn.sntypes_json(D.classes()) == '{"0": "Ia", "1": "II", "2": "Ibc"}'
    tr = [D.Curve(key=str(k), y=k % 3, t=c.t, band=c.band, mag=c.mag, err=c.err, ul=c.ul, w=1.0 + k % 2)
          for k in range(30 + 3)] + [D.Curve(key=f"x{k}", y=0, t=c.t, band=c.band, mag=c.mag, err=c.err, ul=c.ul)
                                     for k in range(10)]
    bal = snn.balanced_train(tr, 3, 0, "subsample")
    assert [sum(b.y == k for b in bal) for k in range(3)] == [11, 11, 11]
    assert len({b.key for b in bal}) == 33 and snn.balanced_train(tr, 3, 0, "none") is tr


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
