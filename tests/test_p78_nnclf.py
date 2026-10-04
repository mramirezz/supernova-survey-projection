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
    assert rep["ece_ts_cv5_val"] < rep["by_subset"]["val"]["ece_raw"] and len(rep["reliability_raw"]) == calib.N_BINS
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
    assert row["n_villar_oids_val"] == 4 and np.isclose(row["coverage_val"], 0.75)
    assert "bal_acc_rep" not in row                                         # sin particion no hay _rep


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


# ---------------------------------------------------------------- particion anidada (revision H1)
def test_real_val_meta_via_splits():
    # data.real_val_meta lee meta_real_ztf.csv con splits.read_val_meta (solo las filas val llegan a pandas, H10)
    tmp = Path(tempfile.mkdtemp())
    _fake_real(tmp)
    v = D.real_val_meta(tmp)
    assert set(v.oid) == {"ZTF00val", "ZTF02val", "ZTF03val"} and set(v.split) == {"val"}
    assert set(D.real_val_meta(tmp, four_classes=True).oid) == {"ZTF00val", "ZTF02val", "ZTF03val", "ZTF06val"}


def test_bootstrap_pareado():
    from sklearn.metrics import balanced_accuracy_score
    from pipeline78.nnclf.experimentos import paired_bootstrap
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [80, 120, 50])
    ok = rng.random(len(y)) < 0.6
    d, p, ci = paired_bootstrap(y, ok, ok)
    assert d == 0 and p == 0.0                                       # empate: no gana (se queda la incumbente)
    mejor = ok | (rng.random(len(y)) < 0.4)                          # acierta todo lo de ok y mas
    d, p, ci = paired_bootstrap(y, mejor, ok)
    yhat_m, yhat_o = np.where(mejor, y, (y + 1) % 3), np.where(ok, y, (y + 1) % 3)
    assert np.isclose(d, balanced_accuracy_score(y, yhat_m) - balanced_accuracy_score(y, yhat_o))
    assert p > 0.99 and ci[0] > 0
    ruido = rng.random(len(y)) < 0.6                                 # misma tasa, otra suerte
    assert paired_bootstrap(y, ruido, ok)[1] < 0.9


def _fake_run(root, name, preds, n_params=1000, cfg=None):
    """Corrida falsa para la cola: pred_real_val.csv (oid, y_true, y_pred) y config.json."""
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(d / "pred_real_val.csv", index=False)
    (d / "metrics.json").write_text("{}")
    base = {"model": "gru", "use_z": False, "band_enc": "onehot", "time_enc": "sin", "gru_pool": "last",
            "trunc": "none", "p_trunc": 1.0, "bidir": False, "n_params": n_params}
    import json
    (d / "config.json").write_text(json.dumps({**base, **(cfg or {})}))


def _preds(oids, y, acc_sel, acc_rep, sel, seed):
    """Predicciones con exactitud acc_sel en val_sel y acc_rep en val_rep (por clase, exacta)."""
    cls = ("Ia", "II", "Ibc")
    rng = np.random.default_rng(seed)
    yp = []
    for o, k in zip(oids, y):
        a = acc_sel if o in sel else acc_rep
        yp.append(cls[k] if rng.random() < a else cls[(k + 1) % 3])
    return {"oid": oids, "y_true": [cls[k] for k in y], "y_pred": yp}


def test_cola_seleccion_anidada():
    import json
    from pipeline78.nnclf import experimentos as X
    root = Path(tempfile.mkdtemp())
    oids = [f"o{k:03d}" for k in range(600)]
    y = [k % 3 for k in range(600)]
    sel = set(oids[:300])
    P = lambda a_sel, a_rep, s: _preds(oids, y, a_sel, a_rep, sel, s)          # noqa: E731
    # gru: lambda y attnpool ganan en val_sel; trunc05 gana (y a trunc); trunc pierde. En val_rep todo al reves.
    _fake_run(root, "gru_base", P(0.50, 0.90, 1), 1000)
    _fake_run(root, "gru_lambda", P(0.65, 0.40, 2), 1000, {"band_enc": "lambda"})
    _fake_run(root, "gru_attnpool", P(0.66, 0.40, 3), 1200, {"bidir": True, "gru_pool": "attn"})
    _fake_run(root, "gru_trunc", P(0.40, 0.99, 4), 1000, {"trunc": "both"})
    _fake_run(root, "gru_trunc05", P(0.70, 0.40, 5), 1000, {"trunc": "both", "p_trunc": 0.5})
    # tf: solo timemod gana; tf_trunc05 no existe (falta)
    _fake_run(root, "tf_base", P(0.50, 0.50, 6), 5000, {"model": "transformer"})
    _fake_run(root, "tf_lambda", P(0.51, 0.99, 7), 5000, {"model": "transformer", "band_enc": "lambda"})
    _fake_run(root, "tf_timemod", P(0.70, 0.50, 8), 6000, {"model": "transformer", "time_enc": "atat"})
    _fake_run(root, "tf_trunc", P(0.30, 0.99, 9), 5000, {"model": "transformer", "trunc": "both"})
    info = X.fase2_select(root, "", sel)
    assert info["gru"]["pasan"] == ["gru_lambda", "gru_attnpool", "gru_trunc05"]     # un ganador por slot
    assert info["gru"]["comb"] == "gru_comb_lambda_attnpool_trunc05"
    jobs = X.fase2_comb_jobs(info)
    assert len(jobs) == 1 and jobs[0][0] == "gru_comb_lambda_attnpool_trunc05"      # tf no necesita combinacion
    fl = jobs[0][2]
    assert fl[:2] == ["--model", "gru"] and "lambda" in fl and "attn" in fl and fl.count("--trunc") == 1
    assert fl[fl.index("--p-trunc") + 1] == "0.5"
    assert info["tf"]["pasan"] == ["tf_timemod"] and info["tf"]["comb"] is None
    assert info["tf"]["faltan"] == ["tf_trunc05"]
    # la combinacion no le gana a la mejor individual en val_sel (aunque en val_rep sea mucho mejor): queda trunc05
    _fake_run(root, "gru_comb_lambda_attnpool_trunc05", P(0.70, 1.00, 10), 1300,
              {"band_enc": "lambda", "bidir": True, "gru_pool": "attn", "trunc": "both", "p_trunc": 0.5})
    jz = X.fase2_decide(root, "", sel, X.fase2_select(root, "", sel))
    e = json.loads((root / "fase2_eleccion.json").read_text())
    assert e["arquitecturas"]["gru"]["mejor"] == "gru_trunc05" and e["arquitecturas"]["tf"]["mejor"] == "tf_timemod"
    # gru tiene menos parametros: es la incumbente y tf_timemod (0.70) no le gana a gru_trunc05 (0.70)
    assert e["arquitectura"] == "gru" and e["mejor_noz"] == "gru_trunc05"
    assert jz == [("gru_trunc05_z", "nn", ["--model", "gru", "--band-enc", "onehot", "--time-enc", "sin", "--gru-pool",
                                          "last", "--trunc", "both", "--p-trunc", "0.5", "--use-z"])]
    # ahora la combinacion si gana en val_sel
    _fake_run(root, "gru_comb_lambda_attnpool_trunc05", P(0.95, 0.10, 11), 1300,
              {"band_enc": "lambda", "bidir": True, "gru_pool": "attn", "trunc": "both", "p_trunc": 0.5})
    jz = X.fase2_decide(root, "", sel, X.fase2_select(root, "", sel))
    assert jz[0][0] == "gru_comb_lambda_attnpool_trunc05_z"
    # fase 3: z gana en val_sel -> folds y ensemble de la corrida con z
    _fake_run(root, "gru_comb_lambda_attnpool_trunc05_z", P(1.0, 0.0, 12), 1310,
              {"band_enc": "lambda", "bidir": True, "gru_pool": "attn", "trunc": "both", "p_trunc": 0.5, "use_z": True})
    j3 = X.fase3(root, "", sel)
    assert [j[0] for j in j3] == [f"gru_comb_lambda_attnpool_trunc05_z_f{k}" for k in range(1, 5)] + \
        ["gru_comb_lambda_attnpool_trunc05_z_ens5"]
    assert j3[0][2][-2:] == ["--fold", "1"] and "--use-z" in j3[0][2]
    assert j3[-1][2][1:] == ["gru_comb_lambda_attnpool_trunc05_z"] + \
        [f"gru_comb_lambda_attnpool_trunc05_z_f{k}" for k in range(1, 5)]
    # con prefijo smoke_ no se mezcla con las corridas sin prefijo (revision H7)
    try:
        X.fase3(root, "smoke_", sel)
        raise AssertionError("fase 3 con prefijo uso la eleccion sin prefijo")
    except SystemExit:
        pass


def test_horizonte_y_muestra_fija():
    from pipeline78.nnclf.evaluate import enumerate_cells, FIXED_MIN, HORIZONS
    c = _curve(n_det_g=30, n_det_r=30, n_ul_pre=3, t_pre=(-50.0, -30.0, -5.0))     # 3 d entre epocas
    td = c.t[~c.ul].min()
    prev = -1
    for h in HORIZONS + (None,):
        d = D.cut_horizon(c, h, ("g", "r"))
        assert d.n_det() >= prev and (h is None or d.t.max() <= td + h)
        assert (d.ul & (d.t < td)).sum() == 3                                       # UL previos conservados
        prev = d.n_det()
    assert D.cut_horizon(c, 10, ("r",)).band.tolist() == [1] * len(D.cut_horizon(c, 10, ("r",)).t)
    assert D.cut_horizon(_curve(n_det_g=1, n_det_r=1), 10, ("g", "r")) is None
    cs = _curvas_reales_falsas() + [_curve(n_det_g=2, n_det_r=12, key="lenta", y=1)]
    cs[-1].t[~cs[-1].ul] = np.sort(np.r_[1000.0, 1000.5, 1100 + 3 * np.arange(12)])  # 2 det en 10 d y luego nada
    meta, dcs = enumerate_cells(cs, n_draws=2, seed=3, fixed=True)
    for bands, bid in (("r", (1,)), ("g+r", (0, 1))):
        hz = meta[(meta["mode"] == "horizon") & (meta.bands == bands)]
        assert set(hz.N) == {"10d", "20d", "50d", "all"} and set(hz.draw) == {0}
        keys = [set(g.key) for _, g in hz.groupby("N")]
        assert all(k == keys[0] for k in keys)                                      # misma muestra en las 4 celdas
        fija = {c.key for c in cs if c.n_det(bid) >= FIXED_MIN}
        assert keys[0] <= fija and "lenta" not in keys[0]
    assert "horizon" not in set(enumerate_cells(cs, 2, 3, fixed=False)[0]["mode"])


def test_correccion_de_priors():
    from pipeline78.nnclf import calib
    rng = np.random.default_rng(2)
    pi = np.array([0.6, 0.3, 0.1])
    n = 6000
    y = rng.choice(3, n, p=pi)
    mu = np.eye(3) * 1.2
    x = mu[y] + rng.normal(0, 1, (n, 3))
    lik = np.exp(-0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1))
    p_bal = lik / lik.sum(1, keepdims=True)                          # posterior con prior uniforme (entreno balanceado)
    sel = np.arange(n) < n // 2
    rep = calib.calibration_report(p_bal, y, sel, ~sel)
    assert np.allclose(rep["priors_val_sel"], np.bincount(y[sel], minlength=3) / sel.sum())
    b = rep["by_subset"]["val_rep"]
    assert b["n"] == n - n // 2 and rep["fit_on"] == "val_sel"
    assert b["nll_ts_prior"] < b["nll_ts"] and b["acc_ts_prior"] > b["acc_raw"]     # la mezcla de val se recupera
    assert b["bal_acc_ts_prior"] < b["bal_acc_raw"]                                 # y la balanceada baja
    assert 0.8 < rep["T_prior"] < 1.25                                              # el modelo ya estaba calibrado
    q = calib.apply_temperature(p_bal, 1.0, calib.prior_adjustment(pi))
    assert np.allclose(q, p_bal * pi / (p_bal * pi).sum(1, keepdims=True))         # softmax(log p + log pi)


def test_summarize_por_subconjunto():
    from pipeline78.nnclf.evaluate import summarize, summary_row, _subset_col
    cls = ("Ia", "II", "Ibc")
    rows = []
    for k in range(60):
        ok = k < 30 or k % 2 == 0                                    # sel (k < 30) perfecto, rep a medias
        p = np.eye(3)[k % 3 if ok else (k + 1) % 3] * 0.8 + 0.2 / 3
        rows.append({"key": f"o{k}", "dataset": "real", "mode": "natural", "bands": "g+r", "N": "all", "draw": 0,
                     "y": k % 3, "w": 1.0, "n_det": 10, **{f"p_{c}": p[i] for i, c in enumerate(cls)}})
    tab = pd.DataFrame(rows)
    subsets = {"val_sel": {f"o{k}" for k in range(30)}, "val_rep": {f"o{k}" for k in range(30, 62)}}
    res, agg = summarize(tab, cls, 62, subsets=subsets)
    by = res["main_by_subset"]
    assert by["val_sel"]["metrics"]["acc"] == 1.0 and by["val_rep"]["metrics"]["acc"] == 0.5
    assert np.isclose(by["val_rep"]["coverage"], 30 / 32) and by["val"]["metrics"]["acc"] == 0.75
    assert res["calibration"]["fit_on"] == "val_sel" and res["calibration"]["by_subset"]["val_rep"]["n"] == 30
    assert set(agg.subset) == {"val_rep", "val_sel", "val"}                     # sin sims en esta tabla
    row = summary_row("x", "nn", res, agg, cls)
    assert row["acc_sel"] == 1.0 and row["acc_rep"] == 0.5 and row["acc_val"] == 0.75 and "bal_acc_rep" in row
    assert list(_subset_col(["o1", "o40", "zz"], subsets)) == ["val_sel", "val_rep", ""]


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
