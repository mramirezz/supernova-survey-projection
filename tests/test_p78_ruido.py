# tests/test_p78_ruido.py
"""Ruido de tres terminos (fondo, fuente, piso): equivalencia con la regla snr, mismos sorteos, parametros por banda,
calibracion contra las detecciones ZTF val y configs que lo usan."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import pytest
from pipeline78.project import project_one, sigma_tres_terminos, SIGMA_MAX
from pipeline78.runcfg import RUNS_CFG

CFG = dict(bands=["g", "r"], pre_ul_days=25.0, rule="ztf", noise_k=5.0, sigma_floor=0.02)


def _ramp(cfg, b="g", n=6000, lo=-1.0, hi=4.5, ml=20.0, seed=1, pre=0):
    """Plantilla lineal en magnitud: dm = ml - m va de lo a hi en n epocas diarias con limite ml constante.
    pre epocas antes de t0 quedan como filas sin flujo (99)."""
    mjd = np.arange(59000.0 - pre, 59000.0 + n, 1.0)
    t_rel = np.array([0.0, float(n - 1)])
    return project_one(t_rel, {b: np.array([ml - lo, ml - hi])}, {b: (mjd, np.full(mjd.size, ml))}, 59000.0,
                       np.random.default_rng(seed), dict(cfg, bands=[b], pre_ul_days=float(pre)))


def test_formula_reduces_to_snr():
    """A = 1, B = 0: el termino de fondo es 1.0857/S/N exacto. Con C = 0 es la regla snr sin piso, byte a byte por
    project_one con deteccion dura (las filas sin flujo quedan como UL; el tope SIGMA_MAX se prueba con deteccion
    logistica en test_sigma_max_con_deteccion_logistica). Con C = sigma_floor coincide con el clip lejos del cruce
    (piso en cuadratura vs clip)."""
    dm = np.linspace(-3.0, 6.0, 2001)
    snr = 5.0 * 10.0 ** (0.4 * dm)
    assert (sigma_tres_terminos(dm, 1.0, 0.0, 0.0) == 1.0857 / snr).all()
    old = np.clip(1.0857 / snr, 0.02, None)
    new = sigma_tres_terminos(dm, 1.0, 0.0, 0.02)
    far = (1.0857 / snr >= 5 * 0.02) | (1.0857 / snr <= 0.02 / 5)
    assert far.sum() > 1000 and np.abs(new[far] / old[far] - 1).max() < 0.021
    assert (new >= old).all() and np.abs(new / old - 1).max() <= np.sqrt(2) - 1 + 1e-12
    tres = dict(CFG, noise_model="tres_terminos", noise_params=dict(A=1.0, B=0.0, C=0.0))
    a, b = _ramp(dict(CFG, sigma_floor=0.0), pre=30), _ramp(tres, pre=30)
    pd.testing.assert_frame_equal(a, b)
    assert (a.magnitud_modelo == 99).sum() == 30 and (a[a.magnitud_modelo == 99].upperlimit == "T").all()
    assert SIGMA_MAX == 1.0857 / 1e-6


def test_same_draws_and_band_params():
    """Un solo sorteo normal por fila en el mismo orden: (m_obs - m)/sigma es el mismo z con las dos reglas. Los
    parametros por banda se aplican a su banda y el global a todas."""
    tres = dict(CFG, noise_model="tres_terminos", noise_params=dict(A=0.9, B=0.12, C=0.03))
    a, b = _ramp(CFG, seed=7), _ramp(tres, seed=7)
    assert (a.mjd.to_numpy() == b.mjd.to_numpy()).all() and (a.upperlimit == b.upperlimit).all()
    d = a.upperlimit == "F"
    za = (a.magnitud_proyectada[d].astype(float) - a.magnitud_modelo[d]) / a.magerr[d]
    zb = (b.magnitud_proyectada[d].astype(float) - b.magnitud_modelo[d]) / b.magerr[d]
    assert d.sum() > 4000 and np.abs(za - zb).max() < 1e-3
    exp = sigma_tres_terminos(b.maglimit[d].astype(float) - b.magnitud_modelo[d].astype(float), 0.9, 0.12, 0.03)
    assert np.abs(b.magerr[d] - exp).max() < 1e-6
    per = dict(CFG, noise_model="tres_terminos", noise_params={"g": dict(A=0.9, B=0.12, C=0.03),
                                                                "r": dict(A=1.0, B=0.0, C=0.5)})
    g, r = _ramp(per, "g", seed=7), _ramp(per, "r", seed=7)
    pd.testing.assert_frame_equal(g, b)
    assert (r.magerr.dropna() >= 0.5).all()
    with pytest.raises(KeyError):
        _ramp(per, "i")
    with pytest.raises(ValueError):
        _ramp(dict(CFG, noise_model="poisson"))


def test_sigma_max_con_deteccion_logistica(monkeypatch):
    """Filas sin flujo (99, antes de t0) con una logistica muy permisiva (m0 = -2): ninguna se detecta y el frame es el
    de la regla snr con A = 1, B = 0, C = 0, porque las dos topan sigma en 1.0857/1e-6. Control: sin el tope, sigma
    ~1e31 mag deja m_obs ~ m_lim - 0.5 en la mitad de esas filas y la logistica detecta varias."""
    from pipeline78 import project
    det = dict(det_model="logistic", det_m0=-2.0, det_w=0.2)
    snr = dict(CFG, sigma_floor=0.0, **det)
    tres = dict(CFG, noise_model="tres_terminos", noise_params=dict(A=1.0, B=0.0, C=0.0), **det)
    a, b = _ramp(snr, n=50, pre=2000, seed=3), _ramp(tres, n=50, pre=2000, seed=3)
    pd.testing.assert_frame_equal(a, b)
    sin = b[b.magnitud_modelo == 99]
    assert len(sin) == 2000 and (sin.upperlimit == "T").all()
    monkeypatch.setattr(project, "SIGMA_MAX", np.inf)
    c = _ramp(tres, n=50, pre=2000, seed=3)
    assert (c[c.magnitud_modelo == 99].upperlimit == "F").sum() > 100


def test_banda_i_usa_los_parametros_de_r():
    """i no tiene detecciones reales: runcfg le da los parametros de r (no los de g), y project_one en i da el mismo
    frame que en r."""
    p = RUNS_CFG["ztf_v78_t9"]["noise_params"]
    assert p["i"] == p["r"] and p["i"] != p["g"]
    cfg = dict(CFG, noise_model="tres_terminos", noise_params=p)
    i, r = _ramp(cfg, "i", seed=11), _ramp(cfg, "r", seed=11)
    assert (i["filter"] == "i").all()
    pd.testing.assert_frame_equal(i.drop(columns="filter"), r.drop(columns="filter"))


def test_bootstrap_sortea_sne_completas(monkeypatch):
    """Cada sorteo del bootstrap toma SNe enteras con reemplazo: todas las filas de una SN sorteada aparecen el mismo
    numero de veces y el total de SNe sorteadas es el numero de SNe. Un bootstrap por filas no cumple esto."""
    from pipeline78 import calib_ruido as c
    rows = [(f"SN{k:02d}", "g" if j % 2 else "r", k + 0.01 * j, 0.1) for k in range(40) for j in range(k % 5 + 2)]
    d = pd.DataFrame(rows, columns=["oid", "filter", "dm", "sig"])
    seen = []

    def fake(dm, sig, band):
        seen.append(np.asarray(dm).copy())
        return {k: np.zeros(3) for k in ("gr", "g", "r")}
    monkeypatch.setattr(c, "_fit_sets", fake)
    out = c.bootstrap(d, n_boot=25, seed=1)
    assert len(seen) == 25 and out["g"].shape == (25, 3)
    oid_de, size = dict(zip(d.dm, d.oid)), d.groupby("oid").size()
    for dm in seen:
        cnt = pd.Series(dm).value_counts()
        per = pd.DataFrame({"oid": [oid_de[x] for x in cnt.index], "n": cnt.to_numpy()})
        g = per.groupby("oid")
        assert (g.n.nunique() == 1).all()                                   # filas de una SN: mismas copias
        assert (g.size() == size[g.size().index]).all()                     # y todas sus filas
        assert int((g.n.first() * 1).sum()) == 40                           # 40 SNe sorteadas con reemplazo


def test_seleccion_de_val(tmp_path):
    """val_meta: solo origen holdout, split val y excluir False (sin final, sin viejas, sin excluidas).
    val_detections: solo las detecciones (upperlimit F) de esas SNe."""
    from pipeline78 import calib_ruido as c
    meta = pd.DataFrame({"oid": ["v1", "v2", "vx", "f1", "vv", "fv"], "sn_type": ["Ia", "II", "Ia", "Ia", "II", "Ia"],
                         "origen": ["holdout"] * 4 + ["viejas"] * 2,
                         "split": ["val", "val", "val", "final", "val_viejo", "final_viejo"],
                         "excluir": [False, False, True, False, False, False]})
    meta.to_csv(tmp_path / "meta_real_ztf.csv", index=False)
    for cl in ("Ia", "II"):
        o = meta.oid[meta.sn_type == cl].tolist()
        pd.DataFrame({"oid": np.repeat(o, 2), "sn_type": cl, "filter": "r", "mjd": 59000.0,
                      "magnitud_proyectada": 19.0, "magerr": [0.1, np.nan] * len(o),
                      "upperlimit": ["F", "T"] * len(o)}).to_parquet(tmp_path / f"{cl}.parquet")
    assert sorted(c.val_meta(tmp_path).oid) == ["v1", "v2"]
    det = c.val_detections(tmp_path)
    assert sorted(det.oid) == ["v1", "v2"] and det.sig.notna().all()


def _datos_o_skip():
    from pipeline78 import calib_ruido as c
    from pipeline78.paths import RUNS
    if not (RUNS / "real_ztf/meta_real_ztf.csv").exists() or not c.CACHE.exists():
        pytest.skip("sin holdout real o sin cache de ALeRCE")
    return c


def test_cache_de_alerce_es_val():
    """El cache de calib_ruido tiene exactamente las SNe de val_meta, y val_meta no tiene final, viejas ni excluidas."""
    c = _datos_o_skip()
    from pipeline78.paths import RUNS
    meta = pd.read_csv(RUNS / "real_ztf/meta_real_ztf.csv")
    v = c.val_meta()
    assert set(pd.read_csv(c.CACHE, usecols=["oid"]).oid) == set(v.oid)
    assert (v.origen == "holdout").all() and (v.split == "val").all() and not v.excluir.astype(bool).any()
    assert not set(v.oid) & set(meta.oid[(meta.split != "val") | meta.excluir.astype(bool)])


def test_ajuste_reproduce_runcfg():
    """El ajuste por banda sobre las detecciones val (m_lim de ALeRCE, 0 <= dm < 4) da los seis numeros de runcfg,
    escritos a mano con 4 decimales; i usa los de r."""
    c = _datos_o_skip()
    from pipeline78.runcfg import NOISE_TRES_TERMINOS as P
    _, d = c.load("alerce")
    d = d[np.isfinite(d.dm) & (d.dm >= 0) & (d.dm < 4)]
    for b in ("g", "r"):
        s = d[d["filter"] == b]
        p = c.fit_abc(c.bin_table(s.dm, s.sig))
        assert np.allclose(p, [P[b]["A"], P[b]["B"], P[b]["C"]], rtol=0, atol=1e-4), (b, np.round(p, 5))
    assert P["i"] == P["r"]


def test_ztf_v78_unchanged_and_t9_uses_tres_terminos():
    """ztf_v78 y las configs de antes de la T9 no tienen noise_model (regla snr); bordes, det* y t9 lo heredan."""
    for k in ("ztf_v78", "ztf_v78_texp", "ztf_v78_tail", "ztf_v78_fireball"):
        assert "noise_model" not in RUNS_CFG[k] and "noise_params" not in RUNS_CFG[k], k
    t9 = [k for k in RUNS_CFG if k.startswith("ztf_v78_t9")]
    assert len(t9) >= 20
    for k in t9:
        assert RUNS_CFG[k]["noise_model"] == "tres_terminos", k
        assert RUNS_CFG[k]["noise_params"] == RUNS_CFG["ztf_v78_t9_bordes"]["noise_params"], k
    p = RUNS_CFG["ztf_v78_t9"]["noise_params"]
    for b in ("g", "r", "i"):
        q = p if "A" in p else p[b]
        assert set(q) == {"A", "B", "C"} and all(v >= 0 for v in q.values()), b


# Mediana de sigma real por bin de dm = m_lim - m (0 a 4 mag, 0.25), holdout ZTF val sin excluidas, m_lim = diffmaglim
# de ALeRCE: salida de python -m pipeline78.calib_ruido (logfix 2026-10-04: reales sin filas repetidas ni restas
# negativas).
REAL = {"g": [0.2155, 0.1850, 0.1560, 0.1345, 0.1150, 0.1030, 0.0890, 0.0795, 0.0700, 0.0610, 0.0570, 0.0490,
              0.0450, 0.0450, 0.0405, 0.0355],
        "r": [0.2000, 0.1700, 0.1420, 0.1220, 0.1060, 0.0920, 0.0800, 0.0710, 0.0640, 0.0560, 0.0500, 0.0460,
              0.0425, 0.0380, 0.0370, 0.0355]}


def test_calibrated_sigma_within_15pct_of_real():
    """Con los parametros de runcfg, project_one sobre una rampa de dm del modelo (-1 a 5 mag) con deteccion casi
    total: la mediana de magerr en cada bin de dm MEDIDO (como en las reales) queda dentro del 15 % de la real."""
    from pipeline78.calib_ruido import bin_table
    cfg = dict(CFG, noise_model="tres_terminos", noise_params=RUNS_CFG["ztf_v78_t9"]["noise_params"],
               det_model="logistic", det_m0=-10.0, det_w=0.2)
    for b in ("g", "r"):
        x = _ramp(cfg, b, n=40000, lo=-1.0, hi=5.0, seed=5)
        x = x[x.upperlimit == "F"]
        sim = bin_table(x.maglimit.astype(float) - x.magnitud_proyectada.astype(float), x.magerr.astype(float))
        assert (sim.n > 1000).all()
        q = sim.sig_med.to_numpy() / np.array(REAL[b])
        assert np.abs(q - 1).max() < 0.15, (b, np.round(q, 3))
        assert np.abs(q - 1).max() < 0.10, (b, np.round(q, 3))     # hoy: peor bin 6.6 % (g, 3.25-3.5)


def test_real_table_is_current():
    """REAL sale de los datos: si el holdout o el cache de ALeRCE cambian, este test lo avisa."""
    from pipeline78 import calib_ruido as c
    from pipeline78.paths import RUNS
    if not (RUNS / "real_ztf/meta_real_ztf.csv").exists() or not c.CACHE.exists():
        pytest.skip("sin holdout real o sin cache de ALeRCE")
    _, d = c.load("alerce")
    d = d[np.isfinite(d.dm) & (d.dm >= 0) & (d.dm < 4)]
    assert len(d) == 18811 and d.oid.nunique() == 578
    for b in ("g", "r"):
        s = d[d["filter"] == b]
        assert np.allclose(c.bin_table(s.dm, s.sig).sig_med.to_numpy(), REAL[b], atol=1e-6), b
