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
    project_one (incluidas las filas sin flujo, por el mismo tope SIGMA_MAX). Con C = sigma_floor coincide con el clip
    lejos del cruce (piso en cuadratura vs clip)."""
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
# de ALeRCE: salida de python -m pipeline78.calib_ruido (2026-10-04).
REAL = {"g": [0.2150, 0.1840, 0.1570, 0.1350, 0.1150, 0.1030, 0.0890, 0.0800, 0.0695, 0.0610, 0.0570, 0.0490,
              0.0450, 0.0455, 0.0405, 0.0355],
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
        assert np.abs(q - 1).max() < 0.10, (b, np.round(q, 3))     # hoy: peor bin 8.4 % (g, 3.25-3.5)


def test_real_table_is_current():
    """REAL sale de los datos: si el holdout o el cache de ALeRCE cambian, este test lo avisa."""
    from pipeline78 import calib_ruido as c
    from pipeline78.paths import RUNS
    if not (RUNS / "real_ztf/meta_real_ztf.csv").exists() or not c.CACHE.exists():
        pytest.skip("sin holdout real o sin cache de ALeRCE")
    _, d = c.load("alerce")
    d = d[np.isfinite(d.dm) & (d.dm >= 0) & (d.dm < 4)]
    assert len(d) == 19171 and d.oid.nunique() == 578
    for b in ("g", "r"):
        s = d[d["filter"] == b]
        assert np.allclose(c.bin_table(s.dm, s.sig).sig_med.to_numpy(), REAL[b], atol=1e-6), b
