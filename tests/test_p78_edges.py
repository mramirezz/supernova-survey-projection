# tests/test_p78_edges.py
"""Bordes de plantilla configurables: window/none identico a antes, texp, tail y determinismo entre variantes."""
import sys, pathlib, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import pytest
from pipeline78.project import project_one

CFG = dict(bands=["g", "r", "i"], pre_ul_days=25.0, rule="ztf", noise_k=5.0, sigma_floor=0.02)
TAIL = dict(CFG, edge_post="tail", tail_days=150, tail_fit_days=20, tail_min_slope=0.005)
Z, T_ANCHOR = 0.05, 59000.0


def _ref_project_one(t_rel, mags, epochs, t_anchor, rng, cfg):
    """project_one tal como estaba antes de los bordes configurables (84aa34c)."""
    t = t_anchor + t_rel
    t0, t1 = float(t[0]), float(t[-1])
    frames = []
    for b in cfg["bands"]:
        if b not in mags or b not in epochs:
            continue
        mjd, mlim = epochs[b]
        sel = (mjd >= t0 - cfg["pre_ul_days"]) & (mjd <= t1)
        if not sel.any():
            continue
        mj, ml = mjd[sel], mlim[sel]
        mm = np.interp(mj, t, mags[b])
        mm[mj < t0] = 99.0
        if cfg["rule"] == "ztf":
            snr = cfg["noise_k"] * 10.0 ** (0.4 * (ml - mm))
            sig = np.clip(1.0857 / np.maximum(snr, 1e-6), cfg["sigma_floor"], None)
            mobs = mm + rng.normal(0.0, sig)
            det = mm < ml
            found = det
        else:
            raise ValueError(cfg["rule"])
        frames.append(pd.DataFrame({
            "mjd": mj, "filter": b, "maglimit": ml.astype(np.float32),
            "magnitud_modelo": mm.astype(np.float32),
            "magnitud_proyectada": np.where(det, mobs, ml).astype(np.float32),
            "magerr": np.where(det, sig, np.nan).astype(np.float32),
            "upperlimit": np.where(det, "F", "T"), "detected": det, "found": found}))
    return pd.concat(frames, ignore_index=True) if frames else None


def _inputs():
    tt = np.arange(-15.0, 61.0, 1.0)                     # plantilla diaria en reposo
    t_rel = tt * (1 + Z)
    g = 18.0 + np.where(tt < 0, 0.01 * tt ** 2, 0.03 * tt)  # declina 0.03 mag/d de reposo
    r = 17.6 - 0.004 * tt                                    # sube hasta el final: la cola usa la pendiente minima
    mags = {"g": g, "r": r, "i": g + 0.3}                    # i sin epocas: se salta
    rng0 = np.random.default_rng(0)
    epochs = {}
    for b in ("g", "r"):
        mjd = np.sort(np.arange(58900.0, 59300.0, 1.5) + rng0.uniform(0, 0.3, 267))
        epochs[b] = (mjd, 20.3 + rng0.normal(0, 0.3, mjd.size))
    return t_rel, mags, epochs


def _run(cfg, seed=3, **kw):
    t_rel, mags, epochs = _inputs()
    return project_one(t_rel, mags, epochs, T_ANCHOR, np.random.default_rng(seed), cfg, **kw)


def _t0_t1():
    t_rel = _inputs()[0]
    return T_ANCHOR + t_rel[0], T_ANCHOR + t_rel[-1]


def test_a_default_identical_to_before():
    t_rel, mags, epochs = _inputs()
    ref = _ref_project_one(t_rel, mags, epochs, T_ANCHOR, np.random.default_rng(3), CFG)
    pd.testing.assert_frame_equal(_run(CFG), ref)
    pd.testing.assert_frame_equal(_run(dict(CFG, edge_pre="window", edge_post="none"), t_exp_rel=-30.0, z=Z), ref)
    assert set(ref["filter"]) == {"g", "r"} and (ref.upperlimit == "T").any() and (ref.upperlimit == "F").any()


def test_b_texp():
    t0, t1 = _t0_t1()
    t_rel = _inputs()[0]
    w = _run(CFG)
    cfg = dict(CFG, edge_pre="texp")
    x = _run(cfg, t_exp_rel=t_rel[0] - 8.0)
    t_exp = t0 - 8.0
    assert not ((x.mjd >= t_exp) & (x.mjd < t0)).any()
    pre = x[x.mjd < t_exp]
    assert len(pre) and (pre.upperlimit == "T").all() and (pre.magnitud_modelo == 99.0).all()
    assert (pre.mjd >= t_exp - CFG["pre_ul_days"]).all()
    for b in ("g", "r"):                                   # todas las epocas de [t_exp - 25, t_exp) quedan como UL
        mjd = _inputs()[2][b][0]
        assert (pre["filter"] == b).sum() == ((mjd >= t_exp - 25.0) & (mjd < t_exp)).sum()
    pd.testing.assert_frame_equal(x[x.mjd >= t0].reset_index(drop=True), w[w.mjd >= t0].reset_index(drop=True))
    pd.testing.assert_frame_equal(_run(cfg, t_exp_rel=None), w)
    pd.testing.assert_frame_equal(_run(cfg, t_exp_rel=t_rel[0] + 1.0), w)   # t_exp >= t0: como window


def test_b_fireball():
    t0, t1 = _t0_t1()
    t_rel, mags, epochs = _inputs()
    w = _run(CFG)
    cfg = dict(CFG, edge_pre="fireball")
    t_exp = t0 - 8.0
    epochs = {b: (np.sort(np.append(e[0], t_exp)), np.append(e[1], 20.3)[np.argsort(np.append(e[0], t_exp))])
              for b, e in epochs.items()}                    # una epoca justo en t_exp (log10(0))
    x = project_one(t_rel, mags, epochs, T_ANCHOR, np.random.default_rng(3), cfg, t_exp_rel=t_rel[0] - 8.0)
    for b in ("g", "r"):
        fb = x[(x["filter"] == b) & (x.mjd >= t_exp) & (x.mjd < t0)].sort_values("mjd")
        assert len(fb) >= 4 and fb.mjd.iloc[0] == t_exp
        assert fb.magnitud_modelo.iloc[0] > 50 and fb.upperlimit.iloc[0] == "T"   # en t_exp no hay flujo: UL, sin inf/nan
        fb = fb.iloc[1:]
        mm, mj = fb.magnitud_modelo.to_numpy(float), fb.mjd.to_numpy(float)
        assert (np.diff(mm) < 0).all()                      # sube (la magnitud baja)
        exp = (mags[b][0] - 5.0 * np.log10((mj - t_exp) / (t0 - t_exp))).astype(np.float32).astype(float)
        assert np.abs(mm - exp).max() < 1e-6                # magnitud_modelo es float32
        assert (mm > mags[b][0]).all()                      # siempre mas debil que el primer punto de la plantilla
    pre = x[x.mjd < t_exp]
    assert len(pre) and (pre.upperlimit == "T").all() and (pre.magnitud_modelo == 99.0).all()
    assert (pre.mjd >= t_exp - CFG["pre_ul_days"]).all()
    fb = x[(x.mjd >= t_exp) & (x.mjd < t0)]
    det = fb.magnitud_modelo < fb.maglimit                   # regla de siempre: deteccion o UL segun el limite
    assert (fb.upperlimit == np.where(det, "F", "T")).all() and det.any()
    w = project_one(t_rel, mags, epochs, T_ANCHOR, np.random.default_rng(3), CFG)
    pd.testing.assert_frame_equal(x[x.mjd >= t0].reset_index(drop=True), w[w.mjd >= t0].reset_index(drop=True))
    # con cola: las filas de la variante tail (<= t1 y cola) salen identicas, el ruido de la bola de fuego va al final
    xt = _run(dict(TAIL, edge_pre="fireball"), t_exp_rel=t_rel[0] - 8.0, z=Z)
    tl = _run(TAIL, z=Z)
    pd.testing.assert_frame_equal(xt[xt.mjd >= t0].reset_index(drop=True), tl[tl.mjd >= t0].reset_index(drop=True))


def test_b_fireball_without_texp_is_window():
    t_rel = _inputs()[0]
    w = _run(CFG)
    cfg = dict(CFG, edge_pre="fireball")
    pd.testing.assert_frame_equal(_run(cfg, t_exp_rel=None), w)
    pd.testing.assert_frame_equal(_run(cfg, t_exp_rel=t_rel[0] + 1.0), w)   # t_exp >= t0


def test_c_tail():
    t0, t1 = _t0_t1()
    n = _run(CFG)
    x = _run(TAIL, z=Z)
    tail = x[x.mjd > t1]
    assert len(tail) and tail.mjd.max() <= t1 + 150 * (1 + Z)
    for b, s_exp in (("g", 0.03 / (1 + Z)), ("r", 0.005)):
        tb = tail[tail["filter"] == b].sort_values("mjd")
        mm, mj = tb.magnitud_modelo.to_numpy(float), tb.mjd.to_numpy(float)
        s = np.diff(mm) / np.diff(mj)
        assert (np.diff(mm) > 0).all() and (s >= 0.005 - 1e-4).all(), b
        assert abs(np.median(s) - s_exp) < 1e-3, (b, np.median(s), s_exp)
    det = tail.magnitud_modelo < tail.maglimit
    assert (tail.upperlimit == np.where(det, "F", "T")).all() and det.any() and (~det).any()
    assert tail.loc[~det, "magerr"].isna().all() and (tail.loc[~det, "magnitud_proyectada"] == tail.loc[~det, "maglimit"]).all()
    pd.testing.assert_frame_equal(x[x.mjd <= t1].reset_index(drop=True), n)
    with pytest.raises(ValueError):
        _run(TAIL)                                         # sin z


def test_c_tail_frame_factors():
    """z alto y quiebre de pendiente dentro de la diferencia entre 20 d de reposo y 20 d observados: la pendiente es
    el ajuste sobre exactamente los ultimos tail_fit_days (1+z) observados, y la cola llega a t1 + tail_days (1+z)."""
    z = 0.3
    tt = np.arange(0.0, 61.0, 1.0)                           # reposo; ventana correcta tt >= 40, la erronea tt >= 44.6
    g = 18.0 + np.where(tt < 42.0, 0.0, 0.06 * (tt - 42.0))      # plana y despues 0.06 mag/d de reposo
    t_rel = tt * (1 + z)
    mjd = np.arange(58990.0, 59400.0, 1.0)
    epochs = {"g": (mjd, np.full(mjd.size, 30.0))}          # limite profundo: toda la cola detectada
    cfg = dict(TAIL, bands=["g"])
    x = project_one(t_rel, {"g": g}, epochs, T_ANCHOR, np.random.default_rng(0), cfg, z=z)
    t, t1 = T_ANCHOR + t_rel, T_ANCHOR + t_rel[-1]
    tail = x[x.mjd > t1]
    s = np.polyfit(tail.mjd - t1, tail.magnitud_modelo.astype(float), 1)[0]
    ok = np.polyfit(t[t >= t1 - 20 * (1 + z)] - t1, g[t >= t1 - 20 * (1 + z)], 1)[0]
    malo = np.polyfit(t[t >= t1 - 20] - t1, g[t >= t1 - 20], 1)[0]
    assert abs(s - ok) < 1e-5 and abs(malo - ok) > 1e-3, (s, ok, malo)
    assert t1 + 150 < tail.mjd.max() <= t1 + 150 * (1 + z)
    assert tail.mjd.max() > t1 + 150 * (1 + z) - 1.0         # epocas diarias: llega al final de la cola


def test_bad_edge_raises():
    with pytest.raises(ValueError):
        _run(dict(CFG, edge_pre="explosion"))


LOG = dict(CFG, det_model="logistic", det_m0=0.5, det_w=0.2)


def _flat(mm, cfg, ml=20.0, n=4000, seed=1):
    """Plantilla plana en g, n epocas diarias con limite ml constante."""
    mjd = np.arange(59000.0, 59000.0 + n, 1.0)
    return project_one(np.array([0.0, float(n)]), {"g": np.full(2, mm)}, {"g": (mjd, np.full(n, ml))}, 59000.0,
                       np.random.default_rng(seed), dict(cfg, bands=["g"]))


def test_e_logistic_limit_is_hard():
    """m0 = 0 y w -> 0: corte duro sobre la magnitud MEDIDA. En estas curvas ninguna fila queda a menos de ~3 sigma del
    limite, asi que sale identico al corte duro de siempre. El ruido no cambia (los uniformes van despues)."""
    pd.testing.assert_frame_equal(_run(dict(CFG, det_model="logistic", det_m0=0.0, det_w=1e-9)), _run(CFG))
    t_rel = _inputs()[0]
    for cfg, kw in ((TAIL, dict(z=Z)), (dict(TAIL, edge_pre="fireball"), dict(z=Z, t_exp_rel=t_rel[0] - 8.0))):
        x, h = _run(dict(cfg, det_model="logistic", det_m0=0.0, det_w=1e-9), **kw), _run(cfg, **kw)   # la cola cruza el limite
        assert (x.mjd.to_numpy() == h.mjd.to_numpy()).all()
        both = (x.upperlimit == "F") & (h.upperlimit == "F")
        assert both.sum() > 20 and (x.loc[both, "magnitud_proyectada"] == h.loc[both, "magnitud_proyectada"]).all()
        xd = x[x.upperlimit == "F"]
        assert (xd.magnitud_proyectada <= xd.maglimit).all()
    with pytest.raises(ValueError):
        _run(dict(CFG, det_model="searcheff"))


def test_e_logistic_measured_magnitude():
    """La deteccion usa la magnitud medida (S/N medida): con w -> 0 ninguna deteccion queda mas debil que el limite 5 sigma,
    que el corte duro sobre el modelo si deja. Las filas sin flujo (99) no se detectan ni con un umbral muy permisivo."""
    h = _flat(19.9, CFG)
    x = _flat(19.9, dict(CFG, det_model="logistic", det_m0=0.0, det_w=1e-9))
    hd, xd = h[h.upperlimit == "F"], x[x.upperlimit == "F"]
    assert len(hd) == 4000 and (hd.magnitud_proyectada > hd.maglimit).mean() > 0.25
    assert 0.6 < len(xd) / 4000 < 0.75 and (xd.magnitud_proyectada <= xd.maglimit).all()
    assert (x.loc[xd.index, "magnitud_proyectada"] == h.loc[xd.index, "magnitud_proyectada"]).all()   # mismo ruido
    t_rel = _inputs()[0]
    loose = dict(TAIL, edge_pre="fireball", det_model="logistic", det_m0=-3.0, det_w=0.2)
    y = _run(loose, z=Z, t_exp_rel=t_rel[0] - 8.0)
    assert (y.magnitud_modelo > 50).sum() > 10 and (y[y.magnitud_modelo > 50].upperlimit == "T").all()


def test_e_logistic_half_at_m0():
    """Fraccion detectada en mm = maglim - m0: ~0.5. Mas brillante detecta casi siempre, mas debil casi nunca."""
    mjd = np.arange(59000.0, 63000.0, 1.0)
    t_rel = np.array([0.0, 4000.0])
    for dm, lo, hi in ((0.0, 0.45, 0.53), (-1.0, 0.98, 1.0), (1.0, 0.0, 0.03)):   # el ruido ensancha la curva
        epochs = {"g": (mjd, np.full(mjd.size, 20.0))}
        x = project_one(t_rel, {"g": np.full(2, 19.5 + dm)}, epochs, 59000.0, np.random.default_rng(1), dict(LOG, bands=["g"]))
        x = x[x.magnitud_modelo < 50]
        assert len(x) == 4000 and lo <= (x.upperlimit == "F").mean() <= hi, (dm, (x.upperlimit == "F").mean())
        d = x[x.upperlimit == "F"]
        assert d.magerr.notna().all() and x.loc[x.upperlimit == "T", "magerr"].isna().all()
        assert (x.loc[x.upperlimit == "T", "magnitud_proyectada"] == x.loc[x.upperlimit == "T", "maglimit"]).all()


def test_e_logistic_shared_rows_keep_noise():
    t0, t1 = _t0_t1()
    t_rel = _inputs()[0]
    h, x = _run(CFG), _run(LOG)
    assert len(h) == len(x) and (h.mjd.to_numpy() == x.mjd.to_numpy()).all()
    both = (h.upperlimit == "F") & (x.upperlimit == "F")
    assert both.sum() > 20 and (h.upperlimit != x.upperlimit).any()
    for c in ("magnitud_proyectada", "magerr", "magnitud_modelo", "maglimit"):
        assert (h.loc[both, c] == x.loc[both, c]).all(), c
    # otro m0: mismos uniformes, filas detectadas en ambas con el mismo ruido
    y = _run(dict(LOG, det_m0=0.0))
    both = (y.upperlimit == "F") & (x.upperlimit == "F")
    assert ((y.upperlimit == "F").sum() > (x.upperlimit == "F").sum()) and (y[both] == x[both]).all().all()
    # variantes de borde: las filas comunes conservan el ruido (el uniforme no, va despues de todo el ruido)
    xt = _run(dict(TAIL, **LOG), z=Z)
    xf = _run(dict(TAIL, edge_pre="fireball", **LOG), z=Z, t_exp_rel=t_rel[0] - 8.0)
    ht = _run(TAIL, z=Z)
    for a, b in ((xt[xt.mjd <= t1].reset_index(drop=True), x), (xt, ht),
                 (xf[xf.mjd >= t0].reset_index(drop=True), xt[xt.mjd >= t0].reset_index(drop=True))):
        assert len(a) == len(b) and (a.mjd.to_numpy() == b.mjd.to_numpy()).all()
        both = (a.upperlimit == "F") & (b.upperlimit == "F")
        assert both.sum() > 20 and (a.loc[both, "magnitud_proyectada"] == b.loc[both, "magnitud_proyectada"]).all()


def test_f_no_ul_after_last_and_clean():
    from pipeline78.lcclean import clean_lc
    cfg = dict(TAIL, **LOG)
    a = _run(cfg, z=Z)
    b = _run(dict(cfg, ul_after_last=False), z=Z)
    last = a.loc[a.upperlimit == "F", "mjd"].max()
    assert (a.mjd > last).any() and not (b.mjd > last).any()
    pd.testing.assert_frame_equal(b, a[a.mjd <= last].reset_index(drop=True))
    # I1: con las dos llaves, el recorte de UL va despues de limpiar. Un punto suelto a +158 d sale por la limpieza y no
    # deja UL despues de la ultima deteccion que queda.
    tt = np.arange(-15.0, 161.0, 0.25)
    m = np.where(tt < 40, 18.0 + 0.02 * np.abs(tt), 25.0)
    m[(tt >= 150.0) & (tt <= 152.0)] = 18.0
    mj = np.arange(58950.0, 59200.0, 1.5)
    ep = {"g": (mj, np.full(mj.size, 20.3))}
    one = lambda **k: project_one(tt * (1 + Z), {"g": m}, ep, T_ANCHOR, np.random.default_rng(3), dict(CFG, bands=["g"], **k))
    a2, c2 = one(), one(ul_after_last=False, lc_clean=True)
    lim = clean_lc(a2, a2.loc[a2.upperlimit == "F", "mjd"].min())[0]
    assert (a2[a2.upperlimit == "F"].mjd > T_ANCHOR + 150).any() and not (lim[lim.upperlimit == "F"].mjd > T_ANCHOR + 100).any()
    last = c2.loc[c2.upperlimit == "F", "mjd"].max()
    assert last < T_ANCHOR + 100 and not (c2.mjd > last).any() and (lim.mjd > last).any()
    pd.testing.assert_frame_equal(c2, lim[lim.mjd <= last].reset_index(drop=True))
    c = _run(dict(cfg, ul_after_last=False, lc_clean=True), z=Z)
    assert not (c.mjd > c.loc[c.upperlimit == "F", "mjd"].max()).any()
    mjd = np.arange(58000.0, 59300.0, 1.0)              # epocas 900 d antes: la ventana desde la primera det las saca
    t_rel, mags, _ = _inputs()
    e = {k: (mjd, np.full(mjd.size, 20.5)) for k in ("g", "r")}
    c = project_one(t_rel, mags, e, T_ANCHOR, np.random.default_rng(3), dict(cfg, pre_ul_days=900.0, lc_clean=True), z=Z)
    first = c.loc[c.upperlimit == "F", "mjd"].min()
    assert c.mjd.min() >= first - 50.0 and c.mjd.max() <= first + 400.0


def _ftpl(sn, span, subtype=None):
    """Plantilla falsa de span d de reposo desde la primera epoca (la explosion en las II)."""
    return dict(sn=sn, clf_class="X", subtype=subtype, M_ref=-17.0, ref_band="r", t_peak=10.0, t_Bmax=10.0, dm15_B=1.1,
                time=np.arange(0.0, span + 0.5, 1.0))


def _fake_sim(monkeypatch, cfg, tpls):
    """run.simulate con plantillas falsas: el motor devuelve curvas lineales en g y r sobre el eje de la plantilla."""
    import pipeline78.run as r
    from pipeline78 import engine, sampling
    for k, v in dict(cfg=cfg, seed=5, bands=None, z=sampling.z_sampler(cfg), tpl=tpls).items():
        monkeypatch.setitem(r._W, k, v)
    monkeypatch.setattr(engine, "observed_lightcurves", lambda tpl, z, e, rv, mw, b, dmag=0.0: (
        (tpl["time"] - tpl["t_peak"]) * (1 + z), {"g": 17.0 + 0.01 * tpl["time"], "r": 17.2 + 0.01 * tpl["time"]}))
    return r


SIMCFG = dict(TAIL, z_mode="fixed", z_fixed=Z, anchor="pivot", n_by_class={"Ia": 2, "II": 2})
MJD = np.arange(58000.0, 60000.0, 1.0)
DEEP = {"g": (MJD, np.full(MJD.size, 30.0)), "r": (MJD + 0.1, np.full(MJD.size, 30.0))}   # todo se detecta


def test_g_anchor_uniform(monkeypatch):
    """uniform: determinista por sim, dentro de [lo, hi] del log y distinta entre campos con el mismo k. pivot repite
    la fecha en todos los campos."""
    from pipeline78.project import anchor_time, _span
    from pipeline78.run import sim_rng
    lo, hi = _span(DEEP)
    one = lambda a, f: anchor_time(dict(anchor=a), sim_rng(7, f, "IIb", 0), 0, 2, DEEP)
    fs = [f"F{i}" for i in range(400)]
    u = np.array([one("uniform", f) for f in fs])
    assert (u == [one("uniform", f) for f in fs]).all() and ((u >= lo) & (u <= hi)).all() and len(set(u)) == len(fs)
    assert 0.18 < np.mean(u < lo + (hi - lo) / 4) < 0.32                # no se amontona en el primer trimestre
    assert len({one("pivot", f) for f in fs}) == 1
    r = _fake_sim(monkeypatch, dict(SIMCFG, anchor="uniform"), {"Ia": [_ftpl("A", 80.0)]})
    a = [r.simulate(f, "Ia", 0, DEEP, 0.02) for f in fs[:20]]
    b = [r.simulate(f, "Ia", 0, DEEP, 0.02) for f in fs[:20]]
    ta = [s["t_anchor"] for s, _ in a]
    assert ta == [s["t_anchor"] for s, _ in b] and len(set(ta)) == 20 and all(lo <= x <= hi for x in ta)
    for (_, d0), (_, d1) in zip(a, b):
        pd.testing.assert_frame_equal(d0, d1)


def test_h_pre_ul_alerce():
    """UL solo si hay una deteccion (cualquier banda) en (t, t + 30]: con det en 100, 105 (g) y 140 (r) y UL en 60, 80,
    90, 120 y 150 (g) quedan 80, 90 y 120; 60 y 150 no. Las filas que quedan son las mismas que sin la regla."""
    t0 = 70.0
    tt = np.arange(0.0, 91.0, 1.0)                           # plantilla de 70 a 160
    m = np.full(tt.size, 25.0)
    m[np.isin(tt + t0, (100.0, 105.0, 140.0))] = 18.0
    ep = {"g": (np.array([60.0, 80.0, 90.0, 100.0, 105.0, 120.0, 150.0]), np.full(7, 20.0)),
          "r": (np.array([140.0]), np.full(1, 20.0))}
    cfg = dict(CFG, bands=["g", "r"], pre_ul_days=30.0)
    run = lambda **k: project_one(tt, {"g": m, "r": m}, ep, t0, np.random.default_rng(3), dict(cfg, **k))
    w, x = run(), run(pre_ul_mode="alerce")
    assert sorted(w[w.upperlimit == "F"].mjd) == [100.0, 105.0, 140.0]
    assert sorted(w[w.upperlimit == "T"].mjd) == [60.0, 80.0, 90.0, 120.0, 150.0]
    assert sorted(x[x.upperlimit == "T"].mjd) == [80.0, 90.0, 120.0]
    pd.testing.assert_frame_equal(x, w[~w.mjd.isin([60.0, 150.0])].reset_index(drop=True))
    with pytest.raises(ValueError):
        run(pre_ul_mode="ztf")
    # frontera, det en 200: UL a 30 d exactos (170) queda, a 30.5 d (169.5) sale, a 27 d (173) queda
    tb = np.arange(0.0, 51.0, 1.0)                           # plantilla de 160 a 210
    eb = {"g": (np.array([169.5, 170.0, 173.0, 200.0]), np.full(4, 20.0))}
    y = project_one(tb, {"g": np.where(tb == 40.0, 18.0, 25.0)}, eb, 160.0, np.random.default_rng(3),
                    dict(cfg, bands=["g"], pre_ul_mode="alerce"))
    assert list(y[y.upperlimit == "F"].mjd) == [200.0] and sorted(y[y.upperlimit == "T"].mjd) == [170.0, 173.0]


def _digest(df):
    """md5 de columnas, dtypes y bytes de cada columna (las de texto unidas con |)."""
    h = hashlib.md5()
    for c in df.columns:
        v = df[c].to_numpy()
        h.update(f"{c}:{v.dtype}".encode())
        h.update("|".join(map(str, v)).encode() if v.dtype == object else v.tobytes())
    return h.hexdigest()


PIN_COUNTS, PIN_MD5 = (138, 104, 32), "ada78c83a5cc5035c8acdceff97cd4c4"     # filas, detecciones, filas sin flujo


def test_j_ztf_v78_pinned():
    """ztf_v78 congelada: ancla pivot (no consume rng) + project_one sobre la plantilla sintetica, con la cfg de
    runcfg. El digest se saco con el codigo de 7e17e57 (antes del Fix H) y coincide con el actual."""
    from pipeline78.project import anchor_time
    from pipeline78.runcfg import RUNS_CFG
    t_rel, mags, epochs = _inputs()
    cfg = RUNS_CFG["ztf_v78"]
    rng = np.random.default_rng(20261002)
    ta = anchor_time(cfg, rng, 3, cfg["n_by_class"]["II"], epochs)
    df = project_one(t_rel, mags, epochs, ta, rng, cfg)
    assert (len(df), int((df.upperlimit == "F").sum()), int((df.magnitud_modelo == 99).sum())) == PIN_COUNTS
    assert _digest(df) == PIN_MD5


def test_i_tail_min_span(monkeypatch):
    """tail_min_span {"II": 120}: una II de 80 d no recibe cola, una de 150 d si, una Ia de 80 d si. Sin cola, las
    filas hasta t1 son las mismas que con cola."""
    cfg = dict(SIMCFG, tail_min_span={"II": 120.0})
    for cls, span, cola in (("II", 80.0, False), ("II", 150.0, True), ("Ia", 80.0, True)):
        tp = [_ftpl("P", span, "IIP"), _ftpl("L", span, "IIL")] if cls == "II" else [_ftpl("A", span)]
        r = _fake_sim(monkeypatch, cfg, {cls: tp})
        s, d = r.simulate("F1", cls, 0, DEEP, 0.02)
        t1 = s["t_anchor"] + (span - 10.0) * (1 + Z)
        assert s["status"] == "ok" and (d.mjd > t1).any() == cola, (cls, span)
        r = _fake_sim(monkeypatch, dict(SIMCFG), {cls: tp})       # sin tail_min_span: siempre cola
        s0, d0 = r.simulate("F1", cls, 0, DEEP, 0.02)
        assert s0["t_anchor"] == s["t_anchor"] and (d0.mjd > t1).any()
        pd.testing.assert_frame_equal(d, d0 if cola else d0[d0.mjd <= t1].reset_index(drop=True))


def _real_tpls(cls):
    from pipeline78.paths import STORE
    from pipeline78.store import load_template
    if not (STORE / "catalog.csv").exists():
        pytest.skip("no hay catalog.csv en el store")
    c = pd.read_csv(STORE / "catalog.csv")
    tp = [load_template(p) for p in c[c.clase == cls].sort_values("sn").store_path]
    # t_Bmax sale del catalogo reconstruido; si el store aun no lo tiene, uno de prueba (solo usa la variante texp)
    return [t if t.get("t_Bmax") is not None else dict(t, t_Bmax=t["t_peak"] - 2.0) for t in tp]


def test_d_variants_same_physics(monkeypatch):
    import pipeline78.run as r
    from pipeline78 import bands as B, sampling, runcfg
    mjd = np.arange(57800.0, 59400.0, 1.0)              # cubre la plantilla mas larga mas la cola
    epochs = {"g": (mjd, np.full(mjd.size, 20.5)), "r": (mjd + 0.1, np.full(mjd.size, 20.5)),
              "i": (mjd[::4] + 0.2, np.full(mjd[::4].size, 20.0))}
    monkeypatch.setitem(r._W, "seed", 20261002)
    monkeypatch.setitem(r._W, "bands", B.survey_bands("ZTF")); monkeypatch.setitem(r._W, "rest", B.rest_bands())
    monkeypatch.setitem(r._W, "tpl", {c: _real_tpls(c) for c in ("Ia", "II")})
    out = {}
    for name in ("ztf_v78", "ztf_v78_tail", "ztf_v78_texp", "ztf_v78_t9_det0.5"):
        cfg = runcfg.RUNS_CFG[name]
        monkeypatch.setitem(r._W, "cfg", cfg); monkeypatch.setitem(r._W, "z", sampling.z_sampler(cfg))
        out[name] = {(c, k): r.simulate("F1", c, k, epochs, 0.02) for c in ("Ia", "II") for k in (1, 2, 3)}
    tpl = {t["sn"]: t for c in ("Ia", "II") for t in r._W["tpl"][c]}
    for key, (s0, d0) in out["ztf_v78"].items():
        assert s0["status"] == "ok"
        t_rel = (tpl[s0["template"]]["time"] - tpl[s0["template"]]["t_peak"]) * (1 + s0["z"])
        t0, t1 = s0["t_anchor"] + t_rel[0], s0["t_anchor"] + t_rel[-1]
        for v in ("ztf_v78_tail", "ztf_v78_texp", "ztf_v78_t9_det0.5"):
            s, d = out[v][key]
            for f in ("template", "z", "ebmv_host", "m_peak_abs", "t_anchor"):
                if not (v == "ztf_v78_t9_det0.5" and f == "t_anchor"):     # t9: ancla uniforme (Fix H)
                    assert s[f] == s0[f], (v, key, f)
        s, d = out["ztf_v78_t9_det0.5"][key]                 # deteccion logistica + UL como ALeRCE + limpieza
        assert mjd.min() <= s["t_anchor"] <= mjd.max() + 0.2 and s["t_anchor"] != s0["t_anchor"]
        if s["found"]:
            dt = d[d.upperlimit == "F"]
            assert d.mjd.max() == dt.mjd.max() and d.mjd.between(dt.mjd.min() - 50, dt.mjd.min() + 400).all()
            assert s["n_det_r"] == ((d["filter"] == "r") & d.detected).sum() and s["n_rows"] == len(d)
            dm = np.sort(dt.mjd.to_numpy())                    # cada UL tiene una det en los 30 d siguientes
            for t in d[d.upperlimit == "T"].mjd:
                assert ((dm > t) & (dm <= t + 30.0)).any()
        d = out["ztf_v78_tail"][key][1]
        assert (d.mjd > t1).any()
        pd.testing.assert_frame_equal(d[d.mjd <= t1].reset_index(drop=True), d0)
        d = out["ztf_v78_texp"][key][1]
        pd.testing.assert_frame_equal(d[d.mjd >= t0].reset_index(drop=True), d0[d0.mjd >= t0].reset_index(drop=True))
        t = tpl[s0["template"]]
        t_exp = s0["t_anchor"] + (t["t_Bmax"] - 18.9 - t["t_peak"]) * (1 + s0["z"]) if key[0] == "Ia" else np.inf
        if t_exp >= t0:                                    # texp solo toca Ia, y solo si t_exp < t0
            pd.testing.assert_frame_equal(d, d0)
        else:
            assert not ((d.mjd >= t_exp) & (d.mjd < t0)).any()
            assert (d[d.mjd < t_exp].upperlimit == "T").all() and (d.mjd < t_exp).any()
    assert sum(s["found"] for s, _ in out["ztf_v78_t9_det0.5"].values()) >= 3


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and n != "test_d_variants_same_physics":
            f(); print("ok", n)
