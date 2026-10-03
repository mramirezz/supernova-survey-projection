# tests/test_p78_edges.py
"""Bordes de plantilla configurables: window/none identico a antes, texp, tail y determinismo entre variantes."""
import sys, pathlib
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


def test_e_logistic_limit_is_hard():
    """m0 = 0 y w -> 0: el corte duro. El ruido no cambia (los uniformes van despues)."""
    pd.testing.assert_frame_equal(_run(dict(CFG, det_model="logistic", det_m0=0.0, det_w=1e-9)), _run(CFG))
    t_rel = _inputs()[0]
    for cfg, kw in ((TAIL, dict(z=Z)), (dict(TAIL, edge_pre="fireball"), dict(z=Z, t_exp_rel=t_rel[0] - 8.0))):
        pd.testing.assert_frame_equal(_run(dict(cfg, det_model="logistic", det_m0=0.0, det_w=1e-9), **kw), _run(cfg, **kw))
    with pytest.raises(ValueError):
        _run(dict(CFG, det_model="searcheff"))


def test_e_logistic_half_at_m0():
    """Fraccion detectada en mm = maglim - m0: ~0.5. Mas brillante detecta casi siempre, mas debil casi nunca."""
    mjd = np.arange(59000.0, 63000.0, 1.0)
    t_rel = np.array([0.0, 4000.0])
    for dm, lo, hi in ((0.0, 0.47, 0.53), (-0.6, 0.94, 1.0), (0.6, 0.0, 0.06)):
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
    c = _run(dict(cfg, ul_after_last=False, lc_clean=True), z=Z)
    pd.testing.assert_frame_equal(c, clean_lc(b, b.loc[b.upperlimit == "F", "mjd"].min())[0])
    mjd = np.arange(58000.0, 59300.0, 1.0)              # epocas 900 d antes: la ventana desde la primera det las saca
    t_rel, mags, _ = _inputs()
    e = {k: (mjd, np.full(mjd.size, 20.5)) for k in ("g", "r")}
    c = project_one(t_rel, mags, e, T_ANCHOR, np.random.default_rng(3), dict(cfg, pre_ul_days=900.0, lc_clean=True), z=Z)
    first = c.loc[c.upperlimit == "F", "mjd"].min()
    assert c.mjd.min() >= first - 50.0 and c.mjd.max() <= first + 400.0


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
                assert s[f] == s0[f], (v, key, f)
        s, d = out["ztf_v78_t9_det0.5"][key]                 # deteccion logistica + sin UL tras la ultima + limpieza
        if s["found"]:
            dt = d[d.upperlimit == "F"]
            assert d.mjd.max() == dt.mjd.max() and d.mjd.between(dt.mjd.min() - 50, dt.mjd.min() + 400).all()
            assert s["n_det_r"] == ((d["filter"] == "r") & d.detected).sum() and s["n_rows"] == len(d)
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
