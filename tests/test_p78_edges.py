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
    for name in ("ztf_v78", "ztf_v78_tail", "ztf_v78_texp"):
        cfg = runcfg.RUNS_CFG[name]
        monkeypatch.setitem(r._W, "cfg", cfg); monkeypatch.setitem(r._W, "z", sampling.z_sampler(cfg))
        out[name] = {(c, k): r.simulate("F1", c, k, epochs, 0.02) for c in ("Ia", "II") for k in (1, 2, 3)}
    tpl = {t["sn"]: t for c in ("Ia", "II") for t in r._W["tpl"][c]}
    for key, (s0, d0) in out["ztf_v78"].items():
        assert s0["status"] == "ok"
        t_rel = (tpl[s0["template"]]["time"] - tpl[s0["template"]]["t_peak"]) * (1 + s0["z"])
        t0, t1 = s0["t_anchor"] + t_rel[0], s0["t_anchor"] + t_rel[-1]
        for v in ("ztf_v78_tail", "ztf_v78_texp"):
            s, d = out[v][key]
            for f in ("template", "z", "ebmv_host", "m_peak_abs", "t_anchor"):
                assert s[f] == s0[f], (v, key, f)
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


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and n != "test_d_variants_same_physics":
            f(); print("ok", n)
