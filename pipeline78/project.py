# pipeline78/project.py
"""Ancla en el cielo y cadencia del survey: interpolacion, ruido, deteccion y limites."""
import numpy as np
import pandas as pd


def _span(epochs):
    lo = min(m.min() for m, _ in epochs.values())
    hi = max(m.max() for m, _ in epochs.values())
    return float(lo), float(hi)


def anchor_time(cfg, rng, k, n, epochs):
    lo, hi = _span(epochs)
    if cfg["anchor"] == "pivot":                       # ZTF: pivote deterministico (tesis cap. 3)
        return lo + (hi - lo) / n * (k + 0.5)
    if cfg["anchor"] == "uniform_window":              # SUDARE: [t0 - 365, tK] como la ec. 3 de SUDARE I
        return float(rng.uniform(lo - cfg["window_pre"], hi))
    raise ValueError(cfg["anchor"])


def _noise(mm, ml, rng, cfg):
    if cfg["rule"] != "ztf":
        raise ValueError(cfg["rule"])
    snr = cfg["noise_k"] * 10.0 ** (0.4 * (ml - mm))      # identico a multiband_projection.py:356-372
    sig = np.clip(1.0857 / np.maximum(snr, 1e-6), cfg["sigma_floor"], None)
    return mm + rng.normal(0.0, sig), sig


def _frame(b, mj, ml, mm, mobs, sig):
    det = mm < ml
    return pd.DataFrame({
        "mjd": mj, "filter": b, "maglimit": ml.astype(np.float32),
        "magnitud_modelo": mm.astype(np.float32),
        "magnitud_proyectada": np.where(det, mobs, ml).astype(np.float32),
        "magerr": np.where(det, sig, np.nan).astype(np.float32),
        "upperlimit": np.where(det, "F", "T"), "detected": det, "found": det})


def project_one(t_rel, mags, epochs, t_anchor, rng, cfg, t_exp_rel=None, z=None):
    """Bordes de la plantilla [t0, t1]:
    edge_pre "window" (default): UL de flujo nulo en [t0 - pre_ul_days, t0).
    edge_pre "texp": sin epocas en [t_exp, t0) y UL de flujo nulo en [t_exp - pre_ul_days, t_exp); sin t_exp o con
    t_exp >= t0 es "window". t_exp_rel es relativo al ancla, como t_rel.
    edge_post "none" (default): nada despues de t1. "tail": recta en magnitud hasta t1 + tail_days (1+z), con la
    pendiente de los ultimos tail_fit_days (1+z) de la plantilla y nunca menor que tail_min_slope (no sube).
    El ruido de [t0 - pre_ul_days, t1] se sortea igual en todas las variantes y el de la cola al final: las filas
    comunes no cambian entre variantes."""
    t = t_anchor + t_rel
    t0, t1 = float(t[0]), float(t[-1])
    pre, edge_pre, edge_post = cfg["pre_ul_days"], cfg.get("edge_pre", "window"), cfg.get("edge_post", "none")
    if edge_pre not in ("window", "texp") or edge_post not in ("none", "tail"):
        raise ValueError(f"edge_pre={edge_pre!r} edge_post={edge_post!r}")
    if edge_post == "tail" and z is None:
        raise ValueError("edge_post='tail' necesita z")
    t_exp = t_anchor + t_exp_rel if edge_pre == "texp" and t_exp_rel is not None else None
    if t_exp is not None and t_exp >= t0:
        t_exp = None
    parts, cola = {}, []
    for b in cfg["bands"]:
        if b not in mags or b not in epochs:
            continue
        mjd, mlim = epochs[b]
        parts[b] = []
        if t_exp is not None:
            ul = (mjd >= t_exp - pre) & (mjd < t_exp)
            if ul.any():
                parts[b].append(_frame(b, mjd[ul], mlim[ul], np.full(ul.sum(), 99.0), mlim[ul], np.full(ul.sum(), np.nan)))
        sel = (mjd >= t0 - pre) & (mjd <= t1)
        if sel.any():
            mj, ml = mjd[sel], mlim[sel]
            mm = np.interp(mj, t, mags[b])
            mm[mj < t0] = 99.0                           # antes de la explosion: no hay flujo
            mobs, sig = _noise(mm, ml, rng, cfg)
            if t_exp is not None:                        # texp: [t_exp, t0) sin informacion, no se finge una no deteccion
                k = mj >= t0
                mj, ml, mm, mobs, sig = mj[k], ml[k], mm[k], mobs[k], sig[k]
            parts[b].append(_frame(b, mj, ml, mm, mobs, sig))
        if edge_post == "tail":
            tl = (mjd > t1) & (mjd <= t1 + cfg["tail_days"] * (1.0 + z))
            if tl.any():
                f = t >= t1 - cfg["tail_fit_days"] * (1.0 + z)
                s = np.polyfit(t[f] - t1, mags[b][f], 1)[0] if f.sum() > 1 else np.nan
                s = s if s >= cfg["tail_min_slope"] else cfg["tail_min_slope"]     # nan -> pendiente minima
                cola.append((b, mjd[tl], mlim[tl], mags[b][-1] + s * (mjd[tl] - t1)))
    for b, mj, ml, mm in cola:                           # ruido de la cola al final: no mueve el de las filas <= t1
        parts[b].append(_frame(b, mj, ml, mm, *_noise(mm, ml, rng, cfg)))
    frames = [f for fr in parts.values() for f in fr if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else None
