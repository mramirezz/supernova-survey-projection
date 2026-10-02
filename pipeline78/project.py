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


def project_one(t_rel, mags, epochs, t_anchor, rng, cfg):
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
        mm[mj < t0] = 99.0                               # antes de la explosion: no hay flujo
        if cfg["rule"] == "ztf":                         # identico a multiband_projection.py:356-372
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
