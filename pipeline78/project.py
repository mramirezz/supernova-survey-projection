# pipeline78/project.py
"""Ancla en el cielo y cadencia del survey: interpolacion, ruido, deteccion y limites."""
import numpy as np
import pandas as pd
from scipy.special import expit
from pipeline78.lcclean import clean_lc

ALERCE_PRV_DAYS = 30        # d: ALeRCE guarda una no deteccion solo si hay una alerta en los 30 d siguientes (prv_candidates)


def _span(epochs):
    lo = min(m.min() for m, _ in epochs.values())
    hi = max(m.max() for m, _ in epochs.values())
    return float(lo), float(hi)


def anchor_time(cfg, rng, k, n, epochs):
    lo, hi = _span(epochs)
    if cfg["anchor"] == "pivot":                       # ZTF: pivote deterministico (tesis cap. 3)
        return lo + (hi - lo) / n * (k + 0.5)
    if cfg["anchor"] == "uniform":                     # ZTF (Fix H): fecha al azar en el log del campo, sin fechas fijas
        return float(rng.uniform(lo, hi))
    if cfg["anchor"] == "uniform_window":              # SUDARE: [t0 - 365, tK] como la ec. 3 de SUDARE I
        return float(rng.uniform(lo - cfg["window_pre"], hi))
    raise ValueError(cfg["anchor"])


SIGMA_MAX = 1.0857 / 1e-6      # mag: el piso de S/N 1e-6 de la regla snr; las filas sin flujo (99) quedan ahi


def sigma_tres_terminos(dm, A, B, C):
    """sigma_m con dm = m_lim - m: fondo (A, S/N 5 en el limite), fuente y host (B, ~F^-1/2) y piso (C), en cuadratura."""
    dm = np.asarray(dm, float)
    return np.sqrt((A * 1.0857 / (5.0 * 10.0 ** (0.4 * dm))) ** 2 + (B * 10.0 ** (-0.2 * dm)) ** 2 + C ** 2)


def _noise_params(cfg, b):
    """noise_params global {"A","B","C"} o por banda {"g": {...}, "r": {...}, ...}."""
    p = cfg["noise_params"]
    p = p if "A" in p else p[b]
    return p["A"], p["B"], p["C"]


def _banda(v, b):
    """Parametro global (numero) o por banda {"g": .., "r": .., "i": ..}; una banda que falta es KeyError."""
    return v[b] if isinstance(v, dict) else v


def _noise(mm, ml, rng, cfg, b):
    """noise_model "snr" (default): sigma = clip(1.0857/S/N, sigma_floor), S/N = noise_k 10^(0.4 (ml - mm)).
    "tres_terminos": sigma_tres_terminos(ml - mm, A, B, C) con cfg["noise_params"], tope SIGMA_MAX (el mismo piso de
    S/N 1e-6: una fila sin flujo queda en m_obs ~ 84 en _det y nunca se detecta). Un solo sorteo normal por fila.
    noise_draw_scale k (numero o por banda, default 1): el sorteo es m_obs = mm + N(0, k sigma) y magerr sigue siendo
    sigma (ZTF: dispersion realizada ~0.5 del sigmapsf). noise_draw_scale_lowsig {"sigma_max", "k"}: otro k donde
    sigma < sigma_max. Mismo sorteo normal en el mismo orden: k = 1 da los mismos bytes."""
    if cfg["rule"] != "ztf":
        raise ValueError(cfg["rule"])
    model = cfg.get("noise_model", "snr")
    if model == "snr":
        snr = cfg["noise_k"] * 10.0 ** (0.4 * (ml - mm))      # identico a multiband_projection.py:356-372
        sig = np.clip(1.0857 / np.maximum(snr, 1e-6), cfg["sigma_floor"], None)
    elif model == "tres_terminos":
        sig = np.minimum(sigma_tres_terminos(ml - mm, *_noise_params(cfg, b)), SIGMA_MAX)
    else:
        raise ValueError(f"noise_model={model!r}")
    k, lo = _banda(cfg.get("noise_draw_scale", 1.0), b), cfg.get("noise_draw_scale_lowsig")
    if lo is not None:
        k = np.where(sig < lo["sigma_max"], _banda(lo["k"], b), k)
    return mm + rng.normal(0.0, k * sig), sig


def _det(mm, ml, mobs, u, cfg, b):
    """Sin uniforme (det_model hard, o UL de flujo nulo antes de t_exp): corte duro mm < ml.
    logistic: P = det_eps/(1 + exp((m_obs - (ml - det_m0))/det_w)) y det = u < P, sobre la magnitud MEDIDA (S/N medida,
    como SEARCHEFF de SNANA). det_m0, det_w y det_eps (techo de eficiencia, default 1) son numeros o dicts por banda.
    m_obs sale del mismo sorteo normal del ruido, pero en flujo: con z = (mobs - mm)/sig y
    S/N = 1.0857/sig, el flujo medido es f (1 - z/S/N) = f (1 - (mobs - mm)/1.0857) y m_obs = mm - 2.5 log10 de eso.
    mobs en magnitud diverge para fuentes debiles (sig ~1e6 mag en las filas sin flujo) y detectaria la mitad de ellas.
    m_obs = mobs a primer orden en las detecciones, mobs <= m_obs siempre, y una fila sin flujo (99) queda en
    m_obs ~ 84: nunca se detecta."""
    if u is None:
        return mm < ml
    f = 1.0 - (mobs - mm) / 1.0857                           # flujo medido / flujo del modelo
    with np.errstate(divide="ignore"):
        m_obs = np.where(f > 0, mm - 2.5 * np.log10(np.maximum(f, 1e-300)), np.inf)
    p = expit((ml - _banda(cfg["det_m0"], b) - m_obs) / _banda(cfg["det_w"], b))
    return u < _banda(cfg.get("det_eps", 1.0), b) * p      # 1.0 * p == p: sin det_eps, los mismos bytes


def _frame(b, mj, ml, mm, mobs, sig, det):
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
    edge_pre "fireball": como texp, pero en [t_exp, t0) hay bola de fuego: flujo ~ (t - t_exp)^2 hasta el primer punto
    de la plantilla, con su color (mm = mags[0] - 5 log10((t - t_exp)/(t0 - t_exp))), con ruido, deteccion y UL de siempre.
    edge_post "none" (default): nada despues de t1. "tail": recta en magnitud hasta t1 + tail_days (1+z), con la
    pendiente de los ultimos tail_fit_days (1+z) de la plantilla y nunca menor que tail_min_slope (no sube).
    El ruido de [t0 - pre_ul_days, t1] se sortea igual en todas las variantes, el de la cola despues y el de la bola de
    fuego al final: las filas comunes no cambian entre variantes.
    det_model "hard" (default) o "logistic" (_det): los uniformes van despues de todo el ruido, en el mismo orden, asi
    las filas comunes conservan ruido y uniforme. lc_clean True: clean_lc con t_ref = primera deteccion en g o r
    (equivale al descubrimiento). ul_after_last False: fuera los UL despues de la ultima deteccion de cualquier banda
    (alertas), aplicado despues de limpiar.
    pre_ul_mode "window" (default): quedan todos los UL. "alerce": un UL queda solo si hay una deteccion de cualquier
    banda en (t, t + ALERCE_PRV_DAYS], como los prv_candidates de ALeRCE (no quedan UL tras la ultima deteccion). Va
    despues de decidir la deteccion y antes de clean_lc; la ventana pre_ul_days tiene que cubrir esos 30 d."""
    t = t_anchor + t_rel
    t0, t1 = float(t[0]), float(t[-1])
    pre, edge_pre, edge_post = cfg["pre_ul_days"], cfg.get("edge_pre", "window"), cfg.get("edge_post", "none")
    if edge_pre not in ("window", "texp", "fireball") or edge_post not in ("none", "tail"):
        raise ValueError(f"edge_pre={edge_pre!r} edge_post={edge_post!r}")
    if cfg.get("det_model", "hard") not in ("hard", "logistic"):
        raise ValueError(f"det_model={cfg['det_model']!r}")
    eps = cfg.get("det_eps", 1.0)
    if not all(0.0 < e <= 1.0 for e in (eps.values() if isinstance(eps, dict) else (eps,))):
        raise ValueError(f"det_eps={eps!r}")
    if cfg.get("pre_ul_mode", "window") not in ("window", "alerce"):
        raise ValueError(f"pre_ul_mode={cfg['pre_ul_mode']!r}")
    if cfg.get("noise_model", "snr") not in ("snr", "tres_terminos"):
        raise ValueError(f"noise_model={cfg['noise_model']!r}")
    if edge_post == "tail" and z is None:
        raise ValueError("edge_post='tail' necesita z")
    t_exp = t_anchor + t_exp_rel if edge_pre in ("texp", "fireball") and t_exp_rel is not None else None
    if t_exp is not None and t_exp >= t0:
        t_exp = None
    parts, cola, fb, ruido = {}, [], [], []              # bloque: [b, mj, ml, mm, mobs, sig, u, k]; ruido en orden de sorteo
    for b in cfg["bands"]:
        if b not in mags or b not in epochs:
            continue
        mjd, mlim = epochs[b]
        parts[b] = []
        if t_exp is not None:
            ul = (mjd >= t_exp - pre) & (mjd < t_exp)
            if ul.any():
                parts[b].append([b, mjd[ul], mlim[ul], np.full(ul.sum(), 99.0), mlim[ul], np.full(ul.sum(), np.nan), None, None])
        sel = (mjd >= t0 - pre) & (mjd <= t1)
        if sel.any():
            mj, ml = mjd[sel], mlim[sel]
            mm = np.interp(mj, t, mags[b])
            mm[mj < t0] = 99.0                           # antes de la explosion: no hay flujo
            mobs, sig = _noise(mm, ml, rng, cfg, b)
            k = mj >= t0 if t_exp is not None else None  # texp/fireball: [t_exp, t0) no queda como no deteccion falsa
            parts[b].append([b, mj, ml, mm, mobs, sig, None, k])
            ruido.append(parts[b][-1])
        if edge_pre == "fireball" and t_exp is not None:
            e = (mjd >= t_exp) & (mjd < t0)
            if e.any():
                dt = np.maximum((mjd[e] - t_exp) / (t0 - t_exp), 1e-12)
                fb.append((b, mjd[e], mlim[e], mags[b][0] - 5.0 * np.log10(dt)))
        if edge_post == "tail":
            tl = (mjd > t1) & (mjd <= t1 + cfg["tail_days"] * (1.0 + z))
            if tl.any():
                f = t >= t1 - cfg["tail_fit_days"] * (1.0 + z)
                s = np.polyfit(t[f] - t1, mags[b][f], 1)[0] if f.sum() > 1 else np.nan
                s = s if s >= cfg["tail_min_slope"] else cfg["tail_min_slope"]     # nan -> pendiente minima
                cola.append((b, mjd[tl], mlim[tl], mags[b][-1] + s * (mjd[tl] - t1)))
    for b, mj, ml, mm in cola + fb:                      # ruido de la cola y despues el de la bola de fuego, al final
        parts[b].append([b, mj, ml, mm, *_noise(mm, ml, rng, cfg, b), None, None])
        ruido.append(parts[b][-1])
    if cfg.get("det_model", "hard") == "logistic":
        for blk in ruido:                                # uniformes despues de todo el ruido, en el orden del ruido
            blk[6] = rng.uniform(size=len(blk[1]))
    frames = []
    for blk in (x for fr in parts.values() for x in fr):
        b, mj, ml, mm, mobs, sig, u, k = blk
        det = _det(mm, ml, mobs, u, cfg, b)
        if k is not None:
            mj, ml, mm, mobs, sig, det = mj[k], ml[k], mm[k], mobs[k], sig[k], det[k]
        if len(mj):
            frames.append(_frame(b, mj, ml, mm, mobs, sig, det))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if cfg.get("pre_ul_mode", "window") == "alerce":   # UL solo con una det (cualquier banda) en (t, t + 30 d]
        dm, t_ul = np.sort(df.loc[df["upperlimit"] == "F", "mjd"].to_numpy(float)), df["mjd"].to_numpy(float)
        nxt = np.append(dm, np.inf)[np.searchsorted(dm, t_ul, side="right")]     # primera det despues de cada fila
        df = df[(df["upperlimit"] == "F").to_numpy() | (nxt <= t_ul + ALERCE_PRV_DAYS)].reset_index(drop=True)
    gr = (df["upperlimit"] == "F") & df["filter"].isin(["g", "r"])
    if cfg.get("lc_clean", False) and gr.any():            # t_ref = primera det en g o r (las reales no tienen i)
        df = clean_lc(df, float(df.loc[gr, "mjd"].min()))[0]
    det = df["upperlimit"] == "F"
    if not cfg.get("ul_after_last", True) and det.any():   # despues de limpiar: sin UL tras la ultima det que queda
        df = df[det | (df["mjd"] <= df.loc[det, "mjd"].max())].reset_index(drop=True)
    return df
