# pipeline78/sampling.py
"""Sorteos de una simulacion con SU propio rng (determinista e independiente del orden de ejecucion)."""
import functools
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from config import EXTINCTION_CONFIG, LUMINOSITY_CONFIG, PHILLIPS_CONFIG
from pipeline78.paths import DATA, STORE

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)          # la misma que core.utils.DL_calculator
EXT_KEY = {"Ia": "SNIa", "II": "SNII_v78", "IIb": "SNIIb_v78", "IIn": "SNIIn_v78", "Ibc": "SNIbc"}


EXT_KEY_SUBTYPE = {"Ib": "SNIb", "Ic": "SNIc", "Ic-BL": "SNIcBL"}


def sample_ebv_host(rng, cls, subtype=None, ii_dust=None):
    key = "SNII_sudare" if (cls == "II" and ii_dust == "sudare") else EXT_KEY_SUBTYPE.get(subtype, EXT_KEY[cls])
    p = EXTINCTION_CONFIG[key]
    if rng.random() < p["frac_zero"]:
        return float(abs(rng.normal(0.0, p["sigma_zero"]))), float(p["Rv"])
    av = min(float(rng.exponential(p["tau"])), float(p["Av_max"]))
    return av / float(p["Rv"]), float(p["Rv"])


def sample_mpeak(rng, cls, dm15=None, subtype=None, ii_dust=None):
    """M intrinseco (libre de polvo) en la banda de referencia del catalogo (D4), salvo las clases en
    LF_AFTER_HOST_DUST, donde M es el pico con el polvo del host adentro. Truncado a clip por re-sorteo."""
    c = dict(LUMINOSITY_CONFIG["clip"])
    c.update(LUMINOSITY_CONFIG.get("clip_by_class", {}).get(cls, {}))     # corte por clase encima del global
    for _ in range(1000):
        if cls == "Ia" and PHILLIPS_CONFIG.get("enabled", False):
            d = dm15 if dm15 is not None and np.isfinite(dm15) else PHILLIPS_CONFIG["dm15_default"]
            m = (PHILLIPS_CONFIG["M0"] + PHILLIPS_CONFIG["slope"] * (d - PHILLIPS_CONFIG["dm15_ref"])
                 + rng.normal(0.0, PHILLIPS_CONFIG["sigma_resid"]))
        else:
            mp = LUMINOSITY_CONFIG["M_peak"]
            p = mp[subtype] if subtype in mp and subtype != "Ia" else mp[cls]
            if cls == "II" and ii_dust == "sudare" and subtype and subtype + "_dered" in mp:     # variante: LF desenrojecida
                p = mp[subtype + "_dered"]
            if "median" in p:      # split-normal asimetrica (SLSN-I): brillante = M mas negativo
                n = rng.normal()
                m = p["median"] + n * (p["sigma_bright"] if n < 0 else p["sigma_faint"])
            else:
                m = rng.normal(p["mean"], p["sigma"])
        if c["min"] <= m <= c["max"]:
            return float(m)
    return float(np.clip(m, c["min"], c["max"]))


def zgrid_cdf(zmin, zmax, n=4000):
    z = np.linspace(zmin, zmax, n)
    dv = COSMO.differential_comoving_volume(z).value
    c = np.concatenate([[0.0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(z))])
    return z, c / c[-1]


@functools.lru_cache(maxsize=None)
def _vol_norm(zmin, zmax, n=4000):
    g = np.linspace(zmin, zmax, n)
    dv = COSMO.differential_comoving_volume(g).value
    return float(np.sum(0.5 * (dv[1:] + dv[:-1]) * np.diff(g)))


def z_volume_weight(z, zmin, zmax):
    """(dV/dz)(z) / int_zmin^zmax dV/dz, con la misma cosmologia que zgrid_cdf (normalizacion en cache)."""
    return COSMO.differential_comoving_volume(np.asarray(z, float)).value / _vol_norm(float(zmin), float(zmax))


def z_sampler(cfg):
    mode = cfg["z_mode"]
    if mode == "fixed":
        return lambda rng, cls: float(cfg["z_fixed"])
    if mode == "empirical":
        vals = {c: np.loadtxt(DATA / f) for c, f in cfg["z_files"].items()}
        return lambda rng, cls: float(max(0.005, rng.choice(vals[cls]) + rng.normal(0.0, 0.003)))
    if mode == "volumetric":
        z, c = zgrid_cdf(cfg["zmin"], cfg["zmax"])
        return lambda rng, cls: float(np.interp(rng.random(), c, z))
    if mode == "uniform_weighted":
        zmin, zmx = cfg["zmin"], cfg["zmax_by_class"]
        return lambda rng, cls: float(rng.uniform(zmin, zmx[cls]))
    raise ValueError(mode)


def load_mw(cfg):
    mode = cfg["mw_mode"]
    if mode == "const":
        return {}
    if mode == "ztf_sfd":
        d = pd.read_parquet(DATA / "sfd98_cache.parquet")
        return dict(zip(d["oid"], d["ebmv_mw"].astype(float)))
    if mode == "sudare_fields":
        d = pd.read_csv(STORE / "sudare_fields.csv")
        return dict(zip(d["field"], d["ebmv_mw"].astype(float)))
    raise ValueError(mode)
