# pipeline78/sampling.py
"""Sorteos de una simulacion con SU propio rng (determinista e independiente del orden de ejecucion)."""
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from config import EXTINCTION_CONFIG, LUMINOSITY_CONFIG, PHILLIPS_CONFIG
from pipeline78.paths import DATA, STORE

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)          # la misma que core.utils.DL_calculator
EXT_KEY = {"Ia": "SNIa", "II": "SNII", "IIb": "SNIIb", "IIn": "SNIIn", "Ibc": "SNIbc"}


def sample_ebv_host(rng, cls):
    p = EXTINCTION_CONFIG[EXT_KEY[cls]]
    if rng.random() < p["frac_zero"]:
        return float(abs(rng.normal(0.0, p["sigma_zero"]))), float(p["Rv"])
    av = min(float(rng.exponential(p["tau"])), float(p["Av_max"]))
    return av / float(p["Rv"]), float(p["Rv"])


def sample_mpeak(rng, cls, dm15=None, subtype=None):
    """M intrinseco (libre de polvo) en la banda de referencia del catalogo (D4)."""
    if cls == "Ia" and PHILLIPS_CONFIG.get("enabled", False):
        d = dm15 if dm15 is not None and np.isfinite(dm15) else PHILLIPS_CONFIG["dm15_default"]
        m = (PHILLIPS_CONFIG["M0"] + PHILLIPS_CONFIG["slope"] * (d - PHILLIPS_CONFIG["dm15_ref"])
             + rng.normal(0.0, PHILLIPS_CONFIG["sigma_resid"]))
    else:
        mp = LUMINOSITY_CONFIG["M_peak"]
        p = mp[subtype] if subtype in mp and subtype != "Ia" else mp[cls]
        if "median" in p:      # split-normal asimetrica (SLSN-I): brillante = M mas negativo
            n = rng.normal()
            m = p["median"] + n * (p["sigma_bright"] if n < 0 else p["sigma_faint"])
        else:
            m = rng.normal(p["mean"], p["sigma"])
    c = LUMINOSITY_CONFIG["clip"]
    return float(np.clip(m, c["min"], c["max"]))


def zgrid_cdf(zmin, zmax, n=4000):
    z = np.linspace(zmin, zmax, n)
    dv = COSMO.differential_comoving_volume(z).value
    c = np.concatenate([[0.0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(z))])
    return z, c / c[-1]


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
