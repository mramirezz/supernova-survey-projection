"""Espectros de reposo a 10 pc -> magnitudes AB observadas por banda.

Orden fisico: brillo intrinseco (dmag) -> host (reposo, R_V del tipo) -> redshift (lambda*(1+z), F/(1+z))
-> Via Lactea (observado, R_V 3.1) -> distancia (10 pc / D_L)^2. Tiempos: (t - t_peak)*(1+z).
Sin LOESS (D2): las series congeladas son diarias y ya estan mangladas a la fotometria.
"""
import numpy as np
from core.correction import redden_spectrum_adjusted
from core.utils import DL_calculator
from pipeline78.bands import synphot, COVERAGE_MIN


def extinction_factor(wave, rv, ebv):
    wave = np.asarray(wave, dtype=float)
    if ebv <= 0:
        return np.ones_like(wave)
    return np.asarray(redden_spectrum_adjusted(wave, np.ones_like(wave), Rv=rv, ebmv=ebv), dtype=float)


def observed_lightcurves(tpl, z, ebv_host, rv_host, ebv_mw, bands, dmag=0.0):
    w = np.asarray(tpl["wave"], dtype=float)
    f = np.asarray(tpl["flux"], dtype=np.float64) * 10.0 ** (-0.4 * dmag)
    f = f * extinction_factor(w, rv_host, ebv_host)[None, :]
    wo = w * (1.0 + z)
    f = f / (1.0 + z)
    f = f * extinction_factor(wo, 3.1, ebv_mw)[None, :]
    f = f * (1e-5 / DL_calculator(z)) ** 2
    mags = {}
    for b in bands:
        F, cov = synphot(wo, f, b)
        if cov > COVERAGE_MIN:
            mags[b.name] = -2.5 * np.log10(np.clip(F, 1e-300, None) / b.f0)
    t_rel = (np.asarray(tpl["time"], dtype=float) - float(tpl["t_peak"])) * (1.0 + z)
    return t_rel, mags
