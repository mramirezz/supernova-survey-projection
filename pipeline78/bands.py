"""Bandas: curvas de transmision, punto cero AB y fotometria sintetica por fotones.

Convencion del pipeline (apendice C de la tesis):
    F = int F_lambda R lambda dlambda / int R lambda dlambda   (denominador sobre la banda completa)
    m = -2.5 log10(F / F0)
Con F0 calculado con la MISMA integral sobre el espectro AB (3631 Jy), m es una magnitud AB.
"""
from dataclasses import dataclass
import numpy as np
from pipeline78.paths import FILTERS, LEGACY_RESP

C_AA = 2.99792458e18          # c en A/s
COVERAGE_MIN = 0.95           # misma regla que la proyeccion historica (porcentaje > 0.95)
TRAPZ = np.trapezoid
SURVEY_FILES = {"ZTF": "ZTF_{}.dat", "SUDARE": "OmegaCAM_{}.dat"}


@dataclass(frozen=True, eq=False)
class Band:
    name: str
    wave: np.ndarray
    resp: np.ndarray
    f0: float


def read_curve(path):
    d = np.loadtxt(path, comments="#")
    d = d[np.argsort(d[:, 0])]
    return d[:, 0].astype(float), np.clip(d[:, 1].astype(float), 0.0, None)


def ab_f0(wave, resp):
    f_ab = 3631e-23 * C_AA / wave**2
    return float(TRAPZ(f_ab * resp * wave, wave) / TRAPZ(resp * wave, wave))


def make_band(name, path, f0=None):
    w, r = read_curve(path)
    return Band(name, w, r, ab_f0(w, r) if f0 is None else float(f0))


def synphot(wave, flux2d, band):
    """Flujo sintetico por epoca y fraccion de la banda cubierta por [wave[0], wave[-1]]."""
    total = TRAPZ(band.resp, band.wave)
    inside = (band.wave >= wave[0]) & (band.wave <= wave[-1])
    cov = float(TRAPZ(band.resp[inside], band.wave[inside]) / total) if inside.sum() > 1 else 0.0
    r = np.interp(wave, band.wave, band.resp, left=0.0, right=0.0)
    num = TRAPZ(np.asarray(flux2d) * (r * wave)[None, :], wave, axis=1)
    return num / TRAPZ(band.resp * band.wave, band.wave), cov


def survey_bands(survey, names=("g", "r", "i")):
    return [make_band(n, FILTERS / SURVEY_FILES[survey].format(n)) for n in names]


def rest_bands():
    """Bandas de reposo del catalogo con los F0 historicos (Vega para BVR, ~AB para r)."""
    from core.utils import cteB, cteV, cteR, cter
    files = {"B_rest": ("spline_B.txt", cteB), "V_rest": ("spline_V.txt", cteV),
             "R_rest": ("bessell_R_ph_lines.dat", cteR), "r_rest": ("spline_r'.txt", cter)}
    return {k: make_band(k, LEGACY_RESP / f, f0) for k, (f, f0) in files.items()}


def legacy_band(name):
    """Curva SDSS y F0 de la proyeccion historica. Solo para la prueba dorada."""
    from core import utils
    f = {"g": "spline_g'.txt", "r": "spline_r'.txt", "i": "spline_i'.txt"}[name]
    return make_band(name, LEGACY_RESP / f, getattr(utils, "cte" + name))
