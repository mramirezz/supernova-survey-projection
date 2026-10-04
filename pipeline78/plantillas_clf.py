"""Clasificador por ajuste bayesiano de plantillas, como SUDARE I (Cappellaro et al. 2015, 2015A&A...584A..62C, Sec.
4.1), que sigue a PSNID (Sako et al. 2011, 2011ApJ...738..162S). Tercer metodo (Mauricio, 2026-10-04), junto a Villar
(pipeline78.clf_villar) y las redes (pipeline78.nnclf). Ventaja nuestra: las plantillas son series de espectros, asi
que la curva observada a cualquier z es fotometria sintetica exacta (sin tablas de correccion K ni banda vecina).

Uso:
    python -m pipeline78.plantillas_clf biblioteca [--force]
    python -m pipeline78.plantillas_clf run --name NOMBRE [--cuatro-clases] [--sin-z] [--subset val_sel] [--limit N]

REGLAS
1. Biblioteca (una por survey, en cache). Por plantilla del catalogo (STORE/catalog.csv), nodo de z, E(B-V) del host y
   banda (ZTF g, r): flujo en una grilla de fase observada (dias desde t_peak, el maximo en r de reposo, paso 1 d) de
   engine.observed_lightcurves(tpl, z, E, R_V, ebv_mw=0, bands, dmag=0) sin la distancia (mu(z) se suma al
   clasificar). synphot es lineal en el flujo: toda la grilla de una plantilla sale de un producto matricial
   (espectros x pesos por z, E y banda), verificado contra engine.observed_lightcurves en celdas al azar al construir
   (|dm| < 1e-6 mag). Entre la primera y la ultima epoca: interpolacion lineal en magnitud, como project.project_one.
   Bordes de RUN_CFG (ztf_v78_t11), los de run.simulate + project.project_one:
     antes   Ia: bola de fuego (t - t_exp)^2 desde t_exp = t_Bmax - rise_Ia_days (reposo) hasta la primera epoca, sin
             flujo antes de t_exp. Las otras clases empiezan en la explosion: sin flujo antes de la primera epoca.
     despues cola recta en magnitud con la pendiente de los ultimos tail_fit_days (1+z), nunca menor que el piso de la
             clase (TAIL_PISO). Las sims cortan la cola a tail_days (1+z). Aca sigue hasta el final de la grilla: la
             verosimilitud necesita un modelo en toda epoca observada. Las plantillas sin cola en las sims
             (tail_min_span: II de menos de 120 d de reposo, que tampoco tienen observaciones despues de su ultima
             epoca) quedan sin flujo despues de la ultima epoca.
   Grillas (fijas a priori): z 0.005-0.20 paso 0.005, extendida hasta 0.42 (paso 0.01 hasta 0.30 y 0.02 despues: la
   val llega a z = 0.4). E(B-V) del host 0-1.0 paso 0.05 con el R_V del polvo de la plantilla (EXTINCTION_CONFIG por
   clase o subtipo, la llave de sampling.sample_ebv_host). Fase -160 a +520 d. Una banda que la plantilla no cubre a
   ese z (cobertura <= bands.COVERAGE_MIN, la regla de engine) queda invalida y ese z no se usa con datos en ella.
   Via Lactea por objeto como corrimiento por banda A_b = R_b E(B-V)_MW, R_b de la plantilla en su pico a ese z
   (E(B-V)_MW = 0.1 contra 0, sin polvo del host). Aproximacion: R_b no cambia con la fase ni con el polvo del host
   (~0.1 en R_b, ~0.005 mag con la mediana E(B-V)_MW = 0.044 de la val).
   Cache: RUNS/plantillas_clf/biblioteca_<clave>, clave = md5 de catalogo, filtros, bandas de reposo, bordes, R_V,
   grillas y VERSION.
2. Modelo: m(t, b) = L(plantilla, z, E, t - T_max, b) + mu(z) + dmag + A_b, con dmag = M - M_ref. En las clases de
   LF_AFTER_HOST_DUST (IIn) dmag = M - M_ref - A_ref(E), como run.simulate.
3. Priors: los aprobados de las simulaciones (SUDARE I usa polvo y escala planos).
   z     N(z_spec, SIGMA_Z = 0.005) (SUDARE I) truncada a [Z_FLOOR, borde de la grilla]. --sin-z o sin z: plana en
         ese rango. La forma de la curva usa el nodo de z de la celda. La distancia mu(z) se integra fino dentro de la
         celda (subceldas con dmu <= DMU_SUB): a z bajo la LF angosta de las Ia no queda como un peine.
   E     masa de la mezcla de sampling.sample_ebv_host en cada celda (la ultima junta todo lo que pasa de 1.0).
   M     la LF de sampling.sample_mpeak (Phillips en Ia con el dm15 de la plantilla, Li, Taddia, Shivvers), gaussiana
         truncada al clip. Integral numerica en dmag.
   T_max plana en [primera det - T_PRE, ultima det], paso 1 d (pico de la plantilla, marco observado).
   plantilla dentro de la clase: II e IIb en la proporcion n_by_class de RUN_CFG (8:2), subtipo con
         SUBTYPE_FRACTIONS y plantillas equiprobables dentro del subtipo (run.choose_template). Sin fracciones,
         equiprobables. Clases equiprobables (el prior de entrenamiento balanceado de los otros dos metodos).
4. Verosimilitud en flujo (unidades 10^(-0.4 (m - m_ref)), m_ref = la deteccion mas brillante). Detecciones
   gaussianas, sigma_f = 0.921 f_obs sqrt(magerr^2 + SIGMA_MOD^2), piso del modelo SIGMA_MOD = 0.05 mag fijado a
   priori y evaluado en el flujo observado: chi2 es cuadratico en la escala y la integral en dmag sale de tres sumas.
   UL: P(f_modelo < f_lim) = Phi((f_lim - f_modelo)/sigma), sigma = f_lim/5 (alertas de ZTF a 5 sigma), evaluado en
   el centro de la integral en dmag de cada (plantilla, z, E, T_max) (aproximacion: la escala la fijan las detecciones).
4b. Que UL entran (--ul). "todos" (por defecto, el modelo pedido): todos los de la curva. Las alertas reales traen UL
   en la misma noche y banda que una deteccion (74 de las 224 curvas de val_sel de 3 clases, 358 de 6290 UL, en la
   prueba chica del 2026-10-04), a veces 4 mag mas profundos que la SN detectada: son restas fallidas, las sims no los
   tienen, y con sigma = f_lim/5 hunden a toda plantilla que ajusta las detecciones. Variantes: "misma_noche" saca los
   UL con una deteccion de su banda a menos de 0.5 d, "previos" deja solo los UL antes de la primera deteccion,
   "ninguno" los saca todos. Elegir entre ellas es una decision de Mauricio (val_sel con el bootstrap pareado).
5. Evidencia: E_tipo = sum_pl pi(pl) sum_{z,E,T} P(z) P(E) P(T) int L(dmag) P(dmag) ddmag. La integral en dmag va en
   len(U) puntos (+-6 sigma) centrados en la combinacion gaussiana de la verosimilitud y el prior. Se calcula solo en
   las celdas (z, E, T_max) cuya aproximacion de Laplace queda a menos de LAPLACE_NATS = 30 del maximo de la
   plantilla. Las demas quedan con Laplace (pesan < e^-30). P_tipo = E_tipo/sum E.
6. Reales: solo la mitad val, con los lectores con guarda (nnclf.data.load_real_val: meta con csv linea a linea,
   pyarrow con filtro por oid; clf_villar.read_val_meta para el subset de splits.val_split). La mitad final no se lee.
   Entran las de >= MIN_DET detecciones en g + r. E(B-V)_MW de data/sfd98_cache.parquet (sampling.load_mw, el de las
   sims; una oid que falta usa mw_const, como run.run_unit).
7. Nada se ajusta con las reales: SIGMA_Z, SIGMA_MOD, el sigma de los UL, T_PRE, U y las grillas son a priori.
8. Salidas en RUNS/plantillas_clf/<nombre>: pred_real_val.csv (oid, subset, y_true, y_pred, p_<clase>, n_det,
   best_template, chi2_min, parametros MAP y medias posteriores dentro de la mejor plantilla), metrics.json (acc, bal_acc con IC 95 % bootstrap,
   F1 por clase, confusion, cobertura por subconjunto, tiempos) y sin_clasificar.csv.
"""
import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtr

from config import EXTINCTION_CONFIG, LF_AFTER_HOST_DUST, LUMINOSITY_CONFIG, PHILLIPS_CONFIG, SUBTYPE_FRACTIONS
from core.utils import DL_calculator
from pipeline78 import bands as B, engine, runcfg, sampling
from pipeline78.nnclf import data as D
from pipeline78.paths import FILTERS, RUNS, STORE
from pipeline78.store import load_template, md5_file

RUN_CFG = "ztf_v78_t11"
SURVEY, BANDS = "ZTF", ("g", "r")
Z_GRID = np.round(np.r_[np.arange(0.005, 0.2001, 0.005), np.arange(0.21, 0.3001, 0.01),
                        np.arange(0.32, 0.4201, 0.02)], 4)
EBV_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)
FASES = (-160.0, 520.0)          # d observados desde el pico de la plantilla; las Ia/IIn mas lentas parten en ~-145
DP = 1.0                         # paso de la fase y de T_max (d)
E_MW_REF = 0.1                   # E(B-V)_MW para medir R_b
BORDES = ("edge_pre", "rise_Ia_days", "edge_post", "tail_fit_days", "tail_min_slope", "tail_min_span")
SIGMA_Z = 0.005                  # SUDARE I: P(z) = N(z_spec, 0.005)
Z_FLOOR = 0.001                  # borde inferior del prior de z (tres val tienen z < 0.005)
DMU_SUB = 0.05                   # mag: paso de mu(z) dentro de la celda de un nodo
SIGMA_MOD = 0.05                 # mag: piso del error del modelo, a priori
UL_NSIG = 5.0                    # alertas de ZTF: limite a 5 sigma
T_PRE = 60.0                     # d: el pico puede caer hasta 60 d antes de la primera deteccion
MIN_DET = D.MIN_DET              # 3 detecciones en g + r, como las redes
U = np.linspace(-6.0, 6.0, 25)   # nodos de la integral en dmag, en unidades del ancho combinado
LAPLACE_NATS = 30.0              # celdas con Laplace mas de 30 nats bajo el maximo de la plantilla: quedan con Laplace
SQ2PI = math.sqrt(2.0 * math.pi)
UL_MODOS = ("todos", "misma_noche", "previos", "ninguno")
MASA_MIN_Z, MASA_MIN_E = 1e-4, 1e-6
LOG0 = -1e4                      # log de densidad nula (exp -> 0 sin nan)
CHUNK = 2_000_000                # elementos por bloque (float64): 16 MB por arreglo (4 procesos --sin-z: ~3 GB RSS)
VERSION = 1
K_MAG = 0.4 * math.log(10.0)     # 0.921: dm = dF/F / K_MAG
OUT_ROOT = RUNS / "plantillas_clf"
REAL_DIR = RUNS / "real_ztf"
SEED = 20261004
SUBSETS = ("val_rep", "val_sel", "val")


# ------------------------------------------------------------------------------------------------ priors
def ext_params(cls, subtype, ii_dust=None):
    """(llave, parametros) del polvo del host: la misma llave que sampling.sample_ebv_host."""
    key = "SNII_sudare" if (cls == "II" and ii_dust == "sudare") else \
        sampling.EXT_KEY_SUBTYPE.get(subtype, sampling.EXT_KEY[cls])
    return key, EXTINCTION_CONFIG[key]


def ebv_prior(cls, subtype, ebv_grid=EBV_GRID, ii_dust=None):
    """Masa de la mezcla de sampling.sample_ebv_host en cada celda de la grilla: |N(0, sigma_zero)| con prob frac_zero
    y A_V ~ Exp(tau) con tope Av_max (E = A_V/R_V) si no. Bordes en los puntos medios, la primera desde 0 y la ultima
    hasta infinito (el tope cae en la celda que lo contiene)."""
    p = ext_params(cls, subtype, ii_dust)[1]
    g = np.asarray(ebv_grid, float)
    e = np.r_[0.0, 0.5 * (g[1:] + g[:-1]), np.inf]
    c0 = 2.0 * ndtr(e / p["sigma_zero"]) - 1.0 if p["sigma_zero"] > 0 else (e > 0).astype(float)
    c1 = np.where(e >= p["Av_max"] / p["Rv"], 1.0, -np.expm1(-e * p["Rv"] / p["tau"]))
    return p["frac_zero"] * np.diff(c0) + (1.0 - p["frac_zero"]) * np.diff(c1)


def lf_params(cls, subtype=None, dm15=None, ii_dust=None, iin_lf=None):
    """(media, sigma, min, max) de M en la banda de referencia: la misma eleccion que sampling.sample_mpeak."""
    c = dict(LUMINOSITY_CONFIG["clip"])
    c.update(LUMINOSITY_CONFIG.get("clip_by_class", {}).get(cls, {}))
    if cls == "Ia" and PHILLIPS_CONFIG.get("enabled", False):
        d = dm15 if dm15 is not None and np.isfinite(dm15) else PHILLIPS_CONFIG["dm15_default"]
        m, s = PHILLIPS_CONFIG["M0"] + PHILLIPS_CONFIG["slope"] * (d - PHILLIPS_CONFIG["dm15_ref"]), \
            PHILLIPS_CONFIG["sigma_resid"]
    else:
        mp = LUMINOSITY_CONFIG["M_peak"]
        p = mp[subtype] if subtype in mp and subtype != "Ia" else mp[cls]
        if cls == "IIn" and iin_lf == "nyholm":
            p = mp["IIn_nyholm"]
        if cls == "II" and ii_dust == "sudare" and subtype and subtype + "_dered" in mp:
            p = mp[subtype + "_dered"]
        if "median" in p:
            raise NotImplementedError("LF split-normal (SLSN-I) no implementada")
        m, s = p["mean"], p["sigma"]
    return float(m), float(s), float(c["min"]), float(c["max"])


@dataclass
class Priors:
    clases: tuple
    cls_idx: np.ndarray          # (nt,) indice de la clase, -1 fuera
    log_pi: np.ndarray           # (nt,) log pi(plantilla | clase)
    log_pe: np.ndarray           # (nt, ne) log masa de cada celda de E(B-V)
    lf: np.ndarray               # (nt, 4) media, sigma, min, max de M
    lf_dust: np.ndarray          # (nt,) True: LF con el polvo adentro (LF_AFTER_HOST_DUST)
    sin_z: bool = False


def tpl_prior(tab, four=False, cfg_name=RUN_CFG):
    """log pi(plantilla | clase): tipo dentro de la clase por n_by_class (II 8 : IIb 2), subtipo por SUBTYPE_FRACTIONS
    y plantillas equiprobables dentro del subtipo (run.choose_template). Un subtipo sin plantillas sale de la
    normalizacion (con aviso). -inf fuera de las clases."""
    n_by = runcfg.RUNS_CFG[cfg_name]["n_by_class"]
    out = np.full(len(tab), -np.inf)
    for i, r in enumerate(tab.itertuples()):
        c = D.class_of(r.clase, four)
        if c is None:
            continue
        tipos = [t for t in n_by if D.class_of(t, four) == c and (tab.clase == t).any()]
        p = n_by[r.clase] / sum(n_by[t] for t in tipos)
        mismo = (tab.clase == r.clase).to_numpy()
        fr = SUBTYPE_FRACTIONS.get(r.clase)
        if fr:
            pres = {st: f for st, f in fr.items() if f > 0 and (mismo & (tab.subtype == st).to_numpy()).any()}
            if set(pres) != {st for st, f in fr.items() if f > 0}:
                warnings.warn(f"{r.clase}: subtipos sin plantillas, se renormaliza entre {sorted(pres)}")
            p *= pres.get(r.subtype, 0.0) / sum(pres.values()) / (mismo & (tab.subtype == r.subtype).to_numpy()).sum()
        else:
            p /= mismo.sum()
        out[i] = np.log(p) if p > 0 else -np.inf
    return out


def priors(lib, four=False, sin_z=False, cfg_name=RUN_CFG):
    cfg = runcfg.RUNS_CFG[cfg_name]
    tab = lib.tab
    cls = D.classes(four)
    with np.errstate(divide="ignore"):
        log_pe = np.stack([np.log(ebv_prior(r.clase, r.subtype, lib.ebv, cfg.get("ii_dust"))) for r in tab.itertuples()])
    lf = np.array([lf_params(r.clase, r.subtype, r.dm15_B, cfg.get("ii_dust"), cfg.get("iin_lf")) for r in tab.itertuples()])
    idx = np.array([cls.index(D.class_of(t, four)) if D.class_of(t, four) else -1 for t in tab.clase])
    return Priors(cls, idx, tpl_prior(tab, four, cfg_name), log_pe, lf,
                  tab.clase.isin(LF_AFTER_HOST_DUST).to_numpy(), sin_z)


# ------------------------------------------------------------------------------------------------ distancia
@functools.lru_cache(maxsize=None)
def _tabla_mu():
    lz = np.linspace(math.log(5e-4), math.log(1.0), 1200)
    return lz, np.array([5.0 * math.log10(DL_calculator(math.exp(x)) / 1e-5) for x in lz])


def mu_z(z):
    """Modulo de distancia de core.utils.DL_calculator (el de engine), interpolado en ln z (error < 1e-5 mag)."""
    lz, mt = _tabla_mu()
    return np.interp(np.log(np.asarray(z, float)), lz, mt)


def celdas_z(z_grid):
    g = np.asarray(z_grid, float)
    mid = 0.5 * (g[1:] + g[:-1])
    return np.r_[Z_FLOOR, mid], np.r_[mid, g[-1] + 0.5 * (g[-1] - g[-2])]


def z_nodos(z_grid, z_spec, sin_z=False):
    """Nodos de z con prior: lista de (indice, log masa, mu de las subceldas, pesos normalizados de las subceldas).
    Subceldas uniformes en ln z con dmu <= DMU_SUB. Prior N(z_spec, SIGMA_Z), o plano con sin_z o z_spec no finito."""
    lo, hi = celdas_z(z_grid)
    usa = (not sin_z) and z_spec is not None and np.isfinite(z_spec)
    zs, ws = [], []
    for a, b in zip(lo, hi):
        n = max(2, int(math.ceil(math.log(b / a) / (DMU_SUB * math.log(10) / 5.0))))
        z = np.exp(math.log(a) + (np.arange(n) + 0.5) * math.log(b / a) / n)
        w = z * math.log(b / a) / n
        zs.append(z)
        ws.append(w * np.exp(-0.5 * ((z - z_spec) / SIGMA_Z) ** 2) if usa else w)
    tot = sum(w.sum() for w in ws)
    if not tot > 0:
        return []
    return [(k, math.log(ws[k].sum() / tot), mu_z(zs[k]), ws[k] / ws[k].sum())
            for k in range(len(lo)) if ws[k].sum() / tot > MASA_MIN_Z]


def _tabla_prior_y(mu_s, w_s, lf):
    """log p(y), y = M + mu(z): la LF truncada corrida por las mu de las subceldas del nodo (mezcla), paso 0.01 mag."""
    m, s, a, b = lf
    lo, hi = max(a, m - 7.0 * s), min(b, m + 7.0 * s)
    y = np.arange(mu_s.min() + lo - 0.02, mu_s.max() + hi + 0.02, 0.01)
    M = y[:, None] - mu_s[None, :]
    ln = -0.5 * ((M - m) / s) ** 2 - math.log(s * math.sqrt(2 * math.pi)) - math.log(ndtr((b - m) / s) - ndtr((a - m) / s))
    ln = np.where((M >= a) & (M <= b), ln, -np.inf)
    with np.errstate(divide="ignore"):
        lp = logsumexp(ln + np.log(w_s)[None, :], axis=1)
    return y, np.where(np.isfinite(lp), lp, LOG0)


# ------------------------------------------------------------------------------------------------ biblioteca
def biblioteca_clave(store=None, cfg_name=RUN_CFG, z_grid=Z_GRID, ebv_grid=EBV_GRID, fases=FASES):
    store = Path(store or STORE)
    cfg = runcfg.RUNS_CFG[cfg_name]
    cat = pd.read_csv(store / "catalog.csv")
    rb = B.rest_bands()
    blob = dict(version=VERSION, survey=SURVEY, bandas=list(BANDS), catalogo=md5_file(store / "catalog.csv"),
                filtros={b: md5_file(FILTERS / B.SURVEY_FILES[SURVEY].format(b)) for b in BANDS},
                reposo={n: hashlib.md5(np.concatenate([b.wave, b.resp, [b.f0]]).tobytes()).hexdigest()
                        for n, b in sorted(rb.items())},
                bordes={k: cfg.get(k) for k in BORDES},
                rv={r.sn: ext_params(r.clase, r.subtype, cfg.get("ii_dust"))[1]["Rv"] for r in cat.itertuples()},
                z=[float(x) for x in z_grid], ebv=[float(x) for x in ebv_grid], fases=[*map(float, fases), DP],
                e_mw_ref=E_MW_REF)
    return hashlib.md5(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def _pesos_trapecio(x):
    d = np.diff(x)
    return np.r_[d[0] / 2, (d[1:] + d[:-1]) / 2, d[-1] / 2]


def mags_sin_distancia(tpl, rv, bands, z_grid, ebv_grid, e_mw=E_MW_REF, nz_bloque=8):
    """engine.observed_lightcurves(tpl, z, E, rv, 0, bands, 0) - mu(z) en toda la grilla, por producto matricial.
    Devuelve m (nz, ne, nb, n_ep), cobertura (nz, nb) y m con E(B-V)_MW = e_mw sin polvo del host (nz, nb, n_ep)."""
    w = np.asarray(tpl["wave"], float)
    f = np.asarray(tpl["flux"], np.float64)
    ext = np.stack([engine.extinction_factor(w, rv, e) for e in ebv_grid])         # polvo del host, reposo
    nz, nb, ne = len(z_grid), len(bands), len(ebv_grid)
    m = np.empty((nz, nb, ne + 1, f.shape[0]))
    cov = np.zeros((nz, nb))
    f0 = np.array([b.f0 for b in bands])
    for z0 in range(0, nz, nz_bloque):
        cols = []
        for iz in range(z0, min(z0 + nz_bloque, nz)):
            z = float(z_grid[iz])
            wo = w * (1.0 + z)
            tw = _pesos_trapecio(wo)
            mw = engine.extinction_factor(wo, 3.1, e_mw)
            for ib, b in enumerate(bands):
                ins = (b.wave >= wo[0]) & (b.wave <= wo[-1])
                cov[iz, ib] = float(B.TRAPZ(b.resp[ins], b.wave[ins]) / B.TRAPZ(b.resp, b.wave)) if ins.sum() > 1 else 0.0
                v = np.interp(wo, b.wave, b.resp, left=0.0, right=0.0) * wo * tw / B.TRAPZ(b.resp * b.wave, b.wave) / (1.0 + z)
                cols += [ext * v, (v * mw)[None]]
        F = f @ np.concatenate(cols).T                                                 # (n_ep, cols del bloque)
        n = min(z0 + nz_bloque, nz) - z0
        F = F.T.reshape(n, nb, ne + 1, -1)
        m[z0:z0 + n] = -2.5 * np.log10(np.clip(F, 1e-300, None) / f0[None, :, None, None])
    return m[:, :, :ne].transpose(0, 2, 1, 3), cov, m[:, :, ne]


def pendiente(x, Y):
    """Pendiente de minimos cuadrados (np.polyfit grado 1) de cada fila de Y contra x."""
    xc = x - x.mean()
    return (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)


def tabla_fases(t_rel, m, ph, z, cls, tpl, cfg):
    """m (n_col, n_ep) en t_rel (observado, desde t_peak) -> (n_col, n_fases) en ph con los bordes de cfg, como
    run.simulate + project.project_one. inf = sin flujo."""
    t0, t1 = float(t_rel[0]), float(t_rel[-1])
    out = np.full((m.shape[0], len(ph)), np.inf)
    ins = (ph >= t0) & (ph <= t1)
    for c in range(m.shape[0]):
        out[c, ins] = np.interp(ph[ins], t_rel, m[c])
    pre, post = cfg.get("edge_pre", "window"), cfg.get("edge_post", "none")
    if pre == "fireball" and cls == "Ia":
        t_exp = (tpl["t_Bmax"] - cfg["rise_Ia_days"] - tpl["t_peak"]) * (1.0 + z)
        if t_exp < t0:
            e = (ph >= t_exp) & (ph < t0)
            out[:, e] = m[:, :1] - 5.0 * np.log10(np.maximum((ph[e] - t_exp) / (t0 - t_exp), 1e-12))[None, :]
    if post == "tail" and not tpl["time"][-1] - tpl["time"][0] < cfg.get("tail_min_span", {}).get(cls, 0.0):
        piso = cfg["tail_min_slope"][cls] if isinstance(cfg["tail_min_slope"], dict) else cfg["tail_min_slope"]
        f = t_rel >= t1 - cfg["tail_fit_days"] * (1.0 + z)
        s = pendiente(t_rel[f] - t1, m[:, f]) if f.sum() > 1 else np.full(m.shape[0], np.nan)
        s = np.where(s >= piso, s, piso)                        # nan -> piso, como project_one
        tl = ph > t1
        out[:, tl] = m[:, -1:] + s[:, None] * (ph[tl] - t1)[None, :]
    return out


def construir_biblioteca(store=None, cfg_name=RUN_CFG, out_root=OUT_ROOT, z_grid=Z_GRID, ebv_grid=EBV_GRID,
                         fases=FASES, force=False, verificar=2, seed=SEED, log=print):
    """Tabla de la biblioteca en out_root/biblioteca_<clave> (no la rehace si ya esta). Devuelve el directorio."""
    store, out_root = Path(store or STORE), Path(out_root)
    key = biblioteca_clave(store, cfg_name, z_grid, ebv_grid, fases)
    d = out_root / f"biblioteca_{key}"
    if (d / "meta.json").exists() and not force:
        return d
    t_ini = time.time()
    cfg = runcfg.RUNS_CFG[cfg_name]
    if cfg.get("edge_pre", "window") not in ("window", "fireball") or cfg.get("edge_post", "none") not in ("none", "tail"):
        raise ValueError(f"bordes no implementados: {cfg.get('edge_pre')} / {cfg.get('edge_post')}")
    cat = pd.read_csv(store / "catalog.csv").sort_values(["clase", "sn"]).reset_index(drop=True)
    bands, rest = B.survey_bands(SURVEY, BANDS), B.rest_bands()
    z_grid, ebv_grid = np.asarray(z_grid, float), np.asarray(ebv_grid, float)
    if ebv_grid[0] != 0.0:
        raise ValueError("la grilla de E(B-V) tiene que partir en 0 (R_b de la Via Lactea)")
    ph = np.arange(fases[0], fases[1] + DP / 2, DP)
    nt, nz, ne, nb, npf = len(cat), len(z_grid), len(ebv_grid), len(bands), len(ph)
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    flux = np.lib.format.open_memmap(tmp / "flux.npy", mode="w+", dtype=np.float16, shape=(nt, nz, ne, nb, npf))
    lp, rmw = np.full((nt, nz, ne), np.nan), np.full((nt, nz, nb), np.nan)
    ok, aref = np.zeros((nt, nz, nb), bool), np.zeros((nt, ne))
    rng, dmax, filas = np.random.default_rng(seed), 0.0, []
    for it, r in enumerate(cat.itertuples()):
        tpl = load_template(r.store_path)
        rv = ext_params(r.clase, r.subtype, cfg.get("ii_dust"))[1]["Rv"]
        m, cov, m_mw = mags_sin_distancia(tpl, rv, bands, z_grid, ebv_grid)
        for _ in range(verificar):                              # la tabla es engine.observed_lightcurves - mu(z)
            iz, ie = int(rng.integers(nz)), int(rng.integers(ne))
            _, me = engine.observed_lightcurves(tpl, z_grid[iz], ebv_grid[ie], rv, 0.0, bands, 0.0)
            mu = 5.0 * math.log10(DL_calculator(float(z_grid[iz])) / 1e-5)
            for ib, b in enumerate(bands):
                if (b.name in me) != (cov[iz, ib] > B.COVERAGE_MIN):
                    raise RuntimeError(f"{r.sn}: cobertura de {b.name} distinta a la de engine en z={z_grid[iz]}")
                if b.name in me:
                    k = m[iz, ie, ib] < 50
                    dmax = max(dmax, float(np.max(np.abs(me[b.name][k] - mu - m[iz, ie, ib][k]), initial=0.0)))
        if dmax > 1e-6:
            raise RuntimeError(f"{r.sn}: la tabla difiere de engine.observed_lightcurves en {dmax:.2e} mag")
        ipk = int(np.argmin(np.abs(np.asarray(tpl["time"], float) - float(tpl["t_peak"]))))
        rmw[it] = (m_mw[:, :, ipk] - m[:, 0, :, ipk]) / E_MW_REF
        ok[it] = cov > B.COVERAGE_MIN
        aref[it] = [engine.host_ext_ref(tpl, e, rv, rest[tpl["ref_band"]]) for e in ebv_grid]
        tt = np.asarray(tpl["time"], float) - float(tpl["t_peak"])
        for iz, z in enumerate(z_grid):
            tab = tabla_fases(tt * (1.0 + z), m[iz].reshape(ne * nb, -1), ph, z, r.clase, tpl, cfg).reshape(ne, nb, npf)
            tab[:, ~ok[it, iz]] = np.inf
            if np.isfinite(tab[:, :, 0]).any():
                raise RuntimeError(f"{r.sn}: hay flujo en la primera fase de la grilla ({ph[0]} d), ampliar FASES")
            pk = tab.reshape(ne, -1).min(1)                     # sin banda valida: inf -> flujo 0, lp nan
            lp[it, iz] = np.where(np.isfinite(pk), pk, np.nan)
            flux[it, iz] = 10.0 ** (-0.4 * (tab - np.where(np.isfinite(pk), pk, 0.0)[:, None, None]))
        t_exp = (tpl["t_Bmax"] - cfg.get("rise_Ia_days", np.nan) - tpl["t_peak"]) if r.clase == "Ia" else np.nan
        filas.append(dict(sn=r.sn, clase=r.clase, subtype=r.subtype, M_ref=float(tpl["M_ref"]),
                          dm15_B=np.nan if tpl.get("dm15_B") is None else float(tpl["dm15_B"]), rv=rv,
                          ext_key=ext_params(r.clase, r.subtype, cfg.get("ii_dust"))[0], t_peak=float(tpl["t_peak"]),
                          t0_reposo=float(tt[0]), t1_reposo=float(tt[-1]), t_exp_reposo=float(t_exp),
                          cola=bool(cfg.get("edge_post") == "tail" and not tt[-1] - tt[0] <
                                    cfg.get("tail_min_span", {}).get(r.clase, 0.0)), ref_band=tpl["ref_band"]))
        log(f"[plantillas_clf] biblioteca {it + 1}/{nt} {r.clase} {r.sn} {time.time() - t_ini:.0f}s", flush=True)
    flux.flush()
    del flux
    np.savez(tmp / "arrays.npz", lp=lp, rmw=rmw, ok=ok, aref=aref, z=z_grid, ebv=ebv_grid, ph=ph)
    pd.DataFrame(filas).to_csv(tmp / "plantillas.csv", index=False)
    (tmp / "meta.json").write_text(json.dumps(dict(clave=key, run_cfg=cfg_name, store=str(store), n_plantillas=nt,
                                                   forma=[nt, nz, ne, nb, npf], max_dif_engine_mag=dmax,
                                                   verificadas_por_plantilla=verificar,
                                                   segundos=round(time.time() - t_ini, 1),
                                                   creada=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    return d


@dataclass
class Biblioteca:
    dir: Path
    flux: np.ndarray             # (nt, nz, ne, nb, n_fases) float16, flujo / flujo del pico de la tabla (memmap)
    lp: np.ndarray               # (nt, nz, ne) magnitud sin distancia del pico de la tabla
    rmw: np.ndarray              # (nt, nz, nb) R_b de la Via Lactea
    ok: np.ndarray               # (nt, nz, nb) banda cubierta
    aref: np.ndarray             # (nt, ne) A_ref del host en la banda de referencia de reposo
    z: np.ndarray
    ebv: np.ndarray
    ph: np.ndarray
    tab: pd.DataFrame
    meta: dict = field(default_factory=dict)


def cargar_biblioteca(d):
    d = Path(d)
    a = np.load(d / "arrays.npz")
    return Biblioteca(d, np.load(d / "flux.npy", mmap_mode="r"), *(a[k] for k in ("lp", "rmw", "ok", "aref", "z", "ebv", "ph")),
                      pd.read_csv(d / "plantillas.csv"), json.loads((d / "meta.json").read_text()))


# ------------------------------------------------------------------------------------------------ clasificacion
def ul_usados(cur, det, ul_modo="todos"):
    """Mascara de los UL que entran a la verosimilitud (regla 4b)."""
    ul = cur.ul & np.isfinite(cur.mag)
    if ul_modo == "todos":
        return ul
    if ul_modo == "ninguno":
        return np.zeros_like(ul)
    if ul_modo == "previos":
        return ul & (cur.t < cur.t[det].min())
    if ul_modo == "misma_noche":
        td, bd = cur.t[det], cur.band[det]
        return ul & ~np.array([np.any((np.abs(td - t) < 0.5) & (bd == b)) for t, b in zip(cur.t, cur.band)], bool)
    raise ValueError(f"ul_modo desconocido: {ul_modo}")


def clasificar(cur, lib, pri, mw=0.0, ul_modo="todos"):
    """Una curva (nnclf.data.Curve: t en MJD, band 0 = g / 1 = r, mag, err, ul, z) -> dict con log E y P por clase,
    mejor plantilla y sus parametros MAP. None si tiene menos de MIN_DET detecciones o ningun modelo valido."""
    det = ~cur.ul & np.isfinite(cur.mag) & np.isfinite(cur.err) & (cur.err > 0)
    ul = ul_usados(cur, det, ul_modo) if det.any() else cur.ul
    nd, nu = int(det.sum()), int(ul.sum())
    if nd < MIN_DET:
        return None
    sel = np.r_[np.flatnonzero(det), np.flatnonzero(ul)]
    t, bo = cur.t[sel].astype(float), cur.band[sel].astype(int)
    mag = cur.mag[sel].astype(float)
    m_ref = float(mag[:nd].min())
    f = 10.0 ** (-0.4 * (mag[:nd] - m_ref))
    wd = 1.0 / (K_MAG * f * np.sqrt(cur.err[sel][:nd].astype(float) ** 2 + SIGMA_MOD ** 2)) ** 2
    wf, sff = wd * f, float(np.sum(wd * f * f))
    flim = 10.0 ** (-0.4 * (mag[nd:] - m_ref))
    T0 = float(t[:nd].min()) - T_PRE
    nT = int(math.floor((float(t[:nd].max()) - T0) / DP)) + 1
    q = (t - T0 - lib.ph[0]) / DP
    k0 = np.floor(q).astype(int)
    fr = (q - k0)[:, None]
    npf = len(lib.ph)
    i0 = np.clip(k0[:, None] - np.arange(nT)[None, :], 0, npf - 1)      # fase t - T_max por debajo de la grilla:
    i1 = np.clip(k0[:, None] - np.arange(nT)[None, :] + 1, 0, npf - 1)  # columna 0, sin flujo
    nod = z_nodos(lib.z, cur.z, pri.sin_z)
    bandas = np.unique(bo)
    dU, ltab = U[1] - U[0], {}
    nt = len(lib.tab)
    logE = np.full(nt, -np.inf)
    mapa = [None] * nt
    chi2_min = math.inf
    for it in range(nt):
        if pri.cls_idx[it] < 0 or not nod:
            continue
        ie = np.flatnonzero(pri.log_pe[it] > math.log(MASA_MIN_E))
        ns = [n for n in nod if lib.ok[it, n[0]][bandas].all()]
        if not ns or not len(ie):
            continue
        lf, Mref = tuple(pri.lf[it]), float(lib.tab.M_ref.iloc[it])
        aref = lib.aref[it, ie] * pri.lf_dust[it]                                  # (ne,)
        lpe = pri.log_pe[it, ie]
        per = max(1, CHUNK // (len(ie) * nT * max(len(bo), len(U))))
        acc, medias, tope = [], [], -math.inf
        for c0 in range(0, len(ns), per):
            ch = ns[c0:c0 + per]
            ks = [n[0] for n in ch]
            sub = np.asarray(lib.flux[it][ks][:, ie], np.float64)                  # (nk, ne, nb, n_fases)
            G0 = sub[:, :, bo[:, None], i0]
            F = G0 + fr * (sub[:, :, bo[:, None], i1] - G0)                         # (nk, ne, n_obs, nT)
            del G0
            if mw:
                F *= (10.0 ** (-0.4 * lib.rmw[it][ks][:, bo] * mw))[:, None, :, None]
            Fd = F[:, :, :nd]
            SFF, SfF = np.matmul(wd, Fd * Fd), np.matmul(wf, Fd)                    # (nk, ne, nT)
            Lp = lib.lp[it][ks][:, ie]                                              # (nk, ne)
            mu_m = np.array([np.sum(n[3] * n[2]) for n in ch])
            mu_v = np.array([np.sum(n[3] * (n[2] - np.sum(n[3] * n[2])) ** 2) for n in ch])
            mp = mu_m[:, None] + lf[0] - Mref - aref[None, :]                       # prior de dmag + mu, gaussiano
            vp = (lf[1] ** 2 + mu_v)[:, None, None]
            good = (SFF > 0) & (SfF > 0)
            Ah = np.where(good, SfF / np.where(good, SFF, 1.0), 1.0)
            chi2_min = min(chi2_min, float(np.min(np.where(good, sff - SfF * Ah, sff))))
            pd_ = np.where(good, (Ah / 1.0857) ** 2 * SFF, 0.0)                     # precision de la verosimilitud en d
            dh = m_ref - Lp[:, :, None] - 2.5 * np.log10(Ah)
            prec = pd_ + 1.0 / vp
            dc = (pd_ * dh + mp[:, :, None] / vp) / prec
            sc = 1.0 / np.sqrt(prec)
            Ac = 10.0 ** (-0.4 * (Lp[:, :, None] + dc - m_ref))                    # escala en el centro
            yc = dc + Mref + aref[None, :, None]                                   # y = M + mu(z)
            lpc = np.empty_like(dc)
            for j, n in enumerate(ch):
                kk = (lf, n[0])
                if kk not in ltab:
                    ltab[kk] = _tabla_prior_y(n[2], n[3], lf)
                lpc[j] = np.interp(yc[j], *ltab[kk], left=LOG0, right=LOG0)
            lul = log_ndtr(UL_NSIG * (1.0 - Ac[:, :, None, :] * F[:, :, nd:] / flim[None, None, :, None])).sum(2) \
                if nu else np.zeros_like(dc)
            base = lul + np.array([n[1] for n in ch])[:, None, None] + lpe[None, :, None] - math.log(nT)
            # Laplace en todas las celdas; la integral en dmag completa solo donde puede pesar
            lE = -0.5 * (sff - 2.0 * Ac * SfF + Ac * Ac * SFF) + lpc + np.log(SQ2PI * sc) + base
            tope = max(tope, float(lE.max()))
            ix = np.nonzero(lE > tope - LAPLACE_NATS)
            Dd = dc[ix][:, None] + sc[ix][:, None] * U                              # (n_sel, nU)
            A = np.exp(-K_MAG * (Lp[ix[0], ix[1]][:, None] + Dd - m_ref))
            ll = -0.5 * (sff - 2.0 * A * SfF[ix][:, None] + A * A * SFF[ix][:, None])
            y = Dd + Mref + aref[ix[1]][:, None]
            for j, n in enumerate(ch):
                r_ = ix[0] == j
                if r_.any():
                    ll[r_] += np.interp(y[r_], *ltab[(lf, n[0])], left=LOG0, right=LOG0)
            lE[ix] = logsumexp(ll, axis=-1) + np.log(sc[ix] * dU) + base[ix]
            j, e, mT = np.unravel_index(int(np.argmax(lE)), lE.shape)
            lse = logsumexp(lE)
            if np.isfinite(lse):                                   # medias posteriores dentro del bloque
                wq = np.exp(lE - lse)
                Mq = dc - mu_m[:, None, None] + Mref + aref[None, :, None]
                acc.append(lse)
                medias.append(dict(z_post=np.sum(wq * lib.z[ks][:, None, None]),
                                   ebv_post=np.sum(wq * lib.ebv[ie][None, :, None]),
                                   tmax_post=T0 + DP * np.sum(wq * np.arange(nT)[None, None, :]), M_post=np.sum(wq * Mq)))
            if mapa[it] is None or lE[j, e, mT] > mapa[it]["lE"]:
                mapa[it] = dict(lE=float(lE[j, e, mT]), z_map=float(lib.z[ks[j]]), ebv_map=float(lib.ebv[ie[e]]),
                                tmax_map=T0 + mT * DP,
                                M_map=float(dc[j, e, mT] - mu_m[j] + Mref + aref[e]),
                                chi2_map=float(sff - 2 * Ac[j, e, mT] * SfF[j, e, mT] + Ac[j, e, mT] ** 2 * SFF[j, e, mT]),
                                chi2_ul_map=float(-2.0 * lul[j, e, mT]))
        if acc:
            logE[it] = logsumexp(acc)
            pw = np.exp(np.array(acc) - logE[it])
            mapa[it].update({k: float(sum(w_ * m_[k] for w_, m_ in zip(pw, medias))) for k in medias[0]})
    lt = logE + pri.log_pi
    if not np.isfinite(lt).any():
        return None
    K = len(pri.clases)
    with np.errstate(divide="ignore"):
        lc = np.array([logsumexp(lt[pri.cls_idx == c]) if (pri.cls_idx == c).any() else -np.inf for c in range(K)])
    p = np.exp(lc - logsumexp(lc))
    ib = int(np.argmax(lt))
    out = dict(n_det=nd, n_ul=nu, p=p, logE=lc, best_template=lib.tab.sn.iloc[ib], best_template_clase=lib.tab.clase.iloc[ib],
               chi2_min=chi2_min, prior_z="plano" if (pri.sin_z or not np.isfinite(cur.z)) else "spec",
               logE_plantillas=lt)
    out.update({k: v for k, v in mapa[ib].items() if k != "lE"})
    return out


# ------------------------------------------------------------------------------------------------ corrida
_W = {}


def _init(lib_dir, four, sin_z, cfg_name, ul_modo="todos"):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    lib = cargar_biblioteca(lib_dir)
    _W.update(lib=lib, pri=priors(lib, four, sin_z, cfg_name), ul_modo=ul_modo)


def _uno(args):
    cur, mw = args
    t0 = time.time()
    r = clasificar(cur, _W["lib"], _W["pri"], mw, _W["ul_modo"])
    return cur.key, r, time.time() - t0


def val_meta(real_dir=REAL_DIR, four=False):
    """Metadatos val con subset (val_sel/val_rep) y la clase: clf_villar.read_val_meta (solo filas val)."""
    from pipeline78.clf_villar import read_val_meta
    return read_val_meta(Path(real_dir) / "meta_real_ztf.csv", four)


def elegir_oids(v, subset="val", limit=None, seed=SEED):
    """Oids del subconjunto pedido; con limit, una muestra fija (permutacion con semilla) para las pruebas chicas."""
    o = sorted(v.oid if subset == "val" else v.oid[v.subset == subset])
    if limit:
        o = sorted(np.random.default_rng(seed).permutation(np.array(o, dtype=object))[:limit])
    return o


def run(name, four=False, sin_z=False, subset="val", limit=None, workers=2, real_dir=REAL_DIR, out_root=OUT_ROOT,
        lib_dir=None, store=None, mw=None, cfg_name=RUN_CFG, n_boot=1000, ul_modo="todos"):
    from pipeline78.clf_villar import bootstrap_ci, calib_metrics, metrics
    t_ini = time.time()
    out_root = Path(out_root)
    lib_dir = Path(lib_dir) if lib_dir else construir_biblioteca(store, cfg_name, out_root)
    t_lib = time.time() - t_ini
    cls = D.classes(four)
    v = val_meta(real_dir, four)
    oids = set(elegir_oids(v, subset, limit))
    v = v[v.oid.isin(oids)].reset_index(drop=True)
    curves, faltan = D.load_real_val(real_dir, four_classes=four, min_det=MIN_DET)
    curves = [c for c in curves if c.key in oids]
    if mw is None:
        mw = sampling.load_mw(dict(mw_mode="ztf_sfd"))
    mw_const = runcfg.RUNS_CFG[cfg_name].get("mw_const", 0.02)
    sin_mw = sorted(c.key for c in curves if c.key not in mw)
    tareas = [(c, float(mw.get(c.key, mw_const))) for c in curves]
    workers = max(1, min(int(workers), 4))
    t1 = time.time()
    if workers == 1:
        _init(lib_dir, four, sin_z, cfg_name, ul_modo)
        res = [_uno(x) for x in tareas]
    else:
        with get_context("spawn").Pool(workers, initializer=_init,
                                       initargs=(lib_dir, four, sin_z, cfg_name, ul_modo)) as pool:
            res = []
            for i, r in enumerate(pool.imap(_uno, tareas, chunksize=1), 1):
                res.append(r)
                if i % 25 == 0 or i == len(tareas):
                    print(f"[plantillas_clf] {i}/{len(tareas)} {time.time() - t1:.0f}s", flush=True)
    t_clf = time.time() - t1
    info = v.set_index("oid")
    filas, nada = [], []
    for (oid, r, dt), (c, m) in zip(res, tareas):
        if r is None:
            nada.append(dict(oid=oid, motivo="sin modelo valido"))
            continue
        k = int(np.argmax(r["p"]))
        filas.append(dict(oid=oid, subset=info.subset[oid], sn_type=info.sn_type[oid], y_true=info.cls[oid],
                          y_pred=cls[k], **{f"p_{c_}": float(r["p"][i]) for i, c_ in enumerate(cls)},
                          n_det=r["n_det"], n_ul=r["n_ul"], z=c.z, prior_z=r["prior_z"], ebv_mw=m,
                          best_template=r["best_template"], best_template_clase=r["best_template_clase"],
                          chi2_min=r["chi2_min"], chi2_map=r["chi2_map"], chi2_ul_map=r["chi2_ul_map"],
                          z_map=r["z_map"], ebv_map=r["ebv_map"], tmax_map=r["tmax_map"], M_map=r["M_map"],
                          z_post=r["z_post"], ebv_post=r["ebv_post"], tmax_post=r["tmax_post"], M_post=r["M_post"],
                          **{f"logE_{c_}": float(r["logE"][i]) for i, c_ in enumerate(cls)}, t_seg=dt))
    nada += [dict(oid=o, motivo=f"menos de {MIN_DET} detecciones g + r") for o in faltan if o in oids]
    P = pd.DataFrame(filas, columns=["oid", "subset", "sn_type", "y_true", "y_pred"] + [f"p_{c}" for c in cls] +
                     ["n_det", "n_ul", "z", "prior_z", "ebv_mw", "best_template", "best_template_clase", "chi2_min",
                      "chi2_map", "chi2_ul_map", "z_map", "ebv_map", "tmax_map", "M_map", "z_post", "ebv_post",
                      "tmax_post", "M_post"] +
                     [f"logE_{c}" for c in cls] + ["t_seg"])
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "pred_real_val.csv", index=False)
    S = pd.DataFrame(nada, columns=["oid", "motivo"])
    S.assign(subset=S.oid.map(info.subset), sn_type=S.oid.map(info.sn_type)).to_csv(out / "sin_clasificar.csv", index=False)
    real, cob = {}, {}
    for s in SUBSETS:
        vs, ps = (v, P) if s == "val" else (v[v.subset == s], P[P.subset == s])
        cob[s] = {"n_real": int(len(vs)), "n_clasificadas": int(len(ps)), "cobertura": float(len(ps) / max(len(vs), 1)),
                  "cobertura_por_clase": {c: float((ps.y_true == c).sum() / max((vs.cls == c).sum(), 1)) for c in cls}}
        if not len(ps):
            real[s] = {"n": 0}
            continue
        y = ps.y_true.map(cls.index).to_numpy()
        yp = ps.y_pred.map(cls.index).to_numpy()
        r = metrics(y, yp, cls)
        r.update(calib_metrics(ps[[f"p_{c}" for c in cls]].to_numpy(), y))
        r.update(bootstrap_ci(y, yp, cls, n_boot, SEED))
        real[s] = r
    ts = P.t_seg.to_numpy() if len(P) else np.zeros(1)
    cfg_txt = dict(run_cfg=cfg_name, cuatro_clases=four, sin_z=sin_z, subset=subset,
                   limit=limit, ul_modo=ul_modo, laplace_nats=LAPLACE_NATS, sigma_z=SIGMA_Z, z_floor=Z_FLOOR, sigma_mod=SIGMA_MOD, ul_nsig=UL_NSIG, t_pre=T_PRE,
                   min_det=MIN_DET, n_u=len(U), u_max=float(U[-1]), dmu_sub=DMU_SUB, masa_min_z=MASA_MIN_Z,
                   masa_min_e=MASA_MIN_E, mw_const=mw_const, n_sin_mw=len(sin_mw))
    res_json = dict(config=cfg_txt, clases=list(cls), biblioteca=str(lib_dir),
                    biblioteca_meta=json.loads((Path(lib_dir) / "meta.json").read_text()), real=real, cobertura=cob,
                    tiempo=dict(total_s=round(time.time() - t_ini, 1), biblioteca_s=round(t_lib, 1),
                                clasificacion_s=round(t_clf, 1), workers=workers, n_sn=len(tareas),
                                por_sn_mediana_s=float(np.median(ts)), por_sn_p90_s=float(np.percentile(ts, 90)),
                                por_sn_max_s=float(ts.max())),
                    creada=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "metrics.json").write_text(json.dumps(res_json, indent=1, default=float))
    rep = real.get("val_rep" if subset in ("val", "val_rep") else "val_sel", {})
    print(f"[plantillas_clf] {name}: {len(P)}/{len(v)} clasificadas en {t_clf:.0f}s ({workers} procesos, mediana "
          f"{np.median(ts):.2f}s por SN) | {subset} n={rep.get('n')} bal_acc={rep.get('bal_acc', float('nan')):.3f} "
          f"-> {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline78.plantillas_clf")
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("biblioteca")
    b.add_argument("--force", action="store_true")
    r = sp.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--cuatro-clases", action="store_true")
    r.add_argument("--sin-z", action="store_true")
    r.add_argument("--subset", choices=("val", "val_sel", "val_rep"), default="val")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--workers", type=int, default=2, help="maximo 4")
    r.add_argument("--ul", choices=UL_MODOS, default="todos", help="UL en la verosimilitud (regla 4b)")
    a = ap.parse_args(argv)
    if a.cmd == "biblioteca":
        from threadpoolctl import threadpool_limits
        with threadpool_limits(int(os.environ.get("P78_PL_THREADS", "2"))):
            print(construir_biblioteca(force=a.force))
        return
    run(a.name, a.cuatro_clases, a.sin_z, a.subset, a.limit, a.workers, ul_modo=a.ul)


if __name__ == "__main__":
    main()
