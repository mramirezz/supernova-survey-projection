"""Clasificador por ajuste bayesiano de plantillas, como SUDARE I (Cappellaro et al. 2015, 2015A&A...584A..62C, Sec.
4.1), que sigue a PSNID (Sako et al. 2011, 2011ApJ...738..162S). Tercer metodo (Mauricio, 2026-10-04), junto a Villar
(pipeline78.clf_villar) y las redes (pipeline78.nnclf). Ventaja nuestra: las plantillas son series de espectros, asi
que la curva observada a cualquier z es fotometria sintetica exacta (sin tablas de correccion K ni banda vecina).

Uso:
    python -m pipeline78.plantillas_clf biblioteca [--force]
    python -m pipeline78.plantillas_clf sigma [--workers 4]            (regla 4c, solo val_sel)
    python -m pipeline78.plantillas_clf run --name NOMBRE [--cuatro-clases] [--sin-z] [--subset val_sel] [--limit N]
                                            [--ul previos] [--sigma-mod S] [--ii-dust cfg|sudare]
    python -m pipeline78.plantillas_clf comparar --name NOMBRE [--contra k=ruta ...] [--tag T]

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
   gaussianas de varianza (0.921 f_obs magerr)^2 + (SIGMA_MOD f_modelo)^2: el error de los datos mas un error del
   modelo, fraccion SIGMA_MOD del flujo del modelo, en cuadratura (regla 4c). Como la varianza depende del modelo, la
   verosimilitud lleva su log det y su cola hacia escalas grandes no es gaussiana (el error del modelo crece con el
   modelo): ver la regla 5. UL: P(f_modelo < f_lim) = Phi((f_lim - f_modelo)/sigma), sigma^2 = (f_lim/5)^2 +
   (SIGMA_MOD f_modelo)^2 (alertas de ZTF a 5 sigma), evaluado en el modo en d de cada (plantilla, z, E, T_max)
   (aproximacion: la escala la fijan las detecciones).
4b. Que UL entran (--ul). Por defecto "previos" (Mauricio 2026-10-04, decision fisica a priori, no ajustada): solo los
   UL antes de la primera deteccion, que acotan la explosion. Los UL despues de la primera deteccion (noches con la SN
   ya detectada) no entran: las alertas reales traen UL en la misma noche y banda que una deteccion (74 de 224 curvas de
   val_sel, 358 de 6290 UL), a veces 4 mag mas profundos que la SN detectada (restas fallidas que las sims no tienen).
   "todos", "misma_noche" (saca los UL con una deteccion de su banda a menos de 0.5 d) y "ninguno" quedan solo como
   sensibilidad reportada en val_sel. Nunca se eligen con val_rep.
4c. SIGMA_MOD (Mauricio 2026-10-04): las series tienen incertidumbres propias que no se propagaron y la diversidad
   real va mas alla de las 78 plantillas. SIGMA_MOD se elige en val_sel (3 clases con z, UL previos, polvo principal)
   entre SIGMA_MOD_GRID = 0.05, 0.10, 0.15, 0.20, 0.30 por la log-verosimilitud de la clase verdadera, media por clase
   (log p con piso 1e-12, como calib_metrics, balanceada como la exactitud con que se elige): la candidata de mayor
   valor reemplaza al a priori SIGMA_MOD_APRIORI = 0.05 solo si el bootstrap pareado de nnclf (paired_bootstrap sobre
   log p por objeto) da P >= P_MIN. Se reportan tambien la exactitud balanceada, el ECE y el chi2 reducido de los mejores
   ajustes por valor (sigma/eleccion.json). El valor queda congelado en SIGMA_MOD antes de mirar val_rep.
5. Evidencia: E_tipo = sum_pl pi(pl) sum_{z,E,T} P(z) P(E) P(T) int L(d) P(d) dd, d = mu + dmag. Por bloque de celdas
   (z, E, T_max) de una plantilla:
   etapa 1 (todas): verosimilitud exacta en el punto de partida (minimos cuadrados con pesos fijos + prior gaussiano de
         d) y en la media del prior (segundo modo posible: con el error del modelo la verosimilitud se aplana a escalas
         grandes y el prior fija el modo), Laplace con el ancho de cada uno. Las celdas a mas de CRIBA_NATS = 150 del
         maximo de la plantilla quedan con ese valor.
   etapa 2 (las demas): Newton con radio de confianza (derivadas analiticas) desde el punto de partida, la media del
         prior y el mejor punto de una grilla gruesa del prior (+-4 sigma); los modos a menos de 4 anchos se juntan.
         Laplace en cada modo. Las celdas a menos de LAPLACE_NATS = 30 del maximo: integral en d de cada modo en su
         tramo (cortes entre modos vecinos), nodos d* + w U (paso w/2, +-8 w), extendida con el mismo paso mientras el
         borde quede a menos de BORDE_NATS del maximo.
   Verificado contra la integral por fuerza bruta (paso 0.002 mag en +-25 mag) en celdas al azar con uno y dos modos
   (tests): error < 0.02 nats, Laplace < 0.2 nats. La criba de la etapa 1 solo cambia plantillas que quedan > 80 nats
   bajo la mejor (P igual): test. P_tipo = E_tipo/sum E.
6. Reales: solo la mitad val, con los lectores con guarda (nnclf.data.load_real_val: meta con csv linea a linea,
   pyarrow con filtro por oid; clf_villar.read_val_meta para el subset de splits.val_split). La mitad final no se lee.
   Entran las de >= MIN_DET detecciones en g + r. E(B-V)_MW de data/sfd98_cache.parquet (sampling.load_mw, el de las
   sims; una oid que falta usa mw_const, como run.run_unit).
7. Nada se ajusta con val_rep: SIGMA_Z, el sigma de los UL, T_PRE, U, las grillas y el modo de UL son a priori;
   SIGMA_MOD sale de val_sel (regla 4c). Polvo de las II: el modelo principal es el de RUN_CFG (IIP/IIL sin polvo del
   host, IIb e IIn con polvo); --ii-dust sudare (half-normal sigma_E 0.2, R_V 3.1, LF IIP/IIL desenrojecida) es una
   sensibilidad, comparada con el principal con el bootstrap pareado (comparar).
8. Salidas en RUNS/plantillas_clf/<nombre>: pred_real_val.csv (oid, subset, y_true, y_pred, p_<clase>, n_det,
   best_template, chi2_min, parametros MAP y medias posteriores dentro de la mejor plantilla, chi2 del MAP con y sin el
   error del modelo), metrics.json (acc, bal_acc con IC 95 % bootstrap, F1 por clase, confusion, log-loss, ECE, chi2
   reducido, cobertura por subconjunto, tiempos), sin_clasificar.csv y comparacion*.json (comparar).
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
SIGMA_MOD_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)   # error del modelo (fraccion del flujo del modelo): candidatos
SIGMA_MOD_APRIORI = 0.05         # el piso a priori de la primera version (referencia del bootstrap pareado)
SIGMA_MOD = 0.05                 # elegido en val_sel (regla 4c): ver sigma/eleccion.json
UL_MODO = "previos"              # regla 4b (Mauricio 2026-10-04): solo los UL antes de la primera deteccion
UL_NSIG = 5.0                    # alertas de ZTF: limite a 5 sigma
T_PRE = 60.0                     # d: el pico puede caer hasta 60 d antes de la primera deteccion
MIN_DET = D.MIN_DET              # 3 detecciones en g + r, como las redes
U = np.linspace(-8.0, 8.0, 33)   # nodos de la integral en d, en unidades del ancho en el modo
LAPLACE_NATS = 30.0              # celdas con Laplace mas de 30 nats bajo el maximo de la plantilla: quedan con Laplace
N_NEWTON = 10                    # iteraciones de Newton para el modo en d
PASO_MAX = 1.0                   # mag: radio de confianza maximo de Newton
CRIBA_NATS = 150.0               # etapa 1 mas de esto bajo el maximo de la plantilla: sin Newton
BORDE_NATS = 20.0                # integrando en un borde a menos de esto de su maximo: se dobla el ancho
N_ENSANCHE = 4
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


def priors(lib, four=False, sin_z=False, cfg_name=RUN_CFG, ii_dust="cfg"):
    """Priors de las sims de cfg_name. ii_dust: "cfg" = el de la config (modelo principal), o la variante ("sudare")."""
    cfg = runcfg.RUNS_CFG[cfg_name]
    ii = cfg.get("ii_dust") if ii_dust == "cfg" else ii_dust
    rv = np.array([ext_params(r.clase, r.subtype, ii)[1]["Rv"] for r in lib.tab.itertuples()])
    if not np.allclose(rv, lib.tab.rv):
        raise ValueError(f"el R_V del polvo ii_dust={ii} no es el de la biblioteca: construirla con ese ii_dust")
    tab = lib.tab
    cls = D.classes(four)
    with np.errstate(divide="ignore"):
        log_pe = np.stack([np.log(ebv_prior(r.clase, r.subtype, lib.ebv, ii)) for r in tab.itertuples()])
    lf = np.array([lf_params(r.clase, r.subtype, r.dm15_B, ii, cfg.get("iin_lf")) for r in tab.itertuples()])
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
def biblioteca_clave(store=None, cfg_name=RUN_CFG, z_grid=Z_GRID, ebv_grid=EBV_GRID, fases=FASES, ii_dust="cfg"):
    store = Path(store or STORE)
    cfg = runcfg.RUNS_CFG[cfg_name]
    ii = cfg.get("ii_dust") if ii_dust == "cfg" else ii_dust
    cat = pd.read_csv(store / "catalog.csv")
    rb = B.rest_bands()
    blob = dict(version=VERSION, survey=SURVEY, bandas=list(BANDS), catalogo=md5_file(store / "catalog.csv"),
                filtros={b: md5_file(FILTERS / B.SURVEY_FILES[SURVEY].format(b)) for b in BANDS},
                reposo={n: hashlib.md5(np.concatenate([b.wave, b.resp, [b.f0]]).tobytes()).hexdigest()
                        for n, b in sorted(rb.items())},
                bordes={k: cfg.get(k) for k in BORDES},
                rv={r.sn: ext_params(r.clase, r.subtype, ii)[1]["Rv"] for r in cat.itertuples()},
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
                         fases=FASES, force=False, verificar=2, seed=SEED, log=print, ii_dust="cfg"):
    """Tabla de la biblioteca en out_root/biblioteca_<clave> (no la rehace si ya esta). Devuelve el directorio.
    ii_dust solo entra por el R_V de las II (la variante "sudare" tiene el mismo R_V 3.1: misma biblioteca)."""
    store, out_root = Path(store or STORE), Path(out_root)
    key = biblioteca_clave(store, cfg_name, z_grid, ebv_grid, fases, ii_dust)
    d = out_root / f"biblioteca_{key}"
    if (d / "meta.json").exists() and not force:
        return d
    t_ini = time.time()
    cfg = runcfg.RUNS_CFG[cfg_name]
    ii = cfg.get("ii_dust") if ii_dust == "cfg" else ii_dust
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
        rv = ext_params(r.clase, r.subtype, ii)[1]["Rv"]
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
                          ext_key=ext_params(r.clase, r.subtype, ii)[0], t_peak=float(tpl["t_peak"]),
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


def _centro(SFF, SfF, q0, mp, vp):
    """Punto de partida en d (= mu + dmag): combinacion gaussiana de la verosimilitud con pesos fijos (cuadratica en la
    escala A: SFF = sum w F^2, SfF = sum w f F) y el prior gaussiano de d (media mp, varianza vp). q0 = Lp - m_ref.
    Devuelve (centro, ancho). Sin flujo del modelo en las detecciones: el prior solo."""
    good = (SFF > 0) & (SfF > 0)
    Ah = np.where(good, SfF / np.where(good, SFF, 1.0), 1.0)              # escala de minimos cuadrados
    pd_ = np.where(good, (Ah * K_MAG) ** 2 * SFF, 0.0)                     # precision de la verosimilitud en d
    prec = pd_ + 1.0 / vp
    return (pd_ * (-q0[:, :, None] - 2.5 * np.log10(Ah)) + mp / vp) / prec, 1.0 / np.sqrt(prec)


def _verosim(fo, so, s2m, a, deriv=False):
    """Verosimilitud de las detecciones (eje -2 = puntos) con el modelo a (flujo): gaussianas de varianza
    V = so + s2m a^2 (datos + error del modelo, fraccion s2m^0.5 del flujo del modelo). Devuelve (log L, chi2) y con
    deriv tambien S1 = sum a phi' y S2 = sum a (phi' + a phi''), phi(a) = -(r^2/V + log V)/2 de un punto, r = f - a:
    d log L/dd = -K S1 y d2 log L/dd2 = K^2 S2 (a = A F, dA/dd = -K A)."""
    V = so + s2m * a * a
    r = fo - a
    iV = 1.0 / V
    q = r * r * iV
    chi2 = q.sum(-2)
    ll = -0.5 * (chi2 + np.log(V).sum(-2))
    if not deriv:
        return ll, chi2
    t = s2m * a * iV
    p1 = r * iV + t * (q - 1.0)
    p2 = (s2m * (q - 1.0) - 1.0 - 4.0 * t * r + 2.0 * t * s2m * a * (1.0 - 2.0 * q)) * iV
    return ll, chi2, (a * p1).sum(-2), (a * (p1 + a * p2)).sum(-2)


def _modo(fo, so, s2m, Fc, q0, mp, vp, d0, n_it=None):
    """Modo local en d de g(d) = log L(d) + log N(d; mp, vp) de cada celda: Newton con radio de confianza desde d0 (un
    paso que empeora g se rehace desde el mejor punto con el radio / 4, uno que mejora lo duplica hasta PASO_MAX). Fc
    (nd, n) forma del modelo en las detecciones, q0 = Lp - m_ref, mp, vp, d0 (n,). Devuelve (d*, ancho 1/sqrt(-g'') en
    d*, log L en d*, chi2 en d*). Sin curvatura negativa en d*: el ancho del prior."""
    tr = np.full(d0.shape, PASO_MAX)
    d, b = d0, None
    for _ in range(N_NEWTON if n_it is None else n_it):
        ll, chi2, s1, s2 = _verosim(fo, so, s2m, np.exp(-K_MAG * (q0 + d)) * Fc, True)
        x = (ll - 0.5 * (d - mp) ** 2 / vp, d, ll, chi2, -K_MAG * s1 - (d - mp) / vp, K_MAG ** 2 * s2 - 1.0 / vp)
        if b is None:
            b = x
        else:
            m = x[0] >= b[0]
            tr = np.where(m, np.minimum(2.0 * tr, PASO_MAX), 0.25 * tr)
            b = tuple(np.where(m, u, v) for u, v in zip(x, b))
        g1, g2 = b[4], b[5]
        st = np.where(g2 < 0, -g1 / np.where(g2 < 0, g2, -1.0), np.sign(g1) * tr)
        d = b[1] + np.clip(st, -tr, tr)
    return b[1], 1.0 / np.sqrt(np.where(b[5] < 0, -b[5], 1.0 / vp)), b[2], b[3]


def _integral_d(fo, so, s2m, Fq, q0, dm, sm, lprior, lo=None, hi=None):
    """log int_lo^hi L(d) p(d) dd de n celdas: nodos dm + sm U (paso sm/2). Un lado cuyo ultimo nodo queda a menos de
    BORDE_NATS del maximo se extiende con el mismo paso, 16 sm por vez (hasta N_ENSANCHE veces): la cola de escalas
    grandes no es gaussiana (el error del modelo crece con el modelo). lprior(i, D) = log p en los nodos D de las
    celdas i. Fq (n, nd), q0, dm, sm, lo, hi (n,)."""
    n, du = len(dm), U[1] - U[0]
    lo = np.full(n, -np.inf) if lo is None else lo
    hi = np.full(n, np.inf) if hi is None else hi
    acc, mx, borde = np.full(n, -np.inf), np.full(n, -np.inf), {}
    nb = max(1, CHUNK // (len(U) * max(Fq.shape[1], 1)))

    def suma(i, Uk):
        for b0 in range(0, len(i), nb):
            j = i[b0:b0 + nb]
            D_ = dm[j][:, None] + sm[j][:, None] * Uk                                  # (n, nU)
            a = np.exp(-K_MAG * (q0[j][:, None] + D_))[:, None, :] * Fq[j][:, :, None]  # (n, nd, nU)
            lw = np.where((D_ >= lo[j][:, None]) & (D_ < hi[j][:, None]), _verosim(fo, so, s2m, a)[0] + lprior(j, D_),
                          -np.inf)
            acc[j] = np.logaddexp(acc[j], logsumexp(lw, axis=-1))
            mx[j] = np.maximum(mx[j], lw.max(1))
            borde[-1][j], borde[1][j] = lw[:, 0], lw[:, -1]
    borde = {-1: np.full(n, -np.inf), 1: np.full(n, -np.inf)}
    suma(np.arange(n), U)
    for lado in (-1, 1):
        ext = lado * (U[-1] + du * np.arange(1, len(U)))                            # 16 sm por extension
        ext = ext if lado > 0 else ext[::-1]
        for _ in range(N_ENSANCHE):
            i = np.flatnonzero(borde[lado] > mx - BORDE_NATS)
            if not len(i):
                break
            otro = borde[-lado].copy()
            suma(i, ext)
            borde[-lado] = otro
            ext = ext + lado * (ext.max() - ext.min() + du)
    return acc + np.log(sm * du)


def _posterior_d(fo, so, s2m, Fc, q0, mp, vp, d0, lpri, ulf):
    """Modos de la posterior en d de n celdas: Newton desde el punto de partida d0, desde la media del prior y desde el
    mejor punto de una grilla gruesa del prior (mp +- 4 sigma): con el error del modelo la verosimilitud puede tener
    mas de un modo (escala de minimos cuadrados y escalas grandes donde se aplana). Modos a menos de 4 anchos se juntan
    (queda el mejor). Devuelve (D, S, L, C, Ul, V) de forma (n, 3) ordenados en d: modo, ancho, Laplace (sin las
    constantes de la celda), chi2 y log P de los UL en el modo, y V = modo distinto (los demas no cuentan).
    lpri(D) = log p exacto en d de cada celda (D (n, k)), ulf(d) = log P de los UL con escala en d."""
    G = mp[:, None] + np.sqrt(vp)[:, None] * np.linspace(-4.0, 4.0, 17)
    gg = np.stack([_verosim(fo, so, s2m, np.exp(-K_MAG * (q0 + G[:, k])) * Fc)[0] for k in range(G.shape[1])], 1) \
        - 0.5 * (G - mp[:, None]) ** 2 / vp[:, None]
    res = []
    for ini in (d0, mp.copy(), G[np.arange(len(mp)), np.argmax(gg, 1)]):
        dk, sk, llk, chk = _modo(fo, so, s2m, Fc, q0, mp, vp, ini)
        ulk = ulf(dk)
        res.append((dk, sk, llk + lpri(dk[:, None])[:, 0] + np.log(SQ2PI * sk) + ulk, chk, ulk))
    D, S, L, C, Ul = (np.stack([r[k] for r in res], 1) for k in range(5))
    o = np.argsort(D, 1, kind="stable")
    D, S, L, C, Ul = (np.take_along_axis(x, o, 1) for x in (D, S, L, C, Ul))
    V = np.ones(D.shape, bool)
    n_ = np.arange(len(D))
    cur = np.zeros(len(D), int)                                    # ultimo modo que queda
    for k in (1, 2):
        cerca = np.abs(D[:, k] - D[n_, cur]) <= 4.0 * np.maximum(S[:, k], S[n_, cur])
        mejor = L[:, k] > L[n_, cur]
        V[n_[cerca & mejor], cur[cerca & mejor]] = False
        V[cerca & ~mejor, k] = False
        cur = np.where(cerca & ~mejor, cur, k)
    return D, S, L, C, Ul, V


def _integral_modos(fo, so, s2m, Fq, q0, D, S, V, Ul, lpr):
    """log int L(d) p(d) P(UL | d) dd de n celdas con los modos de _posterior_d: cada modo que queda (V) en su tramo,
    con cortes entre modos vecinos (mas cerca del angosto), y los UL en su modo. lpr(i, D) = log p de las celdas i."""
    lo, hi = np.full(D.shape, -np.inf), np.full(D.shape, np.inf)
    for a_ in range(3):
        for b_ in range(a_ + 1, 3):                                # b_ = el siguiente modo que queda despues de a_
            sig = V[:, a_] & V[:, b_] & ~V[:, a_ + 1:b_].any(1)
            sp = D[:, a_] + (D[:, b_] - D[:, a_]) * S[:, a_] / (S[:, a_] + S[:, b_])
            hi[sig, a_], lo[sig, b_] = sp[sig], sp[sig]
    I = np.full(len(D), -np.inf)
    for a_ in range(3):
        m = np.flatnonzero(V[:, a_])
        if len(m):
            I[m] = np.logaddexp(I[m], _integral_d(fo, so, s2m, Fq[m], q0[m], D[m, a_], S[m, a_],
                                                  lambda i, D_: lpr(m[i], D_), lo[m, a_], hi[m, a_]) + Ul[m, a_])
    return I


def _lul(flim, s2m, a):
    """log P(f < f_lim) de los UL (eje -2) con el modelo a: Phi((f_lim - a)/sigma), sigma^2 = (f_lim/UL_NSIG)^2 +
    s2m a^2 (el error del modelo tambien en los UL). Sin UL: 0."""
    if not len(flim):
        return np.zeros(a.shape[:-2] + a.shape[-1:])
    fl = flim[:, None]
    return log_ndtr((fl - a) / np.sqrt((fl / UL_NSIG) ** 2 + s2m * a * a)).sum(-2)


def clasificar(cur, lib, pri, mw=0.0, ul_modo=UL_MODO, sig_mod=SIGMA_MOD):
    """Una curva (nnclf.data.Curve: t en MJD, band 0 = g / 1 = r, mag, err, ul, z) -> dict con log E y P por clase,
    mejor plantilla y sus parametros MAP. None si tiene menos de MIN_DET detecciones o ningun modelo valido.
    sig_mod: error del modelo como fraccion del flujo del modelo (regla 4)."""
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
    s2o, s2m = (K_MAG * f * cur.err[sel][:nd].astype(float)) ** 2, float(sig_mod) ** 2   # datos (flujo) y modelo
    w0 = 1.0 / (s2o + s2m * f * f)                     # punto de partida: el error del modelo en el flujo observado
    fo, so = f[:, None], s2o[:, None]
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
    ltab = {}
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
        per = max(1, CHUNK // (len(ie) * nT * len(bo)))
        acc, medias, tope, tope2 = [], [], -math.inf, -math.inf
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
            Lp = lib.lp[it][ks][:, ie]                                              # (nk, ne)
            mu_m = np.array([np.sum(n[3] * n[2]) for n in ch])
            mu_v = np.array([np.sum(n[3] * (n[2] - np.sum(n[3] * n[2])) ** 2) for n in ch])
            mp = mu_m[:, None] + lf[0] - Mref - aref[None, :]                       # prior de dmag + mu, gaussiano
            vp = (lf[1] ** 2 + mu_v)[:, None, None]
            q0, mp3 = Lp - m_ref, mp[:, :, None]
            shp = (len(ch), len(ie), nT)
            mpb, vpb = np.broadcast_to(mp3, shp), np.broadcast_to(vp, shp)
            bconst = np.broadcast_to(np.array([n[1] for n in ch])[:, None, None] + lpe[None, :, None] - math.log(nT), shp)
            for n in ch:
                if (lf, n[0]) not in ltab:
                    ltab[(lf, n[0])] = _tabla_prior_y(n[2], n[3], lf)
            yl = Mref + aref[None, :, None]                                        # y = M + mu(z) = d + yl
            lpri = lambda d: np.stack([np.interp(d[j] + yl[0], *ltab[(lf, n[0])], left=LOG0, right=LOG0)
                                       for j, n in enumerate(ch)])
            Fu = F[:, :, nd:]
            # etapa 1, todas las celdas: verosimilitud exacta en el punto de partida (minimos cuadrados con pesos fijos +
            # prior) y en la media del prior (con el error del modelo la verosimilitud se aplana a escalas grandes: ahi
            # el prior fija un segundo modo). Laplace con el ancho de cada uno.
            d0, s0 = _centro(np.matmul(w0, Fd * Fd), np.matmul(w0 * f, Fd), q0, mp3, vp)
            A0, Ab = np.exp(-K_MAG * (q0[:, :, None] + d0)), np.exp(-K_MAG * (q0[:, :, None] + mpb))
            ll0, ch0 = _verosim(fo, so, s2m, A0[:, :, None, :] * Fd)
            llb, chb = _verosim(fo, so, s2m, Ab[:, :, None, :] * Fd)
            ul0, ulb = _lul(flim, s2m, A0[:, :, None, :] * Fu), _lul(flim, s2m, Ab[:, :, None, :] * Fu)
            e1a = ll0 + lpri(d0) + np.log(SQ2PI * s0) + ul0
            e1b = llb + lpri(mpb) + np.log(SQ2PI * np.sqrt(vpb)) + ulb
            ga = e1a >= e1b
            lE = np.where(ga, e1a, e1b) + bconst
            dmap, chi2c, lulc = np.where(ga, d0, mpb), np.where(ga, ch0, chb), np.where(ga, ul0, ulb)
            del ll0, llb, ch0, chb, ul0, ulb, e1a, e1b, ga
            # etapa 2, celdas a menos de CRIBA_NATS del maximo de la plantilla: Newton desde los dos puntos
            tope = max(tope, float(lE.max()))
            sv = np.nonzero(lE > tope - CRIBA_NATS)
            if len(sv[0]):
                Fc, Fuc = Fd[sv[0], sv[1], :, sv[2]].T, Fu[sv[0], sv[1], :, sv[2]].T     # (nd, n), (nu, n)
                qc, mpc, vpc, yo = q0[sv[0], sv[1]], mp[sv[0], sv[1]], vp[sv[0], 0, 0], Mref + aref[sv[1]]

                def lpri_c(i, D_):
                    y, out = D_ + yo[i][:, None], np.empty_like(D_)
                    for j in np.unique(sv[0][i]):
                        r_ = sv[0][i] == j
                        out[r_] = np.interp(y[r_], *ltab[(lf, ch[j][0])], left=LOG0, right=LOG0)
                    return out
                todas = np.arange(len(qc))
                D, S, L, C, Ul, V = _posterior_d(fo, so, s2m, Fc, qc, mpc, vpc, d0[sv], lambda D_: lpri_c(todas, D_),
                                                 lambda d: _lul(flim, s2m, np.exp(-K_MAG * (qc + d)) * Fuc))
                Lv = np.where(V, L, -np.inf)
                b = np.argmax(Lv, 1)
                L2 = logsumexp(Lv, axis=1) + bconst[sv]
                lE[sv] = L2
                dmap[sv], chi2c[sv], lulc[sv] = (np.take_along_axis(x, b[:, None], 1)[:, 0] for x in (D, C, Ul))
                # integral en d donde puede pesar: cada modo en su tramo (cortes entre modos vecinos, mas cerca del
                # angosto)
                tope2 = max(tope2, float(L2.max()))
                k2 = np.flatnonzero(L2 > tope2 - LAPLACE_NATS)
                if len(k2):
                    I = _integral_modos(fo, so, s2m, Fc[:, k2].T, qc[k2], D[k2], S[k2], V[k2], Ul[k2],
                                        lambda i, D_: lpri_c(k2[i], D_))
                    lE[tuple(a[k2] for a in sv)] = I + bconst[sv][k2]
            chi2_min = min(chi2_min, float(chi2c.min()))
            Ac = np.exp(-K_MAG * (q0[:, :, None] + dmap))
            j, e, mT = np.unravel_index(int(np.argmax(lE)), lE.shape)
            lse = logsumexp(lE)
            if np.isfinite(lse):                                   # medias posteriores dentro del bloque
                wq = np.exp(lE - lse)
                Mq = dmap - mu_m[:, None, None] + yl
                acc.append(lse)
                medias.append(dict(z_post=np.sum(wq * lib.z[ks][:, None, None]),
                                   ebv_post=np.sum(wq * lib.ebv[ie][None, :, None]),
                                   tmax_post=T0 + DP * np.sum(wq * np.arange(nT)[None, None, :]), M_post=np.sum(wq * Mq)))
            if mapa[it] is None or lE[j, e, mT] > mapa[it]["lE"]:
                fm = Ac[j, e, mT] * Fd[j, e, :, mT]
                mapa[it] = dict(lE=float(lE[j, e, mT]), z_map=float(lib.z[ks[j]]), ebv_map=float(lib.ebv[ie[e]]),
                                tmax_map=T0 + mT * DP, M_map=float(dmap[j, e, mT] - mu_m[j] + yl[0, e, 0]),
                                chi2_map=float(chi2c[j, e, mT]),                  # con el error del modelo
                                chi2_dat_map=float(np.sum((f - fm) ** 2 / s2o)),  # solo el error de los datos
                                chi2_ul_map=float(-2.0 * lulc[j, e, mT]))
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


def _init(lib_dir, four, sin_z, cfg_name, ul_modo=UL_MODO, sig_mod=SIGMA_MOD, ii_dust="cfg"):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    lib = cargar_biblioteca(lib_dir)
    _W.update(lib=lib, pri=priors(lib, four, sin_z, cfg_name, ii_dust), ul_modo=ul_modo, sig_mod=sig_mod)


def _uno(args):
    cur, mw = args
    t0 = time.time()
    r = clasificar(cur, _W["lib"], _W["pri"], mw, _W["ul_modo"], _W["sig_mod"])
    return cur.key, r, time.time() - t0


def chi2_red(P, n_par=4):
    """chi2 reducido de la mejor plantilla (MAP): chi2 / (n_det - n_par), n_par = z, E(B-V), T_max y escala."""
    return P.chi2_map / np.maximum(P.n_det - n_par, 1)


def resumen_chi2(P):
    out = {}
    for k, c in (("con_error_del_modelo", "chi2_map"), ("solo_datos", "chi2_dat_map")):
        if c not in P or not len(P):
            continue
        x = chi2_red(P.assign(chi2_map=P[c])).to_numpy(float)
        out[k] = {"mediana": float(np.median(x)), "p16": float(np.percentile(x, 16)), "p84": float(np.percentile(x, 84)),
                  "frac_mayor_2": float(np.mean(x > 2)), "frac_mayor_5": float(np.mean(x > 5))}
    return out


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
        lib_dir=None, store=None, mw=None, cfg_name=RUN_CFG, n_boot=1000, ul_modo=UL_MODO, sig_mod=SIGMA_MOD,
        ii_dust="cfg"):
    from pipeline78.clf_villar import bootstrap_ci, calib_metrics, metrics
    t_ini = time.time()
    out_root = Path(out_root)
    lib_dir = Path(lib_dir) if lib_dir else construir_biblioteca(store, cfg_name, out_root, ii_dust=ii_dust)
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
        _init(lib_dir, four, sin_z, cfg_name, ul_modo, sig_mod, ii_dust)
        res = [_uno(x) for x in tareas]
    else:
        with get_context("spawn").Pool(workers, initializer=_init,
                                       initargs=(lib_dir, four, sin_z, cfg_name, ul_modo, sig_mod, ii_dust)) as pool:
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
                          chi2_min=r["chi2_min"], chi2_map=r["chi2_map"], chi2_dat_map=r["chi2_dat_map"],
                          chi2_ul_map=r["chi2_ul_map"],
                          z_map=r["z_map"], ebv_map=r["ebv_map"], tmax_map=r["tmax_map"], M_map=r["M_map"],
                          z_post=r["z_post"], ebv_post=r["ebv_post"], tmax_post=r["tmax_post"], M_post=r["M_post"],
                          **{f"logE_{c_}": float(r["logE"][i]) for i, c_ in enumerate(cls)}, t_seg=dt))
    nada += [dict(oid=o, motivo=f"menos de {MIN_DET} detecciones g + r") for o in faltan if o in oids]
    P = pd.DataFrame(filas, columns=["oid", "subset", "sn_type", "y_true", "y_pred"] + [f"p_{c}" for c in cls] +
                     ["n_det", "n_ul", "z", "prior_z", "ebv_mw", "best_template", "best_template_clase", "chi2_min",
                      "chi2_map", "chi2_dat_map", "chi2_ul_map", "z_map", "ebv_map", "tmax_map", "M_map", "z_post", "ebv_post",
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
        r["chi2_reducido"] = resumen_chi2(ps)
        real[s] = r
    ts = P.t_seg.to_numpy() if len(P) else np.zeros(1)
    ii = runcfg.RUNS_CFG[cfg_name].get("ii_dust") if ii_dust == "cfg" else ii_dust
    cfg_txt = dict(run_cfg=cfg_name, cuatro_clases=four, sin_z=sin_z, subset=subset, limit=limit, ul_modo=ul_modo,
                   sigma_mod=float(sig_mod), sigma_mod_es="fraccion del flujo del modelo, en cuadratura",
                   ii_dust=ii, laplace_nats=LAPLACE_NATS, sigma_z=SIGMA_Z, z_floor=Z_FLOOR, ul_nsig=UL_NSIG, t_pre=T_PRE,
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


# ------------------------------------------------------------------------------------------------ comparacion
CONTRA = {"villar": "clf_villar/sweep_t11_foco/mejor", "tf_base": "nnclf_t11/tf_base",
          "gru_attn_uni_z": "nnclf_t11/gru_attn_uni_z"}


def comparar(name, contra=None, out_root=OUT_ROOT, runs=RUNS, real_dir=REAL_DIR, villar="villar", tag=None):
    """Plantillas (3 clases) contra otros clasificadores en los mismos objetos, con las funciones de
    informe_clasificadores (mitad val de splits.read_val_meta, leer_pred, met y pareado = bootstrap pareado de nnclf).
    val_sel decide (P >= P_MIN), val_rep solo reporta (delta e IC 90 %). Ademas la exactitud de cada uno en las SNe que
    Villar cubre y en las que no, sobre las oids que clasifican todos menos Villar. Escribe comparacion.json en el run.
    gana: en val_sel el que pasa la regla (P >= P_MIN), en val_rep el que favorece el IC 90 % del delta si excluye 0
    (senal, no decision). None = ninguno."""
    from pipeline78.informe_clasificadores import SUB, _js, leer_pred, met, pareado, particion
    from pipeline78.nnclf.experimentos import P_MIN
    V = particion(real_dir)
    pl, info = leer_pred(Path(out_root) / name / "pred_real_val.csv", V)
    if pl is None:
        raise SystemExit(f"{name}: predice clases fuera de Ia/II/Ibc (corrida de 4 clases)")
    P = {name: pl}
    fuera = {name: info["fuera_de_val"]}
    for k, d in (contra or CONTRA).items():
        p, inf_ = leer_pred(Path(runs) / d / "pred_real_val.csv", V)
        P[k], fuera[k] = p, inf_["fuera_de_val"]
    pares, cubre = {}, {}
    for k in P:
        if k == name:
            continue
        pares[k] = {}
        for s in SUB:
            a, b = P[name][P[name].subset == s], P[k][P[k].subset == s]
            com = set(a.oid) & set(b.oid)
            ac, bc = a[a.oid.isin(com)], b[b.oid.isin(com)]
            c_ab, c_ba = pareado(ac, bc), pareado(bc, ac)
            if s == "val_sel":
                g = name if c_ab.get("p_mejora", 0) >= P_MIN else k if c_ba.get("p_mejora", 0) >= P_MIN else None
            else:
                lo, hi = c_ab.get("ic90_delta") or (0, 0)
                g = name if lo > 0 else k if hi < 0 else None
            pares[k][s] = {"n": len(com), name: met(ac, len(com)), k: met(bc, len(com)),
                           "plantillas_menos_otro": c_ab, "otro_menos_plantillas": c_ba, "gana": g}
    if villar in P:
        otros = [k for k in P if k != villar]
        for s in SUB:
            com = set.intersection(*(set(P[k].oid[P[k].subset == s]) for k in otros))
            cv = com & set(P[villar].oid[P[villar].subset == s])
            cubre[s] = {"n_comun": len(com), "n_cubre_villar": len(cv), "n_no_cubre": len(com - cv)}
            for k in P:
                p = P[k][P[k].subset == s]
                cubre[s][k] = {"cubre_villar": met(p[p.oid.isin(cv)], len(cv)),
                               "no_cubre_villar": met(p[p.oid.isin(com - cv)], len(com - cv)) if k != villar else None}
    res = _js({"plantillas": name, "contra": {k: str(Path(runs) / d) for k, d in (contra or CONTRA).items()},
               "fuera_de_val": fuera, "p_min": P_MIN, "completo": {k: {s: met(P[k][P[k].subset == s],
                                                                                  int((V.subset == s).sum())) for s in SUB}
                                                                    for k in P},
               "pares": pares, "cobertura_villar": cubre, "creada": time.strftime("%Y-%m-%d %H:%M:%S")})
    (Path(out_root) / name / f"comparacion{'_' + tag if tag else ''}.json").write_text(json.dumps(res, indent=1))
    lin = lambda m: (f"acc {m['acc']:.3f} bal {m['bal_acc']:.3f} [{m['bal_acc_ic95'][0]:.3f}, {m['bal_acc_ic95'][1]:.3f}]"
                     if m.get("n") else "-")
    for k, q in res["pares"].items():
        for s in SUB:
            x = q[s]
            c = x["plantillas_menos_otro"]
            print(f"[comparar] {name} - {k} {s} n={x['n']}: {lin(x[name])} | {lin(x[k])} | delta {c.get('delta', 0):+.3f}"
                  f" P(pl) {c.get('p_mejora', float('nan')):.3f} P({k}) "
                  f"{x['otro_menos_plantillas'].get('p_mejora', float('nan')):.3f} IC90 {c.get('ic90_delta')}")
    for s, q in res["cobertura_villar"].items():
        for k in P:
            print(f"[comparar] {s} {k}: cubre Villar {lin(q[k]['cubre_villar'])} (n {q['n_cubre_villar']}) | no cubre "
                  f"{lin(q[k]['no_cubre_villar']) if q[k]['no_cubre_villar'] else '-'} (n {q['n_no_cubre']})")
    return res


# ------------------------------------------------------------------------------------------------ sigma_mod (regla 4c)
def log_p_verdadera(P, cls):
    """log p de la clase verdadera por objeto, con el piso 1e-12 de clf_villar.calib_metrics."""
    pv = P[[f"p_{c}" for c in cls]].to_numpy(float)[np.arange(len(P)), P.y_true.map(list(cls).index).to_numpy()]
    return np.log(np.clip(pv, 1e-12, 1.0))


def elegir_sigma(base="sigma", out_root=OUT_ROOT, grilla=SIGMA_MOD_GRID, apriori=SIGMA_MOD_APRIORI, correr=True,
                 workers=4, **kw):
    """Regla 4c: corre la grilla en val_sel (3 clases con z, UL y polvo por defecto) en out_root/<base>/s<valor> y
    elige por la log-verosimilitud de la clase verdadera media por clase. La de mayor valor reemplaza a la a priori
    solo con P >= P_MIN en el bootstrap pareado (paired_bootstrap de nnclf sobre log p por objeto, mismas oids). Escribe
    <base>/eleccion.json. Solo lee val_sel."""
    from pipeline78.clf_villar import calib_metrics, metrics
    from pipeline78.nnclf.experimentos import BOOT_SEED, N_BOOT, P_MIN, paired_bootstrap
    cls = D.classes(False)
    out_root = Path(out_root)
    P = {}
    for s_ in grilla:
        d = out_root / base / f"s{s_:.2f}"
        if correr and not (d / "metrics.json").exists():
            run(f"{base}/s{s_:.2f}", subset="val_sel", sig_mod=s_, workers=workers, out_root=out_root, **kw)
        q = pd.read_csv(d / "pred_real_val.csv", dtype={"oid": str})
        if set(q.subset) - {"val_sel"}:
            raise ValueError(f"{d}: tiene filas fuera de val_sel")
        P[s_] = q.set_index("oid")
    com = sorted(set.intersection(*(set(q.index) for q in P.values())))
    y = P[apriori].loc[com].y_true.map(cls.index).to_numpy()
    lp, tab = {}, []
    for s_, q in P.items():
        q = q.loc[com].reset_index()
        lp[s_] = log_p_verdadera(q, cls)
        m = metrics(y, q.y_pred.map(cls.index).to_numpy(), cls)
        tab.append(dict(sigma_mod=s_, n=len(q), logp_media=float(lp[s_].mean()),
                        logp_media_por_clase=float(np.mean([lp[s_][y == k].mean() for k in np.unique(y)])),
                        bal_acc=m["bal_acc"], acc=m["acc"], **calib_metrics(q[[f"p_{c}" for c in cls]].to_numpy(), y),
                        chi2_reducido=resumen_chi2(q)))
    cand = max(tab, key=lambda r: (r["logp_media_por_clase"], -r["sigma_mod"]))["sigma_mod"]
    d_, p_, ic = paired_bootstrap(y, lp[cand], lp[apriori], N_BOOT, BOOT_SEED) if cand != apriori else (0.0, 0.0, (0, 0))
    elegido = cand if cand != apriori and p_ >= P_MIN else apriori
    res = dict(regla="4c: max log p(clase verdadera) media por clase en val_sel; reemplaza al a priori con P >= P_MIN",
               subset="val_sel", n=len(com), a_priori=apriori, candidata=cand, p_min=P_MIN,
               pareado=dict(delta_logp_por_clase=float(d_), p_mejora=float(p_), ic90=list(ic)), elegido=elegido,
               tabla=tab, creada=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out_root / base / "eleccion.json").write_text(json.dumps(res, indent=1, default=float))
    for r in tab:
        c = r["chi2_reducido"].get("con_error_del_modelo", {})
        print(f"[sigma] {r['sigma_mod']:.2f}: log p {r['logp_media']:.3f} (por clase {r['logp_media_por_clase']:.3f}) "
              f"bal_acc {r['bal_acc']:.3f} ECE {r['ece']:.3f} chi2_red mediana {c.get('mediana', float('nan')):.2f}")
    print(f"[sigma] candidata {cand} contra {apriori}: P = {p_:.3f} -> elegido {elegido}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline78.plantillas_clf")
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("biblioteca")
    b.add_argument("--force", action="store_true")
    c = sp.add_parser("comparar", help="contra Villar y las redes en los mismos objetos (comparacion.json)")
    c.add_argument("--name", required=True)
    c.add_argument("--contra", nargs="*", help="nombre=ruta relativa a RUNS (por defecto CONTRA)")
    c.add_argument("--tag", help="comparacion_<tag>.json")
    g = sp.add_parser("sigma", help="regla 4c: SIGMA_MOD en val_sel")
    g.add_argument("--workers", type=int, default=4)
    r = sp.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--cuatro-clases", action="store_true")
    r.add_argument("--sin-z", action="store_true")
    r.add_argument("--subset", choices=("val", "val_sel", "val_rep"), default="val")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--workers", type=int, default=2, help="maximo 4")
    r.add_argument("--ul", choices=UL_MODOS, default=UL_MODO, help="UL en la verosimilitud (regla 4b)")
    r.add_argument("--sigma-mod", type=float, default=SIGMA_MOD, help="error del modelo (regla 4c)")
    r.add_argument("--ii-dust", choices=("cfg", "sudare"), default="cfg", help="polvo de las II (regla 7)")
    a = ap.parse_args(argv)
    if a.cmd == "biblioteca":
        from threadpoolctl import threadpool_limits
        with threadpool_limits(int(os.environ.get("P78_PL_THREADS", "2"))):
            print(construir_biblioteca(force=a.force))
        return
    if a.cmd == "comparar":
        comparar(a.name, dict(x.split("=", 1) for x in a.contra) if a.contra else None, tag=a.tag)
        return
    if a.cmd == "sigma":
        elegir_sigma(workers=a.workers)
        return
    run(a.name, a.cuatro_clases, a.sin_z, a.subset, a.limit, a.workers, ul_modo=a.ul, sig_mod=a.sigma_mod,
        ii_dust=a.ii_dust)


if __name__ == "__main__":
    main()
