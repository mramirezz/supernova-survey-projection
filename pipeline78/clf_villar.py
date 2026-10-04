"""Clasificador oficial sobre las features de Villar (SPM + MCMC ref). villar-clf-brief, 2026-10-04.

Uso: python -m pipeline78.clf_villar {train|eval|sweep|gap} --features-sims DIR --name NOMBRE [opciones]

REGLAS
1. Llaves. Todo merge va por (oid, part_index, sn_type): r con g, y features con metadata. En las sims oid = field
   de _sims_all. El pipeline viejo (opt_clasificador/05_faseC*) juntaba r con g por (oid, part_index) sin sn_type
   y el color salia de otro tipo (bug H4: 13 063 filas pasaban a 26 082). Aca un duplicado de llave es un error.
2. Reales. Solo origen == holdout & split == val & ~excluir. meta_real_ztf.csv (pipeline78.splits.read_val_meta) y
   el features.csv real se leen con csv linea a linea y solo las filas val llegan a pandas: la mitad final no se
   carga nunca (test que lo vigila). Por defecto las features reales son las re-extraidas tras el logfix
   (RUNS/features_real_ztf2, --features-real). Seleccion anidada (revision B, mismo protocolo que nnclf): la mitad
   val se parte con pipeline78.splits.val_split en val_sel (elegir y ajustar todo lo que mira reales: S(m), wz_dr,
   prior EM) y val_rep (solo reportar). Columna subset de las reales.
3. Clases. Ia, II (= II + IIb), Ibc. IIn fuera de la metrica principal (--cuatro-clases la agrega).
4. Features por banda b (r, g): m_pk_b = -2.5 log10(A_b) (pico aparente desde A), M_pk_b = m_pk_b - mu(z) (solo
   con z, LCDM plano H0 = 70, Om = 0.3 como core.utils.DL_calculator), f_b, t_rise_b, t_fall_b, gamma_b en reposo
   (divididos por 1 + z, en sims y en reales) y errores relativos err/|valor| de A, f, t_rise, t_fall, gamma.
   Colores: color_gr = -2.5 log10(A_g / A_r) y diferencias g - r de t_rise, t_fall, gamma (en reposo) y de f.
   t0 no entra: cada banda tiene su propio origen (la primera deteccion de esa banda en reader.mjd_to_phase).
   Real sin z (4 de las val): tiempos en el marco observado y M_pk = NaN.
   Faltantes (p. ej. sin g) quedan NaN: HistGradientBoosting los acepta y RF/MLP imputan la mediana con bandera.
   Mitigaciones del estudio del gap (fset rg_cens y g_modo separado, grilla mitig):
     rg_cens  t_rise censurado: si el t_rise observado (antes de 1/(1+z)) esta en el piso del extractor, t_rise_cens
              = NaN y t_rise_piso = 1 (0 si no, NaN sin ajuste). Las sims tienen limites superiores espurios: 9.9 % de
              las Ia y 40.9 % de las II simuladas en el piso contra 0.9 % y 6.1 % de las reales. rg no cambia.
     separado sin g en 36 % de las sims y 20 % de las reales (Ibc: 62 % contra 8 %): los arboles aprenden "sin g =>
              Ibc/II". Modelo A con las sims con g (m_pk_g finito) y el set entero, modelo B con todas las sims y solo
              las columnas de r. Cada objeto (sim o real) se predice con A si tiene g y con B si no.
   Fisica (fset rg_fisica, grilla fisica): rg + hombro de r (residuos de la curva contra el SPM a +15-40 y +40-70 d
   en reposo desde el pico) + color g - r de los SPM a +0, +15, +30 d y su pendiente. Ver la seccion fisica.
5. Pesos de entrenamiento: w = w_z (ya trae 1/(1+z)) * peso de seleccion, y despues balance de clases (cada clase
   suma lo mismo; en el jerarquico, cada nivel queda con prior efectivo uniforme sobre las clases).
   Peso de seleccion (--peso, por defecto wz):
     wz    nada mas. Es la configuracion principal (la mas limpia para las tasas).
     wz_S  S(m) = p_real(m) / p_sim(m), cociente de histogramas de la magnitud de pico aparente (r, o g si no hay
           r) suavizados con una gaussiana. Sims de todas las clases juntas ponderadas por w_z, contra las reales
           de val_sel con features (NUNCA val_rep). NO usa etiquetas (la funcion no las recibe). Variante.
           PARA LAS TASAS S(m) se recalcula sobre la muestra objetivo (sin etiquetas): el de val_sel imita la
           seleccion espectroscopica, no la de la muestra fotometrica de ZTF o SUDARE.
     wz_dr cociente de densidad p_real(x)/p_sim(x) en todo el espacio de features, estimado con un clasificador
           sim contra real (validacion cruzada, sin etiquetas), con las reales de val_sel. Variante.
6. Prior. none: argmax de las probabilidades del modelo (prior de entrenamiento uniforme). La temperatura se ajusta
   en las predicciones fuera de fold de las sims (no mira reales ni etiquetas) y no cambia el argmax.
   em: reajuste de prior por EM sin etiquetas (Saerens, Latinne y Decaestecker 2002) sobre las probabilidades de
   val_sel. El prior resultante se aplica igual a val_sel y val_rep.
7. Seleccion anidada (sweep; revision B, decision del 2026-10-04, mismo protocolo que pipeline78.nnclf):
   a. CV en las sims agrupada por PLANTILLA (por sn_type, II e IIb aparte) como pre-filtro: quedan fuera las
      configuraciones con exactitud balanceada de CV bajo la mediana del barrido.
   b. Las que pasan se ordenan SOLO por la exactitud balanceada en val_sel (prior none). val_rep no entra.
   c. La primera (candidata) se compara con la base simple y fuerte BASELINE (hgb, rg, con z, wz, g nan) por bootstrap
      pareado estratificado por clase sobre val_sel (pipeline78.nnclf.experimentos.paired_bootstrap, 2000
      remuestreos). La candidata queda solo si P(delta > 0) >= 0.9. Si no, queda la base.
   d. Todo se reporta en val_rep (la cifra honesta, con IC 95 % bootstrap), val_sel y val completo, con los sufijos
      _rep, _sel y _val en resumen.csv (como nnclf). Cobertura = reales del subconjunto con features / reales del
      subconjunto. Con --nn-preds se dan tambien las metricas sobre las oids que la red clasifica.
   La cifra de la tesis sale de la mitad final, con la configuracion elegida aca.
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

from pipeline78 import splits
from pipeline78.nnclf.experimentos import BOOT_SEED, N_BOOT as N_BOOT_PAREADO, P_MIN, paired_bootstrap
from pipeline78.paths import RUNS, ZLF

REAL_DIR = RUNS / "real_ztf"
REAL_FEAT = RUNS / "features_real_ztf2" / "features" / "features.csv"   # re-extraidas tras el logfix (bfcedc2)
OUT_ROOT = RUNS / "clf_villar"
SUBSETS = ("val_rep", "val_sel", "val")                 # val = val completo (referencia)
SUFFIX = {"val_rep": "rep", "val_sel": "sel", "val": "val"}
BASELINE = {"model": "hgb", "fset": "rg", "use_z": True, "peso": "wz", "balance": True,   # incumbente del barrido
            "g_modo": "nan"}
G_MODOS = ("nan", "separado")                          # nan: g faltante queda NaN; separado: modelos A (con g) y B (r)
SEED = 20261004
N_JOBS = int(os.environ.get("P78_CLF_THREADS", "2"))   # hilos de RF/HGB/BLAS: la RAM y la CPU se comparten
KEYS = ["oid", "part_index", "sn_type"]
BANDS = ("r", "g")
SN_TYPES = ("Ia", "II", "IIb", "Ibc", "IIn")          # orden fijo: indice para el rng de los folds
SN_TYPE_CLASS = {"Ia": "Ia", "II": "II", "IIb": "II", "Ibc": "Ibc", "IIn": "IIn"}
PARS = ["A", "f", "t_rise", "t_fall", "gamma"]
TIMES = ["t_rise", "t_fall", "gamma"]
QUAL = ["n_points", "time_span", "rms"]
RAW = PARS + [f"{p}_err" for p in PARS] + QUAL
# piso de t_rise del extractor: 1.0 d en el marco OBSERVADO (feature_extraction/ztf_literature_features/config.py,
# MODEL_CONFIG bounds t_rise = (1.0, 100.0)). 1.05 da margen a las medianas del MCMC pegadas al borde.
T_RISE_PISO = 1.05


def classes(four=False):
    return ("Ia", "II", "Ibc", "IIn") if four else ("Ia", "II", "Ibc")


def class_of(sn_type, four=False):
    c = SN_TYPE_CLASS.get(sn_type)
    return c if c in classes(four) else None


def distmod(z, H0=70.0, om=0.3):
    """Modulo de distancia LCDM plano. NaN donde z no es finito o z <= 0."""
    z = np.asarray(z, np.float64)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0)
    if ok.any():
        zg = np.linspace(0.0, float(z[ok].max()) * 1.01 + 1e-3, 4001)
        inv = 1.0 / np.sqrt(om * (1 + zg) ** 3 + 1 - om)
        dc = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zg))]) * 299792.458 / H0
        out[ok] = 5 * np.log10((1 + z[ok]) * np.interp(z[ok], zg, dc) * 1e5)
    return out


# ------------------------------------------------------------------------------------------------ lectura
def widen(df, cols=RAW):
    """Filas por (llave, banda) -> una fila por llave con sufijo _r/_g. Llaves (oid, part_index, sn_type)."""
    d = df.copy()
    d["filter_band"] = d.filter_band.astype(str).str.lower().str[-1]
    d = d[d.filter_band.isin(BANDS)]
    dup = d.duplicated(KEYS + ["filter_band"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} filas repetidas por (oid, part_index, sn_type, banda)")
    cols = [c for c in cols if c in d.columns]
    out = None
    for b in BANDS:
        sub = d.loc[d.filter_band == b, KEYS + cols].rename(columns={c: f"{c}_{b}" for c in cols})
        out = sub if out is None else out.merge(sub, on=KEYS, how="outer", validate="one_to_one")
    return out.reset_index(drop=True)


def _to_num(df):
    for c in df.columns:
        if c in ("oid", "sn_type", "filter_band", "sn_name", "template", "subtipo", "subtype", "split", "origen",
                 "motivo"):
            continue
        num = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
        if num.notna().sum() == (df[c].astype(str) != "").sum():
            df[c] = num
    return df


def read_val_meta(meta_path, four=False):
    """Metadatos de las reales val (regla 2) con la particion anidada en la columna subset (val_sel o val_rep).
    pipeline78.splits.read_val_meta no guarda las filas de otros splits (una segunda pasada solo verifica que ninguna
    oid val aparezca en ellas). La particion se hace sobre todo val (IIn incluida), igual que en nnclf, asi que las
    dos variantes de clases y los dos metodos comparten exactamente val_sel y val_rep."""
    v = splits.read_val_meta(meta_path)
    sel, rep = splits.val_split(v)
    v["oid"] = v.oid.astype(str)
    v["subset"] = np.where(v.oid.isin(set(sel)), "val_sel", np.where(v.oid.isin(set(rep)), "val_rep", ""))
    v["part_index"] = v.part_index.astype(int)
    v["z"] = pd.to_numeric(v.z, errors="coerce")
    v["cls"] = v.sn_type.map(lambda t: class_of(t, four))
    return v[v.cls.notna()].reset_index(drop=True)


def read_rows_for_oids(path, oids):
    """Solo las filas del csv cuya oid esta en oids llegan a pandas."""
    oids = set(oids)
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        k = head.index("oid")
        rows = [row for row in r if row[k] in oids]
    return _to_num(pd.DataFrame(rows, columns=head))


def features_csv(d):
    d = Path(d)
    return d / "features" / "features.csv" if (d / "features" / "features.csv").exists() else d


def run_dir_for(features_sims):
    """features_<proy> -> RUNS/<proy> (donde vive _sims_all.parquet)."""
    n = Path(features_sims).name
    return RUNS / (n[len("features_"):] if n.startswith("features_") else n)


def load_sims(features_sims, run_dir=None, four=False):
    raw = pd.read_csv(features_csv(features_sims), usecols=lambda c: c in KEYS + ["filter_band"] + RAW)
    W = widen(raw)
    s = pd.read_parquet(Path(run_dir or run_dir_for(features_sims)) / "_sims_all.parquet",
                        columns=["field", "part_index", "sn_type", "template", "subtype", "z", "w_z"])
    s = s.rename(columns={"field": "oid"})
    if s.duplicated(KEYS).any():
        raise ValueError("_sims_all con llaves (field, part_index, sn_type) repetidas")
    n0 = len(W)
    W = W.merge(s, on=KEYS, how="inner", validate="one_to_one")
    if len(W) != n0:
        raise ValueError(f"{n0 - len(W)} sims con features sin metadata en _sims_all")
    W["cls"] = W.sn_type.map(lambda t: class_of(t, four))
    return W[W.cls.notna()].reset_index(drop=True)


def load_real_val(features_real=REAL_FEAT, real_dir=REAL_DIR, four=False):
    """(features de las reales val con al menos una banda ajustada, metadatos val de la clase pedida)."""
    meta, feat = Path(real_dir) / "meta_real_ztf.csv", features_csv(features_real)
    if feat.exists() and feat.stat().st_mtime < meta.stat().st_mtime:
        print(f"[clf_villar] AVISO: {feat} es anterior a {meta} (re-extraer las features reales, revision A)",
              flush=True)
    v = read_val_meta(meta, four)
    f = read_rows_for_oids(feat, v.oid)
    if not set(f.oid) <= set(v.oid):
        raise AssertionError("se colo una oid fuera de la mitad val")
    cols = KEYS + ["z", "subtipo", "cls", "subset"]
    if len(f):
        f["part_index"] = f.part_index.astype(int)
        W = widen(f).merge(v[cols], on=KEYS, how="inner", validate="one_to_one")
    else:
        W = pd.DataFrame(columns=cols)
    return W.reset_index(drop=True), v


# ------------------------------------------------------------------------------------------------ features
def derive(W, rest_frame=True):
    """Features derivadas (regla 4) sobre la tabla ancha con columna z. Devuelve una tabla nueva."""
    z = pd.to_numeric(W.z, errors="coerce").to_numpy(float)
    okz = np.isfinite(z) & (z >= 0)
    fac = np.where(okz & rest_frame, 1.0 + np.where(okz, z, 0.0), 1.0)
    mu = distmod(z)
    F = W[[c for c in W.columns if not any(c.startswith(p + "_") for p in RAW)]].copy()
    for b in BANDS:
        A = W.get(f"A_{b}", pd.Series(np.nan, index=W.index)).to_numpy(float)
        m = np.where(A > 0, -2.5 * np.log10(np.where(A > 0, A, 1.0)), np.nan)
        F[f"m_pk_{b}"] = m
        F[f"M_pk_{b}"] = m - mu
        F[f"f_{b}"] = W.get(f"f_{b}", np.nan)
        for t in TIMES:
            F[f"{t}_{b}"] = W.get(f"{t}_{b}", pd.Series(np.nan, index=W.index)).to_numpy(float) / fac
        for p in PARS:
            val = W.get(f"{p}_{b}", pd.Series(np.nan, index=W.index)).to_numpy(float)
            err = W.get(f"{p}_err_{b}", pd.Series(np.nan, index=W.index)).to_numpy(float)
            F[f"rel_{p}_{b}"] = np.where(np.abs(val) > 0, err / np.where(np.abs(val) > 0, np.abs(val), 1.0), np.nan)
        for q in QUAL:
            F[f"{q}_{b}"] = W.get(f"{q}_{b}", np.nan)
    F["color_gr"] = F.m_pk_g - F.m_pk_r
    for t in TIMES + ["f"]:
        F[f"d_{t}_gr"] = F[f"{t}_g"] - F[f"{t}_r"]
    F["m_sel"] = F.m_pk_r.where(F.m_pk_r.notna(), F.m_pk_g)     # magnitud de seleccion S(m); no es feature
    F["tiene_r"] = F.m_pk_r.notna()
    for b in BANDS:                     # censura de t_rise (solo rg_cens): se mira el valor OBSERVADO, sin 1/(1+z)
        tr = W.get(f"t_rise_{b}", pd.Series(np.nan, index=W.index)).to_numpy(float)
        piso = tr < T_RISE_PISO
        F[f"t_rise_piso_{b}"] = np.where(np.isfinite(tr), piso.astype(float), np.nan)
        F[f"t_rise_cens_{b}"] = np.where(piso, np.nan, F[f"t_rise_{b}"].to_numpy(float))
    F["d_t_rise_cens_gr"] = F.t_rise_cens_g - F.t_rise_cens_r
    return F.replace([np.inf, -np.inf], np.nan)


def _forma(b):
    return [f"f_{b}", f"t_rise_{b}", f"t_fall_{b}", f"gamma_{b}"]


COLOR = ["color_gr", "d_t_rise_gr", "d_t_fall_gr", "d_gamma_gr", "d_f_gr"]
REL = [f"rel_{p}_{b}" for b in BANDS for p in PARS]
FSETS = ("viejo", "forma_r", "rg", "rg_err", "rg_m", "rg_robusto", "rg_cens", "rg_fisica")
FSETS_FISICA = ("rg_fisica",)                          # necesitan las curvas: prepare(fisica=True)
CENS = {"t_rise_r": "t_rise_cens_r", "t_rise_g": "t_rise_cens_g", "d_t_rise_gr": "d_t_rise_cens_gr"}
# features fisicas (seccion fisica): hombro de r en los residuos del SPM y evolucion del color g - r
VENTANAS = {"res_r_15_40": (15.0, 40.0), "res_r_40_70": (40.0, 70.0)}
FASES_COLOR = (0, 15, 30)
FIS = list(VENTANAS) + [f"color_gr_{p}" for p in FASES_COLOR] + ["d_color_gr_30_0"]


def fset_cols(name, use_z):
    """(columnas del nivel 1 del jerarquico, columnas de todo lo demas). Sin z no entra la magnitud absoluta."""
    Mr = ["M_pk_r"] if use_z else []
    Mrg = ["M_pk_r", "M_pk_g"] if use_z else []
    if name == "viejo":            # receta congelada de 05_faseC_run1000_v2: N1 forma r, N2 + color, t_rise, M
        n1 = ["f_r", "t_fall_r", "gamma_r"]
        return n1, n1 + ["color_gr", "t_rise_r"] + Mr
    rg = _forma("r") + _forma("g") + COLOR + Mrg
    sets = {"forma_r": _forma("r") + Mr, "rg": rg, "rg_err": rg + REL, "rg_m": rg + ["m_pk_r", "m_pk_g"],
            # sin t_rise: la feature que mas delata a las sims en el diagnostico gap del smoke (1.7 d contra 3.1 d)
            "rg_robusto": [c for c in rg if "t_rise" not in c],
            # t_rise censurado en el piso del extractor y su bandera (regla 4, mitigaciones)
            "rg_cens": [CENS.get(c, c) for c in rg] + ["t_rise_piso_r", "t_rise_piso_g"],
            # hombro de r y color g - r en fases fijas (seccion fisica)
            "rg_fisica": rg + FIS}
    if name not in sets:
        raise ValueError(f"feature set desconocido: {name}")
    return sets[name], sets[name]


def cols_solo_r(cols):
    """Columnas del modelo B de g_modo separado: fuera las de g (terminan en _g) y las de color (llevan _gr)."""
    return [c for c in cols if not c.endswith("_g") and "_gr" not in c]


GAP_COLS = (_forma("r") + _forma("g") + COLOR + ["M_pk_r", "M_pk_g", "m_pk_r", "m_pk_g"] + REL
            + [f"{q}_{b}" for b in BANDS for q in QUAL])


# ------------------------------------------------------------------------------------------------ fisica
# fset rg_fisica = rg + FIS (task V, 2026-10-04). Lo que separa Ia de Ibc mas alla de la forma suave: el segundo
# maximo (hombro) de las Ia en r a +20-35 d en reposo y la evolucion del color. El SPM es suave y de un solo pico, no
# puede representar el hombro: se mide en los RESIDUOS de la curva observada contra el SPM ajustado.
#   res_r_15_40, res_r_40_70  mediana de m_SPM - m_obs (> 0: mas brillante que el modelo suave) de las detecciones r
#                             con fase en reposo en [15, 40) y [40, 70) d desde el pico del SPM de r. NaN con < 2.
#   color_gr_0/15/30          g - r de los SPM de cada banda a +0, +15 y +30 d en reposo desde el pico del SPM de r.
#   d_color_gr_30_0           color_gr_30 - color_gr_0. Sin r o sin g los colores quedan NaN.
# Parametros: A, f, t0, t_rise, t_fall, gamma de features.csv = MEDIANA de cada parametro sobre las 200 mejores curvas
# del MCMC (mcmc_fitter.fit_mcmc 'params'); con ellos el extractor evalua model_flux, rms y mad (no los _moc).
# Mismo codigo para sims y reales (fisica_filas): la curva pasa por reader.prepare_lightcurve del extractor con su
# DATA_FILTER_CONFIG, que da el origen de fase de cada banda (primera deteccion tras el agrupado de 8 h) y los puntos
# que vio el ajuste, y el modelo por model.alerce_model y model.flux_to_mag (importados, no reescritos).
# require_upper_limits=False: la eleccion de UL no cambia ni el origen ni las detecciones (una curva con features ya
# paso ese filtro). Pico del SPM = argmax del modelo en una grilla de 0.05 d. Reposo con (1 + z) como derive; sin z
# (o con --obs-frame), marco observado. Cache csv por corrida al lado de features.csv (fisica_<marco>.csv las sims,
# fisica_val_<marco>.csv las reales val), solo se agrega: una fila sirve si su llave, z y parametros coinciden.
SPM = ["A", "f", "t0", "t_rise", "t_fall", "gamma"]             # orden de alerce_model
SPM_W = [f"{p}_{b}" for b in BANDS for p in SPM]
LC_COLS = ["oid", "part_index", "sn_type", "mjd", "filter", "magnitud_proyectada", "magerr", "upperlimit"]
CACHE_COLS = KEYS + ["z"] + SPM_W + FIS


def _zlf(nombre):
    """Modulo del extractor (paths.ZLF) cargado por ruta con nombre propio: ZLF trae config.py, reader.py y model.py,
    que chocarian con otros del sys.path. reader.py apaga los avisos de numpy y de warnings al importarse: se
    restauran."""
    key = f"_zlf_{nombre}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ZLF / f"{nombre}.py")
        mod, err = importlib.util.module_from_spec(spec), np.geterr()
        with warnings.catch_warnings():
            spec.loader.exec_module(mod)
        np.seterr(**err)
        sys.modules[key] = mod
    return sys.modules[key]


def bandas_lc(rows):
    """Filas LC_COLS de UNA curva -> {banda: DataFrame MJD, MAG, MAGERR, Upperlimit}. La misma conversion que
    parquet_reader.parse_parquet_lightcurve del extractor, que lee el parquet entero (con la mitad final en las
    reales) y por eso no se llama (test que las compara)."""
    out = {}
    for b in BANDS:
        s = rows[rows["filter"] == b]
        d = pd.DataFrame({"MJD": s.mjd.astype(float).to_numpy(), "MAG": s.magnitud_proyectada.astype(float).to_numpy(),
                          "MAGERR": s.magerr.astype(float).to_numpy(),
                          "Upperlimit": s.upperlimit.astype(str).str.strip().str.upper().eq("T").to_numpy()})
        d = d.dropna(subset=["MJD", "MAG"]).sort_values("MJD").reset_index(drop=True)
        if len(d):
            out[b] = d
    return out


def pico_spm(p):
    """Fase del maximo del SPM. La grilla [t0 - 3 t_rise - 10, t0 + gamma + 3 t_rise + 30] lo contiene con los limites
    del extractor (despues de t0 + gamma el maximo cae antes de t0 + t_rise ln(t_fall / t_rise - 1) < t0 + 56 d)."""
    _, _, t0, tr, _, gam = p
    t = np.arange(t0 - 3 * tr - 10, t0 + gam + 3 * tr + 30, 0.05)
    return float(t[np.argmax(_zlf("model").alerce_model(t, *p))])


def fisica_curva(bandas, par, z, rest_frame=True):
    """FIS de UNA curva. bandas: salida de bandas_lc; par: {banda: [A, f, t0, t_rise, t_fall, gamma]} (NaN = banda
    sin ajuste); z (NaN = marco observado)."""
    M, cfg = _zlf("model"), _zlf("config").DATA_FILTER_CONFIG
    out, lc = dict.fromkeys(FIS, np.nan), {}
    for b, p in par.items():
        if b in bandas and np.all(np.isfinite(p)):
            with contextlib.redirect_stdout(io.StringIO()):            # prepare_lightcurve imprime avisos de UL
                d = _zlf("reader").prepare_lightcurve(
                    bandas[b], b, max_days_after_peak=cfg["max_days_after_peak"],
                    max_days_before_peak=cfg["max_days_before_peak"],
                    max_days_before_first_obs=cfg["max_days_before_first_obs"], require_upper_limits=False)
            if d is not None:
                lc[b] = d
    if "r" not in lc:
        return out
    fac = 1.0 + z if rest_frame and np.isfinite(z) and z >= 0 else 1.0
    pk = pico_spm(par["r"])
    det = ~np.asarray(lc["r"]["is_upper_limit"], bool)
    ph = np.asarray(lc["r"]["phase"], float)[det]
    res = M.flux_to_mag(M.alerce_model(ph, *par["r"])) - np.asarray(lc["r"]["mag"], float)[det]
    fase = (ph - pk) / fac
    for c, (lo, hi) in VENTANAS.items():
        k = (fase >= lo) & (fase < hi)
        if k.sum() >= 2:
            out[c] = float(np.median(res[k]))
    if "g" in lc:
        t = lc["r"]["reference_mjd"] + pk + np.asarray(FASES_COLOR, float) * fac            # mjd de cada fase
        m = {b: M.flux_to_mag(M.alerce_model(t - lc[b]["reference_mjd"], *par[b])) for b in BANDS}
        gr = m["g"] - m["r"]
        out.update({f"color_gr_{p}": float(v) for p, v in zip(FASES_COLOR, gr)})
        out["d_color_gr_30_0"] = float(gr[FASES_COLOR.index(30)] - gr[FASES_COLOR.index(0)])
    return out


def fisica_filas(rows, T, rest_frame=True):
    """FIS por llave: el camino comun de sims y reales. rows: filas LC_COLS de las curvas; T: KEYS + z + SPM_W, una
    fila por llave. La llave solo encuentra la curva (sn_type no entra al calculo). Llave sin curva = error."""
    rows = rows.assign(oid=rows.oid.astype(str), part_index=rows.part_index.astype(int))
    grupos = dict(tuple(rows.groupby(KEYS, sort=False))) if len(rows) else {}
    out = []
    with np.errstate(all="ignore"):
        for t in T[KEYS + ["z"] + SPM_W].itertuples(index=False):
            k = (str(t.oid), int(t.part_index), t.sn_type)
            if k not in grupos:
                raise ValueError(f"llave con features y sin curva: {k}")
            par = {b: np.array([getattr(t, f"{p}_{b}") for p in SPM], float) for b in BANDS}
            out.append({"oid": k[0], "part_index": k[1], "sn_type": k[2],
                        **fisica_curva(bandas_lc(grupos[k]), par, float(t.z), rest_frame)})
    return pd.DataFrame(out, columns=KEYS + FIS)


def _tabla_spm(raw, X):
    """Parametros SPM anchos (SPM_W) de las llaves de X, con su z. raw: filas de features.csv por banda."""
    W = widen(raw.assign(part_index=raw.part_index.astype(int)), SPM)
    W = W.reindex(columns=KEYS + SPM_W)
    k = X[KEYS + ["z"]].assign(oid=X.oid.astype(str), part_index=X.part_index.astype(int),
                               z=pd.to_numeric(X.z, errors="coerce"))
    return k.merge(W, on=KEYS, how="left", validate="one_to_one")


def _con_cache(T, cache, fuente, calcular):
    """FIS de las llaves de T (KEYS + z + SPM_W) con cache csv que solo se agrega. Sirve la ultima fila de cada llave
    si su z y sus parametros coinciden con T (rtol 1e-10: ida y vuelta por csv); el resto se calcula con calcular(T')
    y se agrega. Un cache de otras columnas o anterior a la fuente de las curvas se descarta entero. Se lee con
    read_rows_for_oids: solo las oids de T llegan a pandas."""
    cache, T = Path(cache), T.reset_index(drop=True)
    if cache.exists():
        with open(cache, newline="") as fh:
            head = next(csv.reader(fh), [])
        if head != CACHE_COLS or (Path(fuente).exists() and cache.stat().st_mtime < Path(fuente).stat().st_mtime):
            print(f"[clf_villar] fisica: cache {cache} viejo o de otras columnas, se recalcula", flush=True)
            cache.unlink()
    hit = np.zeros(len(T), bool)
    m = T
    if cache.exists() and len(T):
        Cc = read_rows_for_oids(cache, set(T.oid.astype(str)))
        if len(Cc):
            Cc = Cc.assign(part_index=Cc.part_index.astype(int)).drop_duplicates(KEYS, keep="last")
            ren = {c: f"{c}__c" for c in ["z"] + SPM_W}
            m = T.merge(Cc[CACHE_COLS].rename(columns=ren), on=KEYS, how="left", indicator=True, validate="one_to_one")
            hit = (m._merge == "both").to_numpy()
            for c, cc in ren.items():
                hit &= np.isclose(m[c].to_numpy(float), m[cc].to_numpy(float), rtol=1e-10, atol=0, equal_nan=True)
    N = calcular(T[~hit]) if (~hit).any() else pd.DataFrame(columns=KEYS + FIS)
    if len(N):
        nuevo = T[~hit].merge(N, on=KEYS, how="left", validate="one_to_one")[CACHE_COLS]
        cache.parent.mkdir(parents=True, exist_ok=True)
        nuevo.to_csv(cache, mode="a", header=not cache.exists(), index=False)
    print(f"[clf_villar] fisica: {int(hit.sum())} llaves del cache, {len(N)} calculadas ({cache.name})", flush=True)
    partes = ([m.loc[hit, KEYS + FIS]] if hit.any() else []) + ([N[KEYS + FIS]] if len(N) else [])
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=KEYS + FIS)


def _fisica_campo(files, T, rest_frame):
    """Sims de un campo (oid = field): sus parquets <field>__*.parquet, solo g y r."""
    rows = pd.concat([pq.read_table(f, columns=LC_COLS, filters=[("filter", "in", list(BANDS))]).to_pandas()
                      for f in files], ignore_index=True) if files else pd.DataFrame(columns=LC_COLS)
    return fisica_filas(rows, T, rest_frame)


def fisica_sims(S, features_sims, run_dir=None, rest_frame=True, cache=None, n_jobs=None):
    """FIS de las sims de S (KEYS + z). Curvas de RUNS/<proy>/<field>__*.parquet, en paralelo por campo (N_JOBS)."""
    if not len(S):
        return pd.DataFrame(columns=KEYS + FIS)
    feat, run_dir = features_csv(features_sims), Path(run_dir or run_dir_for(features_sims))
    T = _tabla_spm(pd.read_csv(feat, usecols=lambda c: c in KEYS + ["filter_band"] + SPM), S)
    cache = cache or feat.parent / f"fisica_{'rest' if rest_frame else 'obs'}.csv"

    def calcular(Tn):
        files = {}
        for p in sorted(run_dir.glob("*__*.parquet")):
            files.setdefault(p.name.split("__")[0], []).append(p)
        partes = joblib.Parallel(n_jobs=n_jobs or N_JOBS)(
            joblib.delayed(_fisica_campo)(files.get(o, []), g, rest_frame) for o, g in Tn.groupby("oid", sort=True))
        return pd.concat(partes, ignore_index=True)
    return _con_cache(T, cache, run_dir / "_sims_all.parquet", calcular)


def filas_reales(real_dir, oids):
    """Curvas g, r de las oids pedidas (solo val). pyarrow filtra por oid al leer, asi las filas de la mitad final no
    se materializan (como nnclf.data.load_real_val). Se leen todos los parquets de clase: la etiqueta no elige nada."""
    oids = sorted(set(map(str, oids)))
    partes = [pq.read_table(p, columns=LC_COLS, filters=[("oid", "in", oids), ("filter", "in", list(BANDS))])
              .to_pandas() for p in sorted(Path(real_dir).glob("*.parquet")) if not p.name.startswith("_")]
    rows = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=LC_COLS)
    if not set(rows.oid.astype(str)) <= set(oids):
        raise AssertionError("se colo una oid fuera de la mitad val")
    return rows


def fisica_reales(R, v, features_real=REAL_FEAT, real_dir=REAL_DIR, rest_frame=True, cache=None):
    """FIS de las reales val de R. v: metadatos val (read_val_meta): toda oid de R tiene que estar ahi."""
    oids = set(R.oid.astype(str))
    if not oids <= set(v.oid.astype(str)):
        raise AssertionError("se colo una oid fuera de la mitad val")
    if not oids:
        return pd.DataFrame(columns=KEYS + FIS)
    feat = features_csv(features_real)
    raw = read_rows_for_oids(feat, oids)
    T = _tabla_spm(raw[[c for c in raw.columns if c in KEYS + ["filter_band"] + SPM]], R)
    cache = cache or feat.parent / f"fisica_val_{'rest' if rest_frame else 'obs'}.csv"
    return _con_cache(T, cache, Path(real_dir) / "meta_real_ztf.csv",
                      lambda Tn: fisica_filas(filas_reales(real_dir, set(Tn.oid)), Tn, rest_frame))


def con_fisica(F, X):
    """F (tabla derivada) con las columnas FIS de X (KEYS + FIS) por llave."""
    F = F.assign(oid=F.oid.astype(str), part_index=F.part_index.astype(int)).drop(columns=FIS, errors="ignore")
    X = X.assign(oid=X.oid.astype(str), part_index=X.part_index.astype(int))
    out = F.merge(X[KEYS + FIS], on=KEYS, how="left", validate="one_to_one")
    for c in FIS:
        out[c] = out[c].astype(float)
    return out


# ------------------------------------------------------------------------------------------------ pesos
def selection_weight(m_sim, w_sim, m_real, dm=0.1, sigma=0.3, clip=(0.02, 20.0)):
    """S(m) = p_real(m) / p_sim(m) (regla 5). NO recibe etiquetas: m_sim y w_sim son de todas las sims juntas y
    m_real de todas las reales val con features. Histogramas de paso dm suavizados con una gaussiana de sigma mag.
    Devuelve (S en cada sim, normalizado a media ponderada 1 y recortado a clip; tabla m, S)."""
    m_sim, w_sim, m_real = (np.asarray(a, float) for a in (m_sim, w_sim, m_real))
    oks, okr = np.isfinite(m_sim), np.isfinite(m_real)
    lo = min(m_sim[oks].min(), m_real[okr].min()) - 4 * sigma
    hi = max(m_sim[oks].max(), m_real[okr].max()) + 4 * sigma
    edges = np.arange(lo, hi + dm, dm)
    cen = 0.5 * (edges[1:] + edges[:-1])
    hs = gaussian_filter1d(np.histogram(m_sim[oks], edges, weights=w_sim[oks])[0].astype(float), sigma / dm,
                           mode="constant")
    hr = gaussian_filter1d(np.histogram(m_real[okr], edges)[0].astype(float), sigma / dm, mode="constant")
    ps, pr = hs / hs.sum(), hr / hr.sum()
    ratio = np.where(ps > 1e-9, pr / np.maximum(ps, 1e-12), clip[1])
    s = np.where(oks, np.interp(np.where(oks, m_sim, 0.0), cen, ratio), 1.0)
    for _ in range(3):
        s = np.clip(s / np.average(s, weights=w_sim), *clip)
    return s, pd.DataFrame({"m": cen, "S": ratio, "p_sim": ps, "p_real": pr})


def domain_weight(Xs, ws, Xr, seed=SEED, clip=(0.05, 20.0), n_folds=5):
    """Cociente de densidad p_real(x) / p_sim(x) de cada sim (regla 5, wz_dr), con un clasificador sim contra real
    sin etiquetas y prediccion fuera de fold. Devuelve (peso por sim con media ponderada 1, AUC fuera de fold)."""
    p, d, w = _domain_oof(Xs, ws, Xr, seed, n_folds, max_depth=3)
    ps = np.clip(p[d == 0], 1e-4, 1 - 1e-4)
    r = ps / (1 - ps)
    for _ in range(3):
        r = np.clip(r / np.average(r, weights=ws), *clip)
    return r, float(roc_auc_score(d, p, sample_weight=w))


def _domain_oof(Xs, ws, Xr, seed, n_folds, max_depth=None):
    X = np.vstack([np.asarray(Xs, float), np.asarray(Xr, float)])
    d = np.r_[np.zeros(len(Xs), int), np.ones(len(Xr), int)]
    w = np.r_[np.asarray(ws, float) * len(Xr) / np.sum(ws), np.ones(len(Xr))]   # los dos dominios pesan igual
    p = np.zeros(len(X))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed % 2**32).split(X, d):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=max_depth,
                                             max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20,
                                             random_state=seed % 2**32)
        clf.fit(X[tr], d[tr], sample_weight=w[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p, d, w


def class_balance(y, w, prior=None):
    """Cada clase k suma prior[k] del peso total (uniforme por defecto). Normalizado a media 1."""
    y, w = np.asarray(y, int), np.asarray(w, float)
    K = int(y.max()) + 1 if prior is None else len(prior)
    prior = np.full(K, 1.0 / K) if prior is None else np.asarray(prior, float)
    tot = np.bincount(y, weights=w, minlength=K)
    f = np.divide(prior * w.sum(), tot, out=np.zeros(K), where=tot > 0)
    sw = w * f[y]
    return sw / sw.mean()


# ------------------------------------------------------------------------------------------------ modelos
MODELS = {
    "hgb": dict(kind="hgb", max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                min_samples_leaf=20),
    "hgb_lento": dict(kind="hgb", max_iter=600, learning_rate=0.02, max_leaf_nodes=8, l2_regularization=3.0,
                      min_samples_leaf=40),
    "rf": dict(kind="rf", n_estimators=400, min_samples_leaf=3, max_features="sqrt"),
    "rf_suave": dict(kind="rf", n_estimators=400, min_samples_leaf=15, max_features=0.5),
    "mlp": dict(kind="mlp", hidden=(128, 64), alpha=1e-4, n_seeds=3),
    "mlp_reg": dict(kind="mlp", hidden=(64, 32), alpha=1e-2, n_seeds=3),
    "hier_mlp_II": dict(kind="hier", first="II", base="mlp"),     # receta vieja: II contra I, despues Ia contra Ibc
    "hier_mlp_Ia": dict(kind="hier", first="Ia", base="mlp"),     # Ia contra CC, despues II contra Ibc
    "hier_hgb_II": dict(kind="hier", first="II", base="hgb"),
    "hier_hgb_Ia": dict(kind="hier", first="Ia", base="hgb"),
    "hier_rf_Ia": dict(kind="hier", first="Ia", base="rf"),
    "ens": dict(kind="ens", members=("hgb", "rf_suave", "mlp", "hier_mlp_II")),
    "ens_hier": dict(kind="ens", members=("hgb", "hier_hgb_II", "hier_mlp_II")),   # lo mejor del smoke
}


def _base(spec, seed):
    k, s = spec["kind"], seed % 2**32
    if k == "hgb":
        return HistGradientBoostingClassifier(**{a: b for a, b in spec.items() if a != "kind"}, random_state=s)
    if k == "rf":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestClassifier(**{a: b for a, b in spec.items() if a != "kind"},
                                                    n_jobs=N_JOBS, random_state=s))
    if k == "mlp":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=s),
                             MLPClassifier(hidden_layer_sizes=spec["hidden"], alpha=spec["alpha"], max_iter=400,
                                           random_state=s))
    raise ValueError(k)


def _fit_base(spec, seed, X, y, sw):
    """Lista de estimadores ajustados (el MLP promedia n_seeds semillas, como el viejo)."""
    seeds = [seed + i for i in range(spec.get("n_seeds", 1))]
    out = []
    for s in seeds:
        est = _base(spec, s)
        last = est.steps[-1][0] if hasattr(est, "steps") else None
        est.fit(X, y, **({f"{last}__sample_weight": sw} if last else {"sample_weight": sw}))
        out.append(est)
    return out


def _proba(ests, X, K):
    P = np.zeros((len(X), K))
    for e in ests:
        p = e.predict_proba(X)
        P[:, e.classes_.astype(int)] += p
    return P / len(ests)


class Model:
    """Clasificador sobre la tabla de features. spec: entrada de MODELS. cols1/cols2: columnas del nivel 1 y del
    resto (iguales salvo el feature set 'viejo'). fit(F, y, w_phys) aplica el balance de clases."""

    def __init__(self, spec_name, cols1, cols2, K, seed=SEED, balance=True):
        self.spec_name, self.spec = spec_name, MODELS[spec_name]
        self.cols1, self.cols2, self.K, self.seed, self.balance = list(cols1), list(cols2), K, seed, balance

    def _sw(self, y, w, prior=None):
        return class_balance(y, w, prior) if self.balance else np.asarray(w, float) / np.mean(w)

    def fit(self, F, y, w):
        y, w, K, spec = np.asarray(y, int), np.asarray(w, float), self.K, self.spec
        if spec["kind"] == "ens":
            self.members_ = [Model(m, self.cols1, self.cols2, K, self.seed + 101 * i, self.balance).fit(F, y, w)
                             for i, m in enumerate(spec["members"])]
        elif spec["kind"] == "hier":
            base = MODELS[spec["base"]]
            first = self.first_ = list(classes(K == 4)).index(spec["first"])
            y1 = (y == first).astype(int)
            # nivel 1 con prior 1/K para la primera clase: prior efectivo final uniforme sobre las K clases
            self.l1_ = _fit_base(base, self.seed, F[self.cols1].to_numpy(float), y1,
                                 self._sw(y1, w, [1 - 1 / K, 1 / K]))
            rest = y != first
            self.rest_ = [k for k in range(K) if k != first]
            y2 = np.searchsorted(self.rest_, y[rest])
            self.l2_ = _fit_base(base, self.seed + 7, F.loc[rest, self.cols2].to_numpy(float), y2,
                                 self._sw(y2, w[rest]))
        else:
            self.ests_ = _fit_base(spec, self.seed, F[self.cols2].to_numpy(float), y, self._sw(y, w))
        return self

    def predict_proba(self, F):
        kind = self.spec["kind"]
        if kind == "ens":
            return np.mean([m.predict_proba(F) for m in self.members_], axis=0)
        if kind == "hier":
            p1 = _proba(self.l1_, F[self.cols1].to_numpy(float), 2)[:, 1]
            p2 = _proba(self.l2_, F[self.cols2].to_numpy(float), self.K - 1)
            P = np.zeros((len(F), self.K))
            P[:, self.first_] = p1
            P[:, self.rest_] = (1 - p1)[:, None] * p2
            return P
        return _proba(self.ests_, F[self.cols2].to_numpy(float), self.K)


def tiene_g(F):
    return F.m_pk_g.notna().to_numpy(bool)


class ModeloG:
    """g_modo separado (regla 4): A = Model con las sims con g y las columnas del set; B = Model con todas las sims y
    solo las columnas de r (cols_solo_r). predict_proba usa A en los objetos con g y B en los sin g. Misma interfaz
    que Model: run_config, la temperatura, el EM y las metricas no cambian."""

    def __init__(self, spec_name, cols1, cols2, K, seed=SEED, balance=True):
        self.K = K
        self.A = Model(spec_name, cols1, cols2, K, seed, balance)
        self.B = Model(spec_name, cols_solo_r(cols1), cols_solo_r(cols2), K, seed, balance)

    def fit(self, F, y, w):
        y, w, g = np.asarray(y, int), np.asarray(w, float), tiene_g(F)
        self.A.fit(F[g], y[g], w[g])
        self.B.fit(F, y, w)
        return self

    def predict_proba(self, F):
        g, P = tiene_g(F), np.zeros((len(F), self.K))
        if g.any():
            P[g] = self.A.predict_proba(F[g])
        if (~g).any():
            P[~g] = self.B.predict_proba(F[~g])
        return P


def make_model(cfg, cols1, cols2, K, seed=SEED):
    """Model o ModeloG segun cfg['g_modo'] (por defecto nan: el Model de siempre)."""
    gm = cfg.get("g_modo", "nan")
    if gm not in G_MODOS:
        raise ValueError(f"g_modo desconocido: {gm}")
    return (ModeloG if gm == "separado" else Model)(cfg["model"], cols1, cols2, K, seed, cfg.get("balance", True))


# ------------------------------------------------------------------------------------------------ evaluacion
def metrics(y, yp, cls, w=None):
    y, yp = np.asarray(y, int), np.asarray(yp, int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    K = len(cls)
    cm = np.zeros((K, K))
    np.add.at(cm, (y, yp), w)
    tp = np.diag(cm)
    rec = np.divide(tp, cm.sum(1), out=np.full(K, np.nan), where=cm.sum(1) > 0)
    pre = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros(K), where=(pre + rec) > 0)
    out = {"n": int(len(y)), "acc": float(tp.sum() / max(cm.sum(), 1e-12)), "bal_acc": float(np.nanmean(rec)),
           "f1_macro": float(np.mean(f1))}
    out.update({f"recall_{c}": float(rec[i]) for i, c in enumerate(cls)})
    out.update({f"f1_{c}": float(f1[i]) for i, c in enumerate(cls)})
    out["confusion"] = (cm.astype(int) if w is None or np.all(w == 1) else np.round(cm, 3)).tolist()
    return out


def bootstrap_ci(y, yp, cls, n_boot=1000, seed=SEED):
    """IC 95 % (percentiles 2.5 y 97.5) de acc y exactitud balanceada remuestreando las reales. Ancho tipico: +-0.045
    en acc con n ~ 300 (val completo) y +-0.065 con n ~ 150 (val_rep)."""
    y, yp = np.asarray(y, int), np.asarray(yp, int)
    rng = np.random.default_rng(seed)
    acc, bal = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        m = metrics(y[i], yp[i], cls)
        acc.append(m["acc"])
        bal.append(m["bal_acc"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"acc_ic95": q(acc), "bal_acc_ic95": q(bal)}


def calib_metrics(P, y):
    """log-loss y ECE (confianza maxima, 10 bins) de probabilidades contra etiquetas."""
    y = np.asarray(y, int)
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
    conf, hit = P.max(1), (P.argmax(1) == y)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = float(sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                    for b in range(10) if (bins == b).any()))
    return {"logloss": ll, "ece": ece}


def fit_temperature(P, y, w=None):
    """T que minimiza la log-verosimilitud ponderada de softmax(log P / T) (prediccion fuera de fold de sims)."""
    y = np.asarray(y, int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    L = np.log(np.clip(P, 1e-9, 1))

    def nll(lt):
        Q = _softmax(L / np.exp(lt))
        return -np.sum(w * np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, 1))) / w.sum()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def _softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def temper(P, T):
    return _softmax(np.log(np.clip(P, 1e-9, 1)) / T)


def em_prior(P, prior_train, n_iter=200, tol=1e-6):
    """Reajuste de prior sin etiquetas (Saerens et al. 2002). Devuelve (probabilidades reajustadas, prior)."""
    pt = np.asarray(prior_train, float)
    pi = pt.copy()
    Q = P
    for _ in range(n_iter):
        Q = P * (pi / pt)
        Q = Q / Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    return Q, pi


def apply_prior(P, pi, prior_train):
    """Probabilidades con el prior pi en lugar del de entrenamiento (la misma regla de Bayes que usa em_prior)."""
    Q = P * (np.asarray(pi, float) / np.asarray(prior_train, float))
    return Q / Q.sum(1, keepdims=True)


def template_folds(S, n_folds=5, seed=SEED):
    """Fold de cada sim: por sn_type, plantillas ordenadas, permutadas con default_rng(seed + indice del tipo) y
    repartidas en n_folds grupos (misma regla que nnclf.data.split_templates)."""
    fold = np.full(len(S), -1)
    for st in sorted(S.sn_type.unique(), key=SN_TYPES.index):
        tp = sorted(S.template[S.sn_type == st].unique())
        perm = np.random.default_rng(seed + SN_TYPES.index(st)).permutation(tp)
        for k, ch in enumerate(np.array_split(perm, n_folds)):
            fold[(S.sn_type == st).to_numpy() & S.template.isin(set(ch)).to_numpy()] = k
    if len(S) and n_folds > 1 and (fold < 0).any():
        raise AssertionError("sim sin fold")
    return fold


# ------------------------------------------------------------------------------------------------ preparacion
def prepare(features_sims, run_dir=None, features_real=REAL_FEAT, real_dir=REAL_DIR, four=False,
            rest_frame=True, requiere="any", fisica=False):
    """(sims derivadas, reales val derivadas, metadatos val). requiere: 'any' (al menos una banda) o 'r'. fisica: agrega
    las columnas FIS (curvas + SPM, mismo camino en sims y reales, con cache)."""
    cls = classes(four)
    S = derive(load_sims(features_sims, run_dir, four), rest_frame)
    Rw, v = load_real_val(features_real, real_dir, four)
    R = derive(Rw, rest_frame)
    if fisica:
        S = con_fisica(S, fisica_sims(S, features_sims, run_dir, rest_frame))
        R = con_fisica(R, fisica_reales(R, v, features_real, real_dir, rest_frame))
    if requiere == "r":
        S, R = S[S.tiene_r].reset_index(drop=True), R[R.tiene_r].reset_index(drop=True)
    ix = {c: i for i, c in enumerate(cls)}
    S["y"], R["y"] = S.cls.map(ix).astype(int), R.cls.map(ix).astype(int)
    return S, R, v


def phys_weights(S, R, peso, cols, seed=SEED):
    """w_z * peso de seleccion (regla 5). Devuelve (pesos, info). R solo aporta features, nunca etiquetas, y
    run_config le pasa solo las filas de val_sel (val_rep no ajusta nada)."""
    w = S.w_z.to_numpy(float)
    info = {"peso": peso}
    if peso == "wz_S":
        s, tab = selection_weight(S.m_sel, w, R.m_sel)
        info["S_tabla"] = tab[(tab.p_sim > 1e-4) | (tab.p_real > 1e-4)].round(5).to_dict("list")
        w = w * s
    elif peso == "wz_dr":
        r, auc = domain_weight(S[cols].to_numpy(float), w, R[cols].to_numpy(float), seed)
        info["auc_dominio"] = auc
        w = w * r
    elif peso != "wz":
        raise ValueError(peso)
    info["ess_por_clase"] = {str(c): float(w[S.cls == c].sum() ** 2 / np.sum(w[S.cls == c] ** 2))
                             for c in sorted(S.cls.unique())}
    return w, info


def sel_mask(R):
    """Filas de val_sel (las unicas reales que pueden ajustar algo: S(m), wz_dr y el prior EM)."""
    return (R["subset"] == "val_sel").to_numpy(bool)


def evaluate_real(R, P, Q, cls, n_boot=1000, seed=SEED, nn_oids=None):
    """Metricas de las reales por subconjunto (val_rep, val_sel, val) para el prior none (P) y em (Q). Cada una con
    calibracion, IC 95 % bootstrap, el subconjunto con r y, si se da nn_oids, las oids que la red clasifica."""
    y = R.y.to_numpy(int)
    tiene_r = R.tiene_r.to_numpy(bool)
    oid_nn = R.oid.astype(str).isin(nn_oids).to_numpy() if nn_oids is not None else None
    out = {}
    for tag, M in (("none", P), ("em", Q)):
        yp, by = M.argmax(1), {}
        for s in SUBSETS:
            k = np.ones(len(R), bool) if s == "val" else (R["subset"] == s).to_numpy()
            if not k.any():
                by[s] = {"n": 0}
                continue
            r = metrics(y[k], yp[k], cls)
            r.update(calib_metrics(M[k], y[k]))
            r.update(bootstrap_ci(y[k], yp[k], cls, n_boot, seed))
            kr = k & tiene_r
            r["con_r"] = metrics(y[kr], yp[kr], cls) if kr.any() else {"n": 0}
            if oid_nn is not None:
                kn = k & oid_nn
                r["nn_oids"] = metrics(y[kn], yp[kn], cls) if kn.any() else {"n": 0}
            by[s] = r
        out[tag] = by
    return out


def run_config(S, R, cfg, cls, folds=5, seed=SEED, keep_model=False, nn_oids=None):
    """Una configuracion: CV por plantilla en sims (fuera de fold) y modelo con todas las sims evaluado en las
    reales val, con prior none y em, por subconjunto (val_rep, val_sel, val). cfg: model, fset, use_z, peso,
    balance, g_modo. Las reales solo aportan features, y solo las de val_sel (S(m), wz_dr, prior EM)."""
    K = len(cls)
    c1, c2 = fset_cols(cfg["fset"], cfg["use_z"])
    cols = sorted(set(c1) | set(c2), key=(c1 + c2).index)
    sel = sel_mask(R)
    w, winfo = phys_weights(S, R[sel], cfg["peso"], cols, seed)
    winfo["reales_ajuste"] = f"val_sel ({int(sel.sum())})"
    y = S.y.to_numpy()
    res = {"config": cfg, "cols": cols, "pesos": winfo, "n_sims": int(len(S)),
           "n_sims_por_clase": {c: int((S.cls == c).sum()) for c in cls}}
    T = 1.0
    if folds > 1:
        fold = template_folds(S, folds, seed)
        P = np.zeros((len(S), K))
        for k in range(folds):
            te = fold == k
            m = make_model(cfg, c1, c2, K, seed).fit(S[~te], y[~te], w[~te])
            P[te] = m.predict_proba(S[te])
        res["cv_sims"] = metrics(y, P.argmax(1), cls, w)
        T = fit_temperature(P, y, class_balance(y, w) if cfg.get("balance", True) else w)
    res["temperatura"] = T
    model = make_model(cfg, c1, c2, K, seed).fit(S, y, w)
    Pr = temper(model.predict_proba(R), T) if len(R) else np.zeros((0, K))
    prior_tr = np.full(K, 1 / K) if cfg.get("balance", True) else np.bincount(y, weights=w, minlength=K) / w.sum()
    pi = em_prior(Pr[sel], prior_tr)[1] if sel.any() else prior_tr       # EM sin etiquetas, solo val_sel
    Qr = apply_prior(Pr, pi, prior_tr) if len(R) else Pr
    ev = evaluate_real(R, Pr, Qr, cls, 300 if not keep_model else 1000, seed, nn_oids) if len(R) else {}
    for tag in ("none", "em"):
        res[f"real_{tag}"] = ev.get(tag)
    yr = R.y.to_numpy()
    res["prior_train"] = [float(x) for x in prior_tr]
    res["prior_em"] = {c: float(pi[i]) for i, c in enumerate(cls)}
    res["prior_em_ajustado_en"] = "val_sel"
    res["prior_real_val_verdadero"] = {c: float((yr == i).mean()) for i, c in enumerate(cls)} if len(R) else None
    res["_yp"] = {"none": Pr.argmax(1), "em": Qr.argmax(1)}      # para el bootstrap pareado del barrido
    if keep_model:
        res["_model"], res["_P"], res["_Q"] = model, Pr, Qr
    return res


def coverage(R, v, cls, nn_oids=None):
    """Cobertura por subconjunto: reales con features / reales del subconjunto (de las clases pedidas). Con nn_oids,
    tambien la fraccion de las oids que clasifica la red que Villar puede clasificar."""
    out = {}
    for s in SUBSETS:
        vs = v if s == "val" else v[v["subset"] == s]
        rs = R if s == "val" else R[R["subset"] == s]
        c = {"n_real": int(len(vs)), "n_con_features": int(len(rs)), "cobertura": float(len(rs) / max(len(vs), 1)),
             "cobertura_por_clase": {k: float((rs.cls == k).sum() / max((vs.cls == k).sum(), 1)) for k in cls}}
        if nn_oids is not None:
            n_nn = int(vs.oid.astype(str).isin(nn_oids).sum())
            n_com = int(rs.oid.astype(str).isin(nn_oids).sum())
            c.update({"n_nn": n_nn, "n_comun_nn": n_com, "cobertura_sobre_nn": float(n_com / max(n_nn, 1))})
        out[s] = c
    return out


def read_nn_oids(path, R=None):
    """Oids que clasifica una corrida de nnclf (su pred_real_val.csv, o el directorio de la corrida). Con R, verifica
    que las etiquetas coincidan en las oids comunes."""
    p = Path(path)
    if p.is_dir():
        p = next((q for q in (p / "comun" / "pred_real_val.csv", p / "pred_real_val.csv") if q.exists()),
                 p / "pred_real_val.csv")
    nn = pd.read_csv(p, dtype={"oid": str}, usecols=["oid", "y_true"])
    if R is not None:
        m = R[["oid", "cls"]].astype({"oid": str}).merge(nn, on="oid")
        if not (m.cls == m.y_true).all():
            raise ValueError(f"{p}: etiquetas distintas a las de Villar en {int((m.cls != m.y_true).sum())} oids")
    return set(nn.oid)


def perm_importance(model, R, cols, cls, n_rep=10, seed=SEED):
    """Caida de exactitud balanceada en las reales al permutar cada columna (usa etiquetas: declarado)."""
    rng = np.random.default_rng(seed)
    y = R.y.to_numpy()
    base = metrics(y, model.predict_proba(R).argmax(1), cls)["bal_acc"]
    rows = []
    for c in cols:
        d = []
        for _ in range(n_rep):
            Rp = R.copy()
            Rp[c] = rng.permutation(Rp[c].to_numpy())
            d.append(base - metrics(y, model.predict_proba(Rp).argmax(1), cls)["bal_acc"])
        rows.append({"feature": c, "caida_bal_acc": float(np.mean(d)), "std": float(np.std(d))})
    return pd.DataFrame(rows).sort_values("caida_bal_acc", ascending=False)


# ------------------------------------------------------------------------------------------------ salidas
SUB_KEYS = ["n", "coverage", "acc", "bal_acc", "f1_macro", "f1_Ia", "f1_II", "f1_Ibc", "f1_IIn", "acc_con_r",
            "bal_acc_con_r", "logloss", "ece", "acc_ic95", "bal_acc_ic95", "n_nn_oids", "acc_nn_oids",
            "bal_acc_nn_oids", "f1_macro_nn_oids"]
RESUMEN_COLS = (["fecha", "name", "features_sims", "features_real", "model", "fset", "use_z", "peso", "balance",
                 "g_modo", "prior", "n_sims", "cv_bal_acc", "temperatura"]
                + [f"{k}_{SUFFIX[s]}" for s in SUBSETS for k in SUB_KEYS])
# resumen.csv anterior a la particion: sus metricas eran sobre val completo
LEGACY = {"n_real": "n_val", "cobertura": "coverage_val", **{k: f"{k}_val" for k in (
    "acc", "bal_acc", "f1_macro", "f1_Ia", "f1_II", "f1_Ibc", "acc_con_r", "bal_acc_con_r", "logloss", "ece",
    "acc_ic95", "bal_acc_ic95")}}


def resumen_rows(res, name, features_sims, cov, features_real=None):
    """Una fila por prior. Cada metrica de las reales con el sufijo _rep (honesta), _sel (con ella se elige) y _val
    (val completo, referencia), como resumen.csv de nnclf."""
    cfg, rows = res["config"], []
    for tag in ("none", "em"):
        by = res.get(f"real_{tag}") or {}
        row = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "name": name, "features_sims": str(features_sims),
               "features_real": str(features_real) if features_real else None, "model": cfg["model"],
               "fset": cfg["fset"], "use_z": cfg["use_z"], "peso": cfg["peso"], "balance": cfg.get("balance", True),
               "g_modo": cfg.get("g_modo", "nan"), "prior": tag, "n_sims": res["n_sims"],
               "cv_bal_acc": (res.get("cv_sims") or {}).get("bal_acc"), "temperatura": res.get("temperatura")}
        for s in SUBSETS:
            sf, r = SUFFIX[s], by.get(s) or {}
            cr, nn = r.get("con_r") or {}, r.get("nn_oids") or {}
            vals = {"n": r.get("n"), "coverage": cov[s]["cobertura"], "acc_con_r": cr.get("acc"),
                    "bal_acc_con_r": cr.get("bal_acc"), "n_nn_oids": nn.get("n"), "acc_nn_oids": nn.get("acc"),
                    "bal_acc_nn_oids": nn.get("bal_acc"), "f1_macro_nn_oids": nn.get("f1_macro")}
            for k in SUB_KEYS:
                row[f"{k}_{sf}"] = vals[k] if k in vals else r.get(k)
        rows.append(row)
    return rows


def append_resumen(rows, out_root=OUT_ROOT):
    p = Path(out_root) / "resumen.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    if not new:
        with open(p, newline="") as fh:
            head = next(csv.reader(fh), [])
        if head != RESUMEN_COLS:                    # columnas de una version anterior: se reescribe alineado
            old = pd.read_csv(p)
            old = old.rename(columns={k: v for k, v in LEGACY.items() if k in old and v not in old})
            old.reindex(columns=RESUMEN_COLS).to_csv(p, index=False)
    with open(p, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=RESUMEN_COLS, extrasaction="ignore")
        if new:
            wr.writeheader()
        wr.writerows(rows)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_run(out, res, R, cls, cov):
    """metrics.json, confusion_{none,em}_{rep,sel,val}.csv (filas = verdadera) y pred_real_val.csv (con la columna
    subset, el mismo formato que lee nnclf.experimentos.compare)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_jsonable({**res, "cobertura": cov}), indent=1))
    for tag in ("none", "em"):
        for s, r in (res.get(f"real_{tag}") or {}).items():
            if r.get("confusion") is not None:
                pd.DataFrame(r["confusion"], index=[f"true_{c}" for c in cls],
                             columns=[f"pred_{c}" for c in cls]).to_csv(out / f"confusion_{tag}_{SUFFIX[s]}.csv")
    if "_P" in res and len(R):
        P, Q = res["_P"], res["_Q"]
        pd.DataFrame({"oid": R.oid, "subset": R["subset"], "sn_type": R.sn_type, "y_true": R.cls,
                      "tiene_r": R.tiene_r, "y_pred": [cls[i] for i in P.argmax(1)],
                      "y_pred_em": [cls[i] for i in Q.argmax(1)], **{f"p_{c}": P[:, i] for i, c in enumerate(cls)},
                      **{f"p_em_{c}": Q[:, i] for i, c in enumerate(cls)}}).to_csv(out / "pred_real_val.csv",
                                                                                   index=False)


def importances(model, R, cols, cls, seed=SEED):
    """perm_importance en val_rep y val_sel (columna subset)."""
    parts = [perm_importance(model, R[R["subset"] == s].reset_index(drop=True), cols, cls, seed=seed).assign(subset=s)
             for s in ("val_rep", "val_sel") if (R["subset"] == s).any()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _print_res(tag, res):
    cv = res.get("cv_sims") or {}
    nan = float("nan")
    a, b = res.get("real_none") or {}, res.get("real_em") or {}
    txt = " | ".join(f"{SUFFIX[s]} n={(a.get(s) or {}).get('n')} bal {(a.get(s) or {}).get('bal_acc', nan):.3f}"
                     for s in SUBSETS)
    rep = a.get("val_rep") or {}
    print(f"[clf_villar] {tag}: cv sims bal {cv.get('bal_acc', nan):.3f} | {txt} | rep acc {rep.get('acc', nan):.3f} "
          f"IC95 bal {rep.get('bal_acc_ic95')} | em rep bal {(b.get('val_rep') or {}).get('bal_acc', nan):.3f}",
          flush=True)


# ------------------------------------------------------------------------------------------------ comandos
def finish_run(out, res, R, cls, cov, a, cfg):
    """Salidas de una corrida con modelo: metrics.json, confusiones, predicciones, model.joblib e importancias."""
    res["features_real"] = str(features_csv(a.features_real))
    res["nn_preds"] = str(a.nn_preds) if a.nn_preds else None
    write_run(out, res, R, cls, cov)
    joblib.dump({"model": res["_model"], "temperatura": res["temperatura"], "config": cfg, "cols": res["cols"],
                 "classes": cls, "obs_frame": a.obs_frame, "requiere": a.requiere,
                 "prior_train": res["prior_train"]}, out / "model.joblib")
    if len(R):
        importances(res["_model"], R, res["cols"], cls, seed=a.seed).to_csv(out / "importancias_real_val.csv",
                                                                            index=False)


def cmd_train(a):
    cls = classes(a.cuatro_clases)
    S, R, v = prepare(a.features_sims, a.run_dir, a.features_real, a.real_dir, a.cuatro_clases, not a.obs_frame,
                      a.requiere, fisica=a.fset in FSETS_FISICA)
    nn = read_nn_oids(a.nn_preds, R) if a.nn_preds else None
    cfg = {"model": a.model, "fset": a.fset, "use_z": not a.no_z, "peso": a.peso, "balance": not a.sin_balance,
           "g_modo": a.g_modo}
    res = run_config(S, R, cfg, cls, a.folds, a.seed, keep_model=True, nn_oids=nn)
    cov = coverage(R, v, cls, nn)
    finish_run(Path(a.out_root) / a.name, res, R, cls, cov, a, cfg)
    append_resumen(resumen_rows(res, a.name, a.features_sims, cov, features_csv(a.features_real)), a.out_root)
    _print_res(a.name, res)
    return res


def cmd_eval(a):
    """Carga model.joblib de <out_root>/<name> y lo evalua sobre las reales val (p. ej. tras re-extraer). El prior
    EM se reajusta sin etiquetas en val_sel."""
    out = Path(a.out_root) / a.name
    bundle = joblib.load(out / "model.joblib")     # pickle propio, escrito por cmd_train en RUNS local: confiable
    cls = bundle["classes"]
    Rw, v = load_real_val(a.features_real, a.real_dir, len(cls) == 4)
    R = derive(Rw, not bundle["obs_frame"])
    if set(FIS) & set(bundle["cols"]):
        R = con_fisica(R, fisica_reales(R, v, a.features_real, a.real_dir, not bundle["obs_frame"]))
    if bundle["requiere"] == "r":
        R = R[R.tiene_r].reset_index(drop=True)
    R["y"] = R.cls.map({c: i for i, c in enumerate(cls)}).astype(int)
    nn = read_nn_oids(a.nn_preds, R) if a.nn_preds else None
    P = temper(bundle["model"].predict_proba(R), bundle["temperatura"])
    sel, pt = sel_mask(R), np.asarray(bundle["prior_train"], float)
    pi = em_prior(P[sel], pt)[1] if sel.any() else pt
    Q = apply_prior(P, pi, pt)
    ev = evaluate_real(R, P, Q, cls, 1000, a.seed, nn)
    res = {"config": bundle["config"], "cols": bundle["cols"], "temperatura": bundle["temperatura"],
           "n_sims": None, "_P": P, "_Q": Q, "prior_em": {c: float(pi[i]) for i, c in enumerate(cls)},
           "prior_em_ajustado_en": "val_sel", "real_none": ev["none"], "real_em": ev["em"],
           "features_real": str(features_csv(a.features_real)), "nn_preds": str(a.nn_preds) if a.nn_preds else None}
    cov = coverage(R, v, cls, nn)
    write_run(out / "eval", res, R, cls, cov)
    _print_res(f"{a.name} (eval)", res)
    return res


CRITERIO = {"particion": f"pipeline78.splits.val_split (semilla {splits.SEED})",
            "prefiltro": "CV en sims por plantilla: quedan fuera las configuraciones bajo la mediana del barrido",
            "orden": "exactitud balanceada en val_sel, prior none (val_rep no entra)",
            "base": BASELINE,
            "regla": "bootstrap pareado estratificado por clase (nnclf.experimentos.paired_bootstrap) de la candidata "
                     f"contra la base sobre val_sel; la candidata queda solo si P(delta > 0) >= {P_MIN}",
            "n_boot": N_BOOT_PAREADO, "semilla_bootstrap": BOOT_SEED, "empate": "se queda la base",
            "nota": "elegido solo con val_sel; las cifras honestas son las _rep"}


def same_cfg(c, d):
    return (all(c.get(k) == d.get(k) for k in ("model", "fset", "use_z", "peso"))
            and c.get("balance", True) == d.get("balance", True) and c.get("g_modo", "nan") == d.get("g_modo", "nan"))


def select_nested(cfgs, cv, yp, R, cls, base_i, p_min=P_MIN, n_boot=N_BOOT_PAREADO, seed=BOOT_SEED):
    """Regla de eleccion del barrido (regla 7). cfgs: configuraciones; cv: exactitud balanceada de la CV en sims de
    cada una (NaN sin CV); yp: argmax (prior none) sobre las filas de R, None si la corrida fallo; base_i: indice de
    BASELINE en cfgs. Solo mira las filas de val_sel de R."""
    sel = sel_mask(R)
    if not sel.any():
        raise SystemExit("sin reales de val_sel con features: no se puede elegir")
    y = R.y.to_numpy(int)[sel]
    ok = [i for i, p in enumerate(yp) if p is not None]
    if base_i not in ok:
        raise SystemExit(f"la base {BASELINE} fallo: no hay incumbente")
    bal = {i: metrics(y, np.asarray(yp[i])[sel], cls)["bal_acc"] for i in ok}
    cvv = np.array([cv[i] for i in ok], float)
    med = float(np.median(cvv)) if len(ok) > 1 and np.isfinite(cvv).all() else None
    pasa = {i: bool(med is None or cv[i] >= med) for i in ok}
    cand = sorted((i for i in ok if pasa[i]), key=lambda i: (-bal[i], -np.nan_to_num(cv[i], nan=-1.0), i))
    top, comp = cand[0], None
    if top == base_i:
        eleg, motivo = base_i, "la primera en val_sel entre las que pasan el pre-filtro es la base"
    else:
        delta, p, ci = paired_bootstrap(y, np.asarray(yp[top])[sel] == y, np.asarray(yp[base_i])[sel] == y,
                                        n_boot, seed)
        comp = {"subconjunto": "val_sel", "n": int(sel.sum()), "bal_acc_cand_sel": bal[top],
                "bal_acc_base_sel": bal[base_i], "delta": delta, "p_mejora": p, "ic90_delta": list(ci),
                "n_boot": n_boot, "p_min": p_min, "gana": bool(p >= p_min)}
        eleg = top if comp["gana"] else base_i
        motivo = (f"la candidata gana a la base (P(delta > 0) = {p:.3f})" if comp["gana"]
                  else f"la candidata no gana a la base (P(delta > 0) = {p:.3f} < {p_min}); queda la base")
    return {"criterio": CRITERIO, "prefiltro_cv": {"mediana": med, "n_total": len(ok), "n_pasan": sum(pasa.values())},
            "ranking_sel": [{**cfgs[i], "cv_bal_acc": cv[i], "bal_acc_sel": bal[i]} for i in cand[:10]],
            "candidata": cfgs[top], "base": cfgs[base_i], "comparacion": comp, "elegida": cfgs[eleg],
            "motivo": motivo, "i_elegida": eleg, "_pasa": pasa, "_bal_sel": bal}


GRIDS = {
    "rapido": dict(fset=("viejo", "rg"), model=("hgb", "rf", "hier_mlp_II"), use_z=(True,), peso=("wz", "wz_S")),
    "foco": dict(fset=("viejo", "rg", "rg_robusto", "rg_err"),
                 model=("hgb", "hgb_lento", "hier_hgb_II", "hier_hgb_Ia", "hier_mlp_II", "hier_mlp_Ia", "ens_hier"),
                 use_z=(True, False), peso=("wz", "wz_S", "wz_dr")),
    "completo": dict(fset=FSETS, model=tuple(MODELS), use_z=(True, False), peso=("wz", "wz_S", "wz_dr")),
    # mitigaciones del estudio del gap (regla 4): t_rise censurado y modelos separados con y sin g
    "mitig": dict(fset=("rg", "rg_cens"), model=("hgb", "hgb_lento"), use_z=(True,), peso=("wz",),
                  g_modo=("nan", "separado")),
    # hombro de r y evolucion de color (seccion fisica) contra rg, planos y jerarquicos; la base no cambia
    "fisica": dict(fset=("rg", "rg_fisica"), model=("hgb", "hgb_lento", "hier_hgb_II", "ens_hier"), use_z=(True,),
                   peso=("wz",)),
}


def _one(S, R, cfg, cls, folds, seed, nn_oids=None):
    t0 = time.time()
    _silenciar_matmul()
    try:
        with threadpool_limits(N_JOBS):                               # tambien en los workers de joblib
            r = run_config(S, R, cfg, cls, folds, seed, nn_oids=nn_oids)
    except Exception as e:                                           # una configuracion rota no tumba el barrido
        return {"config": cfg, "error": repr(e)[:300]}
    r["segundos"] = round(time.time() - t0, 1)
    return r


def cmd_sweep(a):
    cls = classes(a.cuatro_clases)
    g = GRIDS[a.grid]
    models = a.models.split(",") if a.models else g["model"]
    fsets = a.fsets.split(",") if a.fsets else g["fset"]
    S, R, v = prepare(a.features_sims, a.run_dir, a.features_real, a.real_dir, a.cuatro_clases, not a.obs_frame,
                      a.requiere, fisica=bool(set(fsets) & set(FSETS_FISICA)))
    nn = read_nn_oids(a.nn_preds, R) if a.nn_preds else None
    cov = coverage(R, v, cls, nn)
    cfgs = [{"model": m, "fset": f, "use_z": z, "peso": p, "balance": True, "g_modo": gm}
            for f in fsets for m in models for z in g["use_z"] for p in g["peso"] for gm in g.get("g_modo", ("nan",))]
    base_i = next((i for i, c in enumerate(cfgs) if same_cfg(c, BASELINE)), None)
    if base_i is None:                                               # la base siempre se corre: es la incumbente
        cfgs.append(dict(BASELINE))
        base_i = len(cfgs) - 1
    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[clf_villar] sweep {a.name}: {len(cfgs)} configuraciones, sims {len(S)}, reales val con features "
          f"{len(R)}/{len(v)} (val_sel {int(sel_mask(R).sum())}, val_rep {int((R['subset'] == 'val_rep').sum())})",
          flush=True)
    if a.jobs > 1:
        results = joblib.Parallel(n_jobs=a.jobs, verbose=10)(
            joblib.delayed(_one)(S, R, c, cls, a.folds, a.seed, nn) for c in cfgs)
    else:
        results = []
        for i, c in enumerate(cfgs):
            results.append(_one(S, R, c, cls, a.folds, a.seed, nn))
            if "error" not in results[-1]:
                _print_res(f"{i + 1}/{len(cfgs)} {c}", results[-1])
            else:
                print(f"[clf_villar] {i + 1}/{len(cfgs)} {c}: ERROR {results[-1]['error']}", flush=True)
    rows = []
    for i, r in enumerate(results):
        if "error" in r:
            rows.append({**r["config"], "i_cfg": i, "error": r["error"]})
            continue
        for row in resumen_rows(r, a.name, a.features_sims, cov, features_csv(a.features_real)):
            rows.append({**row, "i_cfg": i, "segundos": r["segundos"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "sweep.csv", index=False)
    append_resumen([r for r in rows if "error" not in r], a.out_root)
    (out / "sweep_full.json").write_text(json.dumps(_jsonable([r for r in results]), indent=1))
    # eleccion anidada (regla 7): solo val_sel
    cv = [np.nan if "error" in r else (r.get("cv_sims") or {}).get("bal_acc", np.nan) for r in results]
    yp = [None if "error" in r else r["_yp"]["none"] for r in results]
    el = select_nested(cfgs, cv, yp, R, cls, base_i)
    tab["pasa_cv"] = tab.i_cfg.map(el["_pasa"])
    tab["es_base"], tab["elegida"] = tab.i_cfg == base_i, tab.i_cfg == el["i_elegida"]
    tab.to_csv(out / "sweep.csv", index=False)
    cfg = el["elegida"]
    res = run_config(S, R, cfg, cls, a.folds, a.seed, keep_model=True, nn_oids=nn)
    finish_run(out / "mejor", res, R, cls, cov, a, cfg)
    honest = {SUFFIX[s]: {k: (res["real_none"].get(s) or {}).get(k) for k in ("n", "acc", "bal_acc", "f1_macro",
                                                                             "acc_ic95", "bal_acc_ic95")}
              for s in SUBSETS}
    (out / "mejor.json").write_text(json.dumps(_jsonable({**el, "metricas_elegida_none": honest,
                                                          "cobertura": cov}), indent=1))
    top = tab[(tab.prior == "none") & tab.pasa_cv.fillna(False).astype(bool)].sort_values(
        ["bal_acc_sel", "cv_bal_acc"], ascending=False)
    print(top.head(10)[["model", "fset", "use_z", "peso", "g_modo", "cv_bal_acc", "bal_acc_sel", "bal_acc_rep",
                        "bal_acc_val"]].round(3).to_string(index=False), flush=True)
    c = el["comparacion"]
    print(f"[clf_villar] pre-filtro CV: {el['prefiltro_cv']['n_pasan']}/{el['prefiltro_cv']['n_total']} "
          f"(mediana {el['prefiltro_cv']['mediana']}). Candidata {el['candidata']}"
          + (f": delta {c['delta']:+.3f}, P(delta > 0) = {c['p_mejora']:.3f}" if c else "") + f". {el['motivo']}",
          flush=True)
    _print_res(f"{a.name} elegida {cfg}", res)
    return tab


def cmd_gap(a):
    """Clasificador sim contra real sin etiquetas (regla 5 del brief): AUC fuera de fold, importancia por
    permutacion (caida de AUC en el fold de prueba) y AUC univariado de cada feature."""
    cls = classes(a.cuatro_clases)
    S, R, v = prepare(a.features_sims, a.run_dir, a.features_real, a.real_dir, a.cuatro_clases, not a.obs_frame,
                      a.requiere)
    cols = [c for c in GAP_COLS if (a.no_z is False or not c.startswith("M_pk"))]
    w = S.w_z.to_numpy(float)
    if a.peso == "wz_S":                                             # S(m) solo de val_sel, como en el barrido
        w = w * selection_weight(S.m_sel, w, R.m_sel[sel_mask(R)])[0]
    Xs, Xr = S[cols].to_numpy(float), R[cols].to_numpy(float)
    X = np.vstack([Xs, Xr])
    d = np.r_[np.zeros(len(Xs), int), np.ones(len(Xr), int)]
    ww = np.r_[w * len(Xr) / w.sum(), np.ones(len(Xr))]
    rng = np.random.default_rng(a.seed)
    p = np.zeros(len(X))
    drop = np.zeros((a.folds, len(cols)))
    for k, (tr, te) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=a.seed % 2**32).split(X, d)):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, min_samples_leaf=20, random_state=a.seed % 2**32)
        clf.fit(X[tr], d[tr], sample_weight=ww[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
        base = roc_auc_score(d[te], p[te], sample_weight=ww[te])
        for j in range(len(cols)):
            aucs = []
            for _ in range(3):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                aucs.append(roc_auc_score(d[te], clf.predict_proba(Xp)[:, 1], sample_weight=ww[te]))
            drop[k, j] = base - np.mean(aucs)
    auc = float(roc_auc_score(d, p, sample_weight=ww))
    rows = []
    for j, c in enumerate(cols):
        xs, xr = Xs[:, j], Xr[:, j]
        oks, okr = np.isfinite(xs), np.isfinite(xr)
        u = np.nan
        if oks.sum() > 5 and okr.sum() > 5:
            u = roc_auc_score(np.r_[np.zeros(oks.sum()), np.ones(okr.sum())], np.r_[xs[oks], xr[okr]],
                              sample_weight=np.r_[w[oks] * okr.sum() / w[oks].sum(), np.ones(okr.sum())])
        rows.append({"feature": c, "caida_auc": float(drop[:, j].mean()), "auc_univariado": float(u),
                     "sep_univariada": float(abs(u - 0.5)) if np.isfinite(u) else np.nan,
                     "mediana_sim": _wmedian(xs[oks], w[oks]), "mediana_real": float(np.median(xr[okr]))
                     if okr.any() else np.nan, "frac_nan_sim": float(np.average(~oks, weights=w)),
                     "frac_nan_real": float((~okr).mean())})
    imp = pd.DataFrame(rows).sort_values("caida_auc", ascending=False)
    # diagnostico extra con etiquetas de la mitad val (declarado): medianas por clase de las features principales
    por_clase = []
    for c in cls:
        for f in _forma("r") + ["M_pk_r", "color_gr", "n_points_r"]:
            xs, xr = S.loc[S.cls == c, f].to_numpy(float), R.loc[R.cls == c, f].to_numpy(float)
            ws = w[(S.cls == c).to_numpy()]
            por_clase.append({"clase": c, "feature": f, "mediana_sim": _wmedian(xs[np.isfinite(xs)],
                                                                                ws[np.isfinite(xs)]),
                              "mediana_real": float(np.nanmedian(xr)) if np.isfinite(xr).any() else np.nan})
    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out / "gap_importancias.csv", index=False)
    pd.DataFrame(por_clase).to_csv(out / "gap_medianas_por_clase.csv", index=False)
    # ponytail: usa val completo (sel + rep). Es diagnostico de la simulacion: no elige configuraciones ni ajusta nada.
    # Restringirlo a val_sel rompe la invariancia a etiquetas (val_split estratifica por clase).
    res = {"auc_sim_vs_real": auc, "n_sims": int(len(S)), "n_real": int(len(R)), "peso": a.peso,
           "reales": "val completo (val_sel + val_rep), solo diagnostico",
           "top": imp.head(10).to_dict("records")}
    (out / "gap.json").write_text(json.dumps(_jsonable(res), indent=1))
    print(f"[clf_villar] gap {a.name}: AUC sim contra real {auc:.3f} (0.5 = indistinguibles)", flush=True)
    print(imp.head(12).round(3).to_string(index=False), flush=True)
    return res


def _wmedian(x, w):
    if not len(x):
        return np.nan
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def parser():
    ap = argparse.ArgumentParser(prog="python -m pipeline78.clf_villar", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("train", "eval", "sweep", "gap"))
    ap.add_argument("--features-sims", type=Path, help="RUNS/features_<proyeccion> (con features/features.csv)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-dir", type=Path, default=None, help="por defecto RUNS/<proyeccion>")
    ap.add_argument("--features-real", type=Path, default=REAL_FEAT,
                    help="features.csv de las reales (o su directorio features_<x>); por defecto las post-logfix")
    ap.add_argument("--real-dir", type=Path, default=REAL_DIR, help="directorio con meta_real_ztf.csv")
    ap.add_argument("--nn-preds", type=Path, default=None,
                    help="pred_real_val.csv (o directorio) de una corrida de nnclf: metricas tambien sobre sus oids")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--model", default="hgb", choices=tuple(MODELS))
    ap.add_argument("--fset", default="rg", choices=FSETS)
    ap.add_argument("--no-z", action="store_true", help="sin magnitud absoluta (los tiempos siguen en reposo)")
    ap.add_argument("--obs-frame", action="store_true", help="tiempos en el marco observado (sin 1/(1+z))")
    ap.add_argument("--peso", default="wz", choices=("wz", "wz_S", "wz_dr"))
    ap.add_argument("--sin-balance", action="store_true", help="sin balance de clases (prior = w_z * seleccion)")
    ap.add_argument("--g-modo", default="nan", choices=G_MODOS,
                    help="train: nan (g faltante = NaN) o separado (modelo A con g, B solo r); sweep: la grilla")
    ap.add_argument("--requiere", default="any", choices=("any", "r"))
    ap.add_argument("--cuatro-clases", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--grid", default="rapido", choices=tuple(GRIDS))
    ap.add_argument("--models", default=None, help="sweep: lista separada por comas (pisa la grilla)")
    ap.add_argument("--fsets", default=None, help="sweep: lista separada por comas (pisa la grilla)")
    ap.add_argument("--jobs", type=int, default=1)
    return ap


def _silenciar_matmul():
    """numpy 2.2 con Accelerate (macOS arm64) avisa divide by zero/overflow en matmul con matrices finitas
    (verificado: un a @ b de numeros aleatorios dispara los tres avisos y el resultado es finito)."""
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def main(argv=None):
    _silenciar_matmul()
    a = parser().parse_args(argv)
    if a.cmd != "eval" and a.features_sims is None:
        raise SystemExit("--features-sims es obligatorio para train, sweep y gap")
    with threadpool_limits(N_JOBS):
        return {"train": cmd_train, "eval": cmd_eval, "sweep": cmd_sweep, "gap": cmd_gap}[a.cmd](a)


if __name__ == "__main__":
    main()
