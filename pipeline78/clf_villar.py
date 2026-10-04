"""Clasificador oficial sobre las features de Villar (SPM + MCMC ref). villar-clf-brief, 2026-10-04.

Uso: python -m pipeline78.clf_villar {train|eval|sweep|gap} --features-sims DIR --name NOMBRE [opciones]

REGLAS
1. Llaves. Todo merge va por (oid, part_index, sn_type): r con g, y features con metadata. En las sims oid = field
   de _sims_all. El pipeline viejo (opt_clasificador/05_faseC*) juntaba r con g por (oid, part_index) sin sn_type
   y el color salia de otro tipo (bug H4: 13 063 filas pasaban a 26 082). Aca un duplicado de llave es un error.
2. Reales. Solo origen == holdout & split == val & ~excluir. meta_real_ztf.csv y el features.csv real se leen con
   csv linea a linea y solo las filas val llegan a pandas: la mitad final no se carga nunca (test que lo vigila).
3. Clases. Ia, II (= II + IIb), Ibc. IIn fuera de la metrica principal (--cuatro-clases la agrega).
4. Features por banda b (r, g): m_pk_b = -2.5 log10(A_b) (pico aparente desde A), M_pk_b = m_pk_b - mu(z) (solo
   con z, LCDM plano H0 = 70, Om = 0.3 como core.utils.DL_calculator), f_b, t_rise_b, t_fall_b, gamma_b en reposo
   (divididos por 1 + z, en sims y en reales) y errores relativos err/|valor| de A, f, t_rise, t_fall, gamma.
   Colores: color_gr = -2.5 log10(A_g / A_r) y diferencias g - r de t_rise, t_fall, gamma (en reposo) y de f.
   t0 no entra: cada banda tiene su propio origen (la primera deteccion de esa banda en reader.mjd_to_phase).
   Real sin z (4 de las val): tiempos en el marco observado y M_pk = NaN.
   Faltantes (p. ej. sin g) quedan NaN: HistGradientBoosting los acepta y RF/MLP imputan la mediana con bandera.
5. Pesos de entrenamiento: w = w_z (ya trae 1/(1+z)) * peso de seleccion, y despues balance de clases (cada clase
   suma lo mismo; en el jerarquico, cada nivel queda con prior efectivo uniforme sobre las clases).
   Peso de seleccion (--peso):
     wz    nada mas.
     wz_S  S(m) = p_real(m) / p_sim(m), cociente de histogramas de la magnitud de pico aparente (r, o g si no hay
           r) suavizados con una gaussiana. Sims de todas las clases juntas ponderadas por w_z, contra las reales
           val con features. NO usa etiquetas (la funcion no las recibe). Usa las reales val: declarado.
     wz_dr cociente de densidad p_real(x)/p_sim(x) en todo el espacio de features, estimado con un clasificador
           sim contra real (validacion cruzada, sin etiquetas). Tambien usa las reales val: declarado.
6. Prior (--prior). none: argmax de las probabilidades del modelo (prior de entrenamiento uniforme).
   em: reajuste de prior por EM sobre las reales sin etiquetas (Saerens, Latinne y Decaestecker 2002). Corre sobre
   las probabilidades calibradas con temperatura (ajustada en las predicciones fuera de fold de las sims).
7. Seleccion de hiperparametros: validacion cruzada en las sims agrupada por PLANTILLA (por sn_type, II e IIb
   aparte) y, como segundo criterio, las metricas sobre las reales val. El barrido ordena por la exactitud
   balanceada en las reales val (declarado: la meta es la clasificacion real, y la mitad final queda intocada).
"""
import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

from pipeline78.paths import RUNS

REAL_DIR = RUNS / "real_ztf"
REAL_FEAT = RUNS / "features_real_ztf"
OUT_ROOT = RUNS / "clf_villar"
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
def widen(df):
    """Filas por (llave, banda) -> una fila por llave con sufijo _r/_g. Llaves (oid, part_index, sn_type)."""
    d = df.copy()
    d["filter_band"] = d.filter_band.astype(str).str.lower().str[-1]
    d = d[d.filter_band.isin(BANDS)]
    dup = d.duplicated(KEYS + ["filter_band"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} filas repetidas por (oid, part_index, sn_type, banda)")
    cols = [c for c in RAW if c in d.columns]
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


def _es_val(row):
    return (row.get("origen") == "holdout" and row.get("split") == "val"
            and str(row.get("excluir", "")).strip().lower() not in ("true", "1"))


def read_val_meta(meta_path, four=False):
    """Metadatos de las reales val (regla 2). Las filas de otros splits no se guardan: la segunda pasada solo
    verifica que ninguna oid val aparezca en ellas."""
    with open(meta_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if _es_val(r)]
    v = _to_num(pd.DataFrame(rows))
    vo = set(v.oid)
    if len(vo) != len(v):
        raise ValueError("oid repetida en la mitad val")
    with open(meta_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("origen") == "holdout" and r.get("split") == "val") and r["oid"] in vo:
                raise ValueError(f"la oid val {r['oid']} aparece tambien fuera de val")
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
    v = read_val_meta(Path(real_dir) / "meta_real_ztf.csv", four)
    f = read_rows_for_oids(features_csv(features_real), v.oid)
    if not set(f.oid) <= set(v.oid):
        raise AssertionError("se colo una oid fuera de la mitad val")
    if len(f):
        f["part_index"] = f.part_index.astype(int)
        W = widen(f).merge(v[KEYS + ["z", "subtipo", "cls"]], on=KEYS, how="inner", validate="one_to_one")
    else:
        W = pd.DataFrame(columns=KEYS + ["z", "subtipo", "cls"])
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
    return F.replace([np.inf, -np.inf], np.nan)


def _forma(b):
    return [f"f_{b}", f"t_rise_{b}", f"t_fall_{b}", f"gamma_{b}"]


COLOR = ["color_gr", "d_t_rise_gr", "d_t_fall_gr", "d_gamma_gr", "d_f_gr"]
REL = [f"rel_{p}_{b}" for b in BANDS for p in PARS]
FSETS = ("viejo", "forma_r", "rg", "rg_err", "rg_m", "rg_robusto")


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
            "rg_robusto": [c for c in rg if "t_rise" not in c]}
    if name not in sets:
        raise ValueError(f"feature set desconocido: {name}")
    return sets[name], sets[name]


GAP_COLS = (_forma("r") + _forma("g") + COLOR + ["M_pk_r", "M_pk_g", "m_pk_r", "m_pk_g"] + REL
            + [f"{q}_{b}" for b in BANDS for q in QUAL])


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
    """IC 95 % de acc y exactitud balanceada remuestreando las reales (con n ~ 300, +-0.025 en acc)."""
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
            rest_frame=True, requiere="any"):
    """(sims derivadas, reales val derivadas, metadatos val). requiere: 'any' (al menos una banda) o 'r'."""
    cls = classes(four)
    S = derive(load_sims(features_sims, run_dir, four), rest_frame)
    Rw, v = load_real_val(features_real, real_dir, four)
    R = derive(Rw, rest_frame)
    if requiere == "r":
        S, R = S[S.tiene_r].reset_index(drop=True), R[R.tiene_r].reset_index(drop=True)
    ix = {c: i for i, c in enumerate(cls)}
    S["y"], R["y"] = S.cls.map(ix).astype(int), R.cls.map(ix).astype(int)
    return S, R, v


def phys_weights(S, R, peso, cols, seed=SEED):
    """w_z * peso de seleccion (regla 5). Devuelve (pesos, info). R solo aporta features, nunca etiquetas."""
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


def run_config(S, R, cfg, cls, folds=5, seed=SEED, keep_model=False):
    """Una configuracion: CV por plantilla en sims (fuera de fold) y modelo con todas las sims evaluado en las
    reales val, con prior none y em. cfg: model, fset, use_z, peso, balance."""
    K = len(cls)
    c1, c2 = fset_cols(cfg["fset"], cfg["use_z"])
    cols = sorted(set(c1) | set(c2), key=(c1 + c2).index)
    w, winfo = phys_weights(S, R, cfg["peso"], cols, seed)
    y = S.y.to_numpy()
    res = {"config": cfg, "cols": cols, "pesos": winfo, "n_sims": int(len(S)),
           "n_sims_por_clase": {c: int((S.cls == c).sum()) for c in cls}}
    T = 1.0
    if folds > 1:
        fold = template_folds(S, folds, seed)
        P = np.zeros((len(S), K))
        for k in range(folds):
            te = fold == k
            m = Model(cfg["model"], c1, c2, K, seed, cfg.get("balance", True)).fit(S[~te], y[~te], w[~te])
            P[te] = m.predict_proba(S[te])
        res["cv_sims"] = metrics(y, P.argmax(1), cls, w)
        T = fit_temperature(P, y, class_balance(y, w) if cfg.get("balance", True) else w)
    res["temperatura"] = T
    model = Model(cfg["model"], c1, c2, K, seed, cfg.get("balance", True)).fit(S, y, w)
    Pr = temper(model.predict_proba(R), T) if len(R) else np.zeros((0, K))
    prior_tr = np.full(K, 1 / K) if cfg.get("balance", True) else np.bincount(y, weights=w, minlength=K) / w.sum()
    Qr, pi = em_prior(Pr, prior_tr) if len(R) else (Pr, prior_tr)
    yr = R.y.to_numpy()
    for tag, Q in (("none", Pr), ("em", Qr)):
        res[f"real_{tag}"] = metrics(yr, Q.argmax(1), cls) if len(R) else None
        if len(R):
            res[f"real_{tag}"].update(calib_metrics(Q, yr))
            res[f"real_{tag}"].update(bootstrap_ci(yr, Q.argmax(1), cls, 300 if not keep_model else 1000, seed))
            con_r = R.tiene_r.to_numpy()
            res[f"real_{tag}"]["con_r"] = metrics(yr[con_r], Q[con_r].argmax(1), cls)
    res["prior_train"] = [float(x) for x in prior_tr]
    res["prior_em"] = {c: float(pi[i]) for i, c in enumerate(cls)}
    res["prior_real_val_verdadero"] = {c: float((yr == i).mean()) for i, c in enumerate(cls)} if len(R) else None
    if keep_model:
        res["_model"], res["_P"], res["_Q"] = model, Pr, Qr
    return res


def coverage(R, v, cls):
    return {"n_real_val": int(len(v)), "n_con_features": int(len(R)), "cobertura": float(len(R) / max(len(v), 1)),
            "cobertura_por_clase": {c: float((R.cls == c).sum() / max((v.cls == c).sum(), 1)) for c in cls}}


def perm_importance(model, R, cols, cls, n_rep=10, seed=SEED):
    """Caida de exactitud balanceada en las reales val al permutar cada columna (usa etiquetas val: declarado)."""
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
RESUMEN_COLS = ["fecha", "name", "features_sims", "model", "fset", "use_z", "peso", "balance", "prior", "n_sims",
                "cv_bal_acc", "n_real", "cobertura", "acc", "bal_acc", "f1_macro", "f1_Ia", "f1_II", "f1_Ibc",
                "acc_con_r", "bal_acc_con_r", "logloss", "ece", "acc_ic95", "bal_acc_ic95"]


def resumen_rows(res, name, features_sims, cov):
    cfg, rows = res["config"], []
    for tag in ("none", "em"):
        r = res.get(f"real_{tag}") or {}
        rows.append({"fecha": time.strftime("%Y-%m-%d %H:%M"), "name": name, "features_sims": str(features_sims),
                     "model": cfg["model"], "fset": cfg["fset"], "use_z": cfg["use_z"], "peso": cfg["peso"],
                     "balance": cfg.get("balance", True), "prior": tag, "n_sims": res["n_sims"],
                     "cv_bal_acc": (res.get("cv_sims") or {}).get("bal_acc"), "n_real": r.get("n"),
                     "cobertura": cov["cobertura"], "acc": r.get("acc"), "bal_acc": r.get("bal_acc"),
                     "f1_macro": r.get("f1_macro"), "f1_Ia": r.get("f1_Ia"), "f1_II": r.get("f1_II"),
                     "f1_Ibc": r.get("f1_Ibc"), "acc_con_r": (r.get("con_r") or {}).get("acc"),
                     "bal_acc_con_r": (r.get("con_r") or {}).get("bal_acc"), "logloss": r.get("logloss"),
                     "ece": r.get("ece"), "acc_ic95": r.get("acc_ic95"), "bal_acc_ic95": r.get("bal_acc_ic95")})
    return rows


def append_resumen(rows, out_root=OUT_ROOT):
    p = Path(out_root) / "resumen.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    if not new:
        with open(p, newline="") as fh:
            head = next(csv.reader(fh), [])
        if head != RESUMEN_COLS:                    # columnas de una version anterior: se reescribe alineado
            pd.read_csv(p).reindex(columns=RESUMEN_COLS).to_csv(p, index=False)
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
    """metrics.json, confusion_real_val_{none,em}.csv y pred_real_val.csv."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_jsonable({**res, "cobertura": cov}), indent=1))
    for tag in ("none", "em"):
        r = res.get(f"real_{tag}")
        if r:
            pd.DataFrame(r["confusion"], index=[f"true_{c}" for c in cls],
                         columns=[f"pred_{c}" for c in cls]).to_csv(out / f"confusion_real_val_{tag}.csv")
    if "_P" in res and len(R):
        P, Q = res["_P"], res["_Q"]
        pd.DataFrame({"oid": R.oid, "sn_type": R.sn_type, "y_true": R.cls, "tiene_r": R.tiene_r,
                      "y_pred": [cls[i] for i in P.argmax(1)], "y_pred_em": [cls[i] for i in Q.argmax(1)],
                      **{f"p_{c}": P[:, i] for i, c in enumerate(cls)},
                      **{f"p_em_{c}": Q[:, i] for i, c in enumerate(cls)}}).to_csv(out / "pred_real_val.csv",
                                                                                   index=False)


def _print_res(tag, res):
    cv = res.get("cv_sims") or {}
    a, b = res.get("real_none") or {}, res.get("real_em") or {}
    print(f"[clf_villar] {tag}: cv sims bal {cv.get('bal_acc', float('nan')):.3f} | reales n={a.get('n')} "
          f"acc {a.get('acc', float('nan')):.3f} bal {a.get('bal_acc', float('nan')):.3f} "
          f"f1 {a.get('f1_macro', float('nan')):.3f} IC95 bal {a.get('bal_acc_ic95')} | em: acc {b.get('acc', float('nan')):.3f} "
          f"bal {b.get('bal_acc', float('nan')):.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ comandos
def cmd_train(a):
    cls = classes(a.cuatro_clases)
    S, R, v = prepare(a.features_sims, a.run_dir, a.features_real, a.real_dir, a.cuatro_clases, not a.obs_frame,
                      a.requiere)
    cfg = {"model": a.model, "fset": a.fset, "use_z": not a.no_z, "peso": a.peso, "balance": not a.sin_balance}
    res = run_config(S, R, cfg, cls, a.folds, a.seed, keep_model=True)
    cov = coverage(R, v, cls)
    out = Path(a.out_root) / a.name
    write_run(out, res, R, cls, cov)
    joblib.dump({"model": res["_model"], "temperatura": res["temperatura"], "config": cfg, "cols": res["cols"],
                 "classes": cls, "obs_frame": a.obs_frame, "requiere": a.requiere,
                 "prior_train": res["prior_train"]}, out / "model.joblib")
    if len(R):
        perm_importance(res["_model"], R, res["cols"], cls, seed=a.seed).to_csv(out / "importancias_real_val.csv",
                                                                                index=False)
    append_resumen(resumen_rows(res, a.name, a.features_sims, cov), a.out_root)
    _print_res(a.name, res)
    return res


def cmd_eval(a):
    """Carga model.joblib de <out_root>/<name> y lo evalua sobre las reales val (p. ej. tras re-extraer)."""
    out = Path(a.out_root) / a.name
    bundle = joblib.load(out / "model.joblib")     # pickle propio, escrito por cmd_train en RUNS local: confiable
    cls = bundle["classes"]
    Rw, v = load_real_val(a.features_real, a.real_dir, len(cls) == 4)
    R = derive(Rw, not bundle["obs_frame"])
    if bundle["requiere"] == "r":
        R = R[R.tiene_r].reset_index(drop=True)
    R["y"] = R.cls.map({c: i for i, c in enumerate(cls)}).astype(int)
    model, K = bundle["model"], len(cls)
    P = temper(model.predict_proba(R), bundle["temperatura"])
    Q, pi = em_prior(P, bundle["prior_train"])
    res = {"config": bundle["config"], "cols": bundle["cols"], "temperatura": bundle["temperatura"],
           "n_sims": None, "_P": P, "_Q": Q, "prior_em": {c: float(pi[i]) for i, c in enumerate(cls)}}
    for tag, M in (("none", P), ("em", Q)):
        res[f"real_{tag}"] = {**metrics(R.y, M.argmax(1), cls), **calib_metrics(M, R.y.to_numpy()),
                              **bootstrap_ci(R.y, M.argmax(1), cls)}
    cov = coverage(R, v, cls)
    write_run(out / "eval", res, R, cls, cov)
    _print_res(f"{a.name} (eval)", res)
    return res


GRIDS = {
    "rapido": dict(fset=("viejo", "rg"), model=("hgb", "rf", "hier_mlp_II"), use_z=(True,), peso=("wz", "wz_S")),
    "foco": dict(fset=("viejo", "rg", "rg_robusto", "rg_err"),
                 model=("hgb", "hgb_lento", "hier_hgb_II", "hier_hgb_Ia", "hier_mlp_II", "hier_mlp_Ia", "ens_hier"),
                 use_z=(True, False), peso=("wz", "wz_S", "wz_dr")),
    "completo": dict(fset=FSETS, model=tuple(MODELS), use_z=(True, False), peso=("wz", "wz_S", "wz_dr")),
}


def _one(S, R, cfg, cls, folds, seed):
    t0 = time.time()
    _silenciar_matmul()
    try:
        with threadpool_limits(N_JOBS):                               # tambien en los workers de joblib
            r = run_config(S, R, cfg, cls, folds, seed)
    except Exception as e:                                           # una configuracion rota no tumba el barrido
        return {"config": cfg, "error": repr(e)[:300]}
    r["segundos"] = round(time.time() - t0, 1)
    return r


def cmd_sweep(a):
    cls = classes(a.cuatro_clases)
    S, R, v = prepare(a.features_sims, a.run_dir, a.features_real, a.real_dir, a.cuatro_clases, not a.obs_frame,
                      a.requiere)
    cov = coverage(R, v, cls)
    g = GRIDS[a.grid]
    models = a.models.split(",") if a.models else g["model"]
    fsets = a.fsets.split(",") if a.fsets else g["fset"]
    cfgs = [{"model": m, "fset": f, "use_z": z, "peso": p, "balance": True}
            for f in fsets for m in models for z in g["use_z"] for p in g["peso"]]
    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[clf_villar] sweep {a.name}: {len(cfgs)} configuraciones, sims {len(S)}, reales val con features "
          f"{len(R)}/{len(v)}", flush=True)
    if a.jobs > 1:
        results = joblib.Parallel(n_jobs=a.jobs, verbose=10)(joblib.delayed(_one)(S, R, c, cls, a.folds, a.seed) for c in cfgs)
    else:
        results = []
        for i, c in enumerate(cfgs):
            results.append(_one(S, R, c, cls, a.folds, a.seed))
            if "error" not in results[-1]:
                _print_res(f"{i + 1}/{len(cfgs)} {c}", results[-1])
            else:
                print(f"[clf_villar] {i + 1}/{len(cfgs)} {c}: ERROR {results[-1]['error']}", flush=True)
    rows = []
    for r in results:
        if "error" in r:
            rows.append({**r["config"], "error": r["error"]})
            continue
        for row in resumen_rows(r, a.name, a.features_sims, cov):
            rows.append({**row, "segundos": r["segundos"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab.get("bal_acc").notna()] if "bal_acc" in tab else tab.iloc[:0]
    ok = ok.sort_values([a.rank_by, "cv_bal_acc"], ascending=False)
    tab.to_csv(out / "sweep.csv", index=False)
    append_resumen([r for r in rows if "error" not in r], a.out_root)
    (out / "sweep_full.json").write_text(json.dumps(_jsonable([r for r in results]), indent=1))
    if len(ok):
        best = ok.iloc[0]
        cfg = {"model": best.model, "fset": best.fset, "use_z": bool(best.use_z), "peso": best.peso,
               "balance": True}
        res = run_config(S, R, cfg, cls, a.folds, a.seed, keep_model=True)
        write_run(out / "mejor", res, R, cls, cov)
        perm_importance(res["_model"], R, res["cols"], cls, seed=a.seed).to_csv(
            out / "mejor" / "importancias_real_val.csv", index=False)
        (out / "mejor.json").write_text(json.dumps(_jsonable({"rank_by": a.rank_by, "prior": best.prior,
                                                              "config": cfg, "top10": ok.head(10).to_dict("records")}),
                                                   indent=1))
        print(ok.head(10)[["model", "fset", "use_z", "peso", "prior", "cv_bal_acc", "acc", "bal_acc",
                           "f1_macro"]].round(3).to_string(index=False), flush=True)
    return tab


def cmd_gap(a):
    """Clasificador sim contra real sin etiquetas (regla 5 del brief): AUC fuera de fold, importancia por
    permutacion (caida de AUC en el fold de prueba) y AUC univariado de cada feature."""
    cls = classes(a.cuatro_clases)
    S, R, v = prepare(a.features_sims, a.run_dir, a.features_real, a.real_dir, a.cuatro_clases, not a.obs_frame,
                      a.requiere)
    cols = [c for c in GAP_COLS if (a.no_z is False or not c.startswith("M_pk"))]
    w = S.w_z.to_numpy(float)
    if a.peso == "wz_S":
        w = w * selection_weight(S.m_sel, w, R.m_sel)[0]
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
    res = {"auc_sim_vs_real": auc, "n_sims": int(len(S)), "n_real": int(len(R)), "peso": a.peso,
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
    ap.add_argument("--features-real", type=Path, default=REAL_FEAT)
    ap.add_argument("--real-dir", type=Path, default=REAL_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--model", default="hgb", choices=tuple(MODELS))
    ap.add_argument("--fset", default="rg", choices=FSETS)
    ap.add_argument("--no-z", action="store_true", help="sin magnitud absoluta (los tiempos siguen en reposo)")
    ap.add_argument("--obs-frame", action="store_true", help="tiempos en el marco observado (sin 1/(1+z))")
    ap.add_argument("--peso", default="wz_S", choices=("wz", "wz_S", "wz_dr"))
    ap.add_argument("--sin-balance", action="store_true", help="sin balance de clases (prior = w_z * seleccion)")
    ap.add_argument("--requiere", default="any", choices=("any", "r"))
    ap.add_argument("--cuatro-clases", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--grid", default="rapido", choices=tuple(GRIDS))
    ap.add_argument("--models", default=None, help="sweep: lista separada por comas (pisa la grilla)")
    ap.add_argument("--fsets", default=None, help="sweep: lista separada por comas (pisa la grilla)")
    ap.add_argument("--rank-by", default="bal_acc", choices=("bal_acc", "acc", "f1_macro", "cv_bal_acc"))
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
