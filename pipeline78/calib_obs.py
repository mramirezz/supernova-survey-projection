# pipeline78/calib_obs.py
"""Calibracion del modelo de observacion por epoca (deteccion logistica por banda y escala del sorteo del ruido)
contra las alertas ALeRCE de las viejas val_viejo (Mauricio 2026-10-04, gap-sintesis.md G1, N1, N2, N3).

DATOS. Calibracion: viejas con split val_viejo (337) menos las excluidas (excluir de meta: primera_det_tardia y las
dudosas de data/excluir_manual_reales.csv) y menos las plantillas (oid -> nombre IAU por internal_names de
TNS_ZTF_df_new.csv, como holdout.tns_index, contra catalog.csv sin el prefijo SN; un oid sin fila en TNS se reporta
con el anio de su prefijo ZTFaa). Confirmacion (solo lectura, parametros ya congelados): mitad val del holdout
(origen holdout, split val, sin excluidas). meta se lee con csv: las filas final y final_viejo nunca llegan a pandas,
y los parquets de real_ztf se leen con filtro de oid.
Detecciones reales: las de real_ztf/<clase>.parquet (misma limpieza que el clasificador: real_to_parquet + lcclean,
sin filas repetidas ni restas negativas), emparejadas con su alerta de ALeRCE (misma banda, |dmjd| <= 1e-3 d,
|dmag| <= 2e-3, alerce.match). De la alerta salen magpsf, sigmapsf y diffmaglim (el limite 5 sigma de esa imagen).
Solo restas positivas (isdiffpos > 0). Alertas: data/ruido_alerce_viejas.csv (calib_ruido.fetch_alerce) y, para la
confirmacion, data/ruido_alerce_val.csv. Sims: detecciones (upperlimit F) de los pilotos, m = magnitud_proyectada,
sigma = magerr, m_lim = maglimit del log.

REGLAS COMUNES (las mismas funciones en reales y sims).
- Una deteccion por objeto, banda y dia entero de mjd: la de m_lim mas hondo (el log de las sims ya es asi,
  survey._best_per_day). Las reales tienen varias visitas por noche.
- Seleccion: >= MIN_NOCHES_R noches con deteccion r (las viejas se eligieron con features r validos,
  n_points_r >= 7 en real_val.parquet). Pico: deteccion r mas brillante (m_pk, t_pk).
- Peso de las sims: w_z por el cociente real/sim de la grilla (clase del clasificador, bin de m_pk de BIN mag). La
  clase entra porque la mezcla sim (Ia 10, II+IIb+IIn 20, Ibc 10 por campo) no es la de las viejas (114/115/107).
  Las reales pesan 1.
BLANCOS por banda (g, r): (1) dm = m_lim - m en las detecciones: fracciones dm < 0, < 0.5, < 1 y mediana. (2) dm de la
primera deteccion de la banda: mediana y fraccion < 0.75. (3) dm de la ultima: mediana. Las fracciones y la mediana
de (1) pesan cada deteccion con el peso de su objeto, (2) y (3) cada objeto.
(4) Tripletes: tres detecciones consecutivas de la banda (por noche) con t3 - t1 <= TRIP_DT. r = m2 - interpolacion
lineal de m1 y m3 en t2; s = sqrt(s2^2 + a^2 s1^2 + b^2 s3^2) con a = (t3 - t2)/(t3 - t1), b = (t2 - t1)/(t3 - t1) (el
error reportado propagado); z = r/s. Dispersion robusta 1.4826 MAD ponderada. k = real/sim con el piloto a k = 1.
PUNTAJE por banda: suma sobre los 7 blancos de (1)-(3) de |sim - real| / sqrt(e_real^2 + e_sim^2), e = desviacion
bootstrap sobre objetos (N_BOOT, semilla SEED_BOOT). Adimensional y comparable entre fracciones y magnitudes.
CHEQUEOS (no entran al puntaje): detecciones por objeto, fraccion sin >= 7 noches g, ultima r - pico por clase y en
las brillantes (m_pk < 18), m_lim en la primera deteccion.

Salidas: tablas en RUNS/calib_obs/ (k.json, tripletes_k.csv, blancos_*.csv, puntajes_*.csv, confirm_*, val_*), figura
FIG (y su pdf), copiada a ATLAS. Los pilotos van en RUNS/calib_obs/<cfg> (pipeline78.run, 200 campos, semilla 20261002).

USO (lo que produjo ztf_v78_t10, 2026-10-04; pilotos uno a uno, --workers 2):
    $PY -m pipeline78.calib_obs fetch                                   # data/ruido_alerce_viejas.csv
    $PY -m pipeline78.calib_obs k --runs ~/thesis_runs/calib_l/m0.0      # k a k = 1 -> runcfg.OBS_K
    $PY -m pipeline78.calib_obs pilots --cfgs ztf_v78_t9_obs-0.5_w0.2,...,ztf_v78_t9_obs0.5_w0.3,ztf_v78_t9_obs_eps0.97
    $PY -m pipeline78.calib_obs score --cfgs <los 11 de arriba> --tag grilla
    $PY -m pipeline78.calib_obs pilots --cfgs ztf_v78_t10
    $PY -m pipeline78.calib_obs confirm --cfg ztf_v78_t10              # contra val_viejo, base calib_l/m1.25 (= t9)
    $PY -m pipeline78.calib_obs val --cfg ztf_v78_t10                  # mitad val, solo lectura, al final
    $PY -m pipeline78.calib_obs fig --cfg ztf_v78_t10
"""
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import DATA, PHD, REPO, RUNS, STORE
from pipeline78.catalog import CLF_CLASS
from pipeline78 import alerce

OUT = RUNS / "calib_obs"
META = RUNS / "real_ztf/meta_real_ztf.csv"
REAL = RUNS / "real_ztf"
CACHE = DATA / "ruido_alerce_viejas.csv"
VAL_CACHE = DATA / "ruido_alerce_val.csv"
FIG = PHD / "paper2_ZTF/figures_templates/pipeline78_ajustes_mcmc/calib_obs.png"
ATLAS = Path.home() / "atlas_local/figures_templates/pipeline78_ajustes_mcmc"
FIELDS = STORE / "ztf_fields_1000.txt"
SEED, LIMIT, WORKERS = 20261002, 200, 2
SPLITS = ("val_viejo", "val")      # nunca final ni final_viejo
BANDS = ("g", "r")
FID = {"g": 1, "r": 2}
TOL_MJD, TOL_MAG = 1e-3, 2e-3      # el .dat trae mjd y magnitud con 3 decimales (real_to_parquet)
MIN_NOCHES_R = 7
BIN = 0.25                         # mag: bins de m_pk
TRIP_DT = 6.0                      # d
SIG_LO = 0.04                      # mag: borde de los tripletes de sigma baja
M_BRILLANTE = 18.0
N_BOOT, SEED_BOOT = 200, 20261004
BLANCOS = ["f_dm<0", "f_dm<0.5", "f_dm<1", "med_dm", "med_dm_1a", "f_dm_1a<0.75", "med_dm_ult"]
CLASES = ("Ia", "II", "Ibc")


# ---------------------------------------------------------------- reales
def read_meta(split, path=META):
    """Filas de meta con ese split. csv.DictReader: las demas filas se descartan antes de armar el DataFrame."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r} no permitido")
    with open(path, newline="") as f:
        rows = [x for x in csv.DictReader(f) if x["split"] == split]
    return pd.DataFrame(rows)


def calib_set(split="val_viejo"):
    """(meta usable, resumen de exclusiones). val_viejo: origen viejas; val: origen holdout."""
    from pipeline78.holdout import tns_index, template_iau
    m = read_meta(split)
    m = m[m.origen == ("viejas" if split == "val_viejo" else "holdout")].reset_index(drop=True)
    excl = m.excluir.str.strip().str.lower().isin(("true", "1"))
    idx, plant = tns_index(), template_iau()
    iau = m.oid.map(lambda o: idx[o][0] if o in idx else None)
    es_pl = iau.isin(plant)
    sin_tns = m.oid[iau.isna()].tolist()
    info = dict(split=split, n=len(m), excluidas=m.oid[excl].tolist(), motivos=m.motivo[excl].tolist(),
                plantillas=sorted(zip(m.oid[es_pl], iau[es_pl])), sin_tns=sin_tns,
                sin_tns_anios=sorted({"20" + o[3:5] for o in sin_tns}))
    out = m[~excl & ~es_pl].reset_index(drop=True)
    out["cls"] = out.sn_type.map(CLF_CLASS)
    info["n_usable"] = len(out)
    info["por_clase"] = out.cls.value_counts().sort_index().to_dict()
    return out, info


def read_alerts(path, oids):
    """Alertas del cache csv solo de esos oid (csv.reader: las demas filas no llegan a pandas)."""
    oids = set(oids)
    with open(path, newline="") as f:
        r = csv.reader(f)
        h = next(r)
        rows = [x for x in r if x[0] in oids]
    a = pd.DataFrame(rows, columns=h)
    for c in ("mjd", "magpsf", "sigmapsf", "diffmaglim"):
        a[c] = pd.to_numeric(a[c])
    for c in ("fid", "isdiffpos"):
        a[c] = pd.to_numeric(a[c]).astype("int64")
    return a.reset_index(drop=True)


def fetch(split="val_viejo", out=CACHE):
    """Alertas de ALeRCE de todas las SNe del split (sin filtrar exclusiones) -> out (calib_ruido.fetch_alerce)."""
    from pipeline78.calib_ruido import fetch_alerce
    m = read_meta(split)
    m = m[m.origen == ("viejas" if split == "val_viejo" else "holdout")]
    df = fetch_alerce(sorted(m.oid), out, workers=4)
    print(f"{out.name}: {df.oid.nunique()} oid de {m.oid.nunique()}, {len(df)} alertas")
    return df


ALERTAS = DATA / "alertas_viejas.csv"
MJD_STAMP = 59580.0                 # 2022-01-01: desde ahi has_stamp == (parent_candid nulo); antes no sirve (ingesta masiva)
PALERT_BINS = [-9, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 9]


def fetch_alertas(split="val_viejo", out=ALERTAS):
    """Detecciones de ALeRCE con has_stamp (alerta propia) y parent_candid de las SNe del split -> out."""
    from concurrent.futures import ThreadPoolExecutor
    from pipeline78.calib_ruido import _get
    m = read_meta(split)
    oids = sorted(m.oid[m.origen == ("viejas" if split == "val_viejo" else "holdout")])
    cols = ["candid", "mjd", "fid", "magpsf", "diffmaglim", "isdiffpos", "has_stamp", "parent_candid"]
    with ThreadPoolExecutor(4) as ex:
        rows = [[oid] + [x.get(c) for c in cols] for oid, js in ex.map(_get, oids) for x in js]
    df = pd.DataFrame(rows, columns=["oid"] + cols).sort_values(["oid", "mjd", "fid"])
    df.to_csv(out, index=False)
    print(f"{out.name}: {df.oid.nunique()} oid, {len(df)} detecciones")


def palert(split="val_viejo", path=ALERTAS):
    """P(alerta | deteccion positiva) contra dm = diffmaglim - magpsf por banda, en detecciones desde 2022 de las SNe
    usables del split. Ajuste de maxima verosimilitud de P = eps expit((dm - m50)/w) por banda. -> (tabla, params)."""
    from scipy.optimize import minimize
    from scipy.special import expit
    meta = calib_set(split)[0]
    d = pd.read_csv(path, dtype={"candid": str, "parent_candid": str})
    d = d[d.oid.isin(set(meta.oid)) & (d.isdiffpos > 0) & (d.mjd >= MJD_STAMP) & d.fid.isin(FID.values())].copy()
    d["band"] = d.fid.map({v: k for k, v in FID.items()})
    d["dm"] = d.diffmaglim - d.magpsf
    d["alerta"] = d.has_stamp.astype(str).str.lower().eq("true")
    d["bin"] = pd.cut(d.dm, PALERT_BINS, right=False)
    tab = d.groupby(["band", "bin"], observed=True).alerta.agg(["mean", "size"]).reset_index()
    tab["lo"], tab["hi"] = [i.left for i in tab["bin"]], [i.right for i in tab["bin"]]
    par = {}
    for b, x in d.groupby("band"):
        dm, y = x.dm.to_numpy(float), x.alerta.to_numpy(float)

        def nll(q):
            p = np.clip(expit(q[2]) * expit((dm - q[0]) / np.exp(q[1])), 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        q = minimize(nll, [0.0, np.log(0.2), 2.5], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6)).x
        par[b] = dict(m50=round(float(q[0]), 3), w=round(float(np.exp(q[1])), 3), eps=round(float(expit(q[2])), 3),
                      n=int(len(x)), n_obj=int(x.oid.nunique()))
    return tab.drop(columns="bin"), par


def ruido(split="val_viejo", n_boot=300):
    """A, B, C del ruido de tres terminos (calib_ruido.calibrate) con las detecciones del split, sin tocar el holdout:
    m = magpsf, sig = sigmapsf, dm = diffmaglim - magpsf de la alerta emparejada (real_epochs). -> {g, r: (p, err, n)}."""
    from pipeline78.calib_ruido import calibrate
    ep = real_epochs(calib_set(split)[0], CACHE)[0]
    d = pd.DataFrame({"oid": ep.oid, "filter": ep.band, "sig": ep.sig, "dm": ep.ml - ep.m})
    d = d[np.isfinite(d.dm) & (d.dm >= 0) & (d.dm < 4)].reset_index(drop=True)
    res = calibrate(d, n_boot)
    return {b: dict(p=[round(float(x), 4) for x in res[b]["p"]], err=[round(float(x), 4) for x in res[b]["err"]],
                    n_sn=int(res[b]["n_sn"]), n_det=int(res[b]["n_det"])) for b in ("g", "r", "gr")}


def real_epochs(meta, cache):
    """Detecciones g/r de los parquets limpios de real_ztf, con magpsf, sigmapsf y diffmaglim de su alerta positiva.
    -> (epocas, resumen del emparejamiento)."""
    oids = sorted(meta.oid)
    fr = [pd.read_parquet(REAL / f"{c}.parquet", filters=[("oid", "in", oids)]) for c in sorted(set(meta.sn_type))]
    d = pd.concat(fr, ignore_index=True)
    if not set(d.oid) <= set(oids):
        raise RuntimeError("el filtro de oid dejo pasar otras SNe")
    d = d[(d.upperlimit == "F") & d["filter"].isin(BANDS)].reset_index(drop=True)
    d["fid"] = d["filter"].map(FID)
    a = read_alerts(cache, oids)
    j = alerce.match(d, a, TOL_MJD, TOL_MAG, mag_col="magnitud_proyectada")
    hit = j >= 0
    pos = np.zeros(len(d), bool)
    pos[hit] = a.isdiffpos.to_numpy()[j[hit]] > 0
    x = a.iloc[j[pos]].reset_index(drop=True)
    cls = dict(zip(meta.oid, meta.cls))
    ep = pd.DataFrame({"oid": x.oid, "cls": x.oid.map(cls), "band": d["filter"].to_numpy()[pos], "mjd": x.mjd,
                       "m": x.magpsf, "sig": x.sigmapsf, "ml": x.diffmaglim, "w_z": 1.0})
    info = dict(n_det=len(d), sin_alerta=int((~hit).sum()), negativas=int((hit & ~pos).sum()), usadas=int(pos.sum()),
                sin_alerta_por_banda=d.loc[~hit, "filter"].value_counts().to_dict(),
                oids_sin_alerta=int(d.loc[~hit, "oid"].nunique()))
    return ep, info


# ---------------------------------------------------------------- sims
def sim_epochs(run_dir):
    """Detecciones g/r de un piloto: oid = sim_id, cls = clase del clasificador, w_z del sorteo de z."""
    cols = ["sim_id", "sn_type", "w_z", "filter", "mjd", "maglimit", "magnitud_proyectada", "magerr", "upperlimit"]
    fr = []
    for p in sorted(Path(run_dir).expanduser().glob("*__*.parquet")):
        d = pd.read_parquet(p, columns=cols)
        fr.append(d[(d.upperlimit == "F") & d["filter"].isin(BANDS)])
    d = pd.concat(fr, ignore_index=True)
    return pd.DataFrame({"oid": d.sim_id.to_numpy(), "cls": d.sn_type.map(CLF_CLASS).to_numpy(),
                         "band": d["filter"].to_numpy(), "mjd": d.mjd.to_numpy(float),
                         "m": d.magnitud_proyectada.to_numpy(float), "sig": d.magerr.to_numpy(float),
                         "ml": d.maglimit.to_numpy(float), "w_z": d.w_z.to_numpy(float)})


# ---------------------------------------------------------------- reglas comunes
def por_noche(ep):
    """Una deteccion por (objeto, banda, dia entero de mjd): la de m_lim mas hondo (survey._best_per_day)."""
    e = ep.assign(day=np.floor(ep.mjd).astype("int64"))
    e = e.sort_values(["oid", "band", "day", "ml", "mjd"], ascending=[True, True, True, False, True], kind="stable")
    e = e.drop_duplicates(["oid", "band", "day"]).drop(columns="day")
    return e.sort_values(["oid", "band", "mjd"], kind="stable").reset_index(drop=True)


def objetos(ep):
    """Una fila por objeto con deteccion r: cls, w_z, m_pk, t_pk y por banda n, dm de la 1a y la ultima deteccion,
    m_lim de la 1a, t de la ultima. ep ya pasado por por_noche (ordenado por oid, banda, mjd)."""
    e = ep.assign(dm=ep.ml - ep.m)
    r = e[e.band == "r"]
    pk = r.loc[r.groupby("oid").m.idxmin(), ["oid", "cls", "w_z", "m", "mjd"]]
    o = pk.rename(columns={"m": "m_pk", "mjd": "t_pk"}).set_index("oid")
    for b in BANDS:
        x = e[e.band == b].groupby("oid")
        f, l = x.first(), x.last()
        o[f"n_{b}"] = x.size().reindex(o.index).fillna(0).astype(int)
        o[f"dm1_{b}"], o[f"ml1_{b}"] = f.dm.reindex(o.index), f.ml.reindex(o.index)
        o[f"dmL_{b}"], o[f"tL_{b}"] = l.dm.reindex(o.index), l.mjd.reindex(o.index)
    o["dur_r"] = o.tL_r - o.t_pk
    return o


def seleccion(o):
    return o[o.n_r >= MIN_NOCHES_R]


def _bin(m):
    return np.floor(np.asarray(m, float) / BIN).astype(int)


def pesos(o_sim, o_real, por_clase=True):
    """w_z * n_real(clase, bin de m_pk) / suma de w_z sim de esa celda; 0 en celdas sin reales. -> (pesos, cobertura).
    por_clase False: solo el bin de m_pk (sensibilidad)."""
    cs, cr = (o_sim.cls, o_real.cls) if por_clase else (np.zeros(len(o_sim), int), np.zeros(len(o_real), int))
    ks = pd.MultiIndex.from_arrays([cs, _bin(o_sim.m_pk)])
    kr = pd.MultiIndex.from_arrays([cr, _bin(o_real.m_pk)])
    nr = pd.Series(1.0, index=kr).groupby(level=[0, 1]).sum()
    ws = pd.Series(o_sim.w_z.to_numpy(), index=ks).groupby(level=[0, 1]).sum()
    f = (nr / ws).reindex(ks).fillna(0.0).to_numpy()
    w = pd.Series(o_sim.w_z.to_numpy() * f, index=o_sim.index)
    cub = float(nr[nr.index.isin(ws.index)].sum() / nr.sum())     # fraccion de reales con sims en su celda
    return w, cub


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w > 0).any() else 0.0


# ---------------------------------------------------------------- estadisticas
def _wq(x, w, q=0.5):
    """Cuantil ponderado (x ordenado o no)."""
    i = np.argsort(x, kind="stable")
    c = np.cumsum(w[i])
    return float(x[i][min(np.searchsorted(c, q * c[-1]), len(x) - 1)]) if len(x) and c[-1] > 0 else np.nan


class Muestra:
    """Arreglos para los blancos de una banda: detecciones (dm, indice de objeto) y objetos (dm 1a, dm ultima)."""

    def __init__(self, ep, o, w, b):
        idx = pd.Series(np.arange(len(o)), index=o.index)
        e = ep[(ep.band == b) & ep.oid.isin(o.index)]
        self.w = np.asarray(w, float)
        self.dm = (e.ml - e.m).to_numpy(float)
        self.io = idx[e.oid].to_numpy()
        self.d1, self.dL = o[f"dm1_{b}"].to_numpy(float), o[f"dmL_{b}"].to_numpy(float)
        self.ok1 = np.isfinite(self.d1)

    def stats(self, c=None):
        """Los 7 blancos con multiplicidad c por objeto (bootstrap) o 1."""
        wo = self.w if c is None else self.w * c
        we = wo[self.io]
        s, dm = we.sum(), self.dm
        w1 = wo[self.ok1]
        return {"f_dm<0": we[dm < 0].sum() / s, "f_dm<0.5": we[dm < 0.5].sum() / s, "f_dm<1": we[dm < 1].sum() / s,
                "med_dm": _wq(dm, we), "med_dm_1a": _wq(self.d1[self.ok1], w1),
                "f_dm_1a<0.75": w1[self.d1[self.ok1] < 0.75].sum() / w1.sum(),
                "med_dm_ult": _wq(self.dL[self.ok1], w1)}

    def boot(self, rng, n=N_BOOT):
        k = len(self.w)
        bs = [self.stats(np.bincount(rng.integers(0, k, k), minlength=k)) for _ in range(n)]
        return pd.DataFrame(bs).std(ddof=1)


def tripletes(ep, dt=TRIP_DT):
    """Ternas consecutivas por objeto y banda con t3 - t1 <= dt: oid, band, sig (del punto central), z."""
    e = ep.sort_values(["oid", "band", "mjd"], kind="stable")
    o, b = e.oid.to_numpy(), e.band.to_numpy()
    t, m, s = e.mjd.to_numpy(float), e.m.to_numpy(float), e.sig.to_numpy(float)
    same = (o[:-2] == o[1:-1]) & (o[1:-1] == o[2:]) & (b[:-2] == b[1:-1]) & (b[1:-1] == b[2:])
    t1, t2, t3 = t[:-2], t[1:-1], t[2:]
    D = t3 - t1
    ok = same & (D <= dt) & (t2 > t1) & (t3 > t2)
    a, bb = (t3 - t2)[ok] / D[ok], (t2 - t1)[ok] / D[ok]
    r = m[1:-1][ok] - (a * m[:-2][ok] + bb * m[2:][ok])
    sr = np.sqrt(s[1:-1][ok] ** 2 + (a * s[:-2][ok]) ** 2 + (bb * s[2:][ok]) ** 2)
    return pd.DataFrame({"oid": o[1:-1][ok], "band": b[1:-1][ok], "sig": s[1:-1][ok], "z": r / sr})


def rstd(z, w):
    """1.4826 MAD ponderada."""
    z, w = np.asarray(z, float), np.asarray(w, float)
    if not len(z) or w.sum() <= 0:
        return np.nan
    return 1.4826 * _wq(np.abs(z - _wq(z, w)), w)


def trip_table(ep, o, w, rng=None, n_boot=N_BOOT):
    """Dispersion robusta de z por banda y rango de sigma (todas, sigma < SIG_LO, >= SIG_LO), con error bootstrap."""
    tr = tripletes(ep[ep.oid.isin(o.index)])
    wo = pd.Series(np.asarray(w, float), index=o.index)
    rows = []
    for bnd in BANDS:
        for lab, sel in (("todas", None), (f"sig<{SIG_LO}", True), (f"sig>={SIG_LO}", False)):
            x = tr[tr.band == bnd]
            if sel is not None:
                x = x[(x.sig < SIG_LO) == sel]
            z, ww = x.z.to_numpy(), wo[x.oid].to_numpy()
            err = np.nan
            if rng is not None and len(z) > 10:
                io = pd.Series(np.arange(len(o)), index=o.index)[x.oid].to_numpy()
                bs = []
                for _ in range(n_boot):
                    c = np.bincount(rng.integers(0, len(o), len(o)), minlength=len(o))
                    bs.append(rstd(z, ww * c[io]))
                err = float(np.nanstd(bs, ddof=1))
            rows.append(dict(band=bnd, rango=lab, n=len(z), n_obj=x.oid.nunique(), rstd=rstd(z, ww), err=err,
                             f_gt3=float(ww[np.abs(z) > 3 * rstd(z, ww)].sum() / ww.sum()) if len(z) else np.nan))
    return pd.DataFrame(rows)


def chequeos(o, w):
    """Chequeos por objeto (no entran al puntaje)."""
    w = pd.Series(np.asarray(w, float), index=o.index)
    out = {}
    for b in BANDS:
        n = o[f"n_{b}"].to_numpy(float)
        out[f"n_noches_{b}_media"] = float((n * w).sum() / w.sum())
        out[f"n_noches_{b}_med"] = _wq(n, w.to_numpy())
        ok = np.isfinite(o[f"ml1_{b}"].to_numpy())
        out[f"ml_1a_{b}_med"] = _wq(o[f"ml1_{b}"].to_numpy()[ok], w.to_numpy()[ok])
    out["f_sin_7g"] = float(w[o.n_g < 7].sum() / w.sum())
    for c in CLASES:
        for lab, s in (("", o.cls == c), ("_brill", (o.cls == c) & (o.m_pk < M_BRILLANTE))):
            out[f"dur_r_{c}{lab}"] = _wq(o.dur_r[s].to_numpy(float), w[s].to_numpy())
            out[f"n_{c}{lab}"] = int(s.sum())
    return out


# ---------------------------------------------------------------- piloto contra reales
_REAL = {}


def real(split="val_viejo"):
    """(epocas por noche, objetos seleccionados, info) de las reales del split, cacheado en memoria."""
    if split not in _REAL:
        meta, info = calib_set(split)
        todas, mi = real_epochs(meta, CACHE if split == "val_viejo" else VAL_CACHE)
        ep = por_noche(todas)
        o = objetos(ep)
        info.update(emparejamiento=mi, n_det_todas=len(todas), n_det_noche=len(ep), n_obj_con_r=len(o))
        o = seleccion(o)
        info["n_sel"] = len(o)
        # sensibilidad: los blancos de (1) con todas las visitas de la noche (las sims tienen una por dia)
        info["dm_todas_visitas"] = {b: Muestra(todas, o, np.ones(len(o)), b).stats() for b in BANDS}
        _REAL[split] = (ep[ep.oid.isin(o.index)].reset_index(drop=True), o, info)
    return _REAL[split]


def evaluar(run_dir, split="val_viejo", boot=True, trip=False, por_clase=True):
    """Blancos, errores y chequeos de un piloto contra las reales del split. -> dict."""
    epr, orl, _ = real(split)
    eps_ = por_noche(sim_epochs(run_dir))
    os_ = seleccion(objetos(eps_))
    w, cub = pesos(os_, orl, por_clase)
    rng = np.random.default_rng(SEED_BOOT)
    res = dict(run=str(run_dir), n_sim_sel=len(os_), ess=ess(w), cobertura=cub, bandas={})
    for b in BANDS:
        ms, mr = Muestra(eps_, os_, w, b), Muestra(epr, orl, np.ones(len(orl)), b)
        res["bandas"][b] = dict(sim=ms.stats(), real=mr.stats(), e_sim=ms.boot(rng) if boot else None,
                                e_real=mr.boot(rng) if boot else None)
    res["cheq_sim"], res["cheq_real"] = chequeos(os_, w), chequeos(orl, np.ones(len(orl)))
    if trip:
        res["trip_sim"] = trip_table(eps_, os_, w, rng)
        res["trip_real"] = trip_table(epr, orl, np.ones(len(orl)), rng)
    res["_sim"] = (eps_, os_, w)
    return res


def puntaje(rb):
    """Suma de |sim - real| / sqrt(e_real^2 + e_sim^2) sobre BLANCOS."""
    s, r = pd.Series(rb["sim"]), pd.Series(rb["real"])
    e = np.sqrt(rb["e_real"] ** 2 + rb["e_sim"] ** 2)
    return float(((s - r).abs() / e)[BLANCOS].sum())


def tabla_blancos(res, label):
    rows = []
    for b in BANDS:
        rb = res["bandas"][b]
        for k in BLANCOS:
            rows.append(dict(piloto=label, band=b, blanco=k, real=rb["real"][k], sim=rb["sim"][k],
                             dif=rb["sim"][k] - rb["real"][k],
                             e_real=None if rb["e_real"] is None else rb["e_real"][k],
                             e_sim=None if rb["e_sim"] is None else rb["e_sim"][k]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pilotos
def pilot_dir(cfg):
    return OUT / cfg


def run_pilot(cfg):
    """Un piloto como los de calib_l: 200 campos, semilla 20261002, 2 workers, --allow-dirty. Uno a la vez."""
    out = pilot_dir(cfg)
    if (out / "_sims_all.parquet").exists():
        print(f"{cfg}: ya existe, no se corre")
        return out
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pipeline78.run", "--run", cfg, "--out", str(out), "--fields-file", str(FIELDS),
           "--limit", str(LIMIT), "--seed", str(SEED), "--workers", str(WORKERS), "--allow-dirty"]
    print(" ".join(cmd), flush=True)
    with open(OUT / f"{cfg}.log", "w") as log:
        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out


# ---------------------------------------------------------------- comandos
def _save(df, name):
    df.to_csv(OUT / name, index=False)
    print(f"-> {OUT / name}")


def cmd_k(runs):
    """k por banda = dispersion de tripletes real / sim, con el primer run (a k = 1). Los demas, solo de referencia."""
    rows, k = [], {}
    for i, run in enumerate(runs):
        r = evaluar(run, boot=False, trip=True)
        t = r["trip_sim"].merge(r["trip_real"], on=["band", "rango"], suffixes=("_sim", "_real"))
        t["cociente"] = t.rstd_real / t.rstd_sim
        t["err_cociente"] = t.cociente * np.hypot(t.err_real / t.rstd_real, t.err_sim / t.rstd_sim)
        rows.append(t.assign(run=str(run)))
        if i == 0:
            k = {b: round(float(t.cociente[(t.band == b) & (t.rango == "todas")].iloc[0]), 2) for b in BANDS}
    t = pd.concat(rows, ignore_index=True)
    print(t.round(3).to_string(index=False))
    _save(t, "tripletes_k.csv")
    print(f"k (con {runs[0]}): {k}")
    (OUT / "k.json").write_text(json.dumps(dict(run=str(runs[0]), k=k, trip_dt=TRIP_DT, sig_lo=SIG_LO), indent=1))
    return k


def _run(c):
    """Nombre de config (piloto en OUT) o carpeta de un run -> (etiqueta, carpeta)."""
    p = Path(c).expanduser()
    return (p.name, p) if p.is_dir() else (c, pilot_dir(c))


def params(run_dir, b):
    """det_m0, det_w, det_eps y noise_draw_scale de la banda b segun el manifiesto del run."""
    from pipeline78.project import _banda
    c = json.loads((Path(run_dir) / "run_manifest.json").read_text())["cfg"]
    return dict(m0=_banda(c["det_m0"], b), w=_banda(c["det_w"], b), eps=_banda(c.get("det_eps", 1.0), b),
                k=_banda(c.get("noise_draw_scale", 1.0), b), k_lo=c.get("noise_draw_scale_lowsig"))


def cmd_score(cfgs, tag):
    """Blancos y puntaje por banda de cada piloto; el mejor por banda."""
    tabs, rows = [], []
    for c in cfgs:
        lab, d = _run(c)
        r = evaluar(d)
        tabs.append(tabla_blancos(r, lab))
        for b in BANDS:
            rows.append(dict(piloto=lab, band=b, **params(d, b), puntaje=puntaje(r["bandas"][b]), n_sim_sel=r["n_sim_sel"],
                             ess=r["ess"], cobertura=r["cobertura"]))
        print(f"{lab}: g {rows[-2]['puntaje']:.2f}  r {rows[-1]['puntaje']:.2f}  (n_sel {r['n_sim_sel']}, ESS {r['ess']:.0f})",
              flush=True)
    t, p = pd.concat(tabs, ignore_index=True), pd.DataFrame(rows).drop(columns="k_lo")
    _save(p, f"puntajes_{tag}.csv")
    print(p.pivot_table(index=["m0", "w", "eps"], columns="band", values="puntaje").round(2).to_string())
    best = p.loc[p.groupby("band").puntaje.idxmin()]
    print("mejor por banda:\n", best.round(3).to_string(index=False))
    # desglose: aporte de cada blanco y el mejor sin (3) y solo con (1) (sensibilidad del puntaje)
    t["aporte"] = (t.sim - t.real).abs() / np.sqrt(t.e_real ** 2 + t.e_sim ** 2)
    print(t.pivot_table(index=["band", "blanco"], columns="piloto", values="aporte", sort=False).round(1).to_string())
    for lab, fuera in (("sin (3)", ["med_dm_ult"]), ("solo (1)", ["med_dm_1a", "f_dm_1a<0.75", "med_dm_ult"])):
        x = t[~t.blanco.isin(fuera)].groupby(["band", "piloto"]).aporte.sum().unstack(0)
        print(f"mejor {lab}: {x.idxmin().to_dict()}")
    _save(t, f"blancos_{tag}.csv")
    w = t.pivot_table(index=["band", "blanco"], columns="piloto", values="sim", sort=False)
    w.insert(0, "real", t.drop_duplicates(["band", "blanco"]).set_index(["band", "blanco"]).real)
    print(w.round(3).to_string())
    return t, p


def dur_table(o, w, lab):
    """Ultima deteccion r - pico (d, mediana ponderada) por clase y bin de m_pk."""
    w = pd.Series(np.asarray(w, float), index=o.index)
    bins = [0, 17.5, 18.0, 18.5, 19.0, 99]
    k = pd.cut(o.m_pk, bins, right=False)
    rows = []
    for c in CLASES:
        for iv in k.cat.categories:
            s = (o.cls == c) & (k == iv)
            rows.append(dict(muestra=lab, cls=c, m_pk=f"[{iv.left:g}, {iv.right:g})", n=int(s.sum()),
                             dur_r_med=_wq(o.dur_r[s].to_numpy(float), w[s].to_numpy()) if s.any() else np.nan))
    return pd.DataFrame(rows)


def _comparar(cfg, base, split, pref):
    """Blancos, chequeos, tripletes y duraciones del piloto cfg y del run base contra las reales del split."""
    (lc, dc), (lb, db) = _run(cfg), _run(base)
    rc, rb = evaluar(dc, split, trip=True), evaluar(db, split, trip=True)
    t = tabla_blancos(rc, lc).merge(tabla_blancos(rb, lb)[["band", "blanco", "sim", "e_sim"]], on=["band", "blanco"],
                                     suffixes=("", "_base"))
    print(t.round(3).to_string(index=False))
    sc = {b: (puntaje(rc["bandas"][b]), puntaje(rb["bandas"][b])) for b in BANDS}
    print("puntaje (piloto, base):", {b: tuple(round(x, 2) for x in v) for b, v in sc.items()})
    ch = pd.DataFrame({"real": rc["cheq_real"], lc: rc["cheq_sim"], lb: rb["cheq_sim"]})
    print(ch.round(3).to_string())
    tr = pd.concat([rc["trip_real"].assign(muestra="real"), rc["trip_sim"].assign(muestra=lc),
                    rb["trip_sim"].assign(muestra=lb)], ignore_index=True)
    print(tr.round(3).to_string(index=False))
    epr, orl, info = real(split)
    du = pd.concat([dur_table(orl, np.ones(len(orl)), "real"), dur_table(rc["_sim"][1], rc["_sim"][2], lc),
                    dur_table(rb["_sim"][1], rb["_sim"][2], lb)], ignore_index=True)
    print(du.pivot_table(index=["cls", "m_pk"], columns="muestra", values=["dur_r_med", "n"], sort=False).round(1).to_string())
    _save(t, f"{pref}_blancos.csv")
    _save(ch.rename_axis("chequeo").reset_index(), f"{pref}_chequeos.csv")
    _save(tr, f"{pref}_tripletes.csv")
    _save(du, f"{pref}_duracion.csv")
    meta = dict(split=split, piloto=str(dc), base=str(db), puntaje=sc, n_sim_sel=rc["n_sim_sel"], ess=rc["ess"],
                cobertura=rc["cobertura"], n_sim_sel_base=rb["n_sim_sel"], ess_base=rb["ess"],
                real={k: v for k, v in info.items()})
    (OUT / f"{pref}_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return rc, rb, t


def cmd_confirm(cfg, base):
    rc, rb, t = _comparar(cfg, base, "val_viejo", "confirm")
    # sensibilidades: pesos solo por m_pk (sin clase) y reales con todas las visitas de la noche
    rs = evaluar(_run(cfg)[1], por_clase=False)
    s = tabla_blancos(rs, "sin_clase")[["band", "blanco", "sim"]].rename(columns={"sim": "sim_pesos_sin_clase"})
    v = pd.DataFrame([dict(band=b, blanco=k, real_todas_visitas=x) for b, d in real()[2]["dm_todas_visitas"].items()
                      for k, x in d.items()])
    sens = t[["band", "blanco", "real", "sim"]].merge(s, on=["band", "blanco"]).merge(v, on=["band", "blanco"], how="left")
    print(sens.round(3).to_string(index=False))
    print("puntaje con pesos sin clase:", {b: round(puntaje(rs["bandas"][b]), 2) for b in BANDS})
    _save(sens, "confirm_sensibilidad.csv")


def cmd_val(cfg, base):
    """Confirmacion final sobre la mitad val del holdout (solo lectura, parametros ya congelados)."""
    _comparar(cfg, base, "val", "val")


def _arrays(ep, o, w, b):
    """dm en las detecciones (con su peso), dm de la 1a y de la ultima (por objeto) y z de los tripletes."""
    m = Muestra(ep, o, w, b)
    tr = tripletes(ep[ep.oid.isin(o.index)])
    tr = tr[tr.band == b]
    wo = pd.Series(np.asarray(w, float), index=o.index)
    return dict(dm=(m.dm, m.w[m.io]), d1=(m.d1[m.ok1], m.w[m.ok1]), dL=(m.dL[m.ok1], m.w[m.ok1]),
                z=(tr.z.to_numpy(), wo[tr.oid].to_numpy()))


def _cdf(ax, x, w, **kw):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    ax.step(x[i], c / c[-1], where="post", **kw)


def cmd_fig(cfg, base, tag, out=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight"})
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    epr, orl, _ = real("val_viejo")
    rc, rb = evaluar(_run(cfg)[1], boot=False), evaluar(_run(base)[1], boot=False)
    p = pd.read_csv(OUT / f"puntajes_{tag}.csv")
    col = {"g": "tab:green", "r": "tab:red"}
    fig, axs = plt.subplots(2, 5, figsize=(7.09, 3.3))
    for i, b in enumerate(BANDS):
        A = {"real": _arrays(epr, orl, np.ones(len(orl)), b), "new": _arrays(*rc["_sim"], b), "old": _arrays(*rb["_sim"], b)}
        sty = {"real": dict(color="0.75", lw=2.6), "new": dict(color=col[b], lw=1.2), "old": dict(color="0.35", lw=0.9, ls="--")}
        ax = axs[i, 0]
        bins = np.arange(-1.5, 4.01, 0.25)
        ax.hist(A["real"]["dm"][0], bins, weights=A["real"]["dm"][1], density=True, color="0.75")
        for k in ("new", "old"):
            ax.hist(A[k]["dm"][0], bins, weights=A[k]["dm"][1], density=True, histtype="step", **sty[k])
        ax.set_xlim(-1.5, 4)
        for j, key in ((1, "d1"), (2, "dL")):
            for k in ("real", "new", "old"):
                _cdf(axs[i, j], *A[k][key], **sty[k])
            axs[i, j].set_xlim(-1.0, 3.5)
            axs[i, j].set_ylim(0, 1)
        ax = axs[i, 3]
        zb = np.arange(-4, 4.01, 0.4)
        ax.hist(A["real"]["z"][0], zb, weights=A["real"]["z"][1], density=True, color="0.75")
        for k in ("new", "old"):
            ax.hist(A[k]["z"][0], zb, weights=A[k]["z"][1], density=True, histtype="step", **sty[k])
        ax.set_xlim(-4, 4)
        ax = axs[i, 4]
        q = p[(p.band == b) & (p.eps == 1.0)]
        for w_, mk in ((0.2, "o"), (0.3, "s")):
            x = q[q.w == w_].sort_values("m0")
            ax.plot(x.m0, x.puntaje, "-" + mk, color=col[b], ms=3, lw=0.8, mfc="w" if w_ == 0.3 else col[b])
        pc = params(_run(cfg)[1], b)
        ax.plot([pc["m0"]], [puntaje(evaluar(_run(cfg)[1])["bandas"][b])], "*", color="k", ms=7, zorder=5)
        ax.set_ylim(0, 1.1 * q.puntaje.max())
        axs[i, 0].set_ylabel(f"${b}$ band")
        for j, lab in enumerate((r"$m_{\rm lim} - m$, all", r"$m_{\rm lim} - m$, first", r"$m_{\rm lim} - m$, last",
                                 r"triplet residual / $\sigma$", r"$m_0$ [mag]")):
            axs[i, j].set_xlabel(lab + (" [mag]" if j in (0, 1, 2) else ""))
    axs[0, 1].set_title("cumulative fraction")
    axs[0, 2].set_title("cumulative fraction")
    axs[0, 4].set_title("score")
    axs[0, 0].set_title("density")
    axs[0, 3].set_title("density")
    hs = [Line2D([], [], color="0.75", lw=5),
          (Line2D([], [], color=col["g"], lw=1.2), Line2D([], [], color=col["r"], lw=1.2)),
          Line2D([], [], color="0.35", lw=0.9, ls="--"),
          Line2D([], [], color="0.3", marker="o", ms=3, lw=0.8), Line2D([], [], color="0.3", marker="s", ms=3, mfc="w", lw=0.8),
          Line2D([], [], color="k", marker="*", ms=7, ls="")]
    fig.legend(hs, ["ZTF alerts, calibration set", "recalibrated (per band)", r"previous ($m_0 = 1.25$, $k = 1$)",
                    r"score, $w = 0.2$ mag", r"score, $w = 0.3$ mag", "adopted"],
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.6, h_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    ATLAS.mkdir(parents=True, exist_ok=True)
    for f in (out, out.with_suffix(".pdf")):
        shutil.copy2(f, ATLAS / f.name)
    url = f"http://127.0.0.1:8899/figures_templates/pipeline78_ajustes_mcmc/{out.name}"
    code = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url], capture_output=True,
                          text=True).stdout
    print(f"figura: {out} y {ATLAS / out.name}; {url} -> {code}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fetch")
    sp.add_parser("fetch_alertas")
    sp.add_parser("palert")
    sp.add_parser("ruido")
    a = sp.add_parser("k")
    a.add_argument("--runs", default=str(RUNS / "calib_l/m0.0"), help="el primero (a k = 1) define k")
    a = sp.add_parser("pilots")
    a.add_argument("--cfgs", required=True)
    a = sp.add_parser("score")
    a.add_argument("--cfgs", required=True)
    a.add_argument("--tag", default="grilla")
    a = sp.add_parser("confirm")
    a.add_argument("--cfg", required=True)
    a.add_argument("--base", default=str(RUNS / "calib_l/m1.25"))
    a = sp.add_parser("val")
    a.add_argument("--cfg", required=True)
    a.add_argument("--base", default=str(RUNS / "calib_l/m1.25"))
    a = sp.add_parser("fig")
    a.add_argument("--cfg", required=True)
    a.add_argument("--base", default=str(RUNS / "calib_l/m1.25"))
    a.add_argument("--tag", default="grilla")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    if a.cmd == "fetch":
        fetch()
    elif a.cmd == "fetch_alertas":
        fetch_alertas()
    elif a.cmd == "ruido":
        r = ruido()
        (OUT / "ruido_abc_viejas.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    elif a.cmd == "palert":
        tab, par = palert()
        _save(tab, "palert.csv")
        (OUT / "palert_params.json").write_text(json.dumps(par, indent=1))
        print(tab.round(3).to_string(index=False))
        print(json.dumps(par, indent=1))
    elif a.cmd == "k":
        cmd_k(a.runs.split(","))
    elif a.cmd == "pilots":
        for c in a.cfgs.split(","):
            run_pilot(c)
    elif a.cmd == "score":
        cmd_score(a.cfgs.split(","), a.tag)
    elif a.cmd == "confirm":
        cmd_confirm(a.cfg, a.base)
    elif a.cmd == "val":
        cmd_val(a.cfg, a.base)
    else:
        cmd_fig(a.cfg, a.base, a.tag)


if __name__ == "__main__":
    main()
