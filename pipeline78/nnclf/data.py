"""Datos del clasificador NN sobre curvas crudas (nn-brief 2026-10-03). Sin torch: lo usa tambien el baseline.

REGLAS
1. Entrenamiento: sims de la T9 final (RUNS/ztf_v78_t9_final). Bandas g y r solamente (ZTF publico no tiene i).
   Deteccion = upperlimit 'F'. Entra la sim con >= MIN_DET detecciones en g+r (no las 7 de Villar).
2. Clases: Ia, II (= II + IIb), Ibc. IIn fuera, salvo four_classes=True (Ia, II, Ibc, IIn). El mapeo sale de
   sn_type y no de clf_class, porque clf_class de la T9 junta IIn con II. Cinco clases (Mauricio 2026-10-06): Ia,
   II (= IIP + IIL + II sin subtipo), IIb, Ibc, IIn, con la IIb como clase propia. El argumento four_classes de todas
   las funciones acepta la bandera de siempre (False = 3, True = 4) o el numero de clases (3, 4, 5): n_classes. Con 3
   y 4 el resultado es el mismo de antes.
3. Validacion real: RUNS/real_ztf, origen holdout & split val & ~excluir. meta_real_ztf.csv se lee con
   pipeline78.splits.read_val_meta (csv, solo las filas val llegan a pandas) y la fotometria filtra por oid dentro de
   pyarrow (filters), asi las filas de la mitad final nunca se materializan. No se usan las viejas ni las plantillas.
   La mitad val se parte en val_sel (elegir y calibrar) y val_rep (reportar) con pipeline78.splits.val_split.
4. Token por observacion: [dt/100, m - m_ref, 10 magerr (0 en UL), es_UL, g, r]. dt en dias (marco observado)
   desde la primera deteccion g/r de la curva de entrada. m_ref = MEDIANA de las detecciones de la curva de entrada
   (despues del aumento o la degradacion). dt va aparte al embedding temporal continuo del modelo.
5. UL: entran con su magnitud limite y es_UL = 1, solo en la ventana [t_primera - PRE_UL_DAYS, t_ultima] de las
   detecciones de la curva de entrada. Las sims no tienen UL despues de la ultima deteccion y las reales casi no
   (2 % de sus UL): cortar ahi deja a las dos con la misma regla.
6. Largo maximo MAX_LEN tokens. Si sobra, se recortan primero los UL (quedan al menos MAX_LEN // 8 si hay) y luego
   las detecciones, con indices equiespaciados en el tiempo (determinista).
7. Globales: sin z -> [(m_ref - 19) / 2] (brillo aparente, fotometria pura). Con z -> ademas [10 z, (M_ref + 18) / 2]
   con M_ref = m_ref - mu(z), LCDM plano H0 = 70, Om = 0.3 (la misma cosmologia de core.utils.DL_calculator).
8. Aumento (solo entrenamiento, cada epoca): con prob p_ronly se deja solo r (si r tiene >= MIN_DET detecciones),
   y con prob p_thin se conservan k ~ U{MIN_DET..n} detecciones al azar. Nunca quedan menos de MIN_DET. Los UL no
   se tocan.
9. Degradacion (evaluacion): bandas r o g+r, N detecciones al azar (N = 3, 5, 7) o todas. La curva con menos de
   max(N, MIN_DET) detecciones en esas bandas no entra a esa celda.
10. Split interno de las sims por PLANTILLA, estratificado por sn_type (II e IIb por separado): por tipo, plantillas
   ordenadas, permutadas con default_rng(seed + indice del tipo) y repartidas en n_folds grupos. El grupo `fold`
   es la validacion interna. La 3 y la 4 clases comparten la particion de los tipos comunes.
11. Banda (band_enc): "onehot" = [g, r] (N_FEAT = 6). "lambda" = un escalar (lambda_p - 5500 A) / 1000 A con la
   longitud de onda PIVOTE de la curva de transmision de data/filters, lambda_p = sqrt(int l T dl / int T/l dl)
   (N_FEAT = 5). La banda como longitud de onda en un solo canal es la entrada de Gupta et al. 2025
   (2025MNRAS.542L.132G, "median wavelength of the passband") y de ORACLE-2 (Shah et al. 2026, 2026arXiv260700228S,
   "mean channel wavelength"). Aca se usa la pivote calculada de nuestras curvas, no un valor de tabla.
   tokenize devuelve ademas el indice de banda por token (0 = g, 1 = r) para el TimeModulator de ATAT.
12. Truncamiento ORACLE-2 (aumento, trunc != "none"): "pow2" corta en t_cut = t_primera + 2^n dias con n ~ U(0, 10)
   continuo, re-sorteado por curva y por epoca (Shah et al. 2026, Sec. IV.1; el codigo del repo usa U(0, 11)).
   "frac" conserva las primeras floor(f n_det) detecciones con f ~ U(0.1, 1) (preset ZTF_Sims-lite del repo
   dev-ved30/Oracle, truncate_ZTF_SIM_light_curve_fractionally, que cuenta filas y no detecciones). "both" elige uno
   de los dos al azar. Adaptacion propia: el corte nunca deja menos de MIN_DET detecciones, y se aplica despues del
   modo solo r y antes del raleo. Con p_trunc < 1 solo esa fraccion de las vistas se trunca (revision H4: p = 1 casi
   nunca deja la curva entera, y la evaluacion principal es la curva completa).
13. Horizonte temporal (evaluacion, revision H4): solo las bandas de la celda y t <= t_primera + H dias (marco
   observado), con t_primera la primera deteccion en esas bandas. Los UL previos se conservan. None si quedan menos de
   MIN_DET detecciones.
"""
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pipeline78.paths import RUNS, FILTERS
from pipeline78.splits import read_val_meta

SIM_RUN = RUNS / "ztf_v78_t9_final"
REAL_DIR = RUNS / "real_ztf"
OUT_ROOT = RUNS / "nnclf"
SEED = 20261003
BANDS = ("g", "r")
BAND_ID = {"g": 0, "r": 1}
MIN_DET = 3
MAX_LEN = 128
PRE_UL_DAYS = 60.0
N_FEAT = 6                                   # band_enc "onehot"; con "lambda" son 5 (n_feat)
BAND_ENCODINGS = ("onehot", "lambda")
TRUNC_MODES = ("none", "pow2", "frac", "both")
LAM0, LAM_SCALE = 5500.0, 1000.0


def pivot_wavelength(path):
    """Longitud de onda pivote (A) de una curva de transmision de dos columnas (lambda en A, T)."""
    lam, T = np.loadtxt(path, unpack=True)
    return float(np.sqrt(np.trapezoid(lam * T, lam) / np.trapezoid(T / lam, lam)))


LAMBDA_PIVOT = {b: pivot_wavelength(FILTERS / f"ZTF_{b}.dat") for b in BANDS}     # g 4783.5, r 6417.1 A
LAMBDA_TOKEN = np.array([(LAMBDA_PIVOT[b] - LAM0) / LAM_SCALE for b in BANDS], np.float32)


def n_feat(band_enc="onehot"):
    if band_enc not in BAND_ENCODINGS:
        raise ValueError(f"band_enc desconocido: {band_enc}")
    return 6 if band_enc == "onehot" else 5


def curve_rng(seed, key, *salt):
    """rng propio de una curva: los sorteos de la degradacion no dependen de la lista ni del metodo (rev. B2)."""
    return np.random.default_rng([int(seed), zlib.crc32(str(key).encode()), *[int(s) for s in salt]])
SN_TYPES = ("Ia", "II", "IIb", "Ibc", "IIn")          # orden fijo: indice para el rng del split
SN_TYPE_CLASS = {"Ia": "Ia", "II": "II", "IIb": "II", "Ibc": "Ibc", "IIn": "IIn"}
SN_TYPE_CLASS_5 = {**SN_TYPE_CLASS, "IIb": "IIb"}      # 5 clases: la IIb sale de II
CLASES = {3: ("Ia", "II", "Ibc"), 4: ("Ia", "II", "Ibc", "IIn"), 5: ("Ia", "II", "IIb", "Ibc", "IIn")}
_COLS = ["mjd", "filter", "magnitud_proyectada", "magerr", "upperlimit"]


def n_classes(four_classes=False):
    """Numero de clases del modo pedido (regla 2): la bandera de siempre (False o None = 3, True = 4) o el numero."""
    if four_classes is None or isinstance(four_classes, (bool, np.bool_)):
        return 4 if four_classes else 3
    n = int(four_classes)
    if n not in CLASES:
        raise ValueError(f"modo de clases desconocido: {four_classes!r} (3, 4 o 5)")
    return n


def modo(cfg):
    """Modo de clases de una train.Config: 5 con five_classes, si no la bandera four_classes de siempre."""
    return 5 if getattr(cfg, "five_classes", False) else cfg.four_classes


def classes(four_classes=False):
    return CLASES[n_classes(four_classes)]


def class_of(sn_type, four_classes=False):
    """sn_type -> clase del clasificador, o None si queda fuera."""
    n = n_classes(four_classes)
    c = (SN_TYPE_CLASS_5 if n == 5 else SN_TYPE_CLASS).get(sn_type)
    return c if c in CLASES[n] else None


def n_glob(use_z):
    return 3 if use_z else 1


@dataclass
class Curve:
    key: str
    y: int
    t: np.ndarray        # mjd, ordenado
    band: np.ndarray     # 0 = g, 1 = r
    mag: np.ndarray      # magnitud o limite (UL)
    err: np.ndarray      # NaN en UL
    ul: np.ndarray       # bool
    z: float = np.nan
    w: float = 1.0
    template: str = ""
    sn_type: str = ""

    def n_det(self, bands=(0, 1)):
        return int(np.sum(~self.ul & np.isin(self.band, bands)))

    def subset(self, keep):
        return replace(self, t=self.t[keep], band=self.band[keep], mag=self.mag[keep], err=self.err[keep],
                       ul=self.ul[keep])


def _to_curves(rows, key, info):
    """rows: filas g/r con la columna key. info: indexado por key con y, z, w, template, sn_type."""
    rows = rows.sort_values([key, "mjd"], kind="stable")
    out = []
    for k, g in rows.groupby(key, sort=True):
        i = info.loc[k]
        out.append(Curve(key=str(k), y=int(i.y), t=g.mjd.to_numpy(np.float64),
                         band=(g["filter"].to_numpy() == "r").astype(np.int8),
                         mag=g.magnitud_proyectada.to_numpy(np.float32), err=g.magerr.to_numpy(np.float32),
                         ul=(g.upperlimit.to_numpy() != "F"), z=float(i.z), w=float(i.w),
                         template=str(i.template), sn_type=str(i.sn_type)))
    return out


# ---------------------------------------------------------------- sims
def sims_table(run_dir=SIM_RUN, four_classes=False):
    s = pd.read_parquet(Path(run_dir) / "_sims_all.parquet",
                        columns=["sim_id", "field", "part_index", "sn_type", "template", "z", "w_z",
                                 "n_det_g", "n_det_r"])
    s["cls"] = s.sn_type.map(lambda t: class_of(t, four_classes))
    return s[s.cls.notna()].reset_index(drop=True)


def split_templates(pairs, n_folds=5, fold=0, seed=SEED):
    """pairs: iterable de (template, sn_type). Devuelve el set de plantillas de validacion interna (regla 10)."""
    pairs = set(pairs)
    val = set()
    for st in sorted({s for _, s in pairs}, key=SN_TYPES.index):
        tp = sorted({t for t, s in pairs if s == st})
        perm = list(np.random.default_rng(seed + SN_TYPES.index(st)).permutation(tp))
        ch = np.array_split(np.array(perm, dtype=object), n_folds)[fold]
        val |= set(ch) if len(ch) else {perm[fold % len(perm)]}
    return val


def load_sims(run_dir=SIM_RUN, four_classes=False, min_det=MIN_DET, max_sims=None, sim_ids=None, seed=SEED):
    """Curvas de las sims con >= min_det detecciones g+r. max_sims: submuestra estratificada por clase (smoke)."""
    cls = classes(four_classes)
    s = sims_table(run_dir, four_classes)
    s = s[s.n_det_g + s.n_det_r >= min_det]
    if sim_ids is not None:
        s = s[s.sim_id.isin(set(sim_ids))]
    if max_sims and len(s) > max_sims:
        s = s.groupby("cls", group_keys=False).sample(frac=max_sims / len(s), random_state=seed % 2**32)
    by_field = {}
    for p in Path(run_dir).glob("*__*.parquet"):
        by_field.setdefault(p.name.split("__")[0], []).append(str(p))
    files = sorted(f for fld in s.field.unique() for f in by_field.get(fld, []))
    flt = pc.field("sim_id").isin(pa.array(s.sim_id.to_numpy())) & pc.field("filter").isin(list(BANDS))
    rows = ds.dataset(files, format="parquet").to_table(columns=["sim_id"] + _COLS, filter=flt).to_pandas()
    info = s.assign(y=s.cls.map({c: i for i, c in enumerate(cls)}), w=s.w_z).set_index("sim_id")
    return [c for c in _to_curves(rows, "sim_id", info) if c.n_det() >= min_det]


def balance_weights(y, w, n_cls):
    """w_z por balance de clases: cada clase suma el mismo peso total. Normalizado a media 1. La seleccion
    espectroscopica S(m) todavia no existe y no entra aca."""
    y, w = np.asarray(y, np.int64), np.asarray(w, np.float64)
    if not len(w):
        return w
    tot = np.bincount(y, weights=w, minlength=n_cls)
    f = np.divide(w.sum() / n_cls, tot, out=np.zeros(n_cls), where=tot > 0)
    sw = w * f[y]
    return sw / sw.mean()


# ---------------------------------------------------------------- reales (solo la mitad val)
def real_val_meta(real_dir=REAL_DIR, four_classes=False):
    """Metadatos de la mitad val de las clases pedidas. read_val_meta verifica que ninguna oid val este en otro split
    y no guarda las filas de la mitad final (revision H10)."""
    v = read_val_meta(Path(real_dir) / "meta_real_ztf.csv")
    v["oid"] = v.oid.astype(str)
    v["cls"] = v.sn_type.map(lambda t: class_of(t, four_classes))
    return v[v.cls.notna()].reset_index(drop=True)


def load_real_val(real_dir=REAL_DIR, four_classes=False, min_det=MIN_DET):
    """Curvas reales de validacion. Devuelve (curvas con >= min_det det g+r, oids val que no llegan)."""
    cls = classes(four_classes)
    v = real_val_meta(real_dir, four_classes)
    info = v.assign(y=v.cls.map({c: i for i, c in enumerate(cls)}), w=1.0, template="").set_index("oid")
    parts = []
    for st, grp in v.groupby("sn_type"):
        tab = pq.read_table(Path(real_dir) / f"{st}.parquet", columns=["oid"] + _COLS,
                            filters=[("oid", "in", sorted(grp.oid)), ("filter", "in", list(BANDS))])
        parts.append(tab.to_pandas())
    rows = pd.concat(parts, ignore_index=True)
    assert set(rows.oid) <= set(v.oid), "se colo una oid fuera de la mitad val"
    curves = [c for c in _to_curves(rows, "oid", info) if c.n_det() >= min_det]
    return curves, sorted(set(v.oid) - {c.key for c in curves})


# ---------------------------------------------------------------- representacion
def distmod(z, H0=70.0, om=0.3):
    z = np.atleast_1d(np.asarray(z, np.float64))
    zg = np.linspace(0.0, float(z.max()) * 1.01 + 1e-3, 2001)
    inv = 1.0 / np.sqrt(om * (1 + zg) ** 3 + 1 - om)
    dc = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zg))]) * 299792.458 / H0
    return 5 * np.log10((1 + z) * np.interp(z, zg, dc) * 1e5)


def _even(idx, n):
    if n >= len(idx):
        return idx
    if n <= 0:
        return idx[:0]
    return idx[np.round(np.linspace(0, len(idx) - 1, n)).astype(int)]


def token_idx(c, max_len=MAX_LEN):
    """Indices de las observaciones que entran como token (reglas 5 y 6). Los usa tambien el export de SuperNNova."""
    det = ~c.ul
    if not det.any():
        raise ValueError(f"{c.key}: sin detecciones")
    t0, t1 = c.t[det].min(), c.t[det].max()
    idx = np.flatnonzero(det | ((c.t >= t0 - PRE_UL_DAYS) & (c.t <= t1)))
    if len(idx) > max_len:
        d_idx, u_idx = idx[det[idx]], idx[~det[idx]]
        n_ul = min(len(u_idx), max(max_len - len(d_idx), max_len // 8))
        idx = np.sort(np.concatenate([_even(d_idx, max_len - n_ul), _even(u_idx, n_ul)]))
    return idx


def tokenize(c, max_len=MAX_LEN, use_magerr=True, use_z=False, band_enc="onehot"):
    """Curve -> (x [L, n_feat], dt [L] en dias, g [n_glob], b [L] indice de banda). Reglas 4 a 7 y 11."""
    nf = n_feat(band_enc)
    idx = token_idx(c, max_len)
    det = ~c.ul
    t0 = c.t[det].min()
    m_ref = float(np.median(c.mag[det]))
    ul = c.ul[idx]
    b = c.band[idx].astype(np.int64)
    dt = (c.t[idx] - t0).astype(np.float32)
    x = np.zeros((len(idx), nf), np.float32)
    x[:, 0] = dt / 100.0
    x[:, 1] = np.clip(c.mag[idx] - m_ref, -10, 10)
    if use_magerr:
        x[:, 2] = np.where(ul, 0.0, np.nan_to_num(c.err[idx]) * 10.0)
    x[:, 3] = ul
    if band_enc == "onehot":
        x[:, 4] = b == 0
        x[:, 5] = b == 1
    else:
        x[:, 4] = LAMBDA_TOKEN[b]
    g = [(m_ref - 19.0) / 2.0]
    if use_z:
        if not np.isfinite(c.z) or c.z <= 0:
            raise ValueError(f"{c.key}: z no valido ({c.z})")
        g += [10.0 * c.z, (m_ref - float(distmod(c.z)[0]) + 18.0) / 2.0]
    return x, dt, np.asarray(g, np.float32), b


def truncate(c, rng, mode, min_det=MIN_DET):
    """Regla 12. Corta la curva en t_cut desde la primera deteccion. Nunca deja menos de min_det detecciones."""
    if mode == "none":
        return c
    if mode not in TRUNC_MODES:
        raise ValueError(f"trunc desconocido: {mode}")
    td = c.t[~c.ul]
    if len(td) <= min_det:
        return c
    if mode == "both":
        mode = "pow2" if rng.random() < 0.5 else "frac"
    if mode == "pow2":
        t_cut = td[0] + 2.0 ** rng.uniform(0.0, 10.0)
    else:
        t_cut = td[max(min_det, int(len(td) * rng.uniform(0.1, 1.0))) - 1]
    return c.subset(c.t <= max(t_cut, td[min_det - 1]))


def augment(c, rng, p_thin=0.8, p_ronly=0.5, min_det=MIN_DET, trunc="none", p_trunc=1.0):
    """Reglas 8 y 12 (solo r, truncamiento, raleo). Si la curva de entrada trae >= min_det detecciones, la salida
    tambien."""
    if p_ronly > 0 and rng.random() < p_ronly and c.n_det((1,)) >= min_det:
        c = c.subset(c.band == 1)
    if trunc != "none" and p_trunc > 0 and rng.random() < p_trunc:
        c = truncate(c, rng, trunc, min_det)
    det = np.flatnonzero(~c.ul)
    if len(det) > min_det and p_thin > 0 and rng.random() < p_thin:
        k = int(rng.integers(min_det, len(det) + 1))
        keep = np.ones(len(c.t), bool)
        keep[rng.choice(det, len(det) - k, replace=False)] = False
        c = c.subset(keep)
    return c


def degrade(c, n, bands, rng, min_det=MIN_DET):
    """Regla 9. n = None es 'todas'. None si la curva no alcanza."""
    c = c.subset(np.isin(c.band, [BAND_ID[b] for b in bands]))
    det = np.flatnonzero(~c.ul)
    if len(det) < max(n or 0, min_det):
        return None
    if n is not None and len(det) > n:
        keep = np.ones(len(c.t), bool)
        keep[rng.choice(det, len(det) - n, replace=False)] = False
        c = c.subset(keep)
    return c


def cut_horizon(c, h_days, bands, min_det=MIN_DET):
    """Regla 13. h_days = None es la curva completa en esas bandas. None si no quedan min_det detecciones."""
    c = c.subset(np.isin(c.band, [BAND_ID[b] for b in bands]))
    det = ~c.ul
    if det.sum() < min_det:
        return None
    if h_days is not None:
        c = c.subset(c.t <= c.t[det].min() + h_days)
    return c if c.n_det() >= min_det else None
