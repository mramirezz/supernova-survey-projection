"""Puerta de realismo de features: bordes de plantilla (base, texp, tail) contra la validacion real ZTF (holdout
nuevo, mitad val), con la seleccion igualada en ambos lados. Uso:
    $PY -m pipeline78.realism prepare --runs base=DIR,texp=DIR,tail=DIR --out DIR --n 50 --seed 20261003
    run_parquet.py --parquet_dir DIR/<v>/parquet --output_dir DIR/<v>        (v = base, texp, tail, real)
    $PY -m pipeline78.realism report --out DIR
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from pipeline78.paths import PHD, RUNS
from pipeline78 import pilot_report as P

CLASES = ("Ia", "II", "IIb", "IIn", "Ibc")
VARIANTES = ("base", "texp", "tail")
M_CUT = 18.5
G_WIN = 2.0                                    # d: ventana de g alrededor del pico de r para el color
KEYS = ["oid", "part_index", "sn_type"]
FIT = ["f", "t_rise", "t_fall", "gamma"]
PARS = ("t_rise_rest", "t_fall_rest", "gamma_rest", "f", "M_r", "g_r")
SEL_COLS = ["sim_id", "oid", "part_index", "sn_type", "z", "m_peak_r", "peso", "variante"]
PAGE = PHD / "paper2_ZTF/figures_templates/pipeline78_realismo"
COLOR = {"base": "C0", "texp": "C1", "tail": "C2"}


def _det(d, b):
    """Detecciones de la banda b (upperlimit 'F', igual en sims y reales) con los nombres de reader.py."""
    x = d[(d["filter"] == b) & (d["upperlimit"] == "F")]
    return x.rename(columns={"mjd": "MJD", "magnitud_proyectada": "MAG", "magerr": "MAGERR"})[["MJD", "MAG", "MAGERR"]]


def photo(d):
    """m_peak_r (el mismo estimador que peak_mag_8h) y g - r al pico de r: mediana de las detecciones g agrupadas
    en 8 h a +-G_WIN d de la epoca del pico r, menos m_peak_r. NaN si faltan puntos."""
    r = P.apply_time_window_filter(_det(d, "r").dropna(subset=["MJD", "MAG"]), window_hours=8.0)
    if len(r) < P.MIN_PTS:
        return np.nan, np.nan
    i = int(np.argmin(r["MAG"].to_numpy(float)))
    m, tp = float(r["MAG"].iloc[i]), float(r["MJD"].iloc[i])
    g = P.apply_time_window_filter(_det(d, "g").dropna(subset=["MJD", "MAG"]), window_hours=8.0)
    g = g[(g["MJD"].astype(float) - tp).abs() <= G_WIN] if len(g) else g
    return m, (float(g["MAG"].astype(float).median()) - m) if len(g) else np.nan


def z_weights(z_sim, z_real):
    """Mismo metodo que g1_selcut.py: 4 bins por cuantiles de la z real con bordes extremos 0 y 9.
    Peso = fraccion real del bin / conteo simulado del bin."""
    z_sim, z_real = np.asarray(z_sim, float), np.asarray(z_real, float)
    edges = np.quantile(z_real, [0, .25, .5, .75, 1]); edges[0], edges[-1] = 0, 9
    ka = np.clip(np.digitize(z_sim, edges) - 1, 0, 3); kb = np.clip(np.digitize(z_real, edges) - 1, 0, 3)
    fa = np.bincount(ka, minlength=4).astype(float); fb = np.bincount(kb, minlength=4) / len(z_real)
    return np.where(fa[ka] > 0, fb[ka] / np.maximum(fa[ka], 1), 0.0)


def sim_peaks(run_dir):
    """{sim_id: m_peak_r} de un run, un archivo de campo a la vez (solo detecciones r)."""
    pk = {}
    for p in sorted(Path(run_dir).glob("*__*.parquet")):
        d = pd.read_parquet(p, columns=["sim_id", "filter", "mjd", "magnitud_proyectada", "magerr", "upperlimit"])
        d = d[(d["filter"] == "r") & (d["upperlimit"] == "F")]
        pk.update({s: P.peak_mag_8h(_det(x, "r")) for s, x in d.groupby("sim_id")})
    return pk


def _clases_variante(cfg):
    """Clases que pueden cambiar respecto de base: la cola toca todas, texp solo las Ia."""
    if cfg.get("edge_post", "none") != "none":
        return list(CLASES)
    return ["Ia"] if cfg.get("edge_pre", "window") == "texp" else []


def _write_sims(run_dir, sel, dest):
    """Fotometria de las sim_id de sel (una lectura por archivo de campo) -> dest/<clase>.parquet."""
    ids, fr = set(sel.sim_id), []
    for f in sorted(set(sel.oid)):
        for p in sorted(Path(run_dir).glob(f"{f}__*.parquet")):
            d = pd.read_parquet(p)
            fr.append(d[d.sim_id.isin(ids)])
    ph = pd.concat(fr, ignore_index=True)
    if set(ph.sim_id) != ids:
        raise ValueError(f"{run_dir}: {len(ids - set(ph.sim_id))} sim_id sin fotometria")
    dest.mkdir(parents=True, exist_ok=True)
    for c, d in ph.groupby("sn_type"):
        d.reset_index(drop=True).to_parquet(dest / f"{c}.parquet", index=False)


def _pick(rng, k, n, p=None):
    """Indices ordenados de un sorteo sin reemplazo de min(n, k) entre k candidatos (con p: solo los de peso > 0)."""
    m = min(n, k if p is None else int((p > 0).sum()))
    if m == 0:
        return np.array([], int)
    return np.sort(rng.choice(k, size=m, replace=False, p=None if p is None else p / p.sum()))


def prepare(runs, out, n=50, seed=20261003, real_dir=RUNS / "real_ztf"):
    out, real_dir = Path(out).expanduser(), Path(real_dir).expanduser()
    runs = {k: Path(v).expanduser() for k, v in runs.items()}
    if "base" not in runs:
        raise ValueError("--runs necesita base=DIR")
    if (out / "selection.csv").exists():      # run_parquet cachea las tareas por ruta y acumula features.csv
        raise ValueError(f"{out} ya tiene una seleccion. No se mezcla: usa otra carpeta.")
    man = {k: json.loads((d / "run_manifest.json").read_text()) for k, d in runs.items()}
    if len({m["seed"] for m in man.values()}) > 1:
        raise ValueError(f"semillas distintas entre runs: { {k: m['seed'] for k, m in man.items()} }")
    rng = np.random.default_rng(seed)
    meta = pd.read_csv(real_dir / "meta_real_ztf.csv")
    meta = meta[(meta.origen == "holdout") & (meta.split == "val")]
    sims = pd.read_parquet(runs["base"] / "_sims_all.parquet")
    sims = sims[sims.status == "ok"].copy()
    sims["m_peak_r"] = sims.sim_id.map(sim_peaks(runs["base"]))
    reales, sel = [], []
    for c in CLASES:
        m, p = meta[meta.sn_type == c], real_dir / f"{c}.parquet"
        if not len(m) or not p.exists():
            print(f"WARNING: {c}: sin reales en holdout val, la clase queda fuera")
            continue
        ph = pd.read_parquet(p)
        ph = ph[ph.oid.isin(set(m.oid))]
        m = m.assign(m_peak_r=m.oid.map({o: P.peak_mag_8h(_det(d, "r")) for o, d in ph.groupby("oid")}))
        m = m[m.m_peak_r < M_CUT]                        # NaN (< 7 puntos agrupados) no pasa
        m = m.iloc[_pick(rng, len(m), n)]
        if not len(m):
            print(f"WARNING: {c}: ninguna real pasa el corte, la clase queda fuera")
            continue
        reales.append(m)
        (out / "real/parquet").mkdir(parents=True, exist_ok=True)
        ph[ph.oid.isin(set(m.oid))].reset_index(drop=True).to_parquet(out / "real/parquet" / f"{c}.parquet", index=False)
        a = sims[(sims.sn_type == c) & (sims.m_peak_r < M_CUT)]
        w = z_weights(a.z, m.z) if len(a) else np.zeros(0)
        i = _pick(rng, len(a), n, w)
        sel.append(a.iloc[i].assign(peso=w[i]))
    real = pd.concat(reales, ignore_index=True)
    real[["oid", "sn_type", "subtipo", "z", "m_peak_r"]].to_csv(out / "real_selection.csv", index=False)
    base = pd.concat(sel, ignore_index=True).rename(columns={"field": "oid"})
    _write_sims(runs["base"], base, out / "base/parquet")
    rows = [base.assign(variante="base")]
    for v, d in runs.items():
        if v == "base":
            continue
        s = base[base.sn_type.isin(_clases_variante(man[v]["cfg"]))]
        vs = pd.read_parquet(d / "_sims_all.parquet").set_index("sim_id")
        ok = (vs.status.reindex(s.sim_id) == "ok").to_numpy()
        if (~ok).any():
            print(f"WARNING: {v}: {(~ok).sum()} sim_id no ok en la variante, se omiten: {s.sim_id[~ok].tolist()}")
        s = s[ok]
        if not len(s):
            print(f"WARNING: {v}: ninguna clase seleccionada cambia en esta variante (cfg sin edge_pre/edge_post)")
            continue
        fis = ["template", "z", "m_peak_abs"]
        if (vs.loc[s.sim_id, fis].to_numpy() != s[fis].to_numpy()).any():
            raise ValueError(f"{v}: las mismas sim_id tienen otra fisica que base (otra semilla, campos o commit)")
        _write_sims(d, s, out / v / "parquet")
        rows.append(s.assign(variante=v))
    selc = pd.concat(rows, ignore_index=True)[SEL_COLS]
    selc.to_csv(out / "selection.csv", index=False)
    (out / "prepare.json").write_text(json.dumps(dict(runs={k: str(d) for k, d in runs.items()}, real=str(real_dir),
                                                      n=n, seed=seed, m_cut=M_CUT), indent=1))
    cnt = selc.groupby(["sn_type", "variante"]).size().unstack(fill_value=0)
    cnt["real"] = real.groupby("sn_type").size()
    print(cnt.fillna(0).astype(int).to_string())
    med = pd.DataFrame({"z_sim": base.groupby("sn_type").z.median(), "z_real": real.groupby("sn_type").z.median(),
                        "m_sim": base.groupby("sn_type").m_peak_r.median(),
                        "m_real": real.groupby("sn_type").m_peak_r.median()})
    print(med.round(3).to_string())
    return selc, real


def boot_sigma(x, y, rng, n=1000):
    """sigma por bootstrap de (mediana sim - mediana real), remuestreando las dos muestras."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    bx = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    by = np.median(y[rng.integers(0, len(y), (n, len(y)))], axis=1)
    return float(np.std(bx - by, ddof=1))


def load_variant(out, v, s):
    """s: seleccion (KEYS + z). Agrega las features r de run_parquet en reposo y la fotometria (M_r, g_r)."""
    f = out / v / "features/features.csv"
    if f.exists():
        ft = pd.read_csv(f)
        ft = ft[ft.filter_band == "r"].drop_duplicates(KEYS)[KEYS + FIT]
    else:
        print(f"WARNING: {v}: no existe {f}, sin features")
        ft = pd.DataFrame(columns=KEYS + FIT)
    ft = ft.astype({"oid": str, "part_index": int, "sn_type": str})
    x = s[KEYS + ["z"]].astype({"oid": str, "part_index": int}).merge(ft, on=KEYS, how="left")
    for p in ("t_rise", "t_fall", "gamma"):
        x[f"{p}_rest"] = x[p].astype(float) / (1.0 + x.z)
    ph = pd.concat([pd.read_parquet(q) for q in sorted((out / v / "parquet").glob("*.parquet"))], ignore_index=True)
    pc = pd.DataFrame([(str(o), int(k), c, *photo(d)) for (o, k, c), d in ph.groupby(KEYS)],
                      columns=KEYS + ["m_peak_r", "g_r"])
    x = x.merge(pc, on=KEYS, how="left")
    x["M_r"] = x.m_peak_r - np.array([P.mu(z) for z in x.z])
    return x


def report(out, page=PAGE, seed=20261003, n_boot=1000):
    out, page = Path(out).expanduser(), Path(page)
    sel = pd.read_csv(out / "selection.csv")
    real = pd.read_csv(out / "real_selection.csv").assign(part_index=0)
    data = {v: load_variant(out, v, sel[sel.variante == v]) for v in VARIANTES if (sel.variante == v).any()}
    data["real"] = load_variant(out, "real", real)
    rng = np.random.default_rng(seed)
    fin = lambda s: s.to_numpy(float)[np.isfinite(s.to_numpy(float))]
    rows = []
    for c in CLASES:
        y0 = data["real"][data["real"].sn_type == c]
        for v in VARIANTES:
            x0 = data[v][data[v].sn_type == c] if v in data else []
            if not len(x0) or not len(y0):
                continue
            for p in PARS:
                x, y = fin(x0[p]), fin(y0[p])
                ms, mr = (float(np.median(x)) if len(x) else np.nan), (float(np.median(y)) if len(y) else np.nan)
                D, pv = ks_2samp(x, y) if len(x) and len(y) else (np.nan, np.nan)
                rows.append(dict(clase=c, variante=v, par=p, n_sel=len(x0), n_sim=len(x), n_real=len(y),
                                 frac_validos_sim=len(x) / len(x0), frac_validos_real=len(y) / len(y0),
                                 med_sim=ms, med_real=mr, delta=ms - mr, sigma_boot=boot_sigma(x, y, rng, n_boot),
                                 ks_D=float(D), ks_p=float(pv)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "realismo_tabla.csv", index=False)
    clases = [c for c in CLASES if c in set(tab.clase)] if len(tab) else []
    page.mkdir(parents=True, exist_ok=True)
    if clases:
        fig, axs = plt.subplots(len(clases), len(PARS), figsize=(3.0 * len(PARS), 2.3 * len(clases)), squeeze=False)
        for ax, (c, p) in zip(axs.ravel(), [(c, p) for c in clases for p in PARS]):
            vals = {v: fin(d.loc[d.sn_type == c, p]) for v, d in data.items()}
            allv = np.concatenate([a for a in vals.values() if len(a)] or [np.zeros(1)])
            bins = np.linspace(*np.percentile(allv, [1, 99]) + np.array([-1e-6, 1e-6]), 21)
            if len(vals["real"]):
                ax.hist(vals["real"], bins=bins, density=True, color="0.6", alpha=0.5, label="real")
            for v in VARIANTES:
                if v in vals and len(vals[v]):
                    ax.hist(vals[v], bins=bins, density=True, histtype="step", color=COLOR[v], lw=1.3, label=v)
            ax.set_title(f"{c}  {p}", fontsize=8); ax.tick_params(labelsize=7)
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(page / "realismo.png", dpi=90); plt.close(fig)
    prep = json.loads((out / "prepare.json").read_text()) if (out / "prepare.json").exists() else {}
    regla = (f"Seleccion igualada en ambos lados: real = holdout nuevo, mitad val (origen holdout, split val). "
             f"Sims = simulaciones ok del run base. En los dos lados m_peak_r &lt; {M_CUT} y al menos {P.MIN_PTS} puntos r "
             f"agrupados en 8 h (peak_mag_8h). A lo m&aacute;s n = {prep.get('n', '?')} por clase de proyecci&oacute;n "
             f"(semilla {prep.get('seed', '?')}). Las sims se sortean sin reemplazo con un peso que iguala la z de las reales "
             f"seleccionadas (4 bins por cuantiles, bordes 0 y 9, como g1_selcut.py). texp y tail usan las mismas sim_id "
             f"(texp solo cambia las Ia). Features de run_parquet.py en r; t_rise, t_fall y gamma en reposo (/(1+z)). "
             f"M_r = m_peak_r - &mu;(z). g - r: mediana de g agrupada en 8 h a &plusmn;{G_WIN:g} d del pico r, menos m_peak_r. "
             f"&Delta; = mediana sim - mediana real, &sigma; por bootstrap ({n_boot} remuestreos de las dos muestras), KS de dos muestras.")
    (page / "index.html").write_text(
        "<html><head><meta charset='utf-8'><title>Realismo de features v78</title></head>"
        "<body style='font-family:sans-serif;max-width:1500px;margin:auto'>"
        "<h1>Puerta de realismo: bordes de plantilla contra la validaci&oacute;n real ZTF</h1>"
        f"<p>Salida: {out}. Runs: {prep.get('runs', {})}</p><p>{regla}</p>"
        "<p>Real relleno (gris); base, texp y tail como contornos.</p><img src='realismo.png' width='100%'>"
        f"<h2>Tabla</h2>{tab.round(3).to_html(index=False) if len(tab) else '<p>sin filas</p>'}</body></html>")
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("prepare")
    a.add_argument("--runs", required=True, help="base=DIR,texp=DIR,tail=DIR")
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=50)
    a.add_argument("--seed", type=int, default=20261003)
    a.add_argument("--real", default=str(RUNS / "real_ztf"))
    b = sp.add_parser("report")
    b.add_argument("--out", required=True)
    b.add_argument("--page", default=str(PAGE))
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        runs = dict(kv.split("=", 1) for kv in args.runs.split(","))
        prepare(runs, args.out, args.n, args.seed, args.real)
    else:
        tab = report(args.out, args.page)
        print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
