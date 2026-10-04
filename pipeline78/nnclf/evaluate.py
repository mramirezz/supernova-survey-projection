"""Evaluacion: reales de validacion (mitad val, nunca la final) y sims de validacion interna, con curva de degradacion.

TABLA DE PREDICCIONES (preds.parquet): una fila por (curva, modo, bandas, N, sorteo) con y, w, n_det y p_<clase>.
La producen la red (run_eval), el ensemble (ensemble.py), SuperNNova (snn.py) y Villar (baseline.py, solo la celda
principal), y `summarize` calcula para todos las mismas metricas.

CELDAS: bandas r o g+r, N = 3, 5, 7 detecciones al azar o todas (data.degrade). Con N hay n_draws sorteos, 'todas' es
un solo pase. Cada curva tiene su propio rng (data.curve_rng con seed, oid, bandas, N y sorteo), asi que el mismo
sorteo de puntos se repite entre variantes, clases y metodos (revision B2).
- modo "natural": entra a la celda la curva con >= max(N, 3) detecciones en esas bandas. La poblacion cambia con N.
- modo "fixed" (revision M1): solo las curvas con >= FIXED_MIN = 7 detecciones en esas bandas, degradadas a 3, 5, 7 y
  todas. La muestra es la misma en las cuatro celdas, asi que la curva mide solo el efecto de quitar puntos.
- modo "horizon" (revision H4): las curvas de la muestra fija que ademas tienen >= 3 detecciones en esas bandas dentro
  de los primeros HORIZONS[0] = 10 dias, cortadas en t <= t_primera + H con H = 10, 20 y 50 dias (marco observado,
  data.cut_horizon), mas la curva completa. La muestra es la misma en las cuatro celdas. Un solo pase (sin sorteo).
  Es la evaluacion temprana que el truncamiento de ORACLE-2 promete mejorar, al lado del raleo al azar.
Las reales van sin peso. Las sims de validacion van ponderadas por w_z (poblacion volumetrica), sin balance de clases.

SUBCONJUNTOS DE LAS REALES (revision H1, pipeline78.splits): la mitad val se parte en val_sel (con ella se eligen las
configuraciones y se ajustan la temperatura y los priors) y val_rep (solo para reportar: la estimacion honesta). Cada
metrica de las reales se da en val_rep, val_sel y val completo, con el sufijo _rep, _sel y _val en resumen.csv y la
columna subset en degradation*.csv. Las figuras muestran val_rep (y val completo como referencia).

METRICAS (summarize): exactitud, exactitud balanceada, F1 por clase y macro, matriz de confusion (filas = verdadera),
cobertura (reales clasificadas / reales del subconjunto de las clases), la misma celda principal restringida a las oids
que Villar puede clasificar (revision M2), calibracion (calib.py: T, T' y priors ajustados en val_sel) y la exactitud
balanceada en las sims de validacion por plantilla (brecha sim -> real).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from pipeline78 import splits
from pipeline78.nnclf import data as D
from pipeline78.nnclf import calib
from pipeline78.nnclf.train import load_model, logits_of, pick_device

CELLS = [(b, n) for b in (("r",), ("g", "r")) for n in (3, 5, 7, None)]
FIXED_MIN = 7
HORIZONS = (10, 20, 50)
SUBSET_ORDER = ("val_rep", "val_sel", "val")
SUFFIX = {"val_rep": "rep", "val_sel": "sel", "val": "val"}
COLORS = {"Ia": "tab:blue", "II": "tab:green", "Ibc": "tab:red", "IIn": "tab:purple"}
META = ["key", "dataset", "mode", "bands", "N", "draw", "y", "w", "n_det"]


def metrics(y, yhat, classes, w=None):
    lab = list(range(len(classes)))
    f1 = f1_score(y, yhat, labels=lab, average=None, sample_weight=w, zero_division=0)
    return {"n": int(len(y)), "acc": float(accuracy_score(y, yhat, sample_weight=w)),
            "bal_acc": float(balanced_accuracy_score(y, yhat, sample_weight=w)),
            "f1_macro": float(np.mean(f1)), **{f"f1_{c}": float(v) for c, v in zip(classes, f1)},
            "confusion": confusion_matrix(y, yhat, labels=lab, sample_weight=w).round(3).tolist()}


# ---------------------------------------------------------------- celdas y tabla de predicciones
def horizon_label(h):
    return "all" if h is None else f"{h}d"


def enumerate_cells(curves, n_draws=5, seed=D.SEED, fixed=True, horizon=None):
    """(meta DataFrame, curvas degradadas) de todas las celdas. Determinista por curva. horizon = None sigue a fixed."""
    horizon = fixed if horizon is None else horizon
    meta, dcs = [], []
    for bands, n in CELLS:
        bid = tuple(D.BAND_ID[b] for b in bands)
        pools = [("natural", curves)]
        if fixed:
            pools.append(("fixed", [c for c in curves if c.n_det(bid) >= FIXED_MIN]))
        for mode, pool in pools:
            for d in range(1 if n is None else n_draws):
                for c in pool:
                    dc = D.degrade(c, n, bands, D.curve_rng(seed, c.key, len(bands), n or 0, d))
                    if dc is None:
                        continue
                    meta.append((c.key, mode, "+".join(bands), "all" if n is None else str(n), d, c.y, c.w,
                                 dc.n_det()))
                    dcs.append(dc)
    if horizon:
        for bands in dict.fromkeys(b for b, _ in CELLS):
            bid = tuple(D.BAND_ID[b] for b in bands)
            pool = [c for c in curves if c.n_det(bid) >= FIXED_MIN and D.cut_horizon(c, HORIZONS[0], bands) is not None]
            for h in HORIZONS + (None,):
                for c in pool:
                    dc = D.cut_horizon(c, h, bands)
                    meta.append((c.key, "horizon", "+".join(bands), horizon_label(h), 0, c.y, c.w, dc.n_det()))
                    dcs.append(dc)
    return pd.DataFrame(meta, columns=[m for m in META if m != "dataset"]), dcs


def predict_table(prob_fn, curves, classes, dataset, n_draws=5, seed=D.SEED, fixed=True):
    meta, dcs = enumerate_cells(curves, n_draws, seed, fixed)
    p = prob_fn(dcs) if dcs else np.zeros((0, len(classes)))
    meta.insert(1, "dataset", dataset)
    for i, c in enumerate(classes):
        meta[f"p_{c}"] = p[:, i].astype(np.float32)
    return meta


def nn_prob_fn(model, cfg, device, chunk=4096):
    def f(dcs):
        out = []
        for i in range(0, len(dcs), chunk):
            enc = [D.tokenize(dc, cfg.max_len, cfg.use_magerr, cfg.use_z, cfg.band_enc) for dc in dcs[i:i + chunk]]
            out.append(torch.softmax(logits_of(model, enc, device), 1).numpy())
        return np.concatenate(out)
    return f


# ---------------------------------------------------------------- resumen
def _pcols(classes):
    return [f"p_{c}" for c in classes]


def cell_rows(tab, classes, weighted):
    rows = []
    for (mode, bands, N, d), g in tab.groupby(["mode", "bands", "N", "draw"], sort=False):
        p = g[_pcols(classes)].to_numpy()
        rows.append({"mode": mode, "bands": bands, "N": N, "draw": int(d), "n_eval": len(g),
                     **metrics(g.y.to_numpy(), p.argmax(1), classes, g.w.to_numpy() if weighted else None)})
    return rows


def aggregate(rows, classes):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion"} for r in rows])
    cols = [c for c in ["acc", "bal_acc", "f1_macro"] + [f"f1_{k}" for k in classes] if c in df]
    g = df.groupby(["mode", "bands", "N"], sort=False)
    agg = g[["n_eval"]].first().join(g[cols].mean().add_suffix("_mean")).join(g[cols].std().add_suffix("_std"))
    return agg.join(g.size().rename("n_draws")).reset_index()


def main_cell(tab):
    return tab[(tab["mode"] == "natural") & (tab.bands == "g+r") & (tab.N == "all")]


def val_subsets(real_dir=D.REAL_DIR, four_classes=False):
    """{"val_sel": set de oids, "val_rep": set de oids} de las clases del clasificador (pipeline78.splits.val_split)."""
    sel, rep = splits.val_split(D.real_val_meta(real_dir, four_classes))
    return {"val_sel": set(sel), "val_rep": set(rep)}


def _subset_frames(real, subsets):
    """Filas reales por subconjunto, en el orden val_rep, val_sel, val. Sin particion solo hay val."""
    out = {}
    if subsets:
        for s in ("val_rep", "val_sel"):
            out[s] = real[real.key.isin(subsets[s])]
    out["val"] = real
    return out


def _subset_col(keys, subsets):
    keys = pd.Series(np.asarray(keys)).astype(str)
    if not subsets:
        return np.full(len(keys), "", dtype=object)
    return np.where(keys.isin(subsets["val_sel"]), "val_sel", np.where(keys.isin(subsets["val_rep"]), "val_rep", ""))


def summarize(tab, classes, n_real_total, villar_oids=None, subsets=None, n_bins=calib.N_BINS):
    """Metricas comunes a todos los metodos a partir de la tabla de predicciones.

    subsets: {"val_sel": oids, "val_rep": oids} (val_subsets). Con particion, cada metrica de las reales se da por
    subconjunto (main_by_subset, columna subset de agg) y la calibracion se ajusta en val_sel."""
    real, sims = tab[tab.dataset == "real"], tab[tab.dataset == "sims"]
    frames = _subset_frames(real, subsets)
    rows_r = {s: cell_rows(fr, classes, False) for s, fr in frames.items()}
    rows_s = cell_rows(sims, classes, True)
    m = main_cell(real)
    p, y = m[_pcols(classes)].to_numpy(np.float64), m.y.to_numpy()
    vset = set(villar_oids) if villar_oids is not None else None
    by = {}
    for s, fr in frames.items():
        ms = main_cell(fr)
        n_tot = int(n_real_total) if s == "val" else len(subsets[s])
        b = {"n_total": n_tot, "n_main": int(len(ms)), "coverage": float(len(ms) / max(n_tot, 1)),
             "metrics": metrics(ms.y.to_numpy(), ms[_pcols(classes)].to_numpy().argmax(1), classes)
             if len(ms) else None}
        if vset is not None:
            v = ms[ms.key.isin(vset)]
            b["villar_oids"] = {"n_comun": int(len(v)), "metrics": metrics(
                v.y.to_numpy(), v[_pcols(classes)].to_numpy().argmax(1), classes) if len(v) else None}
        by[s] = b
    res = {"classes": list(classes), "n_real_val": int(n_real_total), "n_main": int(len(m)),
           "coverage": float(len(m) / max(n_real_total, 1)),
           "main_real_all_gr": metrics(y, p.argmax(1), classes) if len(m) else None,
           "main_by_subset": by, "particion": "val_sel/val_rep (pipeline78.splits)" if subsets else "sin particion",
           "cells_real": [dict(r, subset=s) for s, rr in rows_r.items() for r in rr], "cells_sims": rows_s}
    if len(m) >= 10:
        sel = rep = None
        if subsets:
            sel, rep = m.key.isin(subsets["val_sel"]).to_numpy(), m.key.isin(subsets["val_rep"]).to_numpy()
            if sel.sum() < 10:
                sel = rep = None
        res["calibration"] = calib.calibration_report(p, y, sel, rep, n_cls=len(classes), n_bins=n_bins)
    sm = main_cell(sims)
    if len(sm):
        res["sims_main_all_gr"] = metrics(sm.y.to_numpy(), sm[_pcols(classes)].to_numpy().argmax(1), classes,
                                          sm.w.to_numpy())
    if vset is not None:
        res["villar_oids"] = {"n_villar_oids": int(len(vset)), **by["val"]["villar_oids"]}
    agg = pd.concat([aggregate(rr, classes).assign(dataset="real", subset=s) for s, rr in rows_r.items()]
                    + [aggregate(rows_s, classes).assign(dataset="sims", subset="sims")], ignore_index=True)
    return res, agg


def summary_row(name, method, res, agg, classes):
    """Una fila de resumen.csv. Las metricas de las reales llevan el sufijo del subconjunto: _rep (honesta), _sel
    (con ella se elige) y _val (val completo, referencia)."""
    row = {"name": name, "method": method, "n_real_val": res["n_real_val"]}
    by = res.get("main_by_subset") or {"val": {"n_total": res["n_real_val"], "n_main": res["n_main"],
                                               "coverage": res["coverage"], "metrics": res.get("main_real_all_gr"),
                                               "villar_oids": res.get("villar_oids")}}
    for s in SUBSET_ORDER:
        if s not in by:
            continue
        sf, b = SUFFIX[s], by[s]
        mr = b.get("metrics") or {}
        row[f"n_{sf}"] = b.get("n_main")
        row[f"coverage_{sf}"] = b.get("coverage")
        for k in ["acc", "bal_acc", "f1_macro"] + [f"f1_{c}" for c in classes]:
            row[f"{k}_{sf}"] = mr.get(k)
        vo = b.get("villar_oids") or {}
        row[f"n_villar_oids_{sf}"] = vo.get("n_comun")
        row[f"bal_acc_villar_oids_{sf}"] = (vo.get("metrics") or {}).get("bal_acc")
        row[f"f1_macro_villar_oids_{sf}"] = (vo.get("metrics") or {}).get("f1_macro")
    cal = res.get("calibration") or {}
    row.update({"T": cal.get("T"), "T_prior": cal.get("T_prior")})
    for s in SUBSET_ORDER:
        b = (cal.get("by_subset") or {}).get(s) or {}
        for k in ("ece_raw", "ece_ts", "ece_ts_prior", "acc_ts_prior", "bal_acc_ts_prior"):
            row[f"{k}_{SUFFIX[s]}"] = b.get(k)
    row["ece_ts_cv5_val"] = cal.get("ece_ts_cv5_val", cal.get("ece_ts_cv5"))
    sm = res.get("sims_main_all_gr") or {}
    row["sims_bal_acc"] = sm.get("bal_acc")
    for sf in ("rep", "val"):
        r = row.get(f"bal_acc_{sf}")
        row[f"gap_sim_real_{sf}"] = (sm["bal_acc"] - r) if sm and r is not None else None
    if len(agg):
        if "subset" not in agg:                                       # corridas anteriores a la particion
            agg = agg.assign(subset=np.where(agg.dataset == "real", "val", "sims"))
        for mode in ("fixed", "horizon"):
            fx = agg[(agg.dataset == "real") & (agg["mode"] == mode)]
            for _, r in fx.iterrows():
                sf = SUFFIX.get(r.subset, r.subset)
                row[f"{mode}_{r.bands}_{r.N}_bal_acc_{sf}"] = r.bal_acc_mean
                row[f"{mode}_{r.bands}_n_{sf}"] = r.n_eval
    return row


def write_outputs(out, tab, res, agg, classes, extra=None, subsets=None):
    """Escribe metrics.json, degradation*.csv, confusion por subconjunto, pred_real_val.csv (con la columna subset) y
    las figuras (val_rep, con val completo como referencia)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out / "preds.parquet", index=False)
    res = {**(extra or {}), **res}
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    if len(agg):
        for mode, fname in (("natural", "degradation.csv"), ("fixed", "degradation_fixed.csv"),
                            ("horizon", "degradation_horizon.csv")):
            agg[agg["mode"] == mode].to_csv(out / fname, index=False)
    m = main_cell(tab[tab.dataset == "real"])
    if res.get("main_real_all_gr"):
        by = res.get("main_by_subset") or {}
        for s, b in by.items():
            if b.get("metrics"):
                pd.DataFrame(b["metrics"]["confusion"], index=classes, columns=classes).to_csv(
                    out / f"confusion_real_all_gr_{SUFFIX[s]}.csv")
        p = m[_pcols(classes)].to_numpy()
        pd.DataFrame({"oid": m.key.to_numpy(), "subset": _subset_col(m.key, subsets),
                      "y_true": [classes[i] for i in m.y], "y_pred": [classes[i] for i in p.argmax(1)],
                      "n_det": m.n_det.to_numpy(),
                      **{c: m[c].to_numpy() for c in m.columns if c.startswith(("p_", "std_"))}}).to_csv(
            out / "pred_real_val.csv", index=False)
        plt = _style()
        if len(agg):
            plot_degradation(agg, out / "fig_degradation", plt)
            if (agg["mode"] == "horizon").any():
                plot_horizon(agg, out / "fig_horizon", plt)
        s_fig = "val_rep" if (by.get("val_rep") or {}).get("metrics") else "val"
        cm = (by.get(s_fig) or {}).get("metrics", res["main_real_all_gr"])["confusion"]
        plot_confusion(cm, classes, out / "fig_confusion_real", plt, title=f"ZTF {s_fig.replace('_', ' ')}")
        if "calibration" in res:
            calib.plot_reliability(res["calibration"], out / "fig_reliability", plt)
    return res


def print_summary(name, res, agg):
    if len(agg):
        cols = ["dataset", "subset", "mode", "bands", "N", "n_eval", "bal_acc_mean", "bal_acc_std", "f1_macro_mean"]
        a = agg[agg.subset.isin(["val_rep", "sims"])] if "subset" in agg and (agg.subset == "val_rep").any() else agg
        print(a[[c for c in cols if c in a]].round(3).to_string(index=False), flush=True)
    by = res.get("main_by_subset") or {}
    cal = res.get("calibration") or {}
    for s in SUBSET_ORDER:
        b = by.get(s) or {}
        mr = b.get("metrics")
        if not mr:
            continue
        vo = (b.get("villar_oids") or {}).get("metrics") or {}
        cb = (cal.get("by_subset") or {}).get(s) or {}
        print(f"[nnclf] {name} {s:7s} todas g+r: acc {mr['acc']:.3f} bal {mr['bal_acc']:.3f} f1 {mr['f1_macro']:.3f} "
              f"(n = {mr['n']}, cobertura {b['coverage']:.3f})"
              + (f" | oids Villar bal {vo['bal_acc']:.3f}" if vo else "")
              + (f" | ECE {cb['ece_raw']:.3f} -> T {cb['ece_ts']:.3f} -> T'+priors {cb['ece_ts_prior']:.3f}"
                 f" (bal {cb['bal_acc_ts_prior']:.3f})" if cb else ""), flush=True)
    if cal:
        print(f"[nnclf] {name} calibracion ajustada en {cal.get('fit_on')}: T {cal['T']:.2f}, T' {cal['T_prior']:.2f},"
              f" priors {np.round(cal['priors_val_sel'], 3).tolist()}" + (" (T EN EL BORDE)" if cal["T_en_borde"]
                                                                          else ""), flush=True)


# ---------------------------------------------------------------- dominio y figuras
def _wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(np.asarray(w, float)[o])
    return float(np.asarray(x, float)[o][np.searchsorted(cw, cw[-1] / 2)])


def domain_check(sims, real):
    """Medianas de la representacion en sims de validacion y reales (las de sims tambien ponderadas por w_z), para
    ver corrimientos de dominio. magerr va ademas por bin de magnitud: a igual m es el ruido, no el brillo."""
    def med(curves):
        if not curves:
            return {}
        m = np.concatenate([c.mag[~c.ul] for c in curves])
        e = np.concatenate([c.err[~c.ul] for c in curves])
        nd = np.array([c.n_det() for c in curves], float)
        mr = np.array([np.median(c.mag[~c.ul]) for c in curves])
        w = np.array([c.w for c in curves])
        b = pd.cut(m, [16, 17, 18, 18.5, 19, 19.5, 20, 21])
        return {"magerr_det_mediana": float(np.nanmedian(e)),
                "magerr_por_mag": {str(k): float(v) for k, v in pd.Series(e).groupby(b, observed=True).median().items()},
                "n_det_mediana": float(np.median(nd)), "n_det_mediana_w": _wmedian(nd, w),
                "m_ref_mediana": float(np.median(mr)), "m_ref_mediana_w": _wmedian(mr, w),
                "frac_ul": float(np.mean(np.concatenate([c.ul for c in curves])))}
    return {"sims_val": med(sims), "real_val": med(real)}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "ytick.minor.visible": True, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight"})
    import matplotlib.pyplot as plt
    return plt


def _rep_subset(agg):
    return "val_rep" if "subset" in agg and (agg.subset == "val_rep").any() else "val"


def _sub(agg, ds_, mode, bands, subset):
    a = agg[(agg.dataset == ds_) & (agg["mode"] == mode) & (agg.bands == bands)]
    return a[a.subset == subset] if "subset" in a else a


def plot_degradation(agg, path, plt):
    """Dos paneles (r, g+r): ZTF val_rep natural y muestra fija (>= 7 detecciones), ZTF val completo (referencia) y
    sinteticas."""
    rs = _rep_subset(agg)
    tag = rs.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    xs = ["3", "5", "7", "all"]
    lines = [("real", "natural", rs, "k", "-", "o", 1.0, f"ZTF {tag}"),
             ("real", "fixed", rs, "tab:red", "-.", "s", 1.0, rf"ZTF {tag}, fixed ($\geq 7$ det.)")]
    if rs != "val":
        lines.append(("real", "natural", "val", "0.3", ":", ".", 0.7, "ZTF val (all, reference)"))
    lines.append(("sims", "natural", "sims", "0.55", "--", "^", 1.0, "Synthetic"))
    for ax, bands in zip(axes, ("r", "g+r")):
        for ds_, mode, sub, color, ls, mk, lw, lab in lines:
            a = _sub(agg, ds_, mode, bands, sub).set_index("N").reindex(xs)
            if a.bal_acc_mean.notna().any():
                ax.errorbar(range(4), a.bal_acc_mean, yerr=a.bal_acc_std.fillna(0), color=color, ls=ls, marker=mk,
                            ms=4, lw=lw, capsize=2, label=lab)
        ax.set_xticks(range(4), ["3", "5", "7", "All"])
        ax.set_xlabel(f"Number of detections ({bands})")
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(frameon=False, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def plot_horizon(agg, path, plt):
    """Dos paneles (r, g+r): exactitud balanceada por horizonte desde la primera deteccion (10, 20, 50 d y la curva
    completa), sobre la muestra fija del modo horizon. ZTF val_rep y val completo (referencia)."""
    rs = _rep_subset(agg)
    xs = [horizon_label(h) for h in HORIZONS + (None,)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    lines = [(rs, "k", "-", "o", 1.0, f"ZTF {rs.replace('_', ' ')}")]
    if rs != "val":
        lines.append(("val", "0.3", ":", ".", 0.7, "ZTF val (all, reference)"))
    for ax, bands in zip(axes, ("r", "g+r")):
        for sub, color, ls, mk, lw, lab in lines:
            a = _sub(agg, "real", "horizon", bands, sub).set_index("N").reindex(xs)
            if a.bal_acc_mean.notna().any():
                n = a.n_eval.dropna()
                ax.plot(range(len(xs)), a.bal_acc_mean, color=color, ls=ls, marker=mk, ms=4, lw=lw,
                        label=lab + (f" (n = {int(n.iloc[0])})" if len(n) else ""))
        ax.set_xticks(range(len(xs)), [f"{h}" for h in HORIZONS] + ["All"])
        ax.set_xlabel(f"Days since first detection ({bands})")
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(frameon=False, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def plot_confusion(cm, classes, path, plt, title=None):
    cm = np.asarray(cm, float)
    frac = cm / np.clip(cm.sum(1, keepdims=True), 1e-12, None)
    fig, ax = plt.subplots(figsize=(3.46, 3.0))
    ax.imshow(frac, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{frac[i, j]:.2f}\n({cm[i, j]:.0f})", ha="center", va="center", fontsize=7,
                    color="w" if frac[i, j] > 0.5 else "k")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.minorticks_off()
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    if title:
        ax.set_title(title, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- red
def villar_oids_or_none(real_dir, four_classes, real_feat=None):
    """Oids val que Villar puede clasificar (tienen features reales). None si no esta el csv. real_feat = None usa el
    de baseline.REAL_FEAT (revision H8: quien compare con un Villar corrido con otro --real-feat tiene que pasarlo; el
    csv usado queda en metrics.json como villar_oids_feat)."""
    from pipeline78.nnclf.baseline import REAL_FEAT, villar_covered_oids
    f = Path(real_feat or REAL_FEAT)
    return villar_covered_oids(f, real_dir, four_classes) if f.exists() else None


def run_eval(out_dir, n_draws=5, device=None, threads=None):
    out = Path(out_dir)
    model, cfg, ck = load_model(out, pick_device(device or cfg_device(out)))
    if threads:
        torch.set_num_threads(threads)
    dev = next(model.parameters()).device
    classes = tuple(ck["classes"])
    real, skipped = D.load_real_val(cfg.real_dir, cfg.four_classes)
    split = json.loads((out / "split.json").read_text())
    sims = D.load_sims(cfg.sim_run, cfg.four_classes, sim_ids=[int(k) for k in split["val_keys"]])
    prob = nn_prob_fn(model, cfg, dev)
    tab = pd.concat([predict_table(prob, real, classes, "real", n_draws, cfg.seed, fixed=True),
                     predict_table(prob, sims, classes, "sims", n_draws, cfg.seed, fixed=False)], ignore_index=True)
    subsets = val_subsets(cfg.real_dir, cfg.four_classes)
    res, agg = summarize(tab, classes, len(real) + len(skipped), villar_oids_or_none(cfg.real_dir, cfg.four_classes),
                         subsets)
    extra = {"method": "nn", "model": cfg.model, "use_z": cfg.use_z, "band_enc": cfg.band_enc,
             "time_enc": cfg.time_enc, "gru_pool": cfg.gru_pool, "bidir": cfg.bidir, "trunc": cfg.trunc,
             "p_trunc": cfg.p_trunc, "jerarquica": cfg.jerarquica, "best_epoch": ck.get("best_epoch"),
             "real_sin_3_det_gr": skipped,
             "villar_oids_feat": _real_feat_path(), "domain": domain_check(sims, real)}
    res = write_outputs(out, tab, res, agg, classes, extra, subsets)
    print_summary(out.name, res, agg)
    return res


def _real_feat_path():
    from pipeline78.nnclf.baseline import REAL_FEAT
    return str(REAL_FEAT)


def cfg_device(out):
    return json.loads((Path(out) / "config.json").read_text()).get("device", "auto")
