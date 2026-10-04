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
Las reales van sin peso. Las sims de validacion van ponderadas por w_z (poblacion volumetrica), sin balance de clases.

METRICAS (summarize): exactitud, exactitud balanceada, F1 por clase y macro, matriz de confusion (filas = verdadera),
cobertura (reales clasificadas / reales val de las clases), la misma celda principal restringida a las oids que
Villar puede clasificar (revision M2), calibracion (calib.py, temperatura ajustada en val_real) y la exactitud
balanceada en las sims de validacion por plantilla (brecha sim -> real).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from pipeline78.nnclf import data as D
from pipeline78.nnclf import calib
from pipeline78.nnclf.train import load_model, logits_of, pick_device

CELLS = [(b, n) for b in (("r",), ("g", "r")) for n in (3, 5, 7, None)]
FIXED_MIN = 7
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
def enumerate_cells(curves, n_draws=5, seed=D.SEED, fixed=True):
    """(meta DataFrame, curvas degradadas) de todas las celdas. Determinista por curva."""
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


def summarize(tab, classes, n_real_total, villar_oids=None, n_bins=calib.N_BINS):
    """Metricas comunes a todos los metodos a partir de la tabla de predicciones."""
    real, sims = tab[tab.dataset == "real"], tab[tab.dataset == "sims"]
    rows_r, rows_s = cell_rows(real, classes, False), cell_rows(sims, classes, True)
    m = main_cell(real)
    p, y = m[_pcols(classes)].to_numpy(np.float64), m.y.to_numpy()
    res = {"classes": list(classes), "n_real_val": int(n_real_total), "n_main": int(len(m)),
           "coverage": float(len(m) / max(n_real_total, 1)),
           "main_real_all_gr": metrics(y, p.argmax(1), classes) if len(m) else None,
           "cells_real": rows_r, "cells_sims": rows_s}
    if len(m) >= 10:
        res["calibration"] = calib.calibration_report(p, y, n_bins)
    sm = main_cell(sims)
    if len(sm):
        res["sims_main_all_gr"] = metrics(sm.y.to_numpy(), sm[_pcols(classes)].to_numpy().argmax(1), classes,
                                          sm.w.to_numpy())
    if villar_oids is not None:
        v = m[m.key.isin(set(villar_oids))]
        res["villar_oids"] = {"n_villar_oids": int(len(set(villar_oids))), "n_comun": int(len(v)),
                              "metrics": metrics(v.y.to_numpy(), v[_pcols(classes)].to_numpy().argmax(1), classes)
                              if len(v) else None}
    agg = pd.concat([aggregate(rows_r, classes).assign(dataset="real"),
                     aggregate(rows_s, classes).assign(dataset="sims")], ignore_index=True)
    return res, agg


def summary_row(name, method, res, agg, classes):
    """Una fila de resumen.csv."""
    mr = res.get("main_real_all_gr") or {}
    row = {"name": name, "method": method, "n_real_val": res["n_real_val"], "n_main": res["n_main"],
           "coverage": res["coverage"], "acc": mr.get("acc"), "bal_acc": mr.get("bal_acc"),
           "f1_macro": mr.get("f1_macro"), **{f"f1_{c}": mr.get(f"f1_{c}") for c in classes}}
    vo = (res.get("villar_oids") or {})
    row["n_villar_oids"] = vo.get("n_comun")
    row["bal_acc_villar_oids"] = (vo.get("metrics") or {}).get("bal_acc")
    row["f1_macro_villar_oids"] = (vo.get("metrics") or {}).get("f1_macro")
    cal = res.get("calibration") or {}
    row.update({"T": cal.get("T"), "ece_raw": cal.get("ece_raw"), "ece_ts_cv5": cal.get("ece_ts_cv5")})
    sm = res.get("sims_main_all_gr") or {}
    row["sims_bal_acc"] = sm.get("bal_acc")
    row["gap_sim_real"] = (sm["bal_acc"] - mr["bal_acc"]) if sm and mr else None
    if len(agg):
        fx = agg[(agg.dataset == "real") & (agg["mode"] == "fixed")]
        for _, r in fx.iterrows():
            row[f"fixed_{r.bands}_{r.N}_bal_acc"] = r.bal_acc_mean
            row[f"fixed_{r.bands}_n"] = r.n_eval
    return row


def write_outputs(out, tab, res, agg, classes, extra=None):
    """Escribe metrics.json, degradation*.csv, confusion, pred_real_val.csv y las figuras."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out / "preds.parquet", index=False)
    res = {**(extra or {}), **res}
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    if len(agg):
        agg[agg["mode"] == "natural"].to_csv(out / "degradation.csv", index=False)
        agg[agg["mode"] == "fixed"].to_csv(out / "degradation_fixed.csv", index=False)
    m = main_cell(tab[tab.dataset == "real"])
    if res.get("main_real_all_gr"):
        pd.DataFrame(res["main_real_all_gr"]["confusion"], index=classes, columns=classes).to_csv(
            out / "confusion_real_all_gr.csv")
        p = m[_pcols(classes)].to_numpy()
        pd.DataFrame({"oid": m.key.to_numpy(), "y_true": [classes[i] for i in m.y],
                      "y_pred": [classes[i] for i in p.argmax(1)], "n_det": m.n_det.to_numpy(),
                      **{c: m[c].to_numpy() for c in m.columns if c.startswith(("p_", "std_"))}}).to_csv(
            out / "pred_real_val.csv", index=False)
        plt = _style()
        if len(agg):
            plot_degradation(agg, out / "fig_degradation", plt)
        plot_confusion(res["main_real_all_gr"]["confusion"], classes, out / "fig_confusion_real", plt)
        if "calibration" in res:
            calib.plot_reliability(res["calibration"], out / "fig_reliability", plt)
    return res


def print_summary(name, res, agg):
    if len(agg):
        cols = ["dataset", "mode", "bands", "N", "n_eval", "bal_acc_mean", "bal_acc_std", "f1_macro_mean"]
        print(agg[[c for c in cols if c in agg]].round(3).to_string(index=False), flush=True)
    mr = res.get("main_real_all_gr")
    if mr:
        cal = res.get("calibration", {})
        vo = (res.get("villar_oids") or {}).get("metrics") or {}
        print(f"[nnclf] {name} real todas g+r: acc {mr['acc']:.3f} bal {mr['bal_acc']:.3f} f1 {mr['f1_macro']:.3f} "
              f"(n = {mr['n']}, cobertura {res['coverage']:.3f})"
              + (f" | oids Villar bal {vo['bal_acc']:.3f}" if vo else "")
              + (f" | ECE {cal['ece_raw']:.3f} -> {cal['ece_ts_cv5']:.3f} (T {cal['T']:.2f})" if cal else ""),
              flush=True)


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


def plot_degradation(agg, path, plt):
    """Dos paneles (r, g+r): ZTF natural, ZTF muestra fija (>= 7 detecciones) y sinteticas."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    xs = ["3", "5", "7", "all"]
    for ax, bands in zip(axes, ("r", "g+r")):
        for ds_, mode, color, ls, mk, lab in (("real", "natural", "k", "-", "o", "ZTF"),
                                              ("real", "fixed", "tab:red", "-.", "s", r"ZTF, fixed ($\geq 7$ det.)"),
                                              ("sims", "natural", "0.55", "--", "^", "Synthetic")):
            a = agg[(agg.dataset == ds_) & (agg["mode"] == mode) & (agg.bands == bands)].set_index("N").reindex(xs)
            if a.bal_acc_mean.notna().any():
                ax.errorbar(range(4), a.bal_acc_mean, yerr=a.bal_acc_std.fillna(0), color=color, ls=ls, marker=mk,
                            ms=4, lw=1, capsize=2, label=lab)
        ax.set_xticks(range(4), ["3", "5", "7", "All"])
        ax.set_xlabel(f"Number of detections ({bands})")
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(frameon=False, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def plot_confusion(cm, classes, path, plt):
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
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- red
def villar_oids_or_none(real_dir, four_classes):
    """Oids val que Villar puede clasificar (tienen features reales). None si no esta el csv."""
    from pipeline78.nnclf.baseline import REAL_FEAT, villar_covered_oids
    return villar_covered_oids(REAL_FEAT, real_dir, four_classes) if Path(REAL_FEAT).exists() else None


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
    res, agg = summarize(tab, classes, len(real) + len(skipped), villar_oids_or_none(cfg.real_dir, cfg.four_classes))
    extra = {"method": "nn", "model": cfg.model, "use_z": cfg.use_z, "band_enc": cfg.band_enc,
             "time_enc": cfg.time_enc, "gru_pool": cfg.gru_pool, "bidir": cfg.bidir, "trunc": cfg.trunc,
             "best_epoch": ck.get("best_epoch"), "real_sin_3_det_gr": skipped, "domain": domain_check(sims, real)}
    res = write_outputs(out, tab, res, agg, classes, extra)
    print_summary(out.name, res, agg)
    return res


def cfg_device(out):
    return json.loads((Path(out) / "config.json").read_text()).get("device", "auto")
