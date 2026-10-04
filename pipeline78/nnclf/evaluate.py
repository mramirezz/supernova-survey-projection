"""Evaluacion: reales de validacion (mitad val, nunca la final) y sims de validacion interna, con curva de degradacion.

CELDAS: bandas r o g+r, por N = 3, 5, 7 detecciones al azar o todas (data.degrade). Con N hay n_draws sorteos por
celda (rng = seed + 100 draw + N). 'Todas' es un solo pase. Metricas: exactitud, exactitud balanceada, F1 por clase,
F1 macro y matriz de confusion (filas = clase verdadera). Las reales van sin peso. Las sims de validacion van
ponderadas por w_z (poblacion volumetrica), sin balance de clases.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from pipeline78.nnclf import data as D
from pipeline78.nnclf.train import load_model, logits_of, pick_device

CELLS = [(b, n) for b in (("r",), ("g", "r")) for n in (3, 5, 7, None)]
COLORS = {"Ia": "tab:blue", "II": "tab:green", "Ibc": "tab:red", "IIn": "tab:purple"}


def metrics(y, yhat, classes, w=None):
    lab = list(range(len(classes)))
    f1 = f1_score(y, yhat, labels=lab, average=None, sample_weight=w, zero_division=0)
    return {"n": int(len(y)), "acc": float(accuracy_score(y, yhat, sample_weight=w)),
            "bal_acc": float(balanced_accuracy_score(y, yhat, sample_weight=w)),
            "f1_macro": float(np.mean(f1)), **{f"f1_{c}": float(v) for c, v in zip(classes, f1)},
            "confusion": confusion_matrix(y, yhat, labels=lab, sample_weight=w).round(3).tolist()}


def eval_cells(model, curves, cfg, device, classes, n_draws=5, weighted=False):
    """Devuelve (filas por celda y sorteo, predicciones de la celda 'todas g+r')."""
    rows, preds = [], None
    for bands, n in CELLS:
        for d in range(1 if n is None else n_draws):
            rng = np.random.default_rng(cfg.seed + 100 * d + (n or 0))
            sel = [(c, dc) for c in curves if (dc := D.degrade(c, n, bands, rng)) is not None]
            row = {"bands": "+".join(bands), "N": "all" if n is None else str(n), "draw": d,
                   "n_eval": len(sel), "n_skip": len(curves) - len(sel)}
            if sel:
                enc = [D.tokenize(dc, cfg.max_len, cfg.use_magerr, cfg.use_z) for _, dc in sel]
                p = torch.softmax(logits_of(model, enc, device), 1).numpy()
                y = np.array([c.y for c, _ in sel])
                w = np.array([c.w for c, _ in sel]) if weighted else None
                row.update(metrics(y, p.argmax(1), classes, w))
                if n is None and bands == ("g", "r"):
                    preds = pd.DataFrame({"oid": [c.key for c, _ in sel], "y_true": [classes[i] for i in y],
                                          "y_pred": [classes[i] for i in p.argmax(1)],
                                          "n_det": [dc.n_det() for _, dc in sel],
                                          **{f"p_{c}": p[:, i] for i, c in enumerate(classes)}})
            rows.append(row)
    return rows, preds


def aggregate(rows, classes):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion"} for r in rows])
    cols = [c for c in ["acc", "bal_acc", "f1_macro"] + [f"f1_{k}" for k in classes] if c in df]
    g = df.groupby(["bands", "N"], sort=False)
    agg = g[["n_eval", "n_skip"]].first().join(g[cols].mean().add_suffix("_mean")).join(g[cols].std().add_suffix("_std"))
    return agg.join(g.size().rename("n_draws")).reset_index()


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


def plot_degradation(agg, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.46, 2.6))
    xs = ["3", "5", "7", "all"]
    for ds_, color, ls in (("real", "k", "-"), ("sims", "0.55", "--")):
        for bands, mk in (("r", "o"), ("g+r", "s")):
            a = agg[(agg.dataset == ds_) & (agg.bands == bands)].set_index("N").reindex(xs)
            ax.errorbar(range(4), a.bal_acc_mean, yerr=a.bal_acc_std.fillna(0), color=color, ls=ls, marker=mk,
                        ms=4, lw=1, capsize=2, label=f"{'ZTF' if ds_ == 'real' else 'Synthetic'} {bands}")
    ax.set_xticks(range(4), ["3", "5", "7", "All"])
    ax.set_xlabel("Number of detections")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(frameon=False, ncol=2, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def plot_confusion(cm, classes, path):
    plt = _style()
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

    rows_r, preds = eval_cells(model, real, cfg, dev, classes, n_draws, weighted=False)
    rows_s, _ = eval_cells(model, sims, cfg, dev, classes, n_draws, weighted=True)
    agg = pd.concat([aggregate(rows_r, classes).assign(dataset="real"),
                     aggregate(rows_s, classes).assign(dataset="sims")], ignore_index=True)
    agg.to_csv(out / "degradation.csv", index=False)
    main = next(r for r in rows_r if r["bands"] == "g+r" and r["N"] == "all")
    pd.DataFrame(main["confusion"], index=classes, columns=classes).to_csv(out / "confusion_real_all_gr.csv")
    preds.to_csv(out / "pred_real_val.csv", index=False)
    res = {"model": cfg.model, "use_z": cfg.use_z, "classes": classes, "best_epoch": ck.get("best_epoch"),
           "n_real_val": len(real) + len(skipped), "real_sin_3_det_gr": skipped, "main_real_all_gr": main,
           "domain": domain_check(sims, real), "cells_real": rows_r, "cells_sims": rows_s}
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    plot_degradation(agg, out / "fig_degradation")
    plot_confusion(main["confusion"], classes, out / "fig_confusion_real")
    cols = ["dataset", "bands", "N", "n_eval", "acc_mean", "bal_acc_mean", "bal_acc_std", "f1_macro_mean"]
    print(agg[cols].round(3).to_string(index=False), flush=True)
    print(f"[nnclf] real todas g+r: acc {main['acc']:.3f} bal {main['bal_acc']:.3f} f1 {main['f1_macro']:.3f} "
          f"(n = {main['n']}, {len(skipped)} sin 3 det) -> {out}", flush=True)
    return res


def cfg_device(out):
    return json.loads((Path(out) / "config.json").read_text()).get("device", "auto")
