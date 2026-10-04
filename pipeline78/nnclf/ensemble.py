"""Ensemble de los modelos de los 5 folds por plantilla (R5 de la revision de literatura; Moller et al. 2022,
2022MNRAS.514.5159M, usa ensembles de SuperNNova sobre DES).

- Reales: la probabilidad del ensemble es la MEDIA de las probabilidades de los modelos, celda por celda (mismos
  sorteos por curva en todos, data.curve_rng). El desvio entre modelos va a pred_real_val.csv (std_<clase>) como
  incertidumbre por objeto y marca de fuera de distribucion.
- Sims: prediccion fuera de fold. Cada sim de validacion la predice solo el modelo que no vio su plantilla, asi la
  brecha sim -> real del ensemble cubre las plantillas de los 5 folds.
- Todas las corridas tienen que compartir clases, use_z, band_enc y seed (assert).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pipeline78.nnclf import data as D
from pipeline78.nnclf.evaluate import (enumerate_cells, nn_prob_fn, summarize, write_outputs, print_summary,
                                       villar_oids_or_none)
from pipeline78.nnclf.train import load_model


def run_ensemble(names, out_name, out_root=D.OUT_ROOT, n_draws=5, threads=2):
    torch.set_num_threads(threads)
    runs = [Path(out_root) / n for n in names]
    loaded = [load_model(r, "cpu") for r in runs]
    cfgs = [c for _, c, _ in loaded]
    classes = tuple(loaded[0][2]["classes"])
    for (_, c, ck), r in zip(loaded, runs):
        assert tuple(ck["classes"]) == classes, f"{r.name}: clases distintas"
        for k in ("use_z", "band_enc", "seed", "four_classes", "sim_run", "real_dir"):
            assert getattr(c, k) == getattr(cfgs[0], k), f"{r.name}: {k} distinto"
    cfg = cfgs[0]
    real, skipped = D.load_real_val(cfg.real_dir, cfg.four_classes)
    meta, dcs = enumerate_cells(real, n_draws, cfg.seed, fixed=True)
    P = np.stack([nn_prob_fn(m, c, "cpu")(dcs) for m, c, _ in loaded])          # [n_mod, n, K]
    meta.insert(1, "dataset", "real")
    for i, c in enumerate(classes):
        meta[f"p_{c}"] = P[:, :, i].mean(0).astype(np.float32)
        meta[f"std_{c}"] = P[:, :, i].std(0).astype(np.float32)
    parts = [meta]
    for (m, c, _), r in zip(loaded, runs):                                       # sims fuera de fold
        keys = json.loads((r / "split.json").read_text())["val_keys"]
        sims = D.load_sims(c.sim_run, c.four_classes, sim_ids=[int(k) for k in keys])
        ms, ds_ = enumerate_cells(sims, n_draws, c.seed, fixed=False)
        p = nn_prob_fn(m, c, "cpu")(ds_)
        ms.insert(1, "dataset", "sims")
        for i, k in enumerate(classes):
            ms[f"p_{k}"] = p[:, i].astype(np.float32)
        ms["fold_run"] = r.name
        parts.append(ms)
    tab = pd.concat(parts, ignore_index=True)
    res, agg = summarize(tab, classes, len(real) + len(skipped), villar_oids_or_none(cfg.real_dir, cfg.four_classes))
    out = Path(out_root) / out_name
    res = write_outputs(out, tab, res, agg, classes, {"method": "nn_ensemble", "members": list(names),
                                                       "use_z": cfg.use_z, "model": cfg.model})
    (out / "config.json").write_text(json.dumps({"ensemble_of": list(names), "use_z": cfg.use_z}, indent=1))
    print_summary(out_name, res, agg)
    return res
