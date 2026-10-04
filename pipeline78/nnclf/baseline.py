"""Baseline Villar+MCMC sobre las mismas reales de validacion, para comparar con la red manzana con manzana.

REGLAS
1. Features: los parametros SPM por banda (r, g) del extractor oficial, la misma lista PARAMS de
   feature_extraction/opt_clasificador/00_prep_data.py, mas m_peak = -2.5 log10(A), color g-r, razones g/r de f,
   t_fall, gamma, t_rise y la razon de duraciones. Con use_z se agregan M_peak = m_peak - mu(z) y z. Las dos bandas
   se juntan con outer join: la SN con una sola banda ajustada queda con NaN en la otra (HistGradientBoosting los
   acepta). En 00_prep_data.py el join partia de r.
2. Train: sims de la T9 final con features (RUNS/features_ztf_v78_t9_final), sin las plantillas de validacion
   interna del MISMO split que la red (data.split_templates, mismos n_folds, fold y seed). Pesos w_z por balance
   de clases (data.balance_weights, misma regla que la red).
3. Validacion: features de las reales (RUNS/features_real_ztf), solo las oids de la mitad val. csv.reader recorre
   todas las lineas pero solo guarda las de oids val: las filas de la mitad final no llegan a pandas ni se guardan.
4. Cobertura: la real de validacion sin features (Villar no ajusto) queda sin clasificar. Se reporta la cobertura y
   las metricas sobre las cubiertas. Con nn_run se calculan tambien las metricas de la red sobre las MISMAS oids, y
   la corrida NN tiene que tener las mismas clases y el mismo use_z (assert, revision B3).
5. Salida comun (nn-lit-brief): preds.parquet con la celda principal (natural, g+r, todas) de las reales cubiertas y
   las sims de validacion, resumida con evaluate.summarize por subconjunto (val_rep, val_sel y val, de
   pipeline78.splits). Villar no pasa por la degradacion (revision M2): con menos de 7 detecciones no ajusta, asi que
   alli la comparacion es de cobertura.
"""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from pipeline78.nnclf import data as D
from pipeline78.nnclf.evaluate import metrics, summarize, write_outputs, print_summary, val_subsets
from pipeline78.paths import RUNS

SIM_FEAT = RUNS / "features_ztf_v78_t9_final/features/features.csv"
REAL_FEAT = RUNS / "features_real_ztf/features/features.csv"
PARAMS = ['A', 'f', 't0', 't_rise', 't_fall', 'gamma', 'A_err', 'f_err', 't_rise_err',
          't_fall_err', 'gamma_err', 'rms', 'mad', 'n_points', 'time_span']


def widen(df, keys):
    out = None
    for band in ("r", "g"):
        sub = df[df.filter_band.astype(str).str.lower().str.endswith(band)]
        cols = [c for c in PARAMS if c in sub.columns]
        sub = sub[keys + cols].drop_duplicates(keys, keep="last").rename(columns={c: f"{c}_{band}" for c in cols})
        out = sub if out is None else pd.merge(out, sub, on=keys, how="outer")
    return out


def add_combined(df, use_z):
    for band in ("r", "g"):
        A = df[f"A_{band}"].to_numpy(float)
        df[f"m_peak_{band}"] = np.where(A > 0, -2.5 * np.log10(np.where(A > 0, A, 1.0)), np.nan)
    df["color_gr"] = df.m_peak_g - df.m_peak_r
    for p in ("f", "t_fall", "gamma", "t_rise"):
        df[f"{p}_ratio_gr"] = df[f"{p}_g"] / df[f"{p}_r"]
    df["dur_ratio"] = df.time_span_g / df.time_span_r
    if use_z:
        mu = D.distmod(df.z.to_numpy(float))
        for band in ("r", "g"):
            df[f"M_peak_{band}"] = df[f"m_peak_{band}"] - mu
    return df.replace([np.inf, -np.inf], np.nan)


def feature_cols(use_z):
    cols = [f"{p}_{b}" for b in ("r", "g") for p in PARAMS] + ["m_peak_r", "m_peak_g", "color_gr", "dur_ratio"]
    cols += [f"{p}_ratio_gr" for p in ("f", "t_fall", "gamma", "t_rise")]
    return cols + (["M_peak_r", "M_peak_g", "z"] if use_z else [])


def load_sim_features(path=SIM_FEAT, run_dir=D.SIM_RUN, four_classes=False, use_z=False):
    f = pd.read_csv(path)
    W = widen(f, ["oid", "sn_type", "part_index"])
    s = D.sims_table(run_dir, four_classes).rename(columns={"field": "oid"})
    assert not s.duplicated(["oid", "sn_type", "part_index"]).any()
    W = W.merge(s[["oid", "sn_type", "part_index", "template", "z", "w_z", "cls"]], how="inner",
                on=["oid", "sn_type", "part_index"])
    return add_combined(W, use_z)


def read_rows_for_oids(path, oids):
    """Lee solo las filas del csv cuya columna oid esta en oids (filtro antes de pandas)."""
    oids = set(oids)
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        k = head.index("oid")
        rows = [row for row in r if row[k] in oids]
    df = pd.DataFrame(rows, columns=head)
    for c in df.columns:
        num = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
        if num.notna().sum() == (df[c] != "").sum():
            df[c] = num
    return df


def load_real_features(path=REAL_FEAT, real_dir=D.REAL_DIR, four_classes=False, use_z=False):
    v = D.real_val_meta(real_dir, four_classes)
    f = read_rows_for_oids(path, v.oid)
    assert set(f.oid) <= set(v.oid)
    W = widen(f.drop(columns=["sn_type"]), ["oid"])
    W = W.merge(v[["oid", "z", "cls"]], on="oid", how="inner")
    return add_combined(W, use_z), v


def villar_covered_oids(path=REAL_FEAT, real_dir=D.REAL_DIR, four_classes=False):
    """Oids de la mitad val con features Villar en alguna banda (las que el baseline puede clasificar)."""
    v = D.real_val_meta(real_dir, four_classes)
    f = read_rows_for_oids(path, v.oid)
    return sorted(set(f.oid.astype(str)))


def _check_nn_run(out_root, nn_run, cls, use_z):
    cfg = json.loads((Path(out_root) / nn_run / "config.json").read_text())
    m = json.loads((Path(out_root) / nn_run / "metrics.json").read_text())
    assert tuple(m["classes"]) == tuple(cls), f"{nn_run}: clases {m['classes']} != {cls}"
    assert bool(cfg.get("use_z", False)) == bool(use_z), f"{nn_run}: use_z {cfg.get('use_z')} != {use_z}"


def run_baseline(name=None, use_z=False, four_classes=False, n_folds=5, fold=0, seed=D.SEED, nn_run=None,
                 sim_feat=SIM_FEAT, real_feat=REAL_FEAT, run_dir=D.SIM_RUN, real_dir=D.REAL_DIR, out_root=D.OUT_ROOT):
    cls = D.classes(four_classes)
    if nn_run:
        _check_nn_run(out_root, nn_run, cls, use_z)
    name = name or "baseline_villar" + ("_z" if use_z else "") + ("_4c" if four_classes else "")
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    S = load_sim_features(sim_feat, run_dir, four_classes, use_z)
    pairs = D.sims_table(run_dir, four_classes)[["template", "sn_type"]].itertuples(index=False)
    val_tpl = D.split_templates(pairs, n_folds, fold, seed)
    S["y"] = S.cls.map({c: i for i, c in enumerate(cls)})
    tr, va = S[~S.template.isin(val_tpl)], S[S.template.isin(val_tpl)]
    cols = feature_cols(use_z)
    sw = D.balance_weights(tr.y.to_numpy(), tr.w_z.to_numpy(), len(cls))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                         random_state=seed % 2**32)
    clf.fit(tr[cols].to_numpy(float), tr.y.to_numpy(), sample_weight=sw)

    R, v = load_real_features(real_feat, real_dir, four_classes, use_z)
    R["y"] = R.cls.map({c: i for i, c in enumerate(cls)})
    p = clf.predict_proba(R[cols].to_numpy(float)) if len(R) else np.zeros((0, len(cls)))
    R["y_pred"] = p.argmax(1) if len(R) else []
    res = {"name": name, "use_z": use_z, "classes": cls, "n_sims_train": int(len(tr)), "n_sims_val": int(len(va)),
           "n_real_val": int(len(v)), "n_real_con_features": int(len(R)),
           "cobertura": float(len(R) / max(len(v), 1)),
           "cobertura_por_clase": {c: float((R.cls == c).sum() / max((v.cls == c).sum(), 1)) for c in cls},
           "real": metrics(R.y.to_numpy(), R.y_pred.to_numpy(), cls) if len(R) else None}
    if len(va):
        res["sims_val"] = metrics(va.y.to_numpy(), clf.predict(va[cols].to_numpy(float)), cls, va.w_z.to_numpy())
    pd.DataFrame({"oid": R.oid, "y_true": [cls[i] for i in R.y], "y_pred": [cls[i] for i in R.y_pred],
                  **{f"p_{c}": p[:, i] for i, c in enumerate(cls)}}).to_csv(out / "pred_real_val.csv", index=False)
    # salida comun: celda principal de las reales cubiertas y sims de validacion (sin degradacion)
    pv = clf.predict_proba(va[cols].to_numpy(float)) if len(va) else np.zeros((0, len(cls)))
    parts = []
    for ds_, keys, y, w, pp in (("real", R.oid.astype(str), R.y, np.ones(len(R)), p),
                                ("sims", va.oid.astype(str) + "|" + va.sn_type + "|" + va.part_index.astype(str),
                                 va.y, va.w_z, pv)):
        t = pd.DataFrame({"key": keys.to_numpy(), "dataset": ds_, "mode": "natural", "bands": "g+r", "N": "all",
                          "draw": 0, "y": np.asarray(y, int), "w": np.asarray(w, float), "n_det": -1})
        for i, c in enumerate(cls):
            t[f"p_{c}"] = pp[:, i].astype(np.float32) if len(t) else []
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    subsets = val_subsets(real_dir, four_classes)
    sres, agg = summarize(tab, cls, len(v), villar_oids=list(R.oid.astype(str)), subsets=subsets)
    write_outputs(out / "comun", tab, sres, agg, cls, {"method": "villar", "use_z": use_z, "sim_feat": str(sim_feat),
                                                       "real_feat": str(real_feat)}, subsets)
    res["comun"] = {k: sres[k] for k in ("coverage", "n_main", "calibration", "sims_main_all_gr") if k in sres}
    if nn_run:
        nn = pd.read_csv(Path(out_root) / nn_run / "pred_real_val.csv")
        both = nn.merge(R[["oid"]], on="oid", how="inner")
        ix = {c: i for i, c in enumerate(cls)}
        bl = R.set_index("oid").loc[both.oid]
        res["comparacion"] = {"nn_run": nn_run, "n_comun": int(len(both)),
                              "nn": metrics(both.y_true.map(ix).to_numpy(), both.y_pred.map(ix).to_numpy(), cls),
                              "villar": metrics(bl.y.to_numpy(), bl.y_pred.to_numpy(), cls),
                              "nn_todas_sus_reales": metrics(nn.y_true.map(ix).to_numpy(),
                                                             nn.y_pred.map(ix).to_numpy(), cls)}
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    print_summary(name, sres, agg)
    r = res["real"]
    print(f"[baseline] {name}: train {len(tr)} sims | reales con features {len(R)}/{len(v)} "
          f"(cobertura {res['cobertura']:.2f})" + (f" | acc {r['acc']:.3f} bal {r['bal_acc']:.3f} "
                                                  f"f1 {r['f1_macro']:.3f}" if r else "") + f" -> {out}", flush=True)
    if nn_run:
        c = res["comparacion"]
        print(f"[baseline] mismas {c['n_comun']} oids: NN bal {c['nn']['bal_acc']:.3f} | "
              f"Villar bal {c['villar']['bal_acc']:.3f}", flush=True)
    return res
