"""Baseline externo SuperNNova (Moller & de Boissiere 2020, 2020MNRAS.491.4277M; github.com/supernnova/SuperNNova,
MIT, version 3.0.51 de PyPI) con las mismas sims, el mismo split por plantilla y las mismas reales que la red.

ENTORNO: SuperNNova vive en ~/venvs/snn (venv con --system-site-packages sobre el env series, mas supernnova sin
dependencias y pandas<3, astropy, h5py, natsort, colorama, tabulate, click, seaborn, pyyaml). Este modulo corre en
series y llama a snn_runner.py con el python del venv (SNN_PY). Los envs series y projection no se tocan.

CONVERSION (formato csv de SuperNNova: *_PHOT.csv con SNID, MJD, FLUXCAL, FLUXCALERR, FLT y *_HEAD.csv con SNID,
SNTYPE, PEAKMJD, HOSTGAL_SPECZ y su error):
- Entran las MISMAS observaciones que los tokens de la red (data.token_idx: detecciones y UL en la ventana
  [t_primera - 60 d, t_ultima], largo maximo 128).
- Deteccion: FLUXCAL = 10^(-0.4 (m - 27.5)) y FLUXCALERR = 0.4 ln(10) FLUXCAL sigma_m (punto cero SNANA 27.5).
- UL (adaptacion propia, SuperNNova espera fotometria forzada): FLUXCAL = 0 y FLUXCALERR = F(m_lim) / 5, porque la
  magnitud limite de ZTF (y la de las sims, que es el maglimit del obslog) es a 5 sigma.
- PEAKMJD = MJD de la deteccion mas brillante (solo la usan los diagnosticos de SuperNNova). HOSTGAL_SPECZ = z (de
  la sim o de meta_real_ztf), con error constante 0.001. SuperNNova no normaliza las features de z.
- Clases: SNTYPE = indice de clase y --sntypes {"0": "Ia", "1": "II", "2": "Ibc"(, "3": "IIn")}, con Ia como clase 0.

SPLIT: el de la red (data.split_templates, mismos n_folds, fold y seed). Train = plantillas de train y val = plantillas
de validacion interna. Por defecto (balance "subsample") el train se baja por clase al tamano de la clase menor, que es
lo que hace SuperNNova por defecto, pero el sorteo va pesado por w_z (sin reemplazo), para que la poblacion en z sea
la misma que ve la red con sus pesos. SuperNNova elige el modelo por la perdida de validacion.

DOS DATABASES: train (sims) y test (reales val en todas las celdas de evaluate, natural y fija, mas la celda principal
de las sims de validacion para la brecha sim -> real). La prediccion usa la normalizacion guardada con el modelo
(data_norm.json), asi que las reales no entran a la normalizacion. Los databases se cachean por parametros y por el
md5 del codigo que arma las celdas y la conversion (snn.py, data.py y evaluate.py; revision H9): si cambia la
conversion o las celdas, se arma un database nuevo.

MODELO: los defaults de SuperNNova 3.0.51 (LSTM bidireccional, 2 x 32, salida "mean", dropout 0.05, batch 128,
lr 1e-3, random_length). --snn-model variational es la variante bayesiana (MC dropout) del paper.
"""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.nnclf import data as D
from pipeline78.nnclf.evaluate import (enumerate_cells, summarize, write_outputs, print_summary, villar_oids_or_none,
                                       val_subsets)

SNN_PY = Path(os.environ.get("SNN_PY", Path.home() / "venvs/snn/bin/python"))
RUNNER = Path(__file__).with_name("snn_runner.py")
ZP = 27.5


def lc_arrays(c, snid, max_len=D.MAX_LEN):
    idx = D.token_idx(c, max_len)
    m, e, ul = c.mag[idx].astype(np.float64), np.nan_to_num(c.err[idx].astype(np.float64)), c.ul[idx]
    f = 10.0 ** (-0.4 * (m - ZP))
    return (np.full(len(idx), snid, dtype=object), c.t[idx], np.where(ul, 0.0, f),
            np.where(ul, f / 5.0, 0.4 * np.log(10.0) * f * e), np.where(c.band[idx] == 0, "g", "r"))


def head_row(c, snid):
    det = ~c.ul
    z = float(c.z) if np.isfinite(c.z) and c.z > 0 else -9.0
    return {"SNID": snid, "SNTYPE": int(c.y), "PEAKMJD": float(c.t[det][np.argmin(c.mag[det])]),
            "HOSTGAL_SPECZ": z, "HOSTGAL_SPECZ_ERR": 0.001, "HOSTGAL_PHOTOZ": z, "HOSTGAL_PHOTOZ_ERR": 0.001,
            "SIM_REDSHIFT_CMB": z}


def write_raw(raw_dir, items, prefix, max_len=D.MAX_LEN):
    """items: lista de (snid, Curve). Un solo archivo PHOT y uno HEAD (SuperNNova paraleliza por archivo)."""
    raw_dir = Path(raw_dir)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)
    cols = list(zip(*[lc_arrays(c, s, max_len) for s, c in items]))
    phot = pd.DataFrame({k: np.concatenate(v) for k, v in zip(["SNID", "MJD", "FLUXCAL", "FLUXCALERR", "FLT"], cols)})
    phot.to_csv(raw_dir / f"{prefix}_PHOT.csv", index=False)
    pd.DataFrame([head_row(c, s) for s, c in items]).to_csv(raw_dir / f"{prefix}_HEAD.csv", index=False)
    return len(phot)


def sntypes_json(classes):
    return json.dumps({str(i): c for i, c in enumerate(classes)})


def _run(args, log, threads):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "SNN_THREADS": str(threads), "PYTORCH_ENABLE_MPS_FALLBACK": "0", "MPLBACKEND": "Agg"}
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(map(str, args))}\n")
        fh.flush()
        r = subprocess.run([str(a) for a in args], stdout=fh, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        raise RuntimeError(f"SuperNNova fallo ({r.returncode}), ver {log}")
    return time.time() - t0


def _common(dump, raw, fits, classes):
    return ["--dump_dir", dump, "--raw_dir", raw, "--fits_dir", fits, "--sntypes", sntypes_json(classes),
            "--list_filters", "g", "r"]


def balanced_train(tr, n_cls, seed, balance):
    if balance == "none":
        return tr
    rng = np.random.default_rng(seed)
    by = [[c for c in tr if c.y == k] for k in range(n_cls)]
    n = min(len(b) for b in by)
    keep = []
    for cs in by:
        w = np.array([c.w for c in cs], float) + 1e-12
        keep += [cs[i] for i in rng.choice(len(cs), size=n, replace=False, p=w / w.sum())]
    return keep


def code_hash():
    """md5 del codigo que define las celdas y la conversion (revision H9)."""
    from pipeline78.nnclf import evaluate
    h = hashlib.md5()
    for f in (Path(__file__), Path(D.__file__), Path(evaluate.__file__)):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def build_dbs(name, sim_run, real_dir, four_classes, n_folds, fold, seed, max_sims, balance, n_draws, out_root,
              threads, log):
    classes = D.classes(four_classes)
    tag = hashlib.md5(json.dumps([str(sim_run), str(real_dir), four_classes, n_folds, fold, seed, max_sims, balance,
                                  n_draws, code_hash()]).encode()).hexdigest()[:10]
    db = Path(out_root) / f"_snn_db_{tag}"
    done = db / "done.json"
    curves = D.load_sims(sim_run, four_classes, max_sims=max_sims or None, seed=seed)
    pairs = D.sims_table(sim_run, four_classes)[["template", "sn_type"]].itertuples(index=False)
    val_tpl = D.split_templates(pairs, n_folds, fold, seed)
    tr = [c for c in curves if c.template not in val_tpl]
    va = [c for c in curves if c.template in val_tpl]
    real, skipped = D.load_real_val(real_dir, four_classes)
    meta_r, dcs_r = enumerate_cells(real, n_draws, seed, fixed=True)
    meta_r.insert(1, "dataset", "real")
    meta_s, dcs_s = enumerate_cells(va, 1, seed, fixed=False)
    main = ((meta_s["mode"] == "natural") & (meta_s.bands == "g+r") & (meta_s.N == "all")).to_numpy()
    meta_s = meta_s[main].reset_index(drop=True)
    dcs_s = [d for d, k in zip(dcs_s, main) if k]
    meta_s.insert(1, "dataset", "sims")
    keymap = pd.concat([meta_r, meta_s], ignore_index=True)
    keymap.insert(0, "SNID", [f"T{i}" for i in range(len(keymap))])
    info = {"tag": tag, "n_train_all": len(tr), "n_val": len(va), "n_real": len(real), "n_skipped": len(skipped),
            "n_test_rows": len(keymap)}
    if done.exists():
        km = pd.read_parquet(db / "keymap.parquet")
        cols = ["SNID", "key", "mode", "bands", "N", "draw"]
        assert len(km) == len(keymap) and (km[cols].astype(str).to_numpy() == keymap[cols].astype(str).to_numpy()
                                           ).all(), "keymap del cache distinto"
        print(f"[snn] reuso databases {db}", flush=True)
        return db, classes, km, {**json.loads(done.read_text()), **info}, len(real) + len(skipped)
    fits = db / "fits_vacio"
    fits.mkdir(parents=True, exist_ok=True)
    trb = balanced_train(tr, len(classes), seed, balance)
    keep = {c.key for c in trb}
    items = [(f"S{c.key}", c) for c in tr + va]
    split = pd.DataFrame({"SNID": [s for s, _ in items],
                          "dataset": [0 if c.key in keep else (1 if c.template in val_tpl else -1) for _, c in items]})
    split.to_csv(db / "split.csv", index=False)
    n_rows = write_raw(db / "raw_train", items, "SIMS")
    t_tr = _run([SNN_PY, RUNNER, "make_train", db / "split.csv", "--"] +
                _common(db / "train_db", db / "raw_train", fits, classes) + ["--seed", seed], log, threads)
    shutil.rmtree(db / "raw_train")
    n_test = write_raw(db / "raw_test", list(zip(keymap.SNID, dcs_r + dcs_s)), "TEST")
    t_te = _run([SNN_PY, RUNNER, "make_test", "--"] + _common(db / "test_db", db / "raw_test", fits, classes) +
                ["--data_testing", "--seed", seed], log, threads)
    shutil.rmtree(db / "raw_test")
    keymap.to_parquet(db / "keymap.parquet", index=False)
    meta = {"n_train_balanced": len(trb), "split_counts": split.dataset.value_counts().to_dict(),
            "n_phot_train": n_rows, "n_phot_test": n_test, "make_train_s": t_tr, "make_test_s": t_te}
    done.write_text(json.dumps(meta, indent=1, default=int))
    return db, classes, keymap, {**meta, **info}, len(real) + len(skipped)


def run_snn(name, use_z=False, four_classes=False, n_folds=5, fold=0, seed=D.SEED, max_sims=0, nb_epoch=90,
            snn_model="vanilla", balance="subsample", n_draws=5, threads=2, sim_run=D.SIM_RUN, real_dir=D.REAL_DIR,
            out_root=D.OUT_ROOT, num_inference_samples=None):
    assert SNN_PY.exists(), f"no esta el venv de SuperNNova: {SNN_PY}"
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    log = out / "snn.log"
    t0 = time.time()
    db, classes, keymap, info, n_real_total = build_dbs(name, sim_run, real_dir, four_classes, n_folds, fold, seed,
                                                        max_sims, balance, n_draws, out_root, threads, log)
    fits = db / "fits_vacio"
    mj = out / "model.json"
    t_train = _run([SNN_PY, RUNNER, "train", mj, "--"] + _common(db / "train_db", db / "raw_train", fits, classes) +
                   ["--nb_classes", len(classes), "--redshift", "zspe" if use_z else "none", "--nb_epoch", nb_epoch,
                    "--model", snn_model, "--seed", seed], log, threads)
    model_file = json.loads(mj.read_text())["model_file"]
    pred_csv = out / "snn_pred.csv"
    extra = ["--num_inference_samples", num_inference_samples] if num_inference_samples else []
    t_pred = _run([SNN_PY, RUNNER, "predict", pred_csv, model_file, "--"] +
                  _common(db / "test_db", db / "raw_test", fits, classes) +
                  ["--nb_classes", len(classes), "--model_files", model_file] + extra, log, threads)
    pr = pd.read_csv(pred_csv, dtype={"SNID": str})
    tab = keymap.merge(pr, on="SNID", how="inner")
    assert len(tab) == len(keymap), f"SuperNNova predijo {len(tab)} de {len(keymap)} curvas"
    for i, c in enumerate(classes):
        tab[f"p_{c}"] = tab.pop(f"all_class{i}").astype(np.float32)
    tab = tab.drop(columns=["SNID"])
    subsets = val_subsets(real_dir, four_classes)
    res, agg = summarize(tab, classes, n_real_total, villar_oids_or_none(real_dir, four_classes), subsets)
    extra = {"method": "supernnova", "snn_model": snn_model, "use_z": use_z, "balance": balance, "nb_epoch": nb_epoch,
             "model_file": model_file, "db": str(db), "info": info,
             "times_s": {"train": t_train, "predict": t_pred, "total": time.time() - t0}}
    res = write_outputs(out, tab, res, agg, classes, extra, subsets)
    (out / "config.json").write_text(json.dumps({"method": "supernnova", "use_z": use_z, "snn_model": snn_model,
                                                 "four_classes": four_classes, "fold": fold, "seed": seed,
                                                 "nb_epoch": nb_epoch, "balance": balance, "sim_run": str(sim_run),
                                                 "real_dir": str(real_dir), "max_sims": max_sims}, indent=1))
    print_summary(name, res, agg)
    return res
