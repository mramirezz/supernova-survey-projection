"""Evaluacion UNICA de los clasificadores congelados en la mitad TEST (split == final) del holdout ZTF.

Orden de Mauricio (2026-10-06): primera y unica vez que se usa esa mitad. Los lectores de val (splits.read_val_meta,
clf_villar.load_real_val, nnclf.data.load_real_val, plantillas_clf.run) no cambian y siguen sin ver la final. Este
modulo es el UNICO camino a la final y exige el flag --mitad-final-autorizada en cada comando.

    python -m pipeline78.evaluar_test villar     --mitad-final-autorizada [--cuatro-clases]       (env projection)
    python -m pipeline78.evaluar_test red        --mitad-final-autorizada [--device mps]          (env series)
    python -m pipeline78.evaluar_test plantillas --mitad-final-autorizada [--cuatro-clases] [--workers 4] (projection)
    python -m pipeline78.evaluar_test resumen    --mitad-final-autorizada                         (cualquiera)

REGLAS
1. Guarda. Sin el flag, main sale antes de abrir cualquier archivo. read_final_meta(path) sin autorizada=True lanza
   PermissionError antes de abrir el csv (test que lo vigila).
2. Filas: origen == holdout & split == final & excluir falso, leidas con csv linea a linea (las de val no llegan a
   pandas). Una oid final que aparezca en otro split es un error. Clases como en val: Ia, II (= II + IIb), Ibc, e IIn
   con --cuatro-clases. Curvas g, r filtradas por oid dentro de pyarrow, como nnclf.data.load_real_val.
3. Nada se ajusta ni se elige con test. Cada metodo usa lo congelado en su corrida de val:
   villar      RUNS/clf_villar/sweep_t11_foco/mejor/model.joblib (3 clases) y sweep_t11_fisica_4c/mejor (4 clases):
               modelo, columnas, temperatura (de las sims) y marco. Mismo camino que clf_villar.cmd_eval: widen,
               derive, fisica (fisica_reales, cache propio en RUNS/test_final), temper. Prior none (el EM no va).
   red         RUNS/nnclf_t11/tf_base/model.pt, celda principal (natural, g+r, todas las detecciones), tokenize de
               nnclf.data, softmax cruda (lo que tiene pred_real_val.csv de val). La calibracion T, T' y los priors
               guardados de val_sel solo se aplican (ECE y NLL), no se reajustan.
   plantillas  plantillas_clf.clasificar con la config de plantillas_t11(_4c)/metrics.json y su biblioteca. Si la
               clave de la biblioteca o una constante actual no es la de la corrida de val, se reporta como problema.
4. Reproducibilidad. Antes de test cada metodo rehace val con el mismo codigo y lo compara con su pred_real_val.csv
   guardado: max |dp| y cambios de argmax. Plantillas: N_REPRO SNe fijas de val_sel. Va en metrics.json (repro).
5. Metricas por metodo (formato de val, funciones de clf_villar): exactitud, exactitud balanceada con IC 95 % bootstrap
   (1000, SEED), F1 por clase, matriz de confusion (filas = verdadera), log-loss y ECE, cobertura (clasificadas /
   SNe final de las clases, total y por clase).
6. Sistema completo (pipeline78.sistema_tasas, 3 clases): cada metodo con respaldo tf_base donde no cubre, sobre las
   oids que la red clasifica. Exactitud y exactitud balanceada, L1 de las fracciones contando clase (argmax) y sumando
   probabilidades, comparaciones pareadas (bootstrap estratificado de la exactitud balanceada y bootstrap de SNe del
   L1) y fracciones corregidas con la matriz de confusion del sistema medida en val_sel (de los pred_real_val.csv
   congelados) aplicada a test. Ademas pares de metodos solos en sus oids comunes (3 y 4 clases).
Salidas: RUNS/test_final/<metodo>/pred_test.csv y metrics.json (metodo = villar, red, plantillas, villar_4c,
plantillas_4c) y RUNS/test_final/resumen.json.
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline78 import splits
from pipeline78.nnclf import data as D
from pipeline78.paths import REPO, RUNS

FLAG = "--mitad-final-autorizada"
OUT = RUNS / "test_final"
REAL_DIR = RUNS / "real_ztf"
REAL_FEAT = RUNS / "features_real_ztf3" / "features" / "features.csv"
SEED = 20261004
N_BOOT_IC = 1000
N_REPRO = 6
CONGELADOS = {
    "villar": RUNS / "clf_villar" / "sweep_t11_foco" / "mejor",
    "villar_4c": RUNS / "clf_villar" / "sweep_t11_fisica_4c" / "mejor",
    "red": RUNS / "nnclf_t11" / "tf_base",
    "plantillas": RUNS / "plantillas_clf" / "plantillas_t11",
    "plantillas_4c": RUNS / "plantillas_clf" / "plantillas_t11_4c",
}
METODOS_3C = ("villar", "red", "plantillas")
METODOS_4C = ("villar_4c", "plantillas_4c")


# ------------------------------------------------------------------------------------------------ guarda y lectura
def _is_final(origen, split, excluir):
    return origen == "holdout" and split == "final" and not splits._true(excluir)


def read_final_meta(path, autorizada=False):
    """meta_real_ztf.csv -> DataFrame con SOLO las filas de la mitad final (regla 2). Sin autorizada=True no abre el
    archivo. Una segunda pasada mira solo la columna oid de las otras filas: ninguna oid final puede estar fuera."""
    if autorizada is not True:
        raise PermissionError(f"la mitad final esta bloqueada: solo se lee con {FLAG}")
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if _is_final(r.get("origen"), r.get("split"), r.get("excluir"))]
    fo = {r["oid"] for r in rows}
    if len(fo) != len(rows):
        raise ValueError("oid repetida en la mitad final")
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != "final" and r["oid"] in fo:
                raise ValueError(f"la oid final {r['oid']} aparece tambien en otro split")
    return splits._typed(pd.DataFrame(rows, columns=None if rows else ["oid", "sn_type", "origen", "split", "excluir"]))


def final_meta(real_dir=REAL_DIR, four=False, autorizada=False):
    """Metadatos final de las clases pedidas, con cls, subset = test, part_index entero y z numerico."""
    v = read_final_meta(Path(real_dir) / "meta_real_ztf.csv", autorizada)
    v["oid"] = v.oid.astype(str)
    v["cls"] = v.sn_type.map(lambda t: D.class_of(t, four))
    v["subset"] = "test"
    if "part_index" in v:
        v["part_index"] = v.part_index.astype(int)
    v["z"] = pd.to_numeric(v.z, errors="coerce")
    return v[v.cls.notna()].reset_index(drop=True)


def load_final_curves(v, real_dir=REAL_DIR, four=False, min_det=D.MIN_DET):
    """Curvas g, r de las oids de v (final_meta), como nnclf.data.load_real_val: pyarrow filtra por oid al leer.
    Devuelve (curvas con >= min_det detecciones g + r, oids sin curva suficiente)."""
    cls = D.classes(four)
    info = v.assign(y=v.cls.map({c: i for i, c in enumerate(cls)}), w=1.0, template="").set_index("oid")
    parts = []
    for st, grp in v.groupby("sn_type"):
        tab = pq.read_table(Path(real_dir) / f"{st}.parquet", columns=["oid"] + D._COLS,
                            filters=[("oid", "in", sorted(grp.oid)), ("filter", "in", list(D.BANDS))])
        parts.append(tab.to_pandas())
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["oid"] + D._COLS)
    if not set(rows.oid) <= set(v.oid):
        raise AssertionError("se colo una oid fuera de las pedidas")
    curves = [c for c in D._to_curves(rows, "oid", info) if c.n_det() >= min_det]
    return curves, sorted(set(v.oid) - {c.key for c in curves})


# ------------------------------------------------------------------------------------------------ metricas
def metricas_metodo(pred, v, cls):
    """Regla 5. pred: oid, y_true, y_pred, p_<clase>. v: SNe final de las clases (denominador de la cobertura)."""
    from pipeline78.clf_villar import bootstrap_ci, calib_metrics, metrics
    out = {"n_real": int(len(v)), "n_clasificadas": int(len(pred)),
           "cobertura": float(len(pred) / max(len(v), 1)),
           "cobertura_por_clase": {c: float((pred.y_true == c).sum() / max((v.cls == c).sum(), 1)) for c in cls},
           "n_real_por_clase": {c: int((v.cls == c).sum()) for c in cls}}
    if not len(pred):
        return out
    y = pred.y_true.map(list(cls).index).to_numpy()
    yp = pred.y_pred.map(list(cls).index).to_numpy()
    out.update(metrics(y, yp, cls))
    out.update(calib_metrics(pred[[f"p_{c}" for c in cls]].to_numpy(float), y))
    out.update(bootstrap_ci(y, yp, cls, N_BOOT_IC, SEED))
    return out


def comparar_val(nuevo, guardado, cls):
    """Regla 4: predicciones rehechas en val contra el pred_real_val.csv congelado, en las oids comunes."""
    g = pd.read_csv(guardado, dtype={"oid": str}).drop_duplicates("oid").set_index("oid")
    n = nuevo.drop_duplicates("oid").set_index("oid")
    com = sorted(set(n.index) & set(g.index))
    pc = [f"p_{c}" for c in cls]
    dp = np.abs(n.loc[com, pc].to_numpy(float) - g.loc[com, pc].to_numpy(float)) if com else np.zeros((0, len(cls)))
    return {"guardado": str(guardado), "n_rehechas": int(len(n)), "n_guardadas": int(len(g)), "n_comun": len(com),
            "solo_rehechas": sorted(set(n.index) - set(g.index))[:20],
            "max_abs_dp": float(dp.max()) if dp.size else None,
            "cambios_argmax": int((n.loc[com, "y_pred"] != g.loc[com, "y_pred"]).sum()) if com else None}


def _git():
    try:
        h = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "pipeline78"], capture_output=True,
                           text=True).stdout.strip()
        return {"commit": h, "pipeline78_sin_commitear": d.splitlines()}
    except OSError:
        return None


def escribir(nombre, pred, res, out_root=OUT):
    d = Path(out_root) / nombre
    d.mkdir(parents=True, exist_ok=True)
    pred.to_csv(d / "pred_test.csv", index=False)
    res = {**res, "git": _git(), "creada": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    m = res.get("test") or {}
    print(f"[test] {nombre}: n {m.get('n')} de {m.get('n_real')} (cobertura {m.get('cobertura', float('nan')):.3f}) "
          f"acc {m.get('acc', float('nan')):.3f} bal {m.get('bal_acc', float('nan')):.3f} IC95 {m.get('bal_acc_ic95')}"
          f" -> {d}", flush=True)
    return d


# ------------------------------------------------------------------------------------------------ villar
def villar_R(Rw, v, bundle, features_real, real_dir, cache=None):
    """Tabla derivada y probabilidades templadas de una tabla ancha de reales: el camino de clf_villar.cmd_eval."""
    from pipeline78 import clf_villar as CV
    cls = tuple(bundle["classes"])
    R = CV.derive(Rw, not bundle["obs_frame"])
    if set(CV.FIS) & set(bundle["cols"]):
        R = CV.con_fisica(R, CV.fisica_reales(R, v, features_real, real_dir, not bundle["obs_frame"], cache=cache))
    if bundle["requiere"] == "r":
        R = R[R.tiene_r].reset_index(drop=True)
    R["y"] = R.cls.map({c: i for i, c in enumerate(cls)}).astype(int)
    P = CV.temper(bundle["model"].predict_proba(R), bundle["temperatura"]) if len(R) else np.zeros((0, len(cls)))
    return R, P


def _pred_villar(R, P, cls):
    return pd.DataFrame({"oid": R.oid.astype(str).to_numpy(), "subset": R["subset"].to_numpy(),
                         "sn_type": R.sn_type.to_numpy(), "y_true": R.cls.to_numpy(), "tiene_r": R.tiene_r.to_numpy(),
                         "y_pred": [cls[i] for i in P.argmax(1)], **{f"p_{c}": P[:, i] for i, c in enumerate(cls)}})


def cmd_villar(a):
    import joblib
    from threadpoolctl import threadpool_limits
    from pipeline78 import clf_villar as CV
    CV._silenciar_matmul()
    nombre = "villar_4c" if a.cuatro_clases else "villar"
    run = CONGELADOS[nombre]
    # pickle propio de clf_villar en RUNS local (confiable). Se escribio con `python -m pipeline78.clf_villar`: sus
    # clases quedaron como __main__.Model / __main__.ModeloG, asi que se exponen en __main__ antes de cargar.
    import __main__
    for k in ("Model", "ModeloG"):
        if not hasattr(__main__, k):
            setattr(__main__, k, getattr(CV, k))
    bundle = joblib.load(run / "model.joblib")
    cls = tuple(bundle["classes"])
    four = len(cls) == 4
    if four != a.cuatro_clases:
        raise SystemExit(f"{run}: {len(cls)} clases")
    feat = CV.features_csv(a.features_real)
    with threadpool_limits(CV.N_JOBS):
        # regla 4: val por el mismo camino (lectores con guarda de clf_villar), contra pred_real_val.csv
        Rw, vv = CV.load_real_val(a.features_real, a.real_dir, four)
        Rv, Pv = villar_R(Rw, vv, bundle, a.features_real, a.real_dir)
        repro = comparar_val(_pred_villar(Rv, Pv, cls), run / "pred_real_val.csv", cls)
        print(f"[test] {nombre} repro val: {repro}", flush=True)
        if a.solo_repro:
            return repro
        # test
        v = final_meta(a.real_dir, four, autorizada=True)
        f = CV.read_rows_for_oids(feat, v.oid)
        if not set(f.oid) <= set(v.oid):
            raise AssertionError("se colo una oid fuera de la mitad final")
        cols = CV.KEYS + ["z", "subtipo", "cls", "subset"]
        f["part_index"] = f.part_index.astype(int)
        W = CV.widen(f).merge(v[cols], on=CV.KEYS, how="inner", validate="one_to_one")
        R, P = villar_R(W, v, bundle, a.features_real, a.real_dir,
                        cache=Path(a.out_root) / nombre / "fisica_test_rest.csv")
    pred = _pred_villar(R, P, cls)
    res = {"metodo": nombre, "congelado": str(run), "config": bundle["config"], "cols": bundle["cols"],
           "temperatura": bundle["temperatura"], "prior": "none (argmax del modelo templado)",
           "obs_frame": bundle["obs_frame"], "requiere": bundle["requiere"], "features_real": str(feat),
           "clases": list(cls), "repro_val": repro, "test": metricas_metodo(pred, v, cls),
           "test_con_r": metricas_metodo(pred[pred.tiene_r], v, cls)}
    return escribir(nombre, pred, res, a.out_root)


# ------------------------------------------------------------------------------------------------ red
def _probs_red(model, cfg, dev, curves):
    from pipeline78.nnclf.evaluate import nn_prob_fn
    dcs, keep = [], []
    for c in curves:                                   # celda principal: natural, g+r, todas (un solo pase)
        dc = D.degrade(c, None, ("g", "r"), D.curve_rng(cfg.seed, c.key, 2, 0, 0))
        if dc is not None:
            dcs.append(dc)
            keep.append(c)
    p = nn_prob_fn(model, cfg, dev)(dcs) if dcs else np.zeros((0, 0))
    return keep, dcs, p


def _pred_red(curves, dcs, p, cls, subset):
    return pd.DataFrame({"oid": [c.key for c in curves], "subset": subset, "sn_type": [c.sn_type for c in curves],
                         "y_true": [cls[c.y] for c in curves], "y_pred": [cls[i] for i in p.argmax(1)],
                         "n_det": [dc.n_det() for dc in dcs], **{f"p_{c}": p[:, i] for i, c in enumerate(cls)}})


def cmd_red(a):
    import torch
    from pipeline78.nnclf import calib
    from pipeline78.nnclf.evaluate import cfg_device
    from pipeline78.nnclf.train import load_model, pick_device
    run = CONGELADOS["red"]
    model, cfg, ck = load_model(run, pick_device(a.device or cfg_device(run)))
    torch.set_num_threads(a.threads)
    dev = next(model.parameters()).device
    cls = tuple(ck["classes"])
    four = len(cls) == 4
    # regla 4: val por el lector con guarda de nnclf
    rv, _ = D.load_real_val(cfg.real_dir, four)
    kv, dv, pv = _probs_red(model, cfg, dev, rv)
    repro = comparar_val(_pred_red(kv, dv, pv, cls, "val"), run / "pred_real_val.csv", cls)
    print(f"[test] red repro val: {repro}", flush=True)
    if a.solo_repro:
        return repro
    v = final_meta(a.real_dir, four, autorizada=True)
    curves, sin = load_final_curves(v, a.real_dir, four)
    kt, dt, pt = _probs_red(model, cfg, dev, curves)
    pred = _pred_red(kt, dt, pt, cls, "test")
    cal = json.loads((run / "metrics.json").read_text())["calibration"]
    y = pred.y_true.map(list(cls).index).to_numpy()
    calib_test = calib._block(pt.astype(np.float64), y, cal["T"], cal["T_prior"], np.asarray(cal["log_prior_adj"]),
                              cal.get("n_bins", calib.N_BINS)) if len(pred) else None
    res = {"metodo": "red", "congelado": str(run), "config": json.loads((run / "config.json").read_text()),
           "device": str(dev), "celda": "natural, g+r, todas las detecciones", "probabilidades": "softmax cruda",
           "clases": list(cls), "repro_val": repro, "sin_3_det_gr": sin,
           "calibracion_congelada_val_sel": {k: cal[k] for k in ("T", "T_prior", "priors_val_sel", "log_prior_adj")},
           "calibracion_test": calib_test, "test": metricas_metodo(pred, v, cls)}
    return escribir("red", pred, res, a.out_root)


# ------------------------------------------------------------------------------------------------ plantillas
def _constantes_plantillas(PC, c, cfg_name):
    """Constantes de la corrida congelada que no coinciden con el codigo actual (regla 3)."""
    from pipeline78 import runcfg
    actual = dict(laplace_nats=PC.LAPLACE_NATS, sigma_z=PC.SIGMA_Z, z_floor=PC.Z_FLOOR, ul_nsig=PC.UL_NSIG,
                  t_pre=PC.T_PRE, min_det=PC.MIN_DET, n_u=len(PC.U), u_max=float(PC.U[-1]), dmu_sub=PC.DMU_SUB,
                  masa_min_z=PC.MASA_MIN_Z, masa_min_e=PC.MASA_MIN_E,
                  mw_const=runcfg.RUNS_CFG[cfg_name].get("mw_const", 0.02),
                  ii_dust=runcfg.RUNS_CFG[cfg_name].get("ii_dust"))
    return {k: {"congelado": c.get(k), "actual": x} for k, x in actual.items() if c.get(k) != x}


def _clasificar(PC, curves, lib_dir, four, c, workers, mw):
    from multiprocessing import get_context
    from pipeline78 import runcfg
    mw_const = runcfg.RUNS_CFG[c["run_cfg"]].get("mw_const", 0.02)
    tareas = [(cur, float(mw.get(cur.key, mw_const))) for cur in curves]
    init = (str(lib_dir), four, c["sin_z"], c["run_cfg"], c["ul_modo"], float(c["sigma_mod"]), "cfg")
    t1, res = time.time(), []
    with get_context("spawn").Pool(workers, initializer=PC._init, initargs=init) as pool:
        for i, r in enumerate(pool.imap(PC._uno, tareas, chunksize=1), 1):
            res.append(r)
            if i % 25 == 0 or i == len(tareas):
                print(f"[test] plantillas {i}/{len(tareas)} {time.time() - t1:.0f}s", flush=True)
    return res, tareas, time.time() - t1


def _pred_plantillas(res, tareas, info, cls, subset):
    filas, nada = [], []
    for (oid, r, dt), (cur, m) in zip(res, tareas):
        if r is None:
            nada.append(dict(oid=oid, motivo="sin modelo valido"))
            continue
        filas.append(dict(oid=oid, subset=subset, sn_type=info.sn_type[oid], y_true=info.cls[oid],
                          y_pred=cls[int(np.argmax(r["p"]))], **{f"p_{k}": float(r["p"][i]) for i, k in enumerate(cls)},
                          n_det=r["n_det"], n_ul=r["n_ul"], z=cur.z, prior_z=r["prior_z"], ebv_mw=m,
                          best_template=r["best_template"], best_template_clase=r["best_template_clase"],
                          chi2_min=r["chi2_min"], chi2_map=r["chi2_map"], chi2_dat_map=r["chi2_dat_map"],
                          z_map=r["z_map"], ebv_map=r["ebv_map"], tmax_map=r["tmax_map"], M_map=r["M_map"],
                          **{f"logE_{k}": float(r["logE"][i]) for i, k in enumerate(cls)}, t_seg=dt))
    return pd.DataFrame(filas), nada


def cmd_plantillas(a):
    from pipeline78 import plantillas_clf as PC, sampling
    nombre = "plantillas_4c" if a.cuatro_clases else "plantillas"
    run = CONGELADOS[nombre]
    m = json.loads((run / "metrics.json").read_text())
    c = m["config"]
    four = bool(c["cuatro_clases"])
    if four != a.cuatro_clases or c["subset"] != "val" or c["limit"]:
        raise SystemExit(f"{run}: config inesperada {c}")
    lib_dir = Path(m["biblioteca"])
    cls = D.classes(four)
    problemas = []
    clave = PC.biblioteca_clave(cfg_name=c["run_cfg"])
    if f"biblioteca_{clave}" != lib_dir.name:
        problemas.append(f"clave actual de la biblioteca {clave} distinta de la congelada {lib_dir.name}: se usa la "
                         "congelada")
    if json.loads((lib_dir / "meta.json").read_text()) != m["biblioteca_meta"]:
        problemas.append("meta.json de la biblioteca distinto del guardado en la corrida de val")
    dif = _constantes_plantillas(PC, c, c["run_cfg"])
    if dif:
        problemas.append(f"constantes distintas a las de la corrida de val: {dif}")
    workers = max(1, min(int(a.workers), 8))
    mw = sampling.load_mw(dict(mw_mode="ztf_sfd"))
    # regla 4: N_REPRO SNe fijas de val_sel (plantillas_clf.elegir_oids, lectores con guarda)
    vv = PC.val_meta(a.real_dir, four)
    ro = set(PC.elegir_oids(vv, "val_sel", a.n_repro))
    cv, _ = D.load_real_val(a.real_dir, four_classes=four, min_det=PC.MIN_DET)
    rres, rtar, _ = _clasificar(PC, [x for x in cv if x.key in ro], lib_dir, four, c, workers, mw)
    rpred, _ = _pred_plantillas(rres, rtar, vv.set_index("oid"), cls, "val_sel")
    repro = comparar_val(rpred, run / "pred_real_val.csv", cls)
    print(f"[test] {nombre} repro val ({len(ro)} de val_sel): {repro}", flush=True)
    if a.solo_repro:
        return repro
    # test
    v = final_meta(a.real_dir, four, autorizada=True)
    curves, sin = load_final_curves(v, a.real_dir, four, PC.MIN_DET)
    res, tareas, t_clf = _clasificar(PC, curves, lib_dir, four, c, workers, mw)
    pred, nada = _pred_plantillas(res, tareas, v.set_index("oid"), cls, "test")
    nada += [dict(oid=o, motivo=f"menos de {PC.MIN_DET} detecciones g + r") for o in sin]
    S = pd.DataFrame(nada, columns=["oid", "motivo"])
    d = Path(a.out_root) / nombre
    d.mkdir(parents=True, exist_ok=True)
    S.assign(sn_type=S.oid.map(v.set_index("oid").sn_type)).to_csv(d / "sin_clasificar.csv", index=False)
    met = metricas_metodo(pred, v, cls)
    if len(pred):
        met["chi2_reducido"] = PC.resumen_chi2(pred)
    ts = pred.t_seg.to_numpy() if len(pred) else np.zeros(1)
    res_json = {"metodo": nombre, "congelado": str(run), "config_congelada": c, "biblioteca": str(lib_dir),
                "clases": list(cls), "problemas": problemas, "repro_val": repro, "workers": workers,
                "tiempo": dict(clasificacion_s=round(t_clf, 1), n_sn=len(tareas),
                               por_sn_mediana_s=float(np.median(ts))),
                "n_sin_mw": int(sum(x.key not in mw for x in curves)), "test": met}
    return escribir(nombre, pred, res_json, a.out_root)


# ------------------------------------------------------------------------------------------------ resumen y sistema
def leer_test(nombre, out_root=OUT, cls=("Ia", "II", "Ibc")):
    """pred_test.csv de un metodo como sistema_tasas.leer: indice oid, subset, y_true, y_pred, p_<clase>."""
    d = pd.read_csv(Path(out_root) / nombre / "pred_test.csv", dtype={"oid": str})
    if set(d.subset) != {"test"}:
        raise ValueError(f"{nombre}: filas fuera de test")
    d = d[d.y_true.isin(cls)].drop_duplicates("oid").set_index("oid")
    return d[["subset", "y_true", "y_pred"] + [f"p_{c}" for c in cls]]


def pares_comunes(P, cls):
    """Bootstrap pareado estratificado (nnclf.experimentos.paired_bootstrap) de la exactitud balanceada entre metodos
    solos, en sus oids comunes. P: {nombre: DataFrame indexado por oid}."""
    from pipeline78.clf_villar import metrics
    from pipeline78.nnclf.experimentos import P_MIN, paired_bootstrap
    out, ks = {}, list(P)
    for i, x in enumerate(ks):
        for z in ks[i + 1:]:
            com = sorted(set(P[x].index) & set(P[z].index))
            if not com:
                continue
            A, B = P[x].loc[com], P[z].loc[com]
            assert (A.y_true == B.y_true).all()
            y = A.y_true.to_numpy()
            dlt, p, ci = paired_bootstrap(y, (A.y_pred == y).to_numpy(), (B.y_pred == y).to_numpy())
            yi = A.y_true.map(list(cls).index).to_numpy()
            out[f"{x}_vs_{z}"] = dict(n=len(com), **{f"bal_acc_{k}": metrics(yi, Q.y_pred.map(list(cls).index)
                                                                                .to_numpy(), cls)["bal_acc"]
                                                     for k, Q in ((x, A), (z, B))},
                                      delta_bal=dlt, P_bal=p, ic90_bal=ci,
                                      gana_bal=(x if p >= P_MIN else z if p <= 1 - P_MIN else None))
    return out


def sistema_test(T, V, respaldo="red"):
    """Regla 6. T: {metodo: preds test (leer_test)}; V: {metodo: preds val (sistema_tasas.leer)}. Devuelve el bloque
    del sistema completo en test con las fracciones corregidas con la matriz de val_sel."""
    from pipeline78 import sistema_tasas as ST
    from pipeline78.nnclf.experimentos import P_MIN, paired_bootstrap
    S = {k: ST.sistema(d, T[respaldo]) for k, d in T.items()}
    SV = {k: ST.sistema(d, V[respaldo]) for k, d in V.items()}
    res = {"respaldo": respaldo, "metricas": {k: ST.metricas(d) for k, d in S.items()}, "pares": {},
           "corregidas": {}}
    ks = list(S)
    for i, x in enumerate(ks):
        for z in ks[i + 1:]:
            A, B = S[x], S[z]
            assert (A.index == B.index).all() and (A.y_true == B.y_true).all()
            y = A.y_true.to_numpy()
            dlt, p, ci = paired_bootstrap(y, (A.y_pred == y).to_numpy(), (B.y_pred == y).to_numpy())
            pa, cia = ST.l1_boot(A, B, col="argmax")
            pp, cip = ST.l1_boot(A, B, col="prob")
            res["pares"][f"{x}_vs_{z}"] = dict(
                delta_bal=dlt, P_bal=p, ic90_bal=ci, P_L1_argmax_menor=pa, ic90_dL1_argmax=cia, P_L1_prob_menor=pp,
                ic90_dL1_prob=cip, gana_bal=(x if p >= P_MIN else z if p <= 1 - P_MIN else None))
    for k, d in S.items():
        sel = SV[k][SV[k].subset == "val_sel"]
        l1, f, cond = ST.corregidas(sel, d)
        med, ic = ST.corregidas_boot(sel, d)
        res["corregidas"][k] = dict(L1=l1, frac=f, cond_M=cond, L1_boot_mediana=med, ic90=ic,
                                    M_de="val_sel (pred_real_val.csv congelados)", n_val_sel=int(len(sel)))
    return res


def cmd_resumen(a):
    from pipeline78 import sistema_tasas as ST
    out = Path(a.out_root)
    falta = [k for k in METODOS_3C + METODOS_4C if not (out / k / "metrics.json").exists()]
    M = {k: json.loads((out / k / "metrics.json").read_text()) for k in METODOS_3C + METODOS_4C
         if (out / k / "metrics.json").exists()}
    res = {"regla": __doc__.split("\n\n")[0], "falta": falta, "metodos": {}, "git": _git(),
           "creada": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k, m in M.items():
        t = m["test"]
        res["metodos"][k] = {"congelado": m["congelado"], "repro_val": m.get("repro_val"),
                             "problemas": m.get("problemas", []),
                             **{q: t.get(q) for q in ("n_real", "n_clasificadas", "cobertura", "cobertura_por_clase",
                                                      "acc", "acc_ic95", "bal_acc", "bal_acc_ic95", "f1_macro",
                                                      "logloss", "ece", "confusion")},
                             **{q: t.get(q) for q in t if q.startswith("f1_") or q.startswith("recall_")}}
    if all(k in M for k in METODOS_3C):
        cls = ("Ia", "II", "Ibc")
        T = {k: leer_test(k, out, cls) for k in METODOS_3C}
        V = {k: ST.leer(CONGELADOS[k] / "pred_real_val.csv") for k in METODOS_3C}
        res["sistema_3c"] = sistema_test(T, V)
        n3 = M["red"]["test"]["n_real"]
        res["sistema_3c"]["cobertura_sobre_final"] = {"n_sistema": int(len(T["red"])), "n_final": n3,
                                                     "cobertura": float(len(T["red"]) / max(n3, 1))}
        res["pares_metodos_3c"] = pares_comunes(T, cls)
    if all(k in M for k in METODOS_4C):
        cls = ("Ia", "II", "Ibc", "IIn")
        res["pares_metodos_4c"] = pares_comunes({k: leer_test(k, out, cls) for k in METODOS_4C}, cls)
    (out / "resumen.json").write_text(json.dumps(res, indent=1, default=float))
    for k, m in res["metodos"].items():
        print(f"[test] {k:14s} n {m['n_clasificadas']}/{m['n_real']} cob {m['cobertura']:.3f} acc {m['acc']:.3f} "
              f"bal {m['bal_acc']:.3f} IC95 [{m['bal_acc_ic95'][0]:.3f}, {m['bal_acc_ic95'][1]:.3f}]", flush=True)
    if "sistema_3c" in res:
        s = res["sistema_3c"]
        for k, m in s["metricas"].items():
            c = s["corregidas"][k]
            print(f"[test] sistema {k:10s} n {m['n']} propia {m['cobertura_propia']:.3f} acc {m['acc']:.3f} bal "
                  f"{m['bal_acc']:.3f} L1 argmax {m['L1_argmax']:.3f} prob {m['L1_prob']:.3f} corregida {c['L1']:.3f}",
                  flush=True)
        for k, p in s["pares"].items():
            print(f"[test] sistema {k}: dbal {p['delta_bal']:+.3f} P {p['P_bal']:.3f} | P(L1 menor) argmax "
                  f"{p['P_L1_argmax_menor']:.3f} prob {p['P_L1_prob_menor']:.3f}", flush=True)
    print(f"[test] -> {out / 'resumen.json'}" + (f" (faltan {falta})" if falta else ""), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ CLI
def parser():
    ap = argparse.ArgumentParser(prog="python -m pipeline78.evaluar_test", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("villar", "red", "plantillas", "resumen"))
    ap.add_argument(FLAG, dest="autorizada", action="store_true",
                    help="obligatorio: autoriza leer la mitad final del holdout (orden de Mauricio 2026-10-06)")
    ap.add_argument("--cuatro-clases", action="store_true", help="villar y plantillas: las corridas de 4 clases")
    ap.add_argument("--real-dir", type=Path, default=REAL_DIR)
    ap.add_argument("--features-real", type=Path, default=REAL_FEAT)
    ap.add_argument("--out-root", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=4, help="plantillas: procesos (maximo 8)")
    ap.add_argument("--n-repro", type=int, default=N_REPRO, help="plantillas: SNe de val_sel para la regla 4")
    ap.add_argument("--device", default=None, help="red: por defecto el de config.json (el de la evaluacion de val)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--solo-repro", action="store_true", help="solo la regla 4 (val): no lee la mitad final")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if not a.autorizada:
        raise SystemExit(f"mitad final bloqueada: agregar {FLAG} (solo por orden explicita de Mauricio)")
    return {"villar": cmd_villar, "red": cmd_red, "plantillas": cmd_plantillas, "resumen": cmd_resumen}[a.cmd](a)


if __name__ == "__main__":
    main()
