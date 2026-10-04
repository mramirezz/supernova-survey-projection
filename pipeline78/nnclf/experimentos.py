"""Cola de experimentos del clasificador NN con las mejoras de la literatura (nn-lit-brief, sec. 4).

Una corrida a la vez, en CPU con --threads (2 por defecto), cada una en su propio proceso (python -m pipeline78.nnclf
...). Todas reciben OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS y VECLIB_MAXIMUM_THREADS = --threads
(revision H2: el HistGradientBoosting de Villar usa OpenMP y pyarrow su propio pool, y sin esto toman todos los
nucleos). Cada corrida deja metrics.json en <out_root>/<nombre> y la cola reescribe <out_root>/resumen.csv despues de
cada una. Es reanudable: la corrida que ya tiene metrics.json se salta (salvo --force).

SELECCION ANIDADA (revision H1, decision del 2026-10-04)
- La mitad val de las reales se parte en val_sel y val_rep (pipeline78.splits.val_split: 50/50 estratificada por
  clase, semilla 20261004). Las fases 2 y 3 eligen SOLO con val_sel. val_rep no entra en ninguna decision y es la
  cifra honesta. resumen.csv da cada metrica de las reales con el sufijo _rep, _sel y _val (val completo, referencia).
- Regla de eleccion (`compare`): bootstrap pareado, estratificado por clase, de los objetos de val_sel que clasifican
  las dos corridas (celda principal: natural, g+r, todas; N_BOOT = 2000 remuestreos con semilla fija). En cada
  remuestreo delta = exactitud balanceada de la candidata menos la de la incumbente. La candidata gana solo si
  P(delta > 0) >= P_MIN = 0.9. Si no, se queda la incumbente, que siempre es la configuracion mas simple.

FASES
1. Base y ablaciones de una sola cosa, sin z, fold 0:
   gru_base, tf_base, gru_lambda, tf_lambda (banda por lambda pivote), gru_bidir (control de gru_attnpool),
   gru_attnpool (BiGRU + attention pooling, ORACLE-2), tf_timemod (TimeModulator de ATAT), gru_trunc y tf_trunc
   (truncamiento ORACLE-2, modo "both", p_trunc = 1), gru_trunc05 y tf_trunc05 (lo mismo con p_trunc = 0.5, revision
   H4), gru_jer y tf_jer (cabeza jerarquica, Ia contra CC y despues la subclase, models.HierHead), snn_noz, snn_z y
   snn_var_noz (SuperNNova estandar y variational), villar_noz y villar_z (si estan las features de las sims).
   Bloque de 4 clases FASE1_4C (Ia, II, Ibc, IIn; --four-classes), al final de la fase 1: gru_base_4c,
   gru_attnpool_4c y gru_jer_4c. IIn es una de las 4 clases de las tasas de SUDARE (Cappellaro et al. 2015,
   2015A&A...584A..62C: Ia 67, II 22, Ib/c 17, IIn 11), y la metrica principal es de 3 clases. El bloque no entra
   en las fases 2 y 3 (no compite con las de 3 clases). Solo ese bloque:
       python -m pipeline78.nnclf.experimentos run --fase 1 --only gru_base_4c gru_attnpool_4c gru_jer_4c
2. Mejor configuracion sin z:
   a. Cada ablacion se compara con <arq>_base (la incumbente). Pasa si gana con la regla. Las ablaciones de un mismo
      slot son excluyentes (trunc y trunc05 tocan los mismos flags): si pasan las dos, queda la de mayor exactitud
      balanceada en val_sel.
   b. Si no pasa ninguna, queda <arq>_base. Si pasa una, queda esa misma corrida (no se repite). Si pasan dos o mas,
      se entrena la COMBINACION de sus flags (<arq>_comb_<ablaciones>) y se compara con la mejor ablacion individual
      (la incumbente, mas simple). La combinacion queda solo si gana con la regla.
   c. Entre arquitecturas, la incumbente es la de menos parametros entrenables (n_params de config.json). La otra
      queda solo si gana con la regla.
   d. Se entrena <mejor>_z (los mismos flags con --use-z). Todas las comparaciones quedan en fase2_eleccion.json.
   La fase va en dos tandas: primero las combinaciones y despues la decision y la corrida con z. `plan` muestra la
   segunda tanda solo cuando la primera ya esta.
3. z o no: <mejor>_z es la candidata y <mejor> la incumbente (mas simple: no necesita z). Con la que gane se entrenan
   los folds 1 a 4 y el ensemble de los 5 (<nombre>_ens5). La comparacion queda en fase3_eleccion.json.
La cifra de la tesis sale de la mitad final, con la configuracion, la T y los priors elegidos aca.

USO (desde la raiz del repo, env series)
    python -m pipeline78.nnclf.experimentos plan  --fase 1            # imprime los comandos
    python -m pipeline78.nnclf.experimentos run   --fase 1 [--sims-dir ...] [--sim-feat ...] [--smoke]
    python -m pipeline78.nnclf.experimentos resumen                   # rehace resumen.csv
--smoke: 1 epoca, 1500 sims, 1 sorteo, nombres con prefijo smoke_. No usar para resultados.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78 import splits
from pipeline78.nnclf import data as D

PY = sys.executable
FASE1 = [
    ("gru_base", "nn", ["--model", "gru"]),
    ("tf_base", "nn", ["--model", "transformer"]),
    ("gru_lambda", "nn", ["--model", "gru", "--band-enc", "lambda"]),
    ("tf_lambda", "nn", ["--model", "transformer", "--band-enc", "lambda"]),
    ("gru_bidir", "nn", ["--model", "gru", "--bidir"]),
    ("gru_attnpool", "nn", ["--model", "gru", "--bidir", "--gru-pool", "attn"]),
    ("tf_timemod", "nn", ["--model", "transformer", "--time-enc", "atat"]),
    ("gru_trunc", "nn", ["--model", "gru", "--trunc", "both"]),
    ("tf_trunc", "nn", ["--model", "transformer", "--trunc", "both"]),
    ("gru_trunc05", "nn", ["--model", "gru", "--trunc", "both", "--p-trunc", "0.5"]),
    ("tf_trunc05", "nn", ["--model", "transformer", "--trunc", "both", "--p-trunc", "0.5"]),
    ("gru_jer", "nn", ["--model", "gru", "--jerarquica"]),
    ("tf_jer", "nn", ["--model", "transformer", "--jerarquica"]),
    ("snn_noz", "snn", []),
    ("snn_z", "snn", ["--use-z"]),
    ("snn_var_noz", "snn", ["--snn-model", "variational"]),
    ("villar_noz", "villar", []),
    ("villar_z", "villar", ["--use-z"]),
]
# 4 clases (con IIn): la base, la ablacion de ORACLE-2 y la jerarquica. Va al final de la fase 1 (--only para correrlo)
FASE1_4C = [
    ("gru_base_4c", "nn", ["--model", "gru", "--four-classes"]),
    ("gru_attnpool_4c", "nn", ["--model", "gru", "--bidir", "--gru-pool", "attn", "--four-classes"]),
    ("gru_jer_4c", "nn", ["--model", "gru", "--jerarquica", "--four-classes"]),
]
# ablaciones de la fase 2: arquitectura -> slot -> {corrida: flags}. Las variantes de un slot son excluyentes.
# gru_bidir es solo el control de gru_attnpool y no compite.
TRUNC1, TRUNC05 = ["--trunc", "both"], ["--trunc", "both", "--p-trunc", "0.5"]
ABLACIONES = {"gru": {"lambda": {"gru_lambda": ["--band-enc", "lambda"]},
                      "attnpool": {"gru_attnpool": ["--bidir", "--gru-pool", "attn"]},
                      "trunc": {"gru_trunc": TRUNC1, "gru_trunc05": TRUNC05},
                      "jer": {"gru_jer": ["--jerarquica"]}},
              "tf": {"lambda": {"tf_lambda": ["--band-enc", "lambda"]},
                     "timemod": {"tf_timemod": ["--time-enc", "atat"]},
                     "trunc": {"tf_trunc": TRUNC1, "tf_trunc05": TRUNC05},
                     "jer": {"tf_jer": ["--jerarquica"]}}}
ARCH = {"gru": "gru", "tf": "transformer"}
N_BOOT = 2000
P_MIN = 0.9
BOOT_SEED = splits.SEED
MIN_COMUN = 30
CRITERIO = {"subconjunto": "val_sel (pipeline78.splits.val_split, semilla %d)" % splits.SEED,
            "metrica": "exactitud balanceada, celda principal (natural, g+r, todas)",
            "regla": "bootstrap pareado estratificado por clase; la candidata gana si P(delta > 0) >= %.2f" % P_MIN,
            "n_boot": N_BOOT, "semilla_bootstrap": BOOT_SEED,
            "empate": "se queda la incumbente (la configuracion mas simple)"}
SMOKE = {"nn": ["--max-epochs", "1", "--max-sims", "1500", "--n-draws", "1"],
         "snn": ["--nb-epoch", "1", "--max-sims", "1500", "--n-draws", "1", "--num-inference-samples", "5"],
         "villar": []}


# ---------------------------------------------------------------- regla de eleccion (solo val_sel)
def sel_oids(real_dir=D.REAL_DIR):
    """Oids de val_sel (todas las clases; cada corrida trae solo las suyas)."""
    sel, _ = splits.val_split(splits.read_val_meta(Path(real_dir) / "meta_real_ztf.csv"))
    return set(sel)


def _bal(y, ok):
    return float(np.mean([ok[y == k].mean() for k in np.unique(y)]))


def paired_bootstrap(y, ok_cand, ok_inc, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap pareado y estratificado por clase de la diferencia de exactitud balanceada.

    y: clase verdadera; ok_*: acierto por objeto (los mismos objetos en las dos). En cada remuestreo se sortean con
    reemplazo los objetos de cada clase (el mismo sorteo para las dos corridas) y delta = media sobre las clases de la
    diferencia de exactitud por clase. Devuelve (delta observado, P(delta > 0), (percentil 5, percentil 95))."""
    y = np.asarray(y)
    a, b = np.asarray(ok_cand, np.float64), np.asarray(ok_inc, np.float64)
    rng = np.random.default_rng(seed)
    cls = np.unique(y)
    d = np.zeros(n_boot)
    for k in cls:
        ix = np.flatnonzero(y == k)
        draw = ix[rng.integers(0, len(ix), (n_boot, len(ix)))]
        d += a[draw].mean(1) - b[draw].mean(1)
    d /= len(cls)
    return _bal(y, a) - _bal(y, b), float(np.mean(d > 0)), (float(np.percentile(d, 5)), float(np.percentile(d, 95)))


def run_dir(out_root, prefix, name):
    d = Path(out_root) / (prefix + name)
    return d / "comun" if (d / "comun" / "metrics.json").exists() else d


def load_preds(out_root, prefix, name):
    f = run_dir(out_root, prefix, name) / "pred_real_val.csv"
    if not f.exists():
        raise SystemExit(f"falta {f}")
    return pd.read_csv(f, dtype={"oid": str})


def compare(out_root, prefix, cand, inc, sel, n_boot=N_BOOT, p_min=P_MIN):
    """Regla de eleccion de la cola: candidata contra incumbente sobre los objetos de val_sel."""
    a, b = load_preds(out_root, prefix, cand), load_preds(out_root, prefix, inc)
    a, b = a[a.oid.isin(sel)], b[b.oid.isin(sel)]
    m = a.merge(b, on="oid", suffixes=("_c", "_i"))
    if len(m) < MIN_COMUN:
        raise SystemExit(f"{cand} contra {inc}: solo {len(m)} objetos de val_sel en comun")
    assert (m.y_true_c == m.y_true_i).all(), f"{cand} y {inc} no tienen las mismas etiquetas"
    y = m.y_true_c.to_numpy()
    ok_c, ok_i = (m.y_pred_c == y).to_numpy(), (m.y_pred_i == y).to_numpy()
    delta, p, ci = paired_bootstrap(y, ok_c, ok_i, n_boot)
    r = {"candidata": cand, "incumbente": inc, "subconjunto": "val_sel", "n_comun": int(len(m)),
         "bal_acc_cand_sel": _bal(y, ok_c), "bal_acc_inc_sel": _bal(y, ok_i), "delta": delta, "p_mejora": p,
         "ic90_delta": list(ci), "n_boot": n_boot, "p_min": p_min, "gana": bool(p >= p_min)}
    print(f"[cola] {cand} vs {inc} (val_sel, n = {len(m)}): delta {delta:+.3f}, P(delta > 0) = {p:.3f} -> "
          f"{'gana ' + cand if r['gana'] else 'queda ' + inc}", flush=True)
    return r


# ---------------------------------------------------------------- fases 2 y 3
def _exists(out_root, prefix, name):
    return (run_dir(out_root, prefix, name) / "pred_real_val.csv").exists()


def read_config(out_root, prefix, name):
    return json.loads((Path(out_root) / (prefix + name) / "config.json").read_text())


def flags_from_config(cfg, use_z=None):
    """Flags de `python -m pipeline78.nnclf train` que reproducen una corrida (sin --fold)."""
    use_z = cfg["use_z"] if use_z is None else use_z
    return (["--model", cfg["model"], "--band-enc", cfg["band_enc"], "--time-enc", cfg["time_enc"],
             "--gru-pool", cfg["gru_pool"], "--trunc", cfg["trunc"], "--p-trunc", str(cfg.get("p_trunc", 1.0))]
            + (["--bidir"] if cfg["bidir"] else []) + (["--jerarquica"] if cfg.get("jerarquica") else [])
            + (["--four-classes"] if cfg.get("four_classes") else []) + (["--use-z"] if use_z else []))


def n_params(out_root, prefix, name):
    """Parametros entrenables (config.json; las corridas anteriores a n_params se cuentan desde model.pt)."""
    cfg = read_config(out_root, prefix, name)
    if "n_params" in cfg:
        return int(cfg["n_params"])
    import torch
    sd = torch.load(Path(out_root) / (prefix + name) / "model.pt", map_location="cpu", weights_only=True)["state_dict"]
    return int(sum(v.numel() for v in sd.values() if v.dtype.is_floating_point))


def fase2_select(out_root, prefix, sel):
    """Tanda a: cada ablacion contra su base, un ganador por slot y la combinacion si pasan dos o mas slots."""
    info = {}
    for arq, slots in ABLACIONES.items():
        base = f"{arq}_base"
        if not _exists(out_root, prefix, base):
            raise SystemExit(f"falta {prefix}{base} (correr la fase 1)")
        comps, pasan, faltan, flags = [], [], [], ["--model", ARCH[arq]]
        for slot, variants in slots.items():
            ok = []
            for n in variants:
                if not _exists(out_root, prefix, n):
                    faltan.append(n)
                    continue
                c = compare(out_root, prefix, n, base, sel)
                comps.append(c)
                if c["gana"]:
                    ok.append((c["bal_acc_cand_sel"], n))
            if ok:
                w = max(ok)[1]
                pasan.append(w)
                flags += variants[w]
        if faltan:
            print(f"[cola] {arq}: faltan {faltan}, no compiten", flush=True)
        comb = f"{arq}_comb_" + "_".join(n[len(arq) + 1:] for n in pasan) if len(pasan) >= 2 else None
        info[arq] = {"base": base, "pasan": pasan, "faltan": faltan, "comb": comb,
                     "comb_flags": flags if comb else None, "comparaciones": comps}
    return info


def fase2_comb_jobs(info):
    return [(r["comb"], "nn", r["comb_flags"]) for r in info.values() if r["comb"]]


def fase2_decide(out_root, prefix, sel, info, write=True):
    """Tanda b: la mejor de cada arquitectura, la arquitectura y la corrida <mejor>_z."""
    mejor = {}
    for arq, r in info.items():
        if not r["pasan"]:
            mejor[arq] = r["mejor"] = r["base"]
            r["motivo"] = "ninguna ablacion gana a la base"
            continue
        bal = {c["candidata"]: c["bal_acc_cand_sel"] for c in r["comparaciones"]}
        single = max(r["pasan"], key=lambda n: bal[n])
        mejor[arq], r["motivo"] = single, f"la mejor ablacion individual que gana a la base es {single}"
        if r["comb"]:
            if not _exists(out_root, prefix, r["comb"]):
                raise SystemExit(f"falta la combinacion {prefix}{r['comb']}")
            c = compare(out_root, prefix, r["comb"], single, sel)
            r["comparaciones"].append(c)
            if c["gana"]:
                mejor[arq], r["motivo"] = r["comb"], f"la combinacion gana a {single}"
            else:
                r["motivo"] += "; la combinacion no le gana"
        r["mejor"] = mejor[arq]
    npar = {arq: n_params(out_root, prefix, n) for arq, n in mejor.items()}
    orden = sorted(mejor, key=lambda k: (npar[k], k))
    inc, arq_comps = orden[0], []
    for cand in orden[1:]:
        c = compare(out_root, prefix, mejor[cand], mejor[inc], sel)
        arq_comps.append(c)
        if c["gana"]:
            inc = cand
    best = mejor[inc]
    flags = flags_from_config(read_config(out_root, prefix, best), use_z=False)
    eleccion = {"criterio": CRITERIO, "arquitecturas": info, "n_params": npar,
                "comparaciones_arquitectura": arq_comps, "arquitectura": inc, "mejor_noz": best,
                "mejor_z": f"{best}_z", "flags": flags,
                "nota": "elegido solo con val_sel; las cifras honestas son las _rep de resumen.csv"}
    if write:
        (Path(out_root) / f"{prefix}fase2_eleccion.json").write_text(json.dumps(eleccion, indent=1))
    print(f"[cola] fase 2: arquitectura {inc}, mejor sin z {best}", flush=True)
    return [(f"{best}_z", "nn", flags + ["--use-z"])]


def fase3(out_root, prefix, sel, write=True):
    f = Path(out_root) / f"{prefix}fase2_eleccion.json"
    if not f.exists():
        raise SystemExit(f"falta {f} (correr la fase 2)")
    e = json.loads(f.read_text())
    noz, z = e["mejor_noz"], e["mejor_z"]
    if not _exists(out_root, prefix, z):
        raise SystemExit(f"falta {prefix}{z} (la fase 2 no termino)")
    c = compare(out_root, prefix, z, noz, sel)
    name = z if c["gana"] else noz
    flags = flags_from_config(read_config(out_root, prefix, name))
    if write:
        (Path(out_root) / f"{prefix}fase3_eleccion.json").write_text(json.dumps(
            {"criterio": CRITERIO, "comparacion_z": c, "elegida": name, "flags": flags}, indent=1))
    jobs = [(f"{name}_f{k}", "nn", flags + ["--fold", str(k)]) for k in range(1, 5)]
    members = [prefix + name] + [f"{prefix}{name}_f{k}" for k in range(1, 5)]
    jobs.append((f"{name}_ens5", "ensemble", ["--members"] + members))
    return jobs


# ---------------------------------------------------------------- ejecucion
def command(name, kind, flags, a, prefix):
    full = prefix + name
    common = ["--out-root", a.out_root]
    data = ["--sims-dir", a.sims_dir, "--real-dir", a.real_dir]
    smoke = SMOKE.get(kind, []) if a.smoke else []
    if kind == "nn":
        return [PY, "-m", "pipeline78.nnclf", "train", "--name", full, "--device", "cpu", "--threads",
                str(a.threads)] + flags + data + common + smoke
    if kind == "snn":
        return [PY, "-m", "pipeline78.nnclf", "snn", "--name", full, "--threads", str(a.threads)] + flags + data + \
            common + smoke
    if kind == "villar":
        sf = ["--sim-feat", a.sim_feat] if a.sim_feat else []
        return [PY, "-m", "pipeline78.nnclf", "baseline", "--name", full] + flags + sf + data + common
    if kind == "ensemble":
        return [PY, "-m", "pipeline78.nnclf", "ensemble", "--name", full, "--threads", str(a.threads)] + flags + \
            ["--out-root", a.out_root] + (["--n-draws", "1"] if a.smoke else [])
    raise ValueError(kind)


def job_env(threads):
    t = str(threads)
    return {**os.environ, "OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t,
            "VECLIB_MAXIMUM_THREADS": t}


def villar_ok(a):
    """Villar necesita las features de la MISMA proyeccion que --sims-dir. Sin --sim-feat solo vale la T9 final."""
    if a.sim_feat:
        return Path(a.sim_feat).exists()
    from pipeline78.nnclf.baseline import SIM_FEAT
    return Path(a.sims_dir).resolve() == Path(D.SIM_RUN).resolve() and Path(SIM_FEAT).exists()


def done(out_root, name, kind):
    d = Path(out_root) / name
    return (d / "comun" / "metrics.json").exists() if kind == "villar" else (d / "metrics.json").exists()


def collect(out_root):
    """Rehace resumen.csv con todas las corridas del out_root que tengan el formato comun."""
    from pipeline78.nnclf.evaluate import summary_row
    rows = []
    for d in sorted(Path(out_root).iterdir()):
        mdir = d / "comun" if (d / "comun" / "metrics.json").exists() else d
        f = mdir / "metrics.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        if "coverage" not in res:                                    # corridas anteriores al formato comun
            continue
        parts = [pd.read_csv(mdir / n) for n in ("degradation.csv", "degradation_fixed.csv",
                                                  "degradation_horizon.csv") if (mdir / n).exists()]
        agg = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(agg):
            agg["N"] = agg["N"].astype(str)
        row = summary_row(d.name, res.get("method", "nn"), res, agg, res["classes"])
        row["use_z"] = res.get("use_z")
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(out_root) / "resumen.csv", index=False)
    return df


def run_jobs(jobs, a, prefix):
    logs = Path(a.out_root) / "logs"
    logs.mkdir(exist_ok=True)
    for name, kind, flags in jobs:
        cmd = command(name, kind, flags, a, prefix)
        if a.accion == "plan":
            print(" ".join(cmd) + ("   # ya esta" if done(a.out_root, prefix + name, kind) else ""))
            continue
        if done(a.out_root, prefix + name, kind) and not a.force:
            print(f"[cola] {prefix + name}: ya esta, se salta", flush=True)
            continue
        if kind == "villar" and not villar_ok(a):
            print(f"[cola] {prefix + name}: faltan las features de las sims de {a.sims_dir} (--sim-feat), se salta",
                  flush=True)
            continue
        t0 = time.time()
        print(f"[cola] {prefix + name}: {' '.join(cmd[3:])}", flush=True)
        with open(logs / f"{prefix + name}.log", "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=job_env(a.threads))
        print(f"[cola] {prefix + name}: {'ok' if r.returncode == 0 else f'FALLO ({r.returncode})'} en "
              f"{time.time() - t0:.0f} s", flush=True)
        collect(a.out_root)


RESUMEN_COLS = ["name", "method", "use_z", "n_rep", "bal_acc_rep", "f1_macro_rep", "bal_acc_sel", "bal_acc_val",
                "ece_raw_rep", "ece_ts_rep", "ece_ts_prior_rep", "bal_acc_ts_prior_rep", "sims_bal_acc"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline78.nnclf.experimentos")
    ap.add_argument("accion", choices=["plan", "run", "resumen"])
    ap.add_argument("--fase", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--sims-dir", default=str(D.SIM_RUN))
    ap.add_argument("--real-dir", default=str(D.REAL_DIR))
    ap.add_argument("--sim-feat", default=None, help="features.csv de las sims para Villar")
    ap.add_argument("--out-root", default=str(D.OUT_ROOT))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="solo estas corridas (nombres sin prefijo)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    Path(a.out_root).mkdir(parents=True, exist_ok=True)
    prefix = "smoke_" if a.smoke else ""
    if a.accion == "resumen":
        df = collect(a.out_root)
        print(df[[c for c in RESUMEN_COLS if c in df]].round(3).to_string(index=False) if len(df) else "(vacio)")
        return

    def only(jobs):
        return jobs if a.only is None else [j for j in jobs if j[0] in a.only]

    if a.fase == 1:
        run_jobs(only(FASE1 + FASE1_4C), a, prefix)
        return
    sel = sel_oids(a.real_dir)
    write = a.accion == "run"
    if a.fase == 2:
        info = fase2_select(a.out_root, prefix, sel)
        comb = fase2_comb_jobs(info)
        run_jobs(only(comb), a, prefix)
        if not all(done(a.out_root, prefix + n, k) for n, k, _ in comb):
            print("# la decision de la fase 2 y la corrida con z esperan a las combinaciones de arriba")
            return
        run_jobs(only(fase2_decide(a.out_root, prefix, sel, info, write)), a, prefix)
    else:
        run_jobs(only(fase3(a.out_root, prefix, sel, write)), a, prefix)


if __name__ == "__main__":
    main()
