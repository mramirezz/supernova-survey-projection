"""Cola de experimentos del clasificador NN con las mejoras de la literatura (nn-lit-brief, sec. 4).

Una corrida a la vez, en CPU con 2 threads, cada una en su propio proceso (python -m pipeline78.nnclf ...). Cada
corrida deja metrics.json en <out_root>/<nombre> y la cola reescribe <out_root>/resumen.csv despues de cada una. Es
reanudable: la corrida que ya tiene metrics.json se salta (salvo --force).

FASES
1. Base y ablaciones de una sola cosa, sin z, fold 0:
   gru_base, tf_base, gru_lambda, tf_lambda (banda por lambda pivote), gru_bidir (control), gru_attnpool (BiGRU +
   attention pooling, ORACLE-2), tf_timemod (TimeModulator de ATAT), gru_trunc, tf_trunc (truncamiento ORACLE-2,
   modo "both"), snn_noz, snn_z y snn_var_noz (SuperNNova estandar y variational, 50 muestras MC por defecto),
   villar_noz y villar_z (si estan las features de las sims).
2. Mejor combinacion, sin y con z: por arquitectura se juntan los flags cuya ablacion supero a su base en exactitud
   balanceada sobre val_real (celda principal), y se elige la arquitectura cuya combinacion (o base, si ningun flag
   ayudo) rinde mas. Se entrenan <arq>_best_noz y <arq>_best_z. La eleccion es sobre val_real, asi que esos numeros
   quedan optimistas (revision B7): la cifra de la tesis sale de la mitad final.
3. Folds 1 a 4 de la mejor corrida de la fase 2 (por exactitud balanceada, con o sin z) y el ensemble de los 5.

USO (desde la raiz del repo, env series)
    python -m pipeline78.nnclf.experimentos plan  --fase 1            # imprime los comandos
    python -m pipeline78.nnclf.experimentos run   --fase 1 [--sims-dir ...] [--sim-feat ...] [--smoke]
    python -m pipeline78.nnclf.experimentos resumen                   # rehace resumen.csv
--smoke: 1 epoca, 1500 sims, 1 sorteo, nombres con prefijo smoke_. No usar para resultados.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
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
    ("snn_noz", "snn", []),
    ("snn_z", "snn", ["--use-z"]),
    ("snn_var_noz", "snn", ["--snn-model", "variational"]),
    ("villar_noz", "villar", []),
    ("villar_z", "villar", ["--use-z"]),
]
# flags de cada ablacion para combinar en la fase 2 (gru_bidir es solo el control de gru_attnpool)
ABLACIONES = {"gru": {"gru_lambda": ["--band-enc", "lambda"], "gru_attnpool": ["--bidir", "--gru-pool", "attn"],
                      "gru_trunc": ["--trunc", "both"]},
              "tf": {"tf_lambda": ["--band-enc", "lambda"], "tf_timemod": ["--time-enc", "atat"],
                     "tf_trunc": ["--trunc", "both"]}}
ARCH = {"gru": "gru", "tf": "transformer"}
SMOKE = {"nn": ["--max-epochs", "1", "--max-sims", "1500", "--n-draws", "1"],
         "snn": ["--nb-epoch", "1", "--max-sims", "1500", "--n-draws", "1", "--num-inference-samples", "5"],
         "villar": []}


def read_resumen(out_root):
    f = Path(out_root) / "resumen.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def _bal(res, name):
    r = res[res.name == name]
    return float(r.bal_acc.iloc[0]) if len(r) and pd.notna(r.bal_acc.iloc[0]) else None


def fase2(out_root, prefix=""):
    res = read_resumen(out_root)
    best = None
    for arq, abl in ABLACIONES.items():
        base = _bal(res, f"{prefix}{arq}_base")
        if base is None:
            raise SystemExit(f"falta {prefix}{arq}_base en resumen.csv (correr la fase 1)")
        flags = ["--model", ARCH[arq]]
        ayudan = []
        for n, fl in abl.items():
            b = _bal(res, prefix + n)
            if b is not None and b > base:
                flags += fl
                ayudan.append(n)
        score = max([base] + [_bal(res, prefix + n) for n in ayudan])
        print(f"[cola] {arq}: base {base:.3f}, ayudan {ayudan or 'ninguna'}", flush=True)
        if best is None or score > best[0]:
            best = (score, arq, flags, ayudan)
    _, arq, flags, ayudan = best
    (Path(out_root) / f"{prefix}fase2_eleccion.json").write_text(json.dumps(
        {"arquitectura": arq, "flags": flags, "ablaciones_que_ayudan": ayudan}, indent=1))
    return [(f"{arq}_best_noz", "nn", flags), (f"{arq}_best_z", "nn", flags + ["--use-z"])]


def fase3(out_root, prefix=""):
    res = read_resumen(out_root)
    cand = [n for n in res.name if n.startswith(prefix) and "_best_" in n and n.endswith(("_noz", "_z"))]
    if not cand:
        raise SystemExit("faltan las corridas *_best_* de la fase 2")
    name = max(cand, key=lambda n: _bal(res, n) or -1)[len(prefix):]
    cfg = json.loads((Path(out_root) / (prefix + name) / "config.json").read_text())
    flags = ["--model", cfg["model"]]
    flags += ["--band-enc", cfg["band_enc"], "--time-enc", cfg["time_enc"], "--gru-pool", cfg["gru_pool"],
              "--trunc", cfg["trunc"]] + (["--bidir"] if cfg["bidir"] else []) + (["--use-z"] if cfg["use_z"] else [])
    jobs = [(f"{name}_f{k}", "nn", flags + ["--fold", str(k)]) for k in range(1, 5)]
    members = [prefix + name] + [f"{prefix}{name}_f{k}" for k in range(1, 5)]
    jobs.append((f"{name}_ens5", "ensemble", ["--members"] + members))
    return jobs


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
        parts = [pd.read_csv(mdir / n) for n in ("degradation.csv", "degradation_fixed.csv") if (mdir / n).exists()]
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
        print(collect(a.out_root).round(3).to_string(index=False))
        return
    jobs = {1: lambda: FASE1, 2: lambda: fase2(a.out_root, prefix), 3: lambda: fase3(a.out_root, prefix)}[a.fase]()
    if a.only is not None:
        jobs = [j for j in jobs if j[0] in a.only]
    logs = Path(a.out_root) / "logs"
    logs.mkdir(exist_ok=True)
    for name, kind, flags in jobs:
        cmd = command(name, kind, flags, a, prefix)
        if a.accion == "plan":
            print(" ".join(cmd))
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
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[cola] {prefix + name}: {'ok' if r.returncode == 0 else f'FALLO ({r.returncode})'} en "
              f"{time.time() - t0:.0f} s", flush=True)
        collect(a.out_root)


if __name__ == "__main__":
    main()
