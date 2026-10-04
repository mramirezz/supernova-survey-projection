"""CLI: python -m pipeline78.nnclf train|eval|baseline. Correr desde la raiz del repo proyeccion con el env series."""
import argparse
from dataclasses import fields
from pathlib import Path
from pipeline78.nnclf import data as D


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline78.nnclf")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="entrena con las sims T9 y evalua sobre las reales de validacion")
    t.add_argument("--name", required=True)
    t.add_argument("--model", choices=["gru", "transformer"], default="transformer")
    t.add_argument("--use-z", action="store_true", help="z y M_ref como features globales")
    t.add_argument("--four-classes", action="store_true", help="Ia, II, Ibc, IIn")
    t.add_argument("--no-magerr", action="store_true", help="anula la columna magerr de los tokens")
    t.add_argument("--max-epochs", type=int, default=60)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", default="auto", help="auto | cpu | mps")
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--max-sims", type=int, default=0, help="submuestra estratificada (smoke). 0 = todas")
    t.add_argument("--n-folds", type=int, default=5)
    t.add_argument("--fold", type=int, default=0)
    t.add_argument("--seed", type=int, default=D.SEED)
    t.add_argument("--p-thin", type=float, default=0.8)
    t.add_argument("--p-ronly", type=float, default=0.5)
    t.add_argument("--no-eval", action="store_true")
    t.add_argument("--n-draws", type=int, default=5, help="sorteos por celda de degradacion")
    t.add_argument("--out-root", default=str(D.OUT_ROOT))

    e = sub.add_parser("eval", help="evalua un modelo ya entrenado")
    e.add_argument("--name", required=True)
    e.add_argument("--n-draws", type=int, default=5)
    e.add_argument("--device", default=None)
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--out-root", default=str(D.OUT_ROOT))

    b = sub.add_parser("baseline", help="Villar SPM + gradient boosting sobre las mismas reales de validacion")
    b.add_argument("--name", default=None)
    b.add_argument("--use-z", action="store_true")
    b.add_argument("--four-classes", action="store_true")
    b.add_argument("--n-folds", type=int, default=5)
    b.add_argument("--fold", type=int, default=0)
    b.add_argument("--seed", type=int, default=D.SEED)
    b.add_argument("--nn-run", default=None, help="corrida NN para comparar sobre las mismas oids")
    b.add_argument("--out-root", default=str(D.OUT_ROOT))

    a = ap.parse_args(argv)
    if a.cmd == "train":
        from pipeline78.nnclf.train import Config, train
        from pipeline78.nnclf.evaluate import run_eval
        kw = {f.name: getattr(a, f.name) for f in fields(Config) if hasattr(a, f.name)}
        cfg = Config(**kw, use_magerr=not a.no_magerr)
        out = train(cfg)
        if not a.no_eval:
            run_eval(out, a.n_draws, threads=a.threads)
    elif a.cmd == "eval":
        from pipeline78.nnclf.evaluate import run_eval
        run_eval(Path(a.out_root) / a.name, a.n_draws, device=a.device, threads=a.threads)
    else:
        from pipeline78.nnclf.baseline import run_baseline
        run_baseline(a.name, a.use_z, a.four_classes, a.n_folds, a.fold, a.seed, a.nn_run, out_root=a.out_root)


if __name__ == "__main__":
    main()
