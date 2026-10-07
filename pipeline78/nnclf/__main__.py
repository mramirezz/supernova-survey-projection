"""CLI: python -m pipeline78.nnclf train|eval|baseline|ensemble|snn. Correr desde la raiz del repo proyeccion con el
env series. La cola de experimentos esta en pipeline78/nnclf/experimentos.py."""
import argparse
from dataclasses import fields
from pathlib import Path
from pipeline78.nnclf import data as D


def _data_args(p):
    p.add_argument("--sims-dir", dest="sim_run", default=str(D.SIM_RUN), help="proyeccion (run) de las sims")
    p.add_argument("--real-dir", dest="real_dir", default=str(D.REAL_DIR))
    p.add_argument("--out-root", default=str(D.OUT_ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline78.nnclf")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="entrena con las sims y evalua sobre las reales de validacion")
    t.add_argument("--name", required=True)
    t.add_argument("--model", choices=["gru", "transformer"], default="transformer")
    t.add_argument("--use-z", action="store_true", help="z y M_ref como features globales")
    t.add_argument("--four-classes", action="store_true", help="Ia, II, Ibc, IIn")
    t.add_argument("--five-classes", action="store_true", help="Ia, II, IIb, Ibc, IIn (IIb propia)")
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
    t.add_argument("--band-enc", choices=D.BAND_ENCODINGS, default="onehot", help="lambda = longitud de onda pivote")
    t.add_argument("--time-enc", choices=["sin", "atat"], default="sin", help="atat = TimeModulator de ATAT")
    t.add_argument("--tm-harmonics", type=int, default=64)
    t.add_argument("--tm-tmax", type=float, default=1500.0)
    t.add_argument("--gru-pool", choices=["last", "attn"], default="last", help="attn = attention pooling ORACLE-2")
    t.add_argument("--bidir", action="store_true", help="GRU bidireccional (ORACLE-2)")
    t.add_argument("--trunc", choices=D.TRUNC_MODES, default="none", help="truncamiento ORACLE-2")
    t.add_argument("--p-trunc", type=float, default=1.0)
    t.add_argument("--jerarquica", action="store_true", help="cabezas Ia contra CC y subclase | CC (models.HierHead)")
    t.add_argument("--no-eval", action="store_true")
    t.add_argument("--n-draws", type=int, default=5, help="sorteos por celda de degradacion")
    _data_args(t)

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
    b.add_argument("--sim-feat", default=None, help="features.csv de las sims (default: el de la T9 final)")
    b.add_argument("--real-feat", default=None, help="features.csv de las reales")
    _data_args(b)

    en = sub.add_parser("ensemble", help="promedia las probabilidades de varias corridas (los 5 folds)")
    en.add_argument("--name", required=True)
    en.add_argument("--members", nargs="+", required=True)
    en.add_argument("--n-draws", type=int, default=5)
    en.add_argument("--threads", type=int, default=2)
    en.add_argument("--out-root", default=str(D.OUT_ROOT))

    s = sub.add_parser("snn", help="baseline SuperNNova (venv ~/venvs/snn)")
    s.add_argument("--name", required=True)
    s.add_argument("--use-z", action="store_true")
    s.add_argument("--four-classes", action="store_true")
    s.add_argument("--n-folds", type=int, default=5)
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--seed", type=int, default=D.SEED)
    s.add_argument("--max-sims", type=int, default=0)
    s.add_argument("--nb-epoch", type=int, default=90)
    s.add_argument("--snn-model", choices=["vanilla", "variational", "bayesian"], default="vanilla")
    s.add_argument("--balance", choices=["subsample", "none"], default="subsample")
    s.add_argument("--n-draws", type=int, default=5)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--num-inference-samples", type=int, default=None)
    _data_args(s)

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
    elif a.cmd == "baseline":
        from pipeline78.nnclf.baseline import run_baseline, SIM_FEAT, REAL_FEAT
        run_baseline(a.name, a.use_z, a.four_classes, a.n_folds, a.fold, a.seed, a.nn_run,
                     sim_feat=a.sim_feat or SIM_FEAT, real_feat=a.real_feat or REAL_FEAT, run_dir=Path(a.sim_run),
                     real_dir=Path(a.real_dir), out_root=a.out_root)
    elif a.cmd == "ensemble":
        from pipeline78.nnclf.ensemble import run_ensemble
        run_ensemble(a.members, a.name, a.out_root, a.n_draws, a.threads)
    else:
        from pipeline78.nnclf.snn import run_snn
        run_snn(a.name, a.use_z, a.four_classes, a.n_folds, a.fold, a.seed, a.max_sims, a.nb_epoch, a.snn_model,
                a.balance, a.n_draws, a.threads, Path(a.sim_run), Path(a.real_dir), a.out_root,
                a.num_inference_samples)


if __name__ == "__main__":
    main()
