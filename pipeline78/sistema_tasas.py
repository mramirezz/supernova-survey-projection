# pipeline78/sistema_tasas.py
"""Regla de eleccion del clasificador para las tasas (Mauricio 2026-10-04, fijada antes de ver val_rep de plantillas).

Cada metodo se evalua como SISTEMA COMPLETO sobre todas las SNe de val (3 clases): las que el metodo no cubre se
asignan con un respaldo fijado a priori (la red elegida). Metricas por subconjunto (val_sel decide, val_rep confirma):
  - exactitud balanceada y exactitud;
  - error de las fracciones por clase, lo que entra a las tasas: L1 = sum_k |f_pred,k - f_real,k|, contando argmax y
    sumando probabilidades (p del metodo, o del respaldo donde no cubre).
Comparaciones pareadas entre sistemas: delta de exactitud balanceada (bootstrap estratificado por clase de nnclf) y
delta de L1 (bootstrap simple de SNe: las fracciones dependen de la mezcla). Regla: gana el que mejora con P >= 0.9 en
val_sel; si empatan, el de mayor cobertura propia; despues el que depende menos de la simulacion.

    python -m pipeline78.sistema_tasas [--red tf_base] [--plantillas plantillas_t11] [--villar sweep_t11_foco]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import RUNS
from pipeline78.nnclf.experimentos import paired_bootstrap, P_MIN

CLS = ("Ia", "II", "Ibc")
OUT = RUNS / "informe_clasificadores"


def leer(f):
    d = pd.read_csv(f, dtype={"oid": str})
    d = d[d.y_true.isin(CLS)].drop_duplicates("oid").set_index("oid")
    return d[["subset", "y_true", "y_pred"] + [f"p_{c}" for c in CLS]]


def sistema(prim, resp):
    """Predicciones del sistema sobre las oids del respaldo: las del metodo donde cubre, las del respaldo si no."""
    s = resp.copy()
    s["propia"] = s.index.isin(prim.index)
    com = s.index[s.propia]
    s.loc[com, ["y_pred"] + [f"p_{c}" for c in CLS]] = prim.loc[com, ["y_pred"] + [f"p_{c}" for c in CLS]].values
    return s


def fracciones(d):
    real = np.array([(d.y_true == c).mean() for c in CLS])
    arg = np.array([(d.y_pred == c).mean() for c in CLS])
    p = d[[f"p_{c}" for c in CLS]].to_numpy(float)
    prob = (p / p.sum(1, keepdims=True)).mean(0)
    return real, arg, prob


def metricas(d):
    y, yp = d.y_true.to_numpy(), d.y_pred.to_numpy()
    real, arg, prob = fracciones(d)
    return dict(n=int(len(d)), cobertura_propia=float(d.propia.mean()), acc=float(np.mean(y == yp)),
                bal_acc=float(np.mean([np.mean(yp[y == c] == c) for c in CLS])),
                frac_real=dict(zip(CLS, real.round(4))), frac_argmax=dict(zip(CLS, arg.round(4))),
                frac_prob=dict(zip(CLS, prob.round(4))),
                L1_argmax=float(np.abs(arg - real).sum()), L1_prob=float(np.abs(prob - real).sum()))


def l1_boot(a, b, n_boot=2000, seed=20261004, col="argmax"):
    """P(L1_a < L1_b) re-sorteando SNe (las mismas en los dos sistemas)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    ya = a.y_true.to_numpy()
    def l1(d, ix):
        real = np.array([(ya[ix] == c).mean() for c in CLS])
        if col == "argmax":
            f = np.array([(d.y_pred.to_numpy()[ix] == c).mean() for c in CLS])
        else:
            p = d[[f"p_{c}" for c in CLS]].to_numpy(float)[ix]
            f = (p / p.sum(1, keepdims=True)).mean(0)
        return np.abs(f - real).sum()
    d = np.array([l1(a, ix) - l1(b, ix) for ix in (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(np.mean(d < 0)), (float(np.percentile(d, 5)), float(np.percentile(d, 95)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", default="tf_base")
    ap.add_argument("--plantillas", default="plantillas_t11")
    ap.add_argument("--villar", default="sweep_t11_foco")
    a = ap.parse_args(argv)
    red = leer(RUNS / "nnclf_t11" / a.red / "pred_real_val.csv")
    S = {"red": sistema(red, red),
         "villar": sistema(leer(RUNS / "clf_villar" / a.villar / "mejor" / "pred_real_val.csv"), red),
         "plantillas": sistema(leer(RUNS / "plantillas_clf" / a.plantillas / "pred_real_val.csv"), red)}
    res = {"regla": __doc__.split("\n\n")[0], "respaldo": a.red, "metodos": {"villar": a.villar,
           "plantillas": a.plantillas, "red": a.red}, "por_subconjunto": {}, "pares": {}}
    for sub in ("val_sel", "val_rep"):
        res["por_subconjunto"][sub] = {k: metricas(d[d.subset == sub]) for k, d in S.items()}
        for i, x in enumerate(S):
            for z in list(S)[i + 1:]:
                A, B = S[x][S[x].subset == sub], S[z][S[z].subset == sub]
                assert (A.index == B.index).all() and (A.y_true == B.y_true).all()
                y = A.y_true.to_numpy()
                dlt, p, ci = paired_bootstrap(y, (A.y_pred == y).to_numpy(), (B.y_pred == y).to_numpy())
                pa, cia = l1_boot(A, B, col="argmax")
                pp, cip = l1_boot(A, B, col="prob")
                res["pares"].setdefault(sub, {})[f"{x}_vs_{z}"] = dict(
                    delta_bal=dlt, P_bal=p, ic90_bal=ci, P_L1_argmax_menor=pa, ic90_dL1_argmax=cia,
                    P_L1_prob_menor=pp, ic90_dL1_prob=cip, gana_bal=(x if p >= P_MIN else z if p <= 1 - P_MIN else None))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sistema_tasas.json").write_text(json.dumps(res, indent=1, default=float))
    for sub, m in res["por_subconjunto"].items():
        print(sub, pd.DataFrame(m).T[["n", "cobertura_propia", "acc", "bal_acc", "L1_argmax", "L1_prob"]].round(3).to_string())
    for sub, pr in res["pares"].items():
        for k, v in pr.items():
            print(sub, k, f"dbal {v['delta_bal']:+.3f} P {v['P_bal']:.3f} | P(L1 menor) argmax {v['P_L1_argmax_menor']:.3f} prob {v['P_L1_prob_menor']:.3f}")
    return res


if __name__ == "__main__":
    main()
