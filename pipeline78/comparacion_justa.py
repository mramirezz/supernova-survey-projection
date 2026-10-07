# pipeline78/comparacion_justa.py
"""Comparacion justa en TEST: cada metodo sobre (a) las SNe que Villar alcanza y (b) todas las SNe.

En (b) Villar se cuenta de dos formas: sus no clasificadas como error (Villar solo) y como sistema con respaldo
(la red; en 4 clases, sin red en TEST, las plantillas). Delta y P del bootstrap pareado de nnclf contra Villar.

    python -m pipeline78.comparacion_justa
"""
import json
import numpy as np
import pandas as pd
from pipeline78.paths import RUNS
from pipeline78.nnclf.experimentos import paired_bootstrap

T, C = RUNS / "test_final", RUNS / "cinco_clases"
VERSIONES = {3: dict(villar=T / "villar", red=T / "red", plantillas=T / "plantillas"),
             4: dict(villar=T / "villar_4c", plantillas=T / "plantillas_4c"),
             5: dict(villar=C / "villar_5c", red=C / "red_5c", plantillas=C / "plantillas_5c")}


def leer(d):
    p = pd.read_csv(d / "pred_test.csv", dtype={"oid": str}).drop_duplicates("oid").set_index("oid")
    return p[["y_true", "y_pred"]], json.loads((d / "metrics.json").read_text())


def met(y, yp):
    y, yp = np.asarray(y), np.asarray(yp)
    return dict(n=int(len(y)), acc=float(np.mean(y == yp)),
                bal=float(np.mean([np.mean(yp[y == c] == c) for c in np.unique(y)])))


def version(k, rutas):
    P = {m: leer(d) for m, d in rutas.items()}
    N = P["villar"][1]["test"]["n_real"] if "test" in P["villar"][1] else P["villar"][1]["n_real"]
    vi = P["villar"][0]
    resp = "red" if "red" in P else "plantillas"
    otros = [m for m in P if m != "villar"]
    U = P[resp][0].index                                     # todas las SNe que clasifica el respaldo
    out = {"N": int(N), "respaldo": resp, "villar_alcanza": {}, "total": {}}
    V = vi.index
    for m in P:
        p = P[m][0].loc[P[m][0].index.intersection(V)]
        out["villar_alcanza"][m] = met(p.y_true, p.y_pred)
        if m != "villar":
            a, b = p, vi.loc[p.index]
            y = a.y_true.to_numpy()
            d, pr, ci = paired_bootstrap(y, (a.y_pred == y).to_numpy(), (b.y_pred == y).to_numpy())
            out["villar_alcanza"][m].update(delta_vs_villar=d, P_mejor_que_villar=pr)
    tv = P[resp][0].copy()
    tv.loc[V.intersection(U), "y_pred"] = vi.loc[V.intersection(U), "y_pred"]
    out["total"]["villar_mas_respaldo"] = met(tv.y_true, tv.y_pred)
    sol = P[resp][0][["y_true"]].copy()
    sol["y_pred"] = "sin_clase"
    sol.loc[V.intersection(U), "y_pred"] = vi.loc[V.intersection(U), "y_pred"]
    out["total"]["villar_solo"] = met(sol.y_true, sol.y_pred)
    for m in otros:
        p = P[m][0].loc[P[m][0].index.intersection(U)]
        out["total"][m] = met(p.y_true, p.y_pred)
        y = p.y_true.to_numpy()
        b = tv.loc[p.index]
        d, pr, ci = paired_bootstrap(y, (p.y_pred == y).to_numpy(), (b.y_pred == y).to_numpy())
        out["total"][m].update(delta_vs_villar_mas_respaldo=d, P_mejor=pr)
    return out


def main():
    res = {k: version(k, r) for k, r in VERSIONES.items()}
    (T / "comparacion_justa.json").write_text(json.dumps(res, indent=1))
    for k, r in res.items():
        print(f"== {k} clases (N {r['N']}, respaldo {r['respaldo']})")
        for g in ("villar_alcanza", "total"):
            for m, x in r[g].items():
                ex = " ".join(f"{a} {b:+.3f}" if "delta" in a else f"{a} {b:.3f}" for a, b in x.items() if a not in ("n", "acc", "bal"))
                print(f"  {g:15s} {m:20s} n {x['n']:4d} acc {x['acc']:.3f} bal {x['bal']:.3f} {ex}")
    return res


if __name__ == "__main__":
    main()
