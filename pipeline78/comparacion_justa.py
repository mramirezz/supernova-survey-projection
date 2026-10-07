# pipeline78/comparacion_justa.py
"""Comparacion justa en TEST: cada metodo sobre (a) las SNe que Villar alcanza y (b) todas las SNe.

En (b) Villar se cuenta de dos formas: sus no clasificadas como error (Villar solo) y como sistema con respaldo
(la red; en 4 clases, sin red en TEST, las plantillas). Delta y P del bootstrap pareado de nnclf contra Villar.

agrupacion_sudare: la agrupacion de SUDARE I (Cappellaro+2015 Sec. 4.1, verificado en el texto): la probabilidad de tipo
junta II e IIn ("we merged regular type II and type IIn templates") y despues marca IIn si la plantilla de mayor
probabilidad es IIn. Sobre las predicciones congeladas de 4 clases (sin reajustar nada): (a) 3 clases con H = II + IIn
(VAL y TEST, plantillas y Villar), (b) 4 clases a la SUDARE contra el argmax de 4 clases (plantillas; val_sel decide).

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


def met(y, yp, cls=None):
    y, yp = np.asarray(y), np.asarray(yp)
    out = dict(n=int(len(y)), acc=float(np.mean(y == yp)),
               bal=float(np.mean([np.mean(yp[y == c] == c) for c in np.unique(y)])))
    if cls:
        out.update(clases=list(cls), confusion=[[int(np.sum((y == a) & (yp == b))) for b in cls] for a in cls],
                   recall={c: float(np.mean(yp[y == c] == c)) for c in cls if (y == c).any()})
    return out


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
    if "red" in P and "plantillas" in P:                      # las dos que clasifican casi todo, cara a cara
        for g, ix in (("villar_alcanza", V), ("total", U)):
            a, b = P["plantillas"][0], P["red"][0]
            c = ix.intersection(a.index).intersection(b.index)
            y = a.loc[c, "y_true"].to_numpy()
            d, pr, ci = paired_bootstrap(y, (a.loc[c, "y_pred"] == y).to_numpy(), (b.loc[c, "y_pred"] == y).to_numpy())
            out[g]["plantillas"].update(n_vs_red=int(len(c)), delta_vs_red=d, P_mejor_que_red=pr)
    return out


H3 = ("Ia", "H", "Ibc")
C4 = ("Ia", "II", "Ibc", "IIn")


def a_sudare(p):
    """(clase real con H = II + IIn, pred de 3 clases con P(H) = P(II) + P(IIn), pred de 4 clases a la SUDARE o None
    si no hay plantilla de mayor probabilidad)."""
    y3 = p.y_true.replace({"II": "H", "IIn": "H"})
    P = pd.DataFrame({"Ia": p.p_Ia, "H": p.p_II + p.p_IIn, "Ibc": p.p_Ibc})
    p3 = P.idxmax(axis=1)
    p4 = (p3.where(p3 != "H", np.where(p.best_template_clase == "IIn", "IIn", "II"))
          if "best_template_clase" in p else None)
    return y3, p3, p4


def leer4(f):
    return pd.read_csv(f, dtype={"oid": str}).drop_duplicates("oid").set_index("oid")


def agrupacion_sudare():
    pv = leer4(RUNS / "plantillas_clf" / "plantillas_t11_4c" / "pred_real_val.csv")
    pt, vt = leer4(T / "plantillas_4c" / "pred_test.csv"), leer4(T / "villar_4c" / "pred_test.csv")
    out = {"h3": {}, "sudare4": {}}
    for sub, d in (("val_sel", pv[pv.subset == "val_sel"]), ("val_rep", pv[pv.subset == "val_rep"]), ("test", pt)):
        y3, p3, p4 = a_sudare(d)
        out["h3"].setdefault(sub, {})["plantillas"] = met(y3, p3, H3)
        y = d.y_true.to_numpy()
        a, b = (p4.to_numpy() == y), (d.y_pred.to_numpy() == y)
        dl, pr, ci = paired_bootstrap(y, a, b)
        out["sudare4"][sub] = dict(argmax=met(y, d.y_pred, C4), sudare=met(y, p4, C4), delta_sudare_menos_argmax=dl,
                                   P_sudare_mejor=pr, ic90=ci)
    yv3, pv3, _ = a_sudare(vt)
    V = vt.index
    yp3, pp3, _ = a_sudare(pt)
    c = V.intersection(pt.index)
    t = out["h3"]["test"]
    t["villar_alcanza"] = {"villar": met(yv3.loc[c], pv3.loc[c], H3), "plantillas": met(yp3.loc[c], pp3.loc[c], H3)}
    y = yp3.loc[c].to_numpy()
    dl, pr, ci = paired_bootstrap(y, (pp3.loc[c] == y).to_numpy(), (pv3.loc[c] == y).to_numpy())
    t["villar_alcanza"]["plantillas"].update(delta_vs_villar=dl, P_mejor_que_villar=pr)
    h = pp3.copy()
    h.loc[c] = pv3.loc[c]
    t["villar_mas_respaldo"] = met(yp3, h, H3)
    y = yp3.to_numpy()
    dl, pr, ci = paired_bootstrap(y, (pp3 == y).to_numpy(), (h == y).to_numpy())
    t["plantillas"].update(delta_vs_villar_mas_respaldo=dl, P_mejor=pr)
    return out


def main():
    res = {k: version(k, r) for k, r in VERSIONES.items()}
    S = agrupacion_sudare()
    (T / "agrupacion_sudare.json").write_text(json.dumps(S, indent=1))
    for sub, x in S["sudare4"].items():
        print(f"== 4 clases {sub}: argmax bal {x['argmax']['bal']:.3f} recall {x['argmax']['recall']} | a la SUDARE bal "
              f"{x['sudare']['bal']:.3f} recall {x['sudare']['recall']} | P(SUDARE mejor) {x['P_sudare_mejor']:.3f}")
    for sub, x in S["h3"].items():
        print(f"== H = II + IIn {sub}: " + " ".join(f"{m} n {v['n']} acc {v['acc']:.3f} bal {v['bal']:.3f}"
                                                 for m, v in x.items() if "n" in v))
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
