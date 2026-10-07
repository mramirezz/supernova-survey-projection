"""Informe de los clasificadores ZTF: Villar (oficial), las redes y las plantillas, pagina del atlas generada desde los runs.

POR QUE. Mauricio pidio un HTML con la comparacion y conclusiones en lenguaje simple (2026-10-04). Todo numero de la
pagina sale de los archivos de los runs al construirla (regla numeros-tesis-con-script): nada esta escrito a mano. Las
conclusiones son plantillas con condiciones explicitas, llenadas con los mismos numeros (numeros.json).

ENTRADA. pipeline78/informe_clasificadores.json (versionado): bloque "actual" (produccion final: raiz nnclf, barrido
clf_villar, gap, plantillas = {name, dir} de pipeline78.plantillas_clf) y "historia" (runs del modelo de observacion
viejo). Si falta un run del bloque actual la pagina dice
"pendiente" y sigue. --actual-nn / --actual-villar / --actual-gap reemplazan el bloque actual (para probar la pagina con
runs viejos; la pagina lo marca en un aviso).

REGLAS
1. Reales: SOLO la mitad val (splits.read_val_meta: las filas final no llegan a pandas), clases Ia, II (= II + IIb) e
   Ibc, partida en val_sel y val_rep con splits.val_split. La columna subset sale de la particion, no del archivo; si el
   archivo trae otra, es un error. Una prediccion con oid fuera de la mitad val actual se descarta y se cuenta.
2. Metricas desde pred_real_val.csv de cada run (y_pred = argmax, prior none en Villar): clf_villar.metrics; IC 95 %
   bootstrap = clf_villar.bootstrap_ci (1000 remuestreos, semilla 20261004) en el orden del archivo. Asi el IC de Villar
   reproduce el de su metrics.json; cada cifra se compara con la del metrics.json del run (seccion verificacion).
   Cobertura = reales del subconjunto con prediccion / reales del subconjunto.
3. Mejor Villar = la configuracion elegida por el barrido (mejor.json). Mejor red = entre las redes (nn, nn_ensemble,
   supernnova) la de mayor exactitud balanceada en val_sel contra la incumbente simple (nn_incumbente, gru_base) con el
   bootstrap pareado de nnclf (experimentos.paired_bootstrap, P >= 0.9 en val_sel); si no gana, queda la incumbente.
4. Mismos objetos: oids de val_rep que Villar clasifica. Hibrido fijo a priori: Villar si cubre, si no la mejor red.
   Las decisiones (que metodo conviene) se leen en val_sel con la misma regla. val_rep no decide, pero cada decision
   dice si val_rep la confirma, la contradice o no la confirma (signo del delta y si su IC 90 % excluye 0), y si
   val_rep separa a dos metodos que val_sel no separa. La meta se mira solo para el metodo que recomienda val_sel; las
   tres cifras se listan en orden fijo con su cobertura, sin elegir la mayor de val_rep.
5. Detecciones: n_det de la mejor red (detecciones g + r de la curva limpia); sin prediccion de la red = menos de 3.
6. Calibracion: el piloto de calib_obs se compara campo a campo con las sims de produccion que espera la config
   (sims_nn, sims_villar). Si difieren, la columna y la conclusion dicen "piloto" con sus valores y piden repetir
   confirm. La figura anterior a las tablas se marca como vieja.
7. Plantillas (tercer metodo, ajuste bayesiano como SUDARE I): la corrida del bloque "plantillas" ({name, dir}) de
   pipeline78.plantillas_clf, con su configuracion congelada (UL previos a priori, error del modelo elegido en
   val_sel): no hay eleccion entre corridas en la pagina. Metricas con la regla 2 y verificacion contra su
   metrics.json. Mismos objetos contra Villar (oids de Villar) y contra la mejor red (oids comunes), con la regla de la
   4: val_sel decide, val_rep contrasta. Exactitud en las SNe que Villar cubre y en las que no, sobre las oids que
   clasifican las plantillas y la red. No entra al hibrido ni a la recomendacion de la regla 4 (fijados antes de
   tenerlas).

USO (desde la raiz del repo)
    PYTHONPATH=. $PY -m pipeline78.informe_clasificadores
    PYTHONPATH=. $PY -m pipeline78.informe_clasificadores --actual-nn nnclf_x2 --actual-villar sweep_x2_mitig
Salidas: <pagina>/index.html + figuras (png y pdf) + numeros.json, copia en <atlas> (curl a <url>), y numeros.json en
RUNS/informe_clasificadores/.
"""
import argparse, datetime, html, json, re, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pipeline78 import splits
from pipeline78.clf_villar import bootstrap_ci, metrics
from pipeline78.nnclf.experimentos import BOOT_SEED, N_BOOT, P_MIN, paired_bootstrap
from pipeline78.paths import PHD, REPO, RUNS

CFG = REPO / "pipeline78" / "informe_clasificadores.json"
CLS = ("Ia", "II", "Ibc")
IX = {c: i for i, c in enumerate(CLS)}
REDES = ("nn", "nn_ensemble", "supernnova")
N_BOOT_IC = 1000
SUB = ("val_sel", "val_rep")
N_MIN_BIN = 5                                                # en la figura por detecciones no se grafica una exactitud con menos
COL = {"Ia": "tab:blue", "II": "tab:green", "Ibc": "tab:red", "villar": "0.15", "red": "tab:purple",
       "hib": "tab:orange", "pl": "tab:brown"}
MODELOS_V = {"hgb": "HistGradientBoosting", "hgb_lento": "HistGradientBoosting lento", "rf": "Random Forest",
             "mlp": "MLP", "ens_hier": "ensemble jerarquico"}


# ------------------------------------------------------------------------------------------------ utilidades
def _p(x, base):
    if x is None:
        return None
    q = Path(str(x)).expanduser()
    return q if q.is_absolute() else Path(base) / q


def _get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _js(o):
    """JSON limpio: numpy a python, NaN a null, claves con _ (tablas internas) fuera."""
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def ok(x):
    return isinstance(x, dict) and x.get("estado") == "ok"


def f3(x, d=3):
    return "&mdash;" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pc(x):
    return "&mdash;" if x is None else f"{100 * x:.1f} %"


def ci(v):
    return "" if not v else f" [{v[0]:.3f}, {v[1]:.3f}]"


def fnum(x):
    return f3(x) if x is None or abs(x) < 10 else f3(x, 1)


def tabla(head, rows, cls_="t"):
    """Celdas que empiezan con un numero van a la derecha (clase n)."""
    num = lambda c: " class='n'" if re.match(r"^\s*([+\-]?\d|&mdash;|&minus;)", str(c)) else ""
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td{num(c)}>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls_}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def git_info():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode != 0
        return {"commit": h, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


# ------------------------------------------------------------------------------------------------ reales val
def particion(real_dir):
    """Mitad val (regla 1): oid, cls, subset. Solo las filas val llegan a pandas."""
    v = splits.read_val_meta(Path(real_dir) / "meta_real_ztf.csv")
    sel, rep = splits.val_split(v)
    v["oid"] = v.oid.astype(str)
    v["cls"] = v.sn_type.map(splits.CLASS_OF)
    sub = {**{o: "val_sel" for o in sel}, **{o: "val_rep" for o in rep}}
    v["subset"] = v.oid.map(sub)
    return v.loc[v.cls.isin(CLS), ["oid", "cls", "subset"]].reset_index(drop=True)


def leer_pred(f, V):
    """pred_real_val.csv -> filas de las clases principales con la subset de la particion (regla 1). Devuelve
    (tabla, info). Error si la subset o la clase del archivo contradicen la particion."""
    p = pd.read_csv(f, dtype={"oid": str})
    p = p[p.y_true.isin(CLS)]
    m = p.merge(V.rename(columns={"subset": "_sub", "cls": "_cls"}), on="oid", how="left")
    fuera = int(m._sub.isna().sum())
    m = m[m._sub.notna()]
    if "subset" in m and (m.subset != m._sub).any():
        raise ValueError(f"{f}: la columna subset no coincide con splits.val_split")
    if (m.y_true != m._cls).any():
        raise ValueError(f"{f}: y_true distinta de la clase de meta_real_ztf.csv")
    if not m.y_pred.isin(CLS).all():
        return None, {"fuera_de_val": fuera, "otras_clases": True}
    m["subset"] = m._sub
    return m.drop(columns=["_sub", "_cls"]).reset_index(drop=True), {"fuera_de_val": fuera}


def met(d, n_tot):
    """Metricas (regla 2) de las filas d: n, cobertura, acc, bal_acc, F1, recall, confusion, IC 95 %."""
    if not len(d):
        return {"n": 0, "n_total": int(n_tot), "cobertura": 0.0 if n_tot else None}
    y, yp = d.y_true.map(IX).to_numpy(int), d.y_pred.map(IX).to_numpy(int)
    r = metrics(y, yp, CLS)
    r.update(bootstrap_ci(y, yp, CLS, N_BOOT_IC, splits.SEED))
    r.update({"n_total": int(n_tot), "cobertura": float(len(d) / n_tot) if n_tot else None})
    return r


def por_subset(p, V):
    return {s: met(p[p.subset == s], int((V.subset == s).sum())) for s in SUB}


def pareado(a, b):
    """Bootstrap pareado (regla 3) de a contra b en sus oids comunes, en el orden de a. Delta = bal_acc(a) - bal_acc(b)."""
    m = a[["oid", "y_true", "y_pred"]].merge(b[["oid", "y_pred"]], on="oid", suffixes=("_a", "_b"))
    if len(m) < 2 or m.y_true.nunique() < 2:
        return {"n": int(len(m))}
    y = m.y_true.to_numpy()
    d, p, (lo, hi) = paired_bootstrap(y, (m.y_pred_a == y).to_numpy(), (m.y_pred_b == y).to_numpy(), N_BOOT, BOOT_SEED)
    return {"n": int(len(m)), "delta": float(d), "p_mejora": float(p), "ic90_delta": [lo, hi], "gana": bool(p >= P_MIN)}


# ------------------------------------------------------------------------------------------------ runs
def n_sims(run_dir):
    f = Path(run_dir) / "_sims_all.parquet" if run_dir else None
    return int(pq.ParquetFile(f).metadata.num_rows) if f and f.exists() else None


def modelo_obs(run_dir):
    """Modelo de observacion de una proyeccion (run_manifest.json)."""
    f = Path(run_dir) / "run_manifest.json" if run_dir else None
    if not f or not f.exists():
        return None
    m = json.loads(f.read_text())
    c = m.get("cfg", {})
    return {"run": m.get("run"), "dir": str(run_dir), "git": m.get("git"), "dirty": m.get("dirty"),
            "det_m0": c.get("det_m0"), "det_w": c.get("det_w"), "det_eps": c.get("det_eps"),
            "noise_draw_scale": c.get("noise_draw_scale"), "alertas": c.get("alert_model") is not None,
            "tail_min_slope": c.get("tail_min_slope"), "n_sims": n_sims(run_dir),
            "etiqueta": "nuevo (stream de alertas)" if c.get("alert_model") is not None else "viejo (sin alertas)"}


def que_es_red(name, metodo, c):
    if metodo == "supernnova":
        return (f"SuperNNova (M&ouml;ller &amp; de Boissi&egrave;re 2020), RNN {c.get('snn_model', '')}, "
                f"{'con' if c.get('use_z') else 'sin'} z")
    if metodo == "villar":
        return f"Villar+2019 con el clasificador interno de nnclf ({'con' if c.get('use_z') else 'sin'} z)"
    arq = {"gru": "GRU", "transformer": "Transformer"}.get(c.get("model"), str(c.get("model")))
    s = [f"{arq} sobre la curva cruda (magnitud, error y bandera de l&iacute;mite)"]
    if c.get("gru_pool") == "attn":
        s.append(f"{'bidireccional' if c.get('bidir') else 'unidireccional'}, con attention pooling (ORACLE-2, Shah+2026)")
    elif c.get("bidir"):
        s.append("BiGRU (control del attention pooling)")
    if c.get("jerarquica"):
        s.append("cabeza jer&aacute;rquica (primero Ia contra colapso del n&uacute;cleo, despu&eacute;s el subtipo)")
    if c.get("band_enc") == "lambda":
        s.append("banda codificada por su &lambda; (Gupta+2025, ORACLE-2)")
    if c.get("time_enc") == "atat":
        s.append("TimeModulator (ATAT, Cabrera-Vives+2024)")
    if c.get("trunc") not in (None, "none"):
        s.append(f"truncamiento 2<sup>n</sup> (ORACLE-2), p = {c.get('p_trunc')}")
    s.append("con z" if c.get("use_z") else "sin z")
    if c.get("fold"):
        s.append(f"fold {c['fold']} de 5")
    return ", ".join(s)


def que_es_villar(c):
    m = MODELOS_V.get(c.get("model"), "jer&aacute;rquico" if str(c.get("model", "")).startswith("hier") else c.get("model"))
    return (f"SPM de Villar+2019 ajustado con MCMC + {m} (features {c.get('fset')}, {'con' if c.get('use_z') else 'sin'}"
            f" z, peso {c.get('peso')}, g {c.get('g_modo', 'nan')})")


def runs_nn(root, V):
    """Corridas de una raiz nnclf (sin smoke_). Una corrida sin metrics.json o sin predicciones queda pendiente."""
    if root is None or not Path(root).is_dir():
        return None
    out = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or d.name.startswith(("smoke_", "_", ".")) or d.name == "logs":
            continue
        rd = d / "comun" if (d / "comun" / "metrics.json").exists() else d
        r = {"name": d.name, "dir": str(rd)}
        if not (rd / "metrics.json").exists() or not (rd / "pred_real_val.csv").exists():
            out.append({**r, "estado": "pendiente"})
            continue
        M = json.loads((rd / "metrics.json").read_text())
        c = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        if "ensemble_of" in c and (Path(root) / c["ensemble_of"][0] / "config.json").exists():
            c = {**json.loads((Path(root) / c["ensemble_of"][0] / "config.json").read_text()), **c}
        p, info = leer_pred(rd / "pred_real_val.csv", V)
        if p is None:                                        # corridas de 4 o 5 clases: van en la seccion de validacion
            continue
        r.update(estado="ok", metodo=M.get("method"), use_z=M.get("use_z"), n_params=c.get("n_params"),
                 sim_run=c.get("sim_run"), que_es=que_es_red(d.name, M.get("method"), {**c, **M}), **info,
                 _p=p, _M=M, _d=rd)
        r.update(por_subset(p, V))
        out.append(r)
    return out


def villar_run(sw, V):
    """Configuracion elegida por un barrido clf_villar (mejor.json + mejor/)."""
    if sw is None or not (Path(sw) / "mejor.json").exists() or not (Path(sw) / "mejor" / "pred_real_val.csv").exists():
        return {"estado": "pendiente", "dir": str(sw)}
    sw = Path(sw)
    mj = json.loads((sw / "mejor.json").read_text())
    M = json.loads((sw / "mejor" / "metrics.json").read_text())
    fs = None
    if (sw / "sweep.csv").exists():
        t = pd.read_csv(sw / "sweep.csv", nrows=200)
        fs = t.features_sims.dropna().iloc[0] if "features_sims" in t and t.features_sims.notna().any() else None
    run_dir = None
    if fs:
        n = Path(fs).name
        run_dir = Path(fs).parent / (n[len("features_"):] if n.startswith("features_") else n)
    p, info = leer_pred(sw / "mejor" / "pred_real_val.csv", V)
    c = mj.get("elegida", {})
    r = {"estado": "ok", "name": f"{sw.name} (elegida)", "dir": str(sw), "metodo": "villar_oficial", "config": c,
         "que_es": que_es_villar(c), "motivo": mj.get("motivo"), "comparacion_barrido": mj.get("comparacion"),
         "features_sims": fs, "features_real": M.get("features_real"), "sim_run": str(run_dir) if run_dir else None,
         "n_sims_con_features": M.get("n_sims"), **info, "_p": p, "_M": M}
    r.update(por_subset(p, V))
    return r


def que_es_plantillas(c):
    return ("ajuste bayesiano de las 78 series espectrales como SUDARE I (Cappellaro+2015, siguiendo PSNID de "
            "Sako+2011): evidencia por tipo marginalizando plantilla, z, E(B&minus;V) del host, T<sub>max</sub> y escala, "
            f"con los priors de las sims ({'z plana' if c.get('sin_z') else 'z espectrosc&oacute;pica'}, UL "
            f"{c.get('ul_modo', 'todos')}, error del modelo {c.get('sigma_mod')} del flujo, polvo II {c.get('ii_dust') or 'el de las sims'})")


def plantillas_run(pb, runs, V):
    """Regla 7: corrida de pipeline78.plantillas_clf en <runs>/<dir>/<name> (pred_real_val.csv + metrics.json)."""
    if not pb or not pb.get("name"):
        return None
    d = _p(pb.get("dir", "plantillas_clf"), runs) / pb["name"]
    r = {"name": pb["name"], "dir": str(d)}
    if not (d / "metrics.json").exists() or not (d / "pred_real_val.csv").exists():
        return {**r, "estado": "pendiente"}
    M = json.loads((d / "metrics.json").read_text())
    p, info = leer_pred(d / "pred_real_val.csv", V)
    if p is None:
        return {**r, "estado": "otras clases", **info}
    c = M.get("config") or {}
    r.update(estado="ok", metodo="plantillas", config=c, que_es=que_es_plantillas(c), **info, _p=p, _M=M)
    r.update(por_subset(p, V))
    return r


def comparar_plantillas(pl, vi, nn):
    """Regla 7: plantillas contra Villar (oids de Villar) y contra la red (oids comunes) por subconjunto, y exactitud
    segun cubra Villar sobre las oids que clasifican las plantillas y la red."""
    out = {}
    for s in SUB:
        a = pl["_p"][pl["_p"].subset == s]
        R = {}
        for k, o in (("villar", vi), ("red", nn)):
            if not ok(o):
                continue
            b = o["_p"][o["_p"].subset == s]
            com = set(a.oid) & set(b.oid)
            ac, bc = a[a.oid.isin(com)], b[b.oid.isin(com)]
            R[f"contra_{k}"] = {"n": len(com), "plantillas": met(ac, len(com)), k: met(bc, len(com)),
                                f"plantillas_menos_{k}": pareado(ac, bc), f"{k}_menos_plantillas": pareado(bc, ac)}
        if ok(vi) and ok(nn):
            b = nn["_p"][nn["_p"].subset == s]
            com = set(a.oid) & set(b.oid)
            cv = com & set(vi["_p"].oid[vi["_p"].subset == s])
            R["cobertura_villar"] = {k: {"n": len(o), "plantillas": met(a[a.oid.isin(o)], len(o)),
                                         "red": met(b[b.oid.isin(o)], len(o))}
                                     for k, o in (("cubre", cv), ("no_cubre", com - cv))}
        out[s] = R
    return out


def elegir_red(runs, inc):
    """Regla 3. Devuelve (mejor corrida, info de la eleccion)."""
    cand = [r for r in runs or [] if ok(r) and r.get("metodo") in REDES and r["val_sel"].get("n")]
    if not cand:
        return None, {"motivo": "sin redes con resultados"}
    cand.sort(key=lambda r: (-r["val_sel"]["bal_acc"], r["name"]))
    top, base = cand[0], next((r for r in cand if r["name"] == inc), None)
    if base is None:
        return top, {"candidata": top["name"], "incumbente": None,
                     "motivo": f"no esta la incumbente {inc}: queda la primera en val_sel"}
    if top is base:
        return base, {"candidata": top["name"], "incumbente": inc, "motivo": "la primera en val_sel es la incumbente"}
    s = lambda r: r["_p"][r["_p"].subset == "val_sel"]
    c = pareado(s(top), s(base))
    gana = bool(c.get("gana"))
    return (top if gana else base), {"candidata": top["name"], "incumbente": inc, "comparacion": c,
                                     "motivo": (f"{top['name']} gana a {inc} en val_sel" if gana else
                                                f"{top['name']} no gana a {inc} en val_sel (P < {P_MIN}): queda {inc}")}


def hibrido(V, pv, pn, s):
    """Regla 4: Villar si cubre, si no la red, sobre las reales del subconjunto s (orden de meta)."""
    Vs = V[V.subset == s]
    a, b = Vs.oid.map(dict(zip(pv.oid, pv.y_pred))), Vs.oid.map(dict(zip(pn.oid, pn.y_pred)))
    yp = a.where(a.notna(), b)
    d = pd.DataFrame({"oid": Vs.oid, "y_true": Vs.cls, "y_pred": yp, "subset": s})
    return d[d.y_pred.notna()].reset_index(drop=True), len(Vs)


def comparar(vi, nn, V):
    """Mismos objetos, hibrido y decisiones en val_sel (regla 4)."""
    out = {}
    for s in SUB:
        pv, pn = vi["_p"][vi["_p"].subset == s], nn["_p"][nn["_p"].subset == s]
        comun = pv[["oid"]].merge(pn[["oid"]], on="oid")
        pvc, pnc = pv[pv.oid.isin(comun.oid)], pn[pn.oid.isin(comun.oid)]
        h, n_tot = hibrido(V, pv, pn, s)
        hn = h[h.oid.isin(pn.oid)]
        out[s] = {"mismos_objetos": {"n": int(len(comun)), "villar": met(pvc, len(comun)), "red": met(pnc, len(comun)),
                                     "red_menos_villar": pareado(pnc, pvc), "villar_menos_red": pareado(pvc, pnc)},
                  "hibrido": {**met(h, n_tot), "n_villar": int(h.oid.isin(pv.oid).sum()),
                              "n_red": int((~h.oid.isin(pv.oid)).sum())},
                  "hibrido_menos_red": pareado(hn, pn), "red_menos_hibrido": pareado(pn, hn)}
    return out


def por_ndet(V, vi, nn, bins, pl=None):
    """Regla 5: cobertura de Villar y exactitud por bin de detecciones g + r, en val_rep (y las plantillas si estan)."""
    Vs = V[V.subset == "val_rep"].copy()
    pn, pv = nn["_p"][nn["_p"].subset == "val_rep"], vi["_p"][vi["_p"].subset == "val_rep"]
    Vs["n_det"] = Vs.oid.map(dict(zip(pn.oid, pn.n_det))) if "n_det" in pn else np.nan
    e = [-np.inf] + list(bins) + [np.inf]
    lab = [f"&lt; {bins[0]} (sin red)"] + [f"{a}&ndash;{b - 1}" for a, b in zip(bins[:-1], bins[1:])] + [f"&ge; {bins[-1]}"]
    Vs["bin"] = pd.cut(Vs.n_det.fillna(-1), e, right=False, labels=range(len(lab))).astype(int)
    h, _ = hibrido(V, pv, pn, "val_rep")
    pp = pl["_p"][pl["_p"].subset == "val_rep"] if ok(pl) else None
    acc = lambda p, o: float((p[p.oid.isin(o)].y_true == p[p.oid.isin(o)].y_pred).mean()) if p.oid.isin(o).any() else None
    rows = []
    for k, l in enumerate(lab):
        o = set(Vs.oid[Vs.bin == k])
        ov = o & set(pv.oid)
        rows.append({"bin": l, "n": len(o), "n_villar": len(ov), "cobertura_villar": len(ov) / len(o) if o else None,
                     "cobertura_red": len(o & set(pn.oid)) / len(o) if o else None,
                     "acc_villar": acc(pv, ov), "acc_red_mismos": acc(pn, ov), "acc_red": acc(pn, o),
                     "acc_hibrido": acc(h, o)})
        if pp is not None:
            rows[-1].update(n_plantillas=len(o & set(pp.oid)), acc_plantillas=acc(pp, o))
    return rows


def degradacion(nn):
    d = Path(nn["_d"])
    out = {}
    for tag, f in (("fija", "degradation_fixed.csv"), ("horizonte", "degradation_horizon.csv")):
        if (d / f).exists():
            t = pd.read_csv(d / f, dtype={"N": str})
            t = t[t.subset == "val_rep"]
            out[tag] = [{"bandas": r.bands, "N": r.N, "n_eval": int(r.n_eval), "bal_acc": float(r.bal_acc_mean),
                         "bal_acc_std": None if pd.isna(r.bal_acc_std) else float(r.bal_acc_std)} for r in t.itertuples()]
    return out


def leer_gap(d):
    f = Path(d) / "gap.json" if d else None
    if not f or not f.exists():
        return {"estado": "pendiente", "dir": str(d)}
    g = json.loads(f.read_text())
    return {"estado": "ok", "dir": str(d), "auc": g.get("auc_sim_vs_real"), "n_sims": g.get("n_sims"),
            "n_real": g.get("n_real"), "peso": g.get("peso"),
            "top": [{"feature": t["feature"], "caida_auc": t["caida_auc"], "mediana_sim": t.get("mediana_sim"),
                     "mediana_real": t.get("mediana_real")} for t in g.get("top", [])[:5]]}


def bloque(b, cfg, V, inc, bins):
    runs, cv = cfg["_runs"], _p(cfg.get("clf_villar_root", "clf_villar"), cfg["_runs"])
    root, sw = _p(b.get("nn_root"), runs), _p(b.get("villar_sweep"), cv)
    B = {"etiqueta": b.get("etiqueta"), "nn_root": str(root), "villar_sweep": str(sw),
         "esperado": {k: b.get(k) for k in ("sims_nn", "sims_villar")}}
    nn = runs_nn(root, V)
    B["nn_estado"] = "ok" if nn else "pendiente"
    vi = villar_run(sw, V)
    B["villar"] = vi
    if ok(vi):
        vi["villar_oids_rep"] = vi["val_rep"]
        ov = set(vi["_p"].oid[vi["_p"].subset == "val_rep"])
        for r in nn or []:
            if ok(r):
                q = r["_p"][(r["_p"].subset == "val_rep") & r["_p"].oid.isin(ov)]
                r["villar_oids_rep"] = met(q, len(ov))
    B["redes"] = nn or []
    pl = plantillas_run(b.get("plantillas"), runs, V)
    if pl is not None:
        B["plantillas"] = pl
        if ok(pl) and ok(vi):
            pl["villar_oids_rep"] = met(pl["_p"][(pl["_p"].subset == "val_rep") & pl["_p"].oid.isin(ov)], len(ov))
    best, el = elegir_red(nn, inc)
    B["eleccion_red"] = el
    B["red"] = best if best is not None else {"estado": "pendiente"}
    B["sims_red"] = modelo_obs(best.get("sim_run")) if best is not None and best.get("sim_run") else None
    B["sims_villar"] = modelo_obs(vi.get("sim_run")) if ok(vi) and vi.get("sim_run") else None
    if ok(pl) and (ok(vi) or best is not None):
        B["respuesta_plantillas"] = comparar_plantillas(pl, vi, best)
    if ok(vi) and best is not None:
        B["respuesta"] = comparar(vi, best, V)
        B["ndet"] = por_ndet(V, vi, best, bins, pl)
    if best is not None:
        B["degradacion"] = degradacion(best)
    B["gap"] = leer_gap(_p(b.get("gap"), cv)) if b.get("gap") else {"estado": "sin gap en la config"}
    return B


# ------------------------------------------------------------------------------------------------ aprendizaje
def aprendizaje(bloques):
    """Curva tamano de las sims: puntos por corrida (redes de cada raiz) y Villar elegido. Pares de la misma corrida y
    el mismo modelo de observacion con mas sims: bootstrap pareado en val_sel (regla 3)."""
    pts, vistos = [], set()
    for tag, B in bloques:                                   # el bloque actual va primero: gana los duplicados
        for r in B["redes"] if B["nn_root"] not in vistos else []:
            if ok(r) and r.get("metodo") in REDES and r.get("sim_run"):
                mo = modelo_obs(r["sim_run"]) or {}
                pts.append({"bloque": tag, "corrida": r["name"], "n_sims": mo.get("n_sims"), "modelo": mo.get("etiqueta"),
                            "sims": Path(r["sim_run"]).name, "bal_sel": r["val_sel"].get("bal_acc"),
                            "bal_rep": r["val_rep"].get("bal_acc"), "_r": r})
        vistos.add(B["nn_root"])
        vi = B["villar"]
        if ok(vi) and vi.get("sim_run") and vi["dir"] not in vistos:
            vistos.add(vi["dir"])
            mo = modelo_obs(vi["sim_run"]) or {}
            pts.append({"bloque": tag, "corrida": "Villar (elegida)", "n_sims": mo.get("n_sims"), "modelo": mo.get("etiqueta"),
                        "sims": Path(vi["sim_run"]).name, "bal_sel": vi["val_sel"].get("bal_acc"),
                        "bal_rep": vi["val_rep"].get("bal_acc"), "_r": vi})
    pares = []
    for a in pts:
        for b in pts:
            if (a["corrida"] == b["corrida"] and a["modelo"] == b["modelo"] and a["n_sims"] and b["n_sims"]
                    and b["n_sims"] > a["n_sims"]):
                s = lambda q: q["_r"]["_p"][q["_r"]["_p"].subset == "val_sel"]
                c = pareado(s(b), s(a))
                pares.append({"corrida": a["corrida"], "modelo": a["modelo"], "de": a["sims"], "a": b["sims"],
                              "n_de": a["n_sims"], "n_a": b["n_sims"], "sel": c,
                              "delta_rep": (b["bal_rep"] - a["bal_rep"]) if None not in (a["bal_rep"], b["bal_rep"]) else None})
    return pts, pares


# ------------------------------------------------------------------------------------------------ simulacion
CAMPOS_OBS = {"det_m0": "det_m0", "det_w": "det_w", "det_eps": "det_eps", "noise_draw_scale": "k",
              "alertas": "stream de alertas", "tail_min_slope": "piso de la cola"}


def _gr(x):
    """Valor de un campo del modelo de observacion en texto (bandas g y r de ZTF)."""
    if isinstance(x, dict):
        return " / ".join(f"{b} {x[b]}" for b in ("g", "r") if b in x) or str(x)
    if isinstance(x, bool):
        return "s&iacute;" if x else "no"
    return "&mdash;" if x is None else str(x)


def calib(cfg, esperadas=()):
    """Tablas confirm_* de calib_obs y su piloto comparado con las sims de produccion esperadas (regla 6)."""
    c = cfg.get("calib_obs") or {}
    d = _p(c.get("dir", "calib_obs"), cfg["_runs"])
    fig = _p(c.get("figura"), PHD)
    out = {"dir": str(d), "figura": str(fig) if fig else None, "figura_existe": bool(fig and fig.exists())}
    if fig and fig.exists():
        out["figura_fecha"] = datetime.datetime.fromtimestamp(fig.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    if not (d / "confirm_blancos.csv").exists():
        return {**out, "estado": "pendiente"}
    out["estado"] = "ok"
    out["tablas_fecha"] = datetime.datetime.fromtimestamp((d / "confirm_blancos.csv").stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M")
    out["figura_vieja"] = bool(out.get("figura_fecha") and out["figura_fecha"] < out["tablas_fecha"])
    meta = json.loads((d / "confirm_meta.json").read_text()) if (d / "confirm_meta.json").exists() else {}
    out.update(split=meta.get("split"), piloto=meta.get("piloto"), base=meta.get("base"), puntaje=meta.get("puntaje"),
               n_real=_get(meta, "real", "n_usable"))
    pm = out["piloto_modelo"] = modelo_obs(meta.get("piloto")) if meta.get("piloto") else None
    out["base_modelo"] = modelo_obs(meta.get("base")) if meta.get("base") else None
    prod = [m for m in esperadas if m]
    out["produccion"] = [m["run"] for m in prod]
    if pm and prod:
        out["diferencias_produccion"] = [{"campo": k, "piloto": pm.get(k), "produccion": m.get(k), "run": m["run"]}
                                         for m in prod for k in CAMPOS_OBS if pm.get(k) != m.get(k)]
        out["piloto_como_produccion"] = not out["diferencias_produccion"]
    else:
        out["piloto_como_produccion"] = None                 # sin piloto o sin run_manifest de produccion: no se sabe
    nom = Path(str(meta.get("piloto"))).name
    if out["piloto_como_produccion"] is True:
        out["etiqueta_despues"] = f"{nom} (misma config que {', '.join(out['produccion'])})"
    else:
        cs = list(dict.fromkeys(x["campo"] for x in out.get("diferencias_produccion") or [])) or ["noise_draw_scale"]
        out["etiqueta_despues"] = (f"piloto {nom} (" + ", ".join(f"{CAMPOS_OBS[k]} {_gr((pm or {}).get(k))}" for k in cs)
                                   + ")")
    t = pd.read_csv(d / "confirm_blancos.csv")
    out["blancos"] = [{"banda": r.band, "blanco": r.blanco, "real": r.real, "antes": r.sim_base, "despues": r.sim,
                       "e_real": r.e_real} for r in t.itertuples()]
    if (d / "confirm_tripletes.csv").exists():
        t = pd.read_csv(d / "confirm_tripletes.csv")
        t = t[t.rango == "todas"]
        out["tripletes"] = [{"banda": r.band, "muestra": r.muestra, "rstd": r.rstd, "err": r.err, "n": int(r.n)}
                            for r in t.itertuples()]
    if (d / "confirm_chequeos.csv").exists():
        t = pd.read_csv(d / "confirm_chequeos.csv")
        cols = [c for c in t.columns if c not in ("chequeo", "real")]
        out["chequeos_cols"] = cols
        out["chequeos"] = [{"chequeo": r["chequeo"], "real": r["real"], **{c: r[c] for c in cols}}
                           for _, r in t.iterrows()]
    return out


# ------------------------------------------------------------------------------------------------ verificacion
def verificar(B):
    """Cada cifra calculada contra la del metrics.json del run (regla 2)."""
    out = []

    def add(fuente, campo, calc, arch):
        if calc is None or arch is None:
            return
        dif = abs(float(calc) - float(arch))
        out.append({"fuente": fuente, "campo": campo, "calculado": calc, "archivo": arch, "dif": dif, "ok": dif < 1e-9})
    vi = B["villar"]
    if ok(vi):
        M, r = vi["_M"], vi["val_rep"]
        f = str(Path(vi["dir"]) / "mejor" / "metrics.json")
        for k in ("acc", "bal_acc", "f1_Ia", "f1_II", "f1_Ibc"):
            add(f, f"real_none.val_rep.{k}", r.get(k), _get(M, "real_none", "val_rep", k))
        for i, k in enumerate(("lo", "hi")):
            add(f, f"real_none.val_rep.bal_acc_ic95.{k}", r["bal_acc_ic95"][i], (_get(M, "real_none", "val_rep", "bal_acc_ic95") or [None, None])[i])
        add(f, "cobertura.val_rep.cobertura", r.get("cobertura"), _get(M, "cobertura", "val_rep", "cobertura"))
    pl = B.get("plantillas")
    if ok(pl):
        M, f = pl["_M"], str(Path(pl["dir"]) / "metrics.json")
        for k in ("acc", "bal_acc", "f1_Ia", "f1_II", "f1_Ibc"):
            add(f, f"real.val_rep.{k}", pl["val_rep"].get(k), _get(M, "real", "val_rep", k))
        for i, k in enumerate(("lo", "hi")):
            add(f, f"real.val_rep.bal_acc_ic95.{k}", pl["val_rep"]["bal_acc_ic95"][i],
                (_get(M, "real", "val_rep", "bal_acc_ic95") or [None, None])[i])
        add(f, "cobertura.val_rep.cobertura", pl["val_rep"].get("cobertura"), _get(M, "cobertura", "val_rep", "cobertura"))
    for q in B["redes"]:
        if ok(q):
            M, f = q["_M"], str(Path(q["dir"]) / "metrics.json")
            for k in ("acc", "bal_acc", "f1_Ia", "f1_II", "f1_Ibc"):
                add(f, f"main_by_subset.val_rep.metrics.{k}", q["val_rep"].get(k), _get(M, "main_by_subset", "val_rep", "metrics", k))
            add(f, "main_by_subset.val_rep.coverage", q["val_rep"].get("cobertura"), _get(M, "main_by_subset", "val_rep", "coverage"))
    return out


# ------------------------------------------------------------------------------------------------ figuras
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({                                    # skill figura-publicable
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight"})
    return plt


def _save(fig, out, name, plt):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    return f"{name}.png"


def fig_respuesta(A, out, plt):
    v, r, R = A["villar"], A["red"], A["respuesta"]["val_rep"]
    items = [("Villar", v["val_rep"], COL["villar"]), (f"Network\n(all)", r["val_rep"], COL["red"]),
             ("Network\n(Villar objects)", R["mismos_objetos"]["red"], COL["red"]), ("Hybrid", R["hibrido"], COL["hib"])]
    if ok(A.get("plantillas")):
        items.append(("Templates", A["plantillas"]["val_rep"], COL["pl"]))
    fig, ax = plt.subplots(figsize=(3.46 if len(items) == 4 else 4.2, 2.6))
    for i, (lab, m, c) in enumerate(items):
        lo, hi = m["bal_acc_ic95"]
        ax.bar(i, m["bal_acc"], 0.6, color=c, alpha=0.35 if i == 2 else 0.8, edgecolor=c)
        ax.errorbar(i, m["bal_acc"], [[m["bal_acc"] - lo], [hi - m["bal_acc"]]], color="k", capsize=2, lw=0.8)
        ax.plot(i, m["acc"], "D", color="k", ms=3, mfc="w")
        ax.text(i, 0.03, f"{100 * m['cobertura']:.0f}%" if i != 2 else f"n={m['n']}", ha="center", fontsize=7,
                color="w" if i != 2 else "k")
    ax.set_xticks(range(len(items)), [x[0] for x in items], fontsize=7)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy (bars), accuracy ($\\diamond$)")
    ax.set_title("real ZTF, reporting half (labels: coverage)")
    return _save(fig, out, "fig_respuesta", plt)


def fig_confusion(A, out, plt):
    fig, axs = plt.subplots(1, 2, figsize=(7.09, 3.0))
    for ax, (tit, m) in zip(axs, ((f"Villar+2019 SPM", A["villar"]["val_rep"]), (f"Network: {A['red']['name']}",
                                                                              A["red"]["val_rep"]))):
        cm = np.asarray(m["confusion"], float)
        fr = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(fr, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLS)):
            for j in range(len(CLS)):
                ax.text(j, i, f"{fr[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                        color="w" if fr[i, j] > 0.55 else "k")
        ax.set_xticks(range(len(CLS)), CLS)
        ax.set_yticks(range(len(CLS)), CLS)
        ax.minorticks_off()
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(f"{tit} (n = {m['n']})")
    fig.tight_layout()
    return _save(fig, out, "fig_confusion", plt)


def fig_ndet(A, out, plt):
    dg, nd = A.get("degradacion") or {}, A.get("ndet") or []
    fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.5))
    sty = {"r": dict(color="tab:red", ls="-", marker="o"), "g+r": dict(color="0.2", ls="--", marker="s")}
    for ax, tag, xl in ((axs[0], "fija", "detections kept"), (axs[1], "horizonte", "days since first detection")):
        t = pd.DataFrame(dg.get(tag) or [])
        for b in ("r", "g+r"):
            q = t[t.bandas == b] if len(t) else t
            if len(q):
                ax.errorbar(range(len(q)), q.bal_acc, q.bal_acc_std.fillna(0), ms=3, lw=0.9, capsize=2, label=b,
                            **sty[b])
                ax.set_xticks(range(len(q)), [str(x) for x in q.N])
        ax.minorticks_off()
        ax.set_ylim(0.2, 0.9)
        ax.set_xlabel(xl)
        ax.legend(frameon=False, loc="lower right")
    axs[0].set_ylabel("balanced accuracy")
    axs[0].set_title("network, fixed sample")
    axs[1].set_title("network, early epochs")
    t = pd.DataFrame(nd)
    if len(t):
        x = np.arange(len(t))
        ax = axs[2]
        ax.bar(x, t.cobertura_villar.fillna(0), 0.7, color="0.85", label="Villar coverage")
        for k, nk, lab, c, ls in (("acc_villar", "n_villar", "Villar", COL["villar"], "-"),
                                  ("acc_red_mismos", "n_villar", "network (Villar obj.)", COL["red"], ":"),
                                  ("acc_red", "n", "network (all)", COL["red"], "-"), ("acc_hibrido", "n", "hybrid", COL["hib"], "--")):
            ax.plot(x, t[k].astype(float).where(t[nk] >= N_MIN_BIN), ls=ls, marker="o", ms=3, lw=0.9, color=c, label=lab)
        ax.set_xticks(x, [html.unescape(b).replace(" (sin red)", "") + f"\n({n})" for b, n in zip(t.bin, t.n)], fontsize=7)
        ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("detections (g + r)")
        ax.set_ylabel("accuracy / coverage")
        ax.set_title("by number of detections")
        ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    return _save(fig, out, "fig_ndet", plt)


def fig_aprendizaje(pts, out, plt):
    t = pd.DataFrame([{k: v for k, v in p.items() if k != "_r"} for p in pts])
    if not len(t) or t.n_sims.isna().all():
        return None
    t = t[t.n_sims.notna()]
    rep = t.groupby("corrida").n_sims.nunique()
    keep = set(rep[rep > 1].index) | {"Villar (elegida)"}
    t = t[t.corrida.isin(keep)] if (rep > 1).any() else t
    fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.7), sharey=True)
    cmap = plt.get_cmap("Dark2")
    xs = sorted(set(t.n_sims))
    for ax, k, tit in ((axs[0], "bal_sel", "selection half (val_sel)"), (axs[1], "bal_rep", "reporting half (val_rep)")):
        for i, (c, q) in enumerate(t.groupby("corrida", sort=True)):
            col = "k" if c.startswith("Villar") else cmap(i % 10)
            for mo, qq in q.groupby("modelo"):
                qq = qq.sort_values("n_sims")
                nuevo = str(mo).startswith("nuevo")
                ax.plot(qq.n_sims, qq[k], "-" if nuevo else ":", marker="s" if c.startswith("Villar") else "o", ms=4,
                        color=col, mfc=col if nuevo else "w", lw=0.9,
                        label=(c.replace("Villar (elegida)", "Villar (chosen)") + ("" if nuevo else ", old model")))
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [f"{x / 1000:.0f}k" for x in xs])
        ax.minorticks_off()
        ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
        ax.set_xlabel("projected simulations")
        ax.set_title(tit)
    axs[0].set_ylabel("balanced accuracy")
    h, l = axs[1].get_legend_handles_labels()
    u = dict(zip(l, h))
    fig.legend(u.values(), u.keys(), frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04 + 0.05 * ((len(u) + 2) // 3), 1, 1))
    return _save(fig, out, "fig_aprendizaje", plt)


# ------------------------------------------------------------------------------------------------ conclusiones
NOM = {"red": "la red", "villar": "Villar", "hibrido": "el h&iacute;brido", "plantillas": "las plantillas"}
A_NOM = {"red": "a la red", "villar": "a Villar", "hibrido": "al h&iacute;brido", "plantillas": "a las plantillas"}
CORTO = {"red": "red", "villar": "Villar", "hibrido": "h&iacute;brido", "plantillas": "plantillas"}
MAY = {"red": "Red", "villar": "Villar", "hibrido": "H&iacute;brido", "plantillas": "Plantillas"}


def _dir_rep(c):
    """Ganador en val_rep segun el IC 90 % del delta (pos - neg): +1 si queda sobre 0, -1 bajo 0, 0 si incluye 0."""
    ic = (c or {}).get("ic90_delta")
    if not ic or None in ic:
        return 0
    return 1 if ic[0] > 0 else -1 if ic[1] < 0 else 0


def _par(sp, sn, rp, pos, neg):
    """Decision de la regla en val_sel (sp = pos - neg, sn = neg - pos) y lo que dice val_rep (rp = pos - neg)."""
    ps, pn = (sp or {}).get("p_mejora"), (sn or {}).get("p_mejora")
    sel = 1 if ps is not None and ps >= P_MIN else -1 if pn is not None and pn >= P_MIN else 0
    rep, g = _dir_rep(rp), {1: pos, -1: neg, 0: None}
    acuerdo = ("confirma" if sel and rep == sel else "contradice" if sel and rep == -sel else "no confirma" if sel else
               "val_sel no decide" if rep else "ninguno decide")
    return {"pos": pos, "neg": neg, "p_sel_pos": ps, "p_sel_neg": pn, "val_sel": g[sel], "val_rep": g[rep],
            "n_rep": (rp or {}).get("n"), "delta_rep": (rp or {}).get("delta"), "ic90_rep": (rp or {}).get("ic90_delta"),
            "acuerdo": acuerdo}


def decidir(A, meta):
    """Decisiones de val_sel (regla 4) con su contraste en val_rep, metodo recomendado y la meta (sin max en val_rep)."""
    S, R = A["respuesta"]["val_sel"], A["respuesta"]["val_rep"]
    rv = _par(S["mismos_objetos"]["red_menos_villar"], S["mismos_objetos"]["villar_menos_red"],
              R["mismos_objetos"]["red_menos_villar"], "red", "villar")
    hr = _par(S["hibrido_menos_red"], S.get("red_menos_hibrido"), R["hibrido_menos_red"], "hibrido", "red")
    rec = ("red" if rv["val_sel"] == "red" else
           "hibrido" if rv["val_sel"] == "villar" and hr["val_sel"] == "hibrido" else None)
    M = (("villar", A["villar"]["val_rep"]), ("red", A["red"]["val_rep"]), ("hibrido", R["hibrido"]))
    lm = []
    for k, m in M:                                           # orden fijo: no se elige la mayor
        b, ic = m.get("bal_acc"), m.get("bal_acc_ic95") or [None, None]
        lm.append({"metodo": k, "bal_acc": b, "ic95": ic, "cobertura": m.get("cobertura"), "n": m.get("n"),
                   "alcanza": None if b is None else bool(b >= meta),
                   "ic_incluye_meta": None if None in ic else bool(ic[0] <= meta <= ic[1])})
    return {"red_vs_villar": rv, "hibrido_vs_red": hr, "recomendado": rec, "meta_bal_acc": meta, "meta": lm}


def decidir_plantillas(A):
    """Regla 7: plantillas contra Villar y contra la red en los mismos objetos, decision en val_sel y contraste en
    val_rep (la misma _par de la regla 4)."""
    S, R = (A.get("respuesta_plantillas") or {}).get("val_sel") or {}, (A.get("respuesta_plantillas") or {}).get("val_rep") or {}
    out = {}
    for k in ("villar", "red"):
        s_, r_ = S.get(f"contra_{k}"), R.get(f"contra_{k}")
        if s_ and r_:
            out[k] = _par(s_[f"plantillas_menos_{k}"], s_[f"{k}_menos_plantillas"], r_[f"plantillas_menos_{k}"],
                          "plantillas", k)
            out[k]["n_sel"] = s_["n"]
    return out


def txt_rep(c):
    """Frase de val_rep para una decision de val_sel (salida de _par)."""
    if c.get("delta_rep") is None or not c.get("ic90_rep"):
        return "En val_rep no hay objetos suficientes para contrastarlo."
    (lo, hi), d = c["ic90_rep"], c["delta_rep"]
    cifra = f"{c['n_rep']} objetos, {CORTO[c['pos']]} &minus; {CORTO[c['neg']]} = {d:+.3f}, IC 90 % [{lo:.3f}, {hi:.3f}]"
    a, s, r = c["acuerdo"], c["val_sel"], c["val_rep"]
    if a == "confirma":
        return f"val_rep lo confirma ({cifra}, excluye 0)."
    if a == "contradice":
        return (f"OJO: val_rep lo contradice ({cifra}, excluye 0 a favor de {NOM[r]}). La elecci&oacute;n de val_sel "
                "no es robusta: revisarla antes de usarla.")
    if a == "no confirma":
        return (f"val_rep no lo confirma ({cifra}, incluye 0" + (", con el signo contrario)." if (d > 0) != (s == c["pos"])
                                                                 else ")."))
    if a == "val_sel no decide":
        return (f"Pero val_rep s&iacute; los separa y favorece {A_NOM[r]} por {abs(d):.3f} ({cifra}, excluye 0). val_rep "
                "no se usa para elegir, as&iacute; que queda como se&ntilde;al y no como decisi&oacute;n.")
    return f"Tampoco val_rep los separa ({cifra}, incluye 0)."


def txt_meta(x, meta):
    """Una cifra contra la meta, con su cobertura (las poblaciones de Villar y la red no son las mismas)."""
    lo, hi = x["ic95"]
    est = "por encima de" if x["alcanza"] else "por debajo de"
    ic = ("pero el IC 95 % la incluye" if x["ic_incluye_meta"] else
          f"con todo el IC 95 % por {'encima' if lo > meta else 'debajo'}")
    return (f"{MAY[x['metodo']]} {f3(x['bal_acc'])} [{lo:.3f}, {hi:.3f}] en el {pc(x['cobertura'])} que clasifica: "
            f"{est} la meta, {ic}")


def conclusiones(J, meta_bal):
    """Plantillas con condiciones explicitas, llenadas solo con numeros.json (J)."""
    A, out = J["actual"], []
    if J.get("override"):
        out.append("OJO: esta p&aacute;gina es una PRUEBA. El bloque actual apunta a runs del modelo viejo "
                   f"({html.escape(str(J['override']))}). Estas conclusiones no son las finales.")
    v, r = A.get("villar") or {}, A.get("red") or {}
    if v.get("estado") != "ok" or r.get("estado") != "ok":
        falta = [n for n, x in (("Villar", v), ("la red", r)) if x.get("estado") != "ok"]
        out.append(f"Todav&iacute;a no hay comparaci&oacute;n: falta {' y '.join(falta)} del bloque actual (pendiente). "
                   "Estas frases se rehacen solas cuando est&eacute;n los runs.")
    else:
        vr, rr = v["val_rep"], r["val_rep"]
        out.append(f"En val_rep, Villar acierta el {pc(vr['acc'])} de las SNe que puede clasificar y cubre el "
                   f"{pc(vr['cobertura'])} del total. La red ({r['name']}) acierta el {pc(rr['acc'])} y cubre el "
                   f"{pc(rr['cobertura'])}.")
        D = J.get("decision") or decidir(A, meta_bal)
        rv, hr, h = D["red_vs_villar"], D["hibrido_vs_red"], A["respuesta"]["val_rep"]["hibrido"]
        p_r, p_v = rv["p_sel_pos"], rv["p_sel_neg"]
        sel = {"red": f"En los mismos objetos la regla elige la red: en val_sel P(red mejor) = {f3(p_r, 2)} &ge; {P_MIN}.",
               "villar": (f"En los mismos objetos la regla elige Villar: en val_sel P(Villar mejor) = {f3(p_v, 2)} "
                          f"&ge; {P_MIN}."),
               None: (f"En los mismos objetos la regla no elige: en val_sel P(red mejor) = {f3(p_r, 2)} y P(Villar "
                      f"mejor) = {f3(p_v, 2)}, las dos bajo {P_MIN}.")}[rv["val_sel"]]
        out.append(sel + " " + txt_rep(rv))
        nd = [x for x in A.get("ndet") or [] if x["n"]]
        if len(nd) >= 3:
            lo, hi = nd[1], nd[-1]
            out.append(f"Villar deja sin clasificar el {pc(1 - vr['cobertura'])} de val_rep. Con {html.unescape(lo['bin'])} "
                       f"detecciones cubre el {pc(lo['cobertura_villar'])} y con {html.unescape(hi['bin'])} el "
                       f"{pc(hi['cobertura_villar'])}. La red deja fuera el {pc(1 - rr['cobertura'])} (las de menos de "
                       "3 detecciones).")
        hsel = {"hibrido": f"La regla en val_sel lo prefiere a la red sola (P = {f3(hr['p_sel_pos'], 2)} &ge; {P_MIN}).",
                "red": f"La regla en val_sel prefiere la red sola (P(red mejor) = {f3(hr['p_sel_neg'], 2)} &ge; {P_MIN}).",
                None: (f"La regla en val_sel no lo separa de la red sola (P(h&iacute;brido mejor) = "
                       f"{f3(hr['p_sel_pos'], 2)} &lt; {P_MIN}).")}[hr["val_sel"]]
        out.append(f"El h&iacute;brido (Villar donde ajusta, la red en el resto) cubre el {pc(h['cobertura'])} con "
                   f"exactitud balanceada {f3(h['bal_acc'])}{ci(h.get('bal_acc_ic95'))}. {hsel} {txt_rep(hr)}")
        rec = D["recomendado"]
        cob = (f"Villar clasifica el {pc(vr['cobertura'])}, la red el {pc(rr['cobertura'])} y el h&iacute;brido el "
               f"{pc(h['cobertura'])}. En las tasas cada SN sin clasificar entra como correcci&oacute;n de eficiencia.")
        if rec == "red":
            t = ("Para las tasas la regla recomienda la red: es mejor en los mismos objetos (val_sel) y cubre el "
                 f"{pc(rr['cobertura'])}.")
        elif rec == "hibrido":
            t = ("Para las tasas la regla recomienda el h&iacute;brido: Villar es mejor donde ajusta y el h&iacute;brido "
                 f"le gana a la red sola en val_sel. Cubre el {pc(h['cobertura'])}.")
        elif rv["val_sel"] == "villar":
            t = ("Villar es mejor donde ajusta, pero el h&iacute;brido no pasa la regla contra la red sola, as&iacute; "
                 f"que la regla no recomienda un m&eacute;todo. {cob} El h&iacute;brido evita esa correcci&oacute;n a "
                 "costa de usar la red en las de pocas detecciones.")
        else:
            t = ("La regla en val_sel no recomienda un m&eacute;todo: no separa a Villar de la red en los mismos objetos. "
                 + (f"val_rep, que no elige, favorece {A_NOM[rv['val_rep']]} en esos objetos (ver arriba). "
                    if rv["val_rep"] else "Tampoco val_rep los separa. ") + f"Adem&aacute;s difieren en cobertura. {cob}")
        base = [rv] + ([hr] if rec == "hibrido" else []) if rec else []
        if any(x["acuerdo"] == "contradice" for x in base):
            t += " OJO: val_rep contradice esta elecci&oacute;n (ver arriba). No usarla sin revisar."
        elif any(x["acuerdo"] == "no confirma" for x in base):
            t += " val_rep no la confirma con un IC 90 % que excluya 0."
        out.append(t + " Esta comparaci&oacute;n es de la primera etapa (Villar contra la red). La propuesta final usa "
                   "las plantillas (secci&oacute;n 1).")
        lm = {x["metodo"]: x for x in D["meta"]}
        out.append((f"El m&eacute;todo que recomienda la regla es {NOM[rec]}. Contra la meta de exactitud balanceada "
                    f"{meta_bal} en val_rep: {txt_meta(lm[rec], meta_bal)}." if rec else
                    f"Meta de exactitud balanceada {meta_bal}: la regla no recomienda un m&eacute;todo, as&iacute; que no "
                    "hay cifra titular contra la meta (no se elige la mayor de val_rep).")
                   + f" Las tres cifras contra la meta, en orden fijo y cada una sobre su propia cobertura (poblaciones "
                     "distintas, no se comparan entre s&iacute;). " + ". ".join(txt_meta(x, meta_bal) for x in D["meta"])
                   + ".")
        fv = {c: vr[f"f1_{c}"] for c in CLS}
        fr = {c: rr[f"f1_{c}"] for c in CLS}
        cv_, cr_ = min(fv, key=fv.get), min(fr, key=fr.get)
        out.append(f"La clase m&aacute;s dif&iacute;cil es {cv_} en Villar (F1 {f3(fv[cv_])}) y {cr_} en la red "
                   f"(F1 {f3(fr[cr_])}).")
    out += conclusiones_plantillas(J)
    pares = [p for p in J.get("aprendizaje_pares") or [] if p["sel"].get("p_mejora") is not None]
    if pares:
        g = [p for p in pares if p["sel"]["gana"]]
        txt = "; ".join(f"{p['corrida']} {p['n_de']} &rarr; {p['n_a']}: &Delta; {p['sel']['delta']:+.3f}, "
                        f"P = {p['sel']['p_mejora']:.2f}" for p in pares)
        out.append(f"M&aacute;s sims: {len(g)} de {len(pares)} comparaciones mejoran en val_sel con la regla "
                   f"(P &ge; {P_MIN}) ({txt}).")
    ga, gd = (J.get("simulacion") or {}).get("gap_antes") or {}, A.get("gap") or {}
    if ga.get("estado") == "ok" and gd.get("estado") == "ok":
        d = gd["auc"] - ga["auc"]
        out.append(f"El clasificador sim contra real pasa de AUC {f3(ga['auc'])} a {f3(gd['auc'])} (0.5 = sims que no se "
                   "distinguen de las reales): " + ("la simulaci&oacute;n nueva se parece m&aacute;s a las reales."
                                                    if d < -0.02 else "no mejor&oacute;." if d > 0.02 else
                                                    "casi igual."))
    elif ga.get("estado") == "ok":
        out.append(f"El clasificador sim contra real daba AUC {f3(ga['auc'])} con el modelo viejo (0.5 = sims "
                   "indistinguibles). Con el modelo nuevo est&aacute; pendiente.")
    c = (J.get("simulacion") or {}).get("calib") or {}
    if c.get("estado") == "ok" and c.get("puntaje"):
        pg, pr = c["puntaje"].get("g"), c["puntaje"].get("r")
        cifra = (f"el puntaje (0 = calce perfecto) baj&oacute; de {pg[1]:.0f} a {pg[0]:.1f} en g y de {pr[1]:.0f} a "
                 f"{pr[0]:.1f} en r, contra las alertas de {c.get('split')}")
        cp, prod = c.get("piloto_como_produccion"), ", ".join(c.get("produccion") or [])
        if cp is True:
            out.append(f"Con la calibraci&oacute;n del modelo de observaci&oacute;n {cifra}. El piloto "
                       f"({c['etiqueta_despues']}) tiene la misma config que la producci&oacute;n.")
        elif cp is False:
            dp = list(dict.fromkeys(f"{CAMPOS_OBS[x['campo']]} {_gr(x['produccion'])}"
                                    for x in c["diferencias_produccion"]))
            out.append(f"Con el {c['etiqueta_despues']} {cifra}. La producci&oacute;n ({prod}) usa {', '.join(dp)}: "
                       "falta confirmar con la config final (repetir confirm). Hasta entonces la cifra es del piloto.")
        else:
            out.append(f"Con el {c.get('etiqueta_despues')} de calib_obs {cifra}. Falta compararlo con las sims "
                       "de producci&oacute;n (no est&aacute; su run_manifest.json).")
    return out


def conclusiones_plantillas(J):
    """Regla 7: frases de las plantillas, llenadas con numeros.json."""
    A, out = J["actual"], []
    pl = A.get("plantillas")
    if not pl:
        return out
    if pl.get("estado") != "ok":
        return [f"Plantillas (tercer m&eacute;todo): {html.escape(str(pl.get('name')))} {pl.get('estado')}."]
    q, c = pl["val_rep"], pl.get("config") or {}
    out.append(f"Plantillas (ajuste bayesiano de las 78 series, como SUDARE I): en val_rep cubren el {pc(q['cobertura'])} y "
               f"aciertan el {pc(q['acc'])}, exactitud balanceada {f3(q['bal_acc'])}{ci(q.get('bal_acc_ic95'))} "
               f"(UL {c.get('ul_modo')}, error del modelo {c.get('sigma_mod')} del flujo del modelo, "
               f"{html.escape(str(c.get('sigma_mod_fuente', 'sin fuente')).split(' (')[0])}).")
    D = J.get("decision_plantillas") or {}
    for k in ("villar", "red"):
        x = D.get(k)
        if not x:
            continue
        nom = "Villar" if k == "villar" else f"la red ({A['red']['name']})"
        sel = (f"En los mismos objetos que {nom} la regla en val_sel elige {NOM[x['val_sel']]} "
               f"(P(plantillas mejor) = {f3(x['p_sel_pos'], 2)}, P({CORTO[k]} mejor) = {f3(x['p_sel_neg'], 2)}, "
               f"{x['n_sel']} objetos)." if x["val_sel"] else
               f"En los mismos objetos que {nom} la regla en val_sel no elige (P(plantillas mejor) = "
               f"{f3(x['p_sel_pos'], 2)}, P({CORTO[k]} mejor) = {f3(x['p_sel_neg'], 2)}, {x['n_sel']} objetos, las dos "
               f"bajo {P_MIN}).")
        out.append(sel + " " + txt_rep(x))
    cv = ((A.get("respuesta_plantillas") or {}).get("val_rep") or {}).get("cobertura_villar")
    if cv and cv["cubre"]["n"] and cv["no_cubre"]["n"]:
        a_, b_ = cv["cubre"], cv["no_cubre"]
        out.append(f"En val_rep, sobre las SNe que clasifican las plantillas y la red: donde Villar ajusta ({a_['n']}) las "
                   f"plantillas aciertan el {pc(a_['plantillas']['acc'])} y la red el {pc(a_['red']['acc'])}. Donde "
                   f"Villar no ajusta ({b_['n']}) las plantillas aciertan el {pc(b_['plantillas']['acc'])} y la red el "
                   f"{pc(b_['red']['acc'])}.")
    return out


def pendientes(J, cfg):
    A, out = J["actual"], []
    if A.get("nn_estado") != "ok":
        out.append(f"Redes del bloque actual: falta {A['nn_root']}.")
    for r in A.get("redes") or []:
        if r.get("estado") != "ok":
            out.append(f"Red {r['name']} ({r['dir']}): {r.get('estado')} (sin metrics.json o sin pred_real_val.csv: "
                       "corriendo o fall&oacute;).")
    if (A.get("villar") or {}).get("estado") != "ok":
        out.append(f"Villar del bloque actual: falta {A['villar_sweep']}/mejor.json (barrido sobre las features MCMC "
                   "de las sims nuevas).")
    if A.get("plantillas") and A["plantillas"].get("estado") != "ok":
        out.append(f"Plantillas del bloque actual: {A['plantillas'].get('dir')} {A['plantillas'].get('estado')} (sin "
                   "metrics.json o sin pred_real_val.csv).")
    if (A.get("gap") or {}).get("estado") != "ok":
        out.append(f"Gap sim contra real del modelo nuevo: {A.get('gap', {}).get('dir', 'sin nombre')} pendiente.")
    for tag, mo in (("red", A.get("sims_red")), ("villar", A.get("sims_villar"))):
        esp = (A.get("esperado") or {}).get(f"sims_{'nn' if tag == 'red' else 'villar'}")
        if mo and esp and mo.get("run") != esp:
            out.append(f"Las sims de {tag} del bloque actual son {mo.get('run')}, la config espera {esp}.")
    c = (J.get("simulacion") or {}).get("calib") or {}
    if c.get("figura_vieja"):
        out.append(f"La figura calib_obs ({c['figura_fecha']}) es anterior a las tablas confirm_* ({c['tablas_fecha']}): "
                   "rehacerla con el piloto de las tablas (python -m pipeline78.calib_obs fig).")
    if c.get("piloto_como_produccion") is False:
        d = c["diferencias_produccion"]
        dp = list(dict.fromkeys(f"{CAMPOS_OBS[x['campo']]} {_gr(x['produccion'])}" for x in d))
        out.append(f"La confirmaci&oacute;n de calib_obs us&oacute; el {c['etiqueta_despues']} y la producci&oacute;n "
                   f"({', '.join(c.get('produccion') or [])}) usa {', '.join(dp)}: repetir confirm con la config final.")
    elif c.get("estado") == "ok" and c.get("piloto_como_produccion") is None:
        out.append("No se pudo comparar el piloto de calib_obs con las sims de producci&oacute;n de la config (falta el "
                   "piloto o el run_manifest.json de sims_nn / sims_villar).")
    for B in [A] + list(J.get("historia") or []):
        for r in [B.get("villar") or {}] + list(B.get("redes") or []):
            if r.get("fuera_de_val"):
                out.append(f"{r.get('name')} ({r.get('dir')}): {r['fuera_de_val']} predicciones con oids que ya no "
                           "est&aacute;n en la mitad val actual (meta regenerada); se descartaron.")
    mal = [v for v in J.get("verificacion") or [] if not v["ok"]]
    if mal:
        out.append(f"{len(mal)} cifras no coinciden con su metrics.json (ver numeros.json, verificacion).")
    out += [html.escape(s) for s in cfg.get("pendientes", [])]
    return out


# ------------------------------------------------------------------------------------------------ pagina
CSS = ("table{border-collapse:collapse;margin:6px 0 14px}td,th{border:1px solid #999;padding:2px 7px;font-size:14px}"
       "th{background:#eee}td{text-align:left}td.n{text-align:right}.pend{color:#b00;font-weight:bold}"
       ".aviso{background:#fff3cd;border:1px solid #e0c060;padding:8px}.viejo{background:#f6e6e6;border:1px solid "
       "#c99;padding:8px}.nota{color:#555;font-size:14px}li{margin:3px 0}details summary{cursor:pointer;font-weight:bold;"
       "margin:8px 0}.resumen{background:#eef6ee;border:1px solid #9c9;padding:6px 14px}.anexo{border-left:3px solid #ccc;"
       "padding-left:12px}img{max-width:100%;height:auto}")
GLOSARIO = [
    ("VAL y TEST", "las SNe de ZTF con tipo espectrosc&oacute;pico (TNS) se partieron al azar en dos mitades. VAL se "
                   "us&oacute; para elegir. TEST se us&oacute; una sola vez, al final, con todo congelado."),
    ("val_sel y val_rep", "VAL se parti&oacute; a su vez en dos: val_sel para elegir configuraciones y val_rep para ver "
                          "si la elecci&oacute;n se sostiene en SNe que no participaron."),
    ("exactitud", "fracci&oacute;n de las SNe clasificadas que quedan en su tipo correcto."),
    ("exactitud balanceada", "promedio de los aciertos de cada clase, para que la clase m&aacute;s numerosa no tape a "
                             "las chicas. Es la m&eacute;trica con la que se elige."),
    ("recall de una clase", "de las SNe de esa clase, la fracci&oacute;n que se clasifica bien."),
    ("cobertura", "fracci&oacute;n de las SNe a las que el m&eacute;todo les da una clase. Villar necesita que su ajuste "
                  "MCMC funcione (al menos 7 detecciones y un l&iacute;mite previo a la primera); las plantillas y la "
                  "red piden 3 detecciones."),
    ("SNe que Villar alcanza", "las que Villar puede ajustar. Comparar ah&iacute; es lo justo para Villar."),
    ("respaldo", "el m&eacute;todo que clasifica las SNe que Villar no alcanza, para comparar sistemas completos."),
    ("P", "se re-sortean las mismas SNe miles de veces (bootstrap) y se cuenta en qu&eacute; fracci&oacute;n de los "
          "sorteos un m&eacute;todo le gana al otro. P &ge; 0.9: mejor. P &le; 0.1: peor. Entre medio: empate."),
    ("matriz de confusi&oacute;n", "filas = tipo real, columnas = tipo asignado. La diagonal son los aciertos."),
]


def _fila_modelo(r, best, villar_oid=True):
    if not ok(r):
        return [f"{html.escape(r.get('name', ''))}", "", f"<span class='pend'>{r.get('estado', 'pendiente')}</span>"] + [""] * 8
    q, vo = r["val_rep"], r.get("villar_oids_rep") or {}
    star = " &#9733;" if best else ""
    return [f"<b>{html.escape(r['name'])}</b>{star}" if best else html.escape(r["name"]), r["que_es"],
            pc(q.get("cobertura")), f3(q.get("acc")), f"{f3(q.get('bal_acc'))}{ci(q.get('bal_acc_ic95'))}",
            f3(q.get("f1_Ia")), f3(q.get("f1_II")), f3(q.get("f1_Ibc")),
            (f"{f3(vo.get('bal_acc'))} ({vo.get('n')})" if vo.get("n") else "&mdash;") if villar_oid else "",
            f3(r["val_sel"].get("bal_acc")), str(q.get("n", ""))]


HEAD_MOD = ["modelo", "qu&eacute; es", "cobertura", "exactitud", "exact. balanceada [IC 95 %]", "F1 Ia", "F1 II",
            "F1 Ibc", "exact. bal. en oids de Villar (n)", "exact. bal. val_sel (elige)", "n"]


def tabla_modelos(B):
    v, best = B["villar"], (B["red"] or {}).get("name")
    rows = [_fila_modelo(v, True) if ok(v) else
            [html.escape(Path(B["villar_sweep"]).name), "Villar+2019 (barrido oficial)", "<span class='pend'>pendiente</span>"] + [""] * 8]
    rows += [_fila_modelo(r, r.get("name") == best) for r in B["redes"]]
    if not B["redes"]:
        rows.append([html.escape(Path(B["nn_root"]).name), "redes", "<span class='pend'>pendiente</span>"] + [""] * 8)
    if B.get("plantillas"):
        rows.append(_fila_modelo(B["plantillas"], False))
    return tabla(HEAD_MOD, rows)


def _fmt_obs(mo, pre=""):
    if not mo:
        return ["&mdash;"] * 6
    f = lambda x: " / ".join(f"{k} {v}" for k, v in x.items()) if isinstance(x, dict) else str(x)
    return [f"{pre}{Path(mo['dir']).name} (config {mo['run']}, {mo['n_sims']} sims)", f(mo["det_m0"]), f(mo["det_w"]), f(mo["noise_draw_scale"] or 1.0),
            "s&iacute;" if mo["alertas"] else "no", f(mo["tail_min_slope"])]


def seccion_plantillas(J):
    """Regla 7: el tercer metodo en la pagina (mismos objetos contra Villar y la red, y segun cubra Villar)."""
    A = J["actual"]
    pl = A.get("plantillas")
    h = ["<h3>Ajuste bayesiano de plantillas (como SUDARE I) en VAL</h3>"]
    if not pl:
        return h[0] + "<p class='pend'>Sin bloque plantillas en la config.</p>"
    if pl.get("estado") != "ok":
        return h[0] + f"<p class='pend'>{html.escape(str(pl.get('dir')))}: {pl.get('estado')}.</p>"
    c = pl.get("config") or {}
    h.append(f"<p>{html.escape(pl['name'])}: {pl['que_es']}. Sin entrenamiento: las 78 series proyectadas a la curva "
             f"observada de cada SN con los priors de las sims (LF, polvo, fracciones de subtipo). UL {c.get('ul_modo')} "
             + ("(solo los anteriores a la primera detecci&oacute;n, decisi&oacute;n a priori), " if c.get("ul_modo") ==
                "previos" else "(OJO: no es el modo a priori, previos), ") + "error "
             f"del modelo {c.get('sigma_mod')} del flujo del modelo "
             f"({html.escape(str(c.get('sigma_mod_fuente', 'sin fuente')))}). Corrida {html.escape(pl['dir'])}.</p>")
    R = A.get("respuesta_plantillas") or {}
    D = J.get("decision_plantillas") or {}
    fil = []
    for k, nom in (("villar", "Villar"), ("red", f"red ({html.escape(str((A.get('red') or {}).get('name')))})")):
        for s_ in SUB:
            x = (R.get(s_) or {}).get(f"contra_{k}")
            if not x:
                continue
            d1, d2 = x[f"plantillas_menos_{k}"], x[f"{k}_menos_plantillas"]
            fil.append([f"contra {nom}", s_, str(x["n"]), f"{f3(x['plantillas'].get('bal_acc'))}{ci(x['plantillas'].get('bal_acc_ic95'))}",
                        f"{f3(x[k].get('bal_acc'))}{ci(x[k].get('bal_acc_ic95'))}",
                        f"{d1.get('delta', float('nan')):+.3f}" if d1.get("delta") is not None else "&mdash;",
                        f3(d1.get("p_mejora"), 3), f3(d2.get("p_mejora"), 3), ci(d1.get("ic90_delta")).strip() or "&mdash;"])
    h.append(tabla(["comparaci&oacute;n", "subconjunto", "n (mismos objetos)", "plantillas: exact. bal. [IC 95 %]",
                    "el otro: exact. bal. [IC 95 %]", "&Delta; plantillas &minus; otro", "P(plantillas mejor)",
                    "P(otro mejor)", "IC 90 % del &Delta;"], fil))
    for k in ("villar", "red"):
        x = D.get(k)
        if x:
            h.append(f"<p>Contra {CORTO[k] if k == 'villar' else 'la red'}: la regla en val_sel "
                     + (f"elige <b>{NOM[x['val_sel']]}</b>" if x["val_sel"] else "<b>no elige</b>")
                     + f" (gana el que pase {P_MIN}). "
                     f"{txt_rep(x)}</p>")
    fil = []
    for s_ in SUB:
        cv = (R.get(s_) or {}).get("cobertura_villar")
        for k, nom in (("cubre", "Villar ajusta"), ("no_cubre", "Villar no ajusta")):
            if cv and cv[k]["n"]:
                fil.append([s_, nom, str(cv[k]["n"]), f3(cv[k]["plantillas"].get("acc")), f3(cv[k]["plantillas"].get("bal_acc")),
                            f3(cv[k]["red"].get("acc")), f3(cv[k]["red"].get("bal_acc"))])
    if fil:
        h.append("<p>Seg&uacute;n Villar pueda ajustar la SN, sobre las que clasifican las plantillas y la red:</p>"
                 + tabla(["subconjunto", "SNe", "n", "exactitud plantillas", "exact. bal. plantillas", "exactitud red",
                          "exact. bal. red"], fil))
    return "".join(h)


TEST_MATRICES = {3: [("Villar", "test_final/villar"), ("Red (transformer)", "test_final/red"),
                     ("Plantillas", "test_final/plantillas")],
                 4: [("Villar", "test_final/villar_4c"), ("Plantillas", "test_final/plantillas_4c")],
                 5: [("Villar", "cinco_clases/villar_5c"), ("Red (transformer)", "cinco_clases/red_5c"),
                     ("Plantillas", "cinco_clases/plantillas_5c")]}


def leer_validacion(runs):
    """TEST congelado (evaluar_test), comparacion justa (comparacion_justa) y 5 clases, para la seccion de validacion."""
    def js(p):
        p = Path(runs) / p
        return json.loads(p.read_text()) if p.exists() else None
    mats = {}
    for k, lst in TEST_MATRICES.items():
        for lab, d in lst:
            m = js(f"{d}/metrics.json")
            if m:
                t = m.get("test", m)
                mats.setdefault(k, []).append(dict(lab=lab, clases=m["clases"], conf=t["confusion"], acc=t["acc"],
                                                   bal=t["bal_acc"], ic=t.get("bal_acc_ic95"), cob=t["cobertura"],
                                                   n=t["n_clasificadas"], N=t["n_real"],
                                                   f1={c: t.get(f"f1_{c}") for c in m["clases"]}))
    return dict(justa=js("test_final/comparacion_justa.json"), resumen3=js("test_final/resumen.json"),
                resumen5=js("cinco_clases/resumen.json"), matrices=mats)


def fig_matrices_test(V, out, plt):
    """Matrices de confusion en TEST: filas 3, 4 y 5 clases; columnas los metodos. Fraccion de la fila y conteo."""
    M = V["matrices"]
    if not M:
        return None
    nr, nc = len(M), max(len(v) for v in M.values())
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.0 * nr), squeeze=False)
    for i, (k, lst) in enumerate(sorted(M.items())):
        for j in range(nc):
            ax = axs[i][j]
            if j >= len(lst):
                ax.axis("off")
                continue
            x = lst[j]
            c = np.array(x["conf"], float)
            f = c / np.maximum(c.sum(1, keepdims=True), 1)
            ax.imshow(f, cmap="Blues", vmin=0, vmax=1)
            for a in range(len(c)):
                for b in range(len(c)):
                    ax.text(b, a, f"{f[a, b]:.2f}\n({int(c[a, b])})", ha="center", va="center", fontsize=6.5,
                            color="white" if f[a, b] > 0.55 else "black")
            ax.set_xticks(range(len(c)), x["clases"])
            ax.set_yticks(range(len(c)), x["clases"])
            ax.minorticks_off()
            ax.tick_params(top=False, right=False)
            ax.set_title(f"{x['lab']}, {k} clases\nacc {x['acc']:.3f}, bal {x['bal']:.3f}, cob {x['cob']:.0%}")
            if j == 0:
                ax.set_ylabel("clase real")
            ax.set_xlabel("clase predicha")
    fig.tight_layout()
    return _save(fig, out, "fig_matrices_test", plt)


def _recall_ii(M):
    """Recall de II en 4 clases (TEST) de cada metodo, leido de las matrices: es la clase mas debil de las plantillas."""
    out = []
    for x in M.get(4, []):
        c = np.array(x["conf"], float)
        i = x["clases"].index("II")
        out.append(f"{x['lab']} {c[i, i] / max(c[i].sum(), 1):.2f}")
    return (" En 4 clases, que es lo que usan las tasas, el recall de II es " + ", ".join(out)
            + " (Villar solo sobre las SNe que ajusta). Es el punto d&eacute;bil de las plantillas.") if out else ""


def _mx(M, k, lab):
    """Matriz de TEST de k clases del metodo lab (o None)."""
    return next((x for x in M.get(k, []) if x["lab"] == lab), None)


def _fila(x, c):
    """Fracciones de la fila de la clase c en una matriz de TEST: {clase predicha: fraccion}."""
    m = np.array(x["conf"], float)
    i = x["clases"].index(c)
    return {d: m[i, j] / max(m[i].sum(), 1) for j, d in enumerate(x["clases"])}


def _veredicto(p):
    return "mejor" if p >= P_MIN else "peor" if p <= 1 - P_MIN else "empate"


def _fp(p):
    """P con 2 decimales, o 3 si al redondear quedaria pegada a un umbral (0.897 no es 0.90)."""
    return f"{p:.3f}" if min(abs(p - P_MIN), abs(p - (1 - P_MIN))) < 0.006 else f"{p:.2f}"


def _son(p):
    """'las plantillas ... Villar' segun P(plantillas mejor)."""
    return "son mejores que" if p >= P_MIN else "son peores que" if p <= 1 - P_MIN else "empatan con"


def resumen(J):
    """Recuadro de arriba: la propuesta y las cifras de TEST que la sostienen, todo leido de los archivos de TEST."""
    V = J.get("validacion") or {}
    jt, M = V.get("justa"), V.get("matrices") or {}
    p4, v4, p3 = _mx(M, 4, "Plantillas"), _mx(M, 4, "Villar"), _mx(M, 3, "Plantillas")
    if not (jt and p4 and v4 and p3):
        return "<p class='pend'>Resumen pendiente: faltan los resultados de TEST.</p>"
    a3, a4 = jt["3"]["villar_alcanza"]["plantillas"], jt["4"]["villar_alcanza"]["plantillas"]
    f = _fila(p4, "II")
    return ("<div class='resumen'><b>En corto</b><ul>"
            "<li><b>Propuesta:</b> clasificar las SNe de ZTF (y despu&eacute;s las de SUDARE) con el <b>ajuste de "
            "plantillas</b>, el m&eacute;todo de SUDARE I, usando como plantillas nuestras 78 series espectrales. Villar "
            "y la red neuronal quedan como comparaci&oacute;n.</li>"
            f"<li>En TEST ({p4['N']} SNe de ZTF que no se usaron para elegir nada), con las 4 clases de las tasas (Ia, "
            f"II, Ibc, IIn), las plantillas clasifican el {p4['cob']:.0%} de las SNe con exactitud balanceada "
            f"{p4['bal']:.2f}. Villar clasifica el {v4['cob']:.0%}. En las SNe que Villar s&iacute; alcanza, las plantillas "
            f"{_son(a4['P_mejor_que_villar'])} Villar (P = {a4['P_mejor_que_villar']:.2f}).</li>"
            f"<li>Con 3 clases (Ia, II, Ibc) las plantillas {_son(a3['P_mejor_que_villar'])} Villar en "
            f"las SNe que Villar alcanza (P = {a3['P_mejor_que_villar']:.2f}) y llegan a exactitud balanceada "
            f"{p3['bal']:.2f} sobre el {p3['cob']:.0%} de las SNe.</li>"
            f"<li><b>Punto d&eacute;bil:</b> las II. En 4 clases las plantillas reconocen el {f['II']:.0%} de las II; "
            f"el resto se va a IIn ({f['IIn']:.0%}) y a Ibc ({f['Ibc']:.0%}).</li>"
            "<li><b>Se pide validar:</b> (1) las plantillas como clasificador oficial; (2) 4 clases para las tasas, "
            "como SUDARE; (3) corregir las fracciones por clase con la matriz de confusi&oacute;n.</li></ul></div>")


def seccion_validacion(J):
    """Seccion 1, para que el profesor guia valide la eleccion: TEST congelado, comparacion justa y matrices de
    confusion. Todas las cifras se leen de los archivos de evaluar_test y comparacion_justa."""
    V = J.get("validacion") or {}
    jt, M = V.get("justa"), V.get("matrices") or {}
    T = J.get("tasas") or {}
    h = ["<h2 id='validacion'>1. Qu&eacute; clasificador usa la tesis</h2>",
         "<p><b>Propuesta.</b> Usar el <b>ajuste de plantillas</b> (Cappellaro et al. 2015, &sect;4.1, que sigue a PSNID "
         "de Sako et al. 2011) con nuestras 78 series espectrales como plantillas. Cada SN se compara con todas las "
         "plantillas en todos los corrimientos al rojo, extinciones y fechas de m&aacute;ximo posibles, y la "
         "probabilidad de cada tipo sale de qu&eacute; tan bien calzan, pesada por lo que se sabe de cada tipo "
         "(funci&oacute;n de luminosidad, polvo y fracci&oacute;n de subtipos). No se entrena con nada. Villar (ajuste "
         "de la curva de Villar et al. 2019 + clasificador entrenado con simulaciones) y la red (transformer entrenado "
         "con simulaciones) quedan como comparaci&oacute;n.</p>"]
    pv = (T.get("pares") or {}).get("val_sel") or {}
    rp, vp = pv.get("red_vs_plantillas"), pv.get("villar_vs_plantillas")
    cob = (T.get("por_subconjunto") or {}).get("val_sel") or {}
    if rp and vp and cob:
        h.append("<p><b>C&oacute;mo se eligi&oacute;.</b> Ning&uacute;n m&eacute;todo vio SNe reales para aprender. La "
                 "regla se fij&oacute; el 2026-10-04, antes de ver val_rep y TEST: gana el que mejora con P &ge; 0.9 en val_sel; si "
                 "empatan, el que clasifica m&aacute;s SNe por s&iacute; mismo; despu&eacute;s, el que depende menos de "
                 f"las simulaciones. En val_sel las plantillas le ganan a la red (P = {1 - rp['P_bal']:.2f}) y empatan "
                 f"con Villar (P = {1 - vp['P_bal']:.2f}). En el empate decide la cobertura: las plantillas clasifican "
                 f"por s&iacute; mismas el {cob['plantillas']['cobertura_propia']:.0%} de val_sel y Villar el "
                 f"{cob['villar']['cobertura_propia']:.0%}. Reci&eacute;n despu&eacute;s se corrieron en TEST, una sola "
                 "vez, los tres m&eacute;todos congelados (antes de TEST cada uno reprodujo exacto sus resultados de "
                 "VAL). Las particiones est&aacute;n en la secci&oacute;n 3.</p>")
    if jt:
        h.append("<h3>Comparaci&oacute;n justa en TEST</h3><p>Cada celda: exactitud / exactitud balanceada. Se compara "
                 "de dos maneras: (a) solo en las SNe que Villar alcanza, que es lo justo para Villar; (b) en todas las "
                 "SNe, donde Villar va solo (las que no alcanza cuentan como error) o con respaldo (la red; en 4 clases, "
                 "las plantillas). P = probabilidad de que el m&eacute;todo sea mejor que Villar.</p>")
        nom = {"villar": "Villar", "red": "Red (transformer)", "plantillas": "Plantillas"}
        for k in sorted(jt, key=int):
            r = jt[k]
            va, to = r["villar_alcanza"], r["total"]
            rows = []
            for m in ("villar", "red", "plantillas"):
                if m not in va:
                    continue
                a, t = va[m], to.get(m)
                pa = f"{a['P_mejor_que_villar']:.2f}" if "P_mejor_que_villar" in a else "&ndash;"
                tt = f"{t['acc']:.3f} / {t['bal']:.3f}" if t else ""
                pt = f"{t['P_mejor']:.2f}" if t and "P_mejor" in t else "&ndash;"
                if m == "villar":
                    tt = (f"{to['villar_solo']['acc']:.3f} / {to['villar_solo']['bal']:.3f} (solo) &middot; "
                          f"{to['villar_mas_respaldo']['acc']:.3f} / {to['villar_mas_respaldo']['bal']:.3f} (con respaldo)")
                rows.append([f"<b>{nom[m]}</b>", f"{a['acc']:.3f} / {a['bal']:.3f}", pa, tt, pt])
            n_va, n_to = va["villar"]["n"], to["villar_mas_respaldo"]["n"]
            cl = {"3": "Ia, II, Ibc", "4": "Ia, II, Ibc, IIn", "5": "Ia, II, IIb, Ibc, IIn"}.get(str(k), "")
            h.append(f"<p><b>{k} clases</b> ({cl}). TEST: {r['N']} SNe, Villar alcanza {n_va}, clasificables {n_to}.</p>")
            h.append(tabla(["m&eacute;todo", f"(a) SNe que Villar alcanza ({n_va})", "P vs Villar", f"(b) todas ({n_to})",
                            "P vs Villar con respaldo"], rows))
    else:
        h.append("<p class='pend'>Pendiente: python -m pipeline78.comparacion_justa.</p>")
    if J.get("figuras", {}).get("matrices_test"):
        h.append("<h3>Matrices de confusi&oacute;n en TEST</h3><p>Filas = tipo real, columnas = tipo asignado. Cada "
                 "celda: fracci&oacute;n de la fila y (n&uacute;mero de SNe). Arriba 3 clases, al medio 4 clases (con "
                 "IIn, las de las tasas), abajo 5 clases (IIb aparte). Villar se muestra solo en las SNe que alcanza.</p>"
                 f"<img src='{J['figuras']['matrices_test']}' width='1000'>")
    if M:
        rows = [[f"{k}", x["lab"], f"{x['n']}/{x['N']} ({x['cob']:.0%})", f"{x['acc']:.3f}", f"{x['bal']:.3f} {ci(x['ic'])}",
                 ", ".join(f"{c} {v:.2f}" for c, v in x["f1"].items() if v is not None)]
                for k, lst in sorted(M.items()) for x in lst]
        h.append("<h3>Cada m&eacute;todo solo, en TEST</h3><p class='nota'>Cada uno sobre las SNe que clasifica (por eso "
                 "no se comparan directo entre filas: para eso est&aacute; la tabla de arriba). F1 combina pureza y "
                 "completitud de cada clase (1 = perfecto).</p>" + tabla(
            ["clases", "m&eacute;todo", "clasificadas", "exactitud", "exact. balanceada [IC 95 %]", "F1 por clase"], rows))
    if jt and M:
        a3, t3 = jt["3"]["villar_alcanza"]["plantillas"], jt["3"]["total"]
        a4, t4 = jt["4"]["villar_alcanza"]["plantillas"], jt["4"]["total"]
        a5 = jt["5"]["villar_alcanza"]["plantillas"]
        cp = [x["cob"] for k in M for x in M[k] if x["lab"] == "Plantillas"]
        cv = [x["cob"] for k in M for x in M[k] if x["lab"] == "Villar"]
        rg = lambda v: f"{min(v):.0%}" if round(min(v), 2) == round(max(v), 2) else f"{min(v):.0%}&ndash;{max(v):.0%}"
        li_red = ""
        if "P_mejor_que_red" in t3["plantillas"]:
            pr = t3["plantillas"]["P_mejor_que_red"]
            li_red = (f"<li>Plantillas contra la red, cara a cara en TEST (3 clases): {_veredicto(pr)} (P = {_fp(pr)} en "
                      f"todas, {_fp(a3['P_mejor_que_red'])} en las que Villar alcanza)."
                      + (f" En val_sel las plantillas {_son(1 - rp['P_bal']).replace(' que', '')} (P = "
                         f"{_fp(1 - rp['P_bal'])}), y eso decidi&oacute; la elecci&oacute;n." if rp else "")
                      + (" Con la misma cobertura, el empate de TEST lo resuelve el &uacute;ltimo criterio de la regla: "
                         "las plantillas no dependen de las simulaciones." if _veredicto(pr) == "empate" else "")
                      + "</li>")
        h.append("<h3>Por qu&eacute; las plantillas</h3><ul>"
                 f"<li>3 clases: en las SNe que Villar alcanza, las plantillas {_son(a3['P_mejor_que_villar'])} Villar "
                 f"(P = {a3['P_mejor_que_villar']:.2f}). En todas las SNe tienen la cifra m&aacute;s alta "
                 f"({t3['plantillas']['bal']:.3f} contra {t3['villar_mas_respaldo']['bal']:.3f} de Villar con respaldo y "
                 f"{t3['red']['bal']:.3f} de la red), pero contra Villar con respaldo P = {_fp(t3['plantillas']['P_mejor'])}: "
                 f"{_veredicto(t3['plantillas']['P_mejor'])} (la red llega a P = {_fp(t3['red']['P_mejor'])}).</li>"
                 + li_red +
                 f"<li>4 y 5 clases, en las SNe que Villar alcanza: 4 clases {_veredicto(a4['P_mejor_que_villar'])} "
                 f"(P = {_fp(a4['P_mejor_que_villar'])}), 5 clases {_veredicto(a5['P_mejor_que_villar'])} "
                 f"(P = {_fp(a5['P_mejor_que_villar'])}). En todas las SNe con 4 clases, Villar con las plantillas de respaldo "
                 f"da {t4['villar_mas_respaldo']['bal']:.3f} y las plantillas solas {t4['plantillas']['bal']:.3f} "
                 f"(P = {t4['plantillas']['P_mejor']:.2f}, {_veredicto(t4['plantillas']['P_mejor'])}). En el empate la "
                 "regla prefiere el que depende menos de las simulaciones: las plantillas solas.</li>"
                 f"<li>Cobertura: las plantillas clasifican el {rg(cp)} de las SNe; Villar el {rg(cv)}. Cada SN sin "
                 "clasificar es una correcci&oacute;n m&aacute;s en la tasa.</li>"
                 "<li>No se entrenan con simulaciones de ZTF: no dependen de que la simulaci&oacute;n de la "
                 "observaci&oacute;n sea perfecta. El error que encontramos en esa simulaci&oacute;n (el umbral de "
                 "detecci&oacute;n, anexo F) afectaba a Villar y a la red, no a las plantillas.</li>"
                 "<li>Es el m&eacute;todo de SUDARE con nuestra biblioteca: al comparar tasas con SUDARE cambia la "
                 "biblioteca y no el m&eacute;todo, y SUDARE se puede clasificar sin simular SUDARE.</li>"
                 "<li>Dan una probabilidad por clase. El error que se le asigna al modelo (0.3 del flujo) se eligi&oacute; "
                 "en val_sel.</li></ul>")
        h.append("<h3>Limitaciones a declarar</h3><ul>"
                 "<li>Las plantillas II tienen plateaus m&aacute;s planos que las II de ZTF: parte de las II reales se va "
                 "a IIn e Ibc (fila II de las matrices)." + _recall_ii(M) + "</li>"
                 "<li>IIb e Ibc no se separan bien con fotometr&iacute;a (matriz de 5 clases). Por eso las tasas usan 4 "
                 "clases, con las IIb dentro de II, como SUDARE (secci&oacute;n 2).</li>"
                 "<li>Las fracciones por clase sin corregir est&aacute;n sesgadas. Para las tasas se corrigen con la matriz "
                 "de confusi&oacute;n (secci&oacute;n 4).</li></ul>"
                 "<p><b>Lo que se pide validar:</b> (1) las plantillas como clasificador oficial; (2) 4 clases para las "
                 "tasas; (3) correcci&oacute;n de las fracciones con la matriz de confusi&oacute;n.</p>")
    return "".join(h)


def seccion_sudare(J):
    """Seccion 2: como agrupa SUDARE I los tipos (verificado en el texto de Cappellaro+2015 y Botticella+2017) y la
    prueba de su agrupacion sobre nuestras predicciones congeladas (comparacion_justa.agrupacion_sudare)."""
    S, M = J.get("sudare"), (J.get("validacion") or {}).get("matrices") or {}
    h = ["<h2 id='sudare'>2. C&oacute;mo agrupa SUDARE los tipos y qu&eacute; cambia para nosotros</h2>",
         "<p><b>Lo que hace SUDARE</b> (le&iacute;do en el texto de Cappellaro et al. 2015, A&amp;A 584, A62, y Botticella "
         "et al. 2017, A&amp;A 598, A50):</p><ul>"
         "<li>Plantillas de Ia, Ib, Ic, IIb, IIP, IIL e IIn, m&aacute;s una superluminosa (SN 2008es) que sacaron porque "
         "ning&uacute;n candidato calzaba (&sect;4.1).</li>"
         "<li><b>Para calcular la probabilidad de tipo juntan II e IIn</b>: &laquo;for the purpose to assign probability, "
         "we merged regular type II and type IIn templates&raquo;. Despu&eacute;s marcan IIn cuando la plantilla de mayor "
         "probabilidad es una IIn (&sect;4.1).</li>"
         "<li>Las IIb van dentro de II: la mezcla de II que usan tiene 60 % IIP, 10 % tipo 2005cs, 10 % tipo 1987A, 10 % "
         "IIL y 10 % IIb (&sect;7.1).</li>"
         "<li><b>Las tasas s&iacute; separan las IIn</b>: &laquo;We considered separately the following main SN types: "
         "Ia, Ib/c, II (including IIP and IIL), IIn and SLSN&raquo; (&sect;7, Tabla 7). La tasa CC suma II + Ib/c + IIn "
         "(&sect;8.1).</li>"
         "<li>Declaran las IIn como lo m&aacute;s incierto: 40 % de incertidumbre en la clasificaci&oacute;n de Ib/c "
         "e IIn, contra 10 % en Ia y 25 % en II (&sect;4.4), y a z alto pierden o clasifican mal m&aacute;s de 2/3 de las IIn (&sect;8.2).</li>"
         "<li>SUDARE II usa la clasificaci&oacute;n del paper I. Da tasas de Ia y CC en el texto, y de II, Ibc e IIn por "
         "separado en el ap&eacute;ndice (Tabla 5).</li></ul>"
         "<p><b>Qu&eacute; significa para nosotros.</b> Las tasas que vamos a comparar son Ia, II (con IIb), Ibc, IIn y "
         "CC total. Por eso el clasificador oficial es de 4 clases. SUDARE junta II e IIn solo para la probabilidad; "
         "probamos esa misma regla con nuestras plantillas.</p>"]
    if not S:
        return "".join(h) + "<p class='pend'>Pendiente: python -m pipeline78.comparacion_justa (agrupacion_sudare).</p>"
    s4 = S["sudare4"]
    rows = []
    for sub in ("val_sel", "val_rep", "test"):
        x = s4[sub]
        rows.append([sub.replace("test", "TEST"), str(x["argmax"]["n"]),
                     f"{x['argmax']['bal']:.3f} (II {x['argmax']['recall']['II']:.2f}, IIn {x['argmax']['recall']['IIn']:.2f})",
                     f"{x['sudare']['bal']:.3f} (II {x['sudare']['recall']['II']:.2f}, IIn {x['sudare']['recall']['IIn']:.2f})",
                     f"{x['P_sudare_mejor']:.3f}"])
    v = s4["val_sel"]["P_sudare_mejor"]
    h.append("<h3>&iquest;Juntar II e IIn como SUDARE o separarlas con su propia probabilidad?</h3>"
             "<p>Las dos maneras sobre las mismas predicciones de 4 clases, ya congeladas (no se reajust&oacute; nada). "
             "&laquo;Separadas&raquo;: cada tipo con su probabilidad (lo que usamos). &laquo;A la SUDARE&raquo;: "
             "probabilidad de II + IIn juntas y despu&eacute;s IIn si la plantilla de mayor probabilidad es IIn. Cada "
             "celda: exactitud balanceada (recall de II y de IIn).</p>"
             + tabla(["conjunto", "SNe", "separadas", "a la SUDARE", "P(SUDARE mejor)"], rows)
             + f"<p>Decide val_sel: {'la regla de SUDARE es mejor' if v >= P_MIN else 'separadas es mejor' if v <= 1 - P_MIN else 'empate'} "
             f"(P = {v:.3f}). " + ("val_rep y TEST lo confirman. " if all(
                 (s4[s]["P_sudare_mejor"] <= 1 - P_MIN) == (v <= 1 - P_MIN) for s in ("val_rep", "test")) else "")
             + ("Con nuestras plantillas la probabilidad propia de las IIn funciona mejor que la regla de SUDARE, que "
                "manda m&aacute;s II a IIn. Nos quedamos con las clases separadas. " if v <= 1 - P_MIN else
                "La regla de val_sel no prefiere las clases separadas: decidir con Mauricio. ")
             + "Esta prueba se hizo despu&eacute;s de "
             "abrir TEST, pero no cambia ning&uacute;n modelo y la elecci&oacute;n se lee en val_sel.</p>")
    H = S["h3"]
    t = H["test"]
    va = t["villar_alcanza"]
    rows = [[sub.replace("test", "TEST"), str(H[sub]["plantillas"]["n"]), f"{H[sub]['plantillas']['acc']:.3f} / "
             f"{H[sub]['plantillas']['bal']:.3f}", f"{H[sub]['plantillas']['recall']['H']:.2f}"] for sub in ("val_sel", "val_rep", "test")]
    h.append("<h3>Si se juntan II e IIn en una sola clase &laquo;rica en hidr&oacute;geno&raquo;</h3>"
             "<p>Gran parte del error de las II es irse a IIn, dentro de la misma familia rica en hidr&oacute;geno. "
             "Juntando II + IIn (3 clases: Ia, II + IIn, Ibc), las plantillas dan (exactitud / exactitud balanceada, y "
             "recall de II + IIn):</p>" + tabla(["conjunto", "SNe", "exactitud / exact. balanceada", "recall II + IIn"], rows)
             + f"<p>En TEST, en las {va['villar']['n']} SNe que Villar alcanza: plantillas {va['plantillas']['bal']:.3f} "
             f"contra Villar {va['villar']['bal']:.3f} (P = {va['plantillas']['P_mejor_que_villar']:.2f}, "
             f"{_veredicto(va['plantillas']['P_mejor_que_villar'])}). En todas: plantillas {t['plantillas']['bal']:.3f} "
             f"contra Villar con respaldo {t['villar_mas_respaldo']['bal']:.3f} (P = {_fp(t['plantillas']['P_mejor'])}, "
             f"{_veredicto(t['plantillas']['P_mejor'])}).</p>")
    rows = []
    for lab in ("Plantillas", "Villar"):
        x = _mx(M, 4, lab)
        if not x:
            continue
        c = np.array(x["conf"], float)
        i = x["clases"].index("Ia")
        cc = [j for j in range(len(c)) if j != i]
        rows.append([lab + (" (solo las que alcanza)" if lab == "Villar" else ""), str(int(c.sum())),
                     f"{c[i, i] / c[i].sum():.2f}", f"{c[np.ix_(cc, cc)].sum() / c[cc].sum():.2f}",
                     f"{c[i, i] / c[:, i].sum():.2f}"])
    if rows:
        h.append("<h3>Ia contra colapso del n&uacute;cleo (lo que usan la tasa Ia y la tasa CC total)</h3>"
                 "<p>De las matrices de 4 clases en TEST: qu&eacute; fracci&oacute;n de las Ia se reconoce como Ia, "
                 "qu&eacute; fracci&oacute;n de las CC (II, Ibc, IIn) queda como CC, y de las que se llaman Ia, "
                 "cu&aacute;ntas lo son.</p>" + tabla(["m&eacute;todo", "SNe", "Ia reconocidas", "CC reconocidas",
                                                        "pureza de las Ia"], rows))
    return "".join(h)


def conclusiones_test(J):
    """Conclusiones principales, de TEST y de la regla de val_sel. Frases con condiciones fijas, cifras de los archivos."""
    V, T = J.get("validacion") or {}, J.get("tasas") or {}
    jt, M = V.get("justa"), V.get("matrices") or {}
    p3, p4 = _mx(M, 3, "Plantillas"), _mx(M, 4, "Plantillas")
    if not (jt and p3 and p4):
        return ["Pendiente: faltan los resultados de TEST."]
    meta = J.get("meta_bal_acc", 0.75)
    lo, hi = p3["ic"]
    out = [f"Clasificador propuesto: el ajuste de plantillas. En TEST, con 3 clases, clasifica el {p3['cob']:.0%} de las "
           f"SNe con exactitud balanceada {p3['bal']:.3f} [{lo:.3f}, {hi:.3f}] (IC 95 %). Contra la meta de {meta}: "
           + ("la supera con todo el intervalo." if lo > meta else "la cifra est&aacute; sobre la meta, pero el intervalo "
              "la incluye." if p3["bal"] >= meta else "queda bajo la meta.")
           + f" Con las 4 clases de las tasas: {p4['bal']:.3f} [{p4['ic'][0]:.3f}, {p4['ic'][1]:.3f}] sobre el "
           f"{p4['cob']:.0%}."]
    a = {k: jt[k]["villar_alcanza"]["plantillas"]["P_mejor_que_villar"] for k in ("3", "4", "5")}
    out.append("Contra Villar, en las SNe que Villar alcanza: " + ", ".join(
        f"{k} clases {_veredicto(p)} (P = {p:.2f})" for k, p in a.items())
        + (". Ninguna comparaci&oacute;n favorece a Villar, y las plantillas clasifican casi todas las SNe."
           if min(a.values()) > 1 - P_MIN else ". OJO: en alguna Villar es mejor."))
    pr = jt["3"]["total"]["plantillas"].get("P_mejor_que_red")
    rv = ((T.get("pares") or {}).get("val_sel") or {}).get("red_vs_plantillas")
    if pr is not None and rv:
        out.append(f"Contra la red, que tambi&eacute;n clasifica casi todo: en val_sel las plantillas {_son(1 - rv['P_bal'])} "
                   f"la red (P = {_fp(1 - rv['P_bal'])}) y en TEST, 3 clases, {_veredicto(pr)} (P = {_fp(pr)}). Con la misma "
                   "cobertura, la regla prefiere el m&eacute;todo que no depende de las simulaciones: las plantillas.")
    s = J.get("sudare")
    if s:
        x = s["sudare4"]["val_sel"]
        out.append(f"La regla de SUDARE para las IIn (juntar II + IIn y separar por la mejor plantilla) es "
                   f"{_veredicto(x['P_sudare_mejor'])} que separarlas con su propia probabilidad (val_sel, P = "
                   f"{x['P_sudare_mejor']:.3f}). Se usan las 4 clases separadas.")
    cr = T.get("corregidas_rep") or {}
    if cr.get("plantillas"):
        out.append("Para las tasas lo que importa es la fracci&oacute;n de cada tipo. Corregidas con la matriz de "
                   "confusi&oacute;n de val_sel, el error de las fracciones en val_rep (3 clases, suma de las diferencias con las "
                   f"fracciones reales, 0 = perfecto) es {cr['plantillas']['L1']:.3f} con las plantillas, "
                   + ", ".join(f"{cr[k]['L1']:.3f} con {NOM[k]}" for k in ("red", "villar") if k in cr) + ".")
    f = _fila(p4, "II")
    out.append(f"Limitaci&oacute;n principal: en 4 clases las plantillas reconocen el {f['II']:.0%} de las II (el resto va "
               f"a IIn {f['IIn']:.0%} y a Ibc {f['Ibc']:.0%}), porque las plantillas II tienen plateaus m&aacute;s "
               "planos que las II de ZTF. La correcci&oacute;n con la matriz de confusi&oacute;n compensa este sesgo "
               "en las fracciones, pero no lo elimina por SN.")
    return out


def particiones_conteo(real_dir):
    """Conteos por origen/split/excluir de meta_real_ztf.csv (solo esas columnas, sin fotometria ni clases)."""
    import csv
    from collections import Counter
    c, e = Counter(), Counter()
    with open(Path(real_dir) / "meta_real_ztf.csv", newline="") as f:
        for r in csv.DictReader(f):
            c[(r["origen"], r["split"])] += 1
            e[(r["origen"], r["split"])] += r["excluir"].strip().lower() in ("true", "1")
    return {f"{o}/{s}": {"n": n, "excluidas": e[(o, s)]} for (o, s), n in c.items()}


def _anexo(titulo, cuerpo, intro=""):
    return (f"<details><summary>{titulo}</summary><div class='anexo'>" + (f"<p class='nota'>{intro}</p>" if intro else "")
            + cuerpo + "</div></details>")


def seccion_tasas(J):
    """Seccion 3: las particiones de las SNe reales y la regla de eleccion (pipeline78/sistema_tasas.py), en simple. La
    tabla completa de VAL va plegada."""
    T, Pc = J.get("tasas"), J.get("particiones_conteo") or {}
    lab = {"holdout/val": "<b>VAL</b>: para elegir. Se parte mitad y mitad (por clase, semilla 20261004) en val_sel "
                          "(elegir) y val_rep (confirmar). La misma partici&oacute;n para los tres m&eacute;todos",
           "holdout/final": "<b>TEST</b>: se us&oacute; una sola vez, al final, con los modelos ya congelados",
           "viejas/val_viejo": "Otra muestra de ZTF, fuera de VAL y TEST: calibrar la simulaci&oacute;n de la "
                               "observaci&oacute;n (umbral de detecci&oacute;n, ruido, alertas)",
           "viejas/final_viejo": "Sin uso"}
    h = ["<h2 id='datos'>3. Los datos y la regla para elegir</h2>",
         "<p>Ning&uacute;n m&eacute;todo aprende de SNe reales. Villar y la red se entrenan con simulaciones (las 78 "
         "series proyectadas a ZTF con su cadencia y sus l&iacute;mites); las plantillas no se entrenan. Las SNe reales de "
         "ZTF con tipo espectrosc&oacute;pico (TNS) solo se usan para elegir y para medir:</p>",
         tabla(["conjunto", "SNe", "excluidas", "para qu&eacute;"],
               [[k, str(Pc[k]["n"]), str(Pc[k]["excluidas"]), lab.get(k, "")] for k in sorted(Pc, key=lambda k: (
                   k not in lab, list(lab).index(k) if k in lab else 0, k))]),
         "<ul><li>Las 78 SNe de las plantillas no est&aacute;n en VAL ni en TEST. Las superluminosas quedan fuera (no hay "
         "plantillas de ellas). Las excluidas son 4 dudosas que decidi&oacute; Mauricio y las de primera "
         "detecci&oacute;n tard&iacute;a.</li>"
         f"<li>Las simulaciones de entrenamiento: {_get(J, 'actual', 'sims_villar', 'n_sims')} para Villar y "
         f"{_get(J, 'actual', 'sims_red', 'n_sims')} para la red. Al validar con simulaciones, ninguna plantilla "
         "est&aacute; a la vez en entrenamiento y prueba.</li>"
         "<li>Incidentes declarados: el ruido de la simulaci&oacute;n se calibr&oacute; primero con VAL y despu&eacute;s "
         "se rehizo con la otra muestra; dos agentes imprimieron por error filas de metadata de TEST (ZTF18aacemcn, "
         "ZTF18aaiykoz), sin fotometr&iacute;a y sin ning&uacute;n uso.</li></ul>"]
    if not T:
        return "".join(h) + "<p class='pend'>Pendiente: correr python -m pipeline78.sistema_tasas.</p>"
    h.append("<p><b>La regla para elegir</b> (acordada con Mauricio el 2026-10-04, antes de ver val_rep y TEST): cada "
             "m&eacute;todo se mide como sistema completo sobre todas las SNe (las que no clasifica las clasifica el "
             f"respaldo, {html.escape(T['respaldo'])}). Gana el que mejora la exactitud balanceada con P &ge; 0.9 en "
             "val_sel. Si empatan, el que clasifica m&aacute;s SNe por s&iacute; mismo. Despu&eacute;s, el que depende "
             "menos de las simulaciones. val_rep no elige: solo confirma.</p>")
    rows = []
    for k in ("plantillas", "villar", "red"):
        a, b = T["por_subconjunto"]["val_sel"][k], T["por_subconjunto"]["val_rep"][k]
        cr = T.get("corregidas_rep", {}).get(k, {})
        rows.append([f"<b>{NOM[k]}</b> ({html.escape(T['metodos'][k])})", pc(a["cobertura_propia"]),
                     f3(a["bal_acc"]), f3(b["bal_acc"]), f3(b["acc"]), f3(a["L1_argmax"]), f3(b["L1_argmax"]),
                     f3(b["L1_prob"]), f"{f3(cr.get('L1'))} [{f3((cr.get('ic90') or [None, None])[0])}, "
                     f"{f3((cr.get('ic90') or [None, None])[1])}]"])
    t = [tabla(["sistema", "cobertura propia (sel)", "exact. bal. sel", "exact. bal. rep", "exactitud rep",
                "error fracciones sel", "error fracciones rep", "error fracciones rep (probabilidades)",
                "error fracciones rep corregidas con la matriz de val_sel [IC 90]"], rows)]
    lin = []
    for sub in ("val_sel", "val_rep"):
        for k, v in T["pares"][sub].items():
            x, z = k.split("_vs_")
            lin.append(f"<li>{sub}, {NOM[x]} contra {NOM[z]}: &Delta; exact. bal. {v['delta_bal']:+.3f} "
                       f"(P({NOM[x]} mejor) = {v['P_bal']:.3f}); P(error de fracciones de {NOM[x]} menor) = "
                       f"{v['P_L1_argmax_menor']:.3f} contando clases, {v['P_L1_prob_menor']:.3f} sumando probabilidades.</li>")
    t.append("<ul>" + "".join(lin) + "</ul>")
    t.append("<p class='nota'>Error de fracciones = suma sobre las clases de |fracci&oacute;n estimada &minus; "
             "fracci&oacute;n real| (0 = perfecto). Lectura: en exactitud balanceada las plantillas le ganan a la red en "
             "las dos mitades y quedan arriba de Villar (sin alcanzar P 0.9 en val_sel). En el error de fracciones sin "
             "corregir, el sistema de Villar sale mejor en val_sel pero no en val_rep. Todas las fracciones sin corregir "
             "est&aacute;n sesgadas: para las tasas se corrigen con la matriz de confusi&oacute;n (&uacute;ltima "
             "columna, medida en val_sel y aplicada a val_rep). Esa columna no es parte de la regla.</p>")
    h.append(_anexo("Tabla completa de VAL (incluye el error de las fracciones por clase)", "".join(t)))
    return "".join(h)


def pagina(J, figs, out):
    A, H, S, P = J["actual"], J.get("historia") or [], J["simulacion"], J["particion"]
    v, r = A.get("villar") or {}, A.get("red") or {}
    s = [f"<html><head><meta charset='utf-8'><title>Clasificador ZTF</title><style>{CSS}</style></head>"
         "<body style='font-family:sans-serif;max-width:1300px;margin:auto;padding:0 16px'>",
         "<h1>Clasificaci&oacute;n fotom&eacute;trica de las SNe de ZTF: qu&eacute; clasificador usa la tesis</h1>",
         f"<p class='nota'>Generada {J['generado']['fecha']} por pipeline78/informe_clasificadores.py (commit "
         f"{(J['generado']['git']['commit'] or '')[:8]}{', repo con cambios' if J['generado']['git']['dirty'] else ''}). "
         "Cada n&uacute;mero sale de los archivos de los runs al construir la p&aacute;gina "
         "(<a href='numeros.json'>numeros.json</a>).</p>"]
    if J.get("override"):
        s.append(f"<p class='aviso'><b>PRUEBA.</b> El bloque actual se apunt&oacute; por l&iacute;nea de comandos a "
                 f"{html.escape(str(J['override']))}. Son runs del modelo de observaci&oacute;n viejo: no es la "
                 "producci&oacute;n final.</p>")
    s.append(resumen(J))
    s.append("<h2>Palabras que se usan</h2><p class='nota'>" + "<br>".join(f"<b>{a}</b>: {b}" for a, b in GLOSARIO)
             + "</p>")
    s.append(seccion_validacion(J))
    s.append(seccion_sudare(J))
    s.append(seccion_tasas(J))
    s.append("<h2>4. Conclusiones</h2><ol>" + "".join(f"<li>{x}</li>" for x in J["conclusiones_test"]) + "</ol>")
    s.append("<h2>5. Pr&oacute;ximos pasos</h2><ul>" + "".join(f"<li>{x}</li>" for x in J.get("proximos") or []) + "</ul>")
    s.append("<h2>Anexos t&eacute;cnicos</h2><p>C&oacute;mo se lleg&oacute; hasta ac&aacute;. No hace falta leerlos para "
             "validar. Salvo que se diga otra cosa, las cifras de los anexos son de VAL (la etapa de elegir), no de TEST, "
             "y con 3 clases.</p>")
    # A: Villar contra la red (primera pregunta del proyecto)
    a = [f"<p>Bloque: {html.escape(str(A.get('etiqueta')))}. Redes: {html.escape(A['nn_root'])}. Villar: "
         f"{html.escape(A['villar_sweep'])}.</p>"]
    if ok(v) and ok(r):
        R, Ssel = A["respuesta"]["val_rep"], A["respuesta"]["val_sel"]
        mo = R["mismos_objetos"]
        row = lambda n, q, m: [n, q, pc(m.get("cobertura")), f3(m.get("acc")),
                               f"{f3(m.get('bal_acc'))}{ci(m.get('bal_acc_ic95'))}", str(m.get("n"))]
        a.append(tabla(["m&eacute;todo", "qu&eacute; es", "cobertura", "exactitud", "exact. balanceada [IC 95 %]", "n"], [
            row("<b>Villar</b>", v["que_es"], v["val_rep"]),
            row(f"<b>Mejor red</b> ({html.escape(r['name'])})", r["que_es"], r["val_rep"]),
            row("Villar, mismos objetos", "las SNe que Villar puede clasificar", mo["villar"]),
            row("Red, mismos objetos", "la red sobre esas mismas SNe", mo["red"]),
            row("<b>H&iacute;brido fijo</b>", f"Villar donde ajusta ({R['hibrido']['n_villar']}), la red en el resto "
                f"({R['hibrido']['n_red']}). Regla fijada antes de mirar", R["hibrido"])]))
        d, ds, D = mo["red_menos_villar"], Ssel["mismos_objetos"], J["decision"]
        rv, hr = D["red_vs_villar"], D["hibrido_vs_red"]
        a.append(f"<ul><li>En los mismos {mo['n']} objetos (val_rep) la diferencia de exactitud balanceada red &minus; "
                 f"Villar es <b>{d.get('delta', float('nan')):+.3f}</b> (intervalo 90 % {ci(d.get('ic90_delta'))}).</li>"
                 f"<li>Para decidir se mira val_sel ({ds['n']} objetos comunes): P(red mejor) = "
                 f"{f3(ds['red_menos_villar'].get('p_mejora'), 3)}, P(Villar mejor) = "
                 f"{f3(ds['villar_menos_red'].get('p_mejora'), 3)}. Gana el que pase {P_MIN}: "
                 f"<b>{NOM[rv['val_sel']] if rv['val_sel'] else 'ninguno'}</b>. {txt_rep(rv)}</li>"
                 f"<li>H&iacute;brido contra la red sola (val_sel, objetos de la red): P(h&iacute;brido mejor) = "
                 f"{f3(hr['p_sel_pos'], 3)}, P(red mejor) = {f3(hr['p_sel_neg'], 3)}: "
                 f"<b>{NOM[hr['val_sel']] if hr['val_sel'] else 'ninguno'}</b>. {txt_rep(hr)}</li>"
                 f"<li>Entre Villar y la red, la regla recomienda: <b>{NOM[D['recomendado']] if D['recomendado'] else 'ninguno'}"
                 "</b>. Esa pregunta qued&oacute; superada por las plantillas (secci&oacute;n 1).</li></ul>")
        if figs.get("respuesta"):
            a.append(f"<img src='{figs['respuesta']}' width='520'>")
    else:
        a.append("<p class='pend'>Pendiente: " + " y ".join(
            n for n, x in (("Villar", v), ("la red", r)) if not ok(x)) + " del bloque actual todav&iacute;a no tienen "
                 "resultados.</p>")
    el = A.get("eleccion_red") or {}
    a.append("<h3>C&oacute;mo se midi&oacute;</h3><ul>"
             f"<li>Entrenamiento: <b>solo simulaciones</b> de las 78 series espectrales proyectadas a ZTF. Red: "
             f"{html.escape(str((A.get('sims_red') or {}).get('run')))} ({(A.get('sims_red') or {}).get('n_sims')} sims). "
             f"Villar: {html.escape(str((A.get('sims_villar') or {}).get('run')))} "
             f"({(A.get('sims_villar') or {}).get('n_sims')} sims proyectadas, {v.get('n_sims_con_features')} con "
             "par&aacute;metros de Villar). Ninguna SN real entra al entrenamiento.</li>"
             f"<li>Reales: VAL, {P['n_val']} SNe Ia/II/Ibc (Ia {P['por_clase']['Ia']}, II {P['por_clase']['II']}, Ibc "
             f"{P['por_clase']['Ibc']}; II incluye IIb, IIn fuera). val_sel {P['n_val_sel']}, val_rep {P['n_val_rep']}, "
             f"semilla {P['semilla']}. Estos scripts no leen TEST (splits.read_val_meta solo guarda VAL). TEST se "
             "evalu&oacute; aparte, una vez, con pipeline78/evaluar_test.py.</li>"
             f"<li>Regla: bootstrap pareado en val_sel ({P['n_boot_pareado']} remuestreos). La candidata reemplaza a la "
             f"incumbente m&aacute;s simple solo si P(mejora) &ge; {P['p_min']}.</li>"
             f"<li>Mejor Villar: la elegida por el barrido ({html.escape(str(v.get('motivo')))}).</li>"
             f"<li>Mejor red: entre las redes de la ra&iacute;z, la primera en val_sel ({el.get('candidata')}) contra "
             f"{el.get('incumbente')}: {html.escape(str(el.get('motivo')))}"
             + (f" (P = {f3(_get(el, 'comparacion', 'p_mejora'), 3)})" if _get(el, "comparacion", "p_mejora") is not None else "")
             + ".</li>"
             f"<li>IC 95 %: bootstrap de {P['n_boot_ic']} remuestreos de las SNe (semilla {P['semilla']}).</li></ul>")
    a.append("<h3>Frases de esta etapa</h3><p class='nota'>Generadas desde numeros.json con condiciones fijas. Son de "
             "VAL y de la comparaci&oacute;n Villar contra la red; las conclusiones finales est&aacute;n en la "
             "secci&oacute;n 4.</p><ol>" + "".join(f"<li>{x}</li>" for x in J["conclusiones"]) + "</ol>")
    s.append(_anexo("A. Primera etapa: Villar contra la red en VAL", "".join(a)))
    s.append(_anexo("B. Las plantillas en VAL", seccion_plantillas(J)))
    c = ["<p>&#9733; = el elegido de cada familia con la regla en val_sel (no necesariamente el de mayor cifra en "
         "val_rep). Las cifras son de val_rep salvo la columna val_sel. Referencias: Villar+2019 (ApJ 884, 83), ATAT = "
         "Cabrera-Vives+2024 (A&amp;A 689, A289), ORACLE-2 = Shah+2026 (arXiv:2607.00228, preprint), SuperNNova = "
         "M&ouml;ller &amp; de Boissi&egrave;re 2020 (MNRAS 491, 4277), Gupta+2025 (MNRAS 542, L132).</p>", tabla_modelos(A)]
    for i, B in enumerate(H):
        c.append(f"<details><summary>Historia {i + 1}: {html.escape(str(B.get('etiqueta')))}</summary><div class='viejo'>"
                 "<b>Modelo de observaci&oacute;n viejo y equivocado</b> (detecci&oacute;n al 50 % 1.25 mag sobre el "
                 "l&iacute;mite, ruido a 1 &sigma;, sin stream de alertas). Solo historia: estas cifras no son las de la "
                 "tesis.</div>" + tabla_modelos(B) + "</details>")
    s.append(_anexo("C. Todos los modelos probados", "".join(c)))
    s.append(_anexo("D. Matrices de confusi&oacute;n en val_rep (Villar y la red)",
                    f"<p>Filas = tipo real, columnas = tipo asignado. Cada celda: fracci&oacute;n de la fila y "
                    f"(n&uacute;mero). Izquierda Villar, derecha {html.escape(r.get('name', ''))}.</p>"
                    f"<img src='{figs['confusion']}' width='900'>" if figs.get("confusion") else "<p class='pend'>Pendiente.</p>"))
    e = []
    if figs.get("ndet"):
        e.append("<p>Izquierda: la red con menos detecciones (misma muestra de curvas con &ge; 7 detecciones, raleada "
                 "a 3, 5, 7 o todas). Centro: la red cortando la curva 10, 20 o 50 d despu&eacute;s de la primera "
                 "detecci&oacute;n. Derecha: por n&uacute;mero de detecciones g + r de cada SN real (val_rep). Villar "
                 "no se puede ralear (necesita el ajuste completo), as&iacute; que se muestra su cobertura.</p>"
                 f"<img src='{figs['ndet']}' width='1000'>")
        con_pl = any("acc_plantillas" in x for x in A.get("ndet") or [])
        e.append(tabla(["detecciones g + r", "n", "n Villar", "cobertura Villar", "cobertura red", "exactitud Villar",
                        "exactitud red (mismos objetos)", "exactitud red (todas)", "exactitud h&iacute;brido"]
                       + (["n plantillas", "exactitud plantillas"] if con_pl else []),
                       [[x["bin"], str(x["n"]), str(x["n_villar"]), pc(x["cobertura_villar"]), pc(x["cobertura_red"]), f3(x["acc_villar"]),
                         f3(x["acc_red_mismos"]), f3(x["acc_red"]), f3(x["acc_hibrido"])]
                        + ([str(x.get("n_plantillas", "")), f3(x.get("acc_plantillas"))] if con_pl else [])
                        for x in A.get("ndet") or []]))
    else:
        e.append("<p class='pend'>Pendiente (falta Villar o la red del bloque actual).</p>")
    e.append("<h3>&iquest;M&aacute;s simulaciones ayudan?</h3>")
    pts = J.get("aprendizaje") or []
    if pts:
        if figs.get("aprendizaje"):
            e.append(f"<img src='{figs['aprendizaje']}' width='900'><p class='nota'>L&iacute;nea punteada y "
                     "s&iacute;mbolo vac&iacute;o: modelo de observaci&oacute;n viejo. L&iacute;nea llena: modelo nuevo. "
                     "No mezclar los dos para leer el efecto del tama&ntilde;o.</p>")
        e.append(tabla(["corrida", "sims", "n sims", "modelo de observaci&oacute;n", "exact. bal. val_sel",
                        "exact. bal. val_rep"],
                       [[html.escape(p["corrida"]), p["sims"], str(p["n_sims"]), p["modelo"], f3(p["bal_sel"]),
                         f3(p["bal_rep"])] for p in sorted(pts, key=lambda p: (p["corrida"], p["n_sims"] or 0))]))
        pr = J.get("aprendizaje_pares") or []
        if pr:
            e.append("<p>Misma corrida, mismo modelo de observaci&oacute;n, m&aacute;s sims (bootstrap pareado en "
                     "val_sel):</p>" + tabla(["corrida", "de", "a", "&Delta; val_sel", "P(mejora)", "gana",
                                              "&Delta; val_rep"],
                                             [[html.escape(p["corrida"]), f"{p['de']} ({p['n_de']})", f"{p['a']} ({p['n_a']})",
                                               f"{p['sel'].get('delta', float('nan')):+.3f}", f3(p["sel"].get("p_mejora"), 3),
                                               "s&iacute;" if p["sel"].get("gana") else "no",
                                               f"{p['delta_rep']:+.3f}" if p["delta_rep"] is not None else "&mdash;"]
                                              for p in pr]))
    else:
        e.append("<p class='pend'>Pendiente.</p>")
    s.append(_anexo("E. D&oacute;nde gana cada uno (n&uacute;mero de detecciones) y curva de aprendizaje", "".join(e)))
    f = ["<p>La primera producci&oacute;n usaba un modelo de observaci&oacute;n equivocado: la simulaci&oacute;n "
         "detectaba una SN reci&eacute;n 1.25 mag sobre el l&iacute;mite (las reales se detectan hasta el l&iacute;mite), "
         "sorteaba el ruido al doble del real y no imitaba el stream de alertas de ALeRCE. Un clasificador &laquo;sim "
         "contra real&raquo; lo encontr&oacute;. Se recalibr&oacute; por &eacute;poca con SNe que no son de VAL ni "
         "TEST. Afecta a Villar y a la red (entrenados con simulaciones), no a las plantillas.</p>"]
    c = S.get("calib") or {}
    filas = [_fmt_obs(mo) for mo in S.get("modelos_obs") or []]
    pm = c.get("piloto_modelo")
    if pm and pm["dir"] not in {m["dir"] for m in S.get("modelos_obs") or []}:
        filas.append(_fmt_obs(pm, "piloto de calib_obs: "))
    f.append(tabla(["proyecci&oacute;n", "det_m0 [mag]", "det_w [mag]", "escala del ruido k", "stream de alertas",
                    "piso de la cola [mag/d]"], filas))
    if c.get("estado") == "ok":
        pil, cp = c["etiqueta_despues"], c.get("piloto_como_produccion")
        mu = {Path(str(c.get("piloto"))).name: pil, Path(str(c.get("base"))).name: "antes"}
        if figs.get("calib"):
            f.append(f"<img src='{figs['calib']}' width='1000'>" + (
                f"<p class='nota pend'>OJO, figura vieja: generada {c.get('figura_fecha')}, antes de las tablas "
                f"confirm_* ({c.get('tablas_fecha')}). Puede mostrar otro piloto, no el de las tablas de abajo. Gris: "
                "alertas reales. Color: el piloto con que se hizo la figura. Guiones: modelo viejo.</p>"
                if c.get("figura_vieja") else
                f"<p class='nota'>Figura de calib_obs (generada {c.get('figura_fecha')}). Gris: alertas reales. Color: "
                f"{pil}. Guiones: modelo viejo.</p>"))
        if cp is False:
            f.append(f"<p class='aviso'><b>Ojo.</b> &laquo;Despu&eacute;s&raquo; es el {pil}, no la producci&oacute;n "
                     f"({', '.join(c.get('produccion') or [])}), que usa " + ", ".join(dict.fromkeys(
                         f"{CAMPOS_OBS[x['campo']]} {_gr(x['produccion'])}" for x in c["diferencias_produccion"]))
                     + ". Falta repetir confirm con la config final.</p>")
        elif cp is None:
            f.append("<p class='aviso'>No se pudo comparar el piloto con las sims de producci&oacute;n de la config.</p>")
        f.append(f"<p>Blancos de la calibraci&oacute;n contra {c.get('n_real')} SNe {c.get('split')} (tablas del "
                 f"{c.get('tablas_fecha')}. Antes = {html.escape(Path(str(c.get('base'))).name)}, despu&eacute;s = "
                 f"{pil}). dm = m_lim &minus; m: qu&eacute; tan cerca del l&iacute;mite se detecta.</p>")
        f.append(tabla(["banda", "blanco", "real", "antes", f"despu&eacute;s ({pil})"],
                       [[b["banda"], b["blanco"].replace("<", "&lt;"), f3(b["real"]), f3(b["antes"]), f3(b["despues"])]
                        for b in c.get("blancos") or []]))
        if c.get("puntaje"):
            f.append(f"<p>Puntaje (suma de |sim &minus; real| / error, 0 = calce perfecto), antes &rarr; {pil}: g "
                     f"{c['puntaje']['g'][1]:.1f} &rarr; {c['puntaje']['g'][0]:.1f}, r {c['puntaje']['r'][1]:.1f} "
                     f"&rarr; {c['puntaje']['r'][0]:.1f}.</p>")
        if c.get("tripletes"):
            f.append("<p>Ruido: dispersi&oacute;n de tres detecciones seguidas en unidades del error reportado (las "
                     "reales dan ~0.55: el error de ZTF sobreestima el ruido).</p>" +
                     tabla(["banda", "muestra", "dispersi&oacute;n", "error", "n"],
                           [[t["banda"], mu.get(str(t["muestra"]), html.escape(str(t["muestra"]))), f3(t["rstd"]),
                             f3(t["err"]), str(t["n"])] for t in c["tripletes"]]))
        if c.get("chequeos"):
            cols = c["chequeos_cols"]
            f.append(tabla(["chequeo", "real"] + [mu.get(x, html.escape(x)) for x in cols],
                           [[x["chequeo"], fnum(x["real"])] + [fnum(x[k]) for k in cols] for x in c["chequeos"]]))
    else:
        f.append("<p class='pend'>Calibraci&oacute;n: pendiente.</p>")
    ga, gd = S.get("gap_antes") or {}, A.get("gap") or {}
    f.append("<h3>Clasificador sim contra real (AUC: 0.5 = no se distinguen, 1 = se distinguen siempre)</h3>")
    f.append(tabla(["", "AUC", "sims", "reales", "lo que m&aacute;s delata a la simulaci&oacute;n"],
                   [[f"antes ({html.escape(Path(str(g.get('dir'))).name)})" if i == 0 else
                     f"despu&eacute;s ({html.escape(Path(str(g.get('dir'))).name)})",
                     f3(g.get("auc")) if g.get("estado") == "ok" else "<span class='pend'>pendiente</span>",
                     str(g.get("n_sims", "")), str(g.get("n_real", "")),
                     ", ".join(f"{t['feature']} ({t['caida_auc']:.3f})" for t in g.get("top") or [])]
                    for i, g in enumerate((ga, gd))]))
    s.append(_anexo("F. Calibraci&oacute;n de la simulaci&oacute;n de ZTF (para Villar y la red)", "".join(f)))
    s.append(_anexo("G. Pendientes t&eacute;cnicos", "<ul>" + "".join(f"<li>{x}</li>" for x in J["pendientes"]) + "</ul>"))
    nv = J.get("verificacion") or []
    s.append(f"<p class='nota'>Verificaci&oacute;n: {sum(x['ok'] for x in nv)} de {len(nv)} cifras de los anexos coinciden "
             "con el metrics.json de su run (diferencia &lt; 1e-9).</p></body></html>")
    (out / "index.html").write_text("".join(s))


# ------------------------------------------------------------------------------------------------ principal
def construir(cfg_path=CFG, actual_nn=None, actual_villar=None, actual_gap=None, atlas=True):
    cfg = json.loads(Path(cfg_path).read_text())
    cfg["_runs"] = _p(cfg["runs_root"], Path.home()) if cfg.get("runs_root") else RUNS
    over = {k: v for k, v in (("nn_root", actual_nn), ("villar_sweep", actual_villar), ("gap", actual_gap)) if v}
    act = {**cfg["actual"], **over}
    if over:
        act["etiqueta"] = f"PRUEBA con {over} (la config dice: {cfg['actual'].get('etiqueta')})"
    V = particion(_p(cfg.get("real_dir", "real_ztf"), cfg["_runs"]))
    inc, bins = cfg.get("nn_incumbente", "gru_base"), cfg.get("bins_ndet", [3, 8, 15, 25])
    A = bloque(act, cfg, V, inc, bins)
    H = [bloque(b, cfg, V, inc, bins) for b in cfg.get("historia", [])]
    pts, pares = aprendizaje([("actual", A)] + [(f"historia {i + 1}", B) for i, B in enumerate(H)])
    mos, vistos = [], set()
    esperadas = [modelo_obs(_p(act.get(k), cfg["_runs"])) for k in ("sims_nn", "sims_villar") if act.get(k)]
    esperadas = list({m["dir"]: m for m in esperadas if m}.values())
    for B in H + [A] + [{"sims_red": m} for m in esperadas]:
        for mo in (B.get("sims_red"), B.get("sims_villar")):
            if mo and mo["dir"] not in vistos:
                vistos.add(mo["dir"])
                mos.append(mo)
    gap_antes = next((B["gap"] for B in H if ok(B["gap"])), {"estado": "pendiente"})
    N = {"generado": {"fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "git": git_info(),
                      "config": str(cfg_path)},
         "override": over or None,
         "particion": {"semilla": splits.SEED, "n_val": int(len(V)), "n_val_sel": int((V.subset == "val_sel").sum()),
                       "n_val_rep": int((V.subset == "val_rep").sum()), "por_clase": V.cls.value_counts().to_dict(),
                       "n_boot_ic": N_BOOT_IC, "n_boot_pareado": N_BOOT, "p_min": P_MIN, "semilla_pareado": BOOT_SEED},
         "actual": A, "historia": H, "aprendizaje": pts, "aprendizaje_pares": pares,
         "simulacion": {"calib": calib(cfg, esperadas), "modelos_obs": mos, "gap_antes": gap_antes},
         "verificacion": [x for B in [A] + H for x in verificar(B)], "meta_bal_acc": cfg.get("meta_bal_acc", 0.75)}
    J = _js(N)
    J["decision"] = decidir(J["actual"], J["meta_bal_acc"]) if J["actual"].get("respuesta") else None
    J["decision_plantillas"] = decidir_plantillas(J["actual"]) if J["actual"].get("respuesta_plantillas") else None
    J["conclusiones"] = conclusiones(J, J["meta_bal_acc"])
    J["pendientes"] = pendientes(J, cfg)
    ft = _p(cfg.get("numeros", "informe_clasificadores/numeros.json"), cfg["_runs"]).parent / "sistema_tasas.json"
    J["tasas"] = json.loads(ft.read_text()) if ft.exists() else None
    J["validacion"] = leer_validacion(cfg["_runs"])
    fs = Path(cfg["_runs"]) / "test_final" / "agrupacion_sudare.json"
    J["sudare"] = json.loads(fs.read_text()) if fs.exists() else None
    J["conclusiones_test"] = conclusiones_test(J)
    J["proximos"] = [html.escape(x) for x in cfg.get("proximos_pasos", [])]
    out = _p(cfg["pagina"], PHD)
    out.mkdir(parents=True, exist_ok=True)
    plt, figs = _mpl(), {}
    if ok(A["villar"]) and ok(A["red"]):
        figs["respuesta"] = fig_respuesta(A, out, plt)
        figs["confusion"] = fig_confusion(A, out, plt)
        figs["ndet"] = fig_ndet(A, out, plt)
    figs["aprendizaje"] = fig_aprendizaje(pts, out, plt)
    cf = Path(J["simulacion"]["calib"]["figura"]) if J["simulacion"]["calib"].get("figura_existe") else None
    if cf:
        shutil.copy2(cf, out / cf.name)
        figs["calib"] = cf.name
    J["figuras"] = {k: v for k, v in figs.items() if v}
    fm = fig_matrices_test(J["validacion"], out, plt)
    if fm:
        figs["matrices_test"] = fm
        J["figuras"]["matrices_test"] = fm
    J["particiones_conteo"] = particiones_conteo(_p(cfg.get("real_dir", "real_ztf"), cfg["_runs"]))
    pagina(J, figs, out)
    txt = json.dumps(J, indent=1, ensure_ascii=False)
    (out / "numeros.json").write_text(txt)
    nj = _p(cfg.get("numeros", "informe_clasificadores/numeros.json"), cfg["_runs"])
    nj.parent.mkdir(parents=True, exist_ok=True)
    nj.write_text(txt)
    res = {"pagina": str(out / "index.html"), "numeros": str(nj), "verificacion_mal": sum(not x["ok"] for x in J["verificacion"])}
    if atlas and cfg.get("atlas"):
        dst = _p(cfg["atlas"], Path.home())
        dst.mkdir(parents=True, exist_ok=True)
        for f in out.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        res["atlas"] = str(dst)
        if cfg.get("url"):
            res["url"] = cfg["url"]
            res["http"] = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", cfg["url"]],
                                         capture_output=True, text=True).stdout
    return J, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--actual-nn", help="raiz nnclf del bloque actual (reemplaza la de la config, prueba)")
    ap.add_argument("--actual-villar", help="barrido clf_villar del bloque actual (reemplaza el de la config, prueba)")
    ap.add_argument("--actual-gap", help="gap clf_villar del bloque actual")
    ap.add_argument("--sin-atlas", action="store_true", help="no copiar a atlas_local")
    a = ap.parse_args(argv)
    J, res = construir(a.config, a.actual_nn, a.actual_villar, a.actual_gap, not a.sin_atlas)
    A = J["actual"]
    for k in ("villar", "red"):
        x = A.get(k) or {}
        print(f"[informe] {k}: {x.get('name', '')} {x.get('estado')}"
              + (f" rep acc {x['val_rep']['acc']:.3f} bal {x['val_rep']['bal_acc']:.3f} cob {x['val_rep']['cobertura']:.3f}"
                 if x.get("estado") == "ok" else ""))
    for c in J["conclusiones"]:
        print("  -", html.unescape(c))
    print(f"[informe] {res}")
    return res


if __name__ == "__main__":
    main()
