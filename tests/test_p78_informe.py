# tests/test_p78_informe.py
"""informe_clasificadores sobre un arbol de runs falso: numeros.json = archivos fuente, y lo que falta = pendiente."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from pipeline78 import splits
from pipeline78.clf_villar import bootstrap_ci
from pipeline78 import informe_clasificadores as I

CLS = ["Ia", "II", "Ibc"]
SIG = {"Ia": "II", "II": "Ibc", "Ibc": "Ia"}
FINAL = "ZTF99final000"


def _meta(runs):
    rows = []
    for c, t in (("Ia", "Ia"), ("II", "II"), ("Ibc", "Ibc")):
        for k in range(12):
            rows.append(dict(oid=f"ZTF20{c}{k:03d}", sn_type="IIb" if (c == "II" and k < 3) else t, subtipo="", z=0.05,
                             split="val", origen="holdout", part_index=0, excluir=False, motivo=""))
    rows += [dict(oid="ZTF20IIn000", sn_type="IIn", subtipo="", z=0.05, split="val", origen="holdout", part_index=0,
                  excluir=False, motivo=""),
             dict(oid="ZTF20exc000", sn_type="Ia", subtipo="", z=0.05, split="val", origen="holdout", part_index=0,
                  excluir=True, motivo="no es SN"),
             dict(oid=FINAL, sn_type="Ia", subtipo="", z=0.05, split="final", origen="holdout", part_index=0,
                  excluir=False, motivo="")]
    (runs / "real_ztf").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(runs / "real_ztf/meta_real_ztf.csv", index=False)
    v = splits.read_val_meta(runs / "real_ztf/meta_real_ztf.csv")
    sel, rep = splits.val_split(v)
    sub = {**{o: "val_sel" for o in sel}, **{o: "val_rep" for o in rep}}
    v["cls"] = v.sn_type.map(splits.CLASS_OF)
    v = v[v.cls.isin(CLS)].sort_values("oid").reset_index(drop=True)
    v["subset"] = v.oid.map(sub)
    v["i"] = np.arange(len(v))
    return v


def _sims(runs, name, n, alertas):
    d = runs / name
    d.mkdir(parents=True)
    pd.DataFrame({"field": [f"f{i}" for i in range(n)]}).to_parquet(d / "_sims_all.parquet")
    cfg = {"det_m0": {"g": -0.25, "r": 0.0} if alertas else 1.25, "det_w": 0.2,
           "noise_draw_scale": {"g": 0.52, "r": 0.54} if alertas else None, "tail_min_slope": 0.005}
    if alertas:
        cfg["alert_model"] = {"m50": {"g": 0.1}}
    (d / "run_manifest.json").write_text(json.dumps({"run": name, "cfg": cfg, "git": "abc", "dirty": False}))
    return d


def _nn_run(root, name, v, sim_run, modo="base"):
    d = root / name
    d.mkdir(parents=True)
    p = v[v.i % 9 != 4].copy()                                   # la red no ve las de menos de 3 detecciones
    p["n_det"] = 3 + p.i % 30
    p["y_true"] = p.cls
    err = {"base": lambda i: i % 3 == 0, "malo": lambda i: i % 2 == 0 or i % 3 == 0, "bueno": lambda i: i % 11 == 0}[modo]
    p["y_pred"] = [SIG[c] if err(i) else c for c, i in zip(p.cls, p.i)]
    for c in CLS:
        p[f"p_{c}"] = (p.y_pred == c).astype(float)
    p[["oid", "subset", "y_true", "y_pred", "n_det"] + [f"p_{c}" for c in CLS]].to_csv(d / "pred_real_val.csv",
                                                                                       index=False)
    (d / "config.json").write_text(json.dumps({"model": "gru", "use_z": False, "sim_run": str(sim_run),
                                               "n_params": 1000, "fold": 0, "trunc": "none"}))
    q = p[p.subset == "val_rep"]
    m = {"acc": accuracy_score(q.y_true, q.y_pred), "bal_acc": balanced_accuracy_score(q.y_true, q.y_pred),
         **{f"f1_{c}": f for c, f in zip(CLS, f1_score(q.y_true, q.y_pred, labels=CLS, average=None, zero_division=0))}}
    cov = len(q) / int((v.subset == "val_rep").sum())
    (d / "metrics.json").write_text(json.dumps({"method": "nn", "use_z": False, "main_by_subset": {
        "val_rep": {"coverage": cov, "metrics": m}}}))
    deg = [dict(mode="fixed", bands=b, N=n, n_eval=10, bal_acc_mean=0.4 + 0.05 * k, bal_acc_std=0.01, subset="val_rep")
           for b in ("r", "g+r") for k, n in enumerate(("3", "5", "7", "all"))]
    pd.DataFrame(deg).to_csv(d / "degradation_fixed.csv", index=False)
    pd.DataFrame([{**x, "mode": "horizon", "N": h} for x, h in zip(deg, ("10d", "20d", "50d", "all") * 2)]).to_csv(
        d / "degradation_horizon.csv", index=False)
    return p, m, cov


def _villar(cv, v, features_sims):
    d = cv / "sw"
    (d / "mejor").mkdir(parents=True)
    p = v[v.i % 2 == 0].copy()
    p["y_true"] = p.cls
    p["y_pred"] = [SIG[c] if i % 4 == 0 else c for c, i in zip(p.cls, p.i)]
    p = pd.concat([p, pd.DataFrame([dict(oid=FINAL, subset="val_rep", cls="Ia", y_true="Ia", y_pred="Ia")])])
    p["sn_type"], p["tiene_r"] = p.cls, True
    for c in CLS:
        p[f"p_{c}"] = (p.y_pred == c).astype(float)
    p[["oid", "subset", "sn_type", "y_true", "tiene_r", "y_pred"] + [f"p_{c}" for c in CLS]].to_csv(
        d / "mejor/pred_real_val.csv", index=False)
    q = p[(p.subset == "val_rep") & (p.oid != FINAL)]
    y, yp = q.y_true.map(I.IX).to_numpy(), q.y_pred.map(I.IX).to_numpy()
    m = {"acc": accuracy_score(y, yp), "bal_acc": balanced_accuracy_score(y, yp),
         "bal_acc_ic95": bootstrap_ci(y, yp, CLS, 1000, splits.SEED)["bal_acc_ic95"]}
    cov = len(q) / int((v.subset == "val_rep").sum())
    (d / "mejor/metrics.json").write_text(json.dumps({"real_none": {"val_rep": m}, "cobertura": {"val_rep": {
        "cobertura": cov}}, "n_sims": 30, "features_real": "x"}))
    cfg = {"model": "hgb", "fset": "rg", "use_z": True, "peso": "wz", "balance": True, "g_modo": "nan"}
    (d / "mejor.json").write_text(json.dumps({"elegida": cfg, "motivo": "queda la base", "comparacion": None}))
    pd.DataFrame([{"name": "sw", "features_sims": str(features_sims)}]).to_csv(d / "sweep.csv", index=False)
    return p[p.oid != FINAL], m, cov


def _arbol(tmp):
    runs = tmp / "runs"
    v = _meta(runs)
    sims_nn, sims_v = _sims(runs, "ztf_nuevo_x4", 80, True), _sims(runs, "ztf_nuevo_x2", 40, True)
    _nn_run(runs / "nnclf_fake", "gru_base", v, sims_nn)                 # incumbente
    _nn_run(runs / "nnclf_fake", "tf_base", v, sims_nn, "malo")
    pn, mn, cn = _nn_run(runs / "nnclf_fake", "gru_attnpool", v, sims_nn, "bueno")   # gana con la regla
    (runs / "nnclf_fake/pend_run").mkdir()                          # corriendo: sin metrics.json
    pv, mv, cvv = _villar(runs / "clf_villar", v, runs / "features_ztf_nuevo_x2")
    (runs / "clf_villar/gap_viejo").mkdir(parents=True)
    (runs / "clf_villar/gap_viejo/gap.json").write_text(json.dumps({
        "auc_sim_vs_real": 0.9123, "n_sims": 30, "n_real": 20, "peso": "wz",
        "top": [{"feature": "rms_g", "caida_auc": 0.1, "mediana_sim": 1.0, "mediana_real": 2.0}]}))
    (runs / "calib_obs").mkdir()
    pd.DataFrame([dict(piloto="p", band=b, blanco=k, real=0.1 * j, sim=0.11 * j, dif=0.0, e_real=0.01, e_sim=0.01,
                       sim_base=0.5, e_sim_base=0.01) for b in ("g", "r") for j, k in enumerate(("f_dm<0", "med_dm"))]
                 ).to_csv(runs / "calib_obs/confirm_blancos.csv", index=False)
    (runs / "calib_obs/confirm_meta.json").write_text(json.dumps({
        "split": "val_viejo", "piloto": str(sims_nn), "base": str(sims_v), "puntaje": {"g": [7.0, 150.0], "r": [10.0, 130.0]},
        "real": {"n_usable": 5}}))
    fig = tmp / "calib_obs.png"
    fig.write_bytes(b"png")
    cfg = {"runs_root": str(runs), "real_dir": "real_ztf", "clf_villar_root": "clf_villar", "pagina": str(tmp / "pagina"),
           "atlas": None, "url": None, "numeros": "informe/numeros.json", "nn_incumbente": "gru_base",
           "meta_bal_acc": 0.75, "bins_ndet": [3, 8, 15, 25],
           "actual": {"etiqueta": "nuevo", "nn_root": "nnclf_fake", "villar_sweep": "sw", "gap": "gap_nuevo",
                      "sims_nn": "ztf_nuevo_x4", "sims_villar": "ztf_nuevo_x2"},
           "historia": [{"etiqueta": "viejo", "nn_root": "nnclf_viejo", "villar_sweep": "sw_viejo", "gap": "gap_viejo"}],
           "calib_obs": {"dir": "calib_obs", "figura": str(fig)}, "pendientes": ["algo a mano"]}
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    return runs, v, (pn, mn, cn), (pv, mv, cvv)


def test_numeros_igual_a_los_archivos(tmp_path):
    runs, v, (pn, mn, cn), (pv, mv, cvv) = _arbol(tmp_path)
    J, res = I.construir(tmp_path / "cfg.json", atlas=False)
    disk = json.loads((runs / "informe/numeros.json").read_text())
    assert disk == json.loads((tmp_path / "pagina/numeros.json").read_text())
    A = disk["actual"]
    # Villar elegido: metrics.json del barrido
    vr = A["villar"]["val_rep"]
    assert abs(vr["acc"] - mv["acc"]) < 1e-12 and abs(vr["bal_acc"] - mv["bal_acc"]) < 1e-12
    assert np.allclose(vr["bal_acc_ic95"], mv["bal_acc_ic95"], atol=1e-12)
    assert abs(vr["cobertura"] - cvv) < 1e-12
    assert A["villar"]["fuera_de_val"] == 1                         # la oid final del archivo se descarta
    # mejor red: gru_attnpool le gana a la incumbente gru_base en val_sel (P >= 0.9); metrics.json de la red
    assert A["red"]["name"] == "gru_attnpool" and A["eleccion_red"]["comparacion"]["gana"]
    rr = A["red"]["val_rep"]
    assert abs(rr["acc"] - mn["acc"]) < 1e-12 and abs(rr["bal_acc"] - mn["bal_acc"]) < 1e-12
    assert all(abs(rr[f"f1_{c}"] - mn[f"f1_{c}"]) < 1e-12 for c in CLS)
    assert abs(rr["cobertura"] - cn) < 1e-12
    assert disk["verificacion"] and all(x["ok"] for x in disk["verificacion"])
    # hibrido calculado a mano: Villar si cubre, si no la red, sobre val_rep
    vv, nn = dict(zip(pv.oid, pv.y_pred)), dict(zip(pn.oid, pn.y_pred))
    q = v[v.subset == "val_rep"]
    yp = [vv.get(o, nn.get(o)) for o in q.oid]
    k = [x is not None for x in yp]
    h = A["respuesta"]["val_rep"]["hibrido"]
    yt, ypk = q.cls[k].tolist(), [x for x in yp if x is not None]
    assert h["n"] == sum(k) and abs(h["cobertura"] - sum(k) / len(q)) < 1e-12
    assert abs(h["acc"] - accuracy_score(yt, ypk)) < 1e-12
    assert abs(h["bal_acc"] - balanced_accuracy_score(yt, ypk)) < 1e-12
    # mismos objetos: la red sobre las oids de Villar en val_rep
    ov = set(pv.oid[pv.subset == "val_rep"])
    qn = pn[(pn.subset == "val_rep") & pn.oid.isin(ov)]
    assert abs(A["respuesta"]["val_rep"]["mismos_objetos"]["red"]["bal_acc"]
               - balanced_accuracy_score(qn.y_true, qn.y_pred)) < 1e-12
    # simulacion: gap viejo, calib y aprendizaje
    assert disk["simulacion"]["gap_antes"]["auc"] == 0.9123 and A["gap"]["estado"] == "pendiente"
    b = pd.read_csv(runs / "calib_obs/confirm_blancos.csv")
    assert [x["real"] for x in disk["simulacion"]["calib"]["blancos"]] == b.real.tolist()
    assert [x["antes"] for x in disk["simulacion"]["calib"]["blancos"]] == b.sim_base.tolist()
    assert {p["n_sims"] for p in disk["aprendizaje"]} == {80, 40}
    # pendientes y pagina
    pend = " ".join(disk["pendientes"])
    assert "pend_run" in pend and "algo a mano" in pend and "ya no" in pend
    page = (tmp_path / "pagina/index.html").read_text()
    assert "pendiente" in page and FINAL not in page and FINAL not in json.dumps(disk)
    for f in ("fig_respuesta.png", "fig_confusion.png", "fig_ndet.png", "fig_aprendizaje.png", "calib_obs.png"):
        assert (tmp_path / "pagina" / f).exists(), f


def test_actual_faltante_queda_pendiente(tmp_path):
    _arbol(tmp_path)
    J, res = I.construir(tmp_path / "cfg.json", actual_nn="nnclf_no_existe", actual_villar="sw_no_existe", atlas=False)
    A = J["actual"]
    assert A["villar"]["estado"] == "pendiente" and A["red"]["estado"] == "pendiente" and A["nn_estado"] == "pendiente"
    assert J["override"] == {"nn_root": "nnclf_no_existe", "villar_sweep": "sw_no_existe"}
    page = (tmp_path / "pagina/index.html").read_text()
    assert "PRUEBA" in page and "pendiente" in page.lower()
    assert any("nnclf_no_existe" in p for p in J["pendientes"])
