"""Tests del clasificador oficial sobre features de Villar (pipeline78/clf_villar.py).

python -m pytest tests/test_p78_clf_villar.py -q
"""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pipeline78 import clf_villar as C

FEAT_COLS = ["sn_name", "filter_band"] + C.RAW + ["sn_type", "oid", "part_index"]


@pytest.fixture(autouse=True)
def _dos_hilos():
    """OpenMP/BLAS con todos los nucleos sobre-suscribe la CPU con modelos chicos (163 s contra 75 s con 2 hilos)."""
    with threadpool_limits(2):
        yield


def _feat_row(oid, part, st, band, A=1e-8, f=0.5, t_rise=3.0, t_fall=40.0, gamma=20.0, rng=None):
    rng = rng or np.random.default_rng(0)
    r = {"sn_name": f"{oid}_{st}_p{part:02d}", "filter_band": band, "A": A, "f": f, "t_rise": t_rise,
         "t_fall": t_fall, "gamma": gamma, "n_points": 12, "time_span": 60.0, "rms": 1e-9,
         "sn_type": st, "oid": oid, "part_index": part}
    for p in C.PARS:
        r[f"{p}_err"] = abs(r[p]) * 0.1 * (1 + rng.random())
    return r


def _write_sims(root, rows, meta):
    fd = root / "features_fake"
    (fd / "features").mkdir(parents=True)
    pd.DataFrame(rows, columns=FEAT_COLS).to_csv(fd / "features" / "features.csv", index=False)
    rd = root / "fake"
    rd.mkdir()
    pd.DataFrame(meta).to_parquet(rd / "_sims_all.parquet")
    return fd, rd


def _sims_meta(field, part, st, z=0.05, w=1.0, template="T1"):
    return {"field": field, "part_index": part, "sn_type": st, "template": template, "subtype": st, "z": z, "w_z": w}


# ----------------------------------------------------------------------------- llaves (bug H4)
def test_merge_por_oid_part_index_y_sn_type(tmp_path):
    """Mismo (oid, part_index) para dos tipos: r con g y la metadata se juntan dentro de cada sn_type."""
    rows = [_feat_row("ZTFa", 0, "Ia", "r", A=1e-8), _feat_row("ZTFa", 0, "Ia", "g", A=1e-8 * 10 ** (0.4 * 0.2)),
            _feat_row("ZTFa", 0, "II", "r", A=2e-8), _feat_row("ZTFa", 0, "II", "g", A=2e-8 * 10 ** (-0.4 * 0.5))]
    meta = [_sims_meta("ZTFa", 0, "Ia", z=0.02, template="Ta"), _sims_meta("ZTFa", 0, "II", z=0.08, template="Tb")]
    fd, rd = _write_sims(tmp_path, rows, meta)
    S = C.derive(C.load_sims(fd, rd))
    assert len(S) == 2                                          # el bug H4 daba 4 filas (2 x 2)
    S = S.set_index("sn_type")
    assert np.isclose(S.loc["Ia", "color_gr"], -0.2) and np.isclose(S.loc["II", "color_gr"], 0.5)
    assert S.loc["Ia", "z"] == 0.02 and S.loc["II", "z"] == 0.08
    assert S.loc["Ia", "template"] == "Ta" and S.loc["II", "cls"] == "II"


def test_widen_rechaza_llave_repetida():
    rows = [_feat_row("ZTFa", 0, "Ia", "r"), _feat_row("ZTFa", 0, "Ia", "r")]
    with pytest.raises(ValueError):
        C.widen(pd.DataFrame(rows))


def test_sims_sin_metadata_es_error(tmp_path):
    rows = [_feat_row("ZTFa", 0, "Ia", "r"), _feat_row("ZTFa", 1, "Ia", "r")]
    fd, rd = _write_sims(tmp_path, rows, [_sims_meta("ZTFa", 0, "Ia")])
    with pytest.raises(ValueError):
        C.load_sims(fd, rd)


# ----------------------------------------------------------------------------- la final no se carga
def _write_real(root, n_val=6, n_final=6):
    rd = root / "real_ztf"
    rd.mkdir()
    meta, feats = [], []
    tipos = ["Ia", "II", "Ibc", "IIb", "IIn"]
    for i in range(n_val + n_final):
        split = "val" if i < n_val else "final"
        st = tipos[i % len(tipos)]
        oid = f"ZTF{split}{i:03d}"
        meta.append({"oid": oid, "sn_type": st, "subtipo": st, "z": 0.05, "split": split, "origen": "holdout",
                     "part_index": 0, "excluir": i == 1})
        feats += [_feat_row(oid, 0, st, "r"), _feat_row(oid, 0, st, "g")]
    meta.append({"oid": "ZTFviejo", "sn_type": "Ia", "subtipo": "Ia", "z": 0.05, "split": "val_viejo",
                 "origen": "viejas", "part_index": 0, "excluir": False})
    pd.DataFrame(meta).to_csv(rd / "meta_real_ztf.csv", index=False)
    fr = root / "features_real"
    (fr / "features").mkdir(parents=True)
    pd.DataFrame(feats, columns=FEAT_COLS).to_csv(fr / "features" / "features.csv", index=False)
    return fr, rd, pd.DataFrame(meta)


def test_mitad_final_nunca_se_carga(tmp_path, monkeypatch):
    fr, rd, meta = _write_real(tmp_path)
    final = set(meta.oid[meta.split != "val"])
    vistos = []
    orig = C._to_num

    def espia(df):                                  # todo lo que llega a pandas desde los csv pasa por _to_num
        vistos.extend(df.oid.astype(str).tolist())
        return orig(df)

    def prohibido(*a, **k):
        raise AssertionError("pd.read_csv sobre los csv de reales: leeria la mitad final")

    monkeypatch.setattr(C, "_to_num", espia)
    monkeypatch.setattr(C.pd, "read_csv", prohibido)
    R, v = C.load_real_val(fr, rd)
    assert vistos and not set(vistos) & final
    assert not set(R.oid) & final and not set(v.oid) & final
    assert "ZTFval001" not in set(v.oid)            # excluir = True
    assert set(v.sn_type) <= {"Ia", "II", "IIb", "Ibc"}  # IIn fuera de las 3 clases
    assert set(R.cls) <= {"Ia", "II", "Ibc"}


def test_oid_val_repetida_fuera_de_val_es_error(tmp_path):
    fr, rd, meta = _write_real(tmp_path)
    meta = pd.concat([meta, meta.iloc[[0]].assign(split="final")])
    meta.to_csv(rd / "meta_real_ztf.csv", index=False)
    with pytest.raises(ValueError):
        C.read_val_meta(rd / "meta_real_ztf.csv")


# ----------------------------------------------------------------------------- S(m) sin etiquetas
def test_peso_S_no_usa_etiquetas():
    assert list(inspect.signature(C.selection_weight).parameters)[:3] == ["m_sim", "w_sim", "m_real"]
    rng = np.random.default_rng(1)
    n = 3000
    S = pd.DataFrame({"m_sel": rng.normal(19.0, 0.8, n), "w_z": rng.uniform(0.2, 2, n),
                      "cls": rng.choice(["Ia", "II", "Ibc"], n)})
    R = pd.DataFrame({"m_sel": rng.normal(18.4, 0.5, 400), "cls": rng.choice(["Ia", "II", "Ibc"], 400)})
    w1, _ = C.phys_weights(S, R, "wz_S", [])
    w2, _ = C.phys_weights(S, R.assign(cls=rng.permutation(R.cls.to_numpy())), "wz_S", [])
    w3, _ = C.phys_weights(S, R[["m_sel"]], "wz_S", [])             # sin columna de etiquetas
    assert np.array_equal(w1, w2) and np.array_equal(w1, w3)
    # y acerca la distribucion de m de las sims a la de las reales
    med = lambda x, w: x[np.argsort(x)][np.searchsorted(np.cumsum(w[np.argsort(x)]), 0.5 * w.sum())]
    m = S.m_sel.to_numpy()
    assert abs(med(m, w1) - 18.4) < 0.1 < abs(med(m, S.w_z.to_numpy()) - 18.4)


def test_peso_S_normalizado_y_recortado():
    rng = np.random.default_rng(2)
    m, w = rng.normal(19, 1, 2000), rng.uniform(0.5, 1.5, 2000)
    s, tab = C.selection_weight(m, w, rng.normal(18.5, 0.6, 300))
    assert np.isclose(np.average(s, weights=w), 1.0, atol=0.05)
    assert s.min() >= 0.02 - 1e-12 and s.max() <= 20 + 1e-12
    assert {"m", "S", "p_sim", "p_real"} <= set(tab.columns)


# ----------------------------------------------------------------------------- (1+z) y magnitud absoluta
def test_tiempos_en_reposo_y_magnitud_absoluta():
    W = C.widen(pd.DataFrame([_feat_row("a", 0, "Ia", "r", t_rise=11.0, t_fall=44.0, gamma=22.0),
                              _feat_row("a", 0, "Ia", "g", t_rise=13.2, t_fall=55.0, gamma=33.0),
                              _feat_row("b", 0, "Ia", "r", t_rise=11.0)]))
    W["z"] = [0.1, np.nan]
    F = C.derive(W)
    a, b = F.iloc[0], F.iloc[1]
    assert np.isclose(a.t_rise_r, 10.0) and np.isclose(a.t_fall_r, 40.0) and np.isclose(a.gamma_r, 20.0)
    assert np.isclose(a.d_t_rise_gr, 2.0) and np.isclose(a.d_gamma_gr, 10.0)
    assert np.isclose(b.t_rise_r, 11.0) and np.isnan(b.M_pk_r)       # sin z: marco observado y sin M
    assert np.isclose(C.distmod(np.array([0.1]))[0], 38.3152, atol=2e-3)   # = core.utils.DL_calculator
    assert np.isclose(a.M_pk_r, a.m_pk_r - C.distmod(np.array([0.1]))[0])
    assert np.isclose(a.rel_t_rise_r, W.t_rise_err_r[0] / 11.0)       # error relativo: no cambia con 1+z
    Fo = C.derive(W, rest_frame=False)
    assert np.isclose(Fo.t_rise_r[0], 11.0)


def test_reales_tambien_en_reposo(tmp_path):
    fr, rd, meta = _write_real(tmp_path)
    m = pd.read_csv(rd / "meta_real_ztf.csv")
    m.loc[m.oid == "ZTFval000", "z"] = 0.25
    m.to_csv(rd / "meta_real_ztf.csv", index=False)
    R, _ = C.load_real_val(fr, rd)
    F = C.derive(R).set_index("oid")
    assert np.isclose(F.loc["ZTFval000", "t_rise_r"], 3.0 / 1.25)


def test_sin_z_no_entra_magnitud_absoluta():
    for fs in C.FSETS:
        c1, c2 = C.fset_cols(fs, use_z=False)
        assert not [c for c in c1 + c2 if c.startswith("M_pk")]
        assert any(c.startswith("M_pk") for c in C.fset_cols(fs, use_z=True)[1])


# ----------------------------------------------------------------------------- piezas
def test_template_folds_no_parten_plantillas():
    S = pd.DataFrame({"sn_type": ["Ia"] * 50 + ["II"] * 30 + ["IIb"] * 20,
                      "template": [f"Ia{i % 10}" for i in range(50)] + [f"II{i % 6}" for i in range(30)]
                      + [f"IIb{i % 5}" for i in range(20)]})
    f = C.template_folds(S, 5, seed=3)
    assert (f >= 0).all()
    assert (S.assign(f=f).groupby("template").f.nunique() == 1).all()
    for st in ("Ia", "II", "IIb"):
        assert set(f[(S.sn_type == st).to_numpy()]) == set(range(5))


def test_class_balance_y_em_prior():
    y = np.r_[np.zeros(100, int), np.ones(50, int), np.full(10, 2)]
    sw = C.class_balance(y, np.ones(len(y)))
    assert np.allclose(np.bincount(y, weights=sw), len(y) / 3)
    sw2 = C.class_balance((y == 2).astype(int), np.ones(len(y)), [2 / 3, 1 / 3])
    assert np.isclose(sw2[y == 2].sum() / sw2.sum(), 1 / 3)
    # EM recupera un prior conocido con probabilidades bien calibradas
    rng = np.random.default_rng(4)
    true = np.array([0.6, 0.3, 0.1])
    yy = rng.choice(3, 6000, p=true)
    L = rng.normal(0, 1, (6000, 3))
    L[np.arange(6000), yy] += 1.5
    lik = np.exp(-0.5 * ((L[:, :, None] - 1.5 * np.eye(3)[None]) ** 2).sum(1))   # p(x|k)
    P = lik / lik.sum(1, keepdims=True)                                            # posterior con prior uniforme
    _, pi = C.em_prior(P, np.full(3, 1 / 3))
    assert np.allclose(pi, true, atol=0.03)


def _fake_world(root, n_per=60, seed=5):
    """Sims y reales sinteticas separables por (f, t_fall, M) con 6 plantillas por tipo."""
    rng = np.random.default_rng(seed)
    proto = {"Ia": (0.5, 35.0, -19.2), "II": (0.15, 110.0, -17.3), "IIb": (0.3, 60.0, -17.5),
             "Ibc": (0.7, 50.0, -18.0), "IIn": (0.2, 150.0, -18.5)}
    rows, meta = [], []
    for st, (f, tf, M) in proto.items():
        for i in range(n_per):
            fld, z = f"ZTF{st}{i:03d}", rng.uniform(0.02, 0.1)
            A = 10 ** (-0.4 * (M + rng.normal(0, 0.3) + C.distmod(np.array([z]))[0]))
            for b in ("r", "g"):
                if b == "g" and rng.random() < 0.3:
                    continue
                rows.append(_feat_row(fld, 0, st, b, A=A, f=np.clip(f + rng.normal(0, 0.08), 0.01, 0.99),
                                      t_fall=tf * (1 + z) * rng.lognormal(0, 0.2), t_rise=3 * (1 + z),
                                      gamma=30.0, rng=rng))
            meta.append(_sims_meta(fld, 0, st, z=z, w=rng.uniform(0.5, 1.5), template=f"{st}_T{i % 6}"))
    fd, rd = _write_sims(root, rows, meta)
    # reales: misma fisica
    real = root / "real_ztf"
    real.mkdir()
    m, feats = [], []
    for st, (f, tf, M) in proto.items():
        for i in range(25):
            oid, z = f"ZTFr{st}{i:03d}", rng.uniform(0.02, 0.1)
            A = 10 ** (-0.4 * (M + rng.normal(0, 0.3) + C.distmod(np.array([z]))[0]))
            for b in ("r", "g"):
                feats.append(_feat_row(oid, 0, st, b, A=A, f=np.clip(f + rng.normal(0, 0.08), 0.01, 0.99),
                                       t_fall=tf * (1 + z) * rng.lognormal(0, 0.2), t_rise=3 * (1 + z),
                                       gamma=30.0, rng=rng))
            m.append({"oid": oid, "sn_type": st, "subtipo": st, "z": z, "split": "val" if i < 15 else "final",
                      "origen": "holdout", "part_index": 0, "excluir": False})
    pd.DataFrame(m).to_csv(real / "meta_real_ztf.csv", index=False)
    fr = root / "features_real"
    (fr / "features").mkdir(parents=True)
    pd.DataFrame(feats, columns=FEAT_COLS).to_csv(fr / "features" / "features.csv", index=False)
    return fd, rd, fr, real


def test_modelos_probabilidades_validas(tmp_path):
    fd, rd, fr, real = _fake_world(tmp_path)
    S, R, v = C.prepare(fd, rd, fr, real)
    c1, c2 = C.fset_cols("viejo", True)
    for name in ("hgb", "rf", "hier_mlp_II", "hier_rf_Ia", "hier_hgb_Ia", "ens", "ens_hier"):
        m = C.Model(name, c1, c2, 3, seed=1).fit(S, S.y.to_numpy(), S.w_z.to_numpy())
        P = m.predict_proba(R)
        assert P.shape == (len(R), 3) and np.allclose(P.sum(1), 1) and (P >= 0).all()
        assert (P.argmax(1) == R.y.to_numpy()).mean() > 0.8, name


def test_cli_train_eval_sweep_gap(tmp_path):
    fd, rd, fr, real = _fake_world(tmp_path)
    out = tmp_path / "out"
    common = ["--features-sims", str(fd), "--run-dir", str(rd), "--features-real", str(fr), "--real-dir", str(real),
              "--out-root", str(out), "--folds", "3"]
    res = C.main(["train", "--name", "t1", "--model", "hgb", "--fset", "rg"] + common)
    rv, rr, rs = (res["real_none"][s] for s in ("val", "val_rep", "val_sel"))
    assert rv["acc"] > 0.8 and rv["n"] == 60                   # 15 val x (Ia, II, IIb, Ibc); IIn fuera
    assert rr["n"] + rs["n"] == 60 and rr["n"] == 31           # val_split: Ia 7/8, II+IIb 15/15, Ibc 7/8
    for r in (rv, rr, rs):
        lo, hi = r["acc_ic95"]
        assert lo <= r["acc"] <= hi
    for f in ("metrics.json", "model.joblib", "pred_real_val.csv", "confusion_none_rep.csv", "confusion_em_sel.csv",
              "confusion_none_val.csv", "importancias_real_val.csv"):
        assert (out / "t1" / f).exists(), f
    pred = pd.read_csv(out / "t1" / "pred_real_val.csv")
    assert pred.oid.str.startswith("ZTFr").all() and len(pred) == 60
    assert set(pred.subset) == {"val_sel", "val_rep"}
    imp = pd.read_csv(out / "t1" / "importancias_real_val.csv")
    assert set(imp.subset) == {"val_sel", "val_rep"}
    ev = C.main(["eval", "--name", "t1", "--features-real", str(fr), "--real-dir", str(real), "--out-root", str(out)])
    for s in C.SUBSETS:
        assert np.isclose(ev["real_none"][s]["acc"], res["real_none"][s]["acc"])
        assert np.isclose(ev["real_em"][s]["bal_acc"], res["real_em"][s]["bal_acc"])
    tab = C.main(["sweep", "--name", "sw", "--grid", "rapido", "--models", "hgb,rf"] + common)
    assert len(tab) == 2 * 2 * 2 * 2 and (out / "sw" / "mejor" / "metrics.json").exists()
    assert tab.es_base.sum() == 2 and tab.elegida.sum() == 2   # una configuracion, dos priors
    best = json.loads((out / "sw" / "mejor.json").read_text())
    assert best["elegida"]["model"] in ("hgb", "rf") and best["base"] == C.BASELINE
    assert (out / "sw" / "mejor" / "model.joblib").exists()
    # la base se agrega si la grilla no la tiene
    tab2 = C.main(["sweep", "--name", "sw2", "--grid", "rapido", "--models", "rf", "--fsets", "viejo"] + common)
    assert tab2.es_base.sum() == 2 and set(tab2[tab2.es_base].model) == {"hgb"}
    g = C.main(["gap", "--name", "gp", "--peso", "wz"] + common)
    assert 0.0 <= g["auc_sim_vs_real"] <= 1.0 and (out / "gp" / "gap_importancias.csv").exists()
    resumen = pd.read_csv(out / "resumen.csv")
    assert set(resumen.name) == {"t1", "sw", "sw2"} and set(resumen.prior) == {"none", "em"}
    assert list(resumen.columns) == C.RESUMEN_COLS
    t1 = resumen[(resumen.name == "t1") & (resumen.prior == "none")].iloc[0]
    assert np.isclose(t1.bal_acc_rep, rr["bal_acc"]) and np.isclose(t1.bal_acc_sel, rs["bal_acc"])
    assert np.isclose(t1.coverage_rep, 1.0) and t1.n_rep == 31


# ----------------------------------------------------------------------------- seleccion anidada (revision B)
CFG_HGB = {"model": "hgb", "fset": "rg", "use_z": True, "peso": "wz_S", "balance": True}


def test_D1_etiquetas_val_no_entrenan_ni_calibran(tmp_path):
    """Permutar las etiquetas de las reales val no cambia P, Q (EM) ni la temperatura (mata M9 y M10)."""
    fd, rd, fr, real = _fake_world(tmp_path)
    S, R, _ = C.prepare(fd, rd, fr, real)
    perm = np.random.default_rng(0).permutation(len(R))
    Rp = R.assign(y=R.y.to_numpy()[perm], cls=R.cls.to_numpy()[perm], sn_type=R.sn_type.to_numpy()[perm])
    for cfg in (CFG_HGB, {**CFG_HGB, "model": "hier_hgb_II", "peso": "wz_dr"}):
        a = C.run_config(S, R, cfg, C.classes(), folds=3, keep_model=True)
        b = C.run_config(S, Rp, cfg, C.classes(), folds=3, keep_model=True)
        assert np.allclose(a["_P"], b["_P"], rtol=0, atol=1e-12) and np.allclose(a["_Q"], b["_Q"], rtol=0, atol=1e-12)
        assert a["temperatura"] == b["temperatura"] and a["prior_em"] == b["prior_em"]
        assert a["real_none"]["val"]["bal_acc"] != b["real_none"]["val"]["bal_acc"]   # el test no es trivial


def test_D1b_val_rep_no_ajusta_nada(tmp_path):
    """Cambiar las features de val_rep no cambia S(m), wz_dr, la temperatura ni el prior EM, ni las predicciones de
    val_sel: val_rep solo se reporta."""
    fd, rd, fr, real = _fake_world(tmp_path)
    S, R, _ = C.prepare(fd, rd, fr, real)
    rep = (R.subset == "val_rep").to_numpy()
    assert 0 < rep.sum() < len(R)
    num = [c for c in R.columns if c.startswith(("m_pk", "M_pk", "f_", "t_", "gamma", "color", "d_", "rel_"))
           or c == "m_sel"]
    Rr = R.copy()
    Rr.loc[rep, num] = Rr.loc[rep, num].to_numpy() + 0.7
    for cfg in (CFG_HGB, {**CFG_HGB, "peso": "wz_dr"}):
        a = C.run_config(S, R, cfg, C.classes(), folds=3, keep_model=True)
        b = C.run_config(S, Rr, cfg, C.classes(), folds=3, keep_model=True)
        assert a["pesos"] == b["pesos"] and a["temperatura"] == b["temperatura"] and a["prior_em"] == b["prior_em"]
        assert np.allclose(a["_P"][~rep], b["_P"][~rep], rtol=0, atol=1e-12)
        assert np.allclose(a["_Q"][~rep], b["_Q"][~rep], rtol=0, atol=1e-12)
        assert not np.allclose(a["_P"][rep], b["_P"][rep])                       # val_rep si se predice


def test_D2_cv_disjunta_por_plantilla(tmp_path, monkeypatch):
    """Espia en Model.fit: en la CV de run_config ninguna plantilla del fold de prueba entrena (mata M8)."""
    fd, rd, fr, real = _fake_world(tmp_path)
    S, R, _ = C.prepare(fd, rd, fr, real)
    modelos, fit0, pred0 = [], C.Model.fit, C.Model.predict_proba

    def fit(self, F, y, w):
        self.tpl_train = set(zip(F.sn_type, F.template)) if "template" in F else None
        modelos.append(self)
        return fit0(self, F, y, w)

    def pred(self, F):
        if "template" in F:                                     # sims (las reales no tienen plantilla)
            self.tpl_test, self.n_test = set(zip(F.sn_type, F.template)), len(F)
        return pred0(self, F)

    monkeypatch.setattr(C.Model, "fit", fit)
    monkeypatch.setattr(C.Model, "predict_proba", pred)
    C.run_config(S, R, {**CFG_HGB, "peso": "wz"}, C.classes(), folds=4)
    cv = [m for m in modelos if hasattr(m, "tpl_test")]
    assert len(cv) == 4 and len(modelos) == 5                   # 4 folds + el modelo con todas las sims
    for m in cv:
        assert m.tpl_test and not (m.tpl_train & m.tpl_test)
    assert sum(m.n_test for m in cv) == len(S)
    assert set().union(*(m.tpl_test for m in cv)) == set(zip(S.sn_type, S.template))


def test_D3_metrics_matriz_conocida_y_con_r():
    """Recall por fila y precision por columna sobre una matriz conocida (mata M11); con_r es el subconjunto con r
    (mata M12)."""
    cls = ("Ia", "II", "Ibc")
    cm = np.array([[8, 1, 1], [4, 5, 1], [0, 3, 2]])          # filas verdaderas, columnas predichas
    y = np.concatenate([np.full(cm[i, j], i) for i in range(3) for j in range(3)]).astype(int)
    yp = np.concatenate([np.full(cm[i, j], j) for i in range(3) for j in range(3)]).astype(int)
    m = C.metrics(y, yp, cls)
    rec, pre = np.array([0.8, 0.5, 0.4]), np.array([8 / 12, 5 / 9, 2 / 4])
    f1 = 2 * pre * rec / (pre + rec)
    assert m["confusion"] == cm.tolist() and np.isclose(m["acc"], 15 / 25) and np.isclose(m["bal_acc"], rec.mean())
    assert np.allclose([m[f"recall_{c}"] for c in cls], rec) and np.allclose([m[f"f1_{c}"] for c in cls], f1)
    assert np.isclose(m["f1_macro"], f1.mean())
    # con_r y subconjuntos en evaluate_real
    n = len(y)
    rng = np.random.default_rng(0)
    R = pd.DataFrame({"oid": [f"o{i}" for i in range(n)], "y": y, "tiene_r": rng.random(n) < 0.6,
                      "subset": np.where(np.arange(n) % 2 == 0, "val_sel", "val_rep")})
    P = np.eye(3)[yp] * 0.9 + 0.1 / 3
    ev = C.evaluate_real(R, P, P, cls, n_boot=50, nn_oids={"o0", "o1", "o2"})
    assert ev["none"]["val"]["confusion"] == cm.tolist()
    for s in C.SUBSETS:
        k = np.ones(n, bool) if s == "val" else (R.subset == s).to_numpy()
        kr = k & R.tiene_r.to_numpy()
        assert ev["none"][s]["n"] == k.sum() and ev["none"][s]["con_r"]["n"] == kr.sum()
        assert ev["none"][s]["con_r"]["confusion"] == C.metrics(y[kr], yp[kr], cls)["confusion"]
    assert ev["none"]["val"]["nn_oids"]["n"] == 3 and ev["none"]["val_sel"]["nn_oids"]["n"] == 2


def test_D4_prior_efectivo_jerarquico_uniforme():
    """Con features sin informacion, la media de P sobre las sims es 1/K por clase, tambien en el jerarquico (mata
    M13: nivel 1 balanceado 50/50 favoreceria a la primera clase)."""
    y = np.repeat([0, 1, 2], [300, 150, 60])
    cols = ["f_r", "t_fall_r"]
    F = pd.DataFrame(0.0, index=range(len(y)), columns=cols)
    for name in ("hgb", "rf", "hier_hgb_II", "hier_hgb_Ia", "hier_rf_Ia"):
        P = C.Model(name, cols, cols, 3, seed=1).fit(F, y, np.ones(len(y))).predict_proba(F)
        assert np.allclose(P.mean(0), 1 / 3, atol=0.02), (name, P.mean(0))


def test_D5_gap_no_usa_etiquetas(tmp_path):
    """El AUC sim contra real y sus importancias no cambian al permutar las etiquetas de las reales (mata M14)."""
    fd, rd, fr, real = _fake_world(tmp_path)
    out = tmp_path / "out"
    common = ["--features-sims", str(fd), "--run-dir", str(rd), "--features-real", str(fr), "--real-dir", str(real),
              "--out-root", str(out), "--folds", "3", "--peso", "wz"]
    g1 = C.main(["gap", "--name", "g1"] + common)
    meta = pd.read_csv(real / "meta_real_ztf.csv")
    ix = meta.index[(meta.split == "val") & meta.sn_type.isin(["Ia", "II", "IIb", "Ibc"])]
    nuevo = dict(zip(meta.oid[ix], np.random.default_rng(1).permutation(meta.sn_type[ix].to_numpy())))
    assert any(nuevo[o] != t for o, t in zip(meta.oid[ix], meta.sn_type[ix]))
    feats = pd.read_csv(fr / "features" / "features.csv")
    for df in (meta, feats):                                   # la llave lleva sn_type: se cambia en los dos
        df["sn_type"] = [nuevo.get(o, t) for o, t in zip(df.oid, df.sn_type)]
    meta.to_csv(real / "meta_real_ztf.csv", index=False)
    feats.to_csv(fr / "features" / "features.csv", index=False)
    g2 = C.main(["gap", "--name", "g2"] + common)
    assert g1["auc_sim_vs_real"] == g2["auc_sim_vs_real"] and g1["n_real"] == g2["n_real"]
    pd.testing.assert_frame_equal(pd.read_csv(out / "g1" / "gap_importancias.csv"),
                                  pd.read_csv(out / "g2" / "gap_importancias.csv"))


def _sel_world(n_per=40, seed=0):
    """R con 3 clases y subset alternado; predicciones fabricadas con una tasa de acierto dada por subconjunto."""
    y = np.repeat([0, 1, 2], n_per)
    sub = np.where(np.arange(len(y)) % 2 == 0, "val_sel", "val_rep")
    R = pd.DataFrame({"oid": [f"o{i}" for i in range(len(y))], "y": y, "subset": sub})
    rng = np.random.default_rng(seed)

    def pred(acc_sel, acc_rep):
        acc = np.where(sub == "val_sel", acc_sel, acc_rep)
        hit = rng.random(len(y)) < acc
        return np.where(hit, y, (y + 1 + rng.integers(0, 2, len(y))) % 3)
    return R, pred


def test_seleccion_anidada_prefiltro_bootstrap_y_solo_val_sel():
    cls = C.classes()
    R, pred = _sel_world()
    cfgs = [dict(C.BASELINE), {**C.BASELINE, "model": "rf"}, {**C.BASELINE, "model": "mlp"},
            {**C.BASELINE, "model": "hier_hgb_II"}]
    base_i = 0
    yp = [pred(0.6, 0.6), pred(0.95, 0.5), pred(1.0, 1.0), pred(0.62, 0.62)]
    cv = [0.6, 0.8, 0.1, 0.7]                                  # mediana 0.65: la 2 (cv 0.1) queda fuera
    el = C.select_nested(cfgs, cv, yp, R, cls, base_i)
    assert np.isclose(el["prefiltro_cv"]["mediana"], 0.65) and el["_pasa"] == {0: False, 1: True, 2: False, 3: True}
    assert el["candidata"] == cfgs[1] and el["comparacion"]["gana"] and el["elegida"] == cfgs[1]
    assert el["comparacion"]["n"] == (R.subset == "val_sel").sum()
    # val_rep no entra: la 3 perfecta y la 1 siempre mal en val_rep no cambian ni el orden ni la comparacion
    y, rep = R.y.to_numpy(), (R.subset == "val_rep").to_numpy()
    yp2 = [np.where(rep, y, yp[0]), np.where(rep, (y + 1) % 3, yp[1]), yp[2], np.where(rep, y, yp[3])]
    el2 = C.select_nested(cfgs, cv, yp2, R, cls, base_i)
    assert el2["comparacion"] == el["comparacion"] and el2["elegida"] == el["elegida"]
    assert el2["ranking_sel"] == el["ranking_sel"]
    # una candidata que acierta un solo objeto mas que la base no gana: queda la base
    uno_mas = yp[0].copy()
    i0 = np.flatnonzero(~rep & (yp[0] != y))[0]
    uno_mas[i0] = y[i0]
    el3 = C.select_nested(cfgs, cv, [yp[0], None, yp[2], uno_mas], R, cls, base_i)
    assert el3["candidata"] == cfgs[3] and el3["comparacion"]["delta"] > 0
    assert not el3["comparacion"]["gana"] and el3["elegida"] == C.BASELINE
    # si la primera en val_sel es la base, no hay comparacion
    el4 = C.select_nested(cfgs, [0.9, 0.95, 0.1, 0.1], [pred(1.0, 0.3)] + yp[1:], R, cls, base_i)
    assert el4["candidata"] == C.BASELINE and el4["comparacion"] is None and el4["elegida"] == C.BASELINE
    with pytest.raises(SystemExit):                            # sin base no hay incumbente
        C.select_nested(cfgs, cv, [None] + yp[1:], R, cls, base_i)


def test_cobertura_por_subconjunto_y_oids_de_la_red(tmp_path):
    fd, rd, fr, real = _fake_world(tmp_path)
    feats = pd.read_csv(fr / "features" / "features.csv")
    sin = {"ZTFrIa000", "ZTFrIa001", "ZTFrIbc002"}
    feats[~feats.oid.isin(sin)].to_csv(fr / "features" / "features.csv", index=False)
    S, R, v = C.prepare(fd, rd, fr, real)
    nn = pd.DataFrame({"oid": ["ZTFrIa000", "ZTFrIa003", "ZTFrII004", "ZTFrIIn000"],
                       "y_true": ["Ia", "Ia", "II", "IIn"]})
    nn.to_csv(tmp_path / "pred_nn.csv", index=False)
    oids = C.read_nn_oids(tmp_path / "pred_nn.csv", R)
    cov = C.coverage(R, v, C.classes(), oids)
    assert cov["val"]["n_real"] == 60 and cov["val"]["n_con_features"] == 57
    assert cov["val"]["n_nn"] == 3 and cov["val"]["n_comun_nn"] == 2           # la IIn no es de las 3 clases
    for s in ("val_sel", "val_rep"):
        vs = v[v.subset == s]
        assert cov[s]["n_real"] == len(vs) and cov[s]["n_con_features"] == len(vs) - len(sin & set(vs.oid))
    nn.assign(y_true=["Ia", "II", "II", "IIn"]).to_csv(tmp_path / "pred_mal.csv", index=False)   # Ia003 como II
    with pytest.raises(ValueError):                            # etiquetas distintas: no es la misma muestra
        C.read_nn_oids(tmp_path / "pred_mal.csv", R)


def test_particion_compartida_con_nnclf(tmp_path):
    """val_sel y val_rep son exactamente los de pipeline78.splits (los mismos que usa nnclf)."""
    from pipeline78 import splits
    fd, rd, fr, real = _fake_world(tmp_path)
    _, R, v = C.prepare(fd, rd, fr, real)
    sel, rep = splits.val_split(splits.read_val_meta(real / "meta_real_ztf.csv"))
    assert set(v.oid[v.subset == "val_sel"]) == set(sel) & set(v.oid)
    assert set(v.oid[v.subset == "val_rep"]) == set(rep) & set(v.oid)
    assert set(R.subset) == {"val_sel", "val_rep"}


def test_resumen_viejo_se_alinea_como_val(tmp_path):
    """Un resumen.csv anterior a la particion se reescribe con sus metricas como _val (eran de val completo)."""
    pd.DataFrame([{"fecha": "x", "name": "viejo", "prior": "none", "acc": 0.7, "bal_acc": 0.65, "n_real": 309,
                   "cobertura": 0.67}]).to_csv(tmp_path / "resumen.csv", index=False)
    C.append_resumen([{"name": "nuevo", "bal_acc_rep": 0.6}], tmp_path)
    r = pd.read_csv(tmp_path / "resumen.csv")
    assert list(r.columns) == C.RESUMEN_COLS
    v = r[r.name == "viejo"].iloc[0]
    assert v.bal_acc_val == 0.65 and v.acc_val == 0.7 and v.n_val == 309 and v.coverage_val == 0.67
    assert np.isnan(v.bal_acc_rep) and r[r.name == "nuevo"].iloc[0].bal_acc_rep == 0.6


# ----------------------------------------------------------------------------- mitigaciones del gap (rg_cens, g_modo)
CFG_MIT = {**CFG_HGB, "fset": "rg_cens", "g_modo": "separado"}
NUEVAS = ["t_rise_piso_r", "t_rise_cens_r", "t_rise_piso_g", "t_rise_cens_g", "d_t_rise_cens_gr"]


def test_censura_t_rise_en_marco_observado_igual_en_sims_y_reales(tmp_path):
    """Se censura por el t_rise OBSERVADO: 1.0 d a z 0.5 (0.67 d en reposo) cae en el piso, 1.2 d a z 0.5 (0.8 d en
    reposo, bajo 1.05) no. Sims y reales pasan por la misma derive: mismas filas crudas, mismas columnas."""
    filas = [("a", 1.0, 1.2), ("b", 1.2, 5.0), ("c", 3.0, None)]          # oid, t_rise r, t_rise g (None: sin g)
    rows = [_feat_row(o, 0, "Ia", b, t_rise=t) for o, tr, tg in filas for b, t in (("r", tr), ("g", tg))
            if t is not None]
    fd, rd = _write_sims(tmp_path, rows, [_sims_meta(o, 0, "Ia", z=0.5, template=f"T{o}") for o, _, _ in filas])
    real = tmp_path / "real_ztf"
    real.mkdir()
    pd.DataFrame([{"oid": o, "sn_type": "Ia", "subtipo": "Ia", "z": 0.5, "split": "val", "origen": "holdout",
                   "part_index": 0, "excluir": False} for o, _, _ in filas]).to_csv(real / "meta_real_ztf.csv",
                                                                                     index=False)
    fr = tmp_path / "features_real"
    (fr / "features").mkdir(parents=True)
    pd.DataFrame(rows, columns=FEAT_COLS).to_csv(fr / "features" / "features.csv", index=False)
    S = C.derive(C.load_sims(fd, rd)).set_index("oid").sort_index()
    R = C.derive(C.load_real_val(fr, real)[0]).set_index("oid").sort_index()
    pd.testing.assert_frame_equal(S[NUEVAS + ["t_rise_r", "t_rise_g"]], R[NUEVAS + ["t_rise_r", "t_rise_g"]])
    for F in (S, R):
        a, b, c = F.loc["a"], F.loc["b"], F.loc["c"]
        assert a.t_rise_piso_r == 1.0 and np.isnan(a.t_rise_cens_r) and np.isnan(a.d_t_rise_cens_gr)
        assert b.t_rise_piso_r == 0.0 and np.isclose(b.t_rise_cens_r, 0.8) and np.isclose(b.t_rise_r, 0.8)
        assert a.t_rise_piso_g == 0.0 and np.isclose(a.t_rise_cens_g, 0.8)
        assert np.isclose(b.d_t_rise_cens_gr, (5.0 - 1.2) / 1.5)
        assert c.t_rise_piso_r == 0.0 and np.isnan(c.t_rise_piso_g) and np.isnan(c.t_rise_cens_g)   # g sin ajuste
    Fo = C.derive(C.load_sims(fd, rd), rest_frame=False).set_index("oid").sort_index()
    assert Fo.t_rise_piso_r.tolist() == S.t_rise_piso_r.tolist() and np.isclose(Fo.loc["b", "t_rise_cens_r"], 1.2)


def test_rg_sin_cambios_y_columnas_de_rg_cens():
    """rg no cambia (lista y valores sin censurar) y las columnas nuevas de derive van al final, despues de las de
    siempre. rg_cens cambia solo t_rise y d_t_rise_gr por sus censuradas y agrega las banderas."""
    rg = ["f_r", "t_rise_r", "t_fall_r", "gamma_r", "f_g", "t_rise_g", "t_fall_g", "gamma_g", "color_gr",
          "d_t_rise_gr", "d_t_fall_gr", "d_gamma_gr", "d_f_gr", "M_pk_r", "M_pk_g"]
    assert C.fset_cols("rg", True) == (rg, rg)
    cens = [{"t_rise_r": "t_rise_cens_r", "t_rise_g": "t_rise_cens_g", "d_t_rise_gr": "d_t_rise_cens_gr"}.get(c, c)
            for c in rg] + ["t_rise_piso_r", "t_rise_piso_g"]
    assert C.fset_cols("rg_cens", True) == (cens, cens)
    assert C.fset_cols("rg_cens", False)[1] == [c for c in cens if not c.startswith("M_pk")]
    W = C.widen(pd.DataFrame([_feat_row("a", 0, "Ia", "r", t_rise=1.0), _feat_row("a", 0, "Ia", "g", t_rise=1.2)]))
    W["z"] = 0.5
    F = C.derive(W).iloc[0]
    assert np.isclose(F.t_rise_r, 1.0 / 1.5) and np.isclose(F.d_t_rise_gr, 0.2 / 1.5)     # rg: sin censura
    assert np.isnan(F.t_rise_cens_r) and np.isnan(F.d_t_rise_cens_gr)
    cols = list(C.derive(W).columns)
    assert cols[-len(NUEVAS):] == NUEVAS and cols[-len(NUEVAS) - 1] == "tiene_r"
    assert C.BASELINE == {"model": "hgb", "fset": "rg", "use_z": True, "peso": "wz", "balance": True,
                          "g_modo": "nan"}
    assert C.same_cfg({k: v for k, v in C.BASELINE.items() if k != "g_modo"}, C.BASELINE)   # sin clave = nan
    assert not C.same_cfg({**C.BASELINE, "g_modo": "separado"}, C.BASELINE)
    assert C.cols_solo_r(cens + ["M_pk_r"]) == ["f_r", "t_rise_cens_r", "t_fall_r", "gamma_r", "M_pk_r",
                                                "t_rise_piso_r", "M_pk_r"]


def test_g_separado_enruta_por_g_y_B_no_ve_g(tmp_path, monkeypatch):
    """A se ajusta solo con las sims con g, B con todas y sin columnas de g ni de color; cada objeto se predice con
    A si tiene g (m_pk_g finito) y con B si no. B no cambia si se alteran todas las columnas de g."""
    fd, rd, fr, real = _fake_world(tmp_path)
    S, R, _ = C.prepare(fd, rd, fr, real)
    c1, c2 = C.fset_cols("rg_cens", True)
    gS, y, w = S.m_pk_g.notna().to_numpy(), S.y.to_numpy(), S.w_z.to_numpy()
    assert 0 < gS.sum() < len(S)
    gcols = [c for c in S.columns if c.endswith("_g") or "_gr" in c]
    vistos, fit0 = [], C.Model.fit

    def fit(self, F, yy, ww):
        vistos.append((self, len(F), bool(F.m_pk_g.notna().all())))
        return fit0(self, F, yy, ww)

    monkeypatch.setattr(C.Model, "fit", fit)
    m = C.ModeloG("hgb", c1, c2, 3, seed=1).fit(S, y, w)
    assert [v[0] for v in vistos] == [m.A, m.B]
    assert vistos[0][1] == gS.sum() and vistos[0][2] and vistos[1][1] == len(S)
    assert m.A.cols2 == c2 and m.B.cols1 == m.B.cols2 == C.cols_solo_r(c2)
    assert not [c for c in m.B.cols2 if c in gcols] and {"M_pk_r", "t_rise_piso_r"} <= set(m.B.cols2)
    assert m.B.ests_[0].n_features_in_ == len(m.B.cols2)
    # B no ve g: alterar todas las columnas de g (los NaN siguen NaN) no cambia a B y si a A
    m2 = C.ModeloG("hgb", c1, c2, 3, seed=1).fit(S.assign(**{c: S[c] * 1.7 + 3.0 for c in gcols}), y, w)
    assert np.array_equal(m.B.predict_proba(R), m2.B.predict_proba(R))
    assert not np.array_equal(m.A.predict_proba(R), m2.A.predict_proba(R))
    # enrutamiento: A da [1, 0, 0] y B da [0, 1, 0]
    sin = np.arange(len(R)) % 3 == 0
    Rm = R.copy()
    Rm.loc[sin, gcols] = np.nan

    def pred(self, F):
        P = np.zeros((len(F), self.K))
        P[:, 0 if self is m.A else 1] = 1.0
        return P

    monkeypatch.setattr(C.Model, "predict_proba", pred)
    P = m.predict_proba(Rm)
    assert (P[~sin, 0] == 1).all() and (P[sin, 1] == 1).all()
    Ps = m.predict_proba(S)
    assert (Ps[gS, 0] == 1).all() and (Ps[~gS, 1] == 1).all()
    with pytest.raises(ValueError):
        C.make_model({**CFG_MIT, "g_modo": "otro"}, c1, c2, 3)


def _mitig_world(root):
    """_fake_world con sims en el piso de t_rise (1.0 d observado) y reales sin g o en el piso."""
    fd, rd, fr, real = _fake_world(root)
    rng = np.random.default_rng(7)
    fs = pd.read_csv(fd / "features" / "features.csv")
    fs.loc[rng.random(len(fs)) < 0.3, "t_rise"] = 1.0
    fs.to_csv(fd / "features" / "features.csv", index=False)
    ff = pd.read_csv(fr / "features" / "features.csv")
    ff = ff[~(ff.oid.isin(ff.oid.unique()[::4]) & (ff.filter_band == "g"))].copy()
    ff.loc[rng.random(len(ff)) < 0.15, "t_rise"] = 1.0
    ff.to_csv(fr / "features" / "features.csv", index=False)
    return fd, rd, fr, real


def test_D1_mitig_etiquetas_val_no_entrenan_ni_calibran(tmp_path):
    """Como D1, con rg_cens y g_modo separado: permutar las etiquetas de las reales val no cambia P, Q ni T."""
    S, R, _ = C.prepare(*_mitig_world(tmp_path))
    assert (S.t_rise_piso_r == 1).any() and (R.t_rise_piso_r == 1).any() and R.m_pk_g.isna().any()
    perm = np.random.default_rng(0).permutation(len(R))
    Rp = R.assign(y=R.y.to_numpy()[perm], cls=R.cls.to_numpy()[perm], sn_type=R.sn_type.to_numpy()[perm])
    for cfg in (CFG_MIT, {**CFG_MIT, "model": "hier_hgb_II", "peso": "wz_dr"}, {**CFG_MIT, "fset": "rg"},
                {**CFG_MIT, "g_modo": "nan"}):
        a = C.run_config(S, R, cfg, C.classes(), folds=3, keep_model=True)
        b = C.run_config(S, Rp, cfg, C.classes(), folds=3, keep_model=True)
        assert isinstance(a["_model"], C.ModeloG if cfg["g_modo"] == "separado" else C.Model)
        assert np.allclose(a["_P"], b["_P"], rtol=0, atol=1e-12) and np.allclose(a["_Q"], b["_Q"], rtol=0, atol=1e-12)
        assert a["temperatura"] == b["temperatura"] and a["prior_em"] == b["prior_em"]
        assert a["real_none"]["val"]["bal_acc"] != b["real_none"]["val"]["bal_acc"]   # el test no es trivial


def test_cli_g_modo_y_grilla_mitig(tmp_path):
    fd, rd, fr, real = _mitig_world(tmp_path)
    out = tmp_path / "out"
    common = ["--features-sims", str(fd), "--run-dir", str(rd), "--features-real", str(fr), "--real-dir", str(real),
              "--out-root", str(out), "--folds", "3"]
    res = C.main(["train", "--name", "m1", "--fset", "rg_cens", "--g-modo", "separado"] + common)
    assert res["config"]["g_modo"] == "separado" and isinstance(res["_model"], C.ModeloG)
    ev = C.main(["eval", "--name", "m1", "--features-real", str(fr), "--real-dir", str(real), "--out-root", str(out)])
    assert np.allclose(ev["_P"], res["_P"], rtol=0, atol=1e-12)
    assert C.GRIDS["mitig"] == dict(fset=("rg", "rg_cens"), model=("hgb", "hgb_lento"), use_z=(True,), peso=("wz",),
                                    g_modo=("nan", "separado"))
    assert all("g_modo" not in g for k, g in C.GRIDS.items() if k != "mitig")   # las otras grillas: solo nan
    tab = C.main(["sweep", "--name", "mt", "--grid", "mitig", "--models", "hgb"] + common)
    assert len(tab) == 2 * 2 * 2 and set(tab.g_modo) == {"nan", "separado"} and set(tab.fset) == {"rg", "rg_cens"}
    base = tab[tab.es_base]
    assert len(base) == 2 and set(base.g_modo) == {"nan"} and set(base.fset) == {"rg"} and set(base.model) == {"hgb"}
    assert json.loads((out / "mt" / "mejor.json").read_text())["base"] == C.BASELINE
    resumen = pd.read_csv(out / "resumen.csv", keep_default_na=False)
    assert list(resumen.columns) == C.RESUMEN_COLS
    assert set(resumen[resumen.name == "mt"].g_modo) == {"nan", "separado"}
    assert set(resumen[resumen.name == "m1"].g_modo) == {"separado"}
