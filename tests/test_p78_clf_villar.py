"""Tests del clasificador oficial sobre features de Villar (pipeline78/clf_villar.py).

python -m pytest tests/test_p78_clf_villar.py -q
"""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline78 import clf_villar as C

FEAT_COLS = ["sn_name", "filter_band"] + C.RAW + ["sn_type", "oid", "part_index"]


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
    assert res["real_none"]["acc"] > 0.8 and res["real_none"]["n"] == 60      # 15 val x (Ia, II, IIb, Ibc); IIn fuera
    lo, hi = res["real_none"]["acc_ic95"]
    assert lo <= res["real_none"]["acc"] <= hi
    for f in ("metrics.json", "model.joblib", "pred_real_val.csv", "confusion_real_val_none.csv",
              "importancias_real_val.csv"):
        assert (out / "t1" / f).exists(), f
    pred = pd.read_csv(out / "t1" / "pred_real_val.csv")
    assert pred.oid.str.startswith("ZTFr").all() and len(pred) == 60
    ev = C.main(["eval", "--name", "t1", "--features-real", str(fr), "--real-dir", str(real), "--out-root", str(out)])
    assert np.isclose(ev["real_none"]["acc"], res["real_none"]["acc"])
    tab = C.main(["sweep", "--name", "sw", "--grid", "rapido", "--models", "hgb,rf"] + common)
    assert len(tab) == 2 * 2 * 2 * 2 and (out / "sw" / "mejor" / "metrics.json").exists()
    best = json.loads((out / "sw" / "mejor.json").read_text())
    assert best["config"]["model"] in ("hgb", "rf")
    g = C.main(["gap", "--name", "gp", "--peso", "wz"] + common)
    assert 0.0 <= g["auc_sim_vs_real"] <= 1.0 and (out / "gp" / "gap_importancias.csv").exists()
    resumen = pd.read_csv(out / "resumen.csv")
    assert set(resumen.name) == {"t1", "sw"} and set(resumen.prior) == {"none", "em"}
