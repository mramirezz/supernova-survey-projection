"""Tests de las features fisicas del clasificador Villar (fset rg_fisica, seccion fisica de pipeline78/clf_villar.py):
hombro de r en los residuos contra el SPM y evolucion del color g - r de los SPM.

python -m pytest tests/test_p78_clf_villar_fisica.py -q
"""
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pipeline78 import clf_villar as C

FEAT = ["sn_name", "filter_band"] + C.SPM + [f"{p}_err" for p in C.PARS] + C.QUAL + ["sn_type", "oid", "part_index"]
P_R = [1e-7, 0.4, 8.0, 3.0, 30.0, 12.0]          # A, f, t0, t_rise, t_fall, gamma en la fase de r
DG = 1.5                                         # g empieza 1.5 d despues que r: otro origen de fase


@pytest.fixture(autouse=True)
def _un_hilo(monkeypatch):
    monkeypatch.setattr(C, "N_JOBS", 1)          # sin workers de joblib en los tests chicos
    with threadpool_limits(2):
        yield


def _p_g(p_r=P_R, dcolor=0.3, t_fall=None, dg=DG):
    """El mismo SPM en g, con g - r = dcolor, en la fase de g (t0 corrido dg)."""
    A, f, t0, tr, tf, gam = p_r
    return [A * 10 ** (-0.4 * dcolor), f, t0 - dg, tr, tf if t_fall is None else t_fall, gam]


def _curva(p_r=P_R, p_g=None, z=0.05, hombro=0.0, ruido=0.0, rng=None, fin=120.0, dg=DG, mjd0=59000.0):
    """Filas LC_COLS (sin llave) de una curva generada con el SPM de cada banda. hombro: bump gaussiano en el FLUJO de r
    centrado a +27 d en reposo desde el pico del SPM (sigma 5 d), que el SPM no tiene. 3 UL antes de cada banda."""
    M = C._zlf("model")
    partes = []
    for b, p, d0 in (("r", p_r, 0.0), ("g", p_g, dg)):
        if p is None:
            continue
        ph = np.arange(0.0, fin, 2.0)
        flux = M.alerce_model(ph, *p)
        if b == "r" and hombro:
            rest = (ph - C.pico_spm(p)) / (1 + z)
            flux = flux * (1 + hombro * np.exp(-0.5 * ((rest - 27.0) / 5.0) ** 2))
        mag = M.flux_to_mag(flux) + (rng.normal(0, ruido, len(ph)) if ruido else 0.0)
        partes.append(pd.DataFrame({"mjd": np.r_[mjd0 + d0 + ph, mjd0 + d0 - np.array([3.0, 6.0, 9.0])], "filter": b,
                                    "magnitud_proyectada": np.r_[mag, np.full(3, mag[0] + 1.5)],
                                    "magerr": np.r_[np.full(len(ph), 0.03), np.full(3, np.nan)],
                                    "upperlimit": ["F"] * len(ph) + ["T"] * 3}))
    return pd.concat(partes, ignore_index=True)


def _fis(lc, p_r=P_R, p_g=None, z=0.05, rest_frame=True):
    par = {"r": np.array(p_r, float) if p_r is not None else np.full(6, np.nan),
           "g": np.array(p_g, float) if p_g is not None else np.full(6, np.nan)}
    with np.errstate(all="ignore"):
        return C.fisica_curva(C.bandas_lc(lc), par, z, rest_frame)


# ----------------------------------------------------------------------------- una curva
def test_hombro_positivo_y_curva_suave_cero():
    """Ia con hombro en r a +27 d: residuo > 0 en [15, 40) y ~0 en [40, 70). Curva suave: residuos ~0 (el SPM es la
    curva). Mismo SPM en g con g - r = 0.3 y otro origen de fase: color 0.3 en todas las fases y pendiente 0."""
    suave = _fis(_curva(p_g=_p_g()), p_g=_p_g())
    hombro = _fis(_curva(p_g=_p_g(), hombro=0.35), p_g=_p_g())
    assert abs(suave["res_r_15_40"]) < 1e-6 and abs(suave["res_r_40_70"]) < 1e-6
    assert hombro["res_r_15_40"] > 0.1 and abs(hombro["res_r_40_70"]) < 0.03
    for f in (suave, hombro):
        assert np.allclose([f["color_gr_0"], f["color_gr_15"], f["color_gr_30"]], 0.3, atol=1e-6)
        assert abs(f["d_color_gr_30_0"]) < 1e-6
    # hombro: m_SPM - m_obs con m_obs = m_SPM - 2.5 log10(1 + bump): la mediana de los puntos de la ventana
    M = C._zlf("model")
    pk = C.pico_spm(P_R)
    ph = np.arange(0.0, 120.0, 2.0)
    rest = (ph - pk) / 1.05
    k = (rest >= 15) & (rest < 40)
    esperado = np.median(2.5 * np.log10(1 + 0.35 * np.exp(-0.5 * ((rest[k] - 27.0) / 5.0) ** 2)))
    assert np.isclose(hombro["res_r_15_40"], esperado, atol=1e-9) and k.sum() >= 2
    assert M.alerce_model(np.array([pk]), *P_R)[0] > M.alerce_model(np.array([pk - 1, pk + 1]), *P_R).max()


def test_color_se_enrojece_y_faltantes():
    """g que cae mas rapido que r: g - r crece (pendiente > 0). Sin g: colores NaN y residuos de r. Sin r: todo NaN.
    Menos de 2 puntos en la ventana: NaN."""
    pg = _p_g(t_fall=18.0)
    f = _fis(_curva(p_g=pg), p_g=pg)
    assert f["color_gr_30"] > f["color_gr_15"] > f["color_gr_0"] and f["d_color_gr_30_0"] > 0.2
    solo_r = _fis(_curva())
    assert np.isfinite(solo_r["res_r_15_40"]) and all(np.isnan(solo_r[c]) for c in C.FIS if "color" in c)
    sin_r = _fis(_curva(p_r=None, p_g=pg), p_r=None, p_g=pg)
    assert all(np.isnan(v) for v in sin_r.values())
    corta = _fis(_curva(fin=40.0, p_g=_p_g()), p_g=_p_g())              # r hasta la fase 38 observada
    assert np.isfinite(corta["res_r_15_40"]) and np.isnan(corta["res_r_40_70"])
    assert np.isfinite(corta["color_gr_30"])                            # el color sale de los modelos


def test_reposo_con_z():
    """Las ventanas van en reposo: con z = 0.5 el hombro a +27 d en reposo cae en [15, 40) en reposo, y en el marco
    observado (+40.5 d) se corre a la ventana siguiente."""
    lc = _curva(z=0.5, hombro=0.35, fin=160.0)
    rep, obs = _fis(lc, z=0.5), _fis(lc, z=0.5, rest_frame=False)
    assert rep["res_r_15_40"] > 0.1 > obs["res_r_15_40"] and obs["res_r_40_70"] > rep["res_r_40_70"] + 0.03
    assert _fis(lc, z=np.nan)["res_r_15_40"] == obs["res_r_15_40"]      # sin z: marco observado


def test_bandas_lc_igual_que_parse_parquet_lightcurve(tmp_path):
    """La conversion de filas a bandas es la del extractor (parquet_reader.parse_parquet_lightcurve)."""
    PR = C._zlf("parquet_reader")
    rows = []
    for k, st in ((0, "Ia"), (0, "II"), (1, "Ia")):
        lc = _curva(p_g=_p_g(), mjd0=59000.0 + 10 * k).assign(oid="ZTFa", part_index=np.int32(k), sn_type=st)
        rows.append(lc.sample(frac=1.0, random_state=k))                 # desordenadas, como pueden venir
    df = pd.concat(rows, ignore_index=True)
    df.loc[df.index[:2], "magnitud_proyectada"] = np.nan
    df.to_parquet(tmp_path / "ZTFa__00000.parquet", index=False)
    for k, st in ((0, "Ia"), (0, "II"), (1, "Ia")):
        fd, _, _ = PR.parse_parquet_lightcurve(tmp_path / "ZTFa__00000.parquet", k, st, oid="ZTFa")
        mine = C.bandas_lc(df[(df.part_index == k) & (df.sn_type == st)])
        assert fd.keys() == mine.keys()
        for b in fd:
            pd.testing.assert_frame_equal(fd[b], mine[b])


def test_pico_spm_contiene_el_maximo():
    M = C._zlf("model")
    t = np.arange(-600.0, 900.0, 0.05)
    for p in ([1e-8, 0.0, 0.0, 30.0, 200.0, 1.0], [1e-8, 0.9, 0.0, 1.0, 200.0, 1.0], [1e-8, 0.5, -100.0, 100.0, 200.0, 150.0],
              [1e-8, 0.05, 10.0, 50.0, 200.0, 1.0], P_R):
        assert abs(C.pico_spm(p) - t[np.argmax(M.alerce_model(t, *p))]) < 0.06


# ----------------------------------------------------------------------------- mundo sintetico (sims + reales)
PROTO = {"Ia": dict(f=0.4, t_fall=30.0, gamma=12.0, hombro=0.35, tf_g=22.0),     # Ia e Ibc: mismo SPM en r y g
         "Ibc": dict(f=0.4, t_fall=30.0, gamma=12.0, hombro=0.0, tf_g=22.0),
         "II": dict(f=0.1, t_fall=60.0, gamma=70.0, hombro=0.0, tf_g=40.0),
         "IIb": dict(f=0.3, t_fall=40.0, gamma=20.0, hombro=0.0, tf_g=30.0)}


def _feat_rows(oid, part, st, par):
    out = []
    for b, p in par.items():
        if p is None:
            continue
        r = {"sn_name": f"{oid}_{st}_p{part:02d}", "filter_band": b, **dict(zip(C.SPM, p)), "n_points": 60,
             "time_span": 118.0, "rms": 1e-9, "sn_type": st, "oid": oid, "part_index": part}
        r.update({f"{q}_err": abs(r[q]) * 0.1 for q in C.PARS})
        out.append(r)
    return out


def _objeto(rng, st):
    """(parametros por banda, hombro, z). Ia e Ibc tienen el mismo SPM en r y en g: solo el hombro las separa."""
    q = PROTO[st]
    z = rng.uniform(0.02, 0.1)
    tf = q["t_fall"] * rng.lognormal(0, 0.05)
    pr = [1e-7 * rng.lognormal(0, 0.2), q["f"] + rng.normal(0, 0.02), 8.0, 3.0, tf, q["gamma"]]
    pg = None if rng.random() < 0.2 else _p_g(pr, 0.2, q["tf_g"] * tf / q["t_fall"])
    return {"r": pr, "g": pg}, q["hombro"] * rng.uniform(0.8, 1.2), z


def _escribir(root, sims, reales):
    """Escribe un mundo. sims: dicts (field, part, st, template, z, w, par, lc); reales: dicts (oid, st, split, z,
    par, lc). Sims: parquet por campo, _sims_all y features.csv (con t0). Reales: parquet por clase (todas las
    mitades), meta_real_ztf.csv y features.csv. Las 'features' son los parametros con que se generan las curvas."""
    rd, fd, real, fr = root / "sim", root / "features_sim", root / "real_ztf", root / "features_real"
    for d in (rd, fd / "features", real, fr / "features"):
        d.mkdir(parents=True)
    filas, feats, meta = {}, [], []
    for s in sims:
        lc = s["lc"].assign(oid=s["field"], part_index=np.int32(s["part"]), sn_type=s["st"])
        filas.setdefault(s["field"], []).append(lc[C.LC_COLS])
        feats += _feat_rows(s["field"], s["part"], s["st"], s["par"])
        meta.append({"field": s["field"], "part_index": s["part"], "sn_type": s["st"], "template": s["template"],
                     "subtype": s["st"], "z": s["z"], "w_z": s["w"]})
    for fld, ls in filas.items():
        pd.concat(ls, ignore_index=True).to_parquet(rd / f"{fld}__00000.parquet", index=False)
    pd.DataFrame(meta).to_parquet(rd / "_sims_all.parquet")
    pd.DataFrame(feats, columns=FEAT).to_csv(fd / "features" / "features.csv", index=False)
    filas, feats, meta = {}, [], []
    for r in reales:
        filas.setdefault(r["st"], []).append(r["lc"].assign(oid=r["oid"], part_index=np.int32(0),
                                                            sn_type=r["st"])[C.LC_COLS])
        feats += _feat_rows(r["oid"], 0, r["st"], r["par"])
        meta.append({"oid": r["oid"], "sn_type": r["st"], "subtipo": r["st"], "z": r["z"], "split": r["split"],
                     "origen": "holdout", "part_index": 0, "excluir": False})
    for st, ls in filas.items():
        pd.concat(ls, ignore_index=True).to_parquet(real / f"{st}.parquet", index=False)
    pd.DataFrame(meta).to_csv(real / "meta_real_ztf.csv", index=False)
    pd.DataFrame(feats, columns=FEAT).to_csv(fr / "features" / "features.csv", index=False)
    return fd, rd, fr, real


def _gen(rng, st):
    par, h, z = _objeto(rng, st)
    return {"st": st, "par": par, "z": z, "lc": _curva(par["r"], par["g"], z, h, 0.02, rng)}


def _mundo(root, n_sim=24, n_real=16, seed=3):
    """Sims (dos por campo, 6 plantillas por tipo) y reales (3/4 val, 1/4 final) de los cuatro tipos."""
    rng = np.random.default_rng(seed)
    sims = [{**_gen(rng, st), "field": f"ZTF{st}{i // 2:03d}", "part": i % 2, "template": f"{st}_T{i % 6}",
             "w": rng.uniform(0.5, 1.5)} for st in PROTO for i in range(n_sim)]
    reales = [{**_gen(rng, st), "oid": f"ZTFr{st}{i:03d}", "split": "val" if i < n_real * 3 // 4 else "final"}
              for st in PROTO for i in range(n_real)]
    return _escribir(root, sims, reales)


def test_mismo_camino_sims_y_reales(tmp_path, monkeypatch):
    """La misma curva y los mismos parametros como sim y como real val dan las mismas FIS, por fisica_filas en los
    dos. Las reales de la mitad final estan en los parquets y en features.csv y nunca se leen."""
    rng = np.random.default_rng(5)
    objs = [{**_gen(rng, st), "oid": f"ZTFc{st}{i}"} for st in PROTO for i in range(2)]
    sims = [{**o, "field": o["oid"], "part": 0, "template": "T", "w": 1.0} for o in objs]
    final = [{**_gen(rng, st), "oid": f"ZTFfin{st}", "split": "final"} for st in PROTO]
    fd, rd, fr, real = _escribir(tmp_path, sims, [{**o, "split": "val"} for o in objs] + final)
    llamadas, orig = [], C.fisica_filas

    def espia(rows, T, rest_frame=True):
        llamadas.append(sorted(T.oid))
        return orig(rows, T, rest_frame)

    leidas, rt = [], C.pq.read_table

    def espia_pq(*a, **k):
        t = rt(*a, **k)
        leidas.extend(t.column("oid").to_pylist() if "oid" in t.column_names else [])
        return t

    monkeypatch.setattr(C, "fisica_filas", espia)
    monkeypatch.setattr(C.pq, "read_table", espia_pq)
    S, R, _ = C.prepare(fd, rd, fr, real, fisica=True)
    fin = {o["oid"] for o in final}
    assert len(llamadas) == len(objs) + 1                          # una por campo de sims y una para las reales
    assert sorted(llamadas[-1]) == sorted(R.oid)
    assert not set(leidas) & fin and not set(R.oid) & fin          # la final no se lee de los parquets
    assert not set(pd.read_csv(fr / "features" / "fisica_val_rest.csv").oid) & fin
    a = S.set_index(C.KEYS)[C.FIS].sort_index()
    b = R.set_index(C.KEYS)[C.FIS].sort_index()
    assert a.index.equals(b.index) and len(a) == len(objs)
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)
    assert np.isfinite(a.res_r_15_40).all()
    ia = a.xs("Ia", level="sn_type").res_r_15_40
    otras = a.drop("Ia", level="sn_type").res_r_15_40
    assert ia.min() > 0.1 > otras.abs().max()                      # el hombro de las Ia se ve en los residuos


def test_cache_no_recalcula_y_detecta_cambios(tmp_path, monkeypatch):
    fd, rd, fr, real = _mundo(tmp_path, n_sim=4, n_real=4)
    S0, R0, _ = C.prepare(fd, rd, fr, real, fisica=True)
    n, orig = [0], C.fisica_curva

    def cuenta(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    monkeypatch.setattr(C, "fisica_curva", cuenta)
    S1, R1, _ = C.prepare(fd, rd, fr, real, fisica=True)
    assert n[0] == 0
    pd.testing.assert_frame_equal(S0, S1)
    pd.testing.assert_frame_equal(R0, R1)
    f = pd.read_csv(fd / "features" / "features.csv")
    f.loc[(f.oid == S0.oid[0]) & (f.part_index == S0.part_index[0]) & (f.sn_type == S0.sn_type[0])
          & (f.filter_band == "r"), "t0"] += 2.0                       # cambia el ajuste de una sim
    f.to_csv(fd / "features" / "features.csv", index=False)
    S2, _, _ = C.prepare(fd, rd, fr, real, fisica=True)
    assert n[0] == 1
    assert not np.isclose(S2.res_r_15_40[0], S0.res_r_15_40[0])
    pd.testing.assert_frame_equal(S2.iloc[1:][C.FIS], S0.iloc[1:][C.FIS])


def test_fisica_no_depende_de_la_etiqueta():
    """La llave solo encuentra la curva: cambiar sn_type en filas y parametros no cambia las FIS."""
    rng = np.random.default_rng(0)
    par, h, z = _objeto(rng, "Ia")
    lc = _curva(par["r"], par["g"], z, h).assign(oid="o", part_index=0, sn_type="Ia")
    T = pd.DataFrame([{"oid": "o", "part_index": 0, "sn_type": "Ia", "z": z,
                       **{f"{p}_{b}": v for b in C.BANDS for p, v in zip(C.SPM, par[b])}}])
    a = C.fisica_filas(lc, T)
    b = C.fisica_filas(lc.assign(sn_type="Ibc"), T.assign(sn_type="Ibc"))
    pd.testing.assert_frame_equal(a[C.FIS], b[C.FIS])


def test_rg_sin_cambios_con_fisica(tmp_path):
    """rg es la misma lista, sus columnas valen lo mismo con o sin las FIS, y el modelo rg da lo mismo. cols_solo_r
    (modelo B de g separado) deja los residuos de r y saca los colores."""
    rg = C.fset_cols("rg", True)[1]
    assert C.fset_cols("rg_fisica", True) == (rg + C.FIS, rg + C.FIS)
    assert C.fset_cols("rg_fisica", False)[1] == [c for c in rg if not c.startswith("M_pk")] + C.FIS
    assert C.cols_solo_r(C.FIS) == ["res_r_15_40", "res_r_40_70"]
    assert C.BASELINE["fset"] == "rg"
    fd, rd, fr, real = _mundo(tmp_path, n_sim=12, n_real=8)
    S0, R0, _ = C.prepare(fd, rd, fr, real)
    S1, R1, _ = C.prepare(fd, rd, fr, real, fisica=True)
    for a, b in ((S0, S1), (R0, R1)):
        assert [c for c in b.columns if c not in C.FIS] == list(a.columns) and set(C.FIS) <= set(b.columns)
    pd.testing.assert_frame_equal(S1[list(S0.columns)], S0)
    pd.testing.assert_frame_equal(R1[list(R0.columns)], R0)
    cfg = {"model": "hgb", "fset": "rg", "use_z": True, "peso": "wz", "balance": True}
    a = C.run_config(S0, R0, cfg, C.classes(), folds=3, keep_model=True)
    b = C.run_config(S1, R1, cfg, C.classes(), folds=3, keep_model=True)
    assert np.array_equal(a["_P"], b["_P"]) and a["temperatura"] == b["temperatura"]


def test_D1_fisica_etiquetas_val_no_entrenan_ni_calibran(tmp_path):
    """Como D1 con rg_fisica: permutar las etiquetas de las reales val no cambia P, Q ni T. Y en este mundo (Ia e Ibc
    con el mismo SPM) el hombro separa Ia de Ibc y rg no puede (CV en sims)."""
    S, R, _ = C.prepare(*_mundo(tmp_path), fisica=True)
    perm = np.random.default_rng(0).permutation(len(R))
    Rp = R.assign(y=R.y.to_numpy()[perm], cls=R.cls.to_numpy()[perm], sn_type=R.sn_type.to_numpy()[perm])
    base = {"model": "hgb", "fset": "rg_fisica", "use_z": True, "peso": "wz", "balance": True}
    res = {}
    for cfg in (base, {**base, "model": "hier_hgb_II"}):
        a = res[cfg["model"]] = C.run_config(S, R, cfg, C.classes(), folds=3, keep_model=True)
        b = C.run_config(S, Rp, cfg, C.classes(), folds=3, keep_model=True)
        assert np.allclose(a["_P"], b["_P"], rtol=0, atol=1e-12) and np.allclose(a["_Q"], b["_Q"], rtol=0, atol=1e-12)
        assert a["temperatura"] == b["temperatura"] and a["prior_em"] == b["prior_em"]
        assert a["real_none"]["val"]["bal_acc"] != b["real_none"]["val"]["bal_acc"]   # el test no es trivial
    rg = C.run_config(S, R, {**base, "fset": "rg"}, C.classes(), folds=3)["cv_sims"]
    rf = res["hgb"]["cv_sims"]
    assert rf["recall_Ia"] + rf["recall_Ibc"] > rg["recall_Ia"] + rg["recall_Ibc"] + 0.3


def test_cli_fisica_train_eval_sweep(tmp_path):
    fd, rd, fr, real = _mundo(tmp_path, n_sim=12, n_real=8)
    out = tmp_path / "out"
    common = ["--features-sims", str(fd), "--run-dir", str(rd), "--features-real", str(fr), "--real-dir", str(real),
              "--out-root", str(out), "--folds", "3"]
    res = C.main(["train", "--name", "f1", "--fset", "rg_fisica"] + common)
    assert set(C.FIS) <= set(res["cols"])
    ev = C.main(["eval", "--name", "f1", "--features-real", str(fr), "--real-dir", str(real), "--out-root", str(out)])
    assert np.allclose(ev["_P"], res["_P"], rtol=0, atol=1e-12)
    assert C.GRIDS["fisica"] == dict(fset=("rg", "rg_fisica"), model=("hgb", "hgb_lento", "hier_hgb_II", "ens_hier"),
                                     use_z=(True,), peso=("wz",))
    tab = C.main(["sweep", "--name", "fs", "--grid", "fisica", "--models", "hgb"] + common)
    assert len(tab) == 2 * 2 and set(tab.fset) == {"rg", "rg_fisica"}
    assert set(tab[tab.es_base].fset) == {"rg"}
    assert json.loads((out / "fs" / "mejor.json").read_text())["base"] == C.BASELINE
    assert (fd / "features" / "fisica_rest.csv").exists() and (fr / "features" / "fisica_val_rest.csv").exists()
