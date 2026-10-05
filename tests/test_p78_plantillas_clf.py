"""Clasificador por ajuste bayesiano de plantillas (pipeline78/plantillas_clf.py) sobre una biblioteca falsa de 7
plantillas espectrales (Ia x2, IIP, IIL, IIb, IIn, Ib) con formas y colores distintos."""
import json, math, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import truncnorm
from pipeline78 import bands as B, catalog, engine, project, runcfg, sampling
from pipeline78 import plantillas_clf as PL
from pipeline78.nnclf import data as D
from pipeline78.store import load_template, save_template

W = np.arange(3005.0, 9195.0, 2.0)
ZG = np.round(np.arange(0.01, 0.1001, 0.005), 4)
T_PK = 55000.0
BANDS = B.survey_bands("ZTF", ("g", "r"))
# (sn, clase, subtipo, t0, t1, curva m(t) desde el pico de reposo, temperatura T(t))
FAKES = [
    ("FIA1", "Ia", None, -15, 80, lambda t: np.where(t < 0, 0.012 * t**2, np.where(t < 30, 0.065 * t, 1.95 + 0.02 * (t - 30))),
     lambda t: np.clip(11000 - 80 * t, 5500, 13000)),
    ("FIA2", "Ia", None, -16, 85, lambda t: np.where(t < 0, 0.01 * t**2, np.where(t < 30, 0.045 * t, 1.35 + 0.02 * (t - 30))),
     lambda t: np.clip(11500 - 60 * t, 6000, 13000)),
    ("FIIP", "II", "IIP", -8, 140, lambda t: np.where(t < 0, 0.05 * t**2, np.where(t < 90, 0.005 * t,
                                               np.where(t < 110, 0.45 + 0.1 * (t - 90), 2.45 + 0.01 * (t - 110)))),
     lambda t: 5500 + 6500 * np.exp(-(t + 8) / 20)),
    ("FIIL", "II", "IIL", -10, 120, lambda t: np.where(t < 0, 0.03 * t**2, 0.035 * t),
     lambda t: 5500 + 5000 * np.exp(-(t + 10) / 25)),
    ("FIIB", "IIb", None, -20, 110, lambda t: np.where(t < 0, 0.006 * t**2, np.where(t < 25, 0.07 * t, 1.75 + 0.02 * (t - 25))),
     lambda t: np.clip(7000 - 30 * t, 4500, 9000)),
    ("FIIN", "IIn", None, -30, 200, lambda t: np.where(t < 0, 0.002 * t**2, 0.012 * t), lambda t: 9000 + 0 * t),
    ("FIB", "Ibc", "Ib", -18, 100, lambda t: np.where(t < 0, 0.007 * t**2, np.where(t < 20, 0.08 * t, 1.6 + 0.02 * (t - 20))),
     lambda t: np.clip(6500 - 40 * t, 4000, 8000)),
]
pytestmark = pytest.mark.filterwarnings("ignore:Ibc. subtipos sin plantillas")   # la biblioteca falsa solo tiene Ib
M0 = {"Ia": -19.3, "II": -17.0, "IIb": -17.5, "IIn": -18.0, "Ibc": -17.3}


def _bb(w, T):
    return 1.0 / (w ** 5 * np.expm1(1.4388e8 / (w * T)))


def _store(root):
    for sn, clase, sub, t0, t1, mfun, Tfun in FAKES:
        t = np.arange(t0, t1 + 1.0)
        sed = np.stack([_bb(W, T) / _bb(np.array([6400.0]), T) for T in Tfun(t)])
        flux = 1.2e-3 * 10 ** (-0.4 * (mfun(t) + M0[clase] + 19.0))[:, None] * sed
        save_template(root / "templates" / clase / sn, T_PK + t, W, flux,
                      dict(sn=sn, clase=clase, md5_src="fake", n_epochs=int(t.size), t_first=float(T_PK + t[0]),
                           t_last=float(T_PK + t[-1]), wmin=float(W[0]), wmax=float(W[-1]), n_neg=0))
    pd.DataFrame({"sn": ["FIB"], "subtype": ["Ib"]}).to_csv(root / "ibc.csv", index=False)
    pd.DataFrame({"sn": ["FIIP", "FIIL"], "subtype": ["IIP", "IIL"]}).to_csv(root / "ii.csv", index=False)
    catalog.build_catalog(root, subtypes_csv=root / "ibc.csv", ii_csv=root / "ii.csv")
    return root


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    root = _store(tmp_path_factory.mktemp("store"))
    d = PL.construir_biblioteca(root, out_root=root / "out", z_grid=ZG, log=lambda *a, **k: None)
    return root, d, PL.cargar_biblioteca(d)


def _tpl(root, sn):
    cat = pd.read_csv(root / "catalog.csv").set_index("sn")
    return load_template(cat.store_path[sn])


def _cfg_gen(cls, tpl):
    """ztf_v78_t11 sin ruido (noise_k enorme), deteccion dura con limite 40 y sin stream de alertas: magnitud_modelo
    de project_one es el modelo de las sims con sus bordes."""
    cfg = dict(runcfg.RUNS_CFG["ztf_v78_t11"], bands=["g", "r"], noise_model="snr", noise_k=1e12, sigma_floor=0.0,
               det_model="hard", alert_model=None, pre_ul_mode="window", lc_clean=False, ul_after_last=True,
               noise_draw_scale=1.0, pre_ul_days=30.0)
    cfg["tail_min_slope"] = cfg["tail_min_slope"][cls]
    if tpl["time"][-1] - tpl["time"][0] < cfg.get("tail_min_span", {}).get(cls, 0.0):
        cfg["edge_post"] = "none"
    return cfg


def _modelo_sims(tpl, z, ebv, mw, dmag, tmax, mjd, g_off=0.5):
    """magnitud_modelo de project.project_one (el camino de run.simulate) en las fechas mjd de g y r."""
    cls = tpl["clase"]
    rv = PL.ext_params(cls, tpl["subtype"])[1]["Rv"]
    t_rel, mags = engine.observed_lightcurves(tpl, z, ebv, rv, mw, BANDS, dmag)
    t_exp = (tpl["t_Bmax"] - 18.9 - tpl["t_peak"]) * (1 + z) if cls == "Ia" else None
    ep = {b: (mjd + (g_off if b == "g" else 0.0), np.full(mjd.size, 40.0)) for b in ("g", "r")}
    df = project.project_one(t_rel, mags, ep, tmax, np.random.default_rng(0), _cfg_gen(cls, tpl), t_exp_rel=t_exp, z=z)
    return df.sort_values("mjd").reset_index(drop=True)


def _fake_sn(root, sn, z=0.03, ebv=0.1, mw=0.03, tmax=59000.3, dM=0.3, lim=21.0, seed=1, key="x"):
    """SN falsa de la plantilla sn: M = media de su LF + dM sigma, detecciones con 0.03 mag de ruido bajo lim, UL en
    lim. Devuelve (Curve, parametros verdaderos)."""
    tpl = _tpl(root, sn)
    cls = tpl["clase"]
    m_, s_, _, _ = PL.lf_params(cls, tpl["subtype"], tpl.get("dm15_B"))
    M = m_ + dM * s_
    rv = PL.ext_params(cls, tpl["subtype"])[1]["Rv"]
    dmag = M - tpl["M_ref"]
    if cls in PL.LF_AFTER_HOST_DUST:
        dmag -= engine.host_ext_ref(tpl, ebv, rv, B.rest_bands()[tpl["ref_band"]])
    df = _modelo_sims(tpl, z, ebv, mw, dmag, tmax, np.arange(tmax - 45.0, tmax + 110.0, 2.0))
    rng = np.random.default_rng(seed)
    mm = df.magnitud_modelo.to_numpy(float)
    det = mm < lim
    mag = np.where(det, mm + rng.normal(0, 0.03, mm.size), lim)
    cur = D.Curve(key=key, y=0, t=df.mjd.to_numpy(float), band=(df["filter"] == "r").to_numpy().astype(np.int8),
                  mag=mag.astype(np.float32), err=np.where(det, 0.03, np.nan).astype(np.float32), ul=~det, z=z)
    return cur, dict(cls=D.class_of(cls, True), M=M, ebv=ebv, tmax=tmax, z=z)


# ----------------------------------------------------------------------------- biblioteca
def test_tabla_igual_a_engine(lib):
    root, d, L = lib
    tpl = _tpl(root, "FIIB")
    m, cov, _ = PL.mags_sin_distancia(tpl, 3.1, BANDS, ZG[[2, 8]], PL.EBV_GRID[[0, 4]])
    for i, z in enumerate(ZG[[2, 8]]):
        for j, e in enumerate(PL.EBV_GRID[[0, 4]]):
            _, me = engine.observed_lightcurves(tpl, z, e, 3.1, 0.0, BANDS, 0.0)
            for k, b in enumerate(BANDS):
                assert np.allclose(me[b.name] - PL.mu_z(z), m[i, j, k], atol=2e-5)     # mu_z interpolada
    assert json.loads((d / "meta.json").read_text())["max_dif_engine_mag"] < 1e-6


@pytest.mark.parametrize("sn", ["FIA1", "FIIP", "FIIN"])
def test_tabla_con_los_bordes_de_las_sims(lib, sn):
    """La tabla (con bola de fuego en Ia y la cola con piso) da la magnitud_modelo de project_one en fases enteras."""
    root, d, L = lib
    it = int(np.flatnonzero(L.tab.sn == sn)[0])
    iz, ie, z = 8, 2, float(ZG[8])
    tpl = _tpl(root, sn)
    ebv = float(L.ebv[ie])
    df = _modelo_sims(tpl, z, ebv, 0.0, 0.0, 59000.0, np.arange(59000.0 - 150, 59000.0 + 400), g_off=0.0)
    for ib, b in enumerate("gr"):
        r = df[df["filter"] == b]
        p = np.round(r.mjd.to_numpy() - 59000.0).astype(int)
        F = np.asarray(L.flux[it, iz, ie, ib], float)[p - int(L.ph[0])]
        mm = r.magnitud_modelo.to_numpy(float)
        con = mm < 90
        assert np.all(F[~con] == 0)                                       # sin flujo: antes de la explosion
        ok = con & (F > 1e-3)
        assert ok.sum() > 100
        m_tab = L.lp[it, iz, ie] - 2.5 * np.log10(F[ok]) + PL.mu_z(z)
        assert np.max(np.abs(m_tab - mm[ok])) < 3e-3                     # float16 de la tabla
    if sn == "FIA1":
        assert (df.magnitud_modelo < 90).sum() > 0 and L.tab.set_index("sn").t_exp_reposo[sn] < L.tab.set_index("sn").t0_reposo[sn]


def test_clave_de_cache(lib, tmp_path):
    root, d, L = lib
    k = PL.biblioteca_clave(root, z_grid=ZG)
    assert d.name == f"biblioteca_{k}" and PL.biblioteca_clave(root, z_grid=ZG) == k
    assert PL.biblioteca_clave(root, z_grid=ZG[:-1]) != k
    assert PL.biblioteca_clave(root, z_grid=ZG, ebv_grid=PL.EBV_GRID[:-1]) != k
    assert PL.biblioteca_clave(root, z_grid=ZG, fases=(-150.0, 520.0)) != k
    otro = tmp_path / "otro"
    otro.mkdir()
    (otro / "catalog.csv").write_bytes((root / "catalog.csv").read_bytes())
    assert PL.biblioteca_clave(otro, z_grid=ZG) == k                     # mismo contenido, misma clave
    cat = pd.read_csv(root / "catalog.csv")
    cat.loc[0, "M_ref"] += 0.01
    cat.to_csv(otro / "catalog.csv", index=False)
    assert PL.biblioteca_clave(otro, z_grid=ZG) != k                     # el catalogo cambia -> otra clave
    t = (d / "flux.npy").stat().st_mtime
    assert PL.construir_biblioteca(root, out_root=root / "out", z_grid=ZG) == d and (d / "flux.npy").stat().st_mtime == t


# ----------------------------------------------------------------------------- priors
def test_priors_iguales_a_los_sorteos():
    rng = np.random.default_rng(3)
    e = np.r_[0.0, 0.5 * (PL.EBV_GRID[1:] + PL.EBV_GRID[:-1]), np.inf]
    for cls, sub in [("Ia", "Ia"), ("Ibc", "Ib"), ("Ibc", "Ic"), ("IIb", "IIb"), ("II", "IIP"), ("IIn", "IIn")]:
        x = np.array([sampling.sample_ebv_host(rng, cls, sub)[0] for _ in range(40000)])
        h = np.histogram(x, e)[0] / x.size
        assert np.max(np.abs(h - PL.ebv_prior(cls, sub))) < 0.01, cls
        assert abs(PL.ebv_prior(cls, sub).sum() - 1) < 1e-12
    for cls, sub, dm15 in [("Ia", "Ia", 1.0), ("II", "IIP", None), ("IIn", "IIn", None), ("Ibc", "Ic-BL", None)]:
        x = np.array([sampling.sample_mpeak(rng, cls, dm15, sub) for _ in range(20000)])
        m, s, a, b = PL.lf_params(cls, sub, dm15)
        tn = truncnorm((a - m) / s, (b - m) / s, loc=m, scale=s)      # IIP: el clip -13 queda a 2.2 sigma
        assert abs(x.mean() - tn.mean()) < 0.03 and abs(x.std() - tn.std()) < 0.03, cls
        assert x.min() >= a and x.max() <= b


def test_prior_de_plantillas(lib):
    _, _, L = lib
    p3, p4 = PL.priors(L, four=False), PL.priors(L, four=True)
    t = L.tab.set_index("sn")
    lp = dict(zip(L.tab.sn, np.exp(p4.log_pi)))
    assert np.isclose(lp["FIIP"], 0.8 * 0.875) and np.isclose(lp["FIIL"], 0.8 * 0.125) and np.isclose(lp["FIIB"], 0.2)
    assert np.isclose(lp["FIA1"], 0.5) and np.isclose(lp["FIIN"], 1.0) and np.isclose(lp["FIB"], 1.0)
    assert p3.cls_idx[t.index.get_loc("FIIN")] == -1 and p4.cls_idx[t.index.get_loc("FIIN")] == 3
    assert list(p4.lf_dust) == list(L.tab.clase == "IIn")


# ----------------------------------------------------------------------------- clasificacion
@pytest.mark.parametrize("sn", [s[0] for s in FAKES])
def test_sn_falsa_recupera_clase_y_parametros(lib, sn):
    root, _, L = lib
    pri = PL.priors(L, four=True)
    tpl = _tpl(root, sn)
    ebv = 0.0 if tpl["clase"] == "II" else 0.1          # el polvo de las II del modelo principal es 0
    cur, v = _fake_sn(root, sn, ebv=ebv)
    # con SIGMA_MOD (0.30, elegido en val_sel): la clase y la plantilla. La SN falsa sale de la plantilla misma (sin
    # error del modelo): los parametros se miden con el error del modelo a priori (0.05). Con 0.30 la escala queda
    # menos amarrada (el error del modelo crece con el modelo) y el prior de la LF la corre ~0.3 mag.
    r = PL.clasificar(cur, L, pri, mw=0.03)
    assert pri.clases[int(np.argmax(r["p"]))] == v["cls"] and r["best_template"] == sn
    assert abs(r["tmax_map"] - v["tmax"]) <= 3.0 and abs(r["p"].sum() - 1) < 1e-12
    r = PL.clasificar(cur, L, pri, mw=0.03, sig_mod=PL.SIGMA_MOD_APRIORI)
    assert pri.clases[int(np.argmax(r["p"]))] == v["cls"]
    assert r["best_template"] == sn
    assert abs(r["tmax_map"] - v["tmax"]) <= 2.0
    assert abs(r["ebv_map"] - v["ebv"]) <= 0.1 + 1e-9
    assert abs(r["z_map"] - v["z"]) <= 0.0101
    # el brillo aparente M + mu(z) lo fijan los datos. M solo, con el prior de z (sigma_mu ~0.36 mag a z = 0.03)
    # contra la LF: en las clases de LF ancha y forma lenta (IIn) el MAP se corre de nodo
    assert abs(r["M_map"] + PL.mu_z(r["z_map"]) - v["M"] - PL.mu_z(v["z"])) <= 0.15
    assert abs(r["M_map"] - v["M"]) <= (0.25 if v["cls"] == "Ia" else 0.5)
    assert abs(r["p"].sum() - 1) < 1e-12 and r["n_det"] >= 10 and r["chi2_min"] <= r["chi2_map"] + 1e-6
    assert abs(r["tmax_post"] - v["tmax"]) <= 2.0 and abs(r["z_post"] - v["z"]) <= 0.01
    assert abs(r["M_post"] - v["M"]) <= (0.15 if v["cls"] == "Ia" else 0.4)


def test_sin_z_y_tres_clases(lib):
    root, _, L = lib
    cur, v = _fake_sn(root, "FIA1", z=0.06)
    r = PL.clasificar(cur, L, PL.priors(L, sin_z=True), mw=0.03)
    assert r["prior_z"] == "plano" and len(r["p"]) == 3 and abs(r["p"].sum() - 1) < 1e-12
    assert int(np.argmax(r["p"])) == 0 and abs(r["z_map"] - 0.06) <= 0.02
    r2 = PL.clasificar(replace(cur, z=np.nan), L, PL.priors(L), mw=0.03)      # sin z -> plano
    assert r2["prior_z"] == "plano" and np.allclose(r2["p"], r["p"])
    cur4, _ = _fake_sn(root, "FIIN")
    r4 = PL.clasificar(cur4, L, PL.priors(L, four=False), mw=0.03)            # 3 clases: sin plantillas IIn
    assert len(r4["p"]) == 3 and abs(r4["p"].sum() - 1) < 1e-12 and r4["best_template_clase"] != "IIn"


def test_minimo_de_detecciones(lib):
    root, _, L = lib
    cur, _ = _fake_sn(root, "FIA1")
    d = np.flatnonzero(~cur.ul)
    keep = np.ones(cur.t.size, bool)
    keep[d[2:]] = False
    assert PL.clasificar(cur.subset(keep), L, PL.priors(L)) is None


def test_la_evidencia_usa_los_priors(lib):
    root, _, L = lib
    cur, v = _fake_sn(root, "FIA1", dM=0.0)
    pri = PL.priors(L)
    r = PL.clasificar(cur, L, pri, mw=0.03)
    ia = (L.tab.clase == "Ia").to_numpy()
    lf = pri.lf.copy()
    lf[ia, 0] += 3.0                                    # LF de las Ia 3 mag mas debil (~8 sigma con el prior de z)
    r2 = PL.clasificar(cur, L, replace(pri, lf=lf), mw=0.03)
    assert r2["logE"][0] < r["logE"][0] - 10 and r2["p"][0] <= r["p"][0]   # P(Ia) ~1 igual: las otras van ~-4000
    assert np.allclose(r2["logE"][1:], r["logE"][1:])   # las otras clases no cambian
    p4 = PL.priors(L, four=True)                         # IIn: forma lenta, el prior de la LF mueve M y z posteriores
    cur_n, _ = _fake_sn(root, "FIIN", ebv=0.0)
    iin = (L.tab.clase == "IIn").to_numpy()
    M = {}
    for dm in (-1.0, 0.0, 1.0):
        lf4 = p4.lf.copy()
        lf4[iin, 0] += dm
        r_ = PL.clasificar(cur_n, L, replace(p4, lf=lf4), mw=0.03)
        M[dm] = (r_["M_post"], r_["z_post"])
    # media posterior de M: correr la LF 1 mag la mueve ~ sigma_mu^2/(sigma_M^2 + sigma_mu^2) = 0.13 mag
    assert M[-1.0][0] < M[0.0][0] - 0.05 and M[1.0][0] > M[0.0][0] + 0.05
    assert M[-1.0][1] > M[0.0][1] > M[1.0][1]                                       # mas brillante = mas lejos
    pe = pri.log_pe.copy()
    pe[ia] = np.where(np.arange(pe.shape[1]) == 0, 0.0, -np.inf)          # Ia sin polvo
    cur_d, _ = _fake_sn(root, "FIA1", ebv=0.3)
    a = PL.clasificar(cur_d, L, pri, mw=0.03)
    b = PL.clasificar(cur_d, L, replace(pri, log_pe=pe), mw=0.03)
    assert b["logE"][0] < a["logE"][0] - 5


# ----------------------------------------------------------------------------- reales: la final no se lee
def _real_dir(root, L, tmp):
    rd = tmp / "real_ztf"
    rd.mkdir()
    meta, rows = [], []
    tipos = [("Ia", "FIA1"), ("Ia", "FIA2"), ("II", "FIIP"), ("II", "FIIL"), ("IIb", "FIIB"), ("Ibc", "FIB"), ("IIn", "FIIN")]
    k = 0
    for split in ("val", "final"):
        for rep in range(2):
            for st, sn in tipos:
                oid = f"ZTF{split}{k:03d}"
                k += 1
                cur, _ = _fake_sn(root, sn, ebv=0.0 if st == "II" else 0.05, tmax=59000.3 + 7 * rep, seed=k, key=oid)
                if split == "val" and k == 1:                         # una con 2 detecciones: no entra
                    d = np.flatnonzero(~cur.ul)
                    keep = np.ones(cur.t.size, bool)
                    keep[d[2:]] = False
                    cur = cur.subset(keep)
                meta.append(dict(oid=oid, sn_type=st, subtipo=st, z=0.05, split=split, origen="holdout",
                                 part_index=0, excluir=(split == "val" and k == 2)))
                for t, b, m, e, u in zip(cur.t, cur.band, cur.mag, cur.err, cur.ul):
                    rows.append(dict(oid=oid, sn_type=st, mjd=t, filter="gr"[b], magnitud_proyectada=float(m),
                                     magerr=float(e), upperlimit="T" if u else "F", part_index=0))
    meta.append(dict(oid="ZTFviejo", sn_type="Ia", subtipo="Ia", z=0.05, split="val_viejo", origen="viejas",
                     part_index=0, excluir=False))
    pd.DataFrame(meta).to_csv(rd / "meta_real_ztf.csv", index=False)
    r = pd.DataFrame(rows)
    for st, g in r.groupby("sn_type"):
        g.to_parquet(rd / f"{st}.parquet", index=False)
    return rd, pd.DataFrame(meta)


def test_corrida_sin_tocar_la_final(lib, tmp_path, monkeypatch):
    root, d, L = lib
    rd, meta = _real_dir(root, L, tmp_path)
    final = set(meta.oid[meta.split != "val"])
    vistos, orig_pq, orig_csv = [], D.pq.read_table, pd.read_csv

    def espia(path, columns=None, filters=None, **kw):
        vistos.extend(next(v for k, op, v in filters if k == "oid"))
        t = orig_pq(path, columns=columns, filters=filters, **kw)
        vistos.extend(t.column("oid").to_pylist())
        return t

    def csv_vigilado(p, *a, **kw):
        if str(rd) in str(p):
            raise AssertionError(f"pd.read_csv sobre {p}: leeria la mitad final")
        return orig_csv(p, *a, **kw)

    monkeypatch.setattr(D.pq, "read_table", espia)
    monkeypatch.setattr(pd, "read_csv", csv_vigilado)
    out = PL.run("prueba", real_dir=rd, out_root=tmp_path / "out", lib_dir=d, mw={}, workers=1, n_boot=50)
    out4 = PL.run("prueba4", four=True, real_dir=rd, out_root=tmp_path / "out", lib_dir=d, mw={}, workers=1, n_boot=50)
    monkeypatch.undo()
    assert vistos and not set(vistos) & final
    P, P4 = pd.read_csv(out / "pred_real_val.csv"), pd.read_csv(out4 / "pred_real_val.csv")
    val = set(meta.oid[(meta.split == "val") & ~meta.excluir])
    assert set(P.oid) <= val and not set(P.oid) & final and set(P4.oid) <= val
    for c in ["oid", "subset", "y_true", "y_pred", "p_Ia", "p_II", "p_Ibc", "n_det", "best_template", "chi2_min"]:
        assert c in P.columns
    assert "p_IIn" not in P.columns and "p_IIn" in P4.columns and "IIn" not in set(P.y_true)
    assert np.allclose(P[["p_Ia", "p_II", "p_Ibc"]].sum(1), 1) and set(P.subset) <= {"val_sel", "val_rep"}
    assert (P.y_pred == P.y_true).mean() > 0.8
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["real"]) == {"val_rep", "val_sel", "val"} and "bal_acc_ic95" in m["real"]["val"]
    sc = pd.read_csv(out / "sin_clasificar.csv")
    assert list(sc.oid) == ["ZTFval000"] and m["cobertura"]["val"]["n_clasificadas"] == len(P)
    assert m["cobertura"]["val"]["n_real"] == len(P) + 1
    assert m["tiempo"]["n_sn"] == len(P)


@pytest.mark.parametrize("sig", [0.05, 0.3])
def test_filtro_de_laplace_y_criba_no_cambian_la_evidencia(lib, monkeypatch, sig):
    root, _, L = lib
    pri = PL.priors(L, four=True)
    for sn in ("FIA1", "FIIP", "FIIN", "FIB"):
        cur, _ = _fake_sn(root, sn, ebv=0.0)
        a = PL.clasificar(cur, L, pri, mw=0.03, sig_mod=sig)
        monkeypatch.setattr(PL, "LAPLACE_NATS", np.inf)                  # integral completa en todas las celdas
        b = PL.clasificar(cur, L, pri, mw=0.03, sig_mod=sig)
        fin = np.isfinite(b["logE_plantillas"])
        # el filtro de Laplace (a 30 nats, Laplace en el modo erra < 0.2 nats): < 1e-4 en log E de cada plantilla
        assert np.allclose(a["logE_plantillas"][fin], b["logE_plantillas"][fin], atol=1e-4, rtol=0)
        assert np.allclose(a["p"], b["p"], atol=1e-9) and a["M_post"] == pytest.approx(b["M_post"], abs=1e-6)
        monkeypatch.setattr(PL, "CRIBA_NATS", np.inf)                    # sin la criba de la etapa 1
        c = PL.clasificar(cur, L, pri, mw=0.03, sig_mod=sig)
        monkeypatch.undo()
        # la criba solo cambia plantillas muy por debajo de la mejor (sin esperanza): la mejor y P no se mueven
        lc = c["logE_plantillas"]
        cerca = np.isfinite(lc) & (lc > np.nanmax(lc) - 80.0)
        assert np.allclose(a["logE_plantillas"][cerca], lc[cerca], atol=1e-4, rtol=0)
        assert np.allclose(a["p"], c["p"], atol=1e-9) and a["best_template"] == c["best_template"]


def test_derivadas_de_la_verosimilitud():
    rng = np.random.default_rng(1)
    nd = 30
    for s2m in (0.05 ** 2, 0.3 ** 2):
        f = 10 ** (-0.4 * rng.uniform(0, 3, nd))
        so = (PL.K_MAG * f * rng.uniform(0.02, 0.3, nd)) ** 2
        F = 10 ** (-0.4 * rng.uniform(0, 3, nd))
        ll = lambda d: PL._verosim(f[:, None], so[:, None], s2m, (np.exp(-PL.K_MAG * d) * F)[:, None])[0][0]
        for d in (-2.0, -0.3, 0.7, 2.5):
            _, _, s1, s2 = PL._verosim(f[:, None], so[:, None], s2m, (np.exp(-PL.K_MAG * d) * F)[:, None], True)
            h = 1e-4
            assert -PL.K_MAG * s1[0] == pytest.approx((ll(d + h) - ll(d - h)) / (2 * h), rel=1e-6, abs=1e-6)
            assert PL.K_MAG ** 2 * s2[0] == pytest.approx((ll(d + h) - 2 * ll(d) + ll(d - h)) / h ** 2, rel=1e-4, abs=1e-3)


@pytest.mark.parametrize("s2m", [0.05 ** 2, 0.1 ** 2, 0.3 ** 2])
def test_integral_en_d_contra_fuerza_bruta(s2m):
    """Celdas al azar (la mitad con un modelo que ajusta, la mitad con formas sin relacion a los datos, que dan dos
    modos): modos + integral por tramos contra la integral por fuerza bruta (paso 0.002 mag en +-25 mag)."""
    rng = np.random.default_rng(7)
    nd, n = 30, 80
    f = 10 ** (-0.4 * rng.uniform(0, 2.5, nd))
    so = (PL.K_MAG * f * rng.uniform(0.02, 0.2, nd)) ** 2
    bien = rng.random(n) < 0.5
    Fq = np.where(bien[:, None], f[None] * 10 ** (-0.4 * rng.normal(0, 0.1, (n, nd))),
                  10 ** (-0.4 * rng.uniform(-1, 4, (n, nd)))) * 10 ** (-0.4 * rng.normal(0, 1.0, n))[:, None]
    q0, mp, vp = np.zeros(n), rng.normal(0, 1.0, n), rng.uniform(0.1, 1.5, n) ** 2
    lpr = lambda i, D_: -0.5 * (D_ - mp[i][:, None]) ** 2 / vp[i][:, None] - 0.5 * np.log(2 * np.pi * vp[i][:, None])
    w = 1 / (so + s2m * f * f)
    Fd = Fq.T[None, None]
    with np.errstate(all="ignore"):
        d0 = PL._centro(np.matmul(w, Fd * Fd), np.matmul(w * f, Fd), np.zeros((1, 1)), mp[None, None], vp[None, None])[0][0, 0]
    fo, so_ = f[:, None], so[:, None]
    todas = np.arange(n)
    D_, S, Lm, C, Ul, V = PL._posterior_d(fo, so_, s2m, Fq.T, q0, mp, vp, d0, lambda X: lpr(todas, X),
                                          lambda d: np.zeros_like(d))
    I = PL._integral_modos(fo, so_, s2m, Fq, q0, D_, S, V, Ul, lpr)
    g = np.arange(-25.0, 25.0, 0.002)
    ref = np.array([logsumexp(PL._verosim(fo, so_, s2m, np.exp(-PL.K_MAG * g)[None, :] * Fq[i][:, None])[0]
                              + lpr(np.array([i]), g[None])[0]) + np.log(0.002) for i in range(n)])
    assert np.max(np.abs(I - ref)) < 0.05
    assert np.max(np.abs(I[bien] - ref[bien])) < 1e-6
    lap = logsumexp(np.where(V, Lm, -np.inf), axis=1)
    assert np.max(np.abs(lap - ref)) < 0.5                                # el filtro de 30 nats queda seguro


def _con_ul_contradictorio(cur):
    """Agrega un UL en la noche y banda de la deteccion mas brillante, 3 mag mas profundo (resta fallida)."""
    d = np.flatnonzero(~cur.ul)
    i = d[int(np.argmin(cur.mag[d]))]
    return replace(cur, t=np.r_[cur.t, cur.t[i] + 0.1], band=np.r_[cur.band, cur.band[i]].astype(np.int8),
                   mag=np.r_[cur.mag, cur.mag[i] + 3.0].astype(np.float32), err=np.r_[cur.err, np.nan].astype(np.float32),
                   ul=np.r_[cur.ul, True])


def test_modos_de_ul(lib):
    root, _, L = lib
    pri = PL.priors(L)
    cur, v = _fake_sn(root, "FIA1")
    malo = _con_ul_contradictorio(cur)
    det = ~malo.ul
    assert PL.ul_usados(malo, det, "todos").sum() == malo.ul.sum()
    assert PL.ul_usados(malo, det, "misma_noche").sum() == malo.ul.sum() - 1
    assert not PL.ul_usados(malo, det, "ninguno").any()
    prev = PL.ul_usados(malo, det, "previos")
    assert prev.any() and np.all(malo.t[prev] < malo.t[det].min())
    with pytest.raises(ValueError):
        PL.ul_usados(malo, det, "otro")
    s0 = PL.SIGMA_MOD_APRIORI                         # con 0.30 el error del modelo en el UL lo suaviza (~7 nats)
    limpio = PL.clasificar(cur, L, pri, mw=0.03, ul_modo="todos", sig_mod=s0)
    todos = PL.clasificar(malo, L, pri, mw=0.03, ul_modo="todos", sig_mod=s0)
    noche = PL.clasificar(malo, L, pri, mw=0.03, ul_modo="misma_noche", sig_mod=s0)
    assert todos["logE"][0] < limpio["logE"][0] - 50                    # el UL imposible hunde a las Ia
    assert np.allclose(noche["logE"], limpio["logE"]) and noche["n_ul"] == limpio["n_ul"]
    assert PL.clasificar(malo, L, pri, mw=0.03, ul_modo="ninguno")["n_ul"] == 0
    # por defecto (regla 4b) solo los UL previos: el UL de una noche con la SN detectada no entra
    assert PL.UL_MODO == "previos"
    pre = PL.clasificar(malo, L, pri, mw=0.03)
    assert pre["n_ul"] == prev.sum() < malo.ul.sum()
    assert np.allclose(pre["logE"], PL.clasificar(cur, L, pri, mw=0.03, ul_modo="previos")["logE"])


def _corrida_sigma(root, s_, ok_frac, conf, seed):
    """pred_real_val.csv falso de una corrida de la grilla de sigma_mod: acierta ok_frac, con confianza conf."""
    rng = np.random.default_rng(seed)
    cls = ["Ia", "II", "Ibc"]
    y = np.repeat(cls, 40)
    ok = rng.random(len(y)) < ok_frac
    yp = np.where(ok, y, [cls[(cls.index(c) + 1) % 3] for c in y])
    P = pd.DataFrame({"oid": [f"o{i:03d}" for i in range(len(y))], "subset": "val_sel", "y_true": y, "y_pred": yp,
                      "n_det": 10, "chi2_map": 6.0 / (1 + s_), "chi2_dat_map": 12.0})
    for c in cls:
        P[f"p_{c}"] = np.where(P.y_pred == c, conf, (1 - conf) / 2)
    d = root / "sigma" / f"s{s_:.2f}"
    d.mkdir(parents=True)
    P.to_csv(d / "pred_real_val.csv", index=False)


def test_eleccion_de_sigma(tmp_path):
    # 0.20 calibrada (confianza ~ su exactitud), las demas sobreconfiadas: gana 0.20 con P >= 0.9
    for i, s_ in enumerate(PL.SIGMA_MOD_GRID):
        _corrida_sigma(tmp_path / "a", s_, 0.75, 0.75 if s_ == 0.20 else 0.999, i)
    r = PL.elegir_sigma(out_root=tmp_path / "a", correr=False)
    assert r["candidata"] == 0.20 and r["elegido"] == 0.20 and r["pareado"]["p_mejora"] >= 0.9
    assert json.loads((tmp_path / "a/sigma/eleccion.json").read_text())["elegido"] == 0.20
    t = {x["sigma_mod"]: x for x in r["tabla"]}
    assert t[0.20]["ece"] < t[0.05]["ece"] and "con_error_del_modelo" in t[0.20]["chi2_reducido"]
    # todas iguales salvo ruido: la candidata no pasa la regla y queda la a priori
    for i, s_ in enumerate(PL.SIGMA_MOD_GRID):
        _corrida_sigma(tmp_path / "b", s_, 0.75, 0.80 + 0.001 * i, 0)
    r = PL.elegir_sigma(out_root=tmp_path / "b", correr=False)
    assert r["elegido"] == PL.SIGMA_MOD_APRIORI
