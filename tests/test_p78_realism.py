# tests/test_p78_realism.py
"""Puerta de realismo con datos falsos chicos: seleccion igualada, re-pesado en z, bootstrap y KS."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import pytest
from pipeline78 import realism as R
from pipeline78.pilot_report import peak_mag_8h

COLS = ["oid", "part_index", "sn_type", "mjd", "filter", "magnitud_proyectada", "magerr", "upperlimit"]


def _lc(oid, k, cls, m_pk, n_nights, rng):
    """r: n_nights detecciones cada 2 d con el pico m_pk en el dia 10; g cerca del pico; 3 UL antes."""
    t = 59000.0 + 2.0 * np.arange(n_nights)
    r = m_pk + 0.05 * np.abs(t - 59010.0)
    g = r[: min(8, n_nights)] - 0.1
    rows = [(oid, k, cls, ti, "r", mi, 0.05, "F") for ti, mi in zip(t, r)]
    rows += [(oid, k, cls, ti + 0.1, "g", mi, 0.05, "F") for ti, mi in zip(t[:g.size], g)]
    rows += [(oid, k, cls, 58990.0 + j, "r", 20.5, np.nan, "T") for j in range(3)]
    return pd.DataFrame(rows, columns=COLS)


def _fake(td):
    rng = np.random.default_rng(11)
    sims, ph = [], []
    for c in ("Ia", "II"):
        for f in ("F0", "F1", "F2"):
            for k in range(80):
                sid = len(sims) + 1
                z, m = rng.uniform(0.005, 0.12), 17.0 + rng.uniform(0, 2.5)
                nn = 12 if k % 5 else 5                     # 1 de cada 5 con < 7 puntos
                sims.append(dict(sim_id=sid, field=f, part_index=k, sn_type=c, template=f"T{c}{k % 3}", z=z,
                                 m_peak_abs=-19.0 + 0.01 * k, status="ok"))
                ph.append(_lc(f, k, c, m, nn, rng).assign(sim_id=sid))
    sims, ph = pd.DataFrame(sims), pd.concat(ph, ignore_index=True)
    runs = {}
    for v, cfg in (("base", {}), ("texp", {"edge_pre": "texp"}), ("tail", {"edge_post": "tail"})):
        d = td / v
        d.mkdir()
        s = sims.copy()
        if v == "tail":
            s.loc[s.part_index % 2 == 0, "status"] = "no_epochs"   # se omiten en la variante
        s.to_parquet(d / "_sims_all.parquet", index=False)
        for f, x in ph.groupby("oid"):
            x.to_parquet(d / f"{f}__00000.parquet", index=False)
        (d / "run_manifest.json").write_text(json.dumps(dict(cfg=cfg, seed=1)))
        runs[v] = d
    rd = td / "real"
    rd.mkdir()
    meta, rph = [], {}
    for c in ("Ia", "II"):
        for j in range(50):
            o = f"ZTFR{c}{j:03d}"
            orig, split = ("holdout", "val") if j < 30 else (("holdout", "final") if j < 40 else ("viejas", "val_viejo"))
            meta.append(dict(oid=o, sn_type=c, subtipo=c, z=rng.uniform(0.01, 0.05), split=split, origen=orig, part_index=0))
            rph.setdefault(c, []).append(_lc(o, 0, c, 16.5 + rng.uniform(0, 2.5), 12 if j % 6 else 5, rng))
    pd.DataFrame(meta).to_csv(rd / "meta_real_ztf.csv", index=False)
    for c, fr in rph.items():
        pd.concat(fr, ignore_index=True).astype({"part_index": np.int32}).to_parquet(rd / f"{c}.parquet", index=False)
    return runs, rd, sims, pd.DataFrame(meta)


def _features(out, v, s, rng, drop=0):
    rows = []
    for o, k, c, z in s[["oid", "part_index", "sn_type", "z"]].itertuples(index=False):
        for b in ("g", "r"):
            rows.append(dict(sn_name=f"{o}_{c}_p{k:02d}", filter_band=b, A=1.0, f=rng.uniform(0, 1),
                             t0=0.0, t_rise=rng.uniform(2, 6) * (1 + z), t_fall=rng.uniform(20, 60) * (1 + z),
                             gamma=rng.uniform(5, 40) * (1 + z), sn_type=c, oid=o, part_index=k))
    ft = pd.DataFrame(rows).iloc[2 * drop:]                 # las primeras `drop` sin ajuste valido
    (out / v / "features").mkdir(parents=True, exist_ok=True)
    ft.to_csv(out / v / "features/features.csv", index=False)


def test_prepare_and_report(tmp_path, capsys):
    runs, rd, sims, meta = _fake(tmp_path)
    out = tmp_path / "out"
    selc, real = R.prepare({k: str(v) for k, v in runs.items()}, out, n=10, seed=1, real_dir=rd)
    printed = capsys.readouterr().out
    base = selc[selc.variante == "base"]
    # n por clase, corte m < 18.5 y >= 7 puntos agrupados (se recalcula de la fotometria escrita)
    assert (base.groupby("sn_type").size() <= 10).all() and (real.groupby("sn_type").size() <= 10).all()
    assert set(base.sn_type) == {"Ia", "II"} and set(real.sn_type) == {"Ia", "II"}
    for v, s in (("base", base), ("real", real.assign(part_index=0))):
        ph = pd.concat([pd.read_parquet(p) for p in (out / v / "parquet").glob("*.parquet")])
        assert set(zip(ph.oid, ph.part_index)) == set(zip(s.oid, s.part_index))
        for (o, k), d in ph.groupby(["oid", "part_index"]):
            x = d[(d["filter"] == "r") & (d.upperlimit == "F")].rename(
                columns={"mjd": "MJD", "magnitud_proyectada": "MAG", "magerr": "MAGERR"})
            m = peak_mag_8h(x)
            assert np.isfinite(m) and m < 18.5 and R.photo(d)[0] == m
    hv = set(meta[(meta.origen == "holdout") & (meta.split == "val")].oid)
    assert set(real.oid) <= hv
    # re-pesado en z: la mediana de z seleccionada queda mas cerca de la real que la del pool que pasa el corte
    pk = R.sim_peaks(runs["base"])
    pool = sims.assign(m=sims.sim_id.map(pk)).query("m < 18.5")
    for c in ("Ia", "II"):
        zr = real[real.sn_type == c].z.median()
        assert abs(base[base.sn_type == c].z.median() - zr) < 0.5 * abs(pool[pool.sn_type == c].z.median() - zr)
    # texp solo Ia; tail omite las que no estan ok y lo reporta
    assert set(selc[selc.variante == "texp"].sn_type) == {"Ia"}
    assert [p.name for p in (out / "texp/parquet").glob("*.parquet")] == ["Ia.parquet"]
    tl = selc[selc.variante == "tail"]
    assert len(tl) and (tl.part_index % 2 == 1).all() and len(tl) == (base.part_index % 2 == 1).sum()
    assert "tail:" in printed and "no ok" in printed
    with pytest.raises(ValueError, match="otra carpeta"):
        R.prepare({k: str(v) for k, v in runs.items()}, out, n=10, seed=1, real_dir=rd)
    # report: features falsas, bootstrap y KS
    frng = np.random.default_rng(2)
    for v in ("base", "texp", "tail"):
        _features(out, v, selc[selc.variante == v], frng, drop=1 if v == "base" else 0)
    _features(out, "real", real.assign(part_index=0), frng)
    tab = R.report(out, page=tmp_path / "page", n_boot=200)
    need = {"clase", "variante", "par", "n_sim", "n_real", "med_sim", "med_real", "delta", "sigma_boot", "ks_D", "ks_p",
            "frac_validos_sim"}
    assert need <= set(tab.columns) and (out / "realismo_tabla.csv").exists()
    assert set(tab.par) == set(R.PARS) and set(tab[tab.clase == "II"].variante) == {"base", "tail"}
    fit = tab[tab.par.isin(["t_rise_rest", "f"])]
    assert (fit.sigma_boot > 0).all() and fit.ks_p.between(0, 1).all() and fit.ks_D.between(0, 1).all()
    b = tab[(tab.variante == "base") & (tab.par == "f")]
    assert (b.frac_validos_sim < 1).any()
    assert tab[tab.par == "M_r"].n_sim.gt(0).all() and tab[tab.par == "g_r"].n_sim.gt(0).all()
    assert (tmp_path / "page/index.html").exists() and (tmp_path / "page/realismo.png").exists()


def test_prepare_refuses_other_physics(tmp_path):
    runs, rd, sims, meta = _fake(tmp_path)
    s = pd.read_parquet(runs["tail"] / "_sims_all.parquet")
    s["z"] = s.z + 0.001
    s.to_parquet(runs["tail"] / "_sims_all.parquet", index=False)
    with pytest.raises(ValueError, match="otra fisica"):
        R.prepare({k: str(v) for k, v in runs.items()}, tmp_path / "out", n=5, seed=1, real_dir=rd)


def test_z_weights_and_boot():
    w = R.z_weights([0.01, 0.02, 0.2, 0.21, 0.22, 0.23], [0.01, 0.015, 0.02, 0.03])
    assert w[2:].sum() < w[:2].sum()
    rng = np.random.default_rng(0)
    assert np.isnan(R.boot_sigma(np.array([1.0]), np.array([1.0, 2.0]), rng))
    assert R.boot_sigma(rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng) > 0


if __name__ == "__main__":
    print("usar pytest (fixtures tmp_path y capsys)")
