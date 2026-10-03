# tests/test_p78_realism.py
"""Puerta de realismo con datos falsos chicos: seleccion igualada, muestreo estratificado en z, lectura con el lector
real de run_parquet (parquet_reader), reposo /(1+z), color con la ventana de +-2 d, bootstrap y KS."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import pytest
from pipeline78 import realism as R
from pipeline78.pilot_report import peak_mag_8h
from parquet_reader import enumerate_tasks, parse_parquet_lightcurve      # ZLF, en sys.path por pilot_report

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
                                 m_peak_abs=-19.0 + 0.01 * k, ebmv_host=0.01 * (k % 7), rv_host=3.1,
                                 t_anchor=59010.0 + k, status="ok"))
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
    # estratificado en z: por bin, las sims seleccionadas son la cuota n * fraccion real, y la mediana se acerca a la real
    pk = R.sim_peaks(runs["base"])
    pool = sims.assign(m=sims.sim_id.map(pk)).query("m < 18.5")
    for c in ("Ia", "II"):
        zr = real[real.sn_type == c].z
        e = R.z_edges(zr)
        f = np.bincount(R.z_bin(zr, e), minlength=4) / len(zr)
        t, notas = R.bin_targets(10, f, np.bincount(R.z_bin(pool[pool.sn_type == c].z, e), minlength=4))
        b = base[base.sn_type == c]
        assert not notas and (np.bincount(b.bin_z, minlength=4) == t).all() and (b.bin_z == R.z_bin(b.z, e)).all()
        assert np.allclose(b.peso, 1.0)                      # cuotas cumplidas: fraccion real = fraccion seleccionada
        assert abs(b.z.median() - zr.median()) < 0.5 * abs(pool[pool.sn_type == c].z.median() - zr.median())
    assert "bordes z" in printed and "cuota" in printed
    # la salida de prepare pasa por el lector real de run_parquet: cada tarea recibe solo sus filas
    for v in ("base", "tail", "real"):
        ph = pd.concat([pd.read_parquet(q) for q in (out / v / "parquet").glob("*.parquet")], ignore_index=True)
        tasks = enumerate_tasks(out / v / "parquet", cache_path=tmp_path / f"_idx_{v}.parquet", workers=1)
        assert len(tasks) == len(ph.groupby(["oid", "part_index", "sn_type"]))
        for o, k, c, path in tasks[["oid", "part_index", "sn_type", "parquet_path"]].itertuples(index=False):
            fd, name, typ = parse_parquet_lightcurve(path, k, c, oid=o)
            assert name == f"{o}_{c}_p{int(k):02d}" and typ == c
            own = ph[(ph.oid == o) & (ph.part_index == k) & (ph.sn_type == c)]
            for b in ("g", "r"):
                assert np.array_equal(fd[b].MJD.to_numpy(), np.sort(own[own["filter"] == b].mjd.to_numpy())), (v, o, k, b)
    # texp solo Ia; tail omite las que no estan ok y lo reporta
    assert set(selc[selc.variante == "texp"].sn_type) == {"Ia"}
    assert [p.name for p in (out / "texp/parquet").glob("*.parquet")] == ["Ia.parquet"]
    tl = selc[selc.variante == "tail"]
    assert len(tl) and (tl.part_index % 2 == 1).all() and len(tl) == (base.part_index % 2 == 1).sum()
    assert "tail:" in printed and "no ok" in printed
    with pytest.raises(ValueError, match="otra carpeta"):
        R.prepare({k: str(v) for k, v in runs.items()}, out, n=10, seed=1, real_dir=rd)
    assert "_tasks_index.parquet" not in {q.name for q in (out / "base/parquet").iterdir()}
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
    # base y texp cumplen las cuotas: w = 1 y z igualada. tail omitio la mitad de las sims (part_index par), asi que su
    # w se recalcula sobre las que quedan y en esta muestra chica deja de ser uniforme
    assert {"w_max_min", "z_igualada"} <= set(tab.columns)
    bt = tab[tab.variante.isin(["base", "texp"])]
    assert (bt.z_igualada == "sí").all() and np.allclose(bt.w_max_min, 1.0)
    tl = tab[tab.variante == "tail"]
    assert ((tl.w_max_min > R.W_RATIO) == (tl.z_igualada == "no")).all() and (tl.z_igualada == "no").any()
    assert "z_igualada = no" in (tmp_path / "page/index.html").read_text()
    # reposo: t/(1+z) con la z de la seleccion, y la mediana de la tabla sale de esos valores
    ft = pd.read_csv(out / "base/features/features.csv").query("filter_band == 'r'")
    x = R.load_variant(out, "base", base)
    m = x.merge(ft[["oid", "part_index", "sn_type", "t_rise", "gamma"]], on=["oid", "part_index", "sn_type"],
                suffixes=("", "_csv"))
    assert len(m) and np.allclose(m.t_rise_rest, m.t_rise_csv / (1 + m.z)) and np.allclose(m.gamma_rest, m.gamma_csv / (1 + m.z))
    row = tab[(tab.clase == "Ia") & (tab.variante == "base") & (tab.par == "t_rise_rest")].iloc[0]
    assert np.isclose(row.med_sim, np.median((m.t_rise_csv / (1 + m.z))[m.sn_type == "Ia"]))
    # m1: un _tasks_index (u otro parquet ajeno) en la carpeta no entra a la fotometria
    pd.DataFrame({"oid": ["X"], "part_index": [0], "sn_type": ["Ia"]}).to_parquet(out / "base/parquet/_tasks_index.parquet")
    pd.testing.assert_frame_equal(R.load_variant(out, "base", base), x)


@pytest.mark.parametrize("campo,delta", [("z", 0.001), ("ebmv_host", 0.01), ("t_anchor", 1.0)])
def test_prepare_refuses_other_physics_and_writes_nothing(tmp_path, campo, delta):
    runs, rd, sims, meta = _fake(tmp_path)
    s = pd.read_parquet(runs["tail"] / "_sims_all.parquet")
    s[campo] = s[campo] + delta
    s.to_parquet(runs["tail"] / "_sims_all.parquet", index=False)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="otra fisica"):
        R.prepare({k: str(v) for k, v in runs.items()}, out, n=5, seed=1, real_dir=rd)
    assert not out.exists()                                  # m2: se valida antes de escribir


def _prepared(tmp_path):
    runs, rd, sims, meta = _fake(tmp_path)
    out = tmp_path / "out"
    selc, real = R.prepare({k: str(v) for k, v in runs.items()}, out, n=10, seed=1, real_dir=rd)
    d = tmp_path / "fireball"                                # run nuevo: mismas sims, solo Ia cambian
    d.mkdir()
    sims.to_parquet(d / "_sims_all.parquet", index=False)
    for f in set(sims.field):                                # fotometria distinta de base: se ve de que run sale
        x = pd.read_parquet(runs["base"] / f"{f}__00000.parquet")
        x["magnitud_proyectada"] = x.magnitud_proyectada + 0.5
        x.to_parquet(d / f"{f}__00000.parquet", index=False)
    return runs, rd, out, selc, real, d


def test_add_variant(tmp_path):
    runs, rd, out, selc, real, d = _prepared(tmp_path)
    s = R.add_variant(out, "fireball", d, ("Ia",))
    base = selc[(selc.variante == "base") & (selc.sn_type == "Ia")]
    assert set(s.sim_id) == set(base.sim_id)                 # mismas sim_id, solo Ia
    assert [p.name for p in (out / "fireball/parquet").glob("*.parquet")] == ["Ia.parquet"]
    ph = pd.read_parquet(out / "fireball/parquet/Ia.parquet")
    assert set(zip(ph.oid, ph.part_index)) == set(zip(base.oid, base.part_index))
    bph = pd.read_parquet(out / "base/parquet/Ia.parquet")
    m = ph.merge(bph, on=["sim_id", "mjd", "filter"], suffixes=("_v", "_b"))
    assert len(m) == len(ph) and np.allclose(m.magnitud_proyectada_v, m.magnitud_proyectada_b + 0.5)   # sale del --run
    sel = pd.read_csv(out / "selection.csv")
    assert set(sel[sel.variante == "fireball"].sim_id) == set(base.sim_id)
    assert len(sel) == len(selc) + len(base)
    with pytest.raises(ValueError, match="ya existe"):
        R.add_variant(out, "fireball", d, ("Ia",))
    assert len(pd.read_csv(out / "selection.csv")) == len(sel)


def test_add_variant_other_physics_writes_nothing(tmp_path):
    runs, rd, out, selc, real, d = _prepared(tmp_path)
    s = pd.read_parquet(d / "_sims_all.parquet")
    s["z"] = s.z + 0.001
    s.to_parquet(d / "_sims_all.parquet", index=False)
    _refuse(out, d)


def test_add_variant_rv_host_checked(tmp_path):
    runs, rd, out, selc, real, d = _prepared(tmp_path)
    s = pd.read_parquet(d / "_sims_all.parquet")
    s["rv_host"] = s.rv_host + 0.1
    s.to_parquet(d / "_sims_all.parquet", index=False)
    _refuse(out, d)


def test_add_variant_nan_physics_is_equal(tmp_path):
    runs, rd, out, selc, real, d = _prepared(tmp_path)
    for p in (d, runs["base"]):                              # NaN en los dos lados: misma fisica
        s = pd.read_parquet(p / "_sims_all.parquet")
        s["ebmv_host"] = s.ebmv_host.astype(float)
        s.loc[s.sn_type == "Ia", "ebmv_host"] = np.nan
        s.to_parquet(p / "_sims_all.parquet", index=False)
    R.add_variant(out, "fireball", d, ("Ia",))


def _refuse(out, d):
    before = (out / "selection.csv").read_text()
    with pytest.raises(ValueError, match="otra fisica"):
        R.add_variant(out, "fireball", d, ("Ia",))
    assert not (out / "fireball").exists() and (out / "selection.csv").read_text() == before


def test_report_takes_all_variants(tmp_path):
    runs, rd, out, selc, real, d = _prepared(tmp_path)
    R.add_variant(out, "fireball", d, ("Ia",))
    R.add_variant(out, "otra", d, ("Ia",))
    sel = pd.read_csv(out / "selection.csv")
    frng = np.random.default_rng(2)
    for v in ("base", "texp", "fireball", "tail", "otra"):
        _features(out, v, sel[sel.variante == v], frng)
    _features(out, "real", real.assign(part_index=0), frng)
    tab = R.report(out, page=tmp_path / "page", n_boot=50)
    assert set(tab[tab.clase == "Ia"].variante) == {"base", "texp", "fireball", "tail", "otra"}
    html = (tmp_path / "page/index.html").read_text()
    assert "base, texp, fireball, tail, otra" in html


def test_photo_color_window():
    """g - r: solo las g a +-2 d del pico r (dos noches) entran en la mediana; las de 3 y 5 d quedan fuera."""
    t = 59000.0 + 2.0 * np.arange(10)                       # pico r en 59010
    rows = [("o", 0, "Ia", ti, "r", 17.0 + 0.05 * abs(ti - 59010.0), 0.05, "F") for ti in t]
    g = [(59008.2, 17.4), (59010.5, 17.2), (59013.0, 15.0), (59005.0, 15.5)]
    rows += [("o", 0, "Ia", ti, "g", mi, 0.05, "F") for ti, mi in g]
    m, gr = R.photo(pd.DataFrame(rows, columns=COLS))
    assert m == 17.0 and np.isclose(gr, np.median([17.4, 17.2]) - 17.0)


def test_z_weights_fix_low_z_concentration():
    """Sims concentradas a z baja: la mediana ponderada por w = f_real / f_sim del bin queda cerca de la real."""
    rng = np.random.default_rng(3)
    zr = rng.uniform(0.02, 0.06, 40)
    zs = np.concatenate([rng.uniform(0.005, 0.03, 40), rng.uniform(0.03, 0.06, 12)])
    w = R.z_match_weights(zs, zr)
    e = R.z_edges(zr)
    assert (np.bincount(R.z_bin(zs, e), minlength=4) > 0).all()
    assert w.max() / w.min() > R.W_RATIO                          # esta clase saldria z_igualada = no
    d_w, d_u = abs(R.wmedian(zs, w) - np.median(zr)), abs(np.median(zs) - np.median(zr))
    assert d_w < 0.5 * d_u, (d_w, d_u)
    assert np.allclose(R.z_match_weights(zr, zr), 1.0)
    # pesos iguales: es np.median (n par y n impar)
    assert R.wmedian([4.0, 1.0, 3.0, 2.0], [1, 1, 1, 1]) == 2.5 and R.wmedian([3.0, 1.0, 2.0], [0.7] * 3) == 2.0
    # bootstrap ponderado: con todo el peso en las sims chicas la diferencia de medianas no varia
    sig = R.boot_sigma(np.array([1.0, 1.0, 9.0]), np.array([1.0, 1.0]), np.random.default_rng(0), 200,
                       np.array([1.0, 1.0, 0.0]))
    assert sig == 0.0


def test_wmedian_equal_weights_any_value():
    """Pesos iguales de cualquier valor (no solo 1) dan np.median, con n par e impar."""
    x = np.array([4.0, 1.0, 3.0, 2.0])
    assert R.wmedian(x, [0.7] * 4) == 2.5 and R.wmedian(np.arange(6.0), [0.7] * 6) == 2.5
    rng = np.random.default_rng(7)
    for n in range(1, 61):
        for w in (0.1, 0.7, 1 / 3, 2.5, *rng.uniform(1e-3, 50.0, 5)):
            x = rng.normal(0.0, 1.0, n)
            assert R.wmedian(x, np.full(n, w)) == np.median(x), (n, w)


def test_bin_targets():
    t, notas = R.bin_targets(50, [13 / 50, 11 / 50, 13 / 50, 13 / 50], [197, 111, 43, 49])
    assert t.tolist() == [13, 11, 13, 13] and not notas
    t, notas = R.bin_targets(50, [0.25] * 4, [24, 36, 10, 5])          # bins altos cortos: el faltante va al vecino
    assert t.sum() == 50 and t[2] == 10 and t[3] == 5 and len(notas) == 2
    t, notas = R.bin_targets(10, [0.25] * 4, [1, 1, 1, 1])             # no llega a n
    assert t.tolist() == [1, 1, 1, 1] and "no hay sims libres" in notas[0]


def test_boot():
    rng = np.random.default_rng(0)
    assert np.isnan(R.boot_sigma(np.array([1.0]), np.array([1.0, 2.0]), rng))
    assert R.boot_sigma(rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng) > 0


if __name__ == "__main__":
    print("usar pytest (fixtures tmp_path y capsys)")
