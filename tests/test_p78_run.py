# tests/test_p78_run.py
import sys, os, pathlib, tempfile, importlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

def _setup(td, n=6):
    os.environ["P78_STORE"] = td; os.environ["P78_RUNS"] = td
    import pipeline78.paths, pipeline78.store, pipeline78.catalog, pipeline78.runcfg, pipeline78.run
    for m in (pipeline78.paths, pipeline78.store, pipeline78.catalog, pipeline78.runcfg, pipeline78.run): importlib.reload(m)
    from tests.p78_fakes import fake_template
    fake_template(pathlib.Path(td) / "templates/Ia/FAKE1", "FAKE1", "Ia", 55000.0)
    fake_template(pathlib.Path(td) / "templates/Ia/FAKE2", "FAKE2", "Ia", 55000.0)
    pipeline78.catalog.build_catalog(pathlib.Path(td))
    mjd = np.arange(54900.0, 55300.0, 2.0)
    log = pd.DataFrame([("F1", m, b, 20.5) for m in mjd for b in "gr"], columns=["field", "mjd", "band", "maglim"])
    log.to_parquet(pathlib.Path(td) / "log.parquet", index=False)
    (pathlib.Path(td) / "fields.txt").write_text("F1\n")
    cfg = dict(survey="ZTF", bands=["g", "r", "i"], classes=["Ia"], n_by_class={"Ia": n}, chunk=None,
               anchor="pivot", z_mode="fixed", z_fixed=0.02, mw_mode="const", mw_const=0.02, rule="ztf",
               pre_ul_days=25.0, noise_k=5.0, sigma_floor=0.02, log_path=str(pathlib.Path(td) / "log.parquet"))
    pipeline78.runcfg.RUNS_CFG["_test"] = cfg
    return pipeline78.run, cfg

def _argv(td, out="r1", seed=1):
    return ["--run", "_test", "--out", str(pathlib.Path(td) / out), "--fields-file", str(pathlib.Path(td) / "fields.txt"),
            "--seed", str(seed), "--workers", "1", "--allow-dirty"]

def test_end_to_end_and_determinism():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        out1 = run.main(_argv(td, "r1")); out2 = run.main(_argv(td, "r2"))
        s = pd.read_parquet(out1 / "_sims_all.parquet")
        assert len(s) == 6 and set(s.status) == {"ok"}
        a = pd.read_parquet(next(out1.glob("F1__*.parquet"))); b = pd.read_parquet(next(out2.glob("F1__*.parquet")))
        pd.testing.assert_frame_equal(a, b)
        need = {"oid", "part_index", "sn_type", "mjd", "filter", "magnitud_proyectada", "magerr", "upperlimit", "sim_id"}
        assert need <= set(a.columns)
        assert (a.loc[a.upperlimit == "T", "magerr"].isna()).all()

def test_no_coverage_is_logged():
    with tempfile.TemporaryDirectory() as td:
        run, cfg = _setup(td, n=2)
        cfg["z_fixed"] = 3.0          # todas las bandas fuera de la biblioteca
        out = run.main(_argv(td, "r3"))
        s = pd.read_parquet(out / "_sims_all.parquet")
        assert len(s) == 2 and set(s.status) == {"no_coverage"}

def test_resume_skips_complete_units():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        out = run.main(_argv(td, "r4"))
        (out / "_sims_all.parquet").unlink()
        t0 = next((out / "_sims").glob("*.parquet")).stat().st_mtime
        run.main(_argv(td, "r4"))
        assert next((out / "_sims").glob("*.parquet")).stat().st_mtime == t0
        assert not list(out.glob("*.tmp"))

def test_refuses_config_mismatch():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        run.main(_argv(td, "r5", seed=1))
        try:
            run.main(_argv(td, "r5", seed=2)); raise AssertionError("debio negarse")
        except SystemExit as e:
            assert "otra configuracion" in str(e)

def test_field_without_log_is_recorded():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        (pathlib.Path(td) / "fields.txt").write_text("F1\nNOLOG\n")
        out = run.main(_argv(td, "r6"))
        assert (out / "_sin_log.txt").read_text().split() == ["NOLOG"]
        s = pd.read_parquet(out / "_sims_all.parquet")
        assert len(s) == 6 and set(s.field) == {"F1"}

def test_refuses_other_commit():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        run.main(_argv(td, "r7"))
        orig = run.git_state
        run.git_state = lambda: ("deadbeef", False)
        try:
            run.main(_argv(td, "r7")); raise AssertionError("debio negarse")
        except SystemExit as e:
            assert "otro commit" in str(e)
        finally:
            run.git_state = orig

def test_refuses_changed_log():
    with tempfile.TemporaryDirectory() as td:
        run, cfg = _setup(td)
        run.main(_argv(td, "r8"))
        h1 = run.config_hash(cfg)
        lp = pathlib.Path(cfg["log_path"]); df = pd.read_parquet(lp); df.loc[0, "maglim"] = 19.0
        df.to_parquet(lp, index=False)
        assert run.config_hash(cfg) != h1
        try:
            run.main(_argv(td, "r8")); raise AssertionError("debio negarse")
        except SystemExit as e:
            assert "otra configuracion" in str(e)

def test_subtype_fractions_choose_template():
    import pipeline78.run as r
    fr = {"Ic-BL": 0.05, "Ib": 0.27, "Ic": 0.68}
    tpls = [dict(sn=f"{st}{i}", subtype=st) for st in fr for i in range(4)]
    rng = np.random.default_rng(5)
    n = 20000
    got = [r.choose_template(1, "F1", "Ibc", k, rng, tpls, fr)["subtype"] for k in range(n)]
    for st, f in fr.items():
        assert abs(np.mean([g == st for g in got]) - f) < 0.01, st
    # sin fracciones: igual que la permutacion original y sin consumir rng
    rng2 = np.random.default_rng(5); st0 = rng2.bit_generator.state
    t = r.choose_template(1, "F1", "Ibc", 3, rng2, tpls, None)
    assert rng2.bit_generator.state == st0
    order = np.random.default_rng([1, r._h("F1") & 0xFFFFFFFF, r._h("Ibc") & 0xFFFFFFFF]).permutation(len(tpls))
    assert t is tpls[order[3 % len(tpls)]]

def test_missing_subtype_templates_raises():
    import pipeline78.run as r
    tpls = [dict(sn="a", subtype="Ib")]
    try:
        r._by_subtype(tpls, {"Ib": 0.5, "Ic-BL": 0.5}, "Ibc"); raise AssertionError("debio fallar")
    except ValueError as e:
        assert "Ic-BL" in str(e)

def test_orphan_subtype_warns():
    import warnings
    import pipeline78.run as r
    tpls = [dict(sn="a", subtype="Ib"), dict(sn="b", subtype="Ic")]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r._by_subtype(tpls, {"Ib": 1.0}, "Ibc")
    assert any("'b'" in str(x.message) for x in w)

def test_w_z_column_and_hash_includes_fractions():
    with tempfile.TemporaryDirectory() as td:
        run, cfg = _setup(td, n=3)
        out = run.main(_argv(td, "r9"))
        s = pd.read_parquet(out / "_sims_all.parquet")
        assert (s.w_z == 1.0).all()
        assert "w_z" in pd.read_parquet(next(out.glob("F1__*.parquet"))).columns
        h1 = run.config_hash(cfg)
        run.SUBTYPE_FRACTIONS["Ia"] = {"Ia": 1.0}
        try:
            assert run.config_hash(cfg) != h1
        finally:
            run.SUBTYPE_FRACTIONS.pop("Ia")

def test_untracked_source_is_dirty():
    import pipeline78.run as r
    f = pathlib.Path(r.REPO) / "pipeline78" / "_zz_untracked_probe.py"
    f.write_text("x=1\n")
    try:
        assert r.git_state()[1] is True
    finally:
        f.unlink()

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)

def _real_ii():
    import pandas as pd
    from pipeline78.store import load_template
    c = pd.read_csv(pathlib.Path.home() / "thesis_store" / "catalog.csv")     # el real, aunque otros tests cambien P78_STORE
    return [load_template(p) for p in c[c.clase == "II"].sort_values("sn").store_path]

def test_fixc_peak_mag_equals_M_after_dust():
    # con la regla nueva, el pico enrojecido en la banda de referencia de reposo es M
    from pipeline78 import bands as B, engine
    from pipeline78.engine import extinction_factor
    rb = B.rest_bands()
    for tpl in _real_ii():
        band = rb[tpl["ref_band"]]
        M, ebv, rv = -16.5, 0.3, 3.1
        A = engine.host_ext_ref(tpl, ebv, rv, band)
        assert A > 0.3
        dmag = M - tpl["M_ref"] - A
        i = int(np.argmin(np.abs(tpl["time"] - tpl["t_peak"])))
        f = tpl["flux"][i:i + 1] * 10 ** (-0.4 * dmag) * extinction_factor(tpl["wave"], rv, ebv)[None, :]
        F, _ = B.synphot(tpl["wave"], f, band)
        m = float(-2.5 * np.log10(F[0] / band.f0))
        assert abs(m - M) < 0.005, (tpl["sn"], m, M)

def test_fixc_dmag_unchanged_other_classes_and_A_ref_zero():
    import pipeline78.run as r
    from config import LF_AFTER_HOST_DUST
    assert not ({"Ia", "IIb", "Ibc"} & LF_AFTER_HOST_DUST)
    from pipeline78.engine import host_ext_ref
    assert host_ext_ref(dict(), 0.0, 3.1, None) == 0.0
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td, n=3)
        out = run.main(_argv(td, "r_fc"))
        s = pd.read_parquet(out / "_sims_all.parquet")
        assert (s.A_ref_host == 0.0).all() and s.m_peak_abs.notna().all()

def test_fixc_ii_subtype_frequencies_real_catalog():
    import pipeline78.run as r
    from config import SUBTYPE_FRACTIONS
    tpls = _real_ii()
    fr = SUBTYPE_FRACTIONS["II"]
    rng = np.random.default_rng(9)
    got = [r.choose_template(1, "F1", "II", k, rng, tpls, fr)["subtype"] for k in range(20000)]
    for st, f in fr.items():
        assert abs(np.mean([g == st for g in got]) - f) < 0.01, st
