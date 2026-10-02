# tests/test_p78_catalog.py
import sys, os, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

def test_peak_and_dm15_gaussian():
    from pipeline78.catalog import peak_and_dm15
    t = np.arange(0.0, 100.0)
    m = 0.01 * (t - 30.0) ** 2
    tp, mp, edge, dm15 = peak_and_dm15(t, m)
    assert tp == 30.0 and not edge and abs(dm15 - 2.25) < 1e-9

def test_main_peak_double_peak():
    from pipeline78.catalog import main_peak_index, peak_and_dm15
    t = np.arange(0.0, 100.0)
    early = -1.0 * np.exp(-0.5 * ((t - 1.0) / 1.5) ** 2)       # pico de enfriamiento, mag -1.0
    main = -0.6 * np.exp(-0.5 * ((t - 20.0) / 8.0) ** 2)       # pico de Ni, 0.4 mag mas debil
    m = 1.5 + early + main
    # dip entre ambos ~0.5 mag
    assert t[int(np.argmin(m))] == 1.0
    assert t[main_peak_index(t, m)] == 20.0
    tp, mp, edge, dm15 = peak_and_dm15(t, m)
    assert tp == 20.0 and not edge

def test_main_peak_tiny_wiggle_keeps_argmin():
    from pipeline78.catalog import main_peak_index
    t = np.arange(0.0, 100.0)
    # maximo en d4, caida a 0.5 mag en d12, rebote de solo 0.03 mag con minimo local de brillo en d20
    m = np.interp(t, [0, 4, 12, 20, 40, 99], [0.3, 0.0, 0.5, 0.47, 1.5, 3.0])
    m = np.where(t >= 20, np.interp(t, [20, 28, 40, 99], [0.47, 0.60, 1.5, 3.0]), m)
    assert t[int(np.argmin(m))] == 4.0
    j = 20
    assert m[j] == m[17:24].min() and 0.0 < m[12] - m[j] < 0.05
    assert main_peak_index(t, m) == int(np.argmin(m))

def test_main_peak_single_gaussian_unchanged():
    from pipeline78.catalog import main_peak_index
    t = np.arange(0.0, 100.0)
    m = 0.01 * (t - 30.0) ** 2
    assert main_peak_index(t, m) == 30

def test_build_catalog_on_fake_store():
    with tempfile.TemporaryDirectory() as td:
        os.environ["P78_STORE"] = td
        import importlib, pipeline78.paths, pipeline78.store, pipeline78.catalog
        for m in (pipeline78.paths, pipeline78.store, pipeline78.catalog): importlib.reload(m)
        from tests.p78_fakes import fake_template
        fake_template(pathlib.Path(td) / "templates/Ia/FAKE1", t_peak=55000.0)
        cat = pipeline78.catalog.build_catalog(pathlib.Path(td))
        r = cat.iloc[0]
        assert abs(r.t_peak - 55000.0) <= 1.0 and r.clf_class == "Ia" and np.isfinite(r.M_ref)
        assert np.isfinite(r.dm15_B)
        assert not r.early_peak and r.t_peak_argmin == r.t_peak

def test_ibc_subtype_from_csv_and_missing_raises():
    with tempfile.TemporaryDirectory() as td:
        os.environ["P78_STORE"] = td
        import importlib, pipeline78.paths, pipeline78.store, pipeline78.catalog
        for m in (pipeline78.paths, pipeline78.store, pipeline78.catalog): importlib.reload(m)
        from tests.p78_fakes import fake_template
        fake_template(pathlib.Path(td) / "templates/Ia/FAKE1", t_peak=55000.0)
        fake_template(pathlib.Path(td) / "templates/Ibc/FAKEIBC", sn="FAKEIBC", clase="Ibc", t_peak=55000.0)
        csv = pathlib.Path(td) / "sub.csv"
        csv.write_text("sn,subtype,source\nFAKEIBC,Ic,test\n")
        cat = pipeline78.catalog.build_catalog(pathlib.Path(td), subtypes_csv=csv)
        d = dict(zip(cat.sn, cat.subtype))
        assert d["FAKEIBC"] == "Ic" and d["FAKE1"] == "Ia"
        assert pipeline78.catalog.REF_BAND["Ibc"] == "R_rest"
        csv.write_text("sn,subtype,source\nOTRA,Ic,test\n")
        try:
            pipeline78.catalog.build_catalog(pathlib.Path(td), subtypes_csv=csv)
        except ValueError as e:
            assert "FAKEIBC" in str(e)
        else:
            raise AssertionError("no levanto ValueError")

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
