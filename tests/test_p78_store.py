import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline78.store import parse_dat, save_template, load_template, stable_md5
from tests.p78_fakes import write_dat

def test_parse_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.dat"
        w = np.arange(3005.0, 3010.0)
        write_dat(p, [55001.0, 55000.0], w, [w * 2, w * 1])
        t, ww, f = parse_dat(p)
        assert list(t) == [55000.0, 55001.0]          # ordena por tiempo
        assert np.allclose(f[0], w) and np.allclose(f[1], 2 * w)
        save_template(pathlib.Path(td) / "s", t, ww, f, dict(sn="x"))
        tpl = load_template(pathlib.Path(td) / "s")
        assert tpl["flux"].dtype == np.float32 and tpl["flux"].shape == (2, 5)

def test_parse_rejects_bad_grid():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.dat"
        with open(p, "w") as fh:
            fh.write("# time: 1\n3005 1\n3006 1\n# time: 2\n3005 1\n3007 1\n")
        try:
            parse_dat(p); raise AssertionError("debio fallar")
        except ValueError as e:
            assert "grilla" in str(e)

def test_parse_rejects_nan():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.dat"
        with open(p, "w") as fh:
            fh.write("# time: 1\n3005 nan\n3006 1\n")
        try:
            parse_dat(p); raise AssertionError("debio fallar")
        except ValueError as e:
            assert "no finito" in str(e)

def test_stable_md5_detects_change():
    vals = iter(["aaa", "bbb"])
    try:
        stable_md5("ignorado", reader=lambda p: next(vals)); raise AssertionError("debio fallar")
    except IOError:
        pass

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
