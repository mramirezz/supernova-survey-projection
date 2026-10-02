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

def test_parse_with_different_edges():
    """Dos bloques con el mismo step pero bordes distintos: parsea al overlap con las flux correctas."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.dat"
        # Bloque 1: 3005-3010
        # Bloque 2: 3006-3011
        # Overlap: 3006-3010
        with open(p, "w") as fh:
            fh.write("# time: 55000\n")
            for w in [3005.0, 3006.0, 3007.0, 3008.0, 3009.0, 3010.0]:
                fh.write(f"{w} {w * 1.0}\n")
            fh.write("# time: 55001\n")
            for w in [3006.0, 3007.0, 3008.0, 3009.0, 3010.0, 3011.0]:
                fh.write(f"{w} {w * 2.0}\n")
        t, ww, f = parse_dat(p)
        # El overlap es 3006-3010, así que 5 puntos
        assert list(t) == [55000.0, 55001.0]
        assert np.allclose(ww, [3006.0, 3007.0, 3008.0, 3009.0, 3010.0])
        assert f.shape == (2, 5)
        assert np.allclose(f[0], [3006.0, 3007.0, 3008.0, 3009.0, 3010.0])
        assert np.allclose(f[1], [6012.0, 6014.0, 6016.0, 6018.0, 6020.0])

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
