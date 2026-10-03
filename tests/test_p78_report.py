# tests/test_p78_report.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pipeline78.pilot_report import peak_mag_8h, wmedian

def _df(mjd, mag):
    return pd.DataFrame(dict(MJD=mjd, MAG=mag, MAGERR=0.05))

def test_peak_mag_8h_groups_and_requires_seven():
    # 7 grupos: el 1o son dos puntos dentro de 8 h (17.0 y 19.0, mediana en flujo entre ambos), el resto separados
    mjd = [0.0, 0.1] + [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    mag = [17.0, 19.0] + [18.5, 18.2, 18.0, 18.3, 18.6, 19.0]
    f = np.median([10 ** (-17.0 / 2.5), 10 ** (-19.0 / 2.5)])
    expect = -2.5 * np.log10(f)                    # ~17.6: mas debil que el punto aislado 17.0
    got = peak_mag_8h(_df(mjd, mag))
    assert abs(got - min(expect, 18.0)) < 1e-9, got
    # sin agrupar el minimo seria 17.0: confirma que agrupo
    assert got > 17.0
    # menos de 7 puntos tras agrupar -> NaN
    assert np.isnan(peak_mag_8h(_df(mjd[:6], mag[:6])))
    assert np.isnan(peak_mag_8h(_df([], [])))

def test_wmedian():
    assert wmedian([1, 2, 3]) == 2
    assert wmedian([1, 2, 3], [10, 1, 1]) == 1

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
