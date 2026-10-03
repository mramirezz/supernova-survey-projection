# tests/test_p78_lcclean.py
"""clean_lc: ventana desde t_ref, grupos aislados fuera, segundo grupo grande y grupo del pico se conservan."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pipeline78.lcclean import clean_lc

T = 59000.0


def _lc(rows):
    """rows: (mjd, filter, mag, upperlimit)."""
    m, f, g, u = zip(*rows)
    return pd.DataFrame({"mjd": np.array(m, float), "filter": f, "magnitud_proyectada": np.array(g, float),
                         "magerr": np.where(np.array(u) == "F", 0.05, np.nan), "upperlimit": u})


def _world():
    main = [(T + d, b, 18.0 + abs(d - 20) * 0.02, "F") for d in range(0, 101, 5) for b in ("g", "r")]   # pico en +20
    fuera = [(T - 100, "r", 20.5, "T"), (T - 200, "r", 19.0, "F"), (T + 500, "g", 19.5, "F")]
    dentro = [(T - 40, "r", 20.5, "T"), (T - 40, "g", 20.6, "T")]
    aislado = [(T + 250, "r", 19.8, "F"), (T + 252, "g", 19.9, "F"), (T + 251, "r", 20.4, "T")]
    iin = [(T + d, "r", 19.5, "F") for d in range(330, 371, 10)]            # temporada siguiente, 5 puntos
    return _lc(main + fuera + dentro + aislado + iin)


def test_ventana_aislado_y_segundo_grupo():
    df = _world()
    out, s = clean_lc(df, T)
    assert out.mjd.between(T - 50, T + 400).all()
    assert not out.mjd.isin([T - 100, T - 200, T + 500]).any()             # ventana: det y UL, todas las bandas
    assert (out.mjd == T - 40).sum() == 2                                   # UL dentro de la ventana se quedan
    det = out[out.upperlimit == "F"]
    assert not det.mjd.isin([T + 250, T + 252]).any()                      # grupo aislado de 2 a +250 d
    assert ((out.mjd == T + 251) & (out.upperlimit == "T")).any()           # sus UL no se tocan
    assert (det.mjd >= T + 330).sum() == 5                                  # segundo grupo grande (IIn) se queda
    assert (det.mjd <= T + 100).sum() == 42                                 # grupo del pico entero
    assert s == dict(n_filas_antes=len(df), n_filas_despues=len(out), n_grupos_eliminados=1,
                     span_det_antes=700.0, span_det_despues=370.0)
    assert len(df) - len(out) == 3 + 2


def test_grupo_del_pico_se_conserva():
    pico = [(T, "r", 17.0, "F"), (T + 2, "r", 17.1, "F")]                  # 2 puntos, el maximo en r
    resto = [(T + 100 + d, "r", 19.0, "F") for d in range(0, 25, 5)]
    out, s = clean_lc(_lc(pico + resto), T)
    assert len(out) == 7 and s["n_grupos_eliminados"] == 0
    # sin r manda g; con r manda r aunque un g sea mas brillante
    g_pico = [(T, "g", 17.0, "F"), (T + 2, "g", 17.1, "F")]
    resto_g = [(T + 100 + d, "g", 19.0, "F") for d in range(0, 25, 5)]
    out, s = clean_lc(_lc(g_pico + resto_g), T)
    assert len(out) == 7 and s["n_grupos_eliminados"] == 0
    out, s = clean_lc(_lc(g_pico + resto), T)
    assert len(out) == 5 and s["n_grupos_eliminados"] == 1 and (out["filter"] == "r").all()


def test_sin_detecciones():
    out, s = clean_lc(_lc([(T, "r", 20.0, "T"), (T + 500, "r", 20.0, "T")]), T)
    assert len(out) == 1 and s["n_grupos_eliminados"] == 0 and s["span_det_despues"] == 0.0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
