# tests/test_p78_lcclean.py
"""clean_lc: ventana desde t_ref, grupo del pico con >= 3 det, fuera lo anterior al pico, despues solo grupos que
siguen bajando, bandera de primera deteccion tardia."""
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


def _sn(d0=0, d1=101, bands=("g", "r")):
    """SN con pico en +20 (18.0) que baja 0.02 mag/d: en +90..+100 r = 19.4-19.6 (mediana de las ultimas 3: 19.5)."""
    return [(T + d, b, 18.0 + abs(d - 20) * 0.02, "F") for d in range(d0, d1, 5) for b in bands]


def _grupo(d0, mag, n=5, b="r"):
    return [(T + d0 + 3 * k, b, mag, "F") for k in range(n)]


def test_ventana_aislado_y_temporada_siguiente():
    fuera = [(T - 100, "r", 20.5, "T"), (T - 200, "r", 19.0, "F"), (T + 500, "g", 19.5, "F")]
    dentro = [(T - 40, "r", 20.5, "T"), (T - 40, "g", 20.6, "T")]
    aislado = [(T + 250, "r", 19.8, "F"), (T + 252, "g", 19.9, "F"), (T + 251, "r", 20.4, "T")]
    df = _lc(_sn() + fuera + dentro + aislado + _grupo(330, 20.0))         # temporada siguiente, mas debil
    out, s = clean_lc(df, T)
    assert out.mjd.between(T - 50, T + 400).all()
    assert not out.mjd.isin([T - 100, T - 200, T + 500]).any()             # ventana: det y UL, todas las bandas
    assert (out.mjd == T - 40).sum() == 2                                   # UL dentro de la ventana se quedan
    det = out[out.upperlimit == "F"]
    assert not det.mjd.isin([T + 250, T + 252]).any()                      # grupo chico a +250 d
    assert ((out.mjd == T + 251) & (out.upperlimit == "T")).any()           # sus UL no se tocan
    assert (det.mjd >= T + 330).sum() == 5 and (det.mjd <= T + 100).sum() == 42
    assert len(df) - len(out) == 3 + 2
    assert {k: s[k] for k in ("n_grupos_eliminados", "n_grupos_antes_pico", "n_grupos_chicos", "n_grupos_no_bajan",
                              "span_det_antes", "span_det_despues", "dt_primera_det", "primera_det_tardia")} == dict(
        n_grupos_eliminados=1, n_grupos_antes_pico=0, n_grupos_chicos=1, n_grupos_no_bajan=0, span_det_antes=700.0,
        span_det_despues=342.0, dt_primera_det=0.0, primera_det_tardia=False)
    assert s["n_filas_antes"] == len(df) and s["n_filas_despues"] == len(out)


def test_pico_ignora_punto_suelto_brillante():
    """ZTF25aajuqtp: un punto a +214 d mas brillante que el pico de la SN no ancla el grupo del pico."""
    out, s = clean_lc(_lc(_sn(0, 41) + [(T + 214, "r", 15.4, "F")]), T)
    assert (out.mjd < T + 100).all() and len(out) == 18
    assert s["n_grupos_chicos"] == 1 and s["n_grupos_antes_pico"] == 0


def test_grupo_posterior_mas_brillante_sale_con_los_siguientes():
    """2018hrt: fuente plana mas brillante que el final de la SN. Sale, y tambien todo lo que viene despues."""
    out, s = clean_lc(_lc(_sn() + _grupo(200, 19.3) + _grupo(300, 20.5)), T)
    assert out.mjd.max() <= T + 100 and len(out) == 42
    assert s["n_grupos_no_bajan"] == 2 and s["n_grupos_eliminados"] == 2


def test_grupo_posterior_mas_debil_se_queda_banda_g():
    out, s = clean_lc(_lc(_sn() + _grupo(200, 19.6) + _grupo(300, 19.8)), T)
    assert len(out) == 52 and s["n_grupos_eliminados"] == 0                 # 19.6 >= 19.5 y 19.8 >= 19.6
    # sin r en el grupo posterior manda g; sin banda comun sale
    out, s = clean_lc(_lc(_sn() + _grupo(200, 19.3, b="g")), T)            # g de la SN en +90..+100: 19.5
    assert len(out) == 42 and s["n_grupos_no_bajan"] == 1
    out, s = clean_lc(_lc(_sn() + _grupo(200, 19.8, b="g")), T)
    assert len(out) == 47 and s["n_grupos_eliminados"] == 0
    out, s = clean_lc(_lc(_sn(bands=("r",)) + _grupo(200, 21.0, b="g")), T)
    assert len(out) == 21 and s["n_grupos_no_bajan"] == 1


def test_grupo_anterior_al_pico_sale():
    """Un grupo grande antes de un salto > 60 d y antes del pico es otra cosa, aunque tenga >= 3 detecciones (los dos
    empiezan antes de t_ref + 30 d: el pico decide)."""
    out, s = clean_lc(_lc(_grupo(-50, 20.5, n=3) + _sn(20, 121)), T)    # -50, -47, -44 y la SN desde +20: salto 64 d
    assert (out.mjd >= T + 20).all() and len(out) == 42
    assert s["n_grupos_antes_pico"] == 1 and s["n_grupos_eliminados"] == 1


def test_grupo_del_descubrimiento_ancla_el_pico():
    """ZTF18abskzjm: un grupo posterior de >= 3 puntos apenas mas brillante que el pico de la SN no desplaza al grupo del
    descubrimiento (la SN no sale como 'anterior al pico'). Despues sale por la regla 3 (no sigue bajando)."""
    out, s = clean_lc(_lc(_sn(0, 101) + _grupo(200, 17.9)), T)              # pico de la SN 18.0, grupo a +200 d en 17.9
    assert len(out) == 42 and out.mjd.max() <= T + 100
    assert s["n_grupos_antes_pico"] == 0 and s["n_grupos_no_bajan"] == 1 and not s["primera_det_tardia"]
    # sin grupo grande que empiece antes de t_ref + 30 d: vuelve a la regla del mas brillante
    out, s = clean_lc(_lc(_grupo(40, 20.5) + _grupo(200, 18.0)), T)
    assert (out.mjd >= T + 200).all() and len(out) == 5 and s["n_grupos_antes_pico"] == 1 and s["primera_det_tardia"]


def test_primera_det_tardia():
    _, s = clean_lc(_lc(_sn(40, 141)), T)
    assert s["dt_primera_det"] == 40.0 and s["primera_det_tardia"]
    _, s = clean_lc(_lc(_sn(30, 141)), T)
    assert s["dt_primera_det"] == 30.0 and not s["primera_det_tardia"]


def test_sin_detecciones():
    out, s = clean_lc(_lc([(T, "r", 20.0, "T"), (T + 500, "r", 20.0, "T")]), T)
    assert len(out) == 1 and s["n_grupos_eliminados"] == 0 and s["span_det_despues"] == 0.0
    assert np.isnan(s["dt_primera_det"]) and not s["primera_det_tardia"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
