# tests/test_p78_obslog.py
"""Log con el diffmaglim real de ALeRCE (obslog_alerce), emparejamiento con alertas (alerce.match) y el log_path de
las configs de la T9."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import pytest
from pipeline78 import alerce, obslog_alerce as ob
from pipeline78.runcfg import RUNS_CFG, LOG_ALERCE
from pipeline78 import runcfg


def _alerts(rows):
    return pd.DataFrame(rows, columns=["oid", "mjd", "fid", "magpsf", "diffmaglim", "isdiffpos"])


def test_match_tolerancias_banda_y_duplicados():
    """Misma oid y banda, dentro de las tolerancias; dos alertas de la misma exposicion (campos que se solapan) se
    separan por la magnitud; sin candidata dentro de la tolerancia da -1."""
    al = _alerts([("A", 100.0, 1, 18.00, 20.0, 1), ("A", 100.0, 1, 18.50, 20.5, 1), ("A", 100.0, 2, 18.0, 21.0, 1),
                  ("A", 101.0, 1, 18.0, 19.0, -1), ("B", 100.0, 1, 18.0, 22.0, 1)])
    rows = pd.DataFrame({"oid": ["A", "A", "A", "A", "A", "C", "A"], "fid": [1, 1, 2, 1, 1, 1, 1],
                         "mjd": [100.0 + 4e-6, 100.0, 100.0, 101.0, 100.0 + 5e-5, 100.0, 101.0],
                         "magpsf": [18.4996, 18.0004, 18.0, 18.0, 18.0, 18.0, 18.01]})
    j = alerce.match(rows, al, 1e-5, 1e-3)
    assert list(j) == [1, 0, 2, 3, -1, -1, -1]
    assert list(alerce.match(rows, al, 1e-5)) == [0, 0, 2, 3, -1, -1, 3]     # sin magnitud: la primera del mjd
    dos = _alerts([("A", 100.0, 1, 18.0000, 20.0, 1), ("A", 100.0, 1, 18.0008, 20.5, 1)])   # las dos dentro de tol_mag
    q = pd.DataFrame({"oid": ["A", "A"], "fid": [1, 1], "mjd": [100.0, 100.0], "magpsf": [18.0007, 18.0001]})
    assert list(alerce.match(q, dos, 1e-5, 1e-3)) == [1, 0]                 # gana la de magnitud mas cercana


def _log_rows():
    """Campo F: dos detecciones (una con alerta negativa), una sin alerta y no detecciones; un dia con dos filas."""
    return pd.DataFrame({
        "oid": ["F"] * 6, "mjd": [10.1, 10.2, 11.1, 12.1, 13.1, 13.3], "fid": [1, 1, 1, 2, 2, 2],
        "magpsf": [18.0, np.nan, 18.2, 17.9, np.nan, 18.0],
        "diffmaglim": [18.5, 20.0, 18.7, 18.4, 19.0, 19.5],
        "diffmaglim_estimated": [True, False, True, True, False, True],
        "is_detection": [True, False, True, True, False, True]})


def test_with_alerce_reemplaza_solo_las_detecciones_emparejadas():
    al = _alerts([("F", 10.1, 1, 18.0, 20.6, 1), ("F", 12.1, 2, 17.9, 20.9, -1), ("F", 13.3, 2, 18.0, 19.2, 1)])
    r = ob.with_alerce(_log_rows(), al)
    assert np.allclose(r.maglim, [20.6, 20.0, 18.7, 20.9, 19.0, 19.2])        # la negativa tambien da el limite
    assert list(r.estimado) == [False, False, True, False, False, False]
    assert list(r.ialerta) == [0, -1, -1, 1, -1, 2]
    new, old = ob.best(r), ob.best(r, "diffmaglim").drop(columns="estimado")
    assert list(old.columns) == ["field", "mjd", "band", "maglim"]
    assert list(new.columns) == ["field", "mjd", "band", "maglim", "estimado"]
    # dia 10 (g): viejo max(18.5, 20.0) = 20.0; nuevo max(20.6, 20.0) = 20.6 de la fila 10.1
    assert new.loc[new.mjd == 10.1, "maglim"].item() == 20.6 and old.loc[old.mjd == 10.2, "maglim"].item() == 20.0
    # dia 13 (r): viejo 19.5 (13.3, estimado), nuevo max(19.0, 19.2) = 19.2
    assert new.loc[new.mjd == 13.3, "maglim"].item() == 19.2
    c = ob.compare(old, new).set_index(["band", "day"])
    assert np.allclose(c.delta.to_numpy(), [0.6, 0.0, 2.5, -0.3])
    assert list(c.estimado) == [False, True, False, False]
    camp = ob.resumen_campos(r, ob.compare(old, new))
    assert camp.loc[0, ["n_epocas", "n_epocas_cambian", "n_det", "n_det_sin_alerta"]].tolist() == [4, 3, 4, 1]
    assert not camp.sin_alerce.item()
    sin = ob.resumen_campos(ob.with_alerce(_log_rows(), al.iloc[:0]), ob.compare(old, old.assign(estimado=False)))
    assert sin.sin_alerce.item()                                             # ninguna alerta: conserva y se marca


def test_t9_usa_el_log_de_alerce_y_ztf_v78_no():
    for k in ("ztf_v78", "ztf_v78_texp", "ztf_v78_tail", "ztf_v78_fireball"):
        assert "log_path" not in RUNS_CFG[k] and runcfg.log_path(RUNS_CFG[k]).endswith("ztf_obslog_best.parquet"), k
    t9 = [k for k in RUNS_CFG if k.startswith("ztf_v78_t9")]
    assert len(t9) >= 20
    for k in t9:
        assert runcfg.log_path(RUNS_CFG[k]) == LOG_ALERCE, k
    assert LOG_ALERCE.endswith("thesis_runs/obslog/ztf_obslog_alerce.parquet")


def test_log_construido():
    """Con los datos: mismas epocas que ztf_obslog_best en los 1000 campos, esquema de load_log y la columna estimado."""
    p = pathlib.Path(LOG_ALERCE)
    if not p.exists() or not ob.OLD.exists() or not ob.FIELDS.exists():
        pytest.skip("sin el log de ALeRCE construido")
    new = pd.read_parquet(p)
    f = ob.read_fields()
    assert list(new.columns) == ["field", "mjd", "band", "maglim", "estimado"] and set(new.field) == set(f)
    old = pd.read_parquet(ob.OLD, filters=[("field", "in", f)])
    c = ob.compare(old, new)
    assert len(c) == len(new) == len(old) and np.isfinite(new.maglim).all()
    from pipeline78.survey import load_log
    lg = load_log(p, f[:3])
    assert set(lg) == set(f[:3])
