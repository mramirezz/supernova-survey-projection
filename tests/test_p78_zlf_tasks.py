# tests/test_p78_zlf_tasks.py
"""Indice de tareas de run_parquet.py (ZLF parquet_reader): orden determinista (no depende del orden en que terminan
los workers), cache reusable aunque haya parquets vacios, y --n_test con --seed fija elige las mismas tareas."""
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pipeline78.paths import ZLF
sys.path.append(str(ZLF))          # al final: ZLF tiene su propio config.py
import parquet_reader as PR

COLS = ["oid", "part_index", "sn_type", "mjd", "filter", "magnitud_proyectada", "magerr", "upperlimit"]


def _dir(tmp_path):
    """5 parquets de una oid (varias part_index y tipos, escritos desordenados) y uno vacio con el esquema completo,
    como los campos sin simulaciones de un run real."""
    d = tmp_path / "pq"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i, o in enumerate(("ZTF19e", "ZTF18a", "ZTF20c", "ZTF18b", "ZTF19d")):
        rows = []
        for k in rng.permutation(6)[: 3 + i % 3]:
            for c in rng.permutation(["Ia", "II", "Ibc", "IIn"])[:3]:
                rows += [(o, int(k), c, 59000.0 + j, b, 18.0, 0.05, "F") for j in range(3) for b in ("g", "r")]
        pd.DataFrame(rows, columns=COLS).sample(frac=1.0, random_state=i).to_parquet(d / f"{o}__00000.parquet",
                                                                                      index=False)
    pd.DataFrame({c: pd.Series([], dtype=float if c in ("mjd", "magnitud_proyectada", "magerr") else
                               (int if c == "part_index" else str)) for c in COLS}).to_parquet(
        d / "ZTF18zzz__00000.parquet", index=False)
    return d


def test_enumeracion_ordenada_y_estable(tmp_path):
    d = _dir(tmp_path)
    a = PR.enumerate_tasks(d, use_cache=False, workers=2)          # as_completed: orden de llegada arbitrario
    b = PR.enumerate_tasks(d, use_cache=False, workers=1)
    pd.testing.assert_frame_equal(a, b)
    assert a.index.tolist() == list(range(len(a)))
    assert a.equals(a.sort_values(["parquet_path", "oid", "part_index", "sn_type"]).reset_index(drop=True))
    assert str(d / "ZTF18zzz__00000.parquet") not in set(a.parquet_path)   # el vacio no aporta tareas
    assert not a.duplicated(["parquet_path", "oid", "part_index", "sn_type"]).any() and a.oid.nunique() == 5
    s1, s2 = PR.stratified_sample(a, 10, seed=7), PR.stratified_sample(b, 10, seed=7)
    pd.testing.assert_frame_equal(s1, s2)


def test_cache_valido_con_parquet_vacio(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    fresh = PR.enumerate_tasks(d, workers=2)                        # escribe el cache
    t = os.path.getmtime(d / "_tasks_index.parquet")

    def no_leer(*a, **k):
        raise AssertionError("se re-enumero: el cache no se reuso")
    monkeypatch.setattr(PR, "_read_one_parquet_index", no_leer)
    cached = PR.enumerate_tasks(d, workers=1)                       # el parquet vacio no invalida el cache
    pd.testing.assert_frame_equal(cached, fresh)
    assert os.path.getmtime(d / "_tasks_index.parquet") == t
    pd.testing.assert_frame_equal(PR.stratified_sample(fresh, 10, seed=42), PR.stratified_sample(cached, 10, seed=42))
    # un cache v2 escrito desordenado (enumeraciones previas al fix) se devuelve en el orden canonico
    fresh.iloc[::-1].assign(index_version=PR._INDEX_VERSION).to_parquet(d / "_tasks_index.parquet", index=False)
    pd.testing.assert_frame_equal(PR.enumerate_tasks(d, workers=1), fresh)
    # un parquet nuevo CON filas si invalida el cache (mtime viejo a proposito: lo detecta la comparacion de rutas)
    monkeypatch.undo()
    f = d / "ZTF21x__00000.parquet"
    pd.read_parquet(next(d.glob("ZTF18a__*.parquet"))).assign(oid="ZTF21x").to_parquet(f, index=False)
    os.utime(f, (t - 100, t - 100))
    nuevo = PR.enumerate_tasks(d, workers=1)
    assert "ZTF21x" in set(nuevo.oid) and len(nuevo) > len(fresh)


if __name__ == "__main__":
    print("usar pytest (fixture tmp_path)")
