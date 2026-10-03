# tests/test_p78_zlf_reader.py
"""Lector de run_parquet.py (ZLF parquet_reader): un parquet con varias SNe que comparten part_index no se mezcla,
y un cache de tareas sin version se regenera."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pipeline78.paths import ZLF
sys.path.append(str(ZLF))          # al final: ZLF tiene su propio config.py
import parquet_reader as PR


def _multi(path):
    """3 oid x 2 part_index x 2 tipos en un solo archivo; cada SN con su propio numero de puntos y offset de mjd."""
    rows = []
    for i, o in enumerate(("A", "B", "C")):
        for k in (0, 1):
            for j, c in enumerate(("Ia", "II")):
                n = 5 + i + 2 * k + 3 * j
                for b in ("g", "r"):
                    t = 59000.0 + 100 * i + 10 * k + 1000 * j + np.arange(n)
                    rows += [(o, k, c, ti, b, 18.0 + 0.1 * x, 0.05, "F" if x else "T") for x, ti in enumerate(t)]
    df = pd.DataFrame(rows, columns=["oid", "part_index", "sn_type", "mjd", "filter", "magnitud_proyectada", "magerr",
                                     "upperlimit"])
    df.to_parquet(path, index=False)
    return df


def test_multi_oid_tasks_get_only_their_rows(tmp_path):
    d = tmp_path / "pq"
    d.mkdir()
    df = _multi(d / "Ia.parquet")
    tasks = PR.enumerate_tasks(d, cache_path=tmp_path / "_idx.parquet", workers=1)
    assert len(tasks) == 12 and not tasks.duplicated(["parquet_path", "oid", "part_index", "sn_type"]).any()
    for o, k, c, p in tasks[["oid", "part_index", "sn_type", "parquet_path"]].itertuples(index=False):
        fd, name, typ = PR.parse_parquet_lightcurve(p, k, c, oid=o)
        assert name == f"{o}_{c}_p{k:02d}" and typ == c
        for b in ("g", "r"):
            own = df[(df.oid == o) & (df.part_index == k) & (df.sn_type == c) & (df["filter"] == b)]
            assert np.array_equal(fd[b].MJD.to_numpy(), own.mjd.to_numpy())
            assert fd[b].Upperlimit.to_numpy().tolist() == (own.upperlimit == "T").tolist()
    # sin oid (llamada vieja) el archivo de varias SNe se mezcla: por eso run_parquet pasa el oid de la tarea
    fd, _, _ = PR.parse_parquet_lightcurve(d / "Ia.parquet", 0, "Ia")
    assert len(fd["r"]) == len(df[(df.part_index == 0) & (df.sn_type == "Ia") & (df["filter"] == "r")])


def test_single_oid_file_same_with_or_without_oid(tmp_path):
    df = _multi(tmp_path / "x.parquet")
    one = df[df.oid == "B"]
    one.to_parquet(tmp_path / "B.parquet", index=False)
    for k in (0, 1):
        for c in ("Ia", "II"):
            a, b = PR.parse_parquet_lightcurve(tmp_path / "B.parquet", k, c), \
                PR.parse_parquet_lightcurve(tmp_path / "B.parquet", k, c, oid="B")
            assert a[1:] == b[1:] and a[0].keys() == b[0].keys()
            for f in a[0]:
                pd.testing.assert_frame_equal(a[0][f], b[0][f])


def test_stale_cache_without_version_is_rebuilt(tmp_path):
    d = tmp_path / "pq"
    d.mkdir()
    _multi(d / "Ia.parquet")
    good = PR.enumerate_tasks(d, use_cache=False, workers=1)
    # cache del formato anterior (sin index_version) y con una sola fila: no se puede reusar
    good.head(1).to_parquet(d / "_tasks_index.parquet", index=False)
    pd.testing.assert_frame_equal(PR.enumerate_tasks(d, workers=1), good)
    assert "index_version" in pd.read_parquet(d / "_tasks_index.parquet").columns
    pd.testing.assert_frame_equal(PR.enumerate_tasks(d, workers=1), good)      # el cache nuevo si se reusa


if __name__ == "__main__":
    print("usar pytest (fixture tmp_path)")
