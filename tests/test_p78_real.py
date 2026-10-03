import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline78.real_to_parquet import photometry_rows

def test_photometry_rows_schema():
    fd = {"g": pd.DataFrame({"MJD": [1.0, 2.0], "MAG": [19.0, 20.5], "MAGERR": [0.1, float("nan")], "Upperlimit": [False, True]})}
    r = photometry_rows("ZTFx", "Ia", fd)
    assert list(r.upperlimit) == ["F", "T"] and set(r["filter"]) == {"g"}
    assert {"oid", "part_index", "sn_type", "mjd", "magnitud_proyectada", "magerr"} <= set(r.columns)

def test_upper_limit_error_nan_and_bands():
    fd = {"g": pd.DataFrame({"MJD": [1.0], "MAG": [20.0], "MAGERR": [0.3], "Upperlimit": [True]}),
          "i": pd.DataFrame({"MJD": [1.0], "MAG": [20.0], "MAGERR": [0.3], "Upperlimit": [False]})}
    r = photometry_rows("ZTFx", "II", fd)
    assert set(r["filter"]) == {"g"} and r.magerr.isna().all() and r.magnitud_proyectada.iloc[0] == 20.0

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
