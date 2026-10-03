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

def _dat(path, oid, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = [f"# SNNAME: {oid}"]
    for b in ("g", "r"):
        txt.append(f"# FILTER {b}")
        txt += [f"{m}\t{mag}\t{0.05 if u == 'F' else 'nan'}\t{u}" for m, f, mag, u in rows if f == b]
    path.write_text("\n".join(txt) + "\n")


def test_real_to_parquet_limpia_desde_tns(tmp_path):
    from astropy.time import Time
    from pipeline78.real_to_parquet import ztf_real_to_parquet
    phot, oc = tmp_path / "phot", tmp_path / "oc"
    (oc / "data").mkdir(parents=True)
    t = float(Time("2020-03-01 12:00:00", format="iso", scale="utc").mjd)
    sn = [(t + d, b, 18.0 + abs(d - 15) * 0.03, "F") for d in range(0, 91, 3) for b in ("g", "r")]
    basura = [(t - 300, "r", 18.9, "F"), (t - 100, "g", 20.5, "T"), (t + 700, "r", 18.9, "F"), (t + 250, "r", 19.9, "F")]
    _dat(phot / "SN Ia" / "ZTF20aaa_photometry.dat", "ZTF20aaa", sn + basura)
    _dat(phot / "SN II" / "ZTF20bbb_photometry.dat", "ZTF20bbb",                 # sin TNS: t_disc = primera deteccion
         [(t + 10 + d, "r", 18.5, "F") for d in range(0, 30, 6)] + [(t, "r", 20.0, "T")])
    pd.DataFrame({"name": ["2020aaa"], "discoverydate": ["2020-03-01 12:00:00"],
                  "internal_names": ["ATLAS20x, ZTF20aaa"]}).to_csv(tmp_path / "tns.csv", index=False)
    pd.DataFrame({"oid": ["ZTF20aaa"], "clase": ["Ia"], "subtipo": ["Ia"], "z": [0.05], "split": ["val"]}).to_csv(
        tmp_path / "holdout.csv", index=False)
    pd.DataFrame({"sn_name": ["ZTF20bbb"], "label": ["II"], "z": [0.02]}).to_parquet(oc / "data/real_val.parquet")
    pd.DataFrame({"sn_name": pd.Series([], dtype=str), "label": pd.Series([], dtype=str),
                  "z": pd.Series([], dtype=float)}).to_parquet(oc / "data/real_final.parquet")
    out = tmp_path / "out"
    m = ztf_real_to_parquet(out, holdout=tmp_path / "holdout.csv", oc=oc, phot=phot, tns=tmp_path / "tns.csv")
    ia = pd.read_parquet(out / "Ia.parquet")
    assert ia.mjd.between(t - 50, t + 400).all() and len(ia) == len(sn)    # ventana desde t_disc y el aislado de +250
    meta = pd.read_csv(out / "meta_real_ztf.csv").set_index("oid")
    nuevas = ["t_disc", "n_filas_antes", "n_filas_despues", "n_grupos_eliminados", "span_det_antes", "span_det_despues"]
    assert set(nuevas) <= set(meta.columns) and "n_det_r" not in meta.columns
    a = meta.loc["ZTF20aaa"]
    assert abs(a.t_disc - t) < 1e-6 and a.n_filas_antes == len(sn) + 4 and a.n_filas_despues == len(sn)
    assert a.n_grupos_eliminados == 1 and a.span_det_antes == 1000.0 and a.span_det_despues == 90.0
    b = meta.loc["ZTF20bbb"]
    assert b.t_disc == t + 10 and b.n_filas_despues == 6 and b.origen == "viejas"
    rev = pd.read_csv(out / "revisar_reales.csv")
    assert list(rev.oid) == ["ZTF20bbb"]                                       # 5 detecciones en r
    assert len(m) == 2


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
