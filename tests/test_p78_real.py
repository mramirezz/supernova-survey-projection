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


def _escenario(tmp_path):
    """3 SNe: ZTF20aaa (Ia con basura alrededor), ZTF20bbb (sin TNS, vieja), ZTF20ccc (primera_det_tardia).
    Devuelve los kwargs de ztf_real_to_parquet (sin exclusiones manuales), t de descubrimiento y las filas de la SN."""
    from astropy.time import Time
    phot, oc = tmp_path / "phot", tmp_path / "oc"
    (oc / "data").mkdir(parents=True)
    t = float(Time("2020-03-01 12:00:00", format="iso", scale="utc").mjd)
    sn = [(t + d, b, 18.0 + abs(d - 15) * 0.03, "F") for d in range(0, 91, 3) for b in ("g", "r")]
    basura = [(t - 300, "r", 18.9, "F"), (t - 100, "g", 20.5, "T"), (t + 700, "r", 18.9, "F"), (t + 250, "r", 19.9, "F")]
    _dat(phot / "SN Ia" / "ZTF20aaa_photometry.dat", "ZTF20aaa", sn + basura)
    _dat(phot / "SN II" / "ZTF20bbb_photometry.dat", "ZTF20bbb",                 # sin TNS: t_disc = primera deteccion
         [(t + 10 + d, "r", 18.5, "F") for d in range(0, 30, 6)] + [(t, "r", 20.0, "T")])
    _dat(phot / "SN II" / "ZTF20ccc_photometry.dat", "ZTF20ccc",                 # descubierta antes de las alertas
         [(t + 100 + 3 * k, "r", 19.3, "F") for k in range(10)])
    pd.DataFrame({"name": ["2020aaa", "2020ccc"], "discoverydate": ["2020-03-01 12:00:00"] * 2,
                  "internal_names": ["ATLAS20x, ZTF20aaa", "ZTF20ccc"]}).to_csv(tmp_path / "tns.csv", index=False)
    pd.DataFrame({"oid": ["ZTF20aaa", "ZTF20ccc"], "clase": ["Ia", "II"], "subtipo": ["Ia", "II"], "z": [0.05, 0.02],
                  "split": ["val", "final"]}).to_csv(tmp_path / "holdout.csv", index=False)
    pd.DataFrame({"sn_name": ["ZTF20bbb"], "label": ["II"], "z": [0.02]}).to_parquet(oc / "data/real_val.parquet")
    pd.DataFrame({"sn_name": pd.Series([], dtype=str), "label": pd.Series([], dtype=str),
                  "z": pd.Series([], dtype=float)}).to_parquet(oc / "data/real_final.parquet")
    (tmp_path / "excluir_vacio.csv").write_text("oid,motivo\n")
    kw = dict(holdout=tmp_path / "holdout.csv", oc=oc, phot=phot, tns=tmp_path / "tns.csv",
              excluir_manual=tmp_path / "excluir_vacio.csv")
    return kw, t, sn


def test_real_to_parquet_limpia_desde_tns(tmp_path):
    from pipeline78.real_to_parquet import ztf_real_to_parquet
    kw, t, sn = _escenario(tmp_path)
    out = tmp_path / "out"
    m = ztf_real_to_parquet(out, **kw)
    ia = pd.read_parquet(out / "Ia.parquet")
    assert ia.mjd.between(t - 50, t + 400).all() and len(ia) == len(sn)    # ventana desde t_disc y el aislado de +250
    meta = pd.read_csv(out / "meta_real_ztf.csv").set_index("oid")
    nuevas = ["t_disc", "n_filas_antes", "n_filas_despues", "n_grupos_eliminados", "span_det_antes", "span_det_despues"]
    nuevas += ["n_grupos_antes_pico", "n_grupos_chicos", "n_grupos_no_bajan", "dt_primera_det", "primera_det_tardia", "excluir"]
    assert set(nuevas) <= set(meta.columns) and "n_det_r" not in meta.columns
    a = meta.loc["ZTF20aaa"]
    assert abs(a.t_disc - t) < 1e-6 and a.n_filas_antes == len(sn) + 4 and a.n_filas_despues == len(sn)
    assert a.n_grupos_eliminados == 1 and a.span_det_antes == 1000.0 and a.span_det_despues == 90.0
    b = meta.loc["ZTF20bbb"]
    assert b.t_disc == t + 10 and b.n_filas_despues == 6 and b.origen == "viejas"
    assert not a.excluir and not b.excluir
    c = meta.loc["ZTF20ccc"]                                                   # marcada, pero sigue en el parquet
    assert c.primera_det_tardia and c.excluir and c.dt_primera_det == 100.0
    assert (pd.read_parquet(out / "II.parquet").oid == "ZTF20ccc").sum() == 10
    ex = pd.read_csv(out / "excluir_reales.csv")
    assert list(ex.oid) == ["ZTF20ccc"] and list(ex.motivo) == ["primera_det_tardia"]
    rev = pd.read_csv(out / "revisar_reales.csv")
    assert list(rev.oid) == ["ZTF20bbb"] and list(rev.motivo) == ["<7 det r"]  # 5 detecciones en r
    assert len(m) == 3


def test_exclusiones_manuales(tmp_path):
    """data/excluir_manual_reales.csv: excluir=True y su motivo en meta, sin tocar los parquets (decide el consumidor)."""
    from pipeline78.real_to_parquet import ztf_real_to_parquet
    kw, t, sn = _escenario(tmp_path)
    a = ztf_real_to_parquet(tmp_path / "a", **kw)
    pd.DataFrame({"oid": ["ZTF20aaa", "ZTF20ccc", "ZTF99zzz"],
                  "motivo": ["dudosa de prueba", "manual c", "no esta en las listas"]}).to_csv(tmp_path / "ex.csv", index=False)
    b = ztf_real_to_parquet(tmp_path / "b", **dict(kw, excluir_manual=tmp_path / "ex.csv"))
    for label in ("Ia", "II"):                                                   # parquets identicos
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / f"{label}.parquet"),
                                      pd.read_parquet(tmp_path / "b" / f"{label}.parquet"))
    assert len(pd.read_parquet(tmp_path / "b" / "Ia.parquet")) == len(sn)
    meta = pd.read_csv(tmp_path / "b" / "meta_real_ztf.csv").set_index("oid")
    assert meta.excluir.to_dict() == {"ZTF20aaa": True, "ZTF20ccc": True, "ZTF20bbb": False}
    assert meta.loc["ZTF20aaa", "motivo"] == "dudosa de prueba" and not meta.loc["ZTF20aaa", "primera_det_tardia"]
    assert meta.loc["ZTF20ccc", "motivo"] == "primera_det_tardia + manual c"
    assert pd.isna(meta.loc["ZTF20bbb", "motivo"])
    ex = pd.read_csv(tmp_path / "b" / "excluir_reales.csv")
    assert list(ex.oid) == ["ZTF20aaa", "ZTF20ccc"] and list(ex.motivo) == ["dudosa de prueba", "primera_det_tardia + manual c"]
    rev = pd.read_csv(tmp_path / "b" / "revisar_reales.csv")
    assert list(rev.oid) == ["ZTF20bbb"] and list(rev.motivo) == ["<7 det r"] and rev.motivo_excluir.isna().all()
    # sin exclusiones manuales solo queda la automatica
    ma = pd.read_csv(tmp_path / "a" / "meta_real_ztf.csv").set_index("oid")
    assert ma.excluir.to_dict() == {"ZTF20aaa": False, "ZTF20ccc": True, "ZTF20bbb": False}
    assert len(a) == len(b) == 3


def test_lista_versionada_de_exclusiones_manuales(tmp_path):
    from pipeline78.real_to_parquet import load_excluir_manual
    man = load_excluir_manual()
    assert sorted(man) == ["ZTF21abxlmuw", "ZTF23aamanim", "ZTF24abqndnz", "ZTF24abymeet"]
    assert all(v.startswith("dudosa revisada por Mauricio 2026-10-03") for v in man.values())
    (tmp_path / "dup.csv").write_text("oid,motivo\nZTFa,x\nZTFa,y\n")
    (tmp_path / "sin.csv").write_text("oid,motivo\nZTFa,\n")
    import pytest
    for f in ("dup.csv", "sin.csv"):
        with pytest.raises(ValueError):
            load_excluir_manual(tmp_path / f)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
