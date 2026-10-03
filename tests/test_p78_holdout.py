import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline78.holdout import build_holdout


def _world(tmp, n_ia=10):
    tmp = pathlib.Path(tmp)
    phot, oc, store = tmp / "phot", tmp / "oc", tmp / "store"
    (oc / "data").mkdir(parents=True); store.mkdir()
    rows, names = [], {}
    def add(folder, oid, iau, z=0.05):
        (phot / folder).mkdir(parents=True, exist_ok=True)
        (phot / folder / f"{oid}_photometry.dat").write_text("x")
        rows.append({"name": iau, "redshift": z, "type": folder, "internal_names": f"{oid}, ATLAS1"})
    for i in range(n_ia):
        add("SN Ia", f"ZTF20a{i:03d}", f"2020a{i:03d}")
    add("SN IIP", "ZTF21aaa", "2021aaa"); add("SN IIL", "ZTF21aab", "2021aab"); add("SN II", "ZTF21aac", "2021aac")
    add("SN Ic-BL", "ZTF21bbb", "2021bbb"); add("SN IIn", "ZTF21ccc", "2021ccc"); add("SN IIb", "ZTF21ddd", "2021ddd")
    add("SN Ibn", "ZTF21eee", "2021eee")                       # carpeta fuera del mapeo
    add("SN Ia", "ZTF21old", "2021old")                        # vieja
    add("SN II", "ZTF21tpl", "2021tpl")                        # plantilla por IAU
    add("SN II", "ZTF21zzz", "2021zzz", z=-1)                  # z invalido
    add("SN Ia", "ZTF21nan", "2021nan", z=float("nan"))
    (phot / "SN Ic").mkdir(); (phot / "SN Ic" / "ZTF21aac_photometry.dat").write_text("x")   # duplicado de SN II
    pd.DataFrame(rows).to_csv(tmp / "tns.csv", index=False)
    pd.DataFrame({"sn_name": ["ZTF21old"], "label": ["Ia"], "z": [0.1]}).to_parquet(oc / "data/real_val.parquet")
    pd.DataFrame({"sn_name": ["ZTFnada"], "label": ["Ia"], "z": [0.1]}).to_parquet(oc / "data/real_final.parquet")
    pd.DataFrame({"sn": ["SN2021tpl"]}).to_csv(store / "catalog.csv", index=False)
    return dict(phot=phot, tns=tmp / "tns.csv", oc=oc, store=store)


def test_mapeo_exclusiones_y_split():
    with tempfile.TemporaryDirectory() as t:
        res, ex = build_holdout(**_world(t))
        d = res.set_index("oid")
        assert d.loc["ZTF21aaa", "subtipo"] == "IIP" and d.loc["ZTF21aaa", "clase"] == "II"
        assert d.loc["ZTF21aab", "subtipo"] == "IIL"
        assert d.loc["ZTF21aac", "carpeta"] == "SN II"           # primera carpeta gana
        assert d.loc["ZTF21bbb", "clase"] == "Ibc" and d.loc["ZTF21bbb", "subtipo"] == "Ic-BL"
        assert d.loc["ZTF21ccc", "clase"] == "IIn" and d.loc["ZTF21ddd", "clase"] == "IIb"
        for bad in ("ZTF21eee", "ZTF21old", "ZTF21tpl", "ZTF21zzz", "ZTF21nan"):
            assert bad not in d.index, bad
        assert ex["viejas"] == ["ZTF21old"] and ex["plantilla"] == [("ZTF21tpl", "2021tpl")]
        assert len(ex["duplicado"]) == 1 and sorted(ex["z_invalido"]) == ["ZTF21nan", "ZTF21zzz"]
        ia = res[res.clase == "Ia"]
        assert len(ia) == 10 and (ia.split == "val").sum() == 5 and (ia.split == "final").sum() == 5
        ii = res[res.clase == "II"]                                # 3 -> 1 val, 2 final (impar al final)
        assert (ii.split == "val").sum() == 1 and (ii.split == "final").sum() == 2


def test_tope_reproducible():
    with tempfile.TemporaryDirectory() as t:
        w = _world(t, n_ia=40)
        a, _ = build_holdout(**w, cap=12); b, _ = build_holdout(**w, cap=12)
        assert len(a[a.clase == "Ia"]) == 12 and a.equals(b)
        c, _ = build_holdout(**w, cap=12, seed=1)
        assert set(a[a.clase == "Ia"].oid) != set(c[c.clase == "Ia"].oid)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
