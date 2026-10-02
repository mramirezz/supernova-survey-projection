"""Logs de observacion -> esquema comun (field, mjd, band, maglim). Una epoca por dia, campo y banda."""
import numpy as np
import pandas as pd
from pipeline78.paths import SUDARE_DIR

SUDARE_SEASON_SPLIT = [(56300.0, "cosmos1"), (56900.0, "cosmos2"), (1e9, "cosmos3")]  # MJD de corte


def _best_per_day(df):
    df = df.dropna(subset=["band", "maglim"]).copy()
    df["mjd_day"] = np.floor(df["mjd"]).astype("int64")
    idx = df.groupby(["field", "mjd_day", "band"])["maglim"].idxmax()
    return df.loc[idx].drop(columns="mjd_day").sort_values(["field", "mjd"]).reset_index(drop=True)


def build_ztf_log(csv_path, out):
    df = pd.read_csv(csv_path, usecols=["oid", "mjd", "fid", "diffmaglim"])
    df = df.rename(columns={"oid": "field", "diffmaglim": "maglim"})
    df["band"] = df["fid"].map({1: "g", 2: "r", 3: "i"})
    best = _best_per_day(df[["field", "mjd", "band", "maglim"]])
    best.to_parquet(out, index=False)
    return len(best)


def build_sudare_log(out, ref_epochs=None):
    """obslog_I + obslog_II (las lineas con '#' quedan fuera). cosmos se parte en sus tres temporadas.
    ref_epochs: set de (field, band, mjd_day) que son imagen template, se marcan is_ref."""
    rows = []
    for fname in ("obslog_I.txt", "obslog_II.txt"):
        for line in open(SUDARE_DIR / fname):
            p = line.split()
            if len(p) < 6 or line.lstrip().startswith("#") or p[0] == "field":
                continue
            field = p[0].split("_")[0]
            mjd = float(p[3])
            if field == "cosmos":
                field = next(name for cut, name in SUDARE_SEASON_SPLIT if mjd < cut)
            rows.append((field, mjd, p[1], float(p[5]), float(p[4])))
    df = pd.DataFrame(rows, columns=["field", "mjd", "band", "maglim", "seeing"]).sort_values(["field", "mjd"])
    ref = ref_epochs or set()
    df["is_ref"] = [(f, b, int(np.floor(m))) in ref for f, b, m in zip(df.field, df.band, df.mjd)]
    df.reset_index(drop=True).to_parquet(out, index=False)
    return df


def load_log(path, fields=None):
    df = pd.read_parquet(path)
    if "is_ref" in df.columns:
        df = df[~df["is_ref"]]
    if fields is not None:
        df = df[df["field"].isin(set(fields))]
    out = {}
    for field, g in df.groupby("field", sort=False):
        out[field] = {b: (gb["mjd"].to_numpy(float), gb["maglim"].to_numpy(float))
                      for b, gb in g.sort_values("mjd").groupby("band")}
    return out
