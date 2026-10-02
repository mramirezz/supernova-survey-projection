# tests/test_p78_survey.py
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline78.survey import build_ztf_log, build_sudare_log, load_log

def test_ztf_best_per_day():
    with tempfile.TemporaryDirectory() as td:
        csv = pathlib.Path(td) / "log.csv"
        pd.DataFrame({"oid": ["A", "A", "A", "B"], "mjd": [100.1, 100.7, 101.2, 100.1], "fid": [2, 2, 1, 2],
                      "diffmaglim": [20.0, 20.5, 19.0, 21.0]}).to_csv(csv, index=False)
        out = pathlib.Path(td) / "best.parquet"
        build_ztf_log(csv, out)
        log = load_log(out, ["A"])
        assert list(log) == ["A"]
        assert list(log["A"]["r"][1]) == [20.5] and list(log["A"]["g"][0]) == [101.2]

def test_sudare_counts_match_paper_table1():
    with tempfile.TemporaryDirectory() as td:
        df = build_sudare_log(pathlib.Path(td) / "s.parquet")
        n = df.groupby(["field", "band"]).size().to_dict()
        assert (n[("cdfs1", "r")], n[("cdfs1", "g")], n[("cdfs1", "i")]) == (29, 7, 11)
        assert (n[("cdfs2", "r")], n[("cdfs2", "g")], n[("cdfs2", "i")]) == (23, 6, 4)
        assert (n[("cosmos1", "r")], n[("cosmos1", "g")], n[("cosmos1", "i")]) == (28, 7, 7)
        assert n[("cdfs3", "r")] == 30 and n[("cdfs4", "r")] == 29
        assert n[("cosmos2", "r")] == 24 and n[("cosmos3", "r")] == 13

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
