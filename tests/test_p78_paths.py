import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline78 import paths

def test_paths_exist():
    for p in (paths.REPO, paths.DATA, paths.LIB, paths.SUDARE_DIR, paths.LEGACY_RESP, paths.ZLF, paths.OC):
        assert p.exists(), p
    assert (paths.LIB / "MANIFIESTO.csv").exists()

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
