import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline78.bands import C_AA, synphot, survey_bands, legacy_band, rest_bands

def test_flat_ab_spectrum_is_zero_mag():
    w = np.arange(2500.0, 11000.0, 1.0)
    f = (3631e-23 * C_AA / w**2)[None, :]
    for b in survey_bands("ZTF") + survey_bands("SUDARE"):
        F, cov = synphot(w, f, b)
        assert cov > 0.999, b.name
        assert abs(-2.5 * np.log10(F[0] / b.f0)) < 1e-3, b.name

def test_coverage_drops_when_blue_edge_missing():
    g = survey_bands("ZTF", ("g",))[0]
    w = np.arange(4600.0, 9000.0, 1.0)
    _, cov = synphot(w, np.ones((1, w.size)), g)
    assert cov < 0.95

def test_legacy_band_keeps_historic_zero_point():
    from core.utils import cter
    assert legacy_band("r").f0 == cter

def test_rest_bands_present():
    rb = rest_bands()
    assert set(rb) == {"B_rest", "V_rest", "R_rest", "r_rest"}

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
