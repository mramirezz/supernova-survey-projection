# tests/test_p78_sampling.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import math
import numpy as np
from pipeline78.sampling import sample_ebv_host, sample_mpeak, zgrid_cdf, z_sampler
from config import EXTINCTION_CONFIG, PHILLIPS_CONFIG

def test_ebv_cdf_matches_mixture():
    # P(E<t) analitica de la mezcla: f0*erf(t/(s0*sqrt2)) + (1-f0)*(1-exp(-t*Rv/tau)), t=0.05
    p = EXTINCTION_CONFIG["SNII"]
    t = 0.05
    expected = (p["frac_zero"] * math.erf(t / (p["sigma_zero"] * math.sqrt(2)))
                + (1 - p["frac_zero"]) * (1 - math.exp(-t * p["Rv"] / p["tau"])))
    rng = np.random.default_rng(1)
    e = np.array([sample_ebv_host(rng, "II")[0] for _ in range(20000)])
    assert abs((e < t).mean() - expected) < 0.02

def test_phillips_mean():
    rng = np.random.default_rng(2)
    m = np.array([sample_mpeak(rng, "Ia", PHILLIPS_CONFIG["dm15_ref"]) for _ in range(20000)])
    assert abs(m.mean() - PHILLIPS_CONFIG["M0"]) < 0.01

def test_same_seed_same_draws():
    a = sample_mpeak(np.random.default_rng([7, 1, 2]), "II"); b = sample_mpeak(np.random.default_rng([7, 1, 2]), "II")
    assert a == b

def test_subtype_ic_bl_mean():
    rng = np.random.default_rng(3)
    m = np.array([sample_mpeak(rng, "Ibc", subtype="Ic-BL") for _ in range(20000)])
    assert abs(m.mean() - (-19.0)) < 0.03

def test_unknown_subtype_falls_back_to_class():
    from config import LUMINOSITY_CONFIG
    rng = np.random.default_rng(4)
    m = np.array([sample_mpeak(rng, "Ibc", subtype="XYZ") for _ in range(20000)])
    assert abs(m.mean() - LUMINOSITY_CONFIG["M_peak"]["Ibc"]["mean"]) < 0.03

def test_volumetric_favours_high_z():
    z, c = zgrid_cdf(0.05, 1.0)
    assert np.interp(0.5, c, z) > 0.6

def test_fixed_z_sampler():
    assert z_sampler({"z_mode": "fixed", "z_fixed": 0.2})(np.random.default_rng(0), "Ia") == 0.2

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
