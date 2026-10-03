# tests/test_p78_sampling.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import math
import numpy as np
from pipeline78.sampling import sample_ebv_host, sample_mpeak, zgrid_cdf, z_sampler, z_volume_weight
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

_UW = dict(z_mode="uniform_weighted", zmin=0.005, zmax_by_class={"Ia": 0.25, "II": 0.15})

def _uw_draws(cls, n=20000, seed=11):
    rng = np.random.default_rng(seed); f = z_sampler(_UW)
    z = np.array([f(rng, cls) for _ in range(n)])
    return z, z_volume_weight(z, _UW["zmin"], _UW["zmax_by_class"][cls]) * (_UW["zmax_by_class"][cls] - _UW["zmin"])

def test_uniform_weighted_mean_weight_is_one():
    for cls in ("Ia", "II"):
        z, w = _uw_draws(cls)
        assert z.min() >= _UW["zmin"] and z.max() <= _UW["zmax_by_class"][cls]
        assert abs(w.mean() - 1.0) < 0.02, w.mean()

def test_uniform_weighted_reproduces_volumetric_cdf():
    z, w = _uw_draws("Ia")
    i = np.argsort(z); ecdf = np.cumsum(w[i]) / w.sum()
    g, c = zgrid_cdf(_UW["zmin"], _UW["zmax_by_class"]["Ia"])
    assert np.abs(ecdf - np.interp(z[i], g, c)).max() < 0.02

def test_volume_weight_cache_equals_direct():
    from pipeline78.sampling import COSMO, _vol_norm
    g = np.linspace(0.005, 0.25, 4000); dv = COSMO.differential_comoving_volume(g).value
    direct = float(np.sum(0.5 * (dv[1:] + dv[:-1]) * np.diff(g)))
    assert _vol_norm(0.005, 0.25) == direct and _vol_norm(0.005, 0.25) == direct
    z = np.array([0.01, 0.1])
    assert np.array_equal(z_volume_weight(z, 0.005, 0.25), COSMO.differential_comoving_volume(z).value / direct)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
