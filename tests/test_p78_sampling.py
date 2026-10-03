# tests/test_p78_sampling.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import math
import numpy as np
from pipeline78.sampling import sample_ebv_host, sample_mpeak, zgrid_cdf, z_sampler, z_volume_weight
from config import EXTINCTION_CONFIG, PHILLIPS_CONFIG

def test_ebv_cdf_matches_mixture():
    # P(E<t) analitica de la mezcla: f0*erf(t/(s0*sqrt2)) + (1-f0)*(1-exp(-t*Rv/tau)), t=0.05
    p = EXTINCTION_CONFIG["SNII_v78"]
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
    assert abs(m.mean() - (-18.57)) < 0.03

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

def test_ebv_subtype_ic_mean_av():
    rng = np.random.default_rng(5)
    av = np.array([(lambda e, r: e * r)(*sample_ebv_host(rng, "Ibc", "Ic")) for _ in range(20000)])
    # El objetivo NO es el 0.628 sin tope: Av_max trunca la exponencial. E[min(Exp(tau),Amax)] = tau*(1-exp(-Amax/tau))
    p = EXTINCTION_CONFIG["SNIc"]
    expected = (1 - p["frac_zero"]) * p["tau"] * (1 - math.exp(-p["Av_max"] / p["tau"]))
    assert abs(av.mean() - expected) < 0.015, (av.mean(), expected)
    rng = np.random.default_rng(5)
    assert all(sample_ebv_host(rng, "Ibc", "Ic")[1] == 4.3 for _ in range(50))

def test_ebv_subtype_without_key_uses_class():
    a = [sample_ebv_host(np.random.default_rng(i), "II", None) for i in range(20)]
    b = [sample_ebv_host(np.random.default_rng(i), "II") for i in range(20)]
    assert a == b

def test_ebv_ii_iin_halfnormal_fixc():
    # Fix C: half-normal sigma 0.2 en E(B-V), nunca negativa, R_V 3.1
    for cls, st in (("II", "IIP"), ("II", "IIL"), ("IIn", "IIn")):
        rng = np.random.default_rng(11)
        d = [sample_ebv_host(rng, cls, st) for _ in range(20000)]
        e = np.array([x[0] for x in d])
        assert abs(e.mean() - 0.2 * math.sqrt(2 / math.pi)) < 0.004, (cls, e.mean())
        assert (e >= 0).all() and all(x[1] == 3.1 for x in d)

def test_fixb_config_values():
    from config import LUMINOSITY_CONFIG, SUBTYPE_FRACTIONS
    mp = LUMINOSITY_CONFIG["M_peak"]
    assert mp["Ib"] == {"mean": -17.33, "sigma": 0.60} and mp["Ic"] == {"mean": -17.78, "sigma": 0.85}
    assert mp["Ic-BL"] == {"mean": -18.57, "sigma": 0.63} and mp["IIb"]["mean"] == -17.57
    assert mp["IIn"] == {"mean": -18.72, "sigma": 1.32}
    f = SUBTYPE_FRACTIONS["Ibc"]
    assert f == {"Ib": 0.556, "Ic": 0.386, "Ic-BL": 0.058} and abs(sum(f.values()) - 1) < 1e-9
    from config import EXTINCTION_CONFIG as E
    assert E["SNIb"] == {"frac_zero": 0.375, "tau": 0.68, "sigma_zero": 0.01, "Av_max": 3.0, "Rv": 2.6}
    assert E["SNIc"] == {"frac_zero": 0.27, "tau": 0.86, "sigma_zero": 0.01, "Av_max": 3.0, "Rv": 4.3}
    assert E["SNIcBL"] == {"frac_zero": 0.74, "tau": 0.58, "sigma_zero": 0.01, "Av_max": 3.0, "Rv": 3.1}
    assert E["SNIIn"]["frac_zero"] == 1.0 and E["SNIIn"]["sigma_zero"] == 0.0     # runner viejo, sin tocar

def test_fixc_config_values():
    from config import LUMINOSITY_CONFIG, SUBTYPE_FRACTIONS, LF_AFTER_HOST_DUST, EXTINCTION_CONFIG as E
    mp = LUMINOSITY_CONFIG["M_peak"]
    assert mp["IIP"] == {"mean": -15.75, "sigma": 1.23} and mp["IIL"] == {"mean": -17.53, "sigma": 0.64}
    assert SUBTYPE_FRACTIONS["II"] == {"IIP": 0.875, "IIL": 0.125}
    assert LF_AFTER_HOST_DUST == {"II", "IIn"}
    v = {"frac_zero": 1.0, "sigma_zero": 0.2, "tau": 0.25, "Av_max": 3.0, "Rv": 3.1}
    assert E["SNII_v78"] == v and E["SNIIn_v78"] == v
    rng = np.random.default_rng(4)
    m = np.array([sample_mpeak(rng, "II", subtype="IIL") for _ in range(20000)])
    assert abs(m.mean() + 17.53) < 0.03

def test_mpeak_truncated_by_resampling():
    from config import LUMINOSITY_CONFIG
    c = LUMINOSITY_CONFIG["clip"]
    rng = np.random.default_rng(6)
    m = np.array([sample_mpeak(rng, "II", subtype="IIP") for _ in range(20000)])
    assert m.min() >= c["min"] and m.max() <= c["max"]
    assert (m == c["max"]).sum() == 0          # sin acumulacion en el borde
    # cortar la cola debil baja (mas negativa) la media respecto de la normal sin truncar
    assert m.mean() < -15.75

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
