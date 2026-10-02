import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline78.engine import extinction_factor, observed_lightcurves
from pipeline78.bands import legacy_band, survey_bands

def _tpl():
    w = np.arange(3005.0, 9195.0, 1.0)
    t = np.arange(-20.0, 100.0)
    prof = np.exp(-0.5 * (t / 15.0) ** 2) + 0.05
    bb = 1.0 / (w**5 * (np.exp(1.4388e8 / (w * 10000.0)) - 1.0))
    return dict(sn="x", time=t, wave=w, flux=prof[:, None] * (bb / bb.max())[None, :], t_peak=0.0)

def test_extinction_factor_v_band():
    f = extinction_factor(np.array([5500.0]), 3.1, 0.1)[0]
    assert abs(f - 10 ** (-0.4 * 0.31)) < 0.02

def test_time_dilation():
    t_rel, _ = observed_lightcurves(_tpl(), 0.5, 0.0, 3.1, 0.0, survey_bands("ZTF", ("r",)))
    assert np.allclose(t_rel, np.arange(-20.0, 100.0) * 1.5)

def test_high_z_drops_blue_band():
    _, mags = observed_lightcurves(_tpl(), 0.6, 0.0, 3.1, 0.0, survey_bands("SUDARE"))
    assert "g" not in mags and "i" in mags

def test_dmag_shifts_all_bands_equally():
    b = survey_bands("ZTF")
    _, a = observed_lightcurves(_tpl(), 0.05, 0.1, 3.1, 0.02, b, dmag=0.0)
    _, c = observed_lightcurves(_tpl(), 0.05, 0.1, 3.1, 0.02, b, dmag=1.3)
    for k in a: assert np.allclose(c[k] - a[k], 1.3, atol=1e-6)

def test_golden_vs_legacy_chain():
    """Mismo template, z, polvo y curva SDSS r: el motor nuevo contra correct_redeening + Syntetic_photometry_v2."""
    from pipeline78.paths import LIB
    from pipeline78.store import parse_dat
    from core.utils import leer_spec, Syntetic_photometry_v2
    from core.correction import correct_redeening
    import contextlib
    path = min(LIB.glob("Ia/mangled/*.dat"), key=lambda p: p.stat().st_size)
    t, w, f = parse_dat(path)
    tpl = dict(sn=path.stem, time=t[:5], wave=w, flux=f[:5], t_peak=float(t[0]))
    rb = legacy_band("r")
    _, mags = observed_lightcurves(tpl, 0.03, 0.10, 3.1, 0.03, [rb])
    esp, fases = leer_spec(str(path), ot=False, as_pandas=True)
    with contextlib.redirect_stdout(open("/dev/null", "w")):
        espc, _ = correct_redeening(sn=path.stem, ESPECTRO=esp[:5], fases=fases[:5], z=0.03, ebmv_host=0.10,
                                    ebmv_mw=0.03, reverse=True, use_DL=True, rv_host=3.1)
    old = []
    for s in espc:
        F, _ = Syntetic_photometry_v2(s["wave"].values, s["flux"].values, rb.wave, rb.resp)
        old.append(-2.5 * np.log10(F / rb.f0))
    d = np.abs(np.asarray(old) - mags["r"])
    assert np.median(d) < 0.003 and d.max() < 0.01, d

def test_speed():
    from pipeline78.store import load_template
    from pipeline78.paths import STORE
    import pandas as pd
    p = pd.read_csv(STORE / "catalog.csv").query("sn == 'SN2011fe'").store_path.iloc[0]
    tpl = load_template(p)
    b = survey_bands("ZTF")
    t0 = time.time()
    for _ in range(20): observed_lightcurves(tpl, 0.05, 0.1, 3.1, 0.02, b)
    per = (time.time() - t0) / 20
    print(f"   {per*1000:.0f} ms por simulacion (SN2011fe, {tpl['n_epochs']} epocas)")
    assert per < 0.5

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
