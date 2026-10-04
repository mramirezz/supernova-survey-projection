# tests/test_p78_zlf_mcmc.py
"""MCMC vectorizado del extractor (ZLF mcmc_fitter): la log-probabilidad en bloque, el filtro de upper limits en bloque
y fit_mcmc completo dan lo mismo que el camino escalar (ZLF_MCMC_VECTORIZE=0).

ZLF corre en un subproceso con el python del env projection: ahi esta emcee, y ZLF va al frente de sys.path con su
propio config.py, como en produccion (este proceso de pytest tiene el config.py de proyeccion)."""
import os, sys, pathlib, subprocess
import numpy as np
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline78.paths import ZLF

PY = pathlib.Path("/opt/anaconda3/envs/projection/bin/python")
pytestmark = pytest.mark.skipif(not PY.exists(), reason="sin el env projection (emcee)")

# Sonda que corre dentro de ZLF. argv: ruta ZLF, modo (lp | filtro | fit), npz de salida.
_SONDA = r'''
import sys
import numpy as np
zlf, modo, salida = sys.argv[1:4]
sys.path.insert(0, zlf)
import mcmc_fitter as M
from config import MCMC_CONFIG, MODEL_CONFIG
from model import alerce_model

VERDAD = np.array([6e-8, 0.3, 10.0, 4.0, 30.0, 25.0])      # A, f, t0, t_rise, t_fall, gamma (mag ~18 en el pico)


def curva(n_det=25, n_ul_antes=10, n_ul_despues=2, seed=3):
    """Curva tipo lector: detecciones con 5 % de error, UL antes y despues (flux_err NaN, como en reader.py)."""
    rng = np.random.default_rng(seed)
    t_det = np.sort(np.r_[0.0, rng.uniform(0.5, 130.0, n_det - 1)])
    f_det = alerce_model(t_det, *VERDAD)
    e_det = 0.05 * f_det
    f_det = np.abs(f_det + e_det * rng.standard_normal(n_det))
    t_ul = np.r_[np.sort(rng.uniform(-30.0, -1.0, n_ul_antes)), np.sort(rng.uniform(135.0, 160.0, n_ul_despues))]
    f_ul = 10 ** (-rng.uniform(20.0, 20.8, len(t_ul)) / 2.5)
    t = np.r_[t_det, t_ul]
    f = np.r_[f_det, f_ul]
    e = np.r_[e_det, np.full(len(t_ul), np.nan)]
    ul = np.r_[np.zeros(n_det, bool), np.ones(len(t_ul), bool)]
    return t, f, e, ul


def saneado(f, e):
    """El mismo reemplazo de flux_err que hace fit_mcmc antes de muestrear."""
    return np.where(np.isfinite(e) & (e > 0), e, np.maximum(np.abs(f) * 0.01, 1e-10))


def limites(t, f):
    """Bounds dinamicos de t0 y A como en fit_mcmc."""
    dyn = dict(MODEL_CONFIG["bounds"])
    marg = max(100.0, 0.5 * (t.max() - t.min()))
    dyn["t0"] = (t.min() - marg, t.max() + marg)
    a0, a1 = MODEL_CONFIG["bounds"]["A"]
    dyn["A"] = (max(a0, f.min() * 0.01), min(a1, f.max() * 50.0))
    return dyn


def vectores(dyn, seed=5):
    """200 vectores: cerca del ajuste (sin exceso de UL), en todo el dominio (muchos exceden UL), fuera de los
    bounds, exactamente en un bound (desigualdad estricta) y con NaN."""
    rng = np.random.default_rng(seed)
    nombres = MODEL_CONFIG["param_names"]
    lo = np.array([dyn[k][0] for k in nombres])
    hi = np.array([dyn[k][1] for k in nombres])
    P = np.empty((200, 6))
    P[:60] = VERDAD * (1 + 0.05 * rng.standard_normal((60, 6)))
    P[60:140] = lo + (hi - lo) * rng.uniform(size=(80, 6))
    P[60:100, 0] = 10 ** rng.uniform(np.log10(lo[0]), np.log10(hi[0]), 40)     # A log-uniforme
    P[140:190] = VERDAD * (1 + 0.05 * rng.standard_normal((50, 6)))
    for i, j in zip(range(140, 190), rng.integers(0, 6, 50)):
        P[i, j] = lo[j] - (hi[j] - lo[j]) * rng.uniform(0.01, 1) if i % 2 else hi[j] + (hi[j] - lo[j]) * rng.uniform(0.01, 1)
    for i, j in zip(range(190, 198), rng.integers(0, 6, 8)):
        P[i] = VERDAD
        P[i, j] = lo[j] if i % 2 else hi[j]
    P[198:] = VERDAD
    P[198, 1] = np.nan
    P[199, 4] = np.nan
    return P


if modo == "lp":
    t, f, e, ul = curva()
    dyn = limites(t[~ul], f[~ul])
    P = vectores(dyn)
    es = saneado(f, e)
    casos = {
        "ul_nan": (t, f, e, ul),
        "ul": (t, f, es, ul),
        "sin_ul_none": (t[~ul], f[~ul], es[~ul], None),
        "sin_ul_false": (t[~ul], f[~ul], es[~ul], np.zeros((~ul).sum(), bool)),
        "solo_ul": (t[ul], f[ul], es[ul], np.ones(ul.sum(), bool)),
    }
    out = {"P": P, "vectorize_default": MCMC_CONFIG["vectorize"]}
    for k, (tt, ff, ee, uu) in casos.items():
        out[k + "_esc"] = np.array([M.log_posterior(p, tt, ff, ee, dynamic_bounds=dyn, is_upper_limit=uu) for p in P])
        out[k + "_vec"] = M.log_posterior_vec(P, tt, ff, ee, dynamic_bounds=dyn, is_upper_limit=uu)
        # emcee llama por mitades del ensemble: el resultado de cada fila no depende del bloque
        out[k + "_partes"] = np.concatenate([M.log_posterior_vec(P[a:b], tt, ff, ee, dynamic_bounds=dyn,
                                                                 is_upper_limit=uu)
                                             for a, b in ((0, 1), (1, 7), (7, 100), (100, 200))])
    np.savez(salida, **out)

elif modo == "filtro":
    t, f, e, ul = curva()
    dyn = limites(t[~ul], f[~ul])
    rng = np.random.default_rng(11)
    nombres = MODEL_CONFIG["param_names"]
    lo = np.array([dyn[k][0] for k in nombres])
    hi = np.array([dyn[k][1] for k in nombres])
    cerca = VERDAD * (1 + 0.15 * rng.standard_normal((18000, 6)))
    todo = lo + (hi - lo) * rng.uniform(size=(12000, 6))
    # muestras justo en el borde del criterio (modelo = 1.01 * UL): el modelo es lineal en A
    base = VERDAD * (1 + 0.05 * rng.standard_normal((300, 6)))
    razon = np.array([np.max(alerce_model(t[ul], *p) / (f[ul] * 1.01)) for p in base])
    base[:, 0] = base[:, 0] / razon * (1 + rng.choice([-1e-9, -1e-13, 0.0, 1e-13, 1e-9], 300))
    S = np.concatenate([cerca, todo, base])
    bucle = np.array(M._filtrar_ul_bucle(S, t, f, ul))
    masc = M._mascara_ul_vec(S, t, f, ul, bloque=7000)
    np.savez(salida, S=S, bucle=bucle, masc=masc, n_borde=len(base))

elif modo == "fit":
    t, f, e, ul = curva(n_det=12, n_ul_antes=9, n_ul_despues=1, seed=7)
    r = M.fit_mcmc(t.copy(), f.copy(), e.copy(), verbose=False, is_upper_limit=ul)
    s = r["sampler"]
    moc = r["params_median_of_curves"]
    np.savez(salida, chain=s.get_chain(), lnp=s.get_log_prob(), acc=s.acceptance_fraction,
             samples_valid=r["samples_valid"], best=r["samples_best_200"], params=r["params"],
             moc=np.full(6, np.nan) if moc is None else moc, vectorize_cfg=MCMC_CONFIG["vectorize"],
             vectorize_sampler=s.vectorize, n_walkers=MCMC_CONFIG["n_walkers"], n_steps=MCMC_CONFIG["n_steps"])
'''


def _sonda(tmp_path, modo, etiqueta="", **env_extra):
    script = tmp_path / "sonda_zlf_mcmc.py"
    script.write_text(_SONDA)
    salida = tmp_path / f"{modo}{etiqueta}.npz"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ZLF_", "ZTF_"))}
    env.update(env_extra, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([str(PY), str(script), str(ZLF), modo, str(salida)], env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    with np.load(salida) as z:
        return {k: z[k] for k in z.files}


def test_log_prob_vectorizada_igual_a_la_escalar(tmp_path):
    """(a) 200 vectores dentro y fuera de los bounds: mismos +-inf exactos y finitos a 1e-10 relativo."""
    z = _sonda(tmp_path, "lp")
    assert int(z["vectorize_default"]) == 1          # sin ZLF_MCMC_VECTORIZE el default es el camino vectorizado
    for caso in ("ul_nan", "ul", "sin_ul_none", "sin_ul_false", "solo_ul"):
        esc, vec, partes = z[caso + "_esc"], z[caso + "_vec"], z[caso + "_partes"]
        assert esc.shape == vec.shape == partes.shape == (200,)
        for v in (vec, partes):
            assert np.array_equal(np.isposinf(esc), np.isposinf(v)), caso
            assert np.array_equal(np.isneginf(esc), np.isneginf(v)), caso
            assert not np.isnan(v).any(), caso
            fin = np.isfinite(esc)
            np.testing.assert_allclose(v[fin], esc[fin], rtol=1e-10, atol=0, err_msg=caso)
            # mas fuerte que el 1e-10: identico bit a bit (si no, una aceptacion al limite podria separar las cadenas)
            assert np.array_equal(v, esc), caso
        assert np.isneginf(esc[140:200]).all(), caso     # fuera de bounds, en el bound o NaN: -inf
    # la curva con UL ejercita los dos regimenes del prior: sin exceso y con la penalizacion -1e8
    esc = z["ul_esc"]
    assert (np.isfinite(esc) & (esc > -1e6)).sum() >= 20
    assert (np.isfinite(esc) & (esc < -1e6)).sum() >= 20


def test_filtro_ul_vectorizado_selecciona_las_mismas_muestras(tmp_path):
    """(b) mismo criterio que el bucle (modelo <= 1.01 * UL en todos los UL), incluso en el borde del criterio."""
    z = _sonda(tmp_path, "filtro")
    S, bucle, masc = z["S"], z["bucle"], z["masc"]
    assert masc.dtype == bool and masc.shape == (len(S),)
    assert np.array_equal(S[masc], bucle)
    assert 0 < masc.sum() < len(S)
    borde = masc[-int(z["n_borde"]):]
    assert 0 < borde.sum() < len(borde)               # las del borde caen a ambos lados


def test_fit_mcmc_misma_cadena_con_y_sin_vectorize(tmp_path):
    """(c) fit_mcmc corto (16 walkers, 200 pasos), misma semilla: misma cadena y mismos productos con 0 y 1."""
    base = dict(ZLF_MCMC_WALKERS="16", ZLF_MCMC_STEPS="200", ZLF_MCMC_BURN="50", ZLF_MCMC_SEED="41")
    r0 = _sonda(tmp_path, "fit", "_v0", ZLF_MCMC_VECTORIZE="0", **base)
    r1 = _sonda(tmp_path, "fit", "_v1", ZLF_MCMC_VECTORIZE="1", **base)
    assert int(r0["vectorize_cfg"]) == 0 and not bool(r0["vectorize_sampler"])
    assert int(r1["vectorize_cfg"]) == 1 and bool(r1["vectorize_sampler"])
    assert int(r1["n_walkers"]) == 16 and int(r1["n_steps"]) == 200
    assert r0["chain"].shape == r1["chain"].shape == (200, 16, 6)
    assert np.allclose(r0["chain"], r1["chain"])
    assert np.allclose(r0["lnp"], r1["lnp"])
    assert np.array_equal(r0["chain"], r1["chain"]) and np.array_equal(r0["lnp"], r1["lnp"])     # bit a bit
    assert r0["acc"].mean() > 0.05 and np.array_equal(r0["acc"], r1["acc"])
    for k in ("samples_valid", "best", "params", "moc"):
        assert r0[k].shape == r1[k].shape and np.allclose(r0[k], r1[k], equal_nan=True), k
