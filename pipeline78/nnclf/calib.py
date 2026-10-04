"""Calibracion: temperature scaling y ECE (Guo et al. 2017, "On Calibration of Modern Neural Networks",
2017arXiv170604599G, Sec. 2 ec. 3 y Sec. 4.2 ec. 9).

- Temperatura: un escalar T > 0, q = softmax(z / T), ajustado minimizando la NLL en un set de validacion. No cambia el
  argmax. Aca z = log p: softmax(log p / T) = softmax(z / T) porque log p y z difieren en una constante por fila, asi
  que vale igual para la red, el ensemble (log de la media de probabilidades) y SuperNNova o Villar (solo dan p).
- ECE = sum_m |B_m| / n |acc(B_m) - conf(B_m)| con M bins de igual ancho en la confianza maxima, intervalos
  ((m - 1)/M, m/M]. Guo usa M = 15 en sus tablas, y es el default.
- El T se ajusta en val_real (R2/R5 de la revision de literatura) para aplicarlo despues a la mitad final. El ECE
  despues de calibrar en la misma muestra es optimista, por eso se reporta tambien con validacion cruzada en K partes
  (T ajustado en K - 1 y ECE en la restante).
"""
import numpy as np
from scipy.optimize import minimize_scalar

N_BINS = 15


def _logp(p):
    return np.log(np.clip(np.asarray(p, np.float64), 1e-12, None))


def apply_temperature(p, T):
    z = _logp(p) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    p = np.asarray(p, np.float64)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def fit_temperature(p, y):
    """T > 0 que minimiza la NLL de softmax(log p / T). Busqueda acotada en log T, T entre 0.05 y 20."""
    y = np.asarray(y, int)
    r = minimize_scalar(lambda lt: nll(apply_temperature(p, np.exp(lt)), y), bounds=(np.log(0.05), np.log(20.0)),
                        method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


def reliability(p, y, n_bins=N_BINS):
    """Por bin: (conf media, acc, n). Bins vacios con NaN."""
    p = np.asarray(p, np.float64)
    conf, pred = p.max(1), p.argmax(1)
    ok = pred == np.asarray(y)
    m = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)     # intervalos ((m-1)/M, m/M]
    rows = []
    for k in range(n_bins):
        s = m == k
        rows.append((float(conf[s].mean()) if s.any() else np.nan, float(ok[s].mean()) if s.any() else np.nan,
                     int(s.sum())))
    return np.array(rows, dtype=float)


def ece(p, y, n_bins=N_BINS):
    r = reliability(p, y, n_bins)
    n = r[:, 2].sum()
    s = r[:, 2] > 0
    return float(np.sum(r[s, 2] / n * np.abs(r[s, 1] - r[s, 0]))) if n else float("nan")


def calibration_report(p, y, n_bins=N_BINS, k_folds=5, seed=0):
    """ECE antes y despues de temperature scaling. T sobre todo el set (para la mitad final) y ECE fuera de muestra
    por K partes estratificadas por clase."""
    p, y = np.asarray(p, np.float64), np.asarray(y, int)
    T = fit_temperature(p, y)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    for c in np.unique(y):
        ix = np.flatnonzero(y == c)
        fold[rng.permutation(ix)] = np.arange(len(ix)) % k_folds
    p_cv = np.empty_like(p)
    for k in range(k_folds):
        te = fold == k
        if te.any() and (~te).any():
            p_cv[te] = apply_temperature(p[te], fit_temperature(p[~te], y[~te]))
    return {"T": T, "T_en_borde": bool(T < 0.051 or T > 19.9), "n": int(len(y)), "n_bins": n_bins,
            "ece_raw": ece(p, y, n_bins),
            "ece_ts_in_sample": ece(apply_temperature(p, T), y, n_bins), f"ece_ts_cv{k_folds}": ece(p_cv, y, n_bins),
            "nll_raw": nll(p, y), "nll_ts_in_sample": nll(apply_temperature(p, T), y),
            "reliability_raw": reliability(p, y, n_bins).tolist(),
            "reliability_ts": reliability(apply_temperature(p, T), y, n_bins).tolist()}


def plot_reliability(rep, path, plt):
    fig, ax = plt.subplots(figsize=(3.46, 3.2))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
    for key, color, mk, lab in (("reliability_raw", "k", "o", f"raw (ECE {rep['ece_raw']:.3f})"),
                                ("reliability_ts", "tab:red", "s",
                                 f"T = {rep['T']:.2f} (ECE {rep['ece_ts_in_sample']:.3f})")):
        r = np.asarray(rep[key], float)
        s = r[:, 2] > 0
        ax.plot(r[s, 0], r[s, 1], color=color, marker=mk, ms=3.5, lw=1, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
