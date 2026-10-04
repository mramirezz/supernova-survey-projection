"""Calibracion: temperature scaling, correccion de priors de clase y ECE.

- Temperatura (Guo et al. 2017, "On Calibration of Modern Neural Networks", 2017arXiv170604599G, Sec. 4.2 ec. 9): un
  escalar T > 0, q = softmax(z / T), ajustado minimizando la NLL en un set de validacion. No cambia el argmax. Aca
  z = log p: softmax(log p / T) = softmax(z / T) porque log p y z difieren en una constante por fila, asi que vale
  igual para la red, el ensemble (log de la media de probabilidades) y SuperNNova o Villar (solo dan p).
- Correccion de priors (revision nn-lit H5, logit adjustment): q = softmax(log p / T' + log pi - log pi_train). La red,
  SuperNNova y el baseline entrenan con clases balanceadas (pi_train uniforme), asi que el termino es log pi. pi son
  las frecuencias de clase de val_sel. T' se ajusta en val_sel con la correccion puesta. A diferencia de T, la
  correccion SI cambia el argmax: sube la exactitud y suele bajar la exactitud balanceada.
  OJO: la composicion de val la fija la construccion del holdout (tope de 300 por clase, holdout.py), no la
  poblacion de un survey. Esta correccion lleva el clasificador a la mezcla de val. Para las tasas hacen falta los
  priors del survey o la inversion de la matriz de confusion, no estos.
- ECE (Guo et al. 2017, Sec. 2 ec. 3) = sum_m |B_m| / n |acc(B_m) - conf(B_m)| con M bins de igual ancho en la
  confianza maxima, intervalos ((m - 1)/M, m/M]. Guo usa M = 15 en sus tablas, y es el default.
- Particion anidada (revision H1, pipeline78.splits): T, T' y pi se ajustan en val_sel y las cifras honestas son las
  de val_rep. Se reportan tambien val_sel (dentro de muestra) y val completo (referencia), y el ECE con T en
  validacion cruzada en K partes sobre todo val (la cifra del commit 528f88b).
"""
import numpy as np
from scipy.optimize import minimize_scalar

N_BINS = 15
T_MIN, T_MAX = 0.05, 20.0


def _logp(p):
    return np.log(np.clip(np.asarray(p, np.float64), 1e-12, None))


def apply_temperature(p, T, log_adj=None):
    """softmax(log p / T + log_adj). log_adj = None es temperature scaling puro."""
    z = _logp(p) / T
    if log_adj is not None:
        z = z + np.asarray(log_adj, np.float64)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    p = np.asarray(p, np.float64)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def fit_temperature(p, y, log_adj=None):
    """T > 0 que minimiza la NLL de softmax(log p / T + log_adj). Busqueda acotada en log T, T entre 0.05 y 20."""
    y = np.asarray(y, int)
    r = minimize_scalar(lambda lt: nll(apply_temperature(p, np.exp(lt), log_adj), y),
                        bounds=(np.log(T_MIN), np.log(T_MAX)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


def class_priors(y, n_cls):
    """Frecuencias de clase. Una clase ausente recibe media cuenta, para que log pi sea finito."""
    n = np.bincount(np.asarray(y, int), minlength=n_cls).astype(np.float64)
    n[n == 0] = 0.5
    return n / n.sum()


def prior_adjustment(priors, train_priors=None):
    """log pi - log pi_train (pi_train uniforme por defecto: el entrenamiento balancea las clases)."""
    pi = np.asarray(priors, np.float64)
    pt = np.full(len(pi), 1.0 / len(pi)) if train_priors is None else np.asarray(train_priors, np.float64)
    return np.log(pi) - np.log(pt)


def balanced_accuracy(y, yhat):
    """Media de la exactitud por clase sobre las clases presentes en y (igual que sklearn)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean([np.mean(yhat[y == k] == k) for k in np.unique(y)])) if len(y) else float("nan")


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


def ece_cv(p, y, n_bins=N_BINS, k_folds=5, seed=0):
    """ECE con T fuera de muestra: K partes estratificadas por clase, T ajustado en K - 1 y aplicado a la restante."""
    p, y = np.asarray(p, np.float64), np.asarray(y, int)
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
    return ece(p_cv, y, n_bins)


def _block(p, y, T, T_prior, adj, n_bins):
    q, qp = apply_temperature(p, T), apply_temperature(p, T_prior, adj)
    return {"n": int(len(y)), "ece_raw": ece(p, y, n_bins), "ece_ts": ece(q, y, n_bins),
            "ece_ts_prior": ece(qp, y, n_bins), "nll_raw": nll(p, y), "nll_ts": nll(q, y),
            "nll_ts_prior": nll(qp, y), "acc_raw": float(np.mean(p.argmax(1) == y)),
            "bal_acc_raw": balanced_accuracy(y, p.argmax(1)), "acc_ts_prior": float(np.mean(qp.argmax(1) == y)),
            "bal_acc_ts_prior": balanced_accuracy(y, qp.argmax(1))}


def calibration_report(p, y, sel=None, rep=None, n_cls=None, n_bins=N_BINS, k_folds=5, seed=0, train_priors=None):
    """T, T' y priors ajustados en sel. Cifras por subconjunto: val_rep (honesta), val_sel (dentro de muestra) y val.

    sel, rep: mascaras booleanas sobre las filas de p. Sin sel, todo es sel y rep (todo dentro de muestra, solo para
    pruebas o corridas sin particion)."""
    p, y = np.asarray(p, np.float64), np.asarray(y, int)
    n_cls = n_cls or p.shape[1]
    sel = np.ones(len(y), bool) if sel is None else np.asarray(sel, bool)
    rep = sel.copy() if rep is None else np.asarray(rep, bool)
    T = fit_temperature(p[sel], y[sel])
    pri = class_priors(y[sel], n_cls)
    adj = prior_adjustment(pri, train_priors)
    T_prior = fit_temperature(p[sel], y[sel], adj)
    by = {}
    for name, m in (("val_rep", rep), ("val_sel", sel), ("val", np.ones(len(y), bool))):
        if m.sum() >= 1:
            by[name] = _block(p[m], y[m], T, T_prior, adj, n_bins)
    rp = p[rep] if rep.any() else p
    ry = y[rep] if rep.any() else y
    return {"fit_on": "val_sel", "T": T, "T_prior": T_prior,
            "T_en_borde": bool(min(T, T_prior) < T_MIN * 1.02 or max(T, T_prior) > T_MAX * 0.995),
            "priors_val_sel": pri.tolist(), "log_prior_adj": adj.tolist(), "n_bins": n_bins, "by_subset": by,
            f"ece_ts_cv{k_folds}_val": ece_cv(p, y, n_bins, k_folds, seed),
            "reliability_subset": "val_rep" if rep.any() else "val",
            "reliability_raw": reliability(rp, ry, n_bins).tolist(),
            "reliability_ts": reliability(apply_temperature(rp, T), ry, n_bins).tolist(),
            "reliability_ts_prior": reliability(apply_temperature(rp, T_prior, adj), ry, n_bins).tolist()}


def plot_reliability(rep, path, plt):
    """Diagrama de confiabilidad sobre val_rep: cruda, con T y con T' + priors (ajustados en val_sel)."""
    sub = rep.get("reliability_subset", "val_rep")
    b = rep["by_subset"].get(sub, {})
    fig, ax = plt.subplots(figsize=(3.46, 3.2))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
    for key, color, mk, lab in (("reliability_raw", "k", "o", f"raw (ECE {b.get('ece_raw', np.nan):.3f})"),
                                ("reliability_ts", "tab:red", "s",
                                 f"$T$ = {rep['T']:.2f} (ECE {b.get('ece_ts', np.nan):.3f})"),
                                ("reliability_ts_prior", "tab:blue", "^",
                                 f"$T'$ = {rep['T_prior']:.2f} + priors (ECE {b.get('ece_ts_prior', np.nan):.3f})")):
        r = np.asarray(rep[key], float)
        s = r[:, 2] > 0
        ax.plot(r[s, 0], r[s, 1], color=color, marker=mk, ms=3.5, lw=1, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ZTF {sub.replace('_', ' ')} (calibrated on val sel)", fontsize=8)
    ax.legend(frameon=False, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
