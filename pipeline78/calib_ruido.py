# pipeline78/calib_ruido.py
"""Calibracion del ruido de tres terminos (fondo, fuente, piso) con las detecciones reales del holdout ZTF val.

sigma_m^2 = (A 1.0857 / (5 10^(0.4 dm)))^2 + (B 10^(-0.2 dm))^2 + C^2, dm = m_lim - m (project.sigma_tres_terminos).
Datos: SOLO la mitad val del holdout ZTF (origen holdout, split val, sin excluir=True), sin final y sin viejas.
Ajuste: minimos cuadrados sobre la mediana de sigma real en bins de dm (0 a 4 mag, 0.25 de ancho), residuos pesados
por sqrt(n), con dm_bin = mediana de dm del bin (sigma es monotona en dm: mediana(sigma) = sigma(mediana dm)).
Error: bootstrap sobre SNe (se re-sortean los oid con reemplazo).

Dos fuentes de m_lim (--maglim):
  alerce (default): diffmaglim de cada alerta, bajado de la API de ALeRCE (cache en data/ruido_alerce_val.csv) y
    emparejado por oid, banda, mjd y magnitud. Es el limite 5 sigma real de la imagen diferencia de esa deteccion.
  log: maglim de ztf_obslog_best.parquet por oid, banda y dia (floor, como survey._best_per_day). En las epocas con
    deteccion ese valor es ESTIMADO en ZTF_observing_log_complete.csv (diffmaglim_original vacio, estimated=True en
    el 100 %) y recortado a >= m + 0.5 (41 % de las detecciones val quedan en dm = 0.5 exacto). Solo se usa como
    comparacion.
Los valores que usan las sims se escriben a mano en runcfg (no se recalculan en cada corrida).
"""
import argparse, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pipeline78.paths import RUNS, STORE, DATA, PHD
from pipeline78.project import sigma_tres_terminos

EDGES = np.round(np.arange(0.0, 4.0 + 1e-9, 0.25), 2)
ALERCE_API = "https://api.alerce.online/ztf/v1/objects/{}/detections"
CACHE = DATA / "ruido_alerce_val.csv"
FIG = PHD / "paper2_ZTF/figures_templates/pipeline78_ajustes_mcmc/ruido_calibrado.png"
BANDS = {1: "g", 2: "r", 3: "i"}


def sigma_snr(dm, noise_k=5.0, sigma_floor=0.02):
    """Regla "snr" de project._noise (ztf_v78)."""
    return np.clip(1.0857 / np.maximum(noise_k * 10.0 ** (0.4 * np.asarray(dm, float)), 1e-6), sigma_floor, None)


def val_meta(real_dir=RUNS / "real_ztf"):
    m = pd.read_csv(real_dir / "meta_real_ztf.csv")
    return m[(m.origen == "holdout") & (m.split == "val") & ~m.excluir.astype(bool)].reset_index(drop=True)


def val_detections(real_dir=RUNS / "real_ztf"):
    """Detecciones (upperlimit F) de la mitad val del holdout, sin excluidas: oid, sn_type, filter, mjd, m, sig."""
    meta = val_meta(real_dir)
    out = []
    for c in sorted(meta.sn_type.unique()):
        d = pd.read_parquet(real_dir / f"{c}.parquet")
        out.append(d[d.oid.isin(set(meta.oid)) & (d.upperlimit == "F")])
    d = pd.concat(out, ignore_index=True).rename(columns={"magnitud_proyectada": "m", "magerr": "sig"})
    return d[["oid", "sn_type", "filter", "mjd", "m", "sig"]].reset_index(drop=True)


def with_log_maglim(det, log_path=STORE / "ztf_obslog_best.parquet"):
    """maglim del log por oid, banda y dia (floor). Ver la advertencia del docstring del modulo."""
    log = pd.read_parquet(log_path, filters=[("field", "in", sorted(set(det.oid)))])
    log = log.assign(day=np.floor(log.mjd).astype("int64")).rename(columns={"field": "oid", "band": "filter"})
    d = det.assign(day=np.floor(det.mjd).astype("int64"))
    d = d.merge(log[["oid", "filter", "day", "maglim"]], on=["oid", "filter", "day"], how="left").drop(columns="day")
    return d.assign(dm=d.maglim - d.m)


def _get(oid, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(ALERCE_API.format(oid), timeout=60) as r:
                return oid, json.load(r)
        except Exception as e:                                # red: reintento con espera creciente
            if k == tries - 1:
                raise RuntimeError(f"{oid}: {e}")
            time.sleep(2.0 * (k + 1))


def fetch_alerce(oids, out=CACHE, workers=4):
    """Detecciones de ALeRCE (mjd, fid, magpsf, sigmapsf, diffmaglim, isdiffpos) de cada oid -> out (csv)."""
    cols = ["oid", "candid", "mjd", "fid", "magpsf", "sigmapsf", "diffmaglim", "isdiffpos"]
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for oid, js in ex.map(_get, sorted(set(oids))):
            for x in js:
                rows.append([oid] + [x.get(c) for c in cols[1:]])
    df = pd.DataFrame(rows, columns=cols).sort_values(["oid", "mjd", "fid"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    return df


def with_alerce_maglim(det, alerce, tol_mjd=1e-3, tol_mag=2e-3):
    """diffmaglim de la alerta que corresponde a cada deteccion: misma banda, |dmjd| <= tol_mjd (el .dat trae 3
    decimales) y |dmag| <= tol_mag; si hay mas de una candidata, la de mjd mas cercano. Solo alertas con
    isdiffpos > 0: las 54 detecciones val que son restas negativas quedan sin m_lim y fuera del ajuste."""
    a = alerce[alerce.isdiffpos > 0]
    a = a.assign(filter=a.fid.map(BANDS)).dropna(subset=["diffmaglim"])
    ga = {k: g.sort_values("mjd") for k, g in a.groupby(["oid", "filter"])}
    lim = np.full(len(det), np.nan)
    for (oid, b), g in det.groupby(["oid", "filter"]):
        if (oid, b) not in ga:
            continue
        x = ga[(oid, b)]
        dt = np.abs(g.mjd.to_numpy()[:, None] - x.mjd.to_numpy()[None, :])
        ok = (dt <= tol_mjd) & (np.abs(g.m.to_numpy()[:, None] - x.magpsf.to_numpy()[None, :]) <= tol_mag)
        dt = np.where(ok, dt, np.inf)
        j = dt.argmin(axis=1)
        hit = np.isfinite(dt[np.arange(len(g)), j])
        lim[g.index.to_numpy()[hit]] = x.diffmaglim.to_numpy()[j[hit]]
    d = det.assign(maglim=lim)
    return d.assign(dm=d.maglim - d.m)


def bin_table(dm, sig, edges=EDGES):
    """n, mediana de dm y mediana de sigma por bin [lo, hi)."""
    dm, sig = np.asarray(dm, float), np.asarray(sig, float)
    k = np.digitize(dm, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        s = k == i
        rows.append((edges[i], edges[i + 1], int(s.sum()),
                     float(np.median(dm[s])) if s.any() else np.nan, float(np.median(sig[s])) if s.any() else np.nan))
    return pd.DataFrame(rows, columns=["lo", "hi", "n", "dm_med", "sig_med"])


def fit_abc(tab, x0=(1.0, 0.03, 0.02)):
    """A, B, C >= 0 por minimos cuadrados sobre las medianas, residuo (modelo - mediana) * sqrt(n)."""
    t = tab[tab.n > 0]
    w = np.sqrt(t.n.to_numpy(float))
    r = least_squares(lambda p: (sigma_tres_terminos(t.dm_med.to_numpy(), *p) - t.sig_med.to_numpy()) * w,
                      x0, bounds=(0.0, np.inf), x_scale=(1.0, 0.03, 0.02))
    return r.x


def _fit_sets(dm, sig, band):
    """Ajustes global (g + r) y por banda sobre las mismas filas."""
    return {"gr": fit_abc(bin_table(dm, sig)),
            "g": fit_abc(bin_table(dm[band == "g"], sig[band == "g"])),
            "r": fit_abc(bin_table(dm[band == "r"], sig[band == "r"]))}


def bootstrap(d, n_boot=300, seed=20261004):
    """Re-sorteo de SNe con reemplazo (todas las detecciones de cada SN sorteada); en cada sorteo se ajustan g + r, g
    y r sobre las mismas SNe, asi la diferencia g - r lleva su correlacion. -> {lab: array (n_boot, 3)}."""
    rng = np.random.default_rng(seed)
    idx = list(d.groupby("oid").indices.values())
    dm, sig, band = d.dm.to_numpy(), d.sig.to_numpy(), d["filter"].to_numpy()
    out = {"gr": [], "g": [], "r": []}
    for _ in range(n_boot):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])
        for k, v in _fit_sets(dm[ii], sig[ii], band[ii]).items():
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def calibrate(d, n_boot=300, seed=20261004):
    """Ajuste global (g + r) y por banda, con errores bootstrap. d: filas con dm finito en [0, 4)."""
    p = _fit_sets(d.dm.to_numpy(), d.sig.to_numpy(), d["filter"].to_numpy())
    bs = bootstrap(d, n_boot, seed)
    res = {}
    for lab, s in (("gr", d), ("g", d[d["filter"] == "g"]), ("r", d[d["filter"] == "r"])):
        res[lab] = dict(p=p[lab], err=bs[lab].std(axis=0, ddof=1), boot=bs[lab], tab=bin_table(s.dm, s.sig),
                        n_sn=s.oid.nunique(), n_det=len(s))
    return res


def bands_differ(res):
    """Diferencia g - r de cada parametro en unidades de su error: el de la diferencia en el bootstrap conjunto
    (mismas SNe en g y r) y, entre parentesis en main, el de sumar los errores en cuadratura."""
    g, r = res["g"], res["r"]
    return (g["p"] - r["p"]) / (g["boot"] - r["boot"]).std(axis=0, ddof=1), \
        (g["p"] - r["p"]) / np.hypot(g["err"], r["err"])


def residual_table(res, lab="gr"):
    t = res[lab]["tab"].copy()
    t["sig_snr"] = sigma_snr(t.dm_med)
    t["sig_fit"] = sigma_tres_terminos(t.dm_med, *res[lab]["p"])
    t["cociente_snr"] = t.sig_med / t.sig_snr
    t["cociente_fit"] = t.sig_med / t.sig_fit
    return t


def plot(d, res, out=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight"})
    from matplotlib.lines import Line2D
    col = {"g": "tab:green", "r": "tab:red"}
    fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.5))
    x = np.linspace(-0.2, 4.2, 300)
    for ax, b in zip(axs[:2], ("g", "r")):
        s = d[d["filter"] == b]
        ax.plot(s.dm, s.sig, ".", ms=1.0, color="0.75", alpha=0.35, rasterized=True)
        t = res[b]["tab"]
        ax.plot(x, sigma_snr(x), "k:", lw=1.0)
        ax.plot(x, sigma_tres_terminos(x, *res["gr"]["p"]), "--", color="k", lw=0.9)
        ax.plot(x, sigma_tres_terminos(x, *res[b]["p"]), "-", color=col[b], lw=1.2)
        ax.plot(t.dm_med, t.sig_med, "o", ms=3.5, mfc=col[b], mec="k", mew=0.4)
        ax.set_yscale("log")
        ax.set_xlim(-0.1, 4.1)
        ax.set_ylim(0.012, 0.6)
        ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel(r"$m_{\rm lim} - m$ [mag]")
        ax.text(0.95, 0.93, f"${b}$ band", transform=ax.transAxes, ha="right", va="top")
    axs[0].set_ylabel(r"$\sigma_m$ [mag]")
    hs = [Line2D([], [], ls="", marker="o", ms=3.5, mfc="0.5", mec="k", mew=0.4, label="ZTF detections, bin median"),
          Line2D([], [], ls="-", color="0.4", lw=1.2, label="three terms, fit per band"),
          Line2D([], [], ls="--", color="k", lw=0.9, label="three terms, $g+r$ fit"),
          Line2D([], [], ls=":", color="k", lw=1.0, label=r"previous: S/N only, floor 0.02 mag")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.0, columnspacing=1.5)
    ax = axs[2]
    for b, mk in (("g", "o"), ("r", "s")):
        t = residual_table(res, b)
        ax.plot(t.dm_med, t.cociente_snr, mk, ms=3.5, mfc="none", mec=col[b], mew=0.8, label=f"${b}$, previous")
        ax.plot(t.dm_med, t.cociente_fit, mk, ms=3.5, mfc=col[b], mec="k", mew=0.4, label=f"${b}$, three terms")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axhspan(0.85, 1.15, color="0.9", zorder=0)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(0.7, 3.8)
    ax.set_xlabel(r"$m_{\rm lim} - m$ [mag]")
    ax.set_ylabel(r"observed / model $\sigma_m$")
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def load(maglim="alerce", fetch=False, real_dir=RUNS / "real_ztf"):
    det = val_detections(real_dir)
    if maglim == "log":
        d = with_log_maglim(det)
    else:
        if fetch or not CACHE.exists():
            fetch_alerce(det.oid.unique(), CACHE)
        d = with_alerce_maglim(det, pd.read_csv(CACHE))
    return det, d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--maglim", choices=["alerce", "log"], default="alerce")
    ap.add_argument("--fetch", action="store_true", help="vuelve a bajar las detecciones de ALeRCE")
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()
    det, d = load(a.maglim, a.fetch)
    ok = np.isfinite(d.dm)
    print(f"detecciones val: {len(det)} de {det.oid.nunique()} SNe; con m_lim ({a.maglim}): {ok.sum()}")
    d = d[ok & (d.dm >= EDGES[0]) & (d.dm < EDGES[-1])].reset_index(drop=True)
    print(f"en 0 <= dm < 4: {len(d)} de {d.oid.nunique()} SNe; dm = 0.5 exacto (+-0.002): "
          f"{int((np.abs(d.dm - 0.5) < 0.002).sum())}")
    res = calibrate(d, a.nboot)
    for lab in ("gr", "g", "r"):
        p, e = res[lab]["p"], res[lab]["err"]
        print(f"{lab:>2}: A = {p[0]:.4f} +- {e[0]:.4f}  B = {p[1]:.4f} +- {e[1]:.4f}  C = {p[2]:.4f} +- {e[2]:.4f}"
              f"  ({res[lab]['n_det']} det, {res[lab]['n_sn']} SNe)")
    dj, dq = bands_differ(res)
    print("g - r en sigmas (A, B, C): bootstrap conjunto", np.round(dj, 2), " cuadratura", np.round(dq, 2))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for lab in ("gr", "g", "r"):
            print(f"\n[{lab}]\n", residual_table(res, lab).to_string(index=False))
    if a.fig:
        print("figura:", plot(d, res))
    return res


if __name__ == "__main__":
    main()
