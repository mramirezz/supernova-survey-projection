"""Diagnostico de un piloto: estado de las sims, M_peak observado contra la muestra real de validacion,
detecciones por banda, ventana pre-explosion y galeria de curvas. Escribe una pagina para el atlas."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.utils import DL_calculator
from pipeline78.paths import OC, PHD, STORE
from pipeline78.catalog import CLF_CLASS


def mu(z):
    return 5.0 * np.log10(DL_calculator(float(z)) * 1e6) - 5.0


def _grupo(t):
    """Grupo de comparacion con la muestra real: IIn va aparte, sin contraparte real."""
    return "IIn" if t == "IIn" else CLF_CLASS[t]


def pre_explosion(ph, sims):
    """Por sn_type, sobre sims con al menos 1 deteccion en r: ver docstring del brief (enmienda 2026-10-02)."""
    r = ph[ph["filter"] == "r"].sort_values(["sim_id", "mjd"])
    rows = []
    for sid, d in r.groupby("sim_id", sort=False):
        det = d[d.detected]
        if not len(det):
            continue
        f = det.iloc[0]
        ul = d[(~d.detected) & (d.mjd < f.mjd)]
        prev = ul[ul.mjd >= f.mjd - 60]
        last = ul.iloc[-1] if len(ul) else None
        win = last is not None and last.magnitud_modelo >= 90
        rows.append(dict(sim_id=sid, ul_previo=len(prev) > 0, ul_ventana=bool(win),
                         dt=(f.mjd - last.mjd) if win else np.nan,
                         dm=(last.maglimit - f.magnitud_proyectada) if win else np.nan))
    x = pd.DataFrame(rows).merge(sims[["sim_id", "sn_type"]], on="sim_id")
    out = []
    for t, g in x.groupby("sn_type"):
        w = g[g.ul_ventana]
        out.append(dict(sn_type=t, n_con_det_r=len(g), frac_ul_previo=g.ul_previo.mean(),
                        frac_ul_ventana=g.ul_ventana.mean(), n_ventana=len(w),
                        dt_med=w.dt.median(), dt_p90=w.dt.quantile(0.9), dm_med=w.dm.median()))
    return pd.DataFrame(out).round(3)


def rise_cubierto():
    c = pd.read_csv(STORE / "catalog.csv")
    t = c[["sn", "clase", "t_first", "t_peak"]].copy()
    t["rise_cubierto_d"] = t.t_peak - t.t_first
    t = t.sort_values("rise_cubierto_d").reset_index(drop=True)
    t["corto"] = np.where(t.rise_cubierto_d < 10, "SI (<10 d)", "")
    return t


def report(run_dir, out_dir):
    run_dir, out_dir = Path(run_dir).expanduser(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sims = pd.read_parquet(run_dir / "_sims_all.parquet")
    ph = pd.concat([pd.read_parquet(p) for p in run_dir.glob("*__*.parquet")], ignore_index=True)
    det = ph[ph.detected & (ph["filter"] == "r")]
    peak = det.groupby("sim_id").magnitud_proyectada.min().rename("m_peak_r").reset_index()
    s = sims.merge(peak, on="sim_id", how="left")
    s = s[(s.n_det_r >= 7) & s.m_peak_r.notna()].copy()
    s["M_obs_r"] = [m - mu(z) for m, z in zip(s.m_peak_r, s.z)]
    s["grupo"] = s.sn_type.map(_grupo)
    sims["grupo"] = sims.sn_type.map(_grupo)
    real = pd.read_parquet(OC / "data/real_val.parquet")
    rows = []
    for c in ("Ia", "II", "IIn", "Ibc"):
        a = s.loc[s.grupo == c, "M_obs_r"]
        b = real.loc[real.label == c, "M_peak_r"].dropna() if c != "IIn" else pd.Series(dtype=float)
        rows.append(dict(clase=c, n_sim=len(a), med_sim=a.median(), std_sim=a.std(), n_real=len(b),
                         med_real=b.median(), std_real=b.std(), delta_med=a.median() - b.median()))
    tab = pd.DataFrame(rows).round(3)
    pre = pre_explosion(ph, sims)
    rc = rise_cubierto()
    cols = (("Ia", "C0"), ("II", "C2"), ("IIn", "C1"), ("Ibc", "C3"))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    sims.groupby(["sn_type", "status"]).size().unstack(fill_value=0).plot.bar(ax=ax[0], stacked=True)
    ax[0].set_title("status per projection class")
    for c, col in cols:
        ax[1].hist(s.loc[s.grupo == c, "M_obs_r"], bins=30, histtype="step", color=col, label=f"{c} sim", density=True)
        if c != "IIn":
            ax[1].hist(real.loc[real.label == c, "M_peak_r"].dropna(), bins=20, histtype="stepfilled", alpha=0.2,
                       color=col, label=f"{c} real", density=True)
    ax[1].invert_xaxis(); ax[1].set_xlabel("observed peak M_r"); ax[1].legend(fontsize=7)
    for c, col in cols:
        ax[2].hist(sims.loc[sims.grupo == c, "n_det_r"], bins=40, histtype="step", color=col, label=c)
    ax[2].set_xlabel("detections in r"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "resumen.png", dpi=110); plt.close(fig)
    pick = s.sample(min(12, len(s)), random_state=1).sim_id
    fig, axs = plt.subplots(3, 4, figsize=(13, 8), sharey=False)
    for ax_, sid in zip(axs.ravel(), pick):
        d = ph[ph.sim_id == sid]
        for b, col in (("g", "g"), ("r", "r")):
            x = d[(d["filter"] == b)]
            ax_.errorbar(x.mjd[x.detected], x.magnitud_proyectada[x.detected], x.magerr[x.detected], fmt="o", ms=2, color=col)
            ax_.plot(x.mjd[~x.detected], x.maglimit[~x.detected], "v", ms=2, color=col, alpha=0.4)
        r = sims[sims.sim_id == sid].iloc[0]
        ax_.set_title(f"{r.sn_type} {r.template} z={r.z:.3f}", fontsize=8); ax_.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_dir / "galeria.png", dpi=100); plt.close(fig)
    (out_dir / "index.html").write_text(
        "<html><head><meta charset='utf-8'><title>Piloto ZTF v78</title></head><body style='font-family:sans-serif;max-width:1300px;margin:auto'>"
        f"<h1>Piloto de proyeccion ZTF con las 78 congeladas</h1><p>Corrida: {run_dir}</p>"
        f"<h2>M_peak observado en r: sims (n_det_r &ge; 7) contra la mitad de validacion real</h2>"
        "<p>II compara II+IIb simulados con las II reales. IIn va aparte y no tiene contraparte real.</p>"
        f"{tab.to_html(index=False)}"
        "<img src='resumen.png' width='100%'>"
        "<h2>Ventana pre-explosi&oacute;n</h2>"
        "<p>Un UL de ventana justo antes de una detecci&oacute;n con un dm grande significa que la curva de luz aparece de golpe, porque el template no alcanza la explosi&oacute;n.</p>"
        "<p>Sims con al menos 1 detecci&oacute;n en r. frac_ul_previo: al menos un UL en r dentro de 60 d antes de la primera detecci&oacute;n. "
        "frac_ul_ventana: el &uacute;ltimo UL en r antes de la primera detecci&oacute;n es de la ventana pre-explosi&oacute;n (magnitud_modelo &ge; 90). "
        "dt: primera detecci&oacute;n menos ese UL (d). dm: maglimit del UL menos magnitud proyectada de la detecci&oacute;n.</p>"
        f"{pre.to_html(index=False)}"
        "<h3>Rise cubierto por template (t_peak - t_first, d rest)</h3>"
        f"{rc.to_html(index=False)}"
        "<h2>Galeria (g verde, r rojo; triangulos = limites)</h2>"
        "<img src='galeria.png' width='100%'></body></html>")
    return tab, pre, rc


if __name__ == "__main__":
    tab, pre, rc = report(sys.argv[1], PHD / "paper2_ZTF/figures_templates/pipeline78_piloto_ztf")
    print(tab.to_string(index=False)); print(pre.to_string(index=False)); print(rc.to_string(index=False))
