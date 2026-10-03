# pipeline78/runcfg.py
"""Configuraciones con nombre. Cambiar una configuracion = nombre nuevo (el hash queda en el manifiesto)."""
from pipeline78.paths import STORE

RUNS_CFG = {
    "ztf_v78": dict(
        survey="ZTF", bands=["g", "r", "i"], classes=["Ia", "II", "IIb", "IIn", "Ibc"],
        n_by_class={"Ia": 10, "II": 8, "IIb": 2, "IIn": 10, "Ibc": 10}, chunk=None,   # D1
        anchor="pivot", z_mode="uniform_weighted", zmin=0.005,
        # horizonte generoso: la seleccion S(m) se aplica al entrenar con pesos; valores revisables tras la tabla de LF
        zmax_by_class={"Ia": 0.25, "II": 0.15, "IIb": 0.15, "IIn": 0.35, "Ibc": 0.25},
        mw_mode="ztf_sfd", mw_const=0.02, rule="ztf",
        pre_ul_days=25.0, noise_k=5.0, sigma_floor=0.02),
}
# Puerta de realismo: variantes de borde de ztf_v78 (sin estas claves, project_one usa window/none)
RUNS_CFG["ztf_v78_texp"] = dict(RUNS_CFG["ztf_v78"], edge_pre="texp",
                                rise_Ia_days=18.9)       # Miller+2020 2020ApJ...902...47M: rise medio 18.9 d hasta el maximo en B
RUNS_CFG["ztf_v78_tail"] = dict(RUNS_CFG["ztf_v78"], edge_post="tail", tail_days=150, tail_fit_days=20,
                                tail_min_slope=0.005)    # cola lineal en magnitud declarada, supuesto, no medicion
RUNS_CFG["ztf_v78_fireball"] = dict(RUNS_CFG["ztf_v78"], edge_pre="fireball",
                                    rise_Ia_days=18.9)   # Miller+2020: alfa_r=2.01, rise 18.9 d; mismo supuesto (t-texpl)^2 que cap 3 para las otras clases
RUNS_CFG["ztf_v78_t9"] = dict(RUNS_CFG["ztf_v78"], edge_pre="fireball", rise_Ia_days=18.9, edge_post="tail",
                              tail_days=150, tail_fit_days=20,
                              tail_min_slope=0.005)      # configuracion de la Tarea 9: cola + borde fireball en Ia, elegidos en la puerta de realismo 2026-10-03 (t_rise Ia: base -1.5 sigma, texp +2.6, fireball +0.9; cola acerca t_fall en 4/5 clases)

RUNS_CFG["ztf_v78_t9_iidust"] = dict(RUNS_CFG["ztf_v78_t9"], ii_dust="sudare")   # variante de sistematico: polvo de SUDARE I en II

RUNS_CFG["ztf_v78_t9_iinnyholm"] = dict(RUNS_CFG["ztf_v78_t9"], iin_lf="nyholm")   # variante de sistematico: LF de IIn de Nyholm+2020

# Calibracion de la eficiencia de deteccion (Fix G, 2026-10-03): logistica en S/N, sin UL tras la ultima deteccion y
# la misma limpieza que las reales. El controlador elige m0 y fija la config T9.
for m0 in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0):
    RUNS_CFG[f"ztf_v78_t9_det{m0}"] = dict(RUNS_CFG["ztf_v78_t9"], det_model="logistic", det_m0=m0, det_w=0.2,
                                          ul_after_last=False, lc_clean=True)
    # w angosto: casi un corte en S/N medido (las alertas de ZTF exigen S/N >= 5), corrido m0 mag
    RUNS_CFG[f"ztf_v78_t9_det{m0}_w05"] = dict(RUNS_CFG[f"ztf_v78_t9_det{m0}"], det_w=0.05)


def units(cfg, fields):
    n = max(cfg["n_by_class"].values())
    step = cfg["chunk"] or n
    return [(f, a, min(a + step, n)) for f in fields for a in range(0, n, step)]


def log_path(cfg):
    if "log_path" in cfg:
        return cfg["log_path"]
    return str(STORE / {"ZTF": "ztf_obslog_best.parquet", "SUDARE": "sudare_obslog.parquet"}[cfg["survey"]])
