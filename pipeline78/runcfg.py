# pipeline78/runcfg.py
"""Configuraciones con nombre. Cambiar una configuracion = nombre nuevo (el hash queda en el manifiesto)."""
from pathlib import Path
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
# Bordes de la Tarea 9: cola + borde fireball en Ia, elegidos en la puerta de realismo 2026-10-03
# (t_rise Ia: base -1.5 sigma, texp +2.6, fireball +0.9; cola acerca t_fall en 4/5 clases)
# Fix H (Mauricio 2026-10-03): ancla al azar en el log del campo (pivot dejaba todas las sims en las mismas fechas),
# UL previos como ALeRCE (solo con una deteccion en los 30 d siguientes) y sin cola en las II de menos de 120 d de
# reposo (SN2024ggi, SN2016esw terminan en el plateau). Lo heredan la grilla det* y ztf_v78_t9.
# Ruido de tres terminos (Mauricio 2026-10-04): sigma^2 = (A 1.0857/(5 10^(0.4 dm)))^2 + (B 10^(-0.2 dm))^2 + C^2,
# dm = m_lim - m_modelo (project.sigma_tres_terminos). Ajuste de pipeline78/calib_ruido.py sobre 18811 detecciones de
# 578 SNe del holdout ZTF val (sin excluidas) con 0 <= dm < 4, m_lim = diffmaglim de cada alerta de ALeRCE
# (data/ruido_alerce_val.csv; el maglim del log en epocas con deteccion es estimado). Reajuste del logfix (2026-10-04):
# las reales ya no traen filas identicas repetidas ni restas negativas (antes 19171 detecciones; g A 0.7738,
# B 0.1644, C 0.0217, r sin cambio en 1e-3). Error bootstrap sobre SNe (300): g A 0.788 +- 0.035, B 0.1621 +- 0.0054,
# C 0.0231 +- 0.0052; r A 0.793 +- 0.019, B 0.1370 +- 0.0037, C 0.0258 +- 0.0026. B difiere entre g y r en 4.1 sigma:
# parametros por banda. i no tiene detecciones reales: usa los de r (supuesto, banda vecina). Medianas por bin dentro
# del 10 % de las reales y el ajuste reproduce estos numeros (tests/test_p78_ruido.py).
NOISE_TRES_TERMINOS = {"g": dict(A=0.7882, B=0.1621, C=0.0231),
                       "r": dict(A=0.7930, B=0.1370, C=0.0258),
                       "i": dict(A=0.7930, B=0.1370, C=0.0258)}
# Log con limites reales (logfix, controlador 2026-10-04): en las epocas con deteccion del objeto del campo el maglim de
# ztf_obslog_best es estimado y recortado a >= m + 0.5. ztf_obslog_alerce lo reemplaza por el diffmaglim de la alerta
# de ALeRCE (pipeline78/obslog_alerce.py). Solo los 1000 campos de ztf_fields_1000.txt. ztf_v78 sigue con el log viejo.
LOG_ALERCE = str(Path.home() / "thesis_runs/obslog/ztf_obslog_alerce.parquet")
RUNS_CFG["ztf_v78_t9_bordes"] = dict(RUNS_CFG["ztf_v78"], edge_pre="fireball", rise_Ia_days=18.9, edge_post="tail",
                                     tail_days=150, tail_fit_days=20, tail_min_slope=0.005,
                                     anchor="uniform", pre_ul_mode="alerce", pre_ul_days=30, tail_min_span={"II": 120.0},
                                     noise_model="tres_terminos", noise_params=NOISE_TRES_TERMINOS,
                                     log_path=LOG_ALERCE)

# Calibracion de la eficiencia de deteccion (Fix G, 2026-10-03): logistica sobre el S/N medido, sin UL tras la ultima
# deteccion y la misma limpieza que las reales.
for m0 in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0):
    RUNS_CFG[f"ztf_v78_t9_det{m0}"] = dict(RUNS_CFG["ztf_v78_t9_bordes"], det_model="logistic", det_m0=m0, det_w=0.2,
                                          ul_after_last=False, lc_clean=True)
    # w angosto: casi un corte en S/N medido, corrido m0 mag
    RUNS_CFG[f"ztf_v78_t9_det{m0}_w05"] = dict(RUNS_CFG[f"ztf_v78_t9_det{m0}"], det_w=0.05)

# Configuracion de la Tarea 9 (Mauricio 2026-10-03): bordes de arriba (fireball+cola, ancla uniforme, UL como ALeRCE,
# sin cola en II que terminan en el plateau) + eficiencia con la forma medida por DES (Kessler+2015: 50% a S/N 5,
# ~100% a S/N 10 -> w=0.2 mag) desplazada m0=1.25 mag, calibrada con la duracion de las curvas del holdout ZTF val limpio
# (ultima det r - pico: II 70.9/70.8, IIb 53.0/50.0, Ibc 49.1/47.4, Ia 36.8/42.0 d; IIn 113.8/79.9 por plantillas
# longevas), procedimiento de Kessler+2019. Calibracion efectiva (host, filtros de alertas), no medicion de ZTF.
RUNS_CFG["ztf_v78_t9"] = dict(RUNS_CFG["ztf_v78_t9_det1.25"])
# el doble de sims por campo y clase: con la misma semilla las k < n de ztf_v78_t9 salen identicas (rng por
# (semilla, campo, clase, k) y ancla uniforme que no depende de n), asi que solo hay que extraer las nuevas
RUNS_CFG["ztf_v78_t9_x2"] = dict(RUNS_CFG["ztf_v78_t9"], n_by_class={c: 2 * n for c, n in RUNS_CFG["ztf_v78_t9"]["n_by_class"].items()})

RUNS_CFG["ztf_v78_t9_iidust"] = dict(RUNS_CFG["ztf_v78_t9"], ii_dust="sudare")   # variante de sistematico: polvo de SUDARE I en II

RUNS_CFG["ztf_v78_t9_iinnyholm"] = dict(RUNS_CFG["ztf_v78_t9"], iin_lf="nyholm")   # variante de sistematico: LF de IIn de Nyholm+2020

def units(cfg, fields):
    n = max(cfg["n_by_class"].values())
    step = cfg["chunk"] or n
    return [(f, a, min(a + step, n)) for f in fields for a in range(0, n, step)]


def log_path(cfg):
    if "log_path" in cfg:
        return cfg["log_path"]
    return str(STORE / {"ZTF": "ztf_obslog_best.parquet", "SUDARE": "sudare_obslog.parquet"}[cfg["survey"]])
