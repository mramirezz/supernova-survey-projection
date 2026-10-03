# pipeline78/lcclean.py
"""Limpieza comun de curvas (reales y sims): solo la SN (decision 2026-10-03, Fix G).
ALeRCE junta todas las detecciones de ZTF en esa posicion, de todos los anios (fuentes planas, AGN, puntos sueltos).
(a) ventana [t_ref - PRE, t_ref + POST]: detecciones y UL, todas las bandas.
(b) las detecciones (todas las bandas, por mjd) se cortan en grupos donde el salto es > GAP d. Se borran las
    detecciones de los grupos con menos de MIN_GROUP, salvo el grupo del maximo de brillo en r (en g si no hay r).
    Los UL no se tocan en (b)."""
import numpy as np

PRE, POST = 50.0, 400.0     # desde el descubrimiento: una IIn de un anio cabe en la ventana
GAP, MIN_GROUP = 60.0, 3    # un grupo grande en la temporada siguiente (IIn) se conserva


def _span(mjd):
    return float(mjd.max() - mjd.min()) if len(mjd) else 0.0


def clean_lc(df, t_ref, pre=PRE, post=POST, gap=GAP, min_group=MIN_GROUP):
    """Esquema de proyeccion (mjd, filter, upperlimit 'T'/'F', magnitud_proyectada). Devuelve (df limpio, resumen)."""
    mjd = df["mjd"].to_numpy(float)
    up = df["upperlimit"].to_numpy() == "F"
    keep = (mjd >= t_ref - pre) & (mjd <= t_ref + post)
    i = np.flatnonzero(keep & up)
    i = i[np.argsort(mjd[i], kind="stable")]
    malos = []
    if len(i):
        grupo = np.concatenate([[0], np.cumsum(np.diff(mjd[i]) > gap)])
        f, mag = df["filter"].to_numpy()[i], df["magnitud_proyectada"].to_numpy(float)[i]
        pico = -1
        for b in ("r", "g"):
            k = np.flatnonzero(f == b)
            if len(k):
                pico = grupo[k[np.argmin(mag[k])]]
                break
        n = np.bincount(grupo)
        malos = [g for g in range(len(n)) if n[g] < min_group and g != pico]
        keep[i[np.isin(grupo, malos)]] = False
    out = df[keep].reset_index(drop=True)
    return out, dict(n_filas_antes=len(df), n_filas_despues=len(out), n_grupos_eliminados=len(malos),
                     span_det_antes=_span(mjd[up]), span_det_despues=_span(mjd[keep & up]))
