# pipeline78/lcclean.py
"""Limpieza comun de curvas (reales y sims): solo la SN (decision 2026-10-03, Fix G). "Todo lo que no es supernova
tienes que sacarlo." ALeRCE junta todas las detecciones de ZTF en esa posicion, de todos los anios (fuentes planas,
AGN, puntos sueltos). Ninguna regla usa la clase espectroscopica: sirve igual para objetos sin clasificar.
(a) Ventana [t_ref - PRE, t_ref + POST]: detecciones y UL, todas las bandas.
(b) Las detecciones (todas las bandas, por mjd) se cortan en grupos donde el salto es > GAP d. Los UL no se tocan.
    1. Grupo del pico: el de la deteccion mas brillante en r (en g si no hay r) entre los grupos con >= MIN_GROUP
       detecciones (si ninguno llega, entre todos). Un punto suelto brillante no ancla el pico.
    2. Antes del pico: fuera todos los grupos. Un salto > GAP antes del pico es otra cosa.
    3. Despues del pico, en orden: fuera los grupos con < MIN_GROUP. Un grupo grande se queda solo si la SN sigue
       bajando a traves del salto: la mediana de su banda de referencia tiene que ser al menos FADE_MIN mas debil que la
       mediana de las ultimas 3 detecciones en esa banda del grupo anterior que se quedo. Banda de referencia: r si
       esta en los dos grupos, si no g. Si no baja (o no hay banda comun), fuera ese grupo y todos los siguientes.
       Saca la fuente plana despues de una SN que ya bajo (2018hrt) y conserva la cola de la temporada siguiente.
(c) primera_det_tardia: la primera deteccion que queda esta > TARDIA d despues de t_ref (SN descubierta antes de las
    alertas publicas de ZTF: la curva no tiene la fase principal). Solo se marca."""
import numpy as np

PRE, POST = 50.0, 400.0     # desde el descubrimiento: una IIn de un anio cabe en la ventana
GAP, MIN_GROUP = 60.0, 3
FADE_MIN = 0.0              # mag: un grupo posterior tiene que estar al menos asi de mas debil
TARDIA = 30.0               # d despues de t_ref


def _span(mjd):
    return float(mjd.max() - mjd.min()) if len(mjd) else 0.0


def _ref_band(fa, fb):
    for b in ("r", "g"):
        if (fa == b).any() and (fb == b).any():
            return b
    return None


def clean_lc(df, t_ref, pre=PRE, post=POST, gap=GAP, min_group=MIN_GROUP, fade_min=FADE_MIN, tardia=TARDIA):
    """Esquema de proyeccion (mjd, filter, upperlimit 'T'/'F', magnitud_proyectada). Devuelve (df limpio, resumen)."""
    mjd = df["mjd"].to_numpy(float)
    up = df["upperlimit"].to_numpy() == "F"
    keep = (mjd >= t_ref - pre) & (mjd <= t_ref + post)
    i = np.flatnonzero(keep & up)
    i = i[np.argsort(mjd[i], kind="stable")]
    fuera = dict(n_grupos_antes_pico=0, n_grupos_chicos=0, n_grupos_no_bajan=0)
    if len(i):
        grupo = np.concatenate([[0], np.cumsum(np.diff(mjd[i]) > gap)])
        f, mag = df["filter"].to_numpy()[i], df["magnitud_proyectada"].to_numpy(float)[i]
        n = np.bincount(grupo)
        cand = n[grupo] >= min_group if (n >= min_group).any() else np.ones(len(i), bool)
        for b in ("r", "g", None):                          # None: cualquier banda
            k = np.flatnonzero(cand & ((f == b) if b else True))
            if len(k):
                break
        pico = grupo[k[np.argmin(mag[k])]]
        sale = np.arange(len(n)) < pico
        fuera["n_grupos_antes_pico"] = int(pico)
        prev = pico
        for g in range(pico + 1, len(n)):
            if n[g] < min_group:
                sale[g] = True
                fuera["n_grupos_chicos"] += 1
                continue
            a, c = grupo == prev, grupo == g
            b = _ref_band(f[a], f[c])
            if b is None or np.median(mag[c & (f == b)]) - np.median(mag[a & (f == b)][-3:]) < fade_min:
                sale[g:] = True
                fuera["n_grupos_no_bajan"] = len(n) - g
                break
            prev = g
        keep[i[sale[grupo]]] = False
    out = df[keep].reset_index(drop=True)
    det = mjd[keep & up]
    dt0 = float(det.min() - t_ref) if len(det) else np.nan
    return out, dict(n_filas_antes=len(df), n_filas_despues=len(out), n_grupos_eliminados=sum(fuera.values()), **fuera,
                     span_det_antes=_span(mjd[up]), span_det_despues=_span(det), dt_primera_det=dt0,
                     primera_det_tardia=bool(dt0 > tardia))
