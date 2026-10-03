# pipeline78/lcclean.py
"""Limpieza comun de curvas (reales y sims): solo la SN (decision 2026-10-03, Fix G). "Todo lo que no es supernova
tienes que sacarlo." ALeRCE junta todas las detecciones de ZTF en esa posicion, de todos los anios (fuentes planas,
AGN, puntos sueltos). Ninguna regla usa la clase espectroscopica: sirve igual para objetos sin clasificar.
(a) Ventana [t_ref - PRE, t_ref + POST]: detecciones y UL, todas las bandas.
(b) Las detecciones en g y r (las reales no tienen i), por mjd, se cortan en grupos donde el salto es > GAP d. El
    tamanio de un grupo cuenta epocas (mjd, banda) distintas: ALeRCE repite detecciones. Los UL y las filas en i solo
    pasan por la ventana.
    1. Grupo del pico: el de la deteccion mas brillante en r (en g si no hay r) entre los grupos con >= MIN_GROUP
       detecciones que empiezan antes de t_ref + TARDIA. La SN es, por definicion, lo que TNS descubrio: el grupo del
       descubrimiento la ancla y un grupo posterior apenas mas brillante (re-brillo, AGN, fuente plana) no lo desplaza.
       Si no hay, entre todos los grupos con >= MIN_GROUP, y si ninguno llega (curvas de 1-2 detecciones), entre todos.
       Un punto suelto brillante no ancla el pico.
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
TARDIA = 30.0               # d despues de t_ref: ancla del grupo del pico y bandera primera_det_tardia


def _span(mjd):
    return float(mjd.max() - mjd.min()) if len(mjd) else 0.0


def _ref_band(fa, fb):
    for b in ("r", "g"):
        if (fa == b).any() and (fb == b).any():
            return b
    return None


def clean_lc(df, t_ref, pre=PRE, post=POST, gap=GAP, min_group=MIN_GROUP, fade_min=FADE_MIN, tardia=TARDIA):
    """Esquema de proyeccion (mjd, filter, upperlimit 'T'/'F', magnitud_proyectada). Devuelve (df limpio, resumen)."""
    mjd, fil = df["mjd"].to_numpy(float), df["filter"].to_numpy()
    up = (df["upperlimit"].to_numpy() == "F") & np.isin(fil, ("g", "r"))     # detecciones en g y r
    keep = (mjd >= t_ref - pre) & (mjd <= t_ref + post)
    i = np.flatnonzero(keep & up)
    i = i[np.lexsort((fil[i] == "r", mjd[i]))]                 # por mjd; los duplicados (mjd, banda) quedan juntos
    fuera = dict(n_grupos_antes_pico=0, n_grupos_chicos=0, n_grupos_no_bajan=0)
    if len(i):
        grupo = np.concatenate([[0], np.cumsum(np.diff(mjd[i]) > gap)])
        f, mag = fil[i], df["magnitud_proyectada"].to_numpy(float)[i]
        nueva = np.r_[True, (np.diff(mjd[i]) != 0) | (f[1:] != f[:-1])]
        n = np.bincount(grupo, weights=nueva).astype(int)         # epocas distintas por grupo
        ini = mjd[i][np.r_[0, np.flatnonzero(np.diff(grupo)) + 1]]          # primera deteccion de cada grupo
        for ok in ((n >= min_group) & (ini <= t_ref + tardia), n >= min_group, np.ones(len(n), bool)):
            if ok.any():
                break
        cand = ok[grupo]
        for b in ("r", "g"):
            k = np.flatnonzero(cand & (f == b))
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
