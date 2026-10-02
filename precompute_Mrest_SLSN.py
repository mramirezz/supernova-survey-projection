#!/usr/bin/env python
"""
Precompute M_peak REST-FRAME en banda g de cada template SLSN-I, desde su SED
a 10 pc (data/<TEMPLATE_DIRS['SLSN-I']>/*.dat, ya deenrojecidos y a escala
absoluta por el pipeline s6-s8).

Se usa para el anclaje rest-frame de la normalizacion de luminosidad
(LUMINOSITY_CONFIG['rest_anchor']): la M de literatura de las SLSN-I esta
definida en g rest con K-correction (Chen+2023, 2023ApJ...943...41C), asi que
el shift de normalizacion se calcula en el rest frame y la K-correction la
hereda el SED del template. Ver proyeccion_IIb_SLSN_literatura.md.

A diferencia del dm15 (que solo usa diferencias), aca el cero SI importa: se
calibra con la misma FILTER_CONSTANTS['g'] del pipeline de proyeccion, para que
el M medido viva en el mismo sistema que las curvas sinteticas.

Salida: data/Mrest_g_SLSN-I.json  ->  {"PTF12dam.dat": -21.6, ...}

Uso:  python precompute_Mrest_SLSN.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd

from config import PATHS, RESPONSE_FILES, TEMPLATE_DIRS
# cteg directo de core.utils (NO importar run_per_field: exige que el json que
# este script genera ya exista -> circulo)
from core.utils import leer_spec, Syntetic_photometry_v2, cteg

OVERLAP_MIN = 0.95
BAND = "g"
FILTER_CONSTANTS = {"g": cteg}


def main():
    resp_path = os.path.join(PATHS['response_folder'], RESPONSE_FILES[BAND])
    rdf = pd.read_csv(resp_path, sep=r'\s+', comment='#', header=None)
    gw, gr = rdf[0].values, rdf[1].values

    data_dir = os.path.join(PATHS['data_dir'], TEMPLATE_DIRS['SLSN-I'])
    files = sorted(glob.glob(os.path.join(data_dir, '*.dat')))
    print(f"Templates SLSN-I: {len(files)}  (banda {BAND}: {RESPONSE_FILES[BAND]})\n")

    out = {}
    for f in files:
        name = os.path.basename(f)
        ESPECTRO, fases = leer_spec(f, ot=False, as_pandas=True)
        mags = []
        for spec, fase in zip(ESPECTRO, fases):
            flux, pct = Syntetic_photometry_v2(
                spec['wave'].values, spec['flux'].values, gw, gr)
            if pct > OVERLAP_MIN and flux > 0:
                mags.append(-2.5 * np.log10(flux / FILTER_CONSTANTS[BAND]))
        if len(mags) < 3:
            print(f"  {name:18s}  sin fotometria {BAND} valida -> skip")
            continue
        m = float(np.min(mags))
        out[name] = round(m, 4)
        print(f"  {name:18s}  M_{BAND},rest = {m:7.2f}  ({len(mags)} epocas)")

    vals = np.array(list(out.values()))
    print(f"\n{len(out)}/{len(files)} templates con M rest.")
    if len(vals):
        print(f"rango {vals.min():.2f} a {vals.max():.2f}, mediana {np.median(vals):.2f}")
        # sanity fisico: SLSN-I observadas en ZTF-I caen en -19.8 a -22.8
        # (Chen+2023); el template puede quedar algo fuera (es solo la forma,
        # se renormaliza), pero un M positivo o ~-15 delata un error de escala.
        assert np.all((vals < -18.0) & (vals > -25.0)), \
            f"M rest fuera de rango fisico para SLSN: {out}"

    out_path = os.path.join(PATHS['data_dir'], 'Mrest_g_SLSN-I.json')
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nGuardado: {out_path}")


if __name__ == '__main__':
    main()
