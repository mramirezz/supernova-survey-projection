#!/usr/bin/env python
"""
Precompute Delta-m15(B) de cada template Ia, en rest-frame, desde su SED crudo.

Se usa para la normalizacion de luminosidad *consciente de Phillips*: en vez de
sortear M_peak independiente de la forma, se ancla al ancho del template
(templates que caen rapido = mas debiles). Ver LUMINOSITY_CONFIG / PHILLIPS_CONFIG
en config.py y el bloque de normalizacion en run_per_field.py.

Delta-m15(B) = B(peak + 15 d) - B(peak), medido sintetizando la banda Bessell B
sobre la serie espectral rest-frame del template (data/Ia/*.dat, ya deenrojecidos).
El cero de magnitud es arbitrario porque solo importa la diferencia.

Salida: data/dm15_Ia.json  ->  {"SN2011fe.dat": 1.10, ...}

Uso:  python precompute_dm15_Ia.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd

from config import PATHS, RESPONSE_FILES, TEMPLATE_DIRS
from core.utils import leer_spec, Syntetic_photometry_v2

OVERLAP_MIN = 0.95   # mismo umbral de solapamiento espectro-filtro que el pipeline
DELTA_T = 15.0       # dias post-peak para Delta-m15


def synth_B_lightcurve(path_spec, Bw, Br):
    """Curva de luz sintetica en Bessell B (fase, mag) del template rest-frame."""
    ESPECTRO, fases = leer_spec(path_spec, ot=False, as_pandas=True)
    ph, mg = [], []
    for spec, fase in zip(ESPECTRO, fases):
        flux, pct = Syntetic_photometry_v2(
            spec['wave'].values, spec['flux'].values, Bw, Br
        )
        if pct > OVERLAP_MIN and flux > 0:
            ph.append(float(fase))
            mg.append(-2.5 * np.log10(flux))
    if len(ph) < 3:
        return None, None
    order = np.argsort(ph)
    return np.array(ph)[order], np.array(mg)[order]


def measure_dm15(ph, mg):
    """Delta-m15(B). Peak = minimo de mag sobre grilla densa.

    None si la serie no cubre peak+15 d, O si el peak cae en el borde inicial
    (serie que empieza en/despues del maximo -> dm15 medido desde un punto
    post-peak, sesgado alto). Guard simetrico al de cobertura final.
    """
    grid = np.linspace(ph.min(), ph.max(), 4000)
    mgi = np.interp(grid, ph, mg)
    ipk = int(np.argmin(mgi))
    fpk, mpk = float(grid[ipk]), float(mgi[ipk])
    if fpk <= ph.min() + 1.0:   # peak pegado al inicio: no hay rise que lo restrinja
        return None, fpk
    if ph.max() < fpk + DELTA_T:
        return None, fpk
    m15 = float(np.interp(fpk + DELTA_T, ph, mg))
    return m15 - mpk, fpk


def main():
    resp_path = os.path.join(PATHS['response_folder'], RESPONSE_FILES['B'])
    rdf = pd.read_csv(resp_path, sep=r'\s+', comment='#', header=None)
    Bw, Br = rdf[0].values, rdf[1].values

    data_ia = os.path.join(PATHS['data_dir'], TEMPLATE_DIRS['Ia'])
    files = sorted(glob.glob(os.path.join(data_ia, '*.dat')))
    print(f"Templates Ia: {len(files)}  (filtro B: {RESPONSE_FILES['B']})\n")

    out = {}
    for f in files:
        name = os.path.basename(f)
        ph, mg = synth_B_lightcurve(f, Bw, Br)
        if ph is None:
            print(f"  {name:16s}  sin fotometria B valida -> skip")
            continue
        dm15, fpk = measure_dm15(ph, mg)
        if dm15 is None:
            reason = ("peak en el borde inicial (sin rise)" if fpk <= ph.min() + 1.0
                      else "no llega a peak+15")
            print(f"  {name:16s}  {reason} (peak@{fpk:+.1f}, "
                  f"cobertura [{ph.min():+.1f},{ph.max():+.1f}]) -> skip")
            continue
        out[name] = round(float(dm15), 4)
        print(f"  {name:16s}  dm15(B)={dm15:5.2f}   peak@{fpk:+5.1f}d   "
              f"cobertura [{ph.min():+.0f},{ph.max():+.0f}]")

    vals = np.array(list(out.values()))
    print(f"\n{len(out)}/{len(files)} templates con dm15.")
    if len(vals):
        print(f"rango {vals.min():.2f}-{vals.max():.2f}, mediana {np.median(vals):.2f}")
        # sanity fisico: Ia normales caen en ~0.8-1.8; permito 0.5-2.2
        assert np.all((vals > 0.5) & (vals < 2.2)), \
            f"dm15 fuera de rango fisico: {out}"

    out_path = os.path.join(PATHS['data_dir'], 'dm15_Ia.json')
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nGuardado: {out_path}")


if __name__ == '__main__':
    main()
