"""
Chequeo de la proyeccion con las 78 congeladas (2026-10-02). Correr antes de
lanzar un run:  python test_v78.py   (~10 s, lee las 78 copias para el md5)

1. data/<clase>_v78 = lo copiado de series_aprobadas (conteo y md5 contra
   data/templates_v78_md5.csv).
2. Dilatacion (1+z): la curva sintetica dura (1+z) veces lo que dura en reposo,
   y el LOESS no se llama con loess_smooth=False.
3. Ancla: el pivote cae en el maximo de r para cualquier tipo (antes Ibc/IIb/IIn
   dependian de tablas externas de maximo).
"""
import csv, glob, hashlib, io, os, contextlib
import numpy as np
import pandas as pd

import run_per_field as R
from config import TEMPLATE_DIRS, PATHS, PROCESSING_CONFIG, LUMINOSITY_CONFIG
from core.multiband_projection import multiband_field_projection

data = PATHS['data_dir']

# 1. integridad de las copias
rows = list(csv.DictReader(open(os.path.join(data, 'templates_v78_md5.csv'))))
for tipo, carpeta in TEMPLATE_DIRS.items():
    esperadas = {r['sn']: r['md5'] for r in rows if r['clase'] == tipo}
    files = sorted(glob.glob(os.path.join(data, carpeta, '*.dat')))
    assert len(files) == len(esperadas) > 0, f"{tipo}: {len(files)} .dat vs {len(esperadas)} en el csv"
    for f in files:
        sn = os.path.basename(f)[:-4]
        assert hashlib.md5(open(f, 'rb').read()).hexdigest() == esperadas[sn], f"md5 distinto: {f}"
print(f"1 OK: {len(rows)} templates, md5 igual al snapshot")

# 2. dilatacion y LOESS apagado
assert PROCESSING_CONFIG['loess_smooth'] is False
def _no_loess(*a, **k):
    raise AssertionError("Loess_fit se llamo con loess_smooth=False")
R.Loess_fit = _no_loess
path = os.path.join(data, TEMPLATE_DIRS['Ia'], 'SN2011fe.dat')
_, fases_rest = R.leer_spec(path, ot=False, as_pandas=True)
lum = dict(LUMINOSITY_CONFIG, enabled=False)
z = 0.05
with contextlib.redirect_stdout(io.StringIO()):
    curves, _ = R.generate_synthetic_curves('SN2011fe', 'Ia', path, z, 0.0, 0.0,
                                            PATHS['response_folder'], PROCESSING_CONFIG, lum)
span_rest = max(fases_rest) - min(fases_rest)
for f, (ph, _) in curves.items():
    assert abs((ph.max() - ph.min()) - span_rest * (1 + z)) < 1e-6, f"{f}: sin dilatacion"
print(f"2 OK: span {span_rest:.0f} d en reposo -> {span_rest*(1+z):.1f} d observado en {list(curves)}")

# 3. ancla en el maximo de r, tipo sin tabla de maximo (IIn)
t = np.arange(0.0, 100.0)
mag_r = 20 + 0.01 * (t - 37.0) ** 2           # maximo de r en t=37
obs = pd.DataFrame({'oid': 'X', 'mjd': np.arange(1000.0, 1400.0, 3.0), 'filter': 'r', 'maglimit': 21.0})
with contextlib.redirect_stdout(io.StringIO()):
    res = multiband_field_projection({'r': (t, mag_r)}, obs, 'IIn', ['r'], np.arange(-30, 30), 'fake',
                                     selected_field='X', offset_search_mode='deterministic',
                                     n_divisions=10, part_index=4)
assert res['anchor_source'] == 'min_mag_r' and res['anchor_time'] == 37.0, res['anchor_source']
assert abs(res['anchor_obs_mjd'] - res['mjd_pivote']) < 1e-9
print("3 OK: ancla = maximo de r, cae en el pivote")
