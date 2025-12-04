"""
Test rápido de proyección multi-banda - DIRECTO
"""
import numpy as np
import pandas as pd
from core.utils import leer_spec, Syntetic_photometry_v2, Loess_fit, maximo_lc, cteg, cter, ctei
from core.correction import correct_redeening
from core.multiband_projection import multiband_field_projection
import os

print("="*60)
print("TEST: Proyección Multi-Banda SN1999gi (DIRECTO)")
print("="*60)

# Parámetros
sn_name = "SN1999gi"
tipo = "II"
z_proy = 0.03
ebmv_host = 0.05
ebmv_mw = 0.03
available_filters = ['g', 'r', 'i']
path_spec = 'data/II/SN1999gi.dat'
path_response_folder = r'G:\Mi unidad\Work\Universidad\Phd\Practica2\Splines_eachfilter_2'
response_files = {
    'g': "spline_g'.txt",
    'r': "spline_r'.txt",
    'i': "spline_i'.txt"
}
FILTER_CONSTANTS = {'g': cteg, 'r': cter, 'i': ctei}

# 1. Leer espectro
print("\n1. Lectura de espectro")
ESPECTRO, fases = leer_spec(path_spec, ot=False, as_pandas=True)
print(f"   Espectros: {len(ESPECTRO)}, Fases: {len(fases)}")

# 2. Correcciones
print("\n2. Correcciones cosmológicas")
ESPECTRO_corr, fases_corr = correct_redeening(
    sn=sn_name, ESPECTRO=ESPECTRO, fases=fases,
    z=z_proy, ebmv_host=ebmv_host, ebmv_mw=ebmv_mw, 
    reverse=True, use_DL=True
)

# 3. Generar curvas sintéticas para cada filtro
print("\n3. Generando curvas sintéticas multi-banda")
curves_by_filter = {}

for filt in available_filters:
    print(f"   Procesando filtro {filt}...")
    
    # Curva de respuesta
    path_response = os.path.join(path_response_folder, response_files[filt])
    response_df = pd.read_csv(path_response, sep='\s+', comment='#', header=None)
    response_df.columns = ['wave', 'response']
    
    # Fotometría sintética
    fases_lc, fluxes_lc, porcentaje_lc = [], [], []
    for spec, fase in zip(ESPECTRO_corr, fases_corr):
        flux, porcentaje = Syntetic_photometry_v2(
            spec['wave'].values, spec['flux'].values,
            response_df['wave'].values, response_df['response'].values
        )
        if porcentaje > 0.95:
            fases_lc.append(fase)
            fluxes_lc.append(flux)
            porcentaje_lc.append(porcentaje)

    lc_df = pd.DataFrame({'fase': fases_lc, 'flux': fluxes_lc})
    
    # Calibración
    mul = FILTER_CONSTANTS[filt]
    flux_calibrado = np.array(lc_df['flux']) / mul
    mag = -2.5 * np.log10(np.clip(flux_calibrado, 1e-20, None))
    
    # Ruido
    flux_from_mag = 10 ** (-0.4 * mag)
    minimo_flux = np.min(flux_from_mag)
    flux_norm = flux_from_mag / minimo_flux
    flux_noisy_norm = np.random.normal(loc=flux_norm, scale=np.sqrt(np.abs(flux_norm)) * 0.15)
    flux_noisy = flux_noisy_norm * minimo_flux
    flux_noisy = np.clip(flux_noisy, 1e-20, None)
    mag_noisy = -2.5 * np.log10(flux_noisy)
    
    curves_by_filter[filt] = (np.array(lc_df['fase']), mag_noisy)
    print(f"      {len(lc_df)} puntos, mag range: {mag.min():.2f} - {mag.max():.2f}")

# 4. Proyección multi-banda
print("\n4. Proyección MULTI-BANDA")
df_obslog = pd.read_csv('data/grid_diffmaglim_ZTF.csv')

# Seleccionar OID al azar
available_oids = df_obslog['oid'].unique()
selected_oid = np.random.choice(available_oids)
print(f"   OID seleccionado: {selected_oid}")

result = multiband_field_projection(
    curves_by_filter=curves_by_filter,
    df_obslog=df_obslog,
    tipo=tipo,
    available_filters=available_filters,
    offset=np.arange(-30, 30, 1),
    sn=sn_name,
    selected_field=selected_oid,
    plot=False
)

print("\n5. RESULTADOS")
print(f"   Total proyectado: {result['n_observations']}")
print(f"   Offset usado: {result['offset_used']}")
if result['n_observations'] > 0:
    df_proj = result['projections']
    for filt in available_filters:
        df_filt = df_proj[df_proj['filter'] == filt]
        if len(df_filt) > 0:
            dets = (df_filt['upperlimit'] == 'F').sum()
            print(f"   Filtro {filt}: {len(df_filt)} obs ({dets} detecciones)")
            
print("\n✅ TEST COMPLETADO")
