"""
RUNNER POR CAMPO — 10 SIMULACIONES DETERMINÍSTICAS POR OID Y TIPO
=================================================================

Para cada campo (OID) del observing log, genera 10 curvas de luz por tipo:
  - tipos de config.TEMPLATE_DIRS (o --types) × 10 posiciones de pivote determinísticas

Cada simulación:
  1. Elige un template (cíclico entre los disponibles por tipo)
  2. Muestrea z volume-weighted
  3. Muestrea E(B-V)_host por tipo
  4. Genera curvas sintéticas multi-banda (AVAILABLE_FILTERS), con dilatación (1+z)
  5. Proyecta sobre la grilla real del campo con pivote determinístico
     (grilla dividida en 10 partes, pivote al centro de cada partición)

Uso:
    python run_per_field.py                      # Todos los OIDs
    python run_per_field.py --oid ZTF18aaqeasu   # Un OID específico
    python run_per_field.py --n-fields 10        # Primeros 10 OIDs
    python run_per_field.py --min-obs 50         # OIDs con >=50 observaciones
"""

import os
import sys
import glob
import argparse
import time
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import (PATHS, RESPONSE_FILES, PROCESSING_CONFIG, LUMINOSITY_CONFIG,
                    Z_CONFIG, EXTINCTION_CONFIG, SURVEY, SN_WHITELIST, PHILLIPS_CONFIG,
                    TEMPLATE_DIRS)
import subprocess
from config_loader import load_and_validate_config
from core.utils import (
    leer_spec, Syntetic_photometry_v2, Loess_fit, DL_calculator,
    cteB, cteV, cteR, cteI, cteU, cteu, cteg, cter, ctei, ctez
)
from core.correction import (
    correct_redeening, sample_extinction_by_type, sample_cosmological_redshift,
    rv_host_for_type
)
from core.multiband_projection import multiband_field_projection
from tools.dust_maps import get_sfd98_extinction_real
import math

# Constantes fotométricas por filtro
FILTER_CONSTANTS = {
    'B': cteB, 'V': cteV, 'R': cteR, 'I': cteI, 'U': cteU,
    'u': cteu, 'g': cteg, 'r': cter, 'i': ctei, 'z': ctez
}

# Filtros ZTF. Solo g y r (2026-10-02): la i es el 0.6 % del log (31 118 de
# 5.0 M epocas, solo campos de partnership) y el clasificador usa g y r. Sacarla
# ahorra ~1/3 de la fotometria sintetica. SUDARE necesitara g, r, i.
AVAILABLE_FILTERS = ['g', 'r']
N_DIVISIONS = 10  # particiones determinísticas

# z empírico (run de ENTRENAMIENTO): distribuciones de z reales por tipo, si existen.
# Se usan solo cuando Z_CONFIG['z_mode'] == 'empirical'.
Z_EMPIRICAL = {}
for _t in TEMPLATE_DIRS:
    _zf = os.path.join(PATHS['data_dir'], f'z_empirical_{_t}.txt')
    if os.path.exists(_zf):
        Z_EMPIRICAL[_t] = np.loadtxt(_zf)

# M_peak empírico (run de ENTRENAMIENTO): M ajustados de la muestra real por tipo.
# Se usan solo cuando LUMINOSITY_CONFIG['M_mode'] == 'empirical' (o --m-mode empirical).
# Paridad con Z_EMPIRICAL. Ver auditoría banda g (bitácora Proyección 2026-08-02).
M_EMPIRICAL = {}
for _t in TEMPLATE_DIRS:
    _mf = os.path.join(PATHS['data_dir'], f'M_empirical_{_t}.txt')
    if os.path.exists(_mf):
        M_EMPIRICAL[_t] = np.loadtxt(_mf)

# dm15(B) por template Ia, para la relacion ancho-luminosidad (Phillips).
# Precomputado con precompute_dm15_Ia.py. Se indexa por basename del .dat.
# Si Phillips esta activo y el json falta, FALLAR RUIDOSO: el fallback silencioso
# dejaria todas las Ia en dm15_default (WLR apagada + LF angostada a sigma_resid)
# y el run pareceria exitoso.
DM15_IA = {}
_DM15_WARNED = set()  # templates ya avisados por fallback a dm15_default
_dm15_path = os.path.join(PATHS['data_dir'], PHILLIPS_CONFIG.get('dm15_file', 'dm15_Ia.json'))
if PHILLIPS_CONFIG.get('enabled', False):
    if not os.path.exists(_dm15_path):
        raise FileNotFoundError(
            f"PHILLIPS_CONFIG['enabled']=True pero falta {_dm15_path}. "
            f"Generalo con: python precompute_dm15_Ia.py "
            f"(sin el, todas las Ia caerian a dm15_default={PHILLIPS_CONFIG.get('dm15_default')} "
            f"y la relacion ancho-luminosidad quedaria apagada en silencio).")
    with open(_dm15_path) as _fh:
        DM15_IA = json.load(_fh)

# Anclaje REST-FRAME por tipo (LUMINOSITY_CONFIG['rest_anchor']): M del template
# en la banda rest de la calibracion de literatura (SLSN-I: g rest, Chen+2023).
# Igual que Phillips: si esta configurado y falta el json, se aborta en vez de
# degradar en silencio al anclaje observado.
M_REST = {}
_MREST_WARNED = set()
for _tipo, _ra in LUMINOSITY_CONFIG.get('rest_anchor', {}).items():
    if _tipo not in TEMPLATE_DIRS:
        continue
    _p = os.path.join(PATHS['data_dir'], _ra['file'])
    if not os.path.exists(_p):
        raise FileNotFoundError(
            f"LUMINOSITY_CONFIG['rest_anchor'] pide {_p} para {_tipo}. "
            f"Generalo con: python precompute_Mrest_SLSN.py")
    with open(_p) as _fh:
        M_REST[_tipo] = json.load(_fh)


def scan_templates(data_dir):
    """Escanea templates disponibles por tipo. Aplica SN_WHITELIST si está definida."""
    templates = {}
    for tipo_label, tipo_dir in TEMPLATE_DIRS.items():
        pattern = os.path.join(data_dir, tipo_dir, '*.dat')
        files = sorted(glob.glob(pattern))
        whitelist = SN_WHITELIST.get(tipo_label)
        if whitelist is not None:
            n_total = len(files)
            files = [f for f in files if os.path.basename(f) in whitelist]
            faltan = whitelist - {os.path.basename(f) for f in files}
            print(f"   [templates] {tipo_label}: {n_total} disponibles, {len(files)} usados (whitelist)")
            if faltan:
                print(f"   [WARNING] whitelist {tipo_label}: no encontrados en data/{tipo_dir}/: {sorted(faltan)}")
        templates[tipo_label] = []
        for f in files:
            sn_name = os.path.splitext(os.path.basename(f))[0]
            templates[tipo_label].append({
                'name': sn_name,
                'path': f,
                'tipo_dir': tipo_dir,
            })
    return templates


def generate_synthetic_curves(sn_name, tipo, path_spec, z_proy, ebmv_host, ebmv_mw,
                              response_folder, processing_config, lum_config):
    """
    Pipeline de generación de curvas sintéticas multi-banda.
    Espectro → corrección → fotometría → LOESS → ruido → normalización luminosidad.
    
    Returns: (curves_by_filter, synthetic_data) o (None, None) si falla.
    """
    try:
        ESPECTRO, fases = leer_spec(path_spec, ot=False, as_pandas=True)
    except Exception as e:
        print(f"      [ERROR] leer_spec: {e}")
        return None, None

    if len(ESPECTRO) == 0:
        return None, None

    try:
        ESPECTRO_corr, fases_corr = correct_redeening(
            sn=sn_name, ESPECTRO=ESPECTRO, fases=fases,
            z=z_proy, ebmv_host=ebmv_host, ebmv_mw=ebmv_mw,
            reverse=True, use_DL=True,
            # R_V del host por tipo: el MISMO que uso el sampler de E(B-V)
            # (EXTINCTION_CONFIG), o la extincion efectiva queda escalada
            rv_host=rv_host_for_type(tipo)
        )
    except Exception as e:
        print(f"      [ERROR] correct_redeening: {e}")
        return None, None

    curves_by_filter = {}
    synthetic_data = {}
    t0_rest = float(np.min(fases_corr))

    for filt in AVAILABLE_FILTERS:
        if filt not in RESPONSE_FILES:
            continue

        response_filename = RESPONSE_FILES[filt]
        path_response = os.path.join(response_folder, response_filename)
        if not os.path.exists(path_response):
            continue

        response_df = pd.read_csv(path_response, sep=r'\s+', comment='#', header=None)
        response_df.columns = ['wave', 'response']

        fases_lc, fluxes_lc = [], []
        for spec, fase in zip(ESPECTRO_corr, fases_corr):
            flux, porcentaje = Syntetic_photometry_v2(
                spec['wave'].values, spec['flux'].values,
                response_df['wave'].values, response_df['response'].values
            )
            if porcentaje > processing_config['overlap_threshold']:
                fases_lc.append(fase)
                fluxes_lc.append(flux)

        if len(fases_lc) == 0:
            continue

        lc_df = pd.DataFrame({'fase': fases_lc, 'flux': fluxes_lc}).sort_values('fase')

        # LOESS smoothing (apagado por defecto desde 2026-10-02, ver config)
        loess_result = None
        if processing_config.get('loess_smooth', True):
            LC_df = pd.DataFrame({
                0: np.array(lc_df['fase']), 1: np.array(lc_df['flux']),
                2: np.zeros(len(lc_df['fase'])), 3: ['F'] * len(lc_df['fase'])
            })
            cutoff = processing_config['loess_cutoff']
            alpha = processing_config['loess_alpha_many'] if len(LC_df) > cutoff else processing_config['loess_alpha_few']
            loess_result = Loess_fit(LC_df, filt, mag_to_flux=False, interactive=False,
                                     fig_title='', use_cte='False', alpha=alpha,
                                     corte=processing_config['loess_corte'], plot=False)

        # Si LOESS tuvo éxito, usar el flux suavizado interpolado de vuelta a la grilla original
        if (loess_result is not None and
                isinstance(loess_result, pd.DataFrame) and
                len(loess_result) >= 2 and
                'flux' in loess_result.columns and
                'mjd' in loess_result.columns):
            lc_df = lc_df.copy()
            lc_df['flux'] = np.interp(
                np.array(lc_df['fase']),
                loess_result['mjd'].values,
                loess_result['flux'].values
            )

        # Calibración (la curva queda LIMPIA; el ruido se aplica después)
        mul = FILTER_CONSTANTS[filt]
        flux_calibrado = np.array(lc_df['flux']) / mul
        mag = -2.5 * np.log10(np.clip(flux_calibrado, 1e-20, None))

        # FIX 2026-06-28: el ruido fotométrico YA NO se inyecta aquí. Antes era un 15%
        # fijo en espacio de FLUJO (Poisson de fuente, σ∝√F), no atado a la profundidad
        # del survey y con blowup al pasar a magnitud (de ahí el clip a 1e-20 que tenía).
        # Ahora se aplica POR ÉPOCA en multiband_field_projection desde el maglimit real
        # (limitado por cielo, survey-agnóstico, en espacio de magnitud, sin blowup).
        # Ver bitácora Proyección 2026-06-28.
        # Dilatación temporal (2026-10-02): el template está en días de reposo;
        # en el marco observado cada intervalo se estira por (1+z). t0 común a
        # todas las bandas para no desfasarlas entre sí.
        fases_obs = t0_rest + (np.array(lc_df['fase']) - t0_rest) * (1.0 + float(z_proy))
        curves_by_filter[filt] = (fases_obs, mag)
        synthetic_data[filt] = {'mag': mag, 'mag_noisy': mag}

    if len(curves_by_filter) == 0:
        return None, None

    # Normalización de luminosidad
    tipo_norm = "Ibc" if tipo in ["Ibc", "Ib", "Ic"] else tipo
    if lum_config.get('enabled', False) and (tipo_norm in lum_config.get('apply_to_types', [])):
        ref_filt = lum_config.get('reference_filter', 'r')
        if ref_filt not in synthetic_data:
            ref_filt = list(synthetic_data.keys())[0]

        dist = lum_config.get('M_peak', {}).get(tipo_norm)
        if dist is not None:
            m_mean = float(dist.get('mean', -17.0))
            m_sigma = float(dist.get('sigma', 1.0))
            # Relacion ancho-luminosidad (Phillips) para Ia: la media del sorteo
            # se ancla al dm15 del template en vez de ser fija, y la dispersion pasa
            # a ser la residual tras Phillips. Preserva bright<->slow, faint<->fast.
            dm15_used = None
            if (PHILLIPS_CONFIG.get('enabled', False)
                    and tipo_norm in PHILLIPS_CONFIG.get('apply_to_types', [])):
                _base = os.path.basename(path_spec)
                if _base not in DM15_IA and _base not in _DM15_WARNED:
                    print(f"      [WARNING] Phillips: {_base} sin dm15 en "
                          f"{PHILLIPS_CONFIG['dm15_file']}, usando "
                          f"dm15_default={PHILLIPS_CONFIG.get('dm15_default', 1.1)}")
                    _DM15_WARNED.add(_base)
                dm15_used = float(DM15_IA.get(_base, PHILLIPS_CONFIG.get('dm15_default', 1.1)))
                m_mean = (PHILLIPS_CONFIG['M0']
                          + PHILLIPS_CONFIG['slope'] * (dm15_used - PHILLIPS_CONFIG['dm15_ref']))
                m_sigma = float(PHILLIPS_CONFIG['sigma_resid'])
                m_peak_abs = float(np.random.normal(loc=m_mean, scale=max(1e-6, m_sigma)))
            elif 'sigma_bright' in dist:
                # Distribucion ASIMETRICA alrededor de la mediana: dos medias
                # normales con sigma distinto por lado y masa 50/50, para que la
                # mediana del sorteo = mediana publicada y los percentiles 16/84
                # caigan en mediana -/+ sigma (fiel a como Chen+2023 reporta
                # -21.48 +1.13/-0.61). Validado: p16/p50/p84 del sorteo
                # reproducen -22.09/-21.48/-20.35 a <0.01 mag.
                _med = float(dist['median'])
                _sb = float(dist['sigma_bright'])
                _sf = float(dist['sigma_faint'])
                if np.random.rand() < 0.5:
                    m_peak_abs = _med - abs(float(np.random.normal(0.0, _sb)))
                else:
                    m_peak_abs = _med + abs(float(np.random.normal(0.0, _sf)))
            elif (lum_config.get('M_mode', 'gaussian') == 'empirical'
                    and tipo_norm in M_EMPIRICAL):
                # Run de ENTRENAMIENTO: resamplear M de la muestra real observada
                # (paridad con z empirico). Jitter 0.1 mag para suavizar el resampleo.
                m_peak_abs = float(np.random.choice(M_EMPIRICAL[tipo_norm])
                                   + np.random.normal(0.0, 0.1))
            else:
                m_peak_abs = float(np.random.normal(loc=m_mean, scale=max(1e-6, m_sigma)))
            clip = lum_config.get('clip', {})
            if 'min' in clip:
                m_peak_abs = max(float(clip['min']), m_peak_abs)
            if 'max' in clip:
                m_peak_abs = min(float(clip['max']), m_peak_abs)

            _mrest = M_REST.get(tipo_norm, {}).get(os.path.basename(path_spec))
            if tipo_norm in M_REST and _mrest is None:
                _k = f"{tipo_norm}/{os.path.basename(path_spec)}"
                if _k not in _MREST_WARNED:
                    print(f"      [WARNING] rest_anchor: {_k} sin M rest en "
                          f"{LUMINOSITY_CONFIG['rest_anchor'][tipo_norm]['file']}, "
                          f"cae al anclaje observado (revisar precompute)")
                    _MREST_WARNED.add(_k)
            if _mrest is not None:
                # Anclaje REST-FRAME (SLSN-I): el template tiene M rest conocido
                # en la banda de la calibracion (precomputado); el shift lo lleva
                # al M sorteado y la K-correction la hereda el SED completo.
                delta_mag = m_peak_abs - float(_mrest)
            else:
                DL_mpc = DL_calculator(float(z_proy))
                mu = 5.0 * math.log10(DL_mpc * 1e6) - 5.0
                m_peak_target = mu + m_peak_abs
                m_peak_current = float(np.min(synthetic_data[ref_filt]['mag']))
                delta_mag = m_peak_target - m_peak_current

            for f in list(curves_by_filter.keys()):
                fases_arr, mag_arr = curves_by_filter[f]
                curves_by_filter[f] = (fases_arr, mag_arr + delta_mag)

            # Provenance del sorteo de luminosidad (auditable por simulacion).
            # '_lum_draw' no es un filtro: run_single_simulation lo extrae y lo
            # vuelca como columnas del parquet.
            synthetic_data['_lum_draw'] = {'m_peak_abs': m_peak_abs, 'dm15_used': dm15_used}

    # (2026-10-02) Se eliminó la "conversión temporal para Ibc" (fase + MJD de
    # máximo de maximum_Ibc.dat): las Ibc ya venían en MJD, así que duplicaba el
    # tiempo. Todos los templates v78 traen MJD.
    return curves_by_filter, synthetic_data


def run_single_simulation(oid, tipo, template, part_index, df_obslog_field,
                          z_proy, ebmv_host, ebmv_mw, response_folder,
                          processing_config, lum_config, offset_arr):
    """
    Ejecuta una simulación individual: genera curvas + proyecta.
    
    Returns: dict con resultado o None si falla.
    """
    sn_name = template['name']
    path_spec = template['path']

    curves, synth = generate_synthetic_curves(
        sn_name=sn_name, tipo=tipo, path_spec=path_spec,
        z_proy=z_proy, ebmv_host=ebmv_host, ebmv_mw=ebmv_mw,
        response_folder=response_folder,
        processing_config=processing_config, lum_config=lum_config
    )

    if curves is None:
        return None

    try:
        result = multiband_field_projection(
            curves_by_filter=curves,
            df_obslog=df_obslog_field,
            tipo=tipo,
            available_filters=list(curves.keys()),
            offset=offset_arr,
            sn=sn_name,
            selected_field=oid,
            plot=False,
            offset_search_mode='deterministic',
            n_divisions=N_DIVISIONS,
            part_index=part_index,
        )
    except Exception as e:
        print(f"      [ERROR] projection: {e}")
        return None

    df_proj = result.get('projections', pd.DataFrame())
    # Sin ninguna epoca con la SN encendida (solo upper limits pre-explosion,
    # magnitud_modelo=99) la simulacion no existe para el survey: FAIL.
    if len(df_proj) == 0 or not (df_proj['magnitud_modelo'] < 90).any():
        return None

    # Enriquecer con metadata
    df_proj['sn_type'] = tipo
    df_proj['template'] = sn_name
    df_proj['oid'] = oid
    df_proj['z'] = z_proy
    df_proj['ebmv_host'] = ebmv_host
    df_proj['ebmv_mw'] = ebmv_mw
    df_proj['part_index'] = part_index
    df_proj['n_divisions'] = N_DIVISIONS
    df_proj['offset_used'] = result.get('offset_used', 0)
    df_proj['desplazamiento'] = result.get('desplazamiento', 0)
    # Provenance del sorteo de luminosidad (M_peak asignado; dm15 solo Ia/Phillips)
    _draw = (synth or {}).get('_lum_draw', {})
    df_proj['m_peak_abs'] = _draw.get('m_peak_abs', np.nan)
    df_proj['dm15_used'] = _draw.get('dm15_used', np.nan)

    return {
        'projections': df_proj,
        'n_detections': int(df_proj['detected'].sum()),
        'n_observations': len(df_proj),
    }


def run_field(oid, df_obslog_field, templates, response_folder,
              processing_config, lum_config, z_min=0.01, z_max=0.5,
              oid_coords=None, z_max_by_type=None, ebmv_mw_oid=None, seed=None,
              tipos=None):
    """
    Ejecuta las simulaciones de un campo (OID): len(tipos) × 10 posiciones. El orden de templates se baraja por (OID, tipo)
    para evitar el sesgo del ciclo alfabético (ver run_field, bloque shuffle).

    z_max_by_type: dict opcional {tipo: z_max} — si se provee, sobreescribe
    z_max por tipo (p.ej. Ia=0.15, II=0.08, Ibc=0.10 para ZTF).
    Si es None, usa z_max escalar para todos los tipos (compat).

    ebmv_mw_oid: float opcional — E(B-V)_MW precomputado para este OID.
    Si es None, se consulta IRSA live (legacy) o se usa fallback 0.02.

    Returns: list of DataFrames con las proyecciones.
    """
    all_projections = []
    if tipos is None:
        tipos = list(TEMPLATE_DIRS)

    # Offset array (from processing config)
    offset_range = processing_config['offset_range']
    offset_step = processing_config['offset_step']
    offset_arr = np.arange(offset_range[0], offset_range[1], offset_step)

    for tipo in tipos:
        available_templates = templates.get(tipo, [])
        if len(available_templates) == 0:
            print(f"   [WARN] No hay templates para tipo {tipo}, saltando")
            continue

        # Barajar templates por (OID, tipo). Con N_DIVISIONS != n_templates el
        # ciclo alfabético sesga: duplica los primeros (II: 8<10) o trunca los
        # últimos (Ia 10/13, Ibc 10/22). El shuffle reproducible reparte parejo
        # sobre los OIDs. RNG aparte (default_rng) para NO tocar los draws de
        # z/E/magnitud, que siguen saliendo del estado global de np.random.
        if seed is not None:
            _h = int(hashlib.md5(f"{oid}_{tipo}".encode()).hexdigest()[:8], 16)
            _rng = np.random.default_rng(seed + _h)
        else:
            _rng = np.random.default_rng()
        sim_templates = list(available_templates)
        _rng.shuffle(sim_templates)

        # Override offset range by type
        offset_range_by_type = processing_config.get('offset_range_by_type', {}) or {}
        if tipo in offset_range_by_type:
            or_tipo = offset_range_by_type[tipo]
            offset_arr_tipo = np.arange(or_tipo[0], or_tipo[1], offset_step)
        else:
            offset_arr_tipo = offset_arr

        # z_max efectivo para este tipo (por-tipo si se especificó)
        z_max_tipo = z_max_by_type.get(tipo, z_max) if z_max_by_type else z_max

        for part_idx in range(N_DIVISIONS):
            # Template del orden barajado (reparte parejo sobre los OIDs)
            tpl = sim_templates[part_idx % len(sim_templates)]

            # Muestrear z y extinción para cada simulación.
            # z_mode 'empirical' (Z_CONFIG): resamplea de la distribución de z REAL
            # de la muestra etiquetada (data/z_empirical_<tipo>.txt) — para el run de
            # ENTRENAMIENTO. 'volumetric' (default): dV/dz hasta z_max (para tasas).
            if Z_CONFIG.get('z_mode') == 'empirical' and tipo in Z_EMPIRICAL:
                zvals = Z_EMPIRICAL[tipo]
                z_proy = float(np.clip(np.random.choice(zvals) + np.random.normal(0.0, 0.003), 0.005, None))
            else:
                z_proy = float(sample_cosmological_redshift(n_samples=1, z_min=z_min, z_max=z_max_tipo)[0])
            ebmv_host = float(sample_extinction_by_type(sn_type=tipo, n_samples=1)[0])
            # E(B-V)_MW: usar valor precomputado por OID si está disponible;
            # si no, consultar IRSA live (compat) o usar fallback 0.02
            if ebmv_mw_oid is not None:
                ebmv_mw = ebmv_mw_oid
            elif oid_coords is not None:
                ra, dec = oid_coords
                ebmv_mw, sfd_ok = get_sfd98_extinction_real(ra, dec)
                if not sfd_ok:
                    ebmv_mw = 0.02  # fallback si falla la conexión
            else:
                ebmv_mw = 0.02  # fallback si no hay coordenadas

            sim_label = f"{tipo}_p{part_idx}"
            result = run_single_simulation(
                oid=oid, tipo=tipo, template=tpl, part_index=part_idx,
                df_obslog_field=df_obslog_field,
                z_proy=z_proy, ebmv_host=ebmv_host, ebmv_mw=ebmv_mw,
                response_folder=response_folder,
                processing_config=processing_config, lum_config=lum_config,
                offset_arr=offset_arr_tipo,
            )

            if result is not None:
                all_projections.append(result['projections'])
                status = f"det={result['n_detections']}/{result['n_observations']}"
            else:
                status = "FAIL"

            print(f"   [{sim_label}] tpl={tpl['name']}, z={z_proy:.3f}, E={ebmv_host:.3f} → {status}")

    return all_projections


def main():
    parser = argparse.ArgumentParser(description='Runner por campo: 10 sims determinísticas por OID y tipo')
    parser.add_argument('--oid', type=str, default=None, help='OID específico a procesar')
    parser.add_argument('--n-fields', type=int, default=None, help='Número máximo de campos a procesar')
    parser.add_argument('--min-obs', type=int, default=30, help='Mínimo de observaciones por campo (default: 30)')
    parser.add_argument('--z-min', type=float, default=None,
                        help='Redshift mínimo (default: desde config.Z_CONFIG)')
    parser.add_argument('--z-max', type=float, default=None,
                        help='Redshift máximo global — si se pasa, sobreescribe z_max_by_type '
                             '(default: None, usa z_max_by_type por tipo desde config)')
    parser.add_argument('--output-dir', type=str, default='outputs/per_field', help='Directorio de salida')
    parser.add_argument('--seed', type=int, default=None, help='Seed global para reproducibilidad')
    parser.add_argument('--sort-by-obs', action='store_true',
                        help='Seleccionar OIDs por # de observaciones descendente (top-cobertura). '
                             'Si no se pasa, se ordenan alfabéticamente.')
    parser.add_argument('--oids-file', type=str, default=None,
                        help='Archivo con un OID por línea. Si se pasa, procesa exactamente '
                             'esa lista (ignora --sort-by-obs, --n-fields, --min-obs).')
    parser.add_argument('--types', type=str, default=None,
                        help='Tipos a simular, separados por coma (ej: "II" o "Ia,II"). '
                             'Default: todos los de config.TEMPLATE_DIRS. Útil para controles rápidos por tipo.')
    parser.add_argument('--m-mode', type=str, default=None, choices=['gaussian', 'empirical'],
                        help="Override de LUMINOSITY_CONFIG['M_mode']: 'empirical' resamplea "
                             "M_peak de data/M_empirical_<tipo>.txt (run de entrenamiento).")
    args = parser.parse_args()

    if args.m_mode is not None:
        LUMINOSITY_CONFIG['M_mode'] = args.m_mode
    tipos_run = list(dict.fromkeys(t.strip() for t in args.types.split(','))) if args.types else list(TEMPLATE_DIRS)
    if not set(tipos_run) <= set(TEMPLATE_DIRS):
        parser.error(f"--types debe ser subconjunto de {sorted(TEMPLATE_DIRS)}, recibido {tipos_run}")
    # Fallar ANTES de cargar el obslog si a un tipo le falta config: sin esto,
    # un tipo sin M_peak se proyecta con la luminosidad propia del template en
    # silencio, y uno sin extincion revienta a mitad del run.
    for _t in tipos_run:
        sample_extinction_by_type(_t)  # ValueError si no hay EXTINCTION_CONFIG
        if (LUMINOSITY_CONFIG.get('enabled') and _t in LUMINOSITY_CONFIG.get('apply_to_types', [])
                and _t not in LUMINOSITY_CONFIG.get('M_peak', {})):
            parser.error(f"{_t} esta en LUMINOSITY_CONFIG['apply_to_types'] pero sin M_peak")
    n_sims_field = N_DIVISIONS * len(tipos_run)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Resolver rango de redshift desde config.Z_CONFIG, con overrides desde CLI
    z_min_eff = args.z_min if args.z_min is not None else Z_CONFIG.get('z_min', 0.01)
    global_override = Z_CONFIG.get('z_max_global_override')
    if args.z_max is not None:
        # CLI --z-max gana y se aplica a todos los tipos
        z_max_eff = args.z_max
        z_max_by_type = None
    elif global_override is not None:
        z_max_eff = global_override
        z_max_by_type = None
    else:
        z_max_by_type = dict(Z_CONFIG.get('z_max_by_type', {}))
        z_max_eff = max(z_max_by_type.values()) if z_max_by_type else 0.5

    # Cargar datos
    print("=" * 60)
    print("RUNNER POR CAMPO — 10 SIMS DETERMINÍSTICAS POR OID Y TIPO")
    print("=" * 60)
    if z_max_by_type:
        print(f"Redshift: z_min={z_min_eff}, z_max_by_type={z_max_by_type}")
    else:
        print(f"Redshift: z_min={z_min_eff}, z_max={z_max_eff} (global)")

    data_dir = PATHS['data_dir']
    response_folder = PATHS['response_folder']
    obslog_path = os.path.join(data_dir, 'ZTF_observing_log_complete.csv')

    # Cargar coordenadas RA/Dec por OID desde catálogo maestro (merge de múltiples
    # fuentes: BTS, TNS, etc.) construido por tools/build_coords_catalog.py.
    # Fallback al CSV de BTS (solo 209 OIDs) si el master no existe.
    master_path = os.path.join(data_dir, 'ztf_coords_master.parquet')
    legacy_path = os.path.join(data_dir, 'ztf_targets_with_coords_multicat_summary.csv')
    oid_coords_map = {}
    if os.path.exists(master_path):
        df_coords = pd.read_parquet(master_path)
        oid_coords_map = dict(zip(df_coords['oid'],
                                  zip(df_coords['ra_deg'].astype(float),
                                      df_coords['dec_deg'].astype(float))))
        print(f"  Coordenadas cargadas: {len(oid_coords_map):,} OIDs desde {os.path.basename(master_path)}")
    elif os.path.exists(legacy_path):
        df_coords = pd.read_csv(legacy_path, usecols=['sn_name', 'ra_used_deg', 'dec_used_deg'])
        oid_coords_map = {
            row['sn_name']: (row['ra_used_deg'], row['dec_used_deg'])
            for _, row in df_coords.iterrows()
            if pd.notna(row['ra_used_deg']) and pd.notna(row['dec_used_deg'])
        }
        print(f"  Coordenadas cargadas (legacy): {len(oid_coords_map)} OIDs desde {os.path.basename(legacy_path)}")
        print(f"  [WARN] Ejecuta 'python tools/build_coords_catalog.py' para catálogo unificado")
    else:
        print(f"  [WARN] Sin catálogo de coords — usando E(B-V)_MW=0.02 fijo")

    # Cargar cache SFD98 (precomputado por tools/precompute_sfd98.py)
    sfd98_cache_path = os.path.join(data_dir, 'sfd98_cache.parquet')
    ebmv_mw_cache = {}
    if os.path.exists(sfd98_cache_path):
        df_sfd = pd.read_parquet(sfd98_cache_path)
        ebmv_mw_cache = dict(zip(df_sfd['oid'], df_sfd['ebmv_mw'].astype(float)))
        print(f"  SFD98 cache: {len(ebmv_mw_cache):,} OIDs cacheados ({sfd98_cache_path})")
    else:
        print(f"  [WARN] Sin cache SFD98 — consultas IRSA en vivo (lento). "
              f"Ejecuta: python tools/precompute_sfd98.py")

    # Templates ANTES del obslog (355 MB): si una carpeta vino vacia (Drive sin
    # hidratar) o a una Ia le falta dm15, se aborta aqui y no a mitad del run.
    templates = scan_templates(data_dir)
    for t, tpls in templates.items():
        print(f"  Templates {t}: {len(tpls)}")
    _vacios = [t for t in tipos_run if not templates.get(t)]
    if _vacios:
        raise SystemExit(f"[ERROR] tipos sin templates en data/: {_vacios}")
    if PHILLIPS_CONFIG.get('enabled', False):
        _sin_dm15 = [os.path.basename(d['path']) for t in tipos_run
                     if t in PHILLIPS_CONFIG.get('apply_to_types', []) for d in templates[t]
                     if os.path.basename(d['path']) not in DM15_IA]
        if _sin_dm15:
            raise SystemExit(f"[ERROR] Phillips: sin dm15 en {PHILLIPS_CONFIG['dm15_file']}: {_sin_dm15} "
                             "(regenerar con precompute_dm15_Ia.py)")
    for t in tipos_run:
        _zsrc = ('empirico' if Z_CONFIG.get('z_mode') == 'empirical' and t in Z_EMPIRICAL
                 else f"dV/dz hasta {(z_max_by_type or {}).get(t, z_max_eff)}")
        _msrc = ('empirico' if LUMINOSITY_CONFIG.get('M_mode') == 'empirical' and t in M_EMPIRICAL
                 else 'distribucion de config')
        print(f"  {t}: z {_zsrc} | M_peak {_msrc}")

    print(f"\nCargando observing log: {obslog_path}")
    t0 = time.time()
    df_obslog = pd.read_csv(obslog_path)
    print(f"  Cargado en {time.time()-t0:.1f}s: {len(df_obslog):,} filas, {df_obslog['oid'].nunique():,} OIDs")

    # Normalizar columnas ZTF
    if 'filter' not in df_obslog.columns and 'fid' in df_obslog.columns:
        fid_to_filter = {1: 'g', 2: 'r', 3: 'i'}
        df_obslog['filter'] = df_obslog['fid'].map(fid_to_filter)
    if 'maglimit' not in df_obslog.columns and 'diffmaglim' in df_obslog.columns:
        df_obslog = df_obslog.rename(columns={'diffmaglim': 'maglimit'})

    # Seleccionar OIDs
    if args.oid:
        oids = [args.oid]
    elif args.oids_file:
        with open(args.oids_file) as f:
            oids = [line.strip() for line in f if line.strip()]
        # Validar que los OIDs existen en el obslog
        obslog_oids = set(df_obslog['oid'].unique())
        missing = [o for o in oids if o not in obslog_oids]
        if missing:
            print(f"  [WARN] {len(missing)} OIDs del archivo no están en obslog: {missing[:5]}...")
            oids = [o for o in oids if o in obslog_oids]
    else:
        obs_counts = df_obslog.groupby('oid').size()
        valid_counts = obs_counts[obs_counts >= args.min_obs]
        if args.sort_by_obs:
            # Top-N por cobertura: OIDs con más observaciones primero
            oids = valid_counts.sort_values(ascending=False).index.tolist()
        else:
            oids = sorted(valid_counts.index.tolist())
        if args.n_fields:
            oids = oids[:args.n_fields]

    print(f"\nCampos a procesar: {len(oids)}")
    print(f"Tipos: {tipos_run} | sims por campo: {n_sims_field}")
    print(f"Simulaciones totales: {len(oids) * n_sims_field}")

    # Crear directorio de salida
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Git commit del repo (para reproducibilidad)
    def _git_info():
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                             cwd=os.path.dirname(os.path.abspath(__file__)),
                                             stderr=subprocess.DEVNULL).decode().strip()
            dirty = bool(subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL).decode().strip())
            return {'commit': commit, 'dirty': dirty}
        except Exception:
            return {'commit': None, 'dirty': None}

    # Guardar metadata COMPLETA (todo lo necesario para reproducir el run)
    metadata = {
        # identificación del run
        'run_id': run_id,
        'start_time': datetime.now().isoformat(),

        # CLI / entrada
        'cli_args': vars(args),
        'survey': SURVEY,

        # campos a procesar
        'n_fields': len(oids),
        'n_sims_per_field': n_sims_field,
        'tipos': tipos_run,
        # templates usados (nombres por tipo) y carpeta; md5 en data/templates_v78_md5.csv
        'template_dirs': dict(TEMPLATE_DIRS),
        'templates': {t: [d['name'] for d in templates[t]] for t in tipos_run},
        'n_divisions': N_DIVISIONS,

        # redshift (efectivo, ya resuelto)
        'z_config': {
            'z_min': z_min_eff,
            'z_max_scalar': z_max_eff,
            'z_max_by_type': z_max_by_type,
            'source': dict(Z_CONFIG),
        },

        # configs completos desde config.py
        'processing_config': dict(PROCESSING_CONFIG),
        'luminosity_config': dict(LUMINOSITY_CONFIG),
        'extinction_config': dict(EXTINCTION_CONFIG),
        # Phillips WLR: config + los dm15 exactos usados (dm15_Ia.json no esta
        # pineado por git, asi que el run debe registrarlos para ser reproducible)
        'phillips_config': dict(PHILLIPS_CONFIG),
        'dm15_ia': dict(DM15_IA),

        # reproducibilidad
        'seed': args.seed,
        'git': _git_info(),
    }
    with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Procesar campos
    t_start = time.time()
    total_sims = 0
    total_fails = 0
    all_results = []
    new_sfd98_rows = []  # entradas nuevas para persistir al cache al final

    for i_field, oid in enumerate(oids):
        t_field = time.time()
        df_field = df_obslog[df_obslog['oid'] == oid]
        n_obs = len(df_field)

        print(f"\n{'='*60}")
        print(f"[{i_field+1}/{len(oids)}] OID: {oid} ({n_obs} obs)")
        print(f"{'='*60}")

        # Resolver E(B-V)_MW UNA vez por OID (era 30x en el loop interno)
        oid_coords = oid_coords_map.get(oid)
        if oid in ebmv_mw_cache:
            ebmv_mw_oid = ebmv_mw_cache[oid]
        elif oid_coords is not None:
            ra, dec = oid_coords
            ebmv_mw_live, sfd_ok = get_sfd98_extinction_real(ra, dec)
            ebmv_mw_oid = ebmv_mw_live if sfd_ok else 0.02
            # auto-guardar en cache en memoria + buffer para persistir al final
            ebmv_mw_cache[oid] = ebmv_mw_oid
            new_sfd98_rows.append({
                'oid': oid, 'ra_deg': float(ra), 'dec_deg': float(dec),
                'ebmv_mw': float(ebmv_mw_oid), 'sfd_ok': bool(sfd_ok),
                'queried_at': datetime.now().isoformat(timespec='seconds'),
            })
        else:
            ebmv_mw_oid = 0.02

        projections = run_field(
            oid=oid, df_obslog_field=df_field, templates=templates,
            response_folder=response_folder,
            processing_config=PROCESSING_CONFIG, lum_config=LUMINOSITY_CONFIG,
            z_min=z_min_eff, z_max=z_max_eff,
            oid_coords=oid_coords,
            z_max_by_type=z_max_by_type,
            ebmv_mw_oid=ebmv_mw_oid,
            seed=args.seed,
            tipos=tipos_run,
        )

        n_ok = len(projections)
        n_fail = n_sims_field - n_ok
        total_sims += n_ok
        total_fails += n_fail

        if projections:
            df_combined = pd.concat(projections, ignore_index=True)
            out_path = os.path.join(output_dir, f'{oid}.parquet')
            df_combined.to_parquet(out_path, index=False)
            print(f"  Guardado: {out_path} ({len(df_combined)} filas, {n_ok}/{n_sims_field} sims OK)")

            all_results.append({
                'oid': oid,
                'n_obs_field': n_obs,
                'n_sims_ok': n_ok,
                'n_sims_fail': n_fail,
                'n_projected_rows': len(df_combined),
                'time_s': time.time() - t_field,
            })
        else:
            print(f"  [WARN] Sin resultados para {oid}")

        elapsed = time.time() - t_start
        rate = (i_field + 1) / elapsed * 60 if elapsed > 0 else 0
        print(f"  Tiempo: {time.time()-t_field:.1f}s | Total: {elapsed:.0f}s | Rate: {rate:.1f} campos/min")

    # Resumen final
    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"Campos procesados: {len(oids)}")
    print(f"Simulaciones exitosas: {total_sims}/{total_sims+total_fails}")
    print(f"Tiempo total: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"Output: {output_dir}")

    # Guardar resumen
    if all_results:
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(os.path.join(output_dir, 'run_summary.csv'), index=False)

    metadata['end_time'] = datetime.now().isoformat()
    metadata['total_sims_ok'] = total_sims
    metadata['total_sims_fail'] = total_fails
    metadata['elapsed_seconds'] = elapsed_total
    with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Persistir entradas nuevas del cache SFD98 (si las hubo)
    if new_sfd98_rows:
        df_new_sfd = pd.DataFrame(new_sfd98_rows)
        if os.path.exists(sfd98_cache_path):
            df_existing = pd.read_parquet(sfd98_cache_path)
            df_merged = pd.concat([df_existing, df_new_sfd], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=['oid'], keep='last')
        else:
            df_merged = df_new_sfd
        tmp = sfd98_cache_path + '.tmp'
        df_merged.to_parquet(tmp, index=False)
        os.replace(tmp, sfd98_cache_path)
        print(f"\nCache SFD98 actualizado: +{len(new_sfd98_rows)} entradas "
              f"→ {len(df_merged):,} totales en {sfd98_cache_path}")


if __name__ == '__main__':
    main()
