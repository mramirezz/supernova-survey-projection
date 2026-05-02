"""
RUNNER POR CAMPO — 30 SIMULACIONES DETERMINÍSTICAS POR OID
==========================================================

Para cada campo (OID) del observing log, genera 30 curvas de luz simuladas:
  - 3 tipos (Ia, II, Ibc) × 10 posiciones de pivote determinísticas

Cada simulación:
  1. Elige un template (cíclico entre los disponibles por tipo)
  2. Muestrea z volume-weighted
  3. Muestrea E(B-V)_host por tipo
  4. Genera curvas sintéticas multi-banda (g, r, i)
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
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import (PATHS, RESPONSE_FILES, PROCESSING_CONFIG, LUMINOSITY_CONFIG,
                    Z_CONFIG, EXTINCTION_CONFIG, SURVEY)
import subprocess
from config_loader import load_and_validate_config
from core.utils import (
    leer_spec, Syntetic_photometry_v2, Loess_fit, maximo_lc, DL_calculator,
    cteB, cteV, cteR, cteI, cteU, cteu, cteg, cter, ctei, ctez
)
from core.correction import (
    correct_redeening, sample_extinction_by_type, sample_cosmological_redshift
)
from core.multiband_projection import multiband_field_projection
from tools.dust_maps import get_sfd98_extinction_real
import math

# Constantes fotométricas por filtro
FILTER_CONSTANTS = {
    'B': cteB, 'V': cteV, 'R': cteR, 'I': cteI, 'U': cteU,
    'u': cteu, 'g': cteg, 'r': cter, 'i': ctei, 'z': ctez
}

# Filtros ZTF
AVAILABLE_FILTERS = ['g', 'r', 'i']
N_DIVISIONS = 10  # particiones determinísticas


def scan_templates(data_dir):
    """Escanea templates disponibles por tipo."""
    templates = {}
    for tipo_dir, tipo_label in [('Ia', 'Ia'), ('II', 'II'), ('Ibc', 'Ibc')]:
        pattern = os.path.join(data_dir, tipo_dir, '*.dat')
        files = sorted(glob.glob(pattern))
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
            reverse=True, use_DL=True
        )
    except Exception as e:
        print(f"      [ERROR] correct_redeening: {e}")
        return None, None

    curves_by_filter = {}
    synthetic_data = {}

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

        # LOESS smoothing
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

        # Calibración + ruido
        mul = FILTER_CONSTANTS[filt]
        flux_calibrado = np.array(lc_df['flux']) / mul
        mag = -2.5 * np.log10(np.clip(flux_calibrado, 1e-20, None))

        flux_from_mag = 10 ** (-0.4 * mag)
        minimo_flux = np.min(flux_from_mag)
        flux_norm = flux_from_mag / minimo_flux

        noise_level = processing_config['noise_level']
        flux_noisy_norm = np.random.normal(
            loc=flux_norm, scale=np.sqrt(np.abs(flux_norm)) * noise_level
        )
        flux_noisy = np.clip(flux_noisy_norm * minimo_flux, 1e-20, None)
        mag_noisy = -2.5 * np.log10(flux_noisy)

        curves_by_filter[filt] = (np.array(lc_df['fase']), mag_noisy)
        synthetic_data[filt] = {'mag': mag, 'mag_noisy': mag_noisy}

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
            m_peak_abs = float(np.random.normal(loc=m_mean, scale=max(1e-6, m_sigma)))
            clip = lum_config.get('clip', {})
            if 'min' in clip:
                m_peak_abs = max(float(clip['min']), m_peak_abs)
            if 'max' in clip:
                m_peak_abs = min(float(clip['max']), m_peak_abs)

            DL_mpc = DL_calculator(float(z_proy))
            mu = 5.0 * math.log10(DL_mpc * 1e6) - 5.0
            m_peak_target = mu + m_peak_abs
            m_peak_current = float(np.min(synthetic_data[ref_filt]['mag']))
            delta_mag = m_peak_target - m_peak_current

            for f in list(curves_by_filter.keys()):
                fases_arr, mag_arr = curves_by_filter[f]
                curves_by_filter[f] = (fases_arr, mag_arr + delta_mag)

    # Conversión temporal para Ibc (fases relativas → MJD absoluto)
    if tipo in ['Ibc', 'Ib', 'Ic']:
        try:
            maximum = maximo_lc(tipo, sn_name)
            for filt in list(curves_by_filter.keys()):
                fases_rel, mag_noisy = curves_by_filter[filt]
                curves_by_filter[filt] = (fases_rel + maximum, mag_noisy)
        except Exception:
            pass  # Si no tiene máximo, dejar las fases como están

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
    if len(df_proj) == 0:
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

    return {
        'projections': df_proj,
        'n_detections': int(df_proj['detected'].sum()),
        'n_observations': len(df_proj),
    }


def run_field(oid, df_obslog_field, templates, response_folder,
              processing_config, lum_config, z_min=0.01, z_max=0.5,
              oid_coords=None, z_max_by_type=None, ebmv_mw_oid=None):
    """
    Ejecuta las 30 simulaciones para un campo (OID).
    3 tipos × 10 posiciones determinísticas.

    z_max_by_type: dict opcional {tipo: z_max} — si se provee, sobreescribe
    z_max por tipo (p.ej. Ia=0.15, II=0.08, Ibc=0.10 para ZTF).
    Si es None, usa z_max escalar para todos los tipos (compat).

    ebmv_mw_oid: float opcional — E(B-V)_MW precomputado para este OID.
    Si es None, se consulta IRSA live (legacy) o se usa fallback 0.02.

    Returns: list of DataFrames con las proyecciones.
    """
    all_projections = []
    tipos = ['Ia', 'II', 'Ibc']

    # Offset array (from processing config)
    offset_range = processing_config['offset_range']
    offset_step = processing_config['offset_step']
    offset_arr = np.arange(offset_range[0], offset_range[1], offset_step)

    for tipo in tipos:
        available_templates = templates.get(tipo, [])
        if len(available_templates) == 0:
            print(f"   [WARN] No hay templates para tipo {tipo}, saltando")
            continue

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
            # Template cíclico
            tpl = available_templates[part_idx % len(available_templates)]

            # Muestrear z y extinción para cada simulación
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
    parser = argparse.ArgumentParser(description='Runner por campo: 30 sims determinísticas por OID')
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
    args = parser.parse_args()

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
    print("RUNNER POR CAMPO — 30 SIMS DETERMINÍSTICAS POR OID")
    print("=" * 60)
    if z_max_by_type:
        print(f"Redshift: z_min={z_min_eff}, z_max_by_type={z_max_by_type}")
    else:
        print(f"Redshift: z_min={z_min_eff}, z_max={z_max_eff} (global)")

    data_dir = PATHS['data_dir']
    response_folder = PATHS['response_folder']
    obslog_path = os.path.join(data_dir, 'ZTF_observing_log_complete.csv')

    # Cargar coordenadas RA/Dec por OID para queries SFD98
    coords_path = os.path.join(data_dir, 'ztf_targets_with_coords_multicat_summary.csv')
    oid_coords_map = {}
    if os.path.exists(coords_path):
        df_coords = pd.read_csv(coords_path, usecols=['sn_name', 'ra_used_deg', 'dec_used_deg'])
        oid_coords_map = {
            row['sn_name']: (row['ra_used_deg'], row['dec_used_deg'])
            for _, row in df_coords.iterrows()
            if pd.notna(row['ra_used_deg']) and pd.notna(row['dec_used_deg'])
        }
        print(f"  Coordenadas cargadas: {len(oid_coords_map)} OIDs con RA/Dec para SFD98")
    else:
        print(f"  [WARN] No se encontró {coords_path} — usando E(B-V)_MW=0.02 fijo")

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

    # Escanear templates
    templates = scan_templates(data_dir)
    for t, tpls in templates.items():
        print(f"  Templates {t}: {len(tpls)}")

    # Seleccionar OIDs
    if args.oid:
        oids = [args.oid]
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
    print(f"Simulaciones totales: {len(oids) * 30}")

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
        'n_sims_per_field': 30,
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
        )

        n_ok = len(projections)
        n_fail = 30 - n_ok
        total_sims += n_ok
        total_fails += n_fail

        if projections:
            df_combined = pd.concat(projections, ignore_index=True)
            out_path = os.path.join(output_dir, f'{oid}.parquet')
            df_combined.to_parquet(out_path, index=False)
            print(f"  Guardado: {out_path} ({len(df_combined)} filas, {n_ok}/30 sims OK)")

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
