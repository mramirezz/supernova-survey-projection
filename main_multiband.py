"""
MAIN MULTIBAND - PROYECCIÓN MULTI-FILTRO SIMULTÁNEA
===================================================

Esta versión corrige el problema donde cada filtro se proyectaba independientemente.
Ahora se proyectan TODOS los filtros simultáneamente con el MISMO offset temporal.
"""

import os
import numpy as np  
import pandas as pd
import math

# Imports desde módulos específicos
from core.utils import (
    leer_spec, Syntetic_photometry_v2, Loess_fit, maximo_lc, DL_calculator,
    # Constantes fotométricas
    cteB, cteV, cteR, cteI, cteU, cteu, cteg, cter, ctei, ctez
)
from core.correction import correct_redeening, sample_extinction_by_type
from core.multiband_projection import multiband_field_projection  
from core.save_functions import save_projection_results, create_master_index

# Sistema de configuración
from config_loader import load_and_validate_config, get_survey_info, get_sn_info, print_config_summary

# Constantes fotométricas por filtro
FILTER_CONSTANTS = {
    'B': cteB, 'V': cteV, 'R': cteR, 'I': cteI, 'U': cteU,
    'u': cteu, 'g': cteg, 'r': cter, 'i': ctei, 'z': ctez
}

def main_multiband(config=None):
    """
    Función principal del pipeline de proyección MULTI-BANDA de supernovas
    
    Args:
        config (dict, optional): Configuración pre-cargada. Si es None, se carga desde archivo.
    """
    # CARGAR CONFIGURACION
    # ====================
    if config is None:
        config = load_and_validate_config()
        print_config_summary(config)

    # Extraer información específica
    survey_info = get_survey_info(config)
    sn_info = get_sn_info(config)
    processing_config = config['processing']
    extinction_config = config['extinction']

    # Variables para compatibilidad
    SURVEY = survey_info['SURVEY']
    path_obslog = survey_info['path_obslog']
    target_column = survey_info['target_column']

    sn_name = sn_info['sn_name']
    tipo = sn_info['tipo']
    z_proy = sn_info['z_proy']
    ebmv_host = sn_info['ebmv_host']
    ebmv_mw = sn_info['ebmv_mw']
    use_synthetic_extinction = sn_info['use_synthetic_extinction']
    path_spec = sn_info['path_spec']
    path_response_folder = sn_info['path_response_folder']
    response_files = sn_info['response_files']

    # DETERMINAR FILTROS DISPONIBLES
    # ===============================
    # Por defecto, proyectar en todos los filtros ZTF disponibles
    if SURVEY == "ZTF":
        available_filters = ['g', 'r', 'i']  # Filtros ZTF
    elif SURVEY == "SUDARE":
        available_filters = ['U', 'B', 'V', 'R', 'I']  # Filtros SUDARE
    else:
        available_filters = list(response_files.keys())
    
    print(f"\n{'='*60}")
    print(f"PROYECCIÓN MULTI-BANDA SIMULTÁNEA")
    print(f"{'='*60}")
    print(f"Filtros a proyectar: {available_filters}")
    print(f"{'='*60}\n")

    # PASO 1: LECTURA DE ESPECTRO
    # ============================
    print(f"PASO 1: Lectura de espectro")
    ESPECTRO, fases = leer_spec(path_spec, ot=False, as_pandas=True)
    print(f"   • Archivo: {sn_name}")
    print(f"   • Espectros leídos: {len(ESPECTRO)}")
    print(f"   • Fases disponibles: {len(fases)} (rango: {min(fases):.1f} a {max(fases):.1f} días)")

    # PASO 2: CORRECCIONES COSMOLÓGICAS Y EXTINCIÓN
    # ==============================================
    print(f"\nPASO 2: Correcciones cosmológicas y extinción")
    
    if use_synthetic_extinction and extinction_config.get('use_reproducible_sampling', False):
        if extinction_config.get('random_seed') is not None:
            np.random.seed(extinction_config['random_seed'])
    
    if ebmv_host is not None:
        ebmv_host_final = ebmv_host
        print(f"   E(B-V) host: Usando valor del batch: {ebmv_host_final:.3f}")
    elif use_synthetic_extinction:
        ebmv_host_final = sample_extinction_by_type(sn_type=tipo, n_samples=1)[0]
        print(f"   E(B-V) host: Valor muestreado: {ebmv_host_final:.3f}")
    else:
        ebmv_host_final = 0.0
    
    print(f"   • Redshift proyectado: z = {z_proy}")
    print(f"   • E(B-V) host: {ebmv_host_final:.3f}")
    print(f"   • E(B-V) MW: {ebmv_mw:.3f}")
    
    ESPECTRO_corr, fases_corr = correct_redeening(
        sn=sn_name, ESPECTRO=ESPECTRO, fases=fases,
        z=z_proy, ebmv_host=ebmv_host_final, ebmv_mw=ebmv_mw, 
        reverse=True, use_DL=True
    )
    print(f"   • Espectros corregidos: {len(ESPECTRO_corr)}")

    # PASO 3-7: GENERAR CURVAS SINTÉTICAS PARA TODOS LOS FILTROS
    # ===========================================================
    print(f"\nPASO 3-7: Generando curvas sintéticas para todos los filtros")
    
    curves_by_filter = {}  # {filter: (fases_mjd, magnitudes_noisy)}
    synthetic_data = {}    # Para guardar resultados
    
    for filt in available_filters:
        if filt not in response_files:
            print(f"   [WARNING] No hay curva de respuesta para filtro {filt}, saltando...")
            continue
            
        print(f"\n   --- Filtro {filt} ---")
        
        # Curva de respuesta
        response_filename = response_files[filt]
        path_response = os.path.join(path_response_folder, response_filename)
        response_df = pd.read_csv(path_response, sep='\s+', comment='#', header=None)
        response_df.columns = ['wave', 'response']
        
        # Fotometría sintética
        fases_lc, fluxes_lc, porcentaje_lc = [], [], []
        n_rejected = 0
        for spec, fase in zip(ESPECTRO_corr, fases_corr):
            flux, porcentaje = Syntetic_photometry_v2(
                spec['wave'].values, spec['flux'].values,
                response_df['wave'].values, response_df['response'].values
            )
            if porcentaje > processing_config['overlap_threshold']:
                fases_lc.append(fase)
                fluxes_lc.append(flux)
                porcentaje_lc.append(porcentaje)
            else:
                n_rejected += 1

        lc_df = pd.DataFrame({
            'fase': fases_lc, 'flux': fluxes_lc, 'overlap': porcentaje_lc
        }).sort_values('fase')
        
        print(f"      • Puntos con overlap >{processing_config['overlap_threshold']*100:.0f}%: {len(lc_df)}/{len(ESPECTRO_corr)} ({n_rejected} rechazados)")
        
        # Saltar filtro si no hay datos suficientes
        if len(lc_df) == 0:
            print(f"      • [WARNING] Sin datos para filtro {filt}, se omite")
            continue
        
        if len(lc_df) > 0:
            print(f"      • [DEBUG Phot {filt}] Rango MJD datos sintéticos: {lc_df['fase'].min():.1f} - {lc_df['fase'].max():.1f} ({lc_df['fase'].max()-lc_df['fase'].min():.0f} días)")
            print(f"      • [DEBUG Phot {filt}] Overlap mín/máx: {lc_df['overlap'].min():.3f} / {lc_df['overlap'].max():.3f}")
        
        # Suavizado LOESS (paso fundamental)
        LC_df = pd.DataFrame({
            0: np.array(lc_df['fase']), 1: np.array(lc_df['flux']),
            2: np.zeros(len(lc_df['fase'])), 3: ['F'] * len(lc_df['fase'])
        })
        
        cutoff = processing_config['loess_cutoff']
        alpha_usado = processing_config['loess_alpha_many'] if len(LC_df) > cutoff else processing_config['loess_alpha_few']
        df_loess = Loess_fit(LC_df, filt, mag_to_flux=False, interactive=False,
                             fig_title='', use_cte='False', alpha=alpha_usado,
                             corte=processing_config['loess_corte'], plot=False)
        # liberar objetos grandes ASAP
        try:
            import gc
            del LC_df
            gc.collect()
        except Exception:
            pass
        
        # Calibración fotométrica
        mul = FILTER_CONSTANTS[filt]
        flux_calibrado = np.array(lc_df['flux']) / mul
        mag = -2.5 * np.log10(np.clip(flux_calibrado, 1e-20, None))
        
        # Aplicación de ruido
        flux_from_mag = 10 ** (-0.4 * mag)
        minimo_flux = np.min(flux_from_mag)
        flux_norm = flux_from_mag / minimo_flux
        
        noise_level = processing_config['noise_level']
        flux_noisy_norm = np.random.normal(
            loc=flux_norm, scale=np.sqrt(np.abs(flux_norm)) * noise_level
        )
        
        flux_noisy = flux_noisy_norm * minimo_flux
        flux_noisy = np.clip(flux_noisy, 1e-20, None)
        mag_noisy = -2.5 * np.log10(flux_noisy)
        
        ruido_promedio = np.std(mag_noisy - mag)
        print(f"      • Ruido: sigma = {ruido_promedio:.3f} mag")
        
        # Convertir Loess a magnitudes para ploteo
        # Prioridad: Loess (si flux > 0), si no → interpolación lineal punto a punto
        if df_loess is not None and len(df_loess) > 0:
            flux_loess_raw = df_loess['flux'].values / mul
            mjd_loess_raw = df_loess['mjd'].values
            
            # Contar puntos con flujo negativo o cero
            n_negative = np.sum(flux_loess_raw <= 0)
            
            if n_negative > 0:
                print(f"      • Loess: {n_negative} puntos con flux<=0 descartados (gaps grandes)")
                # Fallback: interpolación lineal punto a punto con datos sin ruido
                mag_loess = mag
                mjd_loess = np.array(lc_df['fase'])
            else:
                # Loess válido, convertir a magnitudes
                mag_loess = -2.5 * np.log10(flux_loess_raw)
                mjd_loess = mjd_loess_raw
        else:
            mag_loess = None
            mjd_loess = None
        
        # Guardar curva para este filtro
        curves_by_filter[filt] = (np.array(lc_df['fase']), mag_noisy)
        synthetic_data[filt] = {
            'lc_df': lc_df,
            'mag': mag,
            'mag_noisy': mag_noisy,
            'mag_loess': mag_loess,
            'mjd_loess': mjd_loess,
            'ruido': ruido_promedio,
            'alpha': alpha_usado
        }
    
    # PARCHE: Normalización de luminosidad al peak (II / Ibc), consistente entre filtros
    # ================================================================================
    lum_cfg = config.get('luminosity', {}) if isinstance(config, dict) else {}
    tipo_norm = "Ibc" if tipo in ["Ibc", "Ib", "Ic"] else tipo
    if lum_cfg.get('enabled', False) and (tipo_norm in lum_cfg.get('apply_to_types', [])) and synthetic_data:
        # Reproducibilidad opcional: usar RNG local (NO tocar np.random global)
        rng_lum = None
        if lum_cfg.get('use_reproducible_sampling', False) and lum_cfg.get('random_seed') is not None:
            rng_lum = np.random.default_rng(int(lum_cfg['random_seed']))

        # Elegir filtro de referencia
        ref_filt = lum_cfg.get('reference_filter', 'r')
        if ref_filt not in synthetic_data:
            ref_filt = list(synthetic_data.keys())[0]

        dist = lum_cfg.get('M_peak', {}).get(tipo_norm, None)
        if dist is not None:
            m_mean = float(dist.get('mean', -17.0))
            m_sigma = float(dist.get('sigma', 1.0))
            if rng_lum is None:
                m_peak_abs = float(np.random.normal(loc=m_mean, scale=max(1e-6, m_sigma)))
            else:
                m_peak_abs = float(rng_lum.normal(loc=m_mean, scale=max(1e-6, m_sigma)))
            clip = lum_cfg.get('clip', {})
            if 'min' in clip:
                m_peak_abs = max(float(clip['min']), m_peak_abs)
            if 'max' in clip:
                m_peak_abs = min(float(clip['max']), m_peak_abs)

            DL_mpc = DL_calculator(float(z_proy))
            mu = 5.0 * math.log10(DL_mpc * 1e6) - 5.0

            m_peak_target = mu + m_peak_abs
            m_peak_current = float(np.min(synthetic_data[ref_filt]['mag']))
            delta_mag = m_peak_target - m_peak_current

            # Aplicar mismo shift a todos los filtros
            for f in synthetic_data.keys():
                synthetic_data[f]['mag'] = synthetic_data[f]['mag'] + delta_mag
                synthetic_data[f]['mag_noisy'] = synthetic_data[f]['mag_noisy'] + delta_mag
                if synthetic_data[f].get('mag_loess', None) is not None:
                    synthetic_data[f]['mag_loess'] = synthetic_data[f]['mag_loess'] + delta_mag

                # curves_by_filter contiene (fases, mag_noisy)
                if f in curves_by_filter:
                    fases_arr, mag_noisy_arr = curves_by_filter[f]
                    curves_by_filter[f] = (fases_arr, mag_noisy_arr + delta_mag)

            # Guardar metadata en config (para trazabilidad en outputs)
            config['luminosity_shift_mag'] = float(delta_mag)
            config['luminosity_M_peak_abs'] = float(m_peak_abs)
            config['distance_modulus'] = float(mu)

            print(f"\n   • [LUM-NORM] Aplicada normalización peak para tipo {tipo_norm} (ref={ref_filt}):")
            print(f"      - M_peak(abs) muestreado: {m_peak_abs:.2f}")
            print(f"      - mu(z): {mu:.2f}  => m_peak_target: {m_peak_target:.2f}")
            print(f"      - m_peak_current({ref_filt}): {m_peak_current:.2f}  => shift: {delta_mag:+.2f} mag")

    # PASO 7.5: CONVERSIÓN DE UNIDADES TEMPORALES
    # ============================================
    print(f"\nPASO 7.5: Conversión temporal")
    maximum = maximo_lc(tipo, sn_name)
    print(f"   • Fecha de máximo {sn_name}: MJD {maximum:.1f}")
    
    if tipo in ['Ibc', 'Ib', 'Ic']:
        print(f"   • Convirtiendo fases relativas → MJD absoluto")
        for filt in curves_by_filter:
            fases_rel, mag_noisy = curves_by_filter[filt]
            fases_abs = fases_rel + maximum
            curves_by_filter[filt] = (fases_abs, mag_noisy)
            # Actualizar también en synthetic_data
            synthetic_data[filt]['lc_df']['fase'] = synthetic_data[filt]['lc_df']['fase'] + maximum
    else:
        print(f"   • Datos ya en MJD absoluto")

    # PASO 8: PROYECCIÓN MULTI-BANDA SIMULTÁNEA
    # ==========================================
    print(f"\nPASO 8: Proyección MULTI-BANDA sobre observaciones reales ({SURVEY})")
    df_obslog_survey = pd.read_csv(path_obslog)
    
    # Normalizar formato ZTF (soportar archivos con fid/diffmaglim)
    # ZTF_observing_log_complete.csv puede venir con columnas: oid,mjd,fid,diffmaglim
    if SURVEY == "ZTF":
        if 'filter' not in df_obslog_survey.columns and 'fid' in df_obslog_survey.columns:
            fid_to_filter = {1: 'g', 2: 'r', 3: 'i'}
            df_obslog_survey = df_obslog_survey.copy()
            df_obslog_survey['filter'] = df_obslog_survey['fid'].map(fid_to_filter)
        if 'maglimit' not in df_obslog_survey.columns and 'diffmaglim' in df_obslog_survey.columns:
            df_obslog_survey = df_obslog_survey.rename(columns={'diffmaglim': 'maglimit'})

    # Selección de target (puede ser fijo o aleatorio)
    fixed_field = processing_config.get('fixed_field', None)
    
    if SURVEY == "ZTF":
        available_targets = df_obslog_survey[target_column].unique()
        
        if fixed_field and fixed_field in available_targets:
            selected_target = fixed_field
            print(f"   • OIDs disponibles: {len(available_targets):,}")
            print(f"   • OID seleccionado (FIJO): {selected_target}")
        else:
            selected_target = np.random.choice(available_targets)
            print(f"   • OIDs disponibles: {len(available_targets):,}")
            print(f"   • OID seleccionado: {selected_target}")
            if fixed_field:
                print(f"   ⚠️  Campo fijo '{fixed_field}' no encontrado, usando aleatorio")
                
    elif SURVEY == "SUDARE":
        available_fields = survey_info.get('available_fields', ['cdfs1', 'cdfs2', 'cosmos'])
        
        if fixed_field and fixed_field in available_fields:
            selected_target = fixed_field
            print(f"   • Campos SUDARE disponibles: {available_fields}")
            print(f"   • Campo seleccionado (FIJO): {selected_target}")
        else:
            selected_target = np.random.choice(available_fields)
            print(f"   • Campos SUDARE disponibles: {available_fields}")
            print(f"   • Campo seleccionado: {selected_target}")
            if fixed_field:
                print(f"   ⚠️  Campo fijo '{fixed_field}' no encontrado, usando aleatorio")

    # PROYECCIÓN MULTI-BANDA
    offset_range = processing_config['offset_range']
    # Override de offsets por tipo (solo si está definido)
    offset_range_by_type = processing_config.get('offset_range_by_type', {}) or {}
    if isinstance(offset_range_by_type, dict) and tipo in offset_range_by_type:
        offset_range = offset_range_by_type[tipo]
    offset_step = processing_config['offset_step']
    
    projection_result = multiband_field_projection(
        curves_by_filter=curves_by_filter,
        df_obslog=df_obslog_survey,
        tipo=tipo,
        available_filters=list(curves_by_filter.keys()),
        offset=np.arange(offset_range[0], offset_range[1], offset_step),
        sn=sn_name,
        selected_field=selected_target,
        plot=processing_config['show_debug_plots'],
        # Opcionales (runner por lista): fallback determinístico y/o “último recurso”
        required_filter=processing_config.get('required_filter', None),
        min_detections=processing_config.get('min_detections_required', None),
        offset_search_mode=processing_config.get('offset_search_mode', 'random'),
        force_brighten_to_min_detections=processing_config.get('force_brighten_to_min_detections', False),
        max_force_brightening_mag=processing_config.get('max_force_brightening_mag', 3.0),
    )
    
    # Agregar lista de filtros proyectados al resultado
    projection_result['filters_projected'] = list(curves_by_filter.keys())

    df_projections_multi = projection_result['projections']

    # PASO 9: RESULTADOS FINALES
    # ===========================
    print(f"\nPASO 9: Resultados finales ({SURVEY})")
    print(f"   • Target seleccionado: {selected_target}")
    print(f"   • Offset usado: {projection_result['offset_used']}")
    print(f"   • Total puntos proyectados: {projection_result['n_observations']:,}")

    if projection_result['n_observations'] > 0:
        for filt in available_filters:
            df_filt = df_projections_multi[df_projections_multi['filter'] == filt]
            if len(df_filt) > 0:
                detections = len(df_filt[df_filt['upperlimit'] == 'F'])
                upper_limits = len(df_filt[df_filt['upperlimit'] == 'T'])
                print(f"   • Filtro {filt}: {len(df_filt)} obs ({detections} det, {upper_limits} UL)")
    else:
        print("    No se generaron proyecciones (SN no observable)")

    print(f"\nPROYECCIÓN MULTI-BANDA {SURVEY} COMPLETADA")

    # PASO 10: GUARDAR RESULTADOS
    # =======================================================
    print(f"\n[INFO] Guardando resultados multi-banda...")
    
    from core.save_multiband import save_multiband_results
    from datetime import datetime
    
    # Asegurar que tipo y sn_name están en config
    if 'tipo' not in config:
        config['tipo'] = tipo
    if 'sn_type' not in config:
        config['sn_type'] = tipo
    if 'sn_name' not in config:
        config['sn_name'] = sn_name
    if 'redshift' not in config:
        config['redshift'] = z_proy
    if 'ebv_host' not in config:
        config['ebv_host'] = ebmv_host_final
    if 'ebv_mw' not in config:
        config['ebv_mw'] = ebmv_mw
    if 'extinction_total' not in config:
        config['extinction_total'] = ebmv_host_final + ebmv_mw
    
    saved_files = save_multiband_results(
        df_projected=df_projections_multi,
        synthetic_curves=curves_by_filter,
        synthetic_data=synthetic_data,
        projection_result=projection_result,
        config=config,
        batch_id=config.get('batch_id', datetime.now().strftime("%Y%m%d_%H%M%S")),
        iteration_label=config.get('iteration_label', 'single_run')
    )
    
    print(f"\n[SUCCESS] Guardado completado!")
    if len(df_projections_multi) > 0:
        print(f"[SAVED] Archivos guardados:")
        for key, path in saved_files.items():
            print(f"        - {key}: {path}")
    
    return df_projections_multi

if __name__ == "__main__":
    main_multiband()
