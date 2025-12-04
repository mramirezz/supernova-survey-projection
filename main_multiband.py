"""
MAIN MULTIBAND - PROYECCIÓN MULTI-FILTRO SIMULTÁNEA
===================================================

Esta versión corrige el problema donde cada filtro se proyectaba independientemente.
Ahora se proyectan TODOS los filtros simultáneamente con el MISMO offset temporal.
"""

import os
import numpy as np  
import pandas as pd

# Imports desde módulos específicos
from core.utils import (
    leer_spec, Syntetic_photometry_v2, Loess_fit, maximo_lc,
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
        for spec, fase in zip(ESPECTRO_corr, fases_corr):
            flux, porcentaje = Syntetic_photometry_v2(
                spec['wave'].values, spec['flux'].values,
                response_df['wave'].values, response_df['response'].values
            )
            if porcentaje > processing_config['overlap_threshold']:
                fases_lc.append(fase)
                fluxes_lc.append(flux)
                porcentaje_lc.append(porcentaje)

        lc_df = pd.DataFrame({
            'fase': fases_lc, 'flux': fluxes_lc, 'overlap': porcentaje_lc
        }).sort_values('fase')
        
        print(f"      • Puntos con overlap >{processing_config['overlap_threshold']*100:.0f}%: {len(lc_df)}")
        
        # Suavizado LOESS
        LC_df = pd.DataFrame({
            0: np.array(lc_df['fase']), 1: np.array(lc_df['flux']),
            2: np.zeros(len(lc_df['fase'])), 3: ['F'] * len(lc_df['fase'])
        })
        
        cutoff = processing_config['loess_cutoff']
        alpha_usado = processing_config['loess_alpha_many'] if len(LC_df) > cutoff else processing_config['loess_alpha_few']
        df_loess = Loess_fit(LC_df, filt, mag_to_flux=False, interactive=False,
                             fig_title='', use_cte='False', alpha=alpha_usado, 
                             corte=processing_config['loess_corte'], plot=False)
        
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
        
        # Guardar curva para este filtro
        curves_by_filter[filt] = (np.array(lc_df['fase']), mag_noisy)
        synthetic_data[filt] = {
            'lc_df': lc_df,
            'mag': mag,
            'mag_noisy': mag_noisy,
            'ruido': ruido_promedio,
            'alpha': alpha_usado
        }
    
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
    offset_step = processing_config['offset_step']
    
    projection_result = multiband_field_projection(
        curves_by_filter=curves_by_filter,
        df_obslog=df_obslog_survey,
        tipo=tipo,
        available_filters=list(curves_by_filter.keys()),
        offset=np.arange(offset_range[0], offset_range[1], offset_step),
        sn=sn_name,
        selected_field=selected_target,
        plot=processing_config['show_debug_plots']
    )

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
    
    saved_files = save_multiband_results(
        df_projected=df_projections_multi,
        synthetic_curves=curves_by_filter,
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
