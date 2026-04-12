# IMPORTS MODULARES
# =====================
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
from core.projection import field_projection  
from core.save_functions import save_projection_results, create_master_index

# Sistema de configuración
from config_loader import load_and_validate_config, get_survey_info, get_sn_info, print_config_summary

#  WORKFLOW MODULAR - PROYECCIÓN MULTI-SURVEY (CON CONFIGURACIÓN)
# =================================================================

def main(config=None):
    """
    Función principal del pipeline de proyección de supernovas
    
    Args:
        config (dict, optional): Configuración pre-cargada. Si es None, se carga desde archivo.
    """
    # CARGAR CONFIGURACION (solo si no se proporciona)
    # ================================================
    if config is None:
        config = load_and_validate_config()
        print_config_summary(config)
    # Si config viene del batch_runner, ya fue validada y el summary se mostró allí

    # Extraer información específica
    survey_info = get_survey_info(config)
    sn_info = get_sn_info(config)
    processing_config = config['processing']
    extinction_config = config['extinction']

    # Variables para compatibilidad con código existente
    SURVEY = survey_info['SURVEY']
    path_obslog = survey_info['path_obslog']
    projection_filter = survey_info['projection_filter']
    target_column = survey_info['target_column']

    sn_name = sn_info['sn_name']
    tipo = sn_info['tipo']
    selected_filter = sn_info['selected_filter']
    z_proy = sn_info['z_proy']
    ebmv_host = sn_info['ebmv_host']
    ebmv_mw = sn_info['ebmv_mw']
    use_synthetic_extinction = sn_info['use_synthetic_extinction']
    path_spec = sn_info['path_spec']
    path_response_folder = sn_info['path_response_folder']
    response_files = sn_info['response_files']

    # PASO 1: LECTURA DE ESPECTRO (igual para ambos surveys)
    # =========================================================
    print(f"\nPASO 1: Lectura de espectro")
    ESPECTRO, fases = leer_spec(path_spec, ot=False, as_pandas=True)
    print(f"   • Archivo: {sn_name}")
    print(f"   • Espectros leídos: {len(ESPECTRO)}")
    print(f"   • Fases disponibles: {len(fases)} (rango: {min(fases):.1f} a {max(fases):.1f} días)")

    # PASO 2: CORRECCIONES COSMOLÓGICAS Y EXTINCIÓN
    # ================================================
    print(f"\nPASO 2: Correcciones cosmológicas y extinción")
    
    # Configurar semilla si es para reproducibilidad
    if use_synthetic_extinction and extinction_config.get('use_reproducible_sampling', False):
        if extinction_config.get('random_seed') is not None:
            np.random.seed(extinction_config['random_seed'])
            print(f"   🎲 Usando semilla fija para reproducibilidad: {extinction_config['random_seed']}")
    
    # Generar o usar extinción del host
    # EVITAR DOBLE MUESTREO: Si ebmv_host ya viene del batch, usarlo directamente
    if ebmv_host is not None:
        # Valor ya muestreado por batch_runner (evitar doble muestreo)
        ebmv_host_final = ebmv_host
        print(f"   E(B-V) host: Usando valor del batch: {ebmv_host_final:.3f}")
    elif use_synthetic_extinction:
        # Solo muestrear si no viene del batch usando funciones específicas por tipo
        print(f"   E(B-V) host: Muestreando para SN {tipo}...")
        
        # Usar directamente el despachador académicamente correcto
        ebmv_host_final = sample_extinction_by_type(sn_type=tipo, n_samples=1)[0]
        print(f"      - Distribución mixta académicamente correcta según tipo {tipo}")
            
        print(f"   E(B-V) host: Valor muestreado localmente: {ebmv_host_final:.3f}")
    else:
        ebmv_host_final = 0.0
        print(f"   E(B-V) host: Sin extinción sintética del host")
    
    print(f"   • Redshift proyectado: z = {z_proy}")
    print(f"   • E(B-V) host: {ebmv_host_final:.3f}")
    print(f"   • E(B-V) MW: {ebmv_mw:.3f}")
    
    # Aplicar correcciones (redshift, distancia, y extinción completa)
    # NOTA: ebmv_host y ebmv_mw se deben aplicar en orden físico dentro de correct_redeening
    ESPECTRO_corr, fases_corr = correct_redeening(
        sn=sn_name, ESPECTRO=ESPECTRO, fases=fases,
        z=z_proy, ebmv_host=ebmv_host_final, ebmv_mw=ebmv_mw, 
        reverse=True, use_DL=True
    )
    print(f"   • Espectros corregidos: {len(ESPECTRO_corr)}")

    # PASO 3: CURVA DE RESPUESTA (igual para ambos surveys)
    # ========================================================
    print(f"\nPASO 3: Curva de respuesta del filtro {selected_filter}")
    response_filename = response_files[selected_filter]
    path_response = os.path.join(path_response_folder, response_filename)

    response_df = pd.read_csv(path_response, sep='\s+', comment='#', header=None)
    response_df.columns = ['wave', 'response']
    print(f"   • Archivo: {response_filename}")
    print(f"   • Rango longitud de onda: {response_df['wave'].min():.0f} - {response_df['wave'].max():.0f} Å")

    # PASO 4: FOTOMETRÍA SINTÉTICA (igual para ambos surveys)
    # ==========================================================
    print(f"\nPASO 4: Fotometría sintética")
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

    print(f"   • Puntos con overlap >{processing_config['overlap_threshold']*100:.0f}%: {len(lc_df)}/{len(ESPECTRO_corr)}")
    print(f"   • Rango de fases: {lc_df['fase'].min():.1f} a {lc_df['fase'].max():.1f} días")
    print(f"   • Overlap promedio: {np.mean(porcentaje_lc)*100:.1f}%")

    # PASO 5: SUAVIZADO LOESS (usando configuración)
    # ==================================================
    print(f"\nPASO 5: Suavizado LOESS")
    LC_df = pd.DataFrame({
        0: np.array(lc_df['fase']), 1: np.array(lc_df['flux']),
        2: np.zeros(len(lc_df['fase'])), 3: ['F'] * len(lc_df['fase'])
    })

    # Usar configuración para alpha
    cutoff = processing_config['loess_cutoff']
    alpha_usado = processing_config['loess_alpha_many'] if len(LC_df) > cutoff else processing_config['loess_alpha_few']
    df_loess = Loess_fit(LC_df, selected_filter, mag_to_flux=False, interactive=False,
                         fig_title='', use_cte='False', alpha=alpha_usado,
                         corte=processing_config['loess_corte'], plot=False)
    print(f"   • Alpha usado: {alpha_usado}")
    print(f"   • Puntos LOESS generados: {len(df_loess) if len(df_loess) > 0 else 'No generado'}")
    # liberar objetos grandes ASAP
    try:
        import gc
        del LC_df
        gc.collect()
    except Exception:
        pass

    # PASO 6: CALIBRACIÓN Y CONVERSIÓN A MAGNITUDES (igual para ambos surveys)
    # ============================================================================
    print(f"\nPASO 6: Calibración fotométrica")
    arr_ctes = ['cteB', 'cteV', 'cteR', 'cteI', 'cteU', 'cteu', 'cteg', 'cter', 'ctei', 'ctez']
    arr_val_ctes = [cteB, cteV, cteR, cteI, cteU, cteu, cteg, cter, ctei, ctez]

    constante = 'cte' + selected_filter
    for jj in range(len(arr_ctes)):
        if constante == arr_ctes[jj]:
            mul = arr_val_ctes[jj]
            break

    flux_calibrado = np.array(lc_df['flux']) / mul
    mag = -2.5 * np.log10(np.clip(flux_calibrado, 1e-20, None))
    print(f"   • Constante usada: {constante} = {mul:.2e}")
    print(f"   • Rango magnitudes: {mag.min():.2f} a {mag.max():.2f} mag")

    # PASO 7: APLICACIÓN DE RUIDO (usando configuración)
    # ======================================================
    print(f"\nPASO 7: Aplicación de ruido fotométrico")
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
    print(f"   • Ruido poissoniano: {noise_level*100:.0f}% en flujo")
    print(f"   • Ruido resultante: σ = {ruido_promedio:.3f} mag")

    # PARCHE: Normalización de luminosidad al peak (II / Ibc)
    # =======================================================
    # Renormaliza el nivel absoluto para que el peak (mínimo mag) coincida con
    # un M_peak muestreado por tipo, en vez de usar la escala original del template.
    lum_cfg = config.get('luminosity', {}) if isinstance(config, dict) else {}
    tipo_norm = "Ibc" if tipo in ["Ibc", "Ib", "Ic"] else tipo
    if lum_cfg.get('enabled', False) and (tipo_norm in lum_cfg.get('apply_to_types', [])):
        # Reproducibilidad opcional: usar RNG local (NO tocar np.random global)
        rng_lum = None
        if lum_cfg.get('use_reproducible_sampling', False) and lum_cfg.get('random_seed') is not None:
            rng_lum = np.random.default_rng(int(lum_cfg['random_seed']))

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
            m_peak_current = float(np.min(mag))
            delta_mag = m_peak_target - m_peak_current

            mag = mag + delta_mag
            mag_noisy = mag_noisy + delta_mag

            # Guardar metadata para outputs
            config['luminosity_shift_mag'] = float(delta_mag)
            config['luminosity_M_peak_abs'] = float(m_peak_abs)
            config['distance_modulus'] = float(mu)

            print(f"   • [LUM-NORM] Aplicada normalización peak para tipo {tipo_norm}:")
            print(f"      - M_peak(abs) muestreado: {m_peak_abs:.2f}")
            print(f"      - mu(z): {mu:.2f}  => m_peak_target: {m_peak_target:.2f}")
            print(f"      - m_peak_current: {m_peak_current:.2f}  => shift: {delta_mag:+.2f} mag")

    # PASO 7.5: CONVERSIÓN DE UNIDADES TEMPORALES (crítico para consistencia)
    # ========================================================================
    maximum = maximo_lc(tipo, sn_name)
    print(f"   • Fecha de máximo {sn_name}: MJD {maximum:.1f}")
    
    # Conversión de fases relativas → MJD absoluto para SNe core-collapse
    if tipo in ['Ibc', 'Ib', 'Ic']:  # Para SNe core-collapse con fases relativas
        print(f"   • Rango original (fases relativas): {lc_df['fase'].min():.1f} a {lc_df['fase'].max():.1f} días")
        mjd_absolute = maximum + lc_df['fase']  # Convertir fases → MJD
        lc_df['fase'] = mjd_absolute           # Actualizar el DataFrame
        print(f"   • Convertidas fases relativas → MJD absoluto")
        print(f"   • Nuevo rango temporal: MJD {lc_df['fase'].min():.1f} - {lc_df['fase'].max():.1f}")
    else:
        print(f"   • Datos ya en MJD absoluto: {lc_df['fase'].min():.1f} - {lc_df['fase'].max():.1f}")

    # PASO 8: PROYECCIÓN ESPECÍFICA POR SURVEY (usando configuración)
    # ===================================================================
    print(f"\nPASO 8: Proyección sobre observaciones reales ({SURVEY})")
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

    # Selección de target específica por survey
    if SURVEY == "ZTF":
        # ZTF: Seleccionar OID al azar
        available_targets = df_obslog_survey[target_column].unique()
        selected_target = np.random.choice(available_targets)
        print(f"   • OIDs disponibles: {len(available_targets):,}")
        print(f"   • OID seleccionado: {selected_target}")
        
    elif SURVEY == "SUDARE":
        # SUDARE: Seleccionar campo al azar de los campos configurados
        available_fields = survey_info.get('available_fields', ['cdfs1', 'cdfs2', 'cosmos'])
        selected_target = np.random.choice(available_fields)
        print(f"   • Campos SUDARE disponibles: {available_fields}")
        print(f"   • Campo seleccionado: {selected_target}")

    print(f"   • Filtro para grilla: {projection_filter}")

    # Proyección con parámetros de configuración
    offset_range = processing_config['offset_range']
    # Override de offsets por tipo (solo si está definido)
    offset_range_by_type = processing_config.get('offset_range_by_type', {}) or {}
    if isinstance(offset_range_by_type, dict) and tipo in offset_range_by_type:
        offset_range = offset_range_by_type[tipo]
    offset_step = processing_config['offset_step']
    show_debug_plots = processing_config['show_debug_plots']
    df_projected = field_projection(
        fases=lc_df['fase'].values,
        flux_y=mag_noisy,
        df_obslog=df_obslog_survey,
        tipo=tipo,
        selected_filter=projection_filter,  # Filtro específico del survey
        selected_field=selected_target,     # Target seleccionado aleatoriamente
        offset=np.arange(offset_range[0], offset_range[1], offset_step),
        sn=sn_name,
        plot=show_debug_plots  # Controla si mostrar gráfico de debug
    )

    # PASO 9: RESULTADOS FINALES
    # =============================
    print(f"\nPASO 9: Resultados finales ({SURVEY})")
    print(f"   • Target seleccionado: {selected_target}")
    print(f"   • Total puntos proyectados: {len(df_projected):,}")

    if len(df_projected) > 0:
        detecciones = len(df_projected[df_projected['upperlimit'] == 'F'])
        upper_limits = len(df_projected[df_projected['upperlimit'] == 'T'])
        tasa_deteccion = detecciones/len(df_projected)*100
        
        print(f"   • Detecciones: {detecciones:,}")
        print(f"   • Upper limits: {upper_limits:,}")
        print(f"   • Tasa de detección: {tasa_deteccion:.1f}%")
        print(f"   • Rango temporal: MJD {df_projected['mjd'].min():.1f} - {df_projected['mjd'].max():.1f}")
        print(f"   • Rango maglimit: {df_projected['maglimit'].min():.2f} - {df_projected['maglimit'].max():.2f} mag")
    else:
        print("    No se generaron proyecciones (SN no observable)")

    print(f"\nPROYECCIÓN {SURVEY} COMPLETADA")
    print(f" Para cambiar de survey, modifica la variable SURVEY al inicio de la celda")

    # PASO 10: GUARDAR RESULTADOS
    # ==============================
    # Preparar parámetros para la función de guardado
    survey_params = {
        'SURVEY': SURVEY,
        'selected_target': selected_target
    }

    sn_params = {
        'sn_name': sn_name,
        'tipo': tipo,
        'z_proy': z_proy,
        'ebmv_host': ebmv_host_final,
        'ebmv_mw': ebmv_mw,
        'ebmv_total': ebmv_host_final + ebmv_mw  # Solo para registro, corrección ya aplicada por separado
    }

    projection_params = {
        'selected_filter': selected_filter,
        'projection_filter': projection_filter,
        'path_spec': path_spec,
        'path_response_folder': path_response_folder,
        'path_obslog': path_obslog,
        'target_column': target_column
    }

    # Llamar función modular para guardar todo
    saved_files = save_projection_results(
        df_projected=df_projected,
        lc_df=lc_df,
        mag=mag,
        mag_noisy=mag_noisy,
        survey_params=survey_params,
        sn_params=sn_params,
        projection_params=projection_params,
        ruido_promedio=ruido_promedio,
        alpha_usado=alpha_usado,
        maximum=maximum
    )

    # PASO 11: ACTUALIZAR ÍNDICE MAESTRO
    # ====================================
    create_master_index()


# Llamar a la función principal si se ejecuta directamente
if __name__ == "__main__":
    main()