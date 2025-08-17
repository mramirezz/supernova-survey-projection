# 📚 CARGADOR DE CONFIGURACIÓN
# ============================

import os
from config import *

def load_and_validate_config():
    """
    Carga y valida toda la configuración, construyendo rutas absolutas
    y verificando que los archivos existan.
    
    Returns:
    --------
    dict: Configuración validada y procesada
    """
    
    # Construir rutas absolutas
    config = {}
    
    # Configuración básica
    config['survey'] = SURVEY
    config['sn_config'] = SN_CONFIG.copy()
    config['processing'] = PROCESSING_CONFIG.copy()
    config['extinction'] = EXTINCTION_CONFIG.copy()  # Nueva configuración de extinción
    config['plot_config'] = PLOT_CONFIG.copy()
    config['batch_config'] = BATCH_CONFIG.copy()
    
    # Construir rutas absolutas
    base_dir = PATHS['base_dir']
    data_dir = PATHS['data_dir']
    
    config['paths'] = {
        'base_dir': base_dir,
        'data_dir': data_dir,
        'spec_path': os.path.join(data_dir, PATHS['spec_file']),
        'response_folder': PATHS['response_folder'],
        'output_dir': PATHS['output_dir']
    }
    
    # Configuración específica del survey seleccionado
    if SURVEY not in SURVEY_CONFIG:
        raise ValueError(f"Survey '{SURVEY}' no reconocido. Opciones: {list(SURVEY_CONFIG.keys())}")
    
    survey_info = SURVEY_CONFIG[SURVEY].copy()
    survey_info['obslog_path'] = os.path.join(data_dir, survey_info['obslog_file'])
    config['survey_config'] = survey_info
    
    # Archivos de respuesta con rutas completas
    config['response_files'] = {}
    for filtro, filename in RESPONSE_FILES.items():
        config['response_files'][filtro] = os.path.join(PATHS['response_folder'], filename)
    
    # Validaciones si están habilitadas
    if VALIDATION['check_file_existence']:
        _validate_file_paths(config)
    
    # Validaciones de parámetros
    _validate_parameters(config)
    
    return config


def _validate_file_paths(config):
    """Valida que los archivos requeridos existan"""
    
    files_to_check = [
        ('Espectro', config['paths']['spec_path']),
        ('Obslog', config['survey_config']['obslog_path']),
        ('Response folder', config['paths']['response_folder'])
    ]
    
    # Verificar filtro específico
    selected_filter = config['sn_config']['selected_filter']
    if selected_filter in config['response_files']:
        files_to_check.append((
            f'Response {selected_filter}', 
            config['response_files'][selected_filter]
        ))
    
    missing_files = []
    for file_type, file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type}: {file_path}")
    
    if missing_files:
        raise FileNotFoundError(
            f"Archivos faltantes:\n" + "\n".join([f"  - {f}" for f in missing_files])
        )


def _validate_parameters(config):
    """Valida que los parámetros estén en rangos válidos"""
    
    sn_config = config['sn_config']
    processing = config['processing']
    
    # Validar redshift
    if not 0 <= sn_config['z_proy'] <= 2:
        raise ValueError(f"Redshift fuera de rango: {sn_config['z_proy']} (debe estar entre 0-2)")
    
    # Validar extinción
    if sn_config.get('ebmv_host') is not None and not 0 <= sn_config['ebmv_host'] <= 2:
        raise ValueError(f"E(B-V) host fuera de rango: {sn_config['ebmv_host']} (debe estar entre 0-2)")
    
    if not 0 <= sn_config['ebmv_mw'] <= 2:
        raise ValueError(f"E(B-V) MW fuera de rango: {sn_config['ebmv_mw']} (debe estar entre 0-2)")
    
    # Validar filtro
    if sn_config['selected_filter'] not in RESPONSE_FILES:
        raise ValueError(f"Filtro no reconocido: {sn_config['selected_filter']}. Opciones: {list(RESPONSE_FILES.keys())}")
    
    # Validar overlap threshold
    if not 0.5 <= processing['overlap_threshold'] <= 1.0:
        raise ValueError(f"Overlap threshold fuera de rango: {processing['overlap_threshold']} (debe estar entre 0.5-1.0)")
    
    print("   Parametros validados correctamente")


def get_survey_info(config):
    """
    Extrae información específica del survey para usar en el main
    
    Returns:
    --------
    dict: Información del survey lista para usar
    """
    
    survey_config = config['survey_config']
    
    return {
        'SURVEY': config['survey'],
        'path_obslog': survey_config['obslog_path'],
        'projection_filter': survey_config['projection_filter'],
        'target_column': survey_config['target_column'],
        'available_fields': survey_config.get('available_fields', None),
        'description': survey_config['description']
    }


def get_sn_info(config):
    """
    Extrae información de la supernova para usar en el main
    """
    
    sn_config = config['sn_config']
    paths = config['paths']
    
    return {
        'sn_name': sn_config['sn_name'],
        'tipo': sn_config['tipo'],
        'selected_filter': sn_config['selected_filter'],
        'z_proy': sn_config['z_proy'],
        'ebmv_host': sn_config['ebmv_host'],
        'ebmv_mw': sn_config['ebmv_mw'],
        'use_synthetic_extinction': sn_config['use_synthetic_extinction'],
        'path_spec': paths['spec_path'],
        'path_response_folder': paths['response_folder'],
        'response_files': config['response_files']
    }


def print_config_summary(config):
    """Imprime un resumen de la configuración cargada"""
    
    print("\n" + "="*60)
    print("RESUMEN DE CONFIGURACION")
    print("="*60)
    
    # Survey
    survey_info = get_survey_info(config)
    print(f"Survey: {survey_info['SURVEY']} - {survey_info['description']}")
    print(f"   Dataset: {os.path.basename(survey_info['path_obslog'])}")
    print(f"   Target column: {survey_info['target_column']}")
    print(f"   Projection filter: {survey_info['projection_filter']}")
    
    # Supernova
    sn_info = get_sn_info(config)
    print(f"\nSupernova: {sn_info['sn_name']} ({sn_info['tipo']})")
    print(f"   Filtro fotometrico: {sn_info['selected_filter']}")
    print(f"   Redshift: z = {sn_info['z_proy']}")
    
    # Extinción
    if sn_info['use_synthetic_extinction']:
        if sn_info['ebmv_host'] is None:
            print(f"   E(B-V) host: muestreado sinteticamente segun tipo {sn_info['tipo']}")
        else:
            print(f"   E(B-V) host: {sn_info['ebmv_host']} (fijo)")
        print(f"   E(B-V) MW: {sn_info['ebmv_mw']}")
    else:
        ebmv_total = (sn_info['ebmv_host'] or 0) + sn_info['ebmv_mw']
        print(f"   E(B-V) total: {ebmv_total}")
    
    # Procesamiento
    processing = config['processing']
    print(f"\nProcesamiento:")
    print(f"   Overlap threshold: {processing['overlap_threshold']}")
    print(f"   LOESS alpha: {processing['loess_alpha_many']} / {processing['loess_alpha_few']}")
    print(f"   Ruido: {processing['noise_level']*100}%")
    print(f"   Offset range: {processing['offset_range'][0]} a {processing['offset_range'][1]} dias")
    print(f"   Debug plots: {'ON' if processing['show_debug_plots'] else 'OFF'}")
    
    print("="*60)
