# 🔧 ARCHIVO DE CONFIGURACIÓN - PROYECCIÓN DE SUPERNOVAS
# ======================================================

# 🎛️ CONFIGURACIÓN PRINCIPAL
# ===========================

# Survey a utilizar
SURVEY = "ZTF"  # Opciones: "ZTF", "SUDARE"

# 🌟 PARÁMETROS DE LA SUPERNOVA
# ==============================
SN_CONFIG = {
    "sn_name": "ASASSN-14lp",
    "tipo": "Ia",
    "selected_filter": "r",      # Filtro para fotometría sintética
    "z_proy": 0.05,              # Redshift proyectado
    "ebmv_host": None,           # E(B-V) del host (None = muestrear automáticamente)
    "ebmv_mw": 0.05,             # E(B-V) de la Vía Láctea (fijo para línea de visión)
    "use_synthetic_extinction": True  # Si usar distribuciones sintéticas
}

# 📂 RUTAS DE ARCHIVOS
# =====================
PATHS = {
    # Directorio base del proyecto
    "base_dir": r"G:\Mi unidad\Work\Universidad\Phd\paper2_ZTF\Codes\proyeccion",
    
    # Espectros de supernovas
    "data_dir": r"G:\Mi unidad\Work\Universidad\Phd\paper2_ZTF\Codes\proyeccion\data",
    "spec_file": "Ia/ASASSN-14lp.dat",  # Relativo a data_dir
    
    # Curvas de respuesta de filtros
    "response_folder": r"G:\Mi unidad\Work\Universidad\Phd\Practica2\Splines_eachfilter_2",
    
    # Directorio de salida
    "output_dir": "outputs"
}

# 📡 CONFIGURACIÓN POR SURVEY
# ============================
SURVEY_CONFIG = {
    "ZTF": {
        "obslog_file": "grid_diffmaglim_ZTF.csv",  # Relativo a data_dir
        "projection_filter": "r",                  # Filtro para grilla de fechas
        "target_column": "oid",                    # Columna que identifica targets
        "description": "Zwicky Transient Facility"
    },
    
    "SUDARE": {
        "obslog_file": "obslog_I.csv",            # Relativo a data_dir  
        "projection_filter": "r",                 # Filtro para grilla de fechas
        "target_column": "field",                 # Columna que identifica targets
        "available_fields": ["cdfs1", "cdfs2", "cosmos"],  # Campos específicos
        "description": "SUDARE Survey"
    }
}

# 🔬 ARCHIVOS DE RESPUESTA DE FILTROS
# ====================================
RESPONSE_FILES = {
    "U": "spline_U.txt",
    "B": "spline_B.txt", 
    "V": "spline_V.txt",
    "R": "bessell_R_ph_lines.dat",
    "I": "bessell_I_ph_lines.dat",
    "u": "spline_u'.txt",
    "g": "spline_g'.txt", 
    "r": "spline_r'.txt",
    "i": "spline_i'.txt",
    "z": "spline_z'.txt"
}

# ⚙️ PARÁMETROS DE PROCESAMIENTO
# ===============================
PROCESSING_CONFIG = {
    # Fotometría sintética
    "overlap_threshold": 0.95,  # Umbral mínimo de overlap spectro-filtro
    
    # Suavizado LOESS
    "loess_alpha_many": [0.5, 0.5],  # Alpha si muchos puntos (>40)
    "loess_alpha_few": [0.5],        # Alpha si pocos puntos (<=40)
    "loess_cutoff": 40,               # Umbral para decidir alpha
    "loess_corte": 30,                # Parámetro de corte LOESS
    
    # Ruido fotométrico
    "noise_level": 0.15,              # 15% de ruido poissoniano en flujo
    
    # Proyección
    "offset_range": [-30, 30],        # Rango de offsets temporales (días)
    "offset_step": 1,                 # Paso del offset
    
    # Debug y visualización
    "show_debug_plots": False         # Mostrar gráfico de debug en field_projection
}

# 🌫️ CONFIGURACIÓN DE EXTINCIÓN
# ===============================
EXTINCTION_CONFIG = {
    # Parámetros para SNe Ia (distribución exponencial académicamente validada)
    "SNIa": {
        "tau": 0.4,            # Holwerda et al. (2014) - académicamente validado
        "Av_max": 3.0,         # Corte físico en A_V
        "Rv": 3.1              # R_V = A_V / E(B-V)
    },
    
    # Parámetros para SNe core-collapse (UNIFICADO: distribución exponencial)
    # Eliminadas distribuciones mixtas por falta de justificación académica
    "SNII": {
        "tau": 0.4,            # Unificado con SNe Ia por consistencia científica
        "Av_max": 3.0,         # Corte físico en A_V
        "Rv": 3.1              # R_V = A_V / E(B-V)
        # ELIMINADOS: f_dusty, tau_exp, sigma_gauss (distribuciones mixtas)
    },
    
    "SNIbc": {
        "tau": 0.4,            # Unificado con SNe Ia por consistencia científica
        "Av_max": 3.0,         # Corte físico en A_V
        "Rv": 3.1              # R_V = A_V / E(B-V)
        # ELIMINADOS: f_dusty, tau_exp, sigma_gauss (distribuciones mixtas)
    },
    
    # Configuración general
    "random_seed": None,           # None = aleatorio, número = reproducible
    "use_reproducible_sampling": False  # Controla si usar semilla fija
}

# 🎨 CONFIGURACIÓN DE GRÁFICOS
# =============================
PLOT_CONFIG = {
    "dpi": 300,                       # Resolución de gráficos
    "figsize": [15, 12],              # Tamaño de figura
    "style": "default",               # Estilo matplotlib
    "colors": {
        "detections": "green",
        "upper_limits": "red", 
        "synthetic_original": "blue",
        "synthetic_noisy": "red"
    }
}

# 🔍 CONFIGURACIÓN DE VALIDACIÓN
# ===============================
VALIDATION = {
    "min_overlap": 0.90,              # Mínimo overlap aceptable
    "max_noise_sigma": 0.5,           # Máximo ruido aceptable
    "min_detection_rate": 0.0,        # Mínima tasa de detección aceptable
    "check_file_existence": True      # Verificar que archivos existan
}

# 🚀 CONFIGURACIÓN PARA RUNS MÚLTIPLES
# =====================================
BATCH_CONFIG = {
    "default_n_runs": 10,             # Número default de runs
    "pause_between_runs": 0.1,        # Pausa entre runs (segundos)
    "auto_update_index": True,        # Actualizar índice automáticamente
    "parallel_processing": False      # Procesamiento paralelo (futuro)
}
