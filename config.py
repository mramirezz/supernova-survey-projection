# 🔧 ARCHIVO DE CONFIGURACIÓN - PROYECCIÓN DE SUPERNOVAS
# ======================================================

# 🎛️ CONFIGURACIÓN PRINCIPAL
# ===========================

# Survey a utilizar
SURVEY = "ZTF"  # Opciones: "ZTF", "SUDARE"

# 🌟 PARÁMETROS DE LA SUPERNOVA
# ==============================
SN_CONFIG = {
    "sn_name": "SNexamplename",
    "tipo": "exampletype",
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
        "obslog_file": "ZTF_observing_log_complete.csv",  # Relativo a data_dir
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
    # Overrides por tipo (p.ej. para forzar ver fases tardías en II)
    # Formato: {"II": [min_days, max_days], ...}
    # Nota: se aplica ANTES de construir np.arange(min, max, step).
    "offset_range_by_type": {
        "II": [-120, 120],
    },

    # Redshift max para batches (si no se pasa por CLI)
    # Se usa para construir el rango (0.01, redshift_max)
    "redshift_max": 0.5,

    # Redshift fijo para batches (None = muestrear). Si se define, anula el muestreo cosmológico.
    "fixed_redshift": None,

    # Redshift fallback para `run_sn_list_multiband.py` cuando una fila viene sin z.
    # Esto es específico para tu lista (escala local), y evita hardcodearlo en el script.
    "sn_list_redshift_min": 0.01,
    "sn_list_redshift_max": 0.03,

    # Reintentos para cumplir mínimo de detecciones (solo para runner por lista)
    # Si está activo, para cada fila se re-proyecta (cambiando el azar del offset/noise)
    # manteniendo OID y z fijos, hasta llegar al mínimo o agotar los intentos.
    "sn_list_require_min_detections": True,
    "sn_list_min_detections": 7,
    "sn_list_max_attempts": 10,
    # Si tras max_attempts no se cumple el mínimo, reintentar con OTRO template
    # (misma fila: mismo OID/z/extinciones, solo cambia el template).
    "sn_list_try_new_template_on_fail": True,
    "sn_list_max_template_attempts": 5,
    # “Último recurso” (solo en el último intento del runner por lista):
    # cuánto se permite brightenear (en mag) para forzar >= min_detections en la banda requerida.
    "sn_list_force_max_brightening_mag": 3.0,
    
    # Campo fijo para pruebas (None = aleatorio)
    "fixed_field": None,              # Ej: "ZTF18aaqeasu" para siempre usar ese campo
    
    # Debug y visualización
    "show_debug_plots": False         # Mostrar gráfico de debug en field_projection
}

# ⭐ NORMALIZACIÓN DE LUMINOSIDAD (PARCHE PARA TEMPLATES SUBLUMINOSOS)
# ==================================================================
# Problema: muchos templates (especialmente II / Ibc) vienen en una escala de flujo
# que no representa una distribución poblacional "típica". Aunque estén a 10 pc
# (magnitud absoluta), algunos son intrínsecamente subluminosos (p.ej. SN2005cs)
# y otros no; además, a veces los templates no están homogenizados entre sí.
#
# Solución: usar el template SOLO como "forma" (SED + evolución temporal) y
# renormalizarlo para que su máximo (peak) corresponda a un M_peak muestreado
# desde una distribución por tipo. Esto evita que "la mayoría" salga demasiado débil
# por culpa del set de templates.
#
# Nota: Se aplica como un SHIFT en magnitudes (misma corrección para todos los filtros),
# manteniendo colores/forma, solo cambiando el nivel absoluto.
LUMINOSITY_CONFIG = {
    # Activar/desactivar globalmente
    "enabled": True,

    # Tipos a los que se aplica (por defecto: los problemáticos)
    "apply_to_types": ["II"],

    # Filtro de referencia para medir el peak del template (para ZTF, 'r' suele ser estable)
    # En multibanda, si este filtro no existe para el template, se usa el primer filtro disponible.
    "reference_filter": "r",

    # Distribuciones de M_peak (magnitud absoluta al peak) por tipo.
    # Valores aproximados para "sanity check"/parche práctico (puedes afinarlos a tu paper).
    "M_peak": {
        "Ia":  {"mean": -19.3, "sigma": 0.3},
        "Ibc": {"mean": -17.3, "sigma": 0.9},
        "II":  {"mean": -16.9, "sigma": 1.1},#-16.9 para SNII
    },

    # Clip físico para evitar extremos absurdos al muestrear
    "clip": {"min": -21.5, "max": -13.0},

    # Reproducibilidad opcional del muestreo de M_peak
    "random_seed": None,                 # None = aleatorio
    "use_reproducible_sampling": False   # True = usa random_seed si no es None
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
