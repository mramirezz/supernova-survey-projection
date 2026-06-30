# 🔧 ARCHIVO DE CONFIGURACIÓN - PROYECCIÓN DE SUPERNOVAS
# ======================================================
import platform
import os
from pathlib import Path

# 🖥️ DETECCIÓN AUTOMÁTICA DE RUTAS POR SISTEMA OPERATIVO
# ========================================================
_system = platform.system()
if _system == "Darwin":  # macOS
    GDRIVE_BASE = Path(os.path.expanduser(
        "~/Library/CloudStorage/GoogleDrive-mramirez.valenzuela@gmail.com/Mi unidad"
    ))
elif _system == "Linux":
    GDRIVE_BASE = Path("/mnt/g/Mi unidad")
else:  # Windows
    GDRIVE_BASE = Path(r"G:\Mi unidad")

BASE_DIR = GDRIVE_BASE / "Work" / "Universidad" / "Phd" / "paper2_ZTF" / "Codes" / "proyeccion"
RESPONSE_FOLDER = GDRIVE_BASE / "Work" / "Universidad" / "Phd" / "Practica2" / "Splines_eachfilter_2"
MAXIMUM_DIR = GDRIVE_BASE / "Work" / "OT_unidos_2" / "OT_combines_test"
MAXIMUM_IBC_PATH = GDRIVE_BASE / "Work" / "Universidad" / "Phd" / "Practica2" / "maximum_Ibc.dat"
MODULUS_PATH = GDRIVE_BASE / "Work" / "Universidad" / "Phd" / "DATA_OSC" / "all_data" / "modulos_query_all.cvs"

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
    "base_dir": str(BASE_DIR),
    
    # Espectros de supernovas
    "data_dir": str(BASE_DIR / "data"),
    "spec_file": "Ia/ASASSN-14lp.dat",  # Relativo a data_dir
    
    # Curvas de respuesta de filtros
    "response_folder": str(RESPONSE_FOLDER),
    
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

# 🌌 RANGO DE REDSHIFT POR TIPO (adaptado a la sensibilidad de ZTF)
# ================================================================
# ZTF tiene m_lim ≈ 19.5–20.5 en g/r. Con M_peak_Ia ≈ -19.3, M_peak_II ≈ -16.9,
# M_peak_Ibc ≈ -17.3, el redshift al que cada tipo deja de ser detectable es
# muy distinto. Usar el mismo z_max=0.5 para los 3 produce ~0% detecciones en II
# y baja tasa en Ibc. Valores calibrados a m_lim=20.5:
#   Ia  M_peak=-19.3 → z_max ≈ 0.19
#   II  M_peak=-16.9 → z_max ≈ 0.07
#   Ibc M_peak=-17.3 → z_max ≈ 0.08
# Se usa un margen razonable por encima para capturar también la región donde
# la SN sube a upper-limit (útil para el clasificador).
Z_CONFIG = {
    "z_min": 0.01,
    "z_max_by_type": {
        "Ia":  0.15,
        "II":  0.08,
        "Ibc": 0.10,
    },
    # Override global — si no es None, se usa este z_max para todos los tipos
    # (ignorando z_max_by_type). Útil para corridas de debug o exploración.
    "z_max_global_override": None,
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

    # Tipos a los que se aplica
    "apply_to_types": ["Ia", "II", "Ibc"],

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

# 🌫️ CONFIGURACIÓN DE EXTINCIÓN DE HOST  (FUENTE ÚNICA DE VERDAD)
# ================================================================
# Modelo de MEZCLA por tipo: una fracción `frac_zero` SIN polvo (E(B-V)≈0, entornos
# limpios) + el resto con una cola EXPONENCIAL en A_V de escala `tau`. Es el enfoque
# estándar en simulaciones de SNe (Kessler+2009; Brout & Scolnic 2021), y la
# distribución de core-collapse sigue a Hatano+1998 como en los frameworks modernos
# de simulación de surveys (Vincenzi+2019, usado en DES/LSST).
#
# IMPORTANTE (fix 2026-06-28): `sample_extinction_by_type()` LEE de este dict.
# Antes los valores estaban hardcodeados en core/correction.py y este dict solo se
# guardaba en run_metadata.json -> la metadata mentía. Ahora hay una sola fuente.
EXTINCTION_CONFIG = {
    # --- SNe Ia: poblaciones viejas Y jóvenes -> fracción alta sin polvo ---
    "SNIa": {
        "tau":        0.35,  # escala exp. de A_V de la componente con polvo (Holwerda+2015, 2015MNRAS.446.3768H)
        "frac_zero":  0.40,  # Ia en poblaciones limpias+polvorientas (Holwerda+2015; Brout&Scolnic+2021, 2021ApJ...909...26B)
        "sigma_zero": 0.01,  # dispersión (mag) de la componente sin-polvo en E(B-V)
        "Av_max":     3.0,   # cap numérico (P(A_V>3)<0.5% con este tau; inocuo)
        "Rv":         3.1,   # MW canónico, Cardelli+1989
    },
    # --- SNe II: solo en regiones star-forming, PERO reddening de host BAJO ---
    # de Jaeger+2018: "host reddening is not a dominant parameter" para las II.
    # Consistente con la sub-muestra DETECTABLE de CC de Hatano+1998 (<A_V>~0.2).
    "SNII": {
        "tau":        0.25,  # de Jaeger+2018 (II reddening bajo) + Hatano+1998 (CC detectable)
        "frac_zero":  0.20,  # CC en SF -> menos eventos limpios que Ia
        "sigma_zero": 0.01,
        "Av_max":     3.0,
        "Rv":         3.1,   # Cardelli+1989
    },
    # --- SNe Ibc (stripped-envelope): MÁS extinguidas que las II ---
    # II < Ibc por ~2x: Prentice 2016 (vía Vincenzi+2019) da Ib/Ic 2-3x mas que II;
    # Stritzinger+2018 mide <A_V>~0.5 para SE SNe del CSP-I (las Ic, las mas rojas).
    "SNIbc": {
        "tau":        0.50,  # Stritzinger+2018 (CSP-I SE SNe, <A_V>~0.5)
        "frac_zero":  0.20,
        "sigma_zero": 0.01,
        "Av_max":     3.0,
        "Rv":         3.1,   # Cardelli+1989. Nota: Stritzinger ve Rv por subtipo
                             # (IIb~1.1, Ic~4.3); como mezclamos todo Ibc usamos un Rv único.
    },
    "random_seed": None,           # None = aleatorio
    "use_reproducible_sampling": False,
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

# ============================================================
# 🎯 WHITELIST DE TEMPLATES POR TIPO
# ============================================================
# Limita qué templates se proyectan, sin borrar archivos de data/.
# None = usar todos los .dat de la carpeta. Los nombres deben coincidir EXACTO
# con los archivos en data/<tipo>/ (con .dat).
#
# Ia / Ibc: la carpeta data/ ya es la curación (13 Ia, 22 Ibc) -> None.
# II: data/II tiene 31, pero solo 8 pasan el criterio de calidad (forma útil de
#     la serie espectral). Ver bitácora de Proyección, 2026-06-27 (manifiesto de
#     templates) para la justificación y los rangos de cada SN.
SN_WHITELIST = {
    "Ia": None,
    "Ibc": None,
    "II": {
        "SN1999gi.dat",  # premax corto, puede aportar
        "SN2002hx.dat",  # sin premax, la caída sirve
        "SN2003hn.dat",  # sin premax, buena caída
        "SN2004et.dat",  # sin máximo, buena caída
        "SN2005cs.dat",  # rise temprano (~3 d post-explosión)
        "SN2007aa.dat",  # sin subida, buena caída
        "SN2013ej.dat",  # "una maravilla"
        "SN2014cy.dat",  # por si acaso
        # SN2004dj.dat REMOVIDA (2026-06-27): serie espectral solo nebular
        # (1er espectro a +200 d, sin plateau) -> LC sintética irreal.
    },
}
