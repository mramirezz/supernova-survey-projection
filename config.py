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
    # Épocas del obslog ANTERIORES al inicio de la SN, emitidas como upper limits
    # (la SN aún no explota -> no-detección por construcción). Mimetiza que ZTF ya
    # monitoreaba el campo antes de la explosión, igual que para las SNe reales:
    # sin esto, las sims brillantes se detectan desde su 1a época y nunca tienen
    # upper limits pre-detección, y el filtro de calidad del feature-extractor las
    # bota (auditoría banda g, bitácora Proyección 2026-08-02). Días antes del
    # inicio de la curva; 0 = comportamiento antiguo (sin épocas pre-explosión).
    "pre_explosion_ul_days": 25,
    
    # Suavizado LOESS de la fotometria sintetica. APAGADO desde 2026-10-02
    # (decision Mauricio): las 78 series congeladas son diarias y ya suaves, y
    # el LOESS (span 0.5, partido en el punto 30) las deformaba hasta 0.7-0.9 mag
    # (2011fe g en el empalme del dia 30, r -0.39 en el 2do maximo; 2012aw 0.88
    # en la 1a epoca). True solo para reproducir los runs con templates viejos.
    "loess_smooth": False,
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
        # IIb: rise 22±3 d (Taddia+2015, 2015A&A...574A..60T) + declive dm15(r)
        # 0.67 y 0.016-0.021 mag/d tardio (Taddia+2018) -> +100 d cubre la parte
        # detectable a z<=0.11 sin poblar de colas invisibles.
        "IIb": [-30, 100],
        # SLSN-I: rise 10-90 d y decay ~1.5x rise en el REST frame (Chen+2023);
        # [-90,+250] rest x (1+z~0.3) del run tipico. El grid es en dias
        # OBSERVADOS, por eso va pre-multiplicado.
        "SLSN-I": [-120, 330],
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
    # 'empirical': z resampleado de la muestra real etiquetada (data/z_empirical_<tipo>.txt),
    #              para el run de ENTRENAMIENTO del clasificador.
    # 'volumetric': dV/dz hasta z_max_by_type, para el run de TASAS (mide eta(z)).
    "z_mode": "empirical",
    # Esquema DOS RUNS (bitácora Proyección 2026-07-02):
    #  - run de ENTRENAMIENTO (clasificador): z matcheado a la muestra etiquetada
    #    de ZTF -> Ia 0.08 / II 0.05 / Ibc 0.06  (valores ACTIVOS)
    #  - run VOLUMÉTRICO (tasas, mide eta(z)):  Ia 0.15 / II 0.08 / Ibc 0.10
    "z_max_by_type": {
        "Ia":  0.08,
        "II":  0.05,
        "Ibc": 0.06,
        # IIb: M_peak(r)=-17.45+/-0.54 (Taddia+2018) -> cola brillante +1sigma
        # cruza m_lim=20.5 en z~0.11. Consistente con BTS (CC a z<~0.05 con m<19).
        "IIb": 0.11,
        # SLSN-I: z maximo OBSERVADO de la muestra ZTF-I (Chen+2023, 78 SNe,
        # z=0.06-0.67, mediana 0.265). La mediana de M cruza 20.5 en z~0.45.
        # OJO: sin z_empirical_SLSN-I.txt cae a muestreo volumetrico hasta este
        # z_max (pendiente transcribir los z de la Tabla A4 de Chen para el run
        # de entrenamiento).
        "SLSN-I": 0.67,
        # IIn: M_r medio -19.18 (Nyholm+2020), igual de brillante que una Ia ->
        # mismo z_max de entrenamiento que Ia. PROVISIONAL (2026-10-02): no hay
        # z_empirical_IIn.txt (la muestra real ZTF no tiene IIn), cae a dV/dz.
        "IIn": 0.08,
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
    "apply_to_types": ["Ia", "II", "Ibc", "Ib", "Ic", "Ic-BL", "IIb", "IIn", "SLSN-I"],

    # Filtro de referencia para medir el peak del template (para ZTF, 'r' suele ser estable)
    # En multibanda, si este filtro no existe para el template, se usa el primer filtro disponible.
    "reference_filter": "r",

    # Distribuciones de M_peak (magnitud absoluta al peak) por tipo.
    # Valores aproximados para "sanity check"/parche práctico (puedes afinarlos a tu paper).
    "M_peak": {
        "Ia":  {"mean": -19.3, "sigma": 0.3},
        "Ibc": {"mean": -17.3, "sigma": 0.9},
        # Ib, Ic, Ic-BL: Drout+2011 (2011ApJ...741...97D), banda R (Vega), corregidas
        # por extincion del host, decision Mauricio 2026-10-02 (G1). "Ibc" queda para el runner viejo.
        "Ib":    {"mean": -17.9, "sigma": 0.9},
        "Ic":    {"mean": -18.3, "sigma": 0.6},
        "Ic-BL": {"mean": -19.0, "sigma": 1.1},
        "II":  {"mean": -16.9, "sigma": 1.1},#-16.9 para SNII
        # IIb: banda r, MW+host corregido; Taddia+2018 (2018A&A...609A.136T,
        # Tabla 5, 10 IIb CSP-I). Cross-check Richardson+2014: M_B=-16.99+/-0.45.
        "IIb": {"mean": -17.45, "sigma": 0.54},
        # SLSN-I: banda g REST-FRAME con K-corr; Chen+2023 (2023ApJ...943...41C,
        # 78 SLSN-I de ZTF-I): mediana -21.48, percentiles 16/84 = -22.09/-20.35.
        # Distribucion ASIMETRICA (decision Mauricio 2026-08-15): split-normal
        # con sigma_bright hacia el lado brillante y sigma_faint hacia el debil.
        # El anclaje es en g REST (ver 'rest_anchor'), no en r observado.
        "SLSN-I": {"median": -21.48, "sigma_bright": 0.61, "sigma_faint": 1.13},
        # IIn: banda r con K-corr, PTF/iPTF no dirigido, 42 SNe IIn incluidas las
        # "superluminosas" IIn; Nyholm+2020 (2020A&A...637A..73N, verificado en
        # ADS 2026-10-02): M_r,peak = -19.18 +/- 1.32. Muestra limitada en
        # magnitud (sesgo Malmquist hacia brillantes). Decision Mauricio 2026-10-02.
        "IIn": {"mean": -19.18, "sigma": 1.32},
    },

    # Anclaje REST-FRAME por tipo (2026-08-15): para tipos cuya M de literatura
    # esta definida en una banda rest (SLSN-I: g rest con K-corr, Chen+2023), el
    # shift se calcula como M_target - M_template_rest (el M rest del template se
    # precomputa con precompute_Mrest_SLSN.py, analogo al dm15 de Phillips). Asi
    # la K-correction la hereda el SED del template y no se asume color. Para los
    # demas tipos se mantiene el anclaje historico en banda observada (K~0 a
    # z<=0.11, error despreciable).
    "rest_anchor": {
        "SLSN-I": {"filter": "g", "file": "Mrest_g_SLSN-I.json"},
    },

    # Modo de sorteo de M_peak (paridad con Z_CONFIG['z_mode']):
    #  'gaussian':  N(mean, sigma) de la tabla de arriba — para el run VOLUMÉTRICO
    #               (LF intrínseca, tasas).
    #  'empirical': resamplea de data/M_empirical_<tipo>.txt (M ajustados de la
    #               muestra real espectroscópica) — para el run de ENTRENAMIENTO,
    #               que debe matchear la muestra observada (Malmquist: mediana II
    #               real -17.42 vs -16.9 intrínseca; auditoría banda g 2026-08-02).
    #               Solo aplica a tipos con archivo; el resto cae a gaussian.
    #               Ia con Phillips activo ignora esto (WLR manda).
    "M_mode": "gaussian",

    # Clip físico para evitar extremos absurdos al muestrear
    # (min ampliado -21.5 -> -23.5 el 2026-08-15: las SLSN-I llegan a -22.8
    # observado en ZTF-I, Chen+2023; el clip viejo las cortaba)
    "clip": {"min": -23.5, "max": -13.0},

    # Reproducibilidad opcional del muestreo de M_peak
    "random_seed": None,                 # None = aleatorio
    "use_reproducible_sampling": False   # True = usa random_seed si no es None
}

# ==================================================================
# RELACION ANCHO-LUMINOSIDAD (PHILLIPS) PARA Ia
# ==================================================================
# La normalizacion de luminosidad de arriba sortea M_peak independiente de la
# FORMA del template. Para las Ia eso rompe la relacion de Phillips (las que
# caen rapido son intrinsecamente mas debiles; Phillips 1993, 1993ApJ...413L.105P),
# creando Ia no fisicas (brillantes+angostas o debiles+anchas). Con PHILLIPS
# activo, para las Ia la media del sorteo se ancla al ancho del template:
#     M_peak = M0 + slope * (dm15_template - dm15_ref) + N(0, sigma_resid)
# dm15_template = Delta-m15(B) rest-frame, medido sobre el template TAL COMO SE USA
# (post OT/deenrojecido) con precompute_dm15_Ia.py -> data/dm15_Ia.json. Se usa el
# dm15 medido y no el de literatura por autoconsistencia: la curva simulada hereda
# la forma del template procesado, asi que el acople brillo<->ancho debe usar esa
# forma (con Ia_new: SN2011fe medido 1.10; con las 15 v78 congeladas, 2026-10-02:
# 1.02, rango 0.76-1.46, 15/15 medidas; el json previo quedo en dm15_Ia_Ia_new.json).
#
# BANDA: el anclaje del peak es en r (LUMINOSITY_CONFIG reference_filter='r'), y la
# relacion es mas plana hacia el rojo (Phillips 1993). Por eso los TRES parametros
# salen de la calibracion en banda R de Prieto, Rest & Suntzeff 2006
# (2006ApJ...647..501P, Tabla 3, corregida por reddening del host, valida en
# 0.8<=dm15<=1.7): a_R = -19.248+/-0.025 (h=0.72) -> -19.31 con el H0=70 del
# pipeline; b_R = 0.566+/-0.101; sigma_R = 0.13.
# Cross-checks verificados en ADS: Hamuy+1996 (1996AJ....112.2391H) b_V=0.707
# (el 0.70 provisional anterior era ese valor de banda V, no r); Phillips+1999
# (1999AJ....118.1766P) sigma 0.09-0.13; Folatelli+2010 (2010AJ....139..120F)
# dispersion r 0.12-0.16; Burns+2018 (2018ApJ...869...56B) scatter intrinseco
# 0.13-0.18; Richardson+2014 (2014AJ....147..118R) M_B=-19.25+/-0.20 (H0=70).
#
# SEMANTICA: el M_peak muestreado actua como magnitud absoluta r LIBRE de
# extincion (el shift de normalizacion cancela el dimming de extincion en r; solo
# sobrevive la firma de color), consistente con una calibracion reddening-corrected.
# POBLACION resultante con las 15 Ia v78 (2026-10-02; dm15 medio 1.095, std 0.229,
# ciclado equiprobable): media M_peak = -19.31, sigma total
# sqrt((0.566*0.229)^2 + 0.13^2) = 0.18. (Con las 12 Ia_new era dm15 1.016 y media
# -19.36: ese set era mas lento que dm15_ref=1.1.) dm15_ref se mantiene en 1.1 porque
# es el pivote de la calibracion de Prieto (M0 esta definido AHI).
# Solo aplica a Ia: II/Ibc no tienen WLR fotometrica apretada y quedan igual.
PHILLIPS_CONFIG = {
    "enabled": True,
    "apply_to_types": ["Ia"],
    "M0": -19.31,         # a_R de Prieto+2006 Tabla 3, convertido a H0=70
    "slope": 0.566,       # b_R de Prieto+2006 Tabla 3 (banda R = la del anclaje)
    "dm15_ref": 1.1,      # pivote de la calibracion de Prieto+2006
    "sigma_resid": 0.13,  # sigma_R de Prieto+2006 (Burns+2018: 0.13-0.18)
    "dm15_file": "dm15_Ia.json",   # relativo a data/
    "dm15_default": 1.1,  # fallback si un template no esta en el json (avisa con WARNING)
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
    # HISTORICO: RECALIBRADO 2026-08-06 (ver opt_clasificador/DIAGNOSTICO_gcorr_leakage.md):
    # con tau=0.35/frac_zero=0.40 el M_peak_r observado de las Ia sinteticas quedaba
    # std 0.27 vs 0.55 (robusta) de las Ia reales ZTF -> Phillips las dejaba "clones"
    # y el clasificador tiraba a Ibc las Ia reales de las colas (recall 0.79->0.69).
    # tau=0.65/frac_zero=0.25 aporta ~0.50 mag de polvo en r (+0.13 resid Phillips
    # = 0.52-0.55 total ✓). El test ab_dust_test.py confirma que recupera Ia~0.78.
    # D4 2026-10: el M ahora es intrinseco y el polvo atenua; la recalibracion 0.65/0.25 del
    # 2026-08-06 suponia que la normalizacion cancelaba la extincion (H5). Se re-verifica la
    # dispersion en el piloto de la Tarea 8. VIGENTE: tau 0.35 / frac_zero 0.40 (literatura).
    "SNIa": {
        "tau":        0.35,  # escala exp. de A_V con polvo (Holwerda+2015, 2015MNRAS.446.3768H; D4 2026-10, vuelve a literatura)
        "frac_zero":  0.40,  # Holwerda+2015; Brout&Scolnic+2021, 2021ApJ...909...26B (D4 2026-10, vuelve a literatura)
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
    # --- SNe IIb (clase propia desde 2026-08-15) ---
    # Stritzinger+2018 (2018A&A...609A.135S, CSP-I): 3/10 IIb minimamente
    # enrojecidas -> frac_zero 0.30; tau = <A_V> de las 7 enrojecidas = 0.35
    # (calculo sobre Tablas 4-5). Rv=1.1 es el adoptado por ese paper para IIb
    # PERO derivado de UNA sola SN (2006T) — decision Mauricio: usar 1.1 fiel a
    # la fuente, con corrida de sensibilidad a Rv=3.1 pendiente.
    "SNIIb": {
        "tau":        0.35,
        "frac_zero":  0.30,
        "sigma_zero": 0.01,
        "Av_max":     3.0,
        "Rv":         1.1,
    },
    # --- SLSN-I: hosts enanas con muy poco polvo ---
    # Chen+2023: 71/78 SIN correccion de host; las 7 restantes E(B-V)=0.07-0.4
    # (A_V 0.22-1.24, media 0.64 -> tau 0.60). Cross-checks: 21/53 hosts con
    # E(B-V)_gas=0.00 y mediana 0.02 (Schulze+2018); Balmer bajos (Leloudas+2015,
    # Perley+2016). Rv=3.1 es convencion (sin medicion de host en la literatura).
    "SNSLSN": {
        "tau":        0.60,
        "frac_zero":  0.90,
        "sigma_zero": 0.01,
        "Av_max":     3.0,
        "Rv":         3.1,
    },
    # --- SNe IIn: PROVISIONAL = mismos parametros que SNII (decision Mauricio
    # 2026-10-02), hasta tener una distribucion de host propia para IIn.
    "SNIIn": {
        "tau":        0.25,
        "frac_zero":  0.20,
        "sigma_zero": 0.01,
        "Av_max":     3.0,
        "Rv":         3.1,
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
# 📁 CARPETAS DE TEMPLATES POR TIPO (data/<carpeta>/)
# ============================================================
# 2026-10-02: las 78 series CONGELADAS (aprobadas una a una por Mauricio),
# copiadas desde paper2_ZTF/series_aprobadas/<clase>/mangled/ con cmp archivo a
# archivo; md5 de cada copia en data/templates_v78_md5.csv. Reposo,
# desenrojecidas, a 10 pc, tiempo en MJD. NO editar estas carpetas: si cambia
# una serie congelada se re-copia desde series_aprobadas.
# Las carpetas *_new (15-08, EMPCA + mangling GP, metodo descartado) y las
# historicas data/{Ia,II,Ibc} quedan intactas solo como referencia.
# Cada tipo se proyecta con su etiqueta real (IIb, IIn incluidas); el mapeo a
# las clases del clasificador se decide al entrenar (decision Mauricio 2026-10-02).
# SLSN-I fuera: aparcadas sin revisar (SUDARE no tiene superluminosas, 2026-09-30).
TEMPLATE_DIRS = {
    "Ia": "Ia_v78",    # 15
    "II": "II_v78",    # 13
    "IIb": "IIb_v78",  # 10
    "IIn": "IIn_v78",  # 10
    "Ibc": "Ibc_v78",  # 30 (Ib, Ic y 7 Ic-BL)
}

# ============================================================
# 🎯 WHITELIST DE TEMPLATES POR TIPO
# ============================================================
# Limita qué templates se proyectan, sin borrar archivos de data/.
# None = usar todos los .dat de la carpeta. Los nombres deben coincidir EXACTO
# con los archivos en data/<carpeta>/ (con .dat).
#
# II: la whitelist de 4 SNe (curación 2026-08-02, déficit de banda g: solo
# series que parten <6 d de la explosión con g-r <= +0.1 en la 1a época) queda
# OBSOLETA con II_new: las 16 series de la campaña YA cumplen ese criterio por
# construcción (las 27 históricas que partían viejas no se rehicieron). La
# curación ahora vive en la selección de la carpeta, no en una lista aquí.
SN_WHITELIST = {
    "Ia": None,
    "Ibc": None,
    "II": None,
    "IIb": None,
    "SLSN-I": None,
}
