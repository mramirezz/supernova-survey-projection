# CONFIGURACIÓN SIMPLIFICADA PARA BATCH RUNNER
# =============================================
# Solo las configuraciones esenciales para el modo custom

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np
import os
import glob

# Permite silenciar prints al importar (útil para runners con barra de progreso)
_SIMPLE_CONFIG_QUIET = str(os.environ.get("SIMPLE_CONFIG_QUIET", "0")).strip().lower() in {"1", "true", "yes", "on"}

class SNType(Enum):
    IA = "Ia"
    IBC = "Ibc" 
    II = "II"

class Survey(Enum):
    ZTF = "ZTF"
    SUDARE = "SUDARE"

@dataclass
class SimpleConfig:
    """Configuración simplificada para batch runs"""
    n_runs: int = 100
    redshift_range: Tuple[float, float] = (0.01, 0.3)
    fixed_redshift: Optional[float] = None  # Si se define, anula el muestreo cosmológico
    sn_type_distribution: Dict[str, float] = None
    survey_distribution: Dict[str, float] = None
    base_seed: int = 42
    volume_weighted: bool = True
    description: str = "Batch personalizado"
    pause_between_runs: float = 0.0  # Sin pausa entre runs
    filter_band: str = "r"  # Filtro fotométrico acoplado
    
    def __post_init__(self):
        if self.sn_type_distribution is None:
            self.sn_type_distribution = {"Ia": 1.0}
        
        if self.survey_distribution is None:
            self.survey_distribution = {"ZTF": 1.0}

def create_simple_config(n_runs: int = 100, 
                        redshift_max: float = 0.3,
                        fixed_redshift: Optional[float] = None,
                        sn_types: List[str] = None,
                        sn_type_distribution: dict = None,
                        survey: str = "ZTF",
                        seed: int = 42,
                        filter_band: str = "r") -> SimpleConfig:
    """
    Crea una configuración simplificada
    
    Parameters:
    -----------
    n_runs : int
        Número de simulaciones
    redshift_max : float
        Redshift máximo
    sn_types : List[str]
        Lista de tipos de SN
    sn_type_distribution : dict
        Distribución custom de tipos. Si None, usa distribución uniforme
    survey : str
        Survey principal
    seed : int
        Semilla para reproducibilidad
    filter_band : str
        Filtro fotométrico para síntesis y proyección
    """
    if sn_types is None:
        sn_types = ["Ia"]
    
    # Usar distribución custom o uniforme
    if sn_type_distribution is None:
        sn_distribution = {sn_type: 1.0/len(sn_types) for sn_type in sn_types}
    else:
        sn_distribution = sn_type_distribution
    
    # Survey único
    survey_distribution = {survey: 1.0}
    
    return SimpleConfig(
        n_runs=n_runs,
        redshift_range=(0.01, redshift_max),
        fixed_redshift=fixed_redshift,
        sn_type_distribution=sn_distribution,
        survey_distribution=survey_distribution,
        base_seed=seed,
        volume_weighted=True,
        filter_band=filter_band,
        description=f"Simulación de {n_runs} runs, z_max={redshift_max}, tipos={sn_types}"
    )

# ============================================================================
# WHITELIST DE TEMPLATES (OPCIONAL)
# ============================================================================
# Define qué templates usar por tipo. Si None, usa todos los encontrados.
# Útil para limitar sin borrar archivos.

SN_WHITELIST = {
    "Ia": None,  # None = usar todos
    "Ibc": None,  # None = usar todos
    # --------------------------------------------------------------------
    # Whitelist SN II (curada manualmente)
    # Nota: los nombres deben coincidir EXACTO con los archivos en data/II/.
    # --------------------------------------------------------------------
    "II": {
        # SN1999gi: falta un buen premax (curva corta) pero puede aportar algo
        "SN1999gi.dat",
        # SN2002hx: no tiene premax, pero la caída puede servir
        "SN2002hx.dat",
        # SN2003hn: sin premax, buena caída
        "SN2003hn.dat",
        # SN2004dj: sí tiene premax
        "SN2004dj.dat",
        # SN2004et: sin máximo, buena caída
        "SN2004et.dat",
        # SN2005cs: incluye fase de transición (sin subida)
        "SN2005cs.dat",
        # SN2007aa: sin subida, buena caída
        "SN2007aa.dat",
        # SN2013ej: “una maravilla”
        "SN2013ej.dat",
        # SN2014cy: por si acaso
        "SN2014cy.dat",
    },
}

def scan_sn_templates() -> Dict[str, List[str]]:
    """
    Escanea automáticamente las carpetas data/ para encontrar plantillas de SN.
    Aplica whitelist si está definida en SN_WHITELIST.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "data")
    
    templates = {
        "Ia": [],
        "Ibc": [], 
        "II": []
    }
    
    # Escanear cada tipo de SN
    for sn_type in templates.keys():
        type_path = os.path.join(data_path, sn_type)
        if os.path.exists(type_path):
            # Buscar archivos .dat
            pattern = os.path.join(type_path, "*.dat")
            files = glob.glob(pattern)
            all_files = [os.path.basename(f) for f in files]
            
            # Aplicar whitelist si existe
            whitelist = SN_WHITELIST.get(sn_type, None)
            if whitelist is not None:
                # Filtrar solo los permitidos
                templates[sn_type] = [f for f in all_files if f in whitelist]
                if not _SIMPLE_CONFIG_QUIET:
                    print(f"[OK] Tipo {sn_type}: {len(all_files)} disponibles, {len(templates[sn_type])} permitidos (whitelist)")
                if len(templates[sn_type]) < len(whitelist):
                    missing = whitelist - set(templates[sn_type])
                    if not _SIMPLE_CONFIG_QUIET:
                        print(f"[WARNING] Templates en whitelist no encontrados: {missing}")
            else:
                # Usar todos
                templates[sn_type] = all_files
                if not _SIMPLE_CONFIG_QUIET:
                    print(f"[OK] Tipo {sn_type}: {len(templates[sn_type])} templates (sin whitelist)")
        else:
            if not _SIMPLE_CONFIG_QUIET:
                print(f"[WARNING] Carpeta {type_path} no encontrada")
    
    return templates

# Plantillas disponibles por tipo (escaneadas automáticamente)
SN_TEMPLATES = scan_sn_templates()

def get_sn_templates() -> Dict[SNType, List[str]]:
    """Retorna diccionario de plantillas por tipo usando SNType enum"""
    return {
        SNType.IA: SN_TEMPLATES["Ia"],
        SNType.IBC: SN_TEMPLATES["Ibc"], 
        SNType.II: SN_TEMPLATES["II"]
    }
