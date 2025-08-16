# CONFIGURACIÓN SIMPLIFICADA PARA BATCH RUNNER
# =============================================
# Solo las configuraciones esenciales para el modo custom

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import numpy as np
import os
import glob

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
    sn_type_distribution: Dict[str, float] = None
    survey_distribution: Dict[str, float] = None
    base_seed: int = 42
    volume_weighted: bool = True
    description: str = "Batch personalizado"
    pause_between_runs: float = 0.0  # Sin pausa entre runs
    
    def __post_init__(self):
        if self.sn_type_distribution is None:
            self.sn_type_distribution = {"Ia": 1.0}
        
        if self.survey_distribution is None:
            self.survey_distribution = {"ZTF": 1.0}

def create_simple_config(n_runs: int = 100, 
                        redshift_max: float = 0.3,
                        sn_types: List[str] = None,
                        survey: str = "ZTF",
                        seed: int = 42) -> SimpleConfig:
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
    survey : str
        Survey principal
    seed : int
        Semilla para reproducibilidad
    """
    if sn_types is None:
        sn_types = ["Ia"]
    
    # Distribución uniforme entre tipos de SN
    sn_distribution = {sn_type: 1.0/len(sn_types) for sn_type in sn_types}
    
    # Survey único
    survey_distribution = {survey: 1.0}
    
    return SimpleConfig(
        n_runs=n_runs,
        redshift_range=(0.01, redshift_max),
        sn_type_distribution=sn_distribution,
        survey_distribution=survey_distribution,
        base_seed=seed,
        volume_weighted=True,
        description=f"Simulación de {n_runs} runs, z_max={redshift_max}, tipos={sn_types}"
    )

def scan_sn_templates() -> Dict[str, List[str]]:
    """Escanea automáticamente las carpetas data/ para encontrar plantillas de SN"""
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
            # Extraer solo los nombres de archivo
            templates[sn_type] = [os.path.basename(f) for f in files]
            print(f"✅ Encontradas {len(templates[sn_type])} SNe {sn_type}: {templates[sn_type]}")
        else:
            print(f"⚠️  Carpeta {type_path} no encontrada")
    
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
