# CONFIGURACIÓN SIMPLIFICADA PARA BATCH RUNNER
# =============================================
# Solo las configuraciones esenciales para el modo custom

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import numpy as np

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

def create_cosmological_sample(redshift_range: Tuple[float, float], 
                              n_samples: int, 
                              volume_weighted: bool = True) -> np.ndarray:
    """
    Genera muestra cosmológica de redshifts
    
    Parameters:
    -----------
    redshift_range : Tuple[float, float]
        Rango (z_min, z_max)
    n_samples : int
        Número de muestras
    volume_weighted : bool
        Si usar volumen comóvil (True) o uniforme (False)
    """
    z_min, z_max = redshift_range
    
    if volume_weighted:
        # Muestreo proporcional al volumen comóvil (∝ z²)
        # Aproximación válida para z < 0.5
        z_samples = []
        for _ in range(n_samples):
            # Método de rechazo simple
            while len(z_samples) < n_samples:
                z_candidate = np.random.uniform(z_min, z_max)
                weight = (z_candidate / z_max) ** 2
                if np.random.uniform(0, 1) < weight:
                    z_samples.append(z_candidate)
                    break
        return np.array(z_samples)
    else:
        # Muestreo uniforme
        return np.random.uniform(z_min, z_max, n_samples)

# Plantillas disponibles por tipo
SN_TEMPLATES = {
    "Ia": [
        "ASASSN-14lp.dat", "SN1994D.dat", "SN1998aq.dat", "SN1998dh.dat",
        "SN2001V.dat", "SN2003du.dat", "SN2005hk.dat", "SN2007af.dat",
        "SN2007le.dat", "SN2009ig.dat", "SN2011fe.dat", "SN2012fr.dat", "SN2012ht.dat"
    ],
    "Ibc": [
        "SN1994I.dat", "SN1998bw.dat", "SN2002ap.dat"  # Placeholder - agregar más
    ],
    "II": [
        "SN1999em.dat", "SN2004et.dat", "SN2013ej.dat"  # Placeholder - agregar más
    ]
}

def get_sn_templates() -> Dict[SNType, List[str]]:
    """Retorna diccionario de plantillas por tipo usando SNType enum"""
    return {
        SNType.IA: SN_TEMPLATES["Ia"],
        SNType.IBC: SN_TEMPLATES["Ibc"], 
        SNType.II: SN_TEMPLATES["II"]
    }
