# 📦 CORE PACKAGE
# ================
# Paquete modular para proyección de supernovas

# Imports principales para facilitar el uso
from .utils import (
    leer_spec, Syntetic_photometry_v2, Loess_fit, maximo_lc, DL_calculator,
    cteB, cteV, cteR, cteI, cteU, cteu, cteg, cter, ctei, ctez
)
from .correction import correct_redeening, redden_spectrum_adjusted
from .projection import field_projection
from .save_functions import save_projection_results

__version__ = "1.0.0"
__author__ = "Tu Nombre"

# Lista de módulos disponibles
__all__ = [
    # Utils
    'leer_spec', 'Syntetic_photometry_v2', 'Loess_fit', 'maximo_lc', 'DL_calculator',
    # Correction
    'correct_redeening', 'redden_spectrum_adjusted', 
    # Projection
    'field_projection',
    # Save functions
    'save_projection_results',
    # Constants
    'cteB', 'cteV', 'cteR', 'cteI', 'cteU', 'cteu', 'cteg', 'cter', 'ctei', 'ctez'
]
