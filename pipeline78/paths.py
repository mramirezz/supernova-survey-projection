"""Rutas del pipeline v78. STORE y RUNS son locales (fuera de Drive) a proposito."""
import os
from pathlib import Path

GD = Path.home() / "Library/CloudStorage/GoogleDrive-mramirez.valenzuela@gmail.com/Mi unidad"
PHD = GD / "Work/Universidad/Phd"
REPO = PHD / "paper2_ZTF/Codes/proyeccion"
DATA = REPO / "data"
FILTERS = DATA / "filters"
LIB = PHD / "paper2_ZTF/series_aprobadas"
SUDARE_DIR = PHD / "paper2_ZTF/obslogsudare"
LEGACY_RESP = PHD / "Practica2/Splines_eachfilter_2"
ZLF = PHD / "paper2_ZTF/Codes/feature_extraction/ztf_literature_features"
OC = PHD / "paper2_ZTF/Codes/feature_extraction/opt_clasificador"
STORE = Path(os.environ.get("P78_STORE", Path.home() / "thesis_store"))
RUNS = Path(os.environ.get("P78_RUNS", Path.home() / "thesis_runs"))
