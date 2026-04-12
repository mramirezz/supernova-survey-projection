# BATCH RUNNER SIMPLIFICADO
# ==========================
# Solo modo custom para simulaciones de supernovas
# Código limpio y directo

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Config global (para defaults cómodos)
import config

# Importar solo lo esencial
from batch_runner import run_scientific_batch
from simple_config import create_simple_config


def print_banner():
    """Banner simple del sistema"""
    print("="*60)
    print(" SIMULACIÓN DE PROYECCIÓN DE SUPERNOVAS")
    print("="*60)
    print(f" Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print()

def setup_environment():
    """Configura directorios básicos"""
    dirs_to_create = [
        "outputs/batch_runs",
        "logs"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def run_custom_batch(n_runs: int, redshift_max: float = None,
                    fixed_redshift: float = None,
                    sn_types: list = None, survey: str = "ZTF", 
                    seed: int = 42, filter_band: str = "r"):
    """
    Ejecuta un batch personalizado de simulaciones
    
    Parameters:
    -----------
    n_runs : int
        Número de simulaciones
    redshift_max : float
        Redshift máximo
    sn_types : list
        Lista de tipos de SN a incluir
    survey : str
        Survey principal
    seed : int
        Semilla para reproducibilidad
    filter_band : str
        Filtro fotométrico para síntesis y proyección
    """
    print(f">> Configurando batch personalizado...")
    
    # Configurar tipos de SN
    if sn_types is None or len(sn_types) == 0:
        print("[ERROR] Debes especificar al menos un tipo de supernova con --sn-types (Ia, Ibc, II)")
        return False

    
    # Si no se pasa por CLI, usar default desde config.py
    if redshift_max is None:
        redshift_max = config.PROCESSING_CONFIG.get('redshift_max', 0.3)
    if fixed_redshift is None:
        fixed_redshift = config.PROCESSING_CONFIG.get('fixed_redshift', None)

    # Crear configuración simplificada
    config = create_simple_config(
        n_runs=n_runs,
        redshift_max=redshift_max,
        fixed_redshift=fixed_redshift,
        sn_types=sn_types,
        survey=survey,
        seed=seed,
        filter_band=filter_band
    )
    
    # Actualizar descripción
    config.description = f"Batch personalizado - {n_runs} simulaciones"
    
    print(f"Configuracion:")
    print(f"    Simulaciones: {n_runs}")
    print(f"    Redshift max: {redshift_max}")
    if fixed_redshift is not None:
        print(f"    Redshift fijo: {fixed_redshift}")
    print(f"    Tipos SN: {sn_types}")
    print(f"    Survey: {survey}")
    print(f"    Filtro: {filter_band}")
    print(f"    Semilla: {seed}")
    print()
    
    # Ejecutar
    try:
        print(">> Iniciando simulaciones...")
        results = run_scientific_batch(config)
        
        print("[OK] Simulaciones completadas")
        print(f" Tasa de éxito: {results['run_statistics']['success_rate']:.1%}")
        print(f"  Duración: {results['batch_metadata']['total_duration_formatted']}")
        print(f" Detecciones: {results['detection_statistics']['total_detections']}")
        print(f" Observaciones: {results['detection_statistics']['total_observations']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error durante las simulaciones: {str(e)}")
        return False

def list_recent_batches():
    """Lista los últimos 10 batches"""
    print("Batches recientes:")
    print("-" * 40)
    
    batch_dir = Path("outputs/batch_runs")
    if not batch_dir.exists():
        print("[INFO] No hay batches disponibles")
        return
    
    batch_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()], reverse=True)[:10]
    
    if not batch_dirs:
        print("[INFO] No hay batches disponibles")
        return
    
    for i, batch_path in enumerate(batch_dirs):
        batch_id = batch_path.name
        print(f"{i+1:2d}. {batch_id}")

def main():
    """Función principal simplificada"""
    parser = argparse.ArgumentParser(
        description="Sistema Simplificado de Simulación de Supernovas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Batch básico (filtro r por defecto)
  python simple_runner.py --runs 100

  # Batch con filtro específico
  python simple_runner.py --runs 100 --filter g --sn-types Ia

  # Batch personalizado completo
  python simple_runner.py --runs 500 --redshift-max 0.2 --sn-types Ia Ibc --filter r --survey ZTF

  # Exploración de filtros múltiples
  python simple_runner.py --runs 200 --filter i --redshift-max 0.3 --sn-types Ibc

  # Ver batches recientes
  python simple_runner.py --list
        """
    )
    
    # Opciones principales
    parser.add_argument("--runs", type=int, default=100,
                       help="Número de simulaciones (default: 100)")
    
    parser.add_argument("--redshift-max", type=float, default=None,
                       help="Redshift máximo (default: config.PROCESSING_CONFIG['redshift_max'])")

    parser.add_argument("--fixed-redshift", type=float, default=None,
                       help="Si se define, fija z para todos los runs (anula el muestreo)")
    
    parser.add_argument("--sn-types", nargs="+", choices=["Ia", "Ibc", "II"],
                       default=["Ia"], help="Tipos de SN a incluir")
    
    parser.add_argument("--survey", choices=["ZTF", "SUDARE"], default="ZTF",
                       help="Survey principal (default: ZTF)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Semilla para reproducibilidad (default: 42)")
    
    parser.add_argument("--filter", choices=["U", "B", "V", "R", "I", "u", "g", "r", "i", "z"], 
                       default="r", help="Filtro fotométrico para síntesis y proyección (default: r)")
    
    parser.add_argument("--list", action="store_true",
                       help="Listar batches recientes")
    
    args = parser.parse_args()
    
    # Mostrar banner
    print_banner()
    
    # Configurar entorno
    setup_environment()
    
    # Ejecutar acción
    if args.list:
        list_recent_batches()
    else:
        success = run_custom_batch(
            n_runs=args.runs,
            redshift_max=args.redshift_max,
            fixed_redshift=args.fixed_redshift,
            sn_types=args.sn_types,
            survey=args.survey,
            seed=args.seed,
            filter_band=args.filter
        )
        
        if success:
            print("\n[OK] Operacion completada exitosamente")
            sys.exit(0)
        else:
            print("\n[ERROR] Operacion fallo")
            sys.exit(1)

if __name__ == "__main__":
    main()
