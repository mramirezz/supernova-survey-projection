"""
SIMPLE RUNNER MULTI-BANDA
=========================
Interfaz simplificada para ejecutar simulaciones multi-banda masivas.

DIFERENCIA CON simple_runner.py:
- Este proyecta en TODOS los filtros disponibles simultáneamente
- Usa el MISMO offset temporal para todos los filtros
- Mantiene la coherencia física: si una noche se observó en g+r, ambos aparecen
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Config global (para defaults cómodos)
import config

# Importar solo lo esencial
from batch_runner_multiband import run_multiband_batch
from simple_config import create_simple_config


def print_banner():
    """Banner del sistema multi-banda"""
    print("="*60)
    print(" SIMULACIÓN MULTI-BANDA DE PROYECCIÓN DE SUPERNOVAS")
    print("="*60)
    print(f" Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(" Modo: PROYECCIÓN SIMULTÁNEA EN MÚLTIPLES FILTROS")
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

def run_custom_multiband_batch(n_runs: int, redshift_max: float = None,
                                fixed_redshift: float = None,
                                sn_types: list = None, sn_ratios: list = None,
                                survey: str = "ZTF", seed: int = 42):
    """
    Ejecuta un batch multi-banda personalizado
    
    Parameters:
    -----------
    n_runs : int
        Número de simulaciones
    redshift_max : float
        Redshift máximo
    sn_types : list
        Lista de tipos de SN a incluir
    sn_ratios : list
        Ratios para cada tipo (debe sumar 1.0). Si None, distribuye equitativamente
    survey : str
        Survey a usar (ZTF o SUDARE)
    seed : int
        Semilla para reproducibilidad
        
    Returns:
    --------
    dict : Resultados del batch
    """
    if sn_types is None:
        sn_types = ["Ia"]
    
    print_banner()
    
    setup_environment()
    
    # Procesar ratios
    if sn_ratios is not None:
        # Validar que tenga la misma longitud que sn_types
        if len(sn_ratios) != len(sn_types):
            raise ValueError(f"--ratios debe tener {len(sn_types)} valores (uno por cada tipo en --sn-types)")
        
        # Validar que sumen ~1.0
        ratio_sum = sum(sn_ratios)
        if abs(ratio_sum - 1.0) > 0.01:
            raise ValueError(f"Los ratios deben sumar 1.0 (suma actual: {ratio_sum})")
        
        # Crear diccionario de distribución
        sn_type_distribution = {sn_type: ratio for sn_type, ratio in zip(sn_types, sn_ratios)}
    else:
        # Distribución equitativa
        equal_ratio = 1.0 / len(sn_types)
        sn_type_distribution = {sn_type: equal_ratio for sn_type in sn_types}
    
    # Crear configuración
    print(f"[INFO] Configuración del batch multi-banda")
    print(f"   • Runs: {n_runs}")
    if redshift_max is None:
        redshift_max = config.PROCESSING_CONFIG.get('redshift_max', 0.3)
    print(f"   • Redshift max: {redshift_max}")
    # Si no se pasa por CLI, usar default desde config.py
    if fixed_redshift is None:
        fixed_redshift = config.PROCESSING_CONFIG.get('fixed_redshift', None)
    if fixed_redshift is not None:
        print(f"   • Redshift fijo: {fixed_redshift}")
    print(f"   • Tipos de SN: {', '.join(sn_types)}")
    print(f"   • Distribución:")
    for sn_type, ratio in sn_type_distribution.items():
        expected_count = int(n_runs * ratio)
        print(f"      - {sn_type}: {ratio:.1%} (~{expected_count} SNe)")
    print(f"   • Survey: {survey}")
    print(f"   • Semilla: {seed}")
    print(f"   • Modo: MULTI-BANDA SIMULTÁNEA (g+r+i para ZTF)")
    print()
    
    # Crear configuración usando el sistema existente
    batch_config = create_simple_config(
        n_runs=n_runs,
        redshift_max=redshift_max,
        fixed_redshift=fixed_redshift,
        sn_types=sn_types,
        sn_type_distribution=sn_type_distribution,  # Pasar distribución custom
        survey=survey,
        seed=seed,
        filter_band="r"  # Valor por defecto, pero proyectará en g+r+i
    )
    
    print("[INFO] Ejecutando batch multi-banda...")
    print()
    
    # Ejecutar batch multi-banda
    results = run_multiband_batch(batch_config)
    
    return results

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Runner simplificado para simulaciones MULTI-BANDA de supernovas"
    )
    
    parser.add_argument("--runs", type=int, default=100,
                       help="Número de simulaciones a ejecutar (default: 100)")
    
    parser.add_argument("--redshift-max", type=float, default=None,
                       help="Redshift máximo para el muestreo (default: config.PROCESSING_CONFIG['redshift_max'])")

    parser.add_argument("--fixed-redshift", type=float, default=None,
                       help="Si se define, fija z para todos los runs (anula el muestreo)")
    
    parser.add_argument("--sn-types", nargs="+", choices=["Ia", "Ibc", "II"],
                       default=["Ia"], help="Tipos de SN a incluir (default: Ia)")
    
    parser.add_argument("--ratios", nargs="+", type=float,
                       help="Ratios para cada tipo en --sn-types (deben sumar 1.0). "
                            "Ejemplo: --sn-types Ia II Ibc --ratios 0.6 0.3 0.1")
    
    parser.add_argument("--survey", choices=["ZTF", "SUDARE"], default="ZTF",
                       help="Survey para proyección (default: ZTF)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Semilla aleatoria para reproducibilidad (default: 42)")
    
    parser.add_argument("--list", action="store_true",
                       help="Listar SNe disponibles por tipo y salir")
    
    args = parser.parse_args()
    
    # Si solo quiere ver la lista
    if args.list:
        from simple_config import get_sn_templates
        print("\n[OK] SNe disponibles por tipo:")
        for sn_type in ["Ia", "Ibc", "II"]:
            templates = get_sn_templates(sn_type)
            print(f"\n{sn_type}: {len(templates)} templates")
            for i, sn in enumerate(templates[:5], 1):
                print(f"  {i}. {sn}")
            if len(templates) > 5:
                print(f"  ... y {len(templates)-5} más")
        return
    
    # Ejecutar batch multi-banda
    results = run_custom_multiband_batch(
        n_runs=args.runs,
        redshift_max=args.redshift_max,
        fixed_redshift=args.fixed_redshift,
        sn_types=args.sn_types,
        sn_ratios=args.ratios,
        survey=args.survey,
        seed=args.seed
    )
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("[OK] Simulaciones multi-banda completadas")
    stats = results['run_statistics']
    batch_meta = results['batch_metadata']
    
    print(f" Tasa de éxito: {stats['success_rate']:.1%}")
    print(f"  Duración: {batch_meta['total_duration_formatted']}")
    
    if 'detections' in stats:
        print(f" Detecciones: {stats['detections']}")
    if 'observations' in stats:
        print(f" Observaciones: {stats['observations']}")
    
    print(f"\n[OK] Operacion completada exitosamente")
    print("="*60)

if __name__ == "__main__":
    main()
