# MULTI-BAND SIMPLE - Solo ejecuta simple_runner.py varias veces
# ============================================================
# Version ultra-simple: solo llama a simple_runner con diferentes filtros

import subprocess
import sys
import argparse
from datetime import datetime


def print_banner():
    print("="*60)
    print(" MULTI-BAND RUNNER SIMPLE")
    print("="*60)
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print()


def run_simple_runner(runs, filter_band, sn_types, redshift_max, survey, seed):
    """
    Llama directamente a simple_runner.py
    """
    cmd = [
        sys.executable,
        "simple_runner.py",
        "--runs", str(runs),
        "--filter", filter_band,
        "--sn-types"] + sn_types + [
        "--redshift-max", str(redshift_max),
        "--survey", survey,
        "--seed", str(seed)
    ]
    
    print(f">> Ejecutando: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Band Simple: Ejecuta simple_runner.py con varios filtros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # ZTF completo (g, r, i)
  python run_multiband.py --runs 100 --filters g r i --sn-types Ia

  # Comparar filtros
  python run_multiband.py --runs 50 --filters r i --sn-types Ibc
        """
    )
    
    parser.add_argument("--runs", type=int, required=True,
                       help="Simulaciones por filtro")
    
    parser.add_argument("--filters", nargs="+", required=True,
                       choices=["U", "B", "V", "R", "I", "u", "g", "r", "i", "z"],
                       help="Lista de filtros")
    
    parser.add_argument("--sn-types", nargs="+", required=True,
                       choices=["Ia", "Ibc", "II"],
                       help="Tipos de SN")
    
    parser.add_argument("--redshift-max", type=float, default=0.3,
                       help="Redshift max (default: 0.3)")
    
    parser.add_argument("--survey", default="ZTF",
                       choices=["ZTF", "SUDARE"],
                       help="Survey (default: ZTF)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Semilla base (default: 42)")
    
    args = parser.parse_args()
    
    print_banner()
    
    print(f"Configuracion:")
    print(f"   Simulaciones por filtro: {args.runs}")
    print(f"   Filtros: {', '.join(args.filters)}")
    print(f"   Total simulaciones: {args.runs * len(args.filters)}")
    print(f"   Tipos SN: {', '.join(args.sn_types)}")
    print(f"   Redshift max: {args.redshift_max}")
    print()
    
    success_count = 0
    
    for i, filter_band in enumerate(args.filters, 1):
        print("="*60)
        print(f"FILTRO {i}/{len(args.filters)}: {filter_band.upper()}")
        print("="*60)
        print()
        
        # Llamar a simple_runner.py con semilla diferente por filtro
        success = run_simple_runner(
            runs=args.runs,
            filter_band=filter_band,
            sn_types=args.sn_types,
            redshift_max=args.redshift_max,
            survey=args.survey,
            seed=args.seed + i * 1000
        )
        
        if success:
            success_count += 1
            print(f"[OK] Filtro {filter_band} completado")
        else:
            print(f"[ERROR] Filtro {filter_band} fallo")
        
        print()
    
    # Resumen final
    print("="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Filtros exitosos: {success_count}/{len(args.filters)}")
    print(f"Total simulaciones: {args.runs * success_count}")
    print("="*60)
    
    sys.exit(0 if success_count == len(args.filters) else 1)


if __name__ == "__main__":
    main()
