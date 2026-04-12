"""
COMBINA TODOS LOS CSV/PARQUET DE MÚLTIPLES BATCHES
==================================================
Combina todos los archivos all_projected_combined.csv.gz o all_projected_combined.parquet
generados por diferentes ejecuciones paralelas de run_sn_list_multiband.py.

Uso:
    python combine_all_batches.py [--survey ZTF] [--output-dir outputs/multiband_runs]
    
Busca todos los batch_id en outputs/multiband_runs/{survey}_* y combina sus
archivos all_projected_combined.* en un solo archivo final.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def find_all_batch_combined_files(survey: str, base_dir: Path, batch_prefix: Optional[str] = None) -> List[Path]:
    """
    Encuentra todos los archivos all_projected_combined.* en los batch folders.
    
    Busca tanto .parquet como .csv.gz
    """
    multiband_dir = base_dir / "multiband_runs"
    if not multiband_dir.exists():
        print(f"[ERROR] No existe el directorio: {multiband_dir}")
        return []
    
    # Buscar todos los directorios que coincidan con {survey}_*
    batch_dirs = [d for d in multiband_dir.iterdir()
                  if d.is_dir() and d.name.startswith(f"{survey}_")]

    # Filtrar por prefijo de batch_id (útil para agrupar una corrida paralela específica)
    if batch_prefix:
        batch_prefix = str(batch_prefix)
        batch_dirs = [d for d in batch_dirs if d.name.startswith(f"{survey}_{batch_prefix}")]
    
    if not batch_dirs:
        print(f"[WARNING] No se encontraron directorios de batch para survey={survey}")
        return []
    
    combined_files = []
    for batch_dir in batch_dirs:
        # Buscar all_projected_combined.parquet primero (preferido)
        parquet_file = batch_dir / "all_projected_combined.parquet"
        csv_gz_file = batch_dir / "all_projected_combined.csv.gz"
        
        if parquet_file.exists():
            combined_files.append(parquet_file)
            print(f"[OK] Encontrado: {parquet_file}")
        elif csv_gz_file.exists():
            combined_files.append(csv_gz_file)
            print(f"[OK] Encontrado: {csv_gz_file}")
        else:
            print(f"[WARNING] No se encontró archivo combinado en {batch_dir}")
    
    return combined_files


def load_combined_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Carga un archivo combinado (parquet o csv.gz)."""
    try:
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.gz' or str(file_path).endswith('.csv.gz'):
            df = pd.read_csv(file_path, compression='gzip')
        else:
            print(f"[WARNING] Formato no reconocido: {file_path}")
            return None
        
        print(f"[OK] Cargado {len(df)} filas desde {file_path.name}")
        return df
    except Exception as e:
        print(f"[ERROR] No pude cargar {file_path}: {e}")
        return None


def combine_all_batches(survey: str = "ZTF", base_dir: Path = Path("outputs"), batch_prefix: Optional[str] = None):
    """
    Combina todos los archivos all_projected_combined.* de múltiples batches.
    """
    print("=" * 60)
    print("COMBINANDO TODOS LOS BATCHES")
    print("=" * 60)
    print(f"Survey: {survey}")
    print(f"Directorio base: {base_dir}")
    print("")
    
    # Encontrar todos los archivos combinados
    combined_files = find_all_batch_combined_files(survey, base_dir, batch_prefix=batch_prefix)
    
    if not combined_files:
        print("[ERROR] No se encontraron archivos para combinar.")
        return None
    
    print(f"\n[INFO] Se encontraron {len(combined_files)} archivos para combinar.")
    print("")
    
    # Cargar todos los DataFrames
    dfs = []
    for file_path in combined_files:
        df = load_combined_file(file_path)
        if df is not None and len(df) > 0:
            # Agregar metadatos del batch_id desde el nombre del directorio
            batch_id = file_path.parent.name.replace(f"{survey}_", "")
            if 'batch_id' not in df.columns or df['batch_id'].isna().all():
                df['batch_id'] = batch_id
            dfs.append(df)
    
    if not dfs:
        print("[ERROR] No se pudieron cargar DataFrames válidos.")
        return None
    
    print(f"\n[INFO] Combinando {len(dfs)} DataFrames...")
    
    # Combinar todos los DataFrames
    try:
        df_combined = pd.concat(dfs, ignore_index=True)
        print(f"[OK] DataFrame combinado: {len(df_combined)} filas, {len(df_combined.columns)} columnas")
    except Exception as e:
        print(f"[ERROR] Error al combinar DataFrames: {e}")
        return None
    
    # Guardar archivo combinado final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_tag = f"_{batch_prefix}" if batch_prefix else ""
    output_dir = base_dir / "multiband_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar como parquet (preferido)
    output_parquet = output_dir / f"{survey}_ALL_BATCHES_COMBINED{prefix_tag}_{timestamp}.parquet"
    try:
        df_combined.to_parquet(output_parquet, index=False)
        print(f"[SAVED] Parquet combinado: {output_parquet}")
        print(f"        Tamaño: {output_parquet.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"[WARNING] No pude guardar parquet: {e}")
    
    # Guardar como CSV comprimido
    output_csv_gz = output_dir / f"{survey}_ALL_BATCHES_COMBINED{prefix_tag}_{timestamp}.csv.gz"
    try:
        df_combined.to_csv(output_csv_gz, index=False, compression='gzip')
        print(f"[SAVED] CSV comprimido: {output_csv_gz}")
        print(f"        Tamaño: {output_csv_gz.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"[WARNING] No pude guardar CSV comprimido: {e}")
    
    # Guardar como CSV sin comprimir (más fácil de abrir)
    output_csv = output_dir / f"{survey}_ALL_BATCHES_COMBINED{prefix_tag}_{timestamp}.csv"
    try:
        df_combined.to_csv(output_csv, index=False)
        print(f"[SAVED] CSV sin comprimir: {output_csv}")
        print(f"        Tamaño: {output_csv.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"[WARNING] No pude guardar CSV: {e}")
    
    # Estadísticas
    print("")
    print("=" * 60)
    print("ESTADÍSTICAS DEL DATASET COMBINADO")
    print("=" * 60)
    print(f"Total de filas: {len(df_combined):,}")
    print(f"Total de columnas: {len(df_combined.columns)}")
    
    if 'batch_id' in df_combined.columns:
        unique_batches = df_combined['batch_id'].nunique()
        print(f"Batches únicos: {unique_batches}")
        print("\nFilas por batch:")
        batch_counts = df_combined['batch_id'].value_counts().sort_index()
        for batch_id, count in batch_counts.items():
            print(f"  {batch_id}: {count:,} filas")
    
    if 'sn_name' in df_combined.columns:
        unique_sn = df_combined['sn_name'].nunique()
        print(f"\nSN únicas: {unique_sn}")
    
    if 'sn_type' in df_combined.columns:
        print("\nFilas por tipo de SN:")
        type_counts = df_combined['sn_type'].value_counts()
        for sn_type, count in type_counts.items():
            print(f"  {sn_type}: {count:,} filas")
    
    print("=" * 60)
    
    return df_combined


def main():
    parser = argparse.ArgumentParser(
        description="Combina todos los archivos all_projected_combined.* de múltiples batches"
    )
    parser.add_argument(
        "--survey", type=str, default="ZTF",
        help="Survey name (default: ZTF)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directorio base de outputs (default: outputs)"
    )
    parser.add_argument(
        "--batch-prefix", type=str, default=None,
        help="Filtrar solo batches cuyo folder sea {survey}_<batch-prefix>* (ej: 20260125_231012)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    if not base_dir.exists():
        print(f"[ERROR] El directorio no existe: {base_dir}")
        sys.exit(1)
    
    df_result = combine_all_batches(survey=args.survey, base_dir=base_dir, batch_prefix=args.batch_prefix)
    
    if df_result is None:
        print("\n[ERROR] No se pudo combinar los batches.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] ¡Combinación completada exitosamente!")


if __name__ == "__main__":
    main()
