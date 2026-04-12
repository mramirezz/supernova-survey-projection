"""
Script para rellenar los valores de redshift (z) en sn_list_to_project.csv
usando los datos de ztf_targets_with_coords_multicat_summary.csv

Calcula el promedio de SDSS_z, NED_z, SIMBAD_z cuando estén disponibles.
Si los tres son NaN, deja el valor vacío (None).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_average_redshift(row):
    """
    Calcula el promedio de los redshifts disponibles (SDSS_z, NED_z, SIMBAD_z).
    Ignora los valores NaN.
    Retorna None si todos son NaN.
    """
    z_values = []
    
    for col in ['SDSS_z', 'NED_z', 'SIMBAD_z']:
        if col in row and pd.notna(row[col]):
            z_values.append(row[col])
    
    if len(z_values) == 0:
        return None
    
    return np.mean(z_values)


def main():
    # Rutas de los archivos
    data_dir = Path(__file__).parent / "data"
    multicat_file = data_dir / "ztf_targets_with_coords_multicat_summary.csv"
    sn_list_file = data_dir / "sn_list_to_project.csv"
    
    # Leer los archivos CSV
    print(f"Leyendo {multicat_file}...")
    df_multicat = pd.read_csv(multicat_file)
    
    print(f"Leyendo {sn_list_file}...")
    df_sn_list = pd.read_csv(sn_list_file)
    
    print(f"\nSupernovas en sn_list_to_project: {len(df_sn_list)}")
    print(f"Supernovas en multicat_summary: {len(df_multicat)}")
    
    # Crear diccionario con los redshifts promedio por sn_name
    redshift_dict = {}
    for _, row in df_multicat.iterrows():
        sn_name = row['sn_name']
        avg_z = calculate_average_redshift(row)
        redshift_dict[sn_name] = avg_z
    
    # Estadísticas
    count_with_z = sum(1 for z in redshift_dict.values() if z is not None)
    count_without_z = sum(1 for z in redshift_dict.values() if z is None)
    print(f"\nEn multicat_summary:")
    print(f"  - Con redshift disponible: {count_with_z}")
    print(f"  - Sin redshift (todos NaN): {count_without_z}")
    
    # Actualizar la columna z en sn_list_to_project
    updated_count = 0
    not_found_count = 0
    already_has_z = 0
    
    for idx, row in df_sn_list.iterrows():
        sn_name = row['sn_name']
        
        if sn_name in redshift_dict:
            new_z = redshift_dict[sn_name]
            if new_z is not None:
                df_sn_list.at[idx, 'z'] = new_z
                updated_count += 1
            # Si new_z es None, dejamos la columna vacía (NaN)
        else:
            not_found_count += 1
            print(f"  Advertencia: {sn_name} no encontrado en multicat_summary")
    
    # Guardar el archivo actualizado
    output_file = sn_list_file  # Sobrescribir el archivo original
    # También podemos crear un backup antes
    backup_file = data_dir / "sn_list_to_project_backup.csv"
    
    # Crear backup del archivo original
    df_original = pd.read_csv(sn_list_file)
    df_original.to_csv(backup_file, index=False)
    print(f"\nBackup creado en: {backup_file}")
    
    # Guardar el archivo actualizado
    df_sn_list.to_csv(output_file, index=False)
    print(f"Archivo actualizado: {output_file}")
    
    print(f"\nResumen:")
    print(f"  - Redshifts actualizados: {updated_count}")
    print(f"  - SNe no encontradas en multicat: {not_found_count}")
    print(f"  - SNe sin redshift disponible: {len(df_sn_list) - updated_count - not_found_count}")
    
    # Mostrar algunos ejemplos de los datos actualizados
    print("\nPrimeras 10 filas del archivo actualizado:")
    print(df_sn_list.head(10).to_string())
    
    # Mostrar distribución de fuentes de redshift
    print("\n--- Detalles adicionales ---")
    for _, row in df_multicat.head(10).iterrows():
        sn_name = row['sn_name']
        sdss_z = row.get('SDSS_z', np.nan)
        ned_z = row.get('NED_z', np.nan)
        simbad_z = row.get('SIMBAD_z', np.nan)
        avg_z = calculate_average_redshift(row)
        
        sources = []
        if pd.notna(sdss_z):
            sources.append(f"SDSS={sdss_z:.6f}")
        if pd.notna(ned_z):
            sources.append(f"NED={ned_z:.6f}")
        if pd.notna(simbad_z):
            sources.append(f"SIMBAD={simbad_z:.6f}")
        
        if sources:
            print(f"{sn_name}: {', '.join(sources)} -> promedio={avg_z:.6f}")
        else:
            print(f"{sn_name}: Sin redshift disponible")


if __name__ == "__main__":
    main()
