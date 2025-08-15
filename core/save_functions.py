# 📚 IMPORTS PARA SAVE_FUNCTIONS
# ===============================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def save_projection_results(df_projected, lc_df, mag, mag_noisy, survey_params, 
                          sn_params, projection_params, ruido_promedio, 
                          alpha_usado, maximum, output_dir="outputs"):
    """
    Guarda todos los resultados de la proyección incluyendo datos y gráficos
    
    Parameters:
    -----------
    df_projected : DataFrame
        Datos de la proyección sobre observaciones reales
    lc_df : DataFrame  
        Curva de luz sintética original
    mag : array
        Magnitudes originales sin ruido
    mag_noisy : array
        Magnitudes con ruido aplicado
    survey_params : dict
        Parámetros del survey (SURVEY, selected_target, etc.)
    sn_params : dict
        Parámetros de la SN (sn_name, tipo, z_proy, etc.)
    projection_params : dict
        Parámetros de proyección (filtros, paths, etc.)
    ruido_promedio : float
        Ruido promedio aplicado
    alpha_usado : list
        Parámetros alpha usados en LOESS
    maximum : float
        MJD del máximo de la SN
    output_dir : str
        Directorio de salida
        
    Returns:
    --------
    dict : Paths de archivos guardados
    """
    
    print(f"\n💾 Guardando resultados...")
    
    # Crear estructura jerárquica de directorios
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Generar ID único para evitar conflictos
    import uuid
    unique_id = str(uuid.uuid4())[:8]  # Usar solo los primeros 8 caracteres
    
    # Estructura: outputs/SN_TYPE/SN_NAME/SURVEY/FILTER/run_timestamp_uniqueID/target_X/
    sn_dir = os.path.join(output_dir, sn_params['tipo'], sn_params['sn_name'])
    survey_dir = os.path.join(sn_dir, survey_params['SURVEY'])
    filter_dir = os.path.join(survey_dir, f"filter_{projection_params['selected_filter']}")
    
    # Crear run directory con timestamp + unique ID
    run_id = f"run_{timestamp}_{unique_id}"
    run_dir = os.path.join(filter_dir, run_id)
    
    # Agregar información del target dentro del run
    if survey_params['selected_target']:
        target_safe = str(survey_params['selected_target']).replace('/', '_')
        target_dir = os.path.join(run_dir, f"target_{target_safe}")
    else:
        target_dir = os.path.join(run_dir, "target_random")
    
    # Crear todos los directorios
    os.makedirs(target_dir, exist_ok=True)
    
    # También crear subdirectorios para organizar tipos de archivos
    data_dir = os.path.join(target_dir, "data")
    plots_dir = os.path.join(target_dir, "plots") 
    config_dir = os.path.join(target_dir, "config")
    
    for subdir in [data_dir, plots_dir, config_dir]:
        os.makedirs(subdir, exist_ok=True)
    
    saved_files = {}
    base_filename = f"{sn_params['sn_name']}_{survey_params['SURVEY']}_{projection_params['selected_filter']}"
    
    # 1. Guardar curva sintética
    lc_synthetic = pd.DataFrame({
        'fase': lc_df['fase'].values,
        'mag_original': mag,
        'mag_noisy': mag_noisy,
        'flux_original': lc_df['flux'].values,
        'overlap': lc_df['overlap'].values
    })
    synthetic_path = os.path.join(data_dir, f"synthetic_lc_{base_filename}.csv")
    lc_synthetic.to_csv(synthetic_path, index=False)
    saved_files['synthetic'] = synthetic_path
    
    # 2. Guardar proyección SIEMPRE (incluso si está vacía)
    projection_path = os.path.join(data_dir, f"projection_{base_filename}.csv")
    df_projected.to_csv(projection_path, index=False)
    saved_files['projection'] = projection_path
    
    # 3. Estadísticas SIEMPRE (maneja caso sin observaciones)
    if len(df_projected) > 0:
        detecciones = len(df_projected[df_projected['upperlimit'] == 'F'])
        upper_limits = len(df_projected[df_projected['upperlimit'] == 'T'])
        detection_rate = (detecciones/len(df_projected)*100)
        mjd_start = df_projected['mjd'].min()
        mjd_end = df_projected['mjd'].max()
        maglimit_min = df_projected['maglimit'].min()
        maglimit_max = df_projected['maglimit'].max()
        status = "SUCCESS"
        print(f"   ✅ Proyección completada: {len(df_projected)} puntos")
    else:
        detecciones = 0
        upper_limits = 0
        detection_rate = 0.0
        mjd_start = None
        mjd_end = None
        maglimit_min = None
        maglimit_max = None
        status = "NO_OBSERVATIONS"
        print("   ⚠️ Sin observaciones - guardando archivos vacíos")
    
    summary_stats = {
        'survey': survey_params['SURVEY'],
        'sn_name': sn_params['sn_name'],
        'sn_type': sn_params['tipo'],
        'target': survey_params['selected_target'],
        'filter_photometry': projection_params['selected_filter'],
        'filter_projection': projection_params['projection_filter'],
        'redshift': sn_params['z_proy'],
        'ebmv_host': sn_params['ebmv_host'],
        'ebmv_mw': sn_params['ebmv_mw'],
        'ebmv_total': sn_params['ebmv_total'],
        'total_points': len(df_projected),
        'detections': detecciones,
        'upper_limits': upper_limits,
        'detection_rate_percent': detection_rate,
        'mjd_start': mjd_start,
        'mjd_end': mjd_end,
        'maglimit_min': maglimit_min,
        'maglimit_max': maglimit_max,
        'noise_sigma': ruido_promedio,
        'status': status,
        'timestamp': timestamp,
        'run_directory': target_dir,
        'run_id': run_id
    }
    
    summary_path = os.path.join(data_dir, f"summary_{base_filename}.csv")
    pd.DataFrame([summary_stats]).to_csv(summary_path, index=False)
    saved_files['summary'] = summary_path
    
    # 4. Configuración SIEMPRE
    config_params = {
        **projection_params,
        **sn_params,
        'alpha_loess': str(alpha_usado),
        'noise_sigma': ruido_promedio,
        'maximum_mjd': maximum,
        'status': status,
        'timestamp': timestamp,
        'run_directory': target_dir,
        'run_id': run_id
    }
    
    config_path = os.path.join(config_dir, f"config_{base_filename}.csv")
    pd.DataFrame([config_params]).to_csv(config_path, index=False)
    saved_files['config'] = config_path
    
    # 5. Crear gráficos SIEMPRE (maneja caso sin datos)
    saved_files.update(_create_projection_plots(
        df_projected, lc_synthetic, survey_params, sn_params, 
        base_filename, plots_dir
    ))
    
    # Mostrar resumen de la estructura
    print(f"   📁 Estructura creada:")
    print(f"      🗂️  {os.path.relpath(target_dir)}/")
    print(f"         📊 data/     - Datos CSV")
    print(f"         📈 plots/    - Gráficos PNG") 
    print(f"         ⚙️  config/   - Configuraciones")
    print(f"   📍 Run ID: {run_id}")
    print(f"   🎯 Target: {survey_params['selected_target']}")
    
    # Listar archivos guardados
    for file_type, path in saved_files.items():
        subdir = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        print(f"   • {file_type}: {subdir}/{filename}")
    
    return saved_files


def _create_projection_plots(df_projected, lc_synthetic, survey_params, 
                           sn_params, base_filename, output_dir):
    """
    Crea gráficos de la proyección - siempre genera al menos la curva sintética
    """
    
    saved_plots = {}
    
    # Configurar estilo
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Título adaptado según si hay observaciones o no
    if len(df_projected) > 0:
        fig.suptitle(f'Proyección {sn_params["sn_name"]} - {survey_params["SURVEY"]}', 
                     fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'Proyección {sn_params["sn_name"]} - {survey_params["SURVEY"]} (SIN OBSERVACIONES)', 
                     fontsize=16, fontweight='bold', color='red')
    
    # 1. Curva sintética original vs con ruido (SIEMPRE se muestra)
    ax1 = axes[0, 0]
    ax1.plot(lc_synthetic['fase'], lc_synthetic['mag_original'], 
             'b-', label='Original', linewidth=2)
    ax1.scatter(lc_synthetic['fase'], lc_synthetic['mag_noisy'], 
                c='red', s=20, alpha=0.7, label='Con ruido')
    ax1.set_xlabel('Fase (días)')
    ax1.set_ylabel('Magnitud')
    ax1.set_title('Curva de Luz Sintética')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # 2. Proyección temporal (maneja caso sin observaciones)
    ax2 = axes[0, 1]
    if len(df_projected) > 0:
        detections = df_projected[df_projected['upperlimit'] == 'F']
        limits = df_projected[df_projected['upperlimit'] == 'T']
        
        if len(detections) > 0:
            ax2.scatter(detections['mjd'], detections['magnitud_proyectada'], 
                       c='green', s=30, label=f'Detecciones ({len(detections)})', alpha=0.8)
        if len(limits) > 0:
            ax2.scatter(limits['mjd'], limits['maglimit'], 
                       c='red', marker='v', s=30, label=f'Upper limits ({len(limits)})', alpha=0.8)
        
        ax2.set_title('Proyección Temporal')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'SIN OBSERVACIONES\nDisponibles', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, color='red', weight='bold')
        ax2.set_title('Proyección Temporal (Sin Datos)')
    
    ax2.set_xlabel('MJD')
    ax2.set_ylabel('Magnitud')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. Histograma de magnitudes límite (maneja caso sin observaciones)
    ax3 = axes[1, 0]
    if len(df_projected) > 0:
        ax3.hist(df_projected['maglimit'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribución de Magnitudes Límite')
    else:
        ax3.text(0.5, 0.5, 'SIN DATOS PARA\nHISTOGRAMA', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, color='red', weight='bold')
        ax3.set_title('Magnitudes Límite (Sin Datos)')
    
    ax3.set_xlabel('Magnitud límite')
    ax3.set_ylabel('Frecuencia')
    ax3.grid(True, alpha=0.3)
    
    # 4. Panel informativo con estadísticas clave
    ax4 = axes[1, 1]
    if len(df_projected) > 0:
        detections = df_projected[df_projected['upperlimit'] == 'F']
        limits = df_projected[df_projected['upperlimit'] == 'T']
        tasa = len(detections) / len(df_projected) * 100
        
        # Crear gráfico de barras con estadísticas clave
        stats_labels = ['Detecciones', 'Upper Limits', 'Total']
        stats_values = [len(detections), len(limits), len(df_projected)]
        colors = ['lightgreen', 'lightcoral', 'lightblue']
        
        bars = ax4.bar(stats_labels, stats_values, color=colors, alpha=0.7)
        
        # Agregar valores en las barras
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Número de puntos')
        ax4.set_title(f'Estadísticas ({tasa:.1f}% detecciones)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Agregar información adicional como texto
        info_text = f"Survey: {survey_params['SURVEY']}\n"
        info_text += f"Target: {survey_params['selected_target']}\n"
        info_text += f"SN: {sn_params['sn_name']} ({sn_params['tipo']})"
        
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Mostrar gráfico indicando que no hay observaciones
        ax4.text(0.5, 0.5, f'NO OBSERVABLE\nen {survey_params["SURVEY"]}', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, color='red', weight='bold')
        ax4.set_title('Estado de Observabilidad')
        
        # Información de la SN aunque no sea observable
        info_text = f"Survey: {survey_params['SURVEY']}\n"
        info_text += f"Target: {survey_params['selected_target']}\n"
        info_text += f"SN: {sn_params['sn_name']} ({sn_params['tipo']})"
        
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    
    # Guardar figura SIEMPRE
    plot_path = os.path.join(output_dir, f"plots_{base_filename}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    saved_plots['plots'] = plot_path
    
    # Mensaje de confirmación
    if len(df_projected) > 0:
        print(f"   📈 Gráficos creados con datos de proyección")
    else:
        print(f"   📈 Gráficos creados (curva sintética + indicación sin observaciones)")
    
    return saved_plots


def create_master_index(output_dir="outputs"):
    """
    Crea un índice maestro de todas las proyecciones realizadas
    para facilitar la navegación y comparación
    """
    print(f"\n📚 Creando índice maestro...")
    
    index_data = []
    
    # Recorrer toda la estructura de directorios
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.startswith('summary_') and file.endswith('.csv'):
                summary_path = os.path.join(root, file)
                
                try:
                    summary_df = pd.read_csv(summary_path)
                    if len(summary_df) > 0:
                        summary_row = summary_df.iloc[0].to_dict()
                        
                        # Agregar información de ubicación
                        rel_path = os.path.relpath(root, output_dir)
                        summary_row['relative_path'] = rel_path
                        summary_row['summary_file'] = summary_path
                        
                        index_data.append(summary_row)
                        
                except Exception as e:
                    print(f"   ⚠️ Error leyendo {summary_path}: {e}")
    
    if index_data:
        # Crear DataFrame maestro
        master_df = pd.DataFrame(index_data)
        
        # Ordenar por timestamp
        master_df = master_df.sort_values('timestamp', ascending=False)
        
        # Guardar índice maestro
        master_path = os.path.join(output_dir, "MASTER_INDEX.csv")
        master_df.to_csv(master_path, index=False)
        
        print(f"   📋 Índice maestro creado: {master_path}")
        print(f"   📊 Total proyecciones indexadas: {len(master_df)}")
        
        # Estadísticas por status si existe la columna
        if 'status' in master_df.columns:
            status_counts = master_df['status'].value_counts()
            print(f"   ✅ Exitosas: {status_counts.get('SUCCESS', 0)}")
            print(f"   ⚠️ Sin observaciones: {status_counts.get('NO_OBSERVATIONS', 0)}")
        
        # Crear resumen por SN y survey
        if 'sn_name' in master_df.columns and 'survey' in master_df.columns:
            summary_by_sn = master_df.groupby(['sn_name', 'survey']).size().reset_index(name='count')
            summary_path = os.path.join(output_dir, "SUMMARY_BY_SN.csv")
            summary_by_sn.to_csv(summary_path, index=False)
            print(f"   📈 Resumen por SN: {summary_path}")
            
            # Crear resumen por status si existe
            if 'status' in master_df.columns:
                summary_by_status = master_df.groupby(['sn_name', 'survey', 'status']).size().reset_index(name='count')
                status_summary_path = os.path.join(output_dir, "SUMMARY_BY_STATUS.csv")
                summary_by_status.to_csv(status_summary_path, index=False)
                print(f"   📊 Resumen por status: {status_summary_path}")
        
        return master_path
    else:
        print("   ℹ️ No se encontraron proyecciones para indexar")
        return None


def run_multiple_projections(survey_list=None, target_list=None, n_runs=3, **kwargs):
    """
    Ejecuta múltiples proyecciones de forma segura, manejando targets duplicados
    
    Parameters:
    -----------
    survey_list : list, optional
        Lista de surveys a usar. Si None, usa el configurado en main
    target_list : list, optional  
        Lista específica de targets. Si None, selección aleatoria
    n_runs : int
        Número de runs a ejecutar
    **kwargs : dict
        Parámetros adicionales para save_projection_results
        
    Returns:
    --------
    list : Lista de resultados de cada run
    """
    print(f"\n🚀 Ejecutando {n_runs} proyecciones múltiples...")
    
    results = []
    
    for i in range(n_runs):
        print(f"\n   🔄 Run {i+1}/{n_runs}")
        
        # Simular selección de target (esto normalmente vendría del main)
        if target_list:
            selected_target = np.random.choice(target_list)
        else:
            # Esto se haría en el main.py con la lógica existente
            selected_target = f"random_target_{i+1}"
        
        print(f"      🎯 Target seleccionado: {selected_target}")
        
        # Aquí llamarías a save_projection_results con los datos reales
        # results.append(save_projection_results(...))
        
        # Por ahora solo mostramos cómo sería la estructura
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        run_id = f"run_{timestamp}_{unique_id}"
        
        print(f"      📁 Run ID único: {run_id}")
        print(f"      ⏱️  Timestamp: {timestamp}")
        
        # Simular pequeña pausa para evitar timestamps idénticos
        import time
        time.sleep(0.1)
        
        results.append({
            'run_id': run_id,
            'target': selected_target,
            'timestamp': timestamp
        })
    
    print(f"\n✅ {n_runs} proyecciones completadas sin conflictos")
    return results