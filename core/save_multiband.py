"""
GUARDADO DE RESULTADOS MULTI-BANDA
===================================
Sistema optimizado para guardar proyecciones multi-banda con:
- Parquet para datos masivos (compresión ~10x vs CSV)
- Plots de debug claros
- Estructura organizada por tipo
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def save_multiband_results(
    df_projected,
    synthetic_curves,
    projection_result,
    config,
    batch_id,
    iteration_label,
    base_output_dir='outputs/multiband_runs'
):
    """
    Guarda resultados de proyección multi-banda
    
    Parameters:
    -----------
    df_projected : DataFrame
        Proyecciones multi-banda (todas las bandas juntas)
    synthetic_curves : dict
        {filter: (mjd, mag)} curvas sintéticas
    projection_result : dict
        {'offset_used', 'field_selected', 'n_observations', 'desplazamiento'}
    config : dict/Config
        Configuración completa
    batch_id : str
        ID del batch (timestamp)
    iteration_label : str
        iter_0001_of_0100
    
    Returns:
    --------
    dict : Paths de archivos guardados
    """
    
    # Extraer info básica - revisar múltiples keys posibles
    sn_type = config.get('tipo') or config.get('sn_type') or config.get('tipo_sn', 'Unknown')
    sn_name = config.get('sn_name', 'Unknown')
    survey = config.get('SURVEY', 'ZTF')
    
    # 1. Crear estructura de directorios
    run_dir = Path(base_output_dir) / f"{survey}_{batch_id}"
    
    # Determinar si exitosa o fallida
    if len(df_projected) > 0:
        status_dir = run_dir / sn_type / 'individual'
        combined_file = run_dir / sn_type / f'{sn_type}_projections.parquet'
    else:
        status_dir = run_dir / sn_type / 'failed'
        combined_file = None
    
    # Crear carpeta individual
    sn_dir = status_dir / f"{sn_name}_{iteration_label}"
    sn_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = sn_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    saved_files = {}
    
    # 2. Guardar datos individuales (CSV solo para debug individual)
    if len(df_projected) > 0:
        # Proyecciones (CSV pequeño para debug)
        proj_file = sn_dir / 'projected.csv'
        df_projected.to_csv(proj_file, index=False)
        saved_files['projected_csv'] = str(proj_file)
        
        # Sintéticas (CSV pequeño para debug)
        synth_df = _synthetic_to_dataframe(synthetic_curves)
        synth_file = sn_dir / 'synthetic.csv'
        synth_df.to_csv(synth_file, index=False)
        saved_files['synthetic_csv'] = str(synth_file)
        
        # Actualizar archivo combinado (PARQUET)
        _append_to_combined_parquet(
            df_projected, combined_file, config, 
            projection_result, iteration_label
        )
        saved_files['combined_parquet'] = str(combined_file)
    
    # 3. Guardar config JSON
    config_file = sn_dir / 'config.json'
    config_data = _extract_config_dict(config, projection_result, iteration_label)
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    saved_files['config'] = str(config_file)
    
    # 4. Generar plot único comprehensivo
    if len(df_projected) > 0:
        # Plot completo del proceso de proyección
        main_plot = plots_dir / 'projection_summary.png'
        _plot_projection_summary(
            df_projected, synthetic_curves, 
            projection_result, config, main_plot
        )
        saved_files['plot_summary'] = str(main_plot)
    else:
        # Plot de por qué falló
        fail_plot = plots_dir / 'failure_reason.png'
        _plot_failure_reason(synthetic_curves, projection_result, config, fail_plot)
        saved_files['plot_failure'] = str(fail_plot)
    
    # 5. Actualizar summary del tipo
    _update_type_summary(run_dir / sn_type, config, projection_result, iteration_label, sn_dir)
    
    return saved_files


def _synthetic_to_dataframe(synthetic_curves):
    """Convierte dict de curvas sintéticas a DataFrame"""
    rows = []
    for filt, (mjd, mag) in synthetic_curves.items():
        for m, mag_val in zip(mjd, mag):
            rows.append({'filter': filt, 'mjd': m, 'mag_synthetic': mag_val})
    return pd.DataFrame(rows)


def _append_to_combined_parquet(df_projected, parquet_file, config, proj_result, iteration_label):
    """Agrega proyección al archivo Parquet combinado"""
    
    # Agregar metadata
    df_with_meta = df_projected.copy()
    df_with_meta['iteration'] = iteration_label
    df_with_meta['sn_type'] = config.get('tipo', config.get('sn_type'))
    df_with_meta['sn_name'] = config.get('sn_name')
    df_with_meta['redshift'] = config.get('redshift', 0.0)
    df_with_meta['ebv_total'] = config.get('extinction_total', 0.0)
    df_with_meta['field_oid'] = proj_result['field_selected']
    df_with_meta['offset'] = proj_result['offset_used']
    
    # Append o crear
    parquet_file = Path(parquet_file)
    if parquet_file.exists():
        df_existing = pd.read_parquet(parquet_file)
        df_combined = pd.concat([df_existing, df_with_meta], ignore_index=True)
    else:
        df_combined = df_with_meta
    
    # Guardar con compresión
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(parquet_file, compression='snappy', index=False)


def _extract_config_dict(config, proj_result, iteration_label):
    """Extrae diccionario de configuración para JSON"""
    
    # Manejar tanto Config objects como dicts
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__.copy()
    else:
        config_dict = dict(config)
    
    # Limpiar objetos no serializables
    clean_dict = {}
    for k, v in config_dict.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean_dict[k] = v
        elif isinstance(v, (list, tuple)):
            clean_dict[k] = list(v)
        elif isinstance(v, dict):
            clean_dict[k] = v
    
    # Agregar info de proyección
    clean_dict['projection'] = {
        'offset_used': int(proj_result['offset_used']),
        'field_selected': proj_result['field_selected'],
        'n_observations': int(proj_result['n_observations']),
        'desplazamiento': float(proj_result['desplazamiento']),
        'iteration': iteration_label,
        'timestamp': datetime.now().isoformat()
    }
    
    return clean_dict


def _plot_projection_summary(df_proj, synth_curves, proj_result, config, output_file):
    """
    Plot comprehensivo del proceso completo de proyección multi-banda
    
    3 paneles:
    - Panel superior: Curva original (antes de offset)
    - Panel medio: Overlap temporal (rangos MJD)
    - Panel inferior: Resultado final (curva desplazada + observaciones proyectadas)
    """
    
    filters = list(synth_curves.keys())
    desplaz = proj_result['desplazamiento']
    offset = proj_result['offset_used']
    
    # Colores profesionales por filtro
    colors = {'g': '#2ca02c', 'r': '#d62728', 'i': '#8c564b', 
              'U': '#9467bd', 'B': '#1f77b4', 'V': '#17becf', 
              'R': '#ff7f0e', 'I': '#e377c2'}
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.8, 1.2], hspace=0.35)
    
    # =========================================================================
    # PANEL 1: CURVA ORIGINAL (antes de aplicar offset)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    
    for filt in filters:
        color = colors.get(filt, 'gray')
        mjd_synth, mag_synth_noisy = synth_curves[filt]
        
        # Plotear con puntos
        ax1.scatter(mjd_synth, mag_synth_noisy, s=80, color=color, 
                   alpha=0.7, edgecolors='black', linewidths=0.5,
                   label=f'Template {filt}', zorder=5)
        
        # Marcar el máximo (peak)
        max_idx = np.argmin(mag_synth_noisy)
        mjd_peak = mjd_synth[max_idx]
        ax1.axvline(mjd_peak, color=color, linestyle='--', 
                   linewidth=2, alpha=0.5)
    
    ax1.invert_yaxis()
    ax1.set_ylabel('Apparent Magnitude', fontsize=13, fontweight='bold')
    ax1.set_title('Step 1: SN Template Light Curve', 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11, ncol=len(filters))
    ax1.tick_params(labelsize=11)
    ax1.set_xlabel('MJD (template frame)', fontsize=12)
    
    # =========================================================================
    # PANEL 2: OBSERVACIONES DE LA GRILLA vs CURVA SN DESPLAZADA
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    
    # Obtener grilla completa de observaciones (si está disponible)
    df_grid = proj_result.get('grid_observations', None)
    
    # Calcular rangos de la SN desplazada
    mjd_synth_example = synth_curves[filters[0]][0]
    mjd_sn_min = mjd_synth_example.min() + desplaz
    mjd_sn_max = mjd_synth_example.max() + desplaz
    mjd_sn_peak = mjd_synth_example[np.argmin(synth_curves[filters[0]][1])] + desplaz
    
    # Si tenemos la grilla completa, la ploteamos
    if df_grid is not None and len(df_grid) > 0:
        # Plotear TODAS las observaciones de la grilla (filtradas por "mejor por día")
        for filt in filters:
            color = colors.get(filt, 'gray')
            df_filt_grid = df_grid[df_grid['filter'] == filt]
            
            if len(df_filt_grid) > 0:
                # Observaciones de la grilla como puntos pequeños grises
                ax2.scatter(df_filt_grid['mjd'], df_filt_grid['maglimit'], 
                           marker='x', s=40, color='gray', alpha=0.3,
                           label=f'Grid {filt} (n={len(df_filt_grid)})' if filt == filters[0] else '',
                           zorder=5)
    
    # Plotear curva SN desplazada
    for filt in filters:
        color = colors.get(filt, 'gray')
        mjd_synth, mag_synth = synth_curves[filt]
        mjd_displaced = mjd_synth + desplaz
        
        # Curva desplazada
        ax2.plot(mjd_displaced, mag_synth, '-', color=color, 
                linewidth=2.5, alpha=0.6, label=f'SN {filt} (shifted)')
    
    # Marcar zona de overlap con sombreado
    overlap_min = max(mjd_sn_min, df_grid['mjd'].min() if df_grid is not None else mjd_sn_min)
    overlap_max = min(mjd_sn_max, df_grid['mjd'].max() if df_grid is not None else mjd_sn_max)
    
    if overlap_min < overlap_max:
        ax2.axvspan(overlap_min, overlap_max, alpha=0.1, color='green', 
                    label=f'Overlap region', zorder=0)
    
    # Líneas verticales clave
    ax2.axvline(mjd_sn_min, color='blue', linestyle='--', linewidth=2, 
                alpha=0.6, label='SN start/end')
    ax2.axvline(mjd_sn_max, color='blue', linestyle='--', linewidth=2, alpha=0.6)
    ax2.axvline(mjd_sn_peak, color='red', linestyle='-', linewidth=2.5, 
                alpha=0.7, label=f'SN peak')
    
    # Plotear observaciones PROYECTADAS (las que pasaron el filtro de overlap)
    for filt in filters:
        color = colors.get(filt, 'gray')
        df_filt_proj = df_proj[df_proj['filter'] == filt]
        
        if len(df_filt_proj) > 0:
            # Observaciones que SÍ se proyectaron (en overlap)
            ax2.scatter(df_filt_proj['mjd'], df_filt_proj['maglimit'], 
                       marker='o', s=200, color=color, edgecolors='black',
                       linewidths=2.5, label=f'Projected {filt} (n={len(df_filt_proj)})', 
                       zorder=10, alpha=0.9)
    
    ax2.invert_yaxis()
    ax2.set_xlabel('Modified Julian Date (MJD)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Limiting Magnitude', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=11)
    ax2.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
    
    # Calcular estadísticas de overlap
    gap_info = ""
    if df_grid is not None and len(df_grid) > 0:
        grid_start = df_grid['mjd'].min()
        if grid_start > mjd_sn_min:
            gap_days = grid_start - mjd_sn_min
            gap_info = f" | GAP: {gap_days:.0f}d before grid starts"
    
    overlap_pct = 100 * (overlap_max - overlap_min) / (mjd_sn_max - mjd_sn_min) if (mjd_sn_max - mjd_sn_min) > 0 else 0
    n_total_grid = len(df_grid) if df_grid is not None else 0
    n_projected = len(df_proj)
    
    ax2.set_title(f'Step 2: Grid Observations vs Shifted SN (offset={offset:+d}d, overlap={overlap_pct:.0f}% | {n_projected}/{n_total_grid} obs in overlap{gap_info})', 
                 fontsize=13, fontweight='bold', pad=10)
    
    # =========================================================================
    # PANEL 3: RESULTADO FINAL (curva desplazada + observaciones)
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    
    # Calcular rangos para sombreado de overlap
    mjd_synth_example = synth_curves[filters[0]][0]
    mjd_sn_min = mjd_synth_example.min() + desplaz
    mjd_sn_max = mjd_synth_example.max() + desplaz
    
    if len(df_proj) > 0:
        mjd_grid_min = df_proj['mjd'].min()
        mjd_grid_max = df_proj['mjd'].max()
        overlap_min = max(mjd_sn_min, mjd_grid_min)
        overlap_max = min(mjd_sn_max, mjd_grid_max)
        
        # Zona de overlap sombreada
        if overlap_min < overlap_max:
            ax3.axvspan(overlap_min, overlap_max, alpha=0.1, color='green', zorder=0)
    
    # Líneas verticales de referencia
    ax3.axvline(mjd_sn_min, color='blue', linestyle='--', linewidth=1.5, 
                alpha=0.4, label='SN range')
    ax3.axvline(mjd_sn_max, color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
    
    for filt in filters:
        color = colors.get(filt, 'gray')
        
        # Curva sintética DESPLAZADA
        mjd_synth, mag_synth = synth_curves[filt]
        mjd_displaced = mjd_synth + desplaz
        
        ax3.plot(mjd_displaced, mag_synth, '-', color=color, 
                alpha=0.5, linewidth=3, label=f'Template {filt}')
        
        # Proyecciones
        df_filt = df_proj[df_proj['filter'] == filt]
        if len(df_filt) > 0:
            detections = df_filt[df_filt['upperlimit'] == 'F']
            upper_limits = df_filt[df_filt['upperlimit'] == 'T']
            
            if len(detections) > 0:
                ax3.scatter(detections['mjd'], detections['magnitud_proyectada'], 
                          marker='o', s=200, color=color,
                          edgecolors='black', linewidths=3, 
                          label=f'Detections {filt} (n={len(detections)})', zorder=10)
            
            if len(upper_limits) > 0:
                ax3.scatter(upper_limits['mjd'], upper_limits['magnitud_proyectada'],
                          marker='v', s=200, facecolors='none',
                          edgecolors=color, linewidths=3,
                          label=f'Upper Limits {filt} (n={len(upper_limits)})', zorder=10)
    
    ax3.invert_yaxis()
    ax3.set_xlabel('Modified Julian Date (MJD)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Apparent Magnitude', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=11)
    ax3.legend(loc='best', fontsize=11, framealpha=0.95, ncol=2)
    
    n_det = len(df_proj[df_proj['upperlimit'] == 'F'])
    n_ul = len(df_proj[df_proj['upperlimit'] == 'T'])
    
    # Calcular coverage temporal (qué % de la SN está cubierto por observaciones)
    if len(df_proj) > 0:
        sn_duration = mjd_sn_max - mjd_sn_min
        overlap_duration = min(mjd_sn_max, df_proj['mjd'].max()) - max(mjd_sn_min, df_proj['mjd'].min())
        coverage_pct = 100 * overlap_duration / sn_duration if sn_duration > 0 else 0
        ax3.set_title(f'Step 3: Final Projected Light Curve ({n_det} det, {n_ul} UL | Coverage: {coverage_pct:.0f}%)', 
                     fontsize=14, fontweight='bold', pad=10)
    else:
        ax3.set_title(f'Step 3: Final Projected Light Curve (NO OBSERVATIONS)', 
                     fontsize=14, fontweight='bold', pad=10)
    
    # =========================================================================
    # TÍTULO GENERAL
    # =========================================================================
    sn_name = config.get('sn_name', 'Unknown')
    sn_type = config.get('tipo', config.get('sn_type', '?'))
    z = config.get('redshift', 0.0)
    oid = proj_result['field_selected']
    
    fig.suptitle(f"{sn_name} (Type {sn_type}) - z={z:.3f} - Field {oid}", 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_multiband_curves(df_proj, synth_curves, proj_result, config, output_file):
    """Plot principal: curvas sintéticas + proyecciones multi-banda"""
    
    filters = list(synth_curves.keys())
    n_filters = len(filters)
    
    fig = plt.figure(figsize=(12, 3*n_filters))
    gs = gridspec.GridSpec(n_filters, 1, hspace=0.3)
    
    colors = {'g': 'green', 'r': 'red', 'i': 'brown', 
              'U': 'purple', 'B': 'blue', 'V': 'cyan', 'R': 'orange', 'I': 'maroon'}
    
    for i, filt in enumerate(filters):
        ax = fig.add_subplot(gs[i])
        
        # Curva sintética
        mjd_synth, mag_synth = synth_curves[filt]
        ax.plot(mjd_synth, mag_synth, '-', color=colors.get(filt, 'gray'), 
                alpha=0.3, linewidth=2, label=f'Sintética {filt}')
        
        # Proyecciones
        df_filt = df_proj[df_proj['filter'] == filt]
        if len(df_filt) > 0:
            detections = df_filt[df_filt['upperlimit'] == 'F']
            upper_limits = df_filt[df_filt['upperlimit'] == 'T']
            
            if len(detections) > 0:
                ax.scatter(detections['mjd'], detections['magnitud_proyectada'], 
                          marker='o', s=100, color=colors.get(filt, 'gray'),
                          edgecolors='black', linewidths=2, 
                          label=f'Detecciones {filt} ({len(detections)})', zorder=5)
            
            if len(upper_limits) > 0:
                ax.scatter(upper_limits['mjd'], upper_limits['magnitud_proyectada'],
                          marker='v', s=100, color='white',
                          edgecolors=colors.get(filt, 'gray'), linewidths=2,
                          label=f'Upper Limits {filt} ({len(upper_limits)})', zorder=5)
        
        ax.invert_yaxis()
        ax.set_ylabel('Magnitud', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        if i == n_filters - 1:
            ax.set_xlabel('MJD', fontsize=12)
        
        # Título en primer panel
        if i == 0:
            sn_name = config.get('sn_name', 'Unknown')
            sn_type = config.get('tipo', config.get('sn_type', '?'))
            z = config.get('redshift', 0.0)
            oid = proj_result['field_selected']
            offset = proj_result['offset_used']
            n_obs = proj_result['n_observations']
            
            n_det = len(df_proj[df_proj['upperlimit'] == 'F'])
            n_ul = len(df_proj[df_proj['upperlimit'] == 'T'])
            
            title = f"{sn_name} ({sn_type}) - z={z:.3f} - {oid}\n"
            title += f"Offset: {offset:+d} días | {n_obs} obs ({n_det} det, {n_ul} UL)"
            ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_temporal_overlap(df_proj, synth_curves, proj_result, config, output_file):
    """Plot de debug: rangos temporales y overlap por filtro"""
    
    filters = list(synth_curves.keys())
    n_filters = len(filters)
    
    fig, ax = plt.subplots(figsize=(12, 2 + n_filters*0.8))
    
    y_pos = 0
    all_mjds = []
    
    for filt in filters:
        mjd_synth, _ = synth_curves[filt]
        sn_min, sn_max = mjd_synth.min(), mjd_synth.max()
        all_mjds.extend([sn_min, sn_max])
        
        # Barra de la SN
        ax.barh(y_pos, sn_max - sn_min, left=sn_min, height=0.3,
                color='lightblue', edgecolor='blue', linewidth=2,
                label='SN range' if filt == filters[0] else '')
        
        # Barra de observaciones
        df_filt = df_proj[df_proj['filter'] == filt]
        if len(df_filt) > 0:
            obs_min, obs_max = df_filt['mjd'].min(), df_filt['mjd'].max()
            all_mjds.extend([obs_min, obs_max])
            
            ax.barh(y_pos + 0.4, obs_max - obs_min, left=obs_min, height=0.3,
                    color='lightgreen', edgecolor='green', linewidth=2,
                    label='Obs range' if filt == filters[0] else '')
            
            # Overlap
            overlap_min = max(sn_min, obs_min)
            overlap_max = min(sn_max, obs_max)
            if overlap_max > overlap_min:
                ax.barh(y_pos + 0.2, overlap_max - overlap_min, left=overlap_min,
                        height=0.5, color='yellow', alpha=0.5, edgecolor='orange',
                        linewidth=2, label='Overlap' if filt == filters[0] else '')
            
            # Texto info
            ax.text(sn_max + 5, y_pos + 0.35, f"{len(df_filt)} obs", 
                   fontsize=10, va='center')
        else:
            ax.text(sn_max + 5, y_pos + 0.35, "Sin obs", 
                   fontsize=10, va='center', color='red')
        
        # Label del filtro
        ax.text(sn_min - 10, y_pos + 0.35, f"Filtro {filt}", 
               fontsize=11, ha='right', va='center', fontweight='bold')
        
        y_pos += 1
    
    # Configuración
    ax.set_ylim(-0.5, y_pos)
    ax.set_yticks([])
    ax.set_xlabel('MJD', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Título
    offset = proj_result['offset_used']
    sn_name = config.get('sn_name', 'Unknown')
    title = f"Análisis Temporal - {sn_name} - Offset: {offset:+d} días\n"
    title += f"Mismo offset para todos los filtros (coherencia multi-banda)"
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_offset_effect(synth_curves, proj_result, config, output_file):
    """Plot de debug: efecto del offset temporal"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Tomar primer filtro como ejemplo
    first_filter = list(synth_curves.keys())[0]
    mjd_synth, mag_synth = synth_curves[first_filter]
    
    desplaz = proj_result['desplazamiento']
    mjd_displaced = mjd_synth + desplaz
    
    # Panel 1: Curva original
    ax1.plot(mjd_synth, mag_synth, 'o-', color='blue', linewidth=2, markersize=4)
    ax1.invert_yaxis()
    ax1.set_ylabel('Magnitud', fontsize=12)
    ax1.set_title('Curva SN Original (sin desplazar)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(mjd_synth[np.argmin(mag_synth)], color='red', linestyle='--', 
                linewidth=2, label=f'Máximo SN (MJD {mjd_synth[np.argmin(mag_synth)]:.1f})')
    ax1.legend()
    
    # Panel 2: Curva desplazada
    ax2.plot(mjd_displaced, mag_synth, 'o-', color='green', linewidth=2, markersize=4)
    ax2.invert_yaxis()
    ax2.set_xlabel('MJD', fontsize=12)
    ax2.set_ylabel('Magnitud', fontsize=12)
    
    offset = proj_result['offset_used']
    ax2.set_title(f'Después de aplicar Offset {offset:+d} días (Desplazamiento total: {desplaz:+.1f} días)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(mjd_displaced[np.argmin(mag_synth)], color='red', linestyle='--',
                linewidth=2, label=f'Máximo desplazado (MJD {mjd_displaced[np.argmin(mag_synth)]:.1f})')
    ax2.legend()
    
    # Añadir texto explicativo
    formula = f"Desplazamiento = MJD_pivote - MJD_máximo + Offset"
    ax2.text(0.02, 0.98, formula, transform=ax2.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_failure_reason(synth_curves, proj_result, config, output_file):
    """Plot explicando por qué no hubo proyección"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.text(0.5, 0.6, '❌ SN NO OBSERVABLE', 
           ha='center', va='center', fontsize=24, fontweight='bold',
           transform=ax.transAxes, color='red')
    
    reason = "No hubo overlap temporal entre la curva SN desplazada\ny las observaciones disponibles en la grilla"
    ax.text(0.5, 0.4, reason,
           ha='center', va='center', fontsize=14,
           transform=ax.transAxes)
    
    # Info adicional
    sn_name = config.get('sn_name', 'Unknown')
    offset = proj_result['offset_used']
    oid = proj_result['field_selected']
    
    info = f"SN: {sn_name} | OID: {oid} | Offset: {offset:+d} días"
    ax.text(0.5, 0.2, info,
           ha='center', va='center', fontsize=11,
           transform=ax.transAxes, style='italic')
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def _update_type_summary(type_dir, config, proj_result, iteration_label, sn_dir):
    """Actualiza CSV summary del tipo"""
    
    summary_file = type_dir / f"{config.get('tipo', 'Unknown')}_summary.csv"
    
    # Crear fila de resumen
    row = {
        'iteration': iteration_label,
        'sn_name': config.get('sn_name'),
        'redshift': config.get('redshift', 0.0),
        'ebv_total': config.get('extinction_total', 0.0),
        'field_oid': proj_result['field_selected'],
        'offset': proj_result['offset_used'],
        'n_observations': proj_result['n_observations'],
        'status': 'success' if proj_result['n_observations'] > 0 else 'failed',
        'path': str(sn_dir.relative_to(type_dir.parent))
    }
    
    # Append o crear
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    type_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_file, index=False)
