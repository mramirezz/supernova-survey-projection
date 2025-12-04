"""
PROYECCIÓN MULTI-BANDA SIMULTÁNEA
==================================

Este módulo maneja la proyección de supernovas en múltiples filtros fotométricos
SIMULTÁNEAMENTE, asegurando que:

1. Se usa el MISMO offset temporal para todos los filtros
2. Se proyecta en las fechas REALES donde cada filtro fue observado
3. Las observaciones multi-banda de la misma noche se mantienen juntas

Esto corrige el problema donde antes cada filtro se proyectaba independientemente
con diferentes offsets, creando una situación NO física.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from .utils import maximo_lc

def multiband_field_projection(
    curves_by_filter,  # Dict: {filter: (fases, flux_y)}
    df_obslog,
    tipo,
    available_filters,  # Lista de filtros disponibles ['r', 'g', 'i']
    offset,
    sn,
    selected_field=None,
    plot=False
):
    """
    Proyecta una SN en múltiples filtros SIMULTÁNEAMENTE.
    
    Parameters:
    -----------
    curves_by_filter : dict
        Diccionario con curvas por filtro: {filter: (fases_mjd, magnitudes)}
    df_obslog : DataFrame
        Grilla de observaciones con columnas: mjd, filter, maglimit, oid/field
    tipo : str
        Tipo de SN (Ia, II, Ibc)
    available_filters : list
        Lista de filtros a proyectar
    offset : array
        Rango de offsets temporales posibles
    sn : str
        Nombre de la SN
    selected_field : str
        OID o campo específico
    plot : bool
        Si mostrar gráfico de debug
        
    Returns:
    --------
    dict : {
        'projections': DataFrame con todas las proyecciones multi-banda,
        'offset_used': offset temporal usado,
        'field_selected': campo/OID seleccionado,
        'n_observations': número total de observaciones proyectadas
    }
    """
    
    obs_log = df_obslog.copy()
    
    # PASO 1: Seleccionar campo/OID
    # ==============================
    if selected_field is not None:
        print(f"   [MULTI-BAND] Campo/OID seleccionado: {selected_field}")
        
        if 'field' in obs_log.columns:
            # SUDARE: columna 'field'
            obs_log['field'] = obs_log['field'].apply(lambda x: x.split('_')[0])
            df_filtered = obs_log[obs_log['field'] == selected_field]
        elif 'oid' in obs_log.columns:
            # ZTF: columna 'oid'
            df_filtered = obs_log[obs_log['oid'] == selected_field]
        else:
            raise ValueError("El archivo obslog debe tener columna 'field' o 'oid'")
    else:
        df_filtered = obs_log.copy()
    
    if len(df_filtered) == 0:
        field_info = f"OID/campo {selected_field}" if selected_field else "datos disponibles"
        raise ValueError(f"[ERROR] No hay observaciones en {field_info}")
    
    # PASO 2: Filtrar por filtros disponibles (NO por uno solo)
    # ==========================================================
    df_multiband = df_filtered[df_filtered['filter'].isin(available_filters)].copy()
    
    if len(df_multiband) == 0:
        raise ValueError(f"[ERROR] No hay observaciones en filtros {available_filters}")
    
    print(f"   [MULTI-BAND] Observaciones totales: {len(df_multiband)}")
    print(f"   [MULTI-BAND] Filtros encontrados: {df_multiband['filter'].unique()}")
    
    # Tomar mejor maglimit por día Y por filtro
    df_multiband['mjd_day'] = df_multiband['mjd'].astype(int)
    
    print(f"   [DEBUG] Antes de filtrar 'mejor por día':")
    for filt in df_multiband['filter'].unique():
        df_filt = df_multiband[df_multiband['filter'] == filt]
        print(f"      - Filtro {filt}: {len(df_filt)} obs | MJD {df_filt['mjd'].min():.1f} - {df_filt['mjd'].max():.1f}")
    
    df_best = df_multiband.loc[
        df_multiband.groupby(['mjd_day', 'filter'])['maglimit'].idxmax()
    ].copy()
    
    print(f"   [MULTI-BAND] Después de filtrar mejor por día+filtro: {len(df_best)}")
    for filt in df_multiband['filter'].unique():
        n_obs = len(df_best[df_best['filter'] == filt])
        print(f"      - Filtro {filt}: {n_obs} observaciones")
    
    # PASO 3: Calcular offset temporal ÚNICO
    # =======================================
    # Seleccionar un MJD pivote ALEATORIO de la grilla (no siempre el primero)
    # Esto evita sesgo de siempre proyectar en el mismo punto de la ventana
    mjd_min_grid = df_best['mjd'].min()
    mjd_max_grid = df_best['mjd'].max()
    
    # Elegir un MJD pivote aleatorio dentro de la ventana observacional
    mjd_pivote = np.random.uniform(mjd_min_grid, mjd_max_grid)
    
    maximum = maximo_lc(tipo, sn)
    select_offset = np.random.choice(offset)
    desplazamiento = mjd_pivote - maximum + select_offset
    
    print(f"   [MULTI-BAND] MJD pivote (aleatorio): {mjd_pivote:.1f}")
    print(f"   [MULTI-BAND] MJD rango grilla: {mjd_min_grid:.1f} - {mjd_max_grid:.1f}")
    print(f"   [MULTI-BAND] Ventana observacional: {mjd_max_grid - mjd_min_grid:.1f} días")
    print(f"   [MULTI-BAND] Máximo SN: {maximum:.1f}")
    print(f"   [MULTI-BAND] Offset seleccionado: {select_offset:+d}")
    print(f"   [MULTI-BAND] Desplazamiento total: {desplazamiento:.1f}")
    
    # PASO 4: Proyectar cada filtro en sus observaciones específicas
    # ===============================================================
    all_projections = []
    
    print("\n   [MULTI-BAND] === PROYECCIÓN POR FILTRO ===")
    
    for filt in available_filters:
        if filt not in curves_by_filter:
            print(f"   [MULTI-BAND] [WARNING] No hay curva sintética para filtro {filt}, saltando...")
            continue
            
        fases, flux_y = curves_by_filter[filt]
        
        # Desplazar curva con el MISMO offset para todos los filtros
        fases_ajustadas = fases + desplazamiento
        
        print(f"\n   Filtro {filt}:")
        print(f"      • Curva SN: {len(fases)} puntos, rango MJD {fases_ajustadas.min():.1f} - {fases_ajustadas.max():.1f}")
        print(f"      • Duración curva SN desplazada: {fases_ajustadas.max() - fases_ajustadas.min():.1f} días")
        
        # Filtrar observaciones de este filtro en el rango de la SN
        df_filter = df_best[df_best['filter'] == filt].copy()
        
        if len(df_filter) > 0:
            print(f"      • Observaciones grilla {filt}: {len(df_filter)} obs, MJD {df_filter['mjd'].min():.1f} - {df_filter['mjd'].max():.1f}")
        else:
            print(f"      • [WARNING] No hay observaciones de filtro {filt} en la grilla")
        
        # DEBUG: Info de overlap (sin ternario dentro del f-string)
        obs_min = f"{df_filter['mjd'].min():.1f}" if len(df_filter) > 0 else "N/A"
        obs_max = f"{df_filter['mjd'].max():.1f}" if len(df_filter) > 0 else "N/A"
        print(f"      • [DEBUG] Buscando overlap entre SN [{fases_ajustadas.min():.1f}, {fases_ajustadas.max():.1f}] y obs [{obs_min}, {obs_max}]")
        
        df_filter_in_range = df_filter[
            (df_filter['mjd'] >= fases_ajustadas.min()) &
            (df_filter['mjd'] <= fases_ajustadas.max())
        ].copy()
        
        print(f"      • Overlap: {len(df_filter_in_range)} observaciones dentro del rango de la SN")
        
        if len(df_filter_in_range) > 0:
            print(f"      • [DEBUG] MJDs con overlap: {sorted(df_filter_in_range['mjd'].values)[:10]}")  # Mostrar hasta 10
        
        if len(df_filter_in_range) == 0:
            continue
        
        # Interpolar curva en las fechas de observación
        interpolation_function = interpolate.interp1d(
            fases_ajustadas, flux_y, kind='linear', fill_value='extrapolate'
        )
        
        df_filter_in_range['magnitud_proyectada'] = interpolation_function(
            df_filter_in_range['mjd']
        )
        
        # Determinar detecciones vs upper limits
        df_filter_in_range['magnitud_proyectada'] = df_filter_in_range[
            ['maglimit', 'magnitud_proyectada']
        ].min(axis=1)
        
        df_filter_in_range['upperlimit'] = (
            df_filter_in_range['maglimit'] == df_filter_in_range['magnitud_proyectada']
        ).map({True: 'T', False: 'F'})
        
        df_filter_in_range['detected'] = (df_filter_in_range['upperlimit'] == 'F')
        
        all_projections.append(df_filter_in_range)
        
        detections = df_filter_in_range['detected'].sum()
        upper_limits = (~df_filter_in_range['detected']).sum()
        print(f"      • Resultado: {detections} detecciones, {upper_limits} upper limits")
    
    # PASO 5: Combinar todas las proyecciones
    # ========================================
    print("\n   [MULTI-BAND] === RESUMEN PROYECCIÓN ===")
    if len(all_projections) == 0:
        print(f"   [MULTI-BAND] [ERROR] SN NO OBSERVABLE en ningún filtro")
        print(f"   [MULTI-BAND] Posible causa: No overlap temporal entre curva SN y grilla de observaciones")
        empty_df = pd.DataFrame(columns=[
            'mjd', 'filter', 'maglimit', 'magnitud_proyectada', 
            'upperlimit', 'detected'
        ])
        return {
            'projections': empty_df,
            'offset_used': select_offset,
            'field_selected': selected_field,
            'n_observations': 0,
            'desplazamiento': desplazamiento
        }
    
    df_projected = pd.concat(all_projections, ignore_index=True).sort_values('mjd')
    
    total_points = len(df_projected)
    total_detections = df_projected['detected'].sum()
    total_ul = (~df_projected['detected']).sum()
    detection_rate = (total_detections / total_points * 100) if total_points > 0 else 0
    
    print(f"\n   [MULTI-BAND] [OK] SN OBSERVABLE")
    print(f"   [MULTI-BAND] Total proyectado: {total_points} observaciones")
    print(f"   [MULTI-BAND] Detecciones: {total_detections}")
    print(f"   [MULTI-BAND] Upper limits: {total_ul}")
    print(f"   [MULTI-BAND] Tasa detección: {detection_rate:.1f}%")
    print(f"   [MULTI-BAND] Rango temporal: MJD {df_projected['mjd'].min():.1f} - "
          f"{df_projected['mjd'].max():.1f}")
    
    # PASO 6: Plot opcional (TODO: mejorar para multi-banda)
    # =======================================================
    if plot:
        print(f"   [MULTI-BAND] [INFO] Gráfico multi-banda aún no implementado")
    
    return {
        'projections': df_projected,
        'offset_used': select_offset,
        'field_selected': selected_field,
        'n_observations': total_points,
        'desplazamiento': desplazamiento,
        'detection_rate': detection_rate,
        'grid_observations': df_multiband  # Grilla COMPLETA (antes de proyección)
    }
