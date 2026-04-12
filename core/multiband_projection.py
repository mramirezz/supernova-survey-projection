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
    plot=False,
    # Opcionales: “modo dataset” para garantizar mínimo de detecciones
    required_filter=None,               # ej: 'r'
    min_detections=None,                # ej: 7
    offset_search_mode='random',        # 'random' | 'grid' | 'deterministic'
    force_brighten_to_min_detections=False,
    max_force_brightening_mag=3.0,
    # Modo determinístico: divide la grilla en n_divisions partes,
    # coloca el ancla en el centro de la partición part_index (0-indexed)
    n_divisions=10,
    part_index=0,
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
    mjd_min_grid = df_best['mjd'].min()
    mjd_max_grid = df_best['mjd'].max()
    # ---------------------------------------------------------------------
    # DEFINICIÓN DE ANCLA ("anchor_time")
    # ---------------------------------------------------------------------
    # Para II: NO usar maximo_lc() (depende de maximum_II.txt con definición externa).
    #         Usamos el máximo flujo del template disponible: min(mag) en la banda
    #         requerida (si existe), si no en 'r', si no el primer filtro disponible.
    # Para otros tipos: mantener maximo_lc() como antes.
    maximum = None
    anchor_source = None

    if str(tipo) == 'II':
        # Elegir filtro para definir el máximo (ancla)
        preferred = None
        if required_filter is not None and required_filter in curves_by_filter:
            preferred = required_filter
        elif 'r' in curves_by_filter:
            preferred = 'r'
        elif len(curves_by_filter) > 0:
            preferred = list(curves_by_filter.keys())[0]

        if preferred is None:
            # extremo raro: sin curvas
            maximum = float(df_best['mjd'].min())
            anchor_source = 'fallback_grid_start'
        else:
            f_arr, mag_arr = curves_by_filter[preferred]
            # fecha de máximo flujo (mínima mag) en esa banda
            idx = int(np.argmin(np.asarray(mag_arr)))
            maximum = float(np.asarray(f_arr)[idx])
            anchor_source = f'min_mag_{preferred}'
    else:
        maximum = float(maximo_lc(tipo, sn))
        anchor_source = 'maximo_lc'

    # Si el anchor cae fuera del rango del template (p.ej. templates sin premax),
    # anclar en el primer MJD del template para no perder la ventana observacional.
    phase_min = None
    phase_max = None
    for _, (f_arr, _) in curves_by_filter.items():
        try:
            f_min = float(np.min(f_arr))
            f_max = float(np.max(f_arr))
        except Exception:
            continue
        phase_min = f_min if phase_min is None else min(phase_min, f_min)
        phase_max = f_max if phase_max is None else max(phase_max, f_max)

    anchor_time = float(maximum)
    if phase_min is not None and phase_max is not None:
        if anchor_time < float(phase_min) or anchor_time > float(phase_max):
            anchor_time = float(phase_min)
            if anchor_source is not None:
                anchor_source = f"{anchor_source}_clipped_to_phase_min"

    # Normalizar offsets a array (int)
    offset_values = np.array(offset, dtype=int)

    # Seleccionar un MJD pivote + offset.
    # Default: random (como antes). Si offset_search_mode='grid' y se pide min_detections,
    # buscamos el mejor desplazamiento para el required_filter.
    mjd_pivote = None
    select_offset = None

    def _count_detections_for_displacement(displ):
        """Cuenta detecciones en required_filter para un desplazamiento dado (solo overlap)."""
        if required_filter is None or required_filter not in curves_by_filter:
            return 0, 0
        df_req = df_best[df_best['filter'] == required_filter].copy()
        if len(df_req) == 0:
            return 0, 0
        fases_req, mag_req = curves_by_filter[required_filter]
        fases_aj = fases_req + float(displ)
        df_in = df_req[(df_req['mjd'] >= fases_aj.min()) & (df_req['mjd'] <= fases_aj.max())].copy()
        if len(df_in) == 0:
            return 0, 0
        interp_fn = interpolate.interp1d(fases_aj, mag_req, kind='linear', fill_value='extrapolate')
        m_model = interp_fn(df_in['mjd'].values)
        n_det = int((m_model < df_in['maglimit'].values).sum())
        return n_det, int(len(df_in))

    use_grid = (
        required_filter is not None
        and min_detections is not None
        and str(offset_search_mode).lower() == 'grid'
        and required_filter in df_best['filter'].unique()
    )

    use_deterministic = (str(offset_search_mode).lower() == 'deterministic')

    if use_deterministic:
        # Modo determinístico: dividir ventana observacional en n_divisions partes
        # y colocar el ancla en el centro de la partición part_index
        part_size = (mjd_max_grid - mjd_min_grid) / float(n_divisions)
        mjd_pivote = mjd_min_grid + part_size * (part_index + 0.5)
        select_offset = 0
        desplazamiento = mjd_pivote - anchor_time + select_offset
        print(f"   [MULTI-BAND] Modo DETERMINÍSTICO: div {part_index+1}/{n_divisions}, "
              f"pivote={mjd_pivote:.1f}")

    elif use_grid:
        df_req = df_best[df_best['filter'] == required_filter].copy()
        pivots = df_req['mjd'].values

        best = None  # (n_det, n_obs, pivot, off, displ)
        for piv in pivots:
            base = float(piv) - float(anchor_time)
            for off in offset_values:
                displ = base + float(off)
                n_det, n_obs = _count_detections_for_displacement(displ)
                cand = (n_det, n_obs, float(piv), int(off), float(displ))
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                    best = cand
                if n_det >= int(min_detections):
                    best = cand
                    break
            if best is not None and best[0] >= int(min_detections):
                break

        if best is None:
            mjd_pivote = float(np.random.uniform(mjd_min_grid, mjd_max_grid))
            select_offset = int(np.random.choice(offset_values))
            desplazamiento = mjd_pivote - anchor_time + select_offset
        else:
            _, _, mjd_pivote, select_offset, desplazamiento = best
    else:
        # Random: pivote uniforme + offset aleatorio
        mjd_pivote = float(np.random.uniform(mjd_min_grid, mjd_max_grid))
        select_offset = int(np.random.choice(offset_values))
        desplazamiento = mjd_pivote - anchor_time + select_offset
    
    # Momento observado donde cae el "anchor_time" del template (típicamente el máximo si existe):
    # anchor_obs_mjd = anchor_time + desplazamiento = mjd_pivote + select_offset
    anchor_obs_mjd = float(anchor_time) + float(desplazamiento)

    print(f"   [MULTI-BAND] MJD pivote: {mjd_pivote:.1f}")
    print(f"   [MULTI-BAND] MJD rango grilla: {mjd_min_grid:.1f} - {mjd_max_grid:.1f}")
    print(f"   [MULTI-BAND] Ventana observacional: {mjd_max_grid - mjd_min_grid:.1f} días")
    print(f"   [MULTI-BAND] Máximo SN (raw): {float(maximum):.1f}")
    if phase_min is not None and phase_max is not None:
        print(f"   [MULTI-BAND] Rango template: {phase_min:.1f} - {phase_max:.1f}")
    print(f"   [MULTI-BAND] Ancla usada: {anchor_time:.1f}")
    print(f"   [MULTI-BAND] Ancla (def): {anchor_source}")
    print(f"   [MULTI-BAND] Offset seleccionado: {int(select_offset):+d}")
    print(f"   [MULTI-BAND] Desplazamiento total: {desplazamiento:.1f}")
    print(f"   [MULTI-BAND] Ancla observada (pivote+offset): {anchor_obs_mjd:.1f}")
    
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
        
        # Guardar magnitud del modelo ANTES de aplicar maglimit (útil para fallbacks)
        df_filter_in_range['magnitud_modelo'] = interpolation_function(
            df_filter_in_range['mjd']
        )
        
        # Determinar detecciones vs upper limits
        df_filter_in_range['magnitud_proyectada'] = df_filter_in_range[
            ['maglimit', 'magnitud_modelo']
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
            'mjd', 'filter', 'maglimit', 'magnitud_modelo', 'magnitud_proyectada',
            'upperlimit', 'detected'
        ])
        return {
            'projections': empty_df,
            'offset_used': select_offset,
            'field_selected': selected_field,
            'n_observations': 0,
            'desplazamiento': desplazamiento,
            'mjd_pivote': mjd_pivote,
            'anchor_time': float(anchor_time),
            'anchor_obs_mjd': float(anchor_obs_mjd),
            'maximum_raw': float(maximum),
            'anchor_source': anchor_source,
            'template_phase_min': phase_min,
            'template_phase_max': phase_max,
            'offset_search_mode_used': 'deterministic' if use_deterministic else ('grid' if use_grid else 'random'),
            'required_filter': required_filter,
            'min_detections_required': min_detections,
        }
    
    df_projected = pd.concat(all_projections, ignore_index=True).sort_values('mjd')

    # “Último recurso”: si no se cumple el mínimo por banda, aplicar brightening global (limitado)
    forced_shift = 0.0
    forced_applied = False
    if (
        required_filter is not None
        and min_detections is not None
        and bool(force_brighten_to_min_detections)
        and 'magnitud_modelo' in df_projected.columns
    ):
        df_req = df_projected[df_projected['filter'] == required_filter].copy()
        # Solo se puede “garantizar” si hay al menos K observaciones en overlap en ese filtro
        if len(df_req) >= int(min_detections):
            deltas = (df_req['magnitud_modelo'].values - df_req['maglimit'].values)
            deltas_sorted = np.sort(deltas)
            kth = float(deltas_sorted[int(min_detections) - 1])
            needed = max(0.0, kth + 1e-3)  # epsilon para quedar justo detectado
            if needed > 0.0 and needed <= float(max_force_brightening_mag):
                forced_shift = float(needed)
                df_projected['magnitud_modelo'] = df_projected['magnitud_modelo'] - forced_shift
                df_projected['magnitud_proyectada'] = df_projected[['maglimit', 'magnitud_modelo']].min(axis=1)
                df_projected['upperlimit'] = (df_projected['maglimit'] == df_projected['magnitud_proyectada']).map({True: 'T', False: 'F'})
                df_projected['detected'] = (df_projected['upperlimit'] == 'F')
                forced_applied = True
    
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
        'grid_observations': df_multiband,  # Grilla COMPLETA (antes de proyección)
        'forced_brightening_mag': forced_shift,
        'forced_brightening_applied': forced_applied,
        'mjd_pivote': mjd_pivote,
        'anchor_time': float(anchor_time),
        'anchor_obs_mjd': float(anchor_obs_mjd),
        'maximum_raw': float(maximum),
        'anchor_source': anchor_source,
        'template_phase_min': phase_min,
        'template_phase_max': phase_max,
        'offset_search_mode_used': 'deterministic' if use_deterministic else ('grid' if use_grid else 'random'),
        'required_filter': required_filter,
        'min_detections_required': min_detections,
        'n_divisions': n_divisions if use_deterministic else None,
        'part_index': part_index if use_deterministic else None,
    }
