# 📚 IMPORTS PARA PROJECTION
# ============================
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from .utils import maximo_lc

def field_projection(fases, flux_y, df_obslog, tipo, selected_filter, offset, sn, selected_field=None, plot=False):
    # ✔️ Seleccionamos un campo/OID
    obs_log = df_obslog.copy()
    
    if selected_field is not None:
        print("selected_field:", selected_field)
        # Caso 1: Archivo con columna 'field' (obslog_I.txt)
        if 'field' in obs_log.columns:
            obs_log['field'] = obs_log['field'].apply(lambda x: x.split('_')[0])
            df_filtered = obs_log[
                (obs_log['field'] == selected_field) &
                (obs_log['filter'] == selected_filter)
            ]
        # Caso 2: Archivo con columna 'oid' (grid_diffmaglim_ZTF.csv)
        elif 'oid' in obs_log.columns:
            # Usar selected_field como OID específico
            df_oid = obs_log[
                (obs_log['oid'] == selected_field) &
                (obs_log['filter'] == selected_filter)
            ]
            
            # 🎯 IMPLEMENTACIÓN OPCIÓN A: Mejor maglimit por día
            if len(df_oid) > 0:
                df_oid['mjd_day'] = df_oid['mjd'].astype(int)
                # Tomar el mejor (más profundo) maglimit por día
                df_filtered = df_oid.loc[df_oid.groupby('mjd_day')['maglimit'].idxmax()].copy()
                print(f"📅 Días únicos con observaciones: {df_filtered['mjd_day'].nunique()}")
                print(f"📊 Observaciones totales → filtradas: {len(df_oid)} → {len(df_filtered)}")
            else:
                print(f"⚠️ No se encontraron observaciones para OID: {selected_field}")
                df_filtered = df_oid
        else:
            raise ValueError("El archivo obslog debe tener columna 'field' o 'oid'")
    else:
        # Sin campo específico: usar todo el archivo filtrado solo por filtro
        df_filtered = obs_log[
            (obs_log['filter'] == selected_filter)
        ]

    # Información ya se muestra en la celda principal, no duplicar aquí

    #if len(df_filtered) < 7:
    #    field_info = f"campo/OID {selected_field}" if selected_field else "datos disponibles"
    #    raise ValueError(f"❌ No hay suficientes observaciones (mínimo 7) en {field_info} con filtro {selected_filter}. Encontradas: {len(df_filtered)}")

    # ✔️ Verificar que tenemos observaciones disponibles
    if len(df_filtered) == 0:
        field_info = f"OID {selected_field}" if selected_field else "datos disponibles"
        raise ValueError(f"❌ No hay observaciones en {field_info} con filtro {selected_filter}")
    
    # ✔️ Usar todas las observaciones disponibles (sin restricción de mínimo 7)
    df_filtered_cut = df_filtered.copy()
    mjd_pivote = df_filtered_cut.iloc[0]['mjd']

    # ✔️ Calcular el máximo usando la función maximo_lc()
    maximum = maximo_lc(tipo, sn)

    # ✔️ Loop para buscar un offset que funcione
    df_filtered_cut_copy = []
    select_offset = np.random.choice(offset)
    desplazamiento = mjd_pivote - maximum + select_offset

    fases_ajustadas = [fecha + desplazamiento for fecha in fases]

    df_filtered_cut_copy = df_filtered_cut[
        (df_filtered_cut['mjd'] >= min(fases_ajustadas)) &
        (df_filtered_cut['mjd'] <= max(fases_ajustadas))
    ]
    
    df_filtered_cut = df_filtered_cut_copy.copy()
    
    # ✔️ Verificar si la supernova es observable
    if len(df_filtered_cut) == 0:
        field_info = f"OID {selected_field}" if selected_field else "campo seleccionado"
        print(f"❌ SUPERNOVA NO OBSERVABLE")
        print(f"   • Motivo: Sin observaciones en rango temporal de la SN")
        print(f"   • Campo/OID: {field_info}")
        print(f"   • Filtro: {selected_filter}")
        print(f"   • Rango SN: MJD {min(fases_ajustadas):.1f} - {max(fases_ajustadas):.1f}")
        print(f"   • Offset usado: {select_offset}")
        print(f"   • Observaciones disponibles: {len(df_filtered)} total")
        
        # Crear DataFrame vacío pero con columnas necesarias para mantener consistencia
        empty_df = pd.DataFrame(columns=['mjd', 'maglimit', 'magnitud_proyectada', 'upperlimit', 'detected'])
        return empty_df

    # ✔️ Interpolación
    interpolation_function = interpolate.interp1d(
        fases_ajustadas, flux_y, kind='linear', fill_value='extrapolate'
    )

    df_filtered_cut['magnitud_proyectada'] = interpolation_function(df_filtered_cut['mjd'])

    # ✔️ Mag limit y detección
    df_filtered_cut['magnitud_proyectada'] = df_filtered_cut[['maglimit', 'magnitud_proyectada']].min(axis=1)
    df_filtered_cut['upperlimit'] = (df_filtered_cut['maglimit'] == df_filtered_cut['magnitud_proyectada']).map({True: 'T', False: 'F'})
    df_filtered_cut['detected'] = (df_filtered_cut['upperlimit'] == 'F')  # True si es detección, False si es upper limit

    # ✔️ Plot (solo si está habilitado)
    if plot:
        df_filtered_cut_T = df_filtered_cut[df_filtered_cut['upperlimit'] == 'T']
        df_filtered_cut_F = df_filtered_cut[df_filtered_cut['upperlimit'] == 'F']

        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        axs[0].plot(fases, flux_y, '.', label='Synthetic LC')
        axs[0].invert_yaxis()
        axs[0].legend()

        axs[1].scatter(df_filtered_cut_F['mjd'], df_filtered_cut_F['magnitud_proyectada'], marker='o', label='Detections')
        axs[1].scatter(df_filtered_cut_T['mjd'], df_filtered_cut_T['magnitud_proyectada'], marker='^', label='Upper Limits')
        #axs[1].set_ylim(np.min(flux_y), np.max(flux_y))
        axs[1].invert_yaxis()
        axs[1].legend()

        axs[2].plot(fases_ajustadas, flux_y, '.', label='Synthetic LC shifted')
        axs[2].plot(df_filtered_cut.mjd, df_filtered_cut.maglimit, '^', label="Mag limit")
        axs[2].invert_yaxis()
        axs[2].legend()

        plt.tight_layout()
        for ax_ in axs:
            ax_.tick_params('both', length=6, width=1, which='major', direction='in')
            ax_.tick_params('both', length=3, width=1, which='minor', direction='in')
            ax_.xaxis.set_ticks_position('both')
            ax_.yaxis.set_ticks_position('both')

        plt.show()
        print("   📊 Gráfico de debug mostrado (plot=True)")
    else:
        print("   📊 Gráfico de debug omitido (plot=False)")
    
    # ✔️ Reporte final de observabilidad
    total_points = len(df_filtered_cut)
    detections = len(df_filtered_cut[df_filtered_cut['detected'] == True])
    upper_limits = len(df_filtered_cut[df_filtered_cut['detected'] == False])
    detection_rate = (detections / total_points * 100) if total_points > 0 else 0
    
    field_info = f"OID {selected_field}" if selected_field else "campo seleccionado"
    print(f"\n✅ SUPERNOVA OBSERVABLE")
    print(f"   • Campo/OID: {field_info}")
    print(f"   • Puntos proyectados: {total_points}")
    print(f"   • Detecciones: {detections}")
    print(f"   • Upper limits: {upper_limits}")
    print(f"   • Tasa detección: {detection_rate:.1f}%")

    return df_filtered_cut