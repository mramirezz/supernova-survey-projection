import pandas as pd

# Cargar grilla
df = pd.read_csv('data/grid_diffmaglim_ZTF.csv')

# Filtrar por OID y filtro
oid_data = df[(df['oid'] == 'ZTF19ablfrui') & (df['filter'] == 'r')]

print(f"Total observaciones: {len(oid_data)}")
print(f"Rango MJD: {oid_data['mjd'].min():.1f} - {oid_data['mjd'].max():.1f}")
print(f"Días únicos: {oid_data['mjd'].astype(int).nunique()}")
print(f"\nMaglimit range: {oid_data['maglimit'].min():.2f} - {oid_data['maglimit'].max():.2f}")

# Ver distribución temporal
oid_data_sorted = oid_data.sort_values('mjd')
print(f"\n=== Primeras 10 observaciones ===")
print(oid_data_sorted[['mjd', 'maglimit']].head(10).to_string())

print(f"\n=== Últimas 10 observaciones ===")
print(oid_data_sorted[['mjd', 'maglimit']].tail(10).to_string())

# Verificar rango de la SN (del config)
print(f"\n=== VERIFICACIÓN CON LA SN ===")
print(f"Máximo de SN1999gi: MJD 51644.1")
print(f"Rango curva sintética: ~51525 - 51660 (135 días)")

# Simular el desplazamiento
mjd_pivote = oid_data_sorted.iloc[0]['mjd']
maximum = 51644.1
offset_usado = -137  # Del log vimos que el offset estaba alrededor de -137

desplazamiento = mjd_pivote - maximum + offset_usado
print(f"\nmjd_pivote: {mjd_pivote:.1f}")
print(f"offset estimado: {offset_usado}")
print(f"desplazamiento: {desplazamiento:.1f}")

# Rango de la SN desplazada
fase_min_original = 51525.0
fase_max_original = 51660.0
fase_min_ajustada = fase_min_original + desplazamiento
fase_max_ajustada = fase_max_original + desplazamiento

print(f"\nRango SN desplazada: {fase_min_ajustada:.1f} - {fase_max_ajustada:.1f}")

# Observaciones en ese rango
obs_en_rango = oid_data[(oid_data['mjd'] >= fase_min_ajustada) & 
                         (oid_data['mjd'] <= fase_max_ajustada)]
print(f"\nObservaciones en rango de la SN: {len(obs_en_rango)}")
print(obs_en_rango[['mjd', 'maglimit']].to_string())
