import pandas as pd
import numpy as np

# Datos del problema
maximum = 51644.1  # Del config
fases_sn = [51525.0, 51660.0]  # Rango aproximado

# Observaciones proyectadas (del archivo projection)
obs_proyectadas = [58681.458, 58684.462, 58692.485]

# Grilla del OID
df = pd.read_csv('data/grid_diffmaglim_ZTF.csv')
oid_data = df[(df['oid'] == 'ZTF19ablfrui') & (df['filter'] == 'r')].sort_values('mjd')
mjd_pivote = oid_data.iloc[0]['mjd']

print(f"=== DATOS ===")
print(f"Maximum SN: {maximum}")
print(f"Rango SN original: {fases_sn[0]} - {fases_sn[1]}")
print(f"mjd_pivote (primera obs OID): {mjd_pivote:.3f}")
print(f"\nObservaciones proyectadas:")
for obs in obs_proyectadas:
    print(f"  MJD {obs:.3f}")

print(f"\n=== CÁLCULO INVERSO ===")
# Si las obs proyectadas son 58681, 58684, 58692
# Y esas deben estar en el rango de fases_ajustadas
# Entonces: fases_ajustadas = fases + desplazamiento
# Y: desplazamiento = mjd_pivote - maximum + offset

# Para que 58681 esté en el rango, necesito:
# min(fases_ajustadas) <= 58681 <= max(fases_ajustadas)
# 51525 + desplazamiento <= 58681
# desplazamiento >= 7156

# max(fases_ajustadas) >= 58692
# 51660 + desplazamiento >= 58692
# desplazamiento >= 7032

# Probemos qué offset genera eso:
# desplazamiento = mjd_pivote - maximum + offset
# offset = desplazamiento - mjd_pivote + maximum

# Si desplazamiento = 7156:
desp_min = 58681 - 51525
offset_para_esto = desp_min - mjd_pivote + maximum
print(f"\nPara que min(fases_ajustadas) = 58681:")
print(f"  desplazamiento needed: {desp_min}")
print(f"  offset needed: {offset_para_esto:.1f}")

# Rango de offsets por defecto
print(f"\n=== VERIFICACIÓN CON CÓDIGO ===")
# Del main.py, offset_range por defecto es [-30, 30]
offsets_posibles = np.arange(-30, 30, 1)
print(f"Offsets posibles: {offsets_posibles[0]} a {offsets_posibles[-1]}")

# Con cada offset, calcular el rango de fases_ajustadas
for offset in [offsets_posibles[0], 0, offsets_posibles[-1]]:
    desplazamiento = mjd_pivote - maximum + offset
    fase_min_aj = fases_sn[0] + desplazamiento
    fase_max_aj = fases_sn[1] + desplazamiento
    print(f"\nOffset {offset:+3d}: desplaz={desplazamiento:.1f}")
    print(f"  Rango SN desplazada: {fase_min_aj:.1f} - {fase_max_aj:.1f}")
    
    # Cuántas obs caen en ese rango
    obs_en_rango = oid_data[(oid_data['mjd'] >= fase_min_aj) & 
                             (oid_data['mjd'] <= fase_max_aj)]
    print(f"  Obs en rango: {len(obs_en_rango)}")
    if len(obs_en_rango) > 0:
        print(f"  MJDs: {obs_en_rango['mjd'].values[:5]}")
