# ARQUITECTURA DE GUARDADO MULTI-BANDA

## Problema Actual
- El sistema viejo guardaba por filtro independiente → difícil de mantener
- Estructura: `outputs/II/SN1999gi/ZTF/filter_r/run_xxx/target_xxx/`
- Cada filtro en carpeta separada → no se ve que son la misma proyección

## Propuesta Nueva: Estructura por Ejecución

```
outputs/
├── multiband_runs/
│   └── ZTF_20251127_195907/              # Survey_Fecha_Hora
│       ├── run_metadata.json             # Metadata del batch completo
│       ├── summary_statistics.csv        # Resumen: cuántas proyecciones, detecciones, etc.
│       │
│       ├── projections/                  # Todas las proyecciones exitosas
│       │   ├── Ia_SN2007af_iter001/
│       │   │   ├── projection_data.csv   # TODAS las bandas en un CSV
│       │   │   │                         # Columnas: mjd, filter, mag_projected, maglimit, detected, upperlimit
│       │   │   ├── synthetic_curves.csv  # Curvas sintéticas de g,r,i
│       │   │   ├── config.json          # z, ebv_host, ebv_mw, offset_used, field/oid
│       │   │   └── plots/
│       │   │       ├── multiband_projection.png  # Curvas g+r+i en MISMO plot
│       │   │       ├── debug_step1_spectra.png   # Debug: espectros leídos
│       │   │       ├── debug_step2_corrections.png
│       │   │       ├── debug_step3_synthphot.png
│       │   │       └── debug_step8_overlap.png   # Rangos MJD de cada filtro
│       │   │
│       │   ├── Ia_SN2011fe_iter002/
│       │   │   └── ... (misma estructura)
│       │   │
│       │   └── Ia_SN2011fe_iter003/
│       │       └── ...
│       │
│       └── failed/                       # Proyecciones sin overlap
│           ├── II_SN2013am_iter001/
│           │   ├── config.json
│           │   ├── failure_reason.txt    # "No temporal overlap"
│           │   └── plots/
│           │       └── debug_ranges.png  # Muestra POR QUÉ no hubo overlap
│           └── ...
```

## Ventajas

### 1. Agrupación por Ejecución
- **Todo en una carpeta** por fecha/survey
- Fácil encontrar: "¿Qué salió el 27 de nov con ZTF?"
- Metadata completo del batch en `run_metadata.json`

### 2. Datos Multi-banda Unificados
```csv
# projection_data.csv - TODO en un archivo
mjd,filter,mag_projected,maglimit,detected,upperlimit,field
60097.2,g,19.5,20.8,True,F,ZTF23aanrksv
60097.2,r,19.3,20.5,True,F,ZTF23aanrksv  # Mismo MJD, mismo offset
60102.3,g,19.8,20.7,True,F,ZTF23aanrksv
60102.3,r,19.6,20.4,True,F,ZTF23aanrksv
```
→ **Fácil de leer para clasificación**: un CSV con todas las bandas

### 3. Plots de Debug Completos
```python
# debug_step8_overlap.png mostraría:
┌─────────────────────────────────────────┐
│ PASO 8: Proyección Multi-banda         │
│ OID: ZTF23aanrksv, Offset: +27         │
├─────────────────────────────────────────┤
│ Filtro g:                               │
│   SN: ████████████████ (60095-60175)    │
│   Obs:    ████████     (60080-60105)    │
│   Overlap: ███ (3 obs)                  │
├─────────────────────────────────────────┤
│ Filtro r:                               │
│   SN: ████████████████ (60095-60175)    │
│   Obs:    ████████████ (60080-60115)    │
│   Overlap: ██████ (6 obs)               │
└─────────────────────────────────────────┘
```

### 4. Para Clasificación
```python
# Lectura simple para clasificar
import pandas as pd

# Cargar proyección específica
df = pd.read_csv('outputs/multiband_runs/ZTF_20251127_195907/projections/Ia_SN2007af_iter001/projection_data.csv')

# Separar por filtro
df_g = df[df['filter'] == 'g']
df_r = df[df['filter'] == 'r']

# Usar para features de ML
features = extract_features(df_g, df_r, df_i)
```

### 5. Tracking y Reproducibilidad
```json
// config.json por cada proyección
{
  "sn_name": "SN2007af",
  "sn_type": "Ia",
  "redshift": 0.0481,
  "ebv_host": 0.053,
  "ebv_mw": 0.067,
  "ebv_total": 0.120,
  "survey": "ZTF",
  "field_oid": "ZTF23aanrksv",
  "offset_used": 27,
  "desplazamiento": 5931.2,
  "mjd_range_sn": [60095.2, 60175.2],
  "filters_projected": ["g", "r"],
  "n_observations": 9,
  "n_detections": 9,
  "detection_rate": 100.0,
  "seed": 42,
  "batch_id": "20251127_195907_b9eb25aa",
  "iteration": "iter_0001_of_0003",
  "timestamp": "2025-11-27 19:59:07"
}
```

## CSV Principal: summary_statistics.csv

```csv
iteration,sn_type,sn_name,z,ebv_total,field_oid,offset,filters,n_obs,n_det,det_rate,status,path
iter001,Ia,SN2007af,0.048,0.120,ZTF23aanrksv,+27,"g,r",9,9,100.0,success,projections/Ia_SN2007af_iter001
iter002,Ia,SN2011fe,0.035,0.164,ZTF20aavxtjo,-24,"g,r",5,0,0.0,success,projections/Ia_SN2011fe_iter002
iter003,Ia,SN2011fe,0.014,0.341,ZTF22aaznxqp,-18,"g,r",17,10,58.8,success,projections/Ia_SN2011fe_iter003
iter004,II,SN2013am,0.048,0.105,ZTF22aazihmk,+13,,0,0,0.0,failed,failed/II_SN2013am_iter004
```

## Implementación

### Función Principal
```python
def save_multiband_projection(
    df_projected,           # DataFrame multi-banda
    synthetic_curves,       # Dict {filter: (mjd, mag)}
    projection_result,      # Dict con offset, field, etc.
    config,                 # Config completo
    batch_id,              # ID del batch
    iteration_label,       # iter_0001_of_0100
    base_output_dir='outputs/multiband_runs'
):
    """
    Guarda proyección multi-banda con estructura clara
    """
    # 1. Crear directorio de la ejecución
    run_dir = f"{base_output_dir}/{config['SURVEY']}_{batch_id}"
    
    # 2. Determinar si exitosa o fallida
    status = 'projections' if len(df_projected) > 0 else 'failed'
    
    # 3. Crear carpeta individual
    sn_dir = f"{run_dir}/{status}/{config['tipo']}_{config['sn_name']}_{iteration_label}"
    
    # 4. Guardar CSVs
    save_projection_csv(df_projected, sn_dir)
    save_synthetic_csv(synthetic_curves, sn_dir)
    
    # 5. Guardar config JSON
    save_config_json(config, projection_result, sn_dir)
    
    # 6. Generar plots
    create_multiband_plot(df_projected, synthetic_curves, sn_dir)
    create_debug_plots(projection_result, sn_dir)
    
    # 7. Actualizar summary CSV del batch
    update_summary_csv(run_dir, config, projection_result)
    
    return sn_dir
```

## ¿Te parece bien esta arquitectura?

- ✅ Todo agrupado por ejecución (fecha/survey)
- ✅ Multi-banda en un solo CSV
- ✅ Plots de debug paso a paso
- ✅ Fácil de leer para clasificación
- ✅ Metadata completo en JSON
- ✅ Summary CSV para análisis masivo
