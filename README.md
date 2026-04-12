# Supernova Survey Projection — ZTF

Simula la detectabilidad de supernovas proyectando templates espectrales reales sobre campos ZTF observados. Genera datos de entrenamiento para un clasificador fotométrico.

## Uso

```bash
conda activate projection

# Un campo específico
python run_per_field.py --oid ZTF18aaqeasu --seed 42

# Primeros N campos con mínimo de observaciones
python run_per_field.py --n-fields 100 --min-obs 50

# Todos los campos (~67,800 OIDs)
python run_per_field.py

# Verificar resultados
python verify_output.py outputs/per_field/<run_dir>/
python verify_output.py outputs/per_field/<run_dir>/ZTF18aaqeasu.parquet --verbose
```

## Pipeline

Para cada campo (OID) genera **30 simulaciones**: 3 tipos (Ia, II, Ibc) × 10 posiciones de pivote determinísticas.

Cada simulación:
1. Elige template espectral (cíclico entre disponibles por tipo)
2. Muestrea redshift volume-weighted (dV/dz)
3. Muestrea E(B-V)\_host por tipo (exponencial, parámetros en `core/correction.py`)
4. Normaliza luminosidad al M\_peak del tipo (distribución gaussiana con clip)
5. Genera fotometría sintética multi-banda (g, r, i simultáneamente)
6. Proyecta sobre la grilla real del campo con pivote determinístico (grilla ÷ 10, centro de cada partición)

Output: un `.parquet` por campo en `outputs/per_field/<timestamp>/`

## Estructura

```
├── run_per_field.py        # Runner principal (30 sims/campo)
├── verify_output.py        # Verificador de resultados
├── config.py               # Configuración global (rutas, parámetros)
├── config_loader.py        # Loader de configuración
├── app.py                  # Dash app de visualización
├── core/
│   ├── multiband_projection.py   # Proyección multi-banda con offset compartido
│   ├── projection.py             # Proyección single-band (base)
│   ├── correction.py             # Extinción, redshift, correcciones
│   ├── utils.py                  # leer_spec, fotometría sintética, LOESS
│   ├── save_functions.py         # Guardado de resultados
│   └── save_multiband.py         # Guardado multi-banda
├── tools/                  # Utilidades del workflow
│   ├── combine_all_batches.py         # Combinar outputs en CSV maestro
│   ├── fill_redshift.py               # Poblar z en lista de targets
│   ├── find_best_fields.py            # Rankear OIDs por cobertura
│   ├── dust_maps.py                   # Queries SFD98 reales
│   ├── check_combined_counts.py       # QA del CSV combinado
│   ├── check_combined_vs_sn_list.py   # Auditoría completeness
│   └── summarize_latest_failures.py   # Diagnóstico de fallos
├── data/
│   ├── ZTF_observing_log_complete.csv  # 5M obs, 67,827 campos
│   ├── sn_list_to_project.csv          # Lista de targets
│   ├── Ia/   (13 templates)
│   ├── II/   (31 templates)
│   └── Ibc/  (22 templates)
├── app_lightcurve_viewer/  # Viewer de curvas de luz
└── legacy/                 # Pipelines obsoletos (batch, single-band)
```

## Física

| Componente | Implementación |
|---|---|
| Cosmología | ΛCDM: H₀=70, Ωₘ=0.3, ΩΛ=0.7 |
| Redshift | Volume-weighted dV/dz via inverse CDF |
| Extinción host | Exponencial por tipo (τ: Ia=0.35, II=0.25, Ibc=0.50) |
| Extinción MW | E(B-V) fijo por línea de visión |
| Normalización | M\_peak gaussiano: Ia=-19.3±0.3, II=-16.9±1.1, Ibc=-17.3±0.9 |
| Ruido | Poisson gaussiano sobre flujo |
| Templates | Espectros reales: 13 Ia + 31 II + 22 Ibc |

## Requisitos

```bash
conda create -n projection python=3.10
conda activate projection
pip install -r requirements.txt
```


