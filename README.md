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
3. Muestrea E(B-V)\_host por tipo (modelo mixto)
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

## Requisitos

```bash
conda create -n projection python=3.10
conda activate projection
pip install -r requirements.txt
```

---

## Fundamentos Físicos

### Resumen de parámetros implementados

| Componente | Implementación | Archivo |
|---|---|---|
| Cosmología | ΛCDM: H₀=70, Ωₘ=0.3, ΩΛ=0.7 | `core/correction.py` |
| Redshift | Volume-weighted dV/dz via inverse CDF | `core/correction.py` |
| Extinción host | Modelo mixto: frac\_zero × Gauss(0, σ) + (1-frac\_zero) × Exp(τ) | `core/correction.py` |
| Extinción MW | E(B-V) fijo por línea de visión (mapas SFD98) | `core/correction.py` |
| Ley de extinción | Cardelli, Clayton & Mathis (1989), R\_V = 3.1 | `core/correction.py` |
| Normalización luminosidad | M\_peak gaussiano con clip | `config.py` |
| Ruido fotométrico | Poisson gaussiano sobre flujo | `run_per_field.py` |
| Fotometría sintética | Integración espectro × respuesta de filtro | `core/utils.py` |
| Templates | Espectros reales observados | `data/{Ia,II,Ibc}/` |

### PASO 1: Muestreo de redshift cosmológico

La distribución de redshifts sigue un muestreo volume-weighted:

```
P(z) ∝ dVc/dz ∝ (1+z)² / E(z)
donde E(z) = √[Ωₘ(1+z)³ + ΩΛ]
```

**Parámetros cosmológicos implementados** (en `core/correction.py`):
```
H₀ = 70 km/s/Mpc
Ωₘ = 0.3
ΩΛ = 0.7
```

**Método**: Inverse Transform Sampling de la CDF del elemento de volumen diferencial, con grilla de 1000 puntos en z.

**Justificación**: El número de SNe a redshift z es proporcional al volumen comóvil disponible. El factor (1+z)² surge de la geometría FLRW; E(z)⁻¹ corrige por el contenido materia/energía.

**Referencias**:
- Hogg (1999, arXiv:astro-ph/9905116) — Distance measures in cosmology
- Planck Collaboration (2020, A&A 641, A6) — Parámetros cosmológicos
- Weinberg (2008, *Cosmology*) — Cap. 2: Geometría cosmológica

### PASO 2: Muestreo de extinción del host galaxy

Se usa un **modelo mixto** (implementado en `sample_host_extinction_mixture()`):

```
P(E(B-V)) = frac_zero × |N(0, σ_zero)| + (1 - frac_zero) × Exp(A_V/τ) / R_V
```

donde la componente Exp está truncada en A\_V\_max = 3.0 mag y R\_V = 3.1.

**Parámetros por tipo** (hardcoded en `sample_extinction_by_type()`):

| Tipo | τ (mag) | frac\_zero | σ\_zero | Justificación |
|---|---|---|---|---|
| **Ia** | 0.35 | 0.4 | 0.01 | ~40% en entornos sin polvo (elípticas, regiones limpias). Holwerda+2015, Hallgren+2023, Pantheon+ |
| **II** | 0.25 | 0.2 | 0.01 | Progenitores de masa intermedia (8-25 M☉), regiones HII moderadas. Hatano+1998, Riello & Patat 2005 |
| **Ibc** | 0.50 | 0.2 | 0.01 | Progenitores masivos (>25 M☉), núcleos densos de formación estelar. Taddia+2015, Galbany+2018 |

**Justificación del modelo mixto**:
- **Componente sin polvo** (frac\_zero): SNe en bordes de regiones HII o "burbujas" despejadas por vientos estelares
- **Componente exponencial** (1 - frac\_zero): SNe embebidas en polvo del medio interestelar local
- Las distribuciones observadas de extinción en hosts muestran un exceso de SNe con E(B-V) ≈ 0 (Jha+2007, Kessler+2009, Brout & Scolnic 2021)

**Referencias extinción host**:
- Burns et al. (2014, ApJ 789, 32) — Carnegie SN Project: colores intrínsecos Ia
- Folatelli et al. (2010, AJ 139, 120) — Carnegie SN Project: análisis Ia
- Hatano et al. (1998, ApJ 502, 177) — Extinción en SNe II
- Riello & Patat (2005, MNRAS 362, 671) — Estadística de 200+ SNe core-collapse
- Taddia et al. (2015, A&A 580, A131) — Extinción en SNe stripped-envelope
- Galbany et al. (2018, ApJ 855, 107) — 888 SNe CC del survey CALIFA
- Hallgren et al. (2023, ApJ 949, 76) — Distribuciones de extinción en hosts
- Pantheon+ (Scolnic et al. 2022, ApJ 938, 113)

### PASO 3: Orden de aplicación de correcciones al espectro

El orden es físicamente crítico (implementado en `correct_redeening()`):

```
Espectro intrínseco
    → Extinción host:      F_1 = F_0 × 10^(-0.4 × A_λ^host)
    → Redshift cosmológico: λ_obs = λ_rest × (1+z),  F_2 = F_1/(1+z) × 10^(-0.4 × μ(z))
    → Extinción MW:         F_3 = F_2 × 10^(-0.4 × A_λ^MW)
```

donde μ(z) = 5 log₁₀(D\_L / 10 pc) es el módulo de distancia.

**Justificación del orden**:
1. **Host primero**: La SN está dentro de la galaxia host
2. **Redshift**: Simula el viaje cosmológico (λ se estira, flujo se atenúa)
3. **MW último**: Último obstáculo antes del telescopio

**Ley de extinción**: Cardelli, Clayton & Mathis (1989), implementación basada en código IDL de G. Pignata (2004). Válida para 1250 Å < λ < 33000 Å, con correcciones separadas para UV, óptico e IR.

**Referencias**:
- Cardelli, Clayton & Mathis (1989, ApJ 345, 245) — Ley de extinción R\_V = 3.1
- Fitzpatrick (1999, PASP 111, 63) — Corrección de extinción interestelar
- Schlegel, Finkbeiner & Davis (1998, ApJ 500, 525) — Mapas SFD98

### PASO 4: Normalización de luminosidad

Se aplica a los 3 tipos (configurado en `config.py: LUMINOSITY_CONFIG`). Consiste en un shift uniforme en magnitud para que el peak del template corresponda a un M\_peak muestreado de una distribución gaussiana.

**Distribuciones de M\_peak**:

| Tipo | M\_peak (mean) | σ | Clip |
|---|---|---|---|
| **Ia** | -19.3 | 0.3 | [-21.5, -13.0] |
| **II** | -16.9 | 1.1 | [-21.5, -13.0] |
| **Ibc** | -17.3 | 0.9 | [-21.5, -13.0] |

**Implementación**: Se mide el peak del template en el filtro de referencia (r), se calcula el target aparente como m\_peak = μ(z) + M\_peak, y se aplica Δm = m\_target - m\_current a todos los filtros (preserva colores y forma de la curva).

### PASO 5: Fotometría sintética multi-banda

Para cada filtro (g, r, i de ZTF):

```
m_AB = -2.5 log₁₀(∫ F_λ × R(λ) dλ / ∫ R(λ) dλ) - 48.6
```

donde R(λ) es la respuesta del filtro (splines precalculados en archivos .dat).

### PASO 6: Ruido fotométrico

```
flux_noisy = N(flux_clean, σ = √flux_clean × noise_level)
```

con noise\_level = 0.15 (15%). La σ proporcional a √flujo simula estadística de Poisson fotónica: fuentes brillantes tienen mejor SNR relativa.

### PASO 7: Proyección sobre grilla observacional

La grilla ZTF real del campo (observing log: MJD, filtro, diffmaglim) se divide en N\_divisions = 10 particiones temporales iguales. Para cada partición i, el pivote se coloca al centro:

```
part_size = (MJD_max - MJD_min) / N_divisions
mjd_pivote = MJD_min + part_size × (i + 0.5)
```

En cada época de la grilla se compara mag\_modelo vs maglimit:
- **Detección**: mag\_modelo < maglimit → se registra mag\_proyectada = mag\_modelo
- **Upper limit**: mag\_modelo ≥ maglimit → se registra mag\_proyectada = maglimit

Multi-banda: el offset temporal es **compartido** entre filtros (g, r, i usan el mismo desplazamiento).

### Nota: Unidades temporales

Los templates Ia están en MJD absoluto; los core-collapse (II, Ibc) en fases relativas al máximo. El pipeline convierte automáticamente fases → MJD usando `maximo_lc()` antes de la proyección.

---

## Referencias completas

### Cosmología
- Hogg, D. W. 1999, arXiv:astro-ph/9905116
- Planck Collaboration 2020, A&A, 641, A6
- Weinberg, S. 2008, *Cosmology*, Oxford University Press

### Extinción galáctica (MW)
- Schlegel, Finkbeiner & Davis 1998, ApJ, 500, 525
- Schlafly & Finkbeiner 2011, ApJ, 737, 103
- Green et al. 2019, ApJ, 887, 93

### Extinción del host — SNe Ia
- Burns et al. 2014, ApJ, 789, 32
- Folatelli et al. 2010, AJ, 139, 120
- Pantheon+ (Scolnic et al.) 2022, ApJ, 938, 113
- Brout et al. 2022, ApJ, 938, 110

### Extinción del host — SNe core-collapse
- Hatano et al. 1998, ApJ, 502, 177
- Riello & Patat 2005, MNRAS, 362, 671
- Eldridge et al. 2013, MNRAS, 436, 774
- Taddia et al. 2015, A&A, 580, A131
- Galbany et al. 2018, ApJ, 855, 107
- Hallgren et al. 2023, ApJ, 949, 76

### Leyes de extinción
- Cardelli, Clayton & Mathis 1989, ApJ, 345, 245
- Fitzpatrick 1999, PASP, 111, 63

### ZTF
- Bellm et al. 2019, PASP, 131, 018002
- Dekany et al. 2020, PASP, 132, 038001
- Masci et al. 2019, PASP, 131, 018003
- Graham et al. 2019, PASP, 131, 078001

### Completitud y rates
- Perrett et al. 2010, A&A, 515, A72
- Frohmaier et al. 2019, MNRAS, 486, 2308
- Vincenzi et al. 2021, MNRAS, 505, 2819

### Software
- Astropy Collaboration 2013, A&A, 558, A33
- Astropy Collaboration 2018, AJ, 156, 123

