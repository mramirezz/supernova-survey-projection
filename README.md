# Supernova Survey Projection — ZTF

Pipeline de simulación de detectabilidad de supernovas sobre el Zwicky Transient Facility.
Transforma espectros teóricos de SNe en curvas de luz observadas proyectadas sobre campos ZTF reales,
aplicando cosmología $\Lambda$CDM, extinción por polvo (host + Vía Láctea), fotometría sintética multi-banda
y ruido observacional. Genera datos de entrenamiento para un clasificador fotométrico de transientes.

---

## Quick Start

```bash
conda create -n projection python=3.10
conda activate projection
pip install -r requirements.txt

# Un campo específico (30 simulaciones: 3 tipos × 10 pivotes)
python run_per_field.py --oid ZTF18aaqeasu --seed 42

# Primeros N campos con mínimo de observaciones
python run_per_field.py --n-fields 100 --min-obs 50

# Todos los campos (~67,800 OIDs)
python run_per_field.py

# Verificar resultados
python verify_output.py outputs/per_field/<run_dir>/
```

---

## Pipeline Overview

```
                         run_per_field.py
                              │
              ┌───────────────┼───────────────┐
              │               │               │
           Tipo Ia         Tipo II        Tipo Ibc
           ×10 pivotes     ×10 pivotes    ×10 pivotes
              │               │               │
              └───────┬───────┘               │
                      │                       │
              generate_synthetic_curves()     │
              (run_per_field.py L71)          │
                      │                       │
         ┌────────────┼────────────┐          │
         │            │            │          │
    leer_spec    correct_redeening  Syntetic_photometry_v2
   (utils.py)   (correction.py)    (utils.py)
         │            │            │
         │   host ext → z → MW    │
         │   → D_L dimming        │
         │            │            │
         └────────────┼────────────┘
                      │
              Calibración + Ruido + Normalización M_peak
                      │
              multiband_field_projection()
              (core/multiband_projection.py)
                      │
              Pivote determinístico → Interpolación → Detección
                      │
                  .parquet por campo
```

Cada simulación individual ejecuta los siguientes pasos en secuencia:

1. **Lectura del template espectral** → `leer_spec()` — espectros multi-época $[\lambda, F_\lambda]$
2. **Correcciones observacionales** → `correct_redeening()` — extinción host, redshift, extinción MW, atenuación por $D_L$
3. **Fotometría sintética** → `Syntetic_photometry_v2()` — convolución con filtros g, r, i de ZTF
4. **Calibración fotométrica** — flujo → magnitud AB usando constantes de filtro
5. **Inyección de ruido** — estadística Poisson-like (15%)
6. **Normalización de luminosidad** — shift $\Delta m$ para igualar $M_\text{peak}$ muestreado
7. **Conversión temporal** — fases relativas → MJD absoluto (solo Ibc)
8. **Proyección multi-banda** → `multiband_field_projection()` — pivote determinístico, interpolación en grilla ZTF, evaluación de detección

---

## Arquitectura

```
├── run_per_field.py              # Orquestador: 30 sims/campo
│     ├── scan_templates()            L55   — glob data/{Ia,II,Ibc}/*.dat
│     ├── generate_synthetic_curves() L71   — espectro → curvas multi-banda
│     ├── run_single_simulation()     L174  — 1 curva + proyección
│     ├── run_field()                 L268  — 3 tipos × 10 pivotes
│     └── main()                      L329  — CLI + loop por OID
│
├── verify_output.py              # Verificador de resultados (6 categorías)
├── config.py                     # Configuración global (rutas auto-detectadas, parámetros)
├── config_loader.py              # Loader de configuración
│
├── core/
│   ├── correction.py             # Extinción, redshift, correcciones espectrales
│   │     ├── sample_host_extinction_mixture()   L8    — modelo mixto E(B-V)
│   │     ├── sample_extinction_by_type()        L72   — dispatch por tipo de SN
│   │     ├── sample_cosmological_redshift()     L149  — z volume-weighted
│   │     ├── correct_redeening()                L244  — pipeline de corrección
│   │     └── redden_spectrum_adjusted()                — ley Cardelli+89
│   │
│   ├── multiband_projection.py   # Proyección multi-banda con offset compartido
│   │     └── multiband_field_projection()       — pivote determinístico, detección
│   │
│   ├── utils.py                  # Funciones base
│   │     ├── DL_calculator()          L8    — distancia de luminosidad
│   │     ├── leer_spec()              L60   — lectura de templates .dat
│   │     ├── Syntetic_photometry_v2() L148  — fotometría sintética
│   │     ├── Loess_fit()                     — suavizado LOESS (ALR/rpy2)
│   │     └── maximo_lc()              L198  — MJD del máximo por tipo/nombre
│   │
│   ├── projection.py             # Proyección single-band (base)
│   ├── save_functions.py         # Guardado de resultados
│   └── save_multiband.py         # Guardado multi-banda
│
├── tools/                        # Utilidades del workflow
│   ├── combine_all_batches.py         — combinar outputs en CSV maestro
│   ├── fill_redshift.py               — poblar z en lista de targets
│   ├── find_best_fields.py            — rankear OIDs por cobertura temporal
│   ├── dust_maps.py                   — queries E(B-V)_MW a mapas SFD98 reales
│   ├── check_combined_counts.py       — QA del CSV combinado
│   ├── check_combined_vs_sn_list.py   — auditoría de completitud
│   └── summarize_latest_failures.py   — diagnóstico de fallos por batch
│
├── data/
│   ├── ZTF_observing_log_complete.csv   — 5M obs, 67,827 campos
│   ├── sn_list_to_project.csv           — lista de targets
│   ├── Ia/   (13 templates espectrales)
│   ├── II/   (31 templates espectrales)
│   └── Ibc/  (22 templates espectrales)
│
├── docs/                         # Documentación complementaria (LaTeX, notebooks)
├── app.py                        # Dash app de visualización
├── app_lightcurve_viewer/        # Viewer de curvas de luz
└── legacy/                       # Pipelines obsoletos (batch, single-band)
```

---

## Modelo Físico

### §1. Muestreo cosmológico de redshift

La distribución de redshifts sigue un muestreo proporcional al volumen comóvil diferencial (*volume-weighted*):

$$P(z) \propto \frac{dV_c}{dz} = \frac{(1+z)^2}{E(z)}$$

donde el factor de expansión adimensional es:

$$E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda}$$

**Parámetros cosmológicos** (implementados en `core/correction.py`, `sample_cosmological_redshift()`):

| Parámetro | Valor | Referencia |
|---|---|---|
| $H_0$ | 70 km/s/Mpc | Planck Collaboration (2020) |
| $\Omega_m$ | 0.3 | Planck Collaboration (2020) |
| $\Omega_\Lambda$ | 0.7 | Universo plano: $\Omega_m + \Omega_\Lambda = 1$ |
| $z_\text{min}$ | 0.01 | Evitar peculiar velocities |
| $z_\text{max}$ | 0.5 | Default (configurable por CLI) |

**Método numérico**: Inverse Transform Sampling de la CDF del elemento de volumen, discretizada en grilla de 1000 puntos en $z$. Se genera $u \sim \text{Uniform}(0, 1)$ y se invierte la CDF mediante interpolación lineal (`np.interp`).

**Justificación física**: El número de SNe observables a redshift $z$ es proporcional al volumen de universo disponible en ese shell. El factor $(1+z)^2$ surge de la geometría del espacio comóvil en el modelo FLRW; $E(z)^{-1}$ corrige por el contenido materia/energía oscura del universo. Un muestreo uniforme en $z$ sobrerepresentaría las distancias cortas.

> **Refs**: Hogg (1999, arXiv:astro-ph/9905116) — Distance measures in cosmology; Planck Collaboration (2020, A&A 641, A6).

---

### §2. Extinción del host galaxy

Se utiliza un **modelo de mezcla** que captura la bimodalidad observada en las distribuciones de extinción de hosts de SNe: una fracción significativa de SNe presenta extinción cercana a cero (entornos con poco polvo), mientras que el resto sigue una distribución exponencial en $A_V$ (entornos con polvo).

Implementado en `core/correction.py`, `sample_host_extinction_mixture()`:

$$P(E(B-V)) = f_\text{zero} \cdot \left| \mathcal{N}(0, \sigma_\text{zero}) \right| + (1 - f_\text{zero}) \cdot \frac{R_V}{\tau} \exp\left( -\frac{E(B-V) \cdot R_V}{\tau} \right)$$

donde:
- $f_\text{zero}$: fracción de SNe en entornos sin polvo (componente "limpia")
- $\sigma_\text{zero} = 0.01$ mag: dispersión de la componente sin polvo (half-normal truncada)
- $\tau$: escala exponencial en $A_V$ (mag), convertida a $E(B-V)$ vía $R_V = 3.1$
- $A_V$ truncado en $[0, 3.0]$ mag

**Parámetros por tipo de SN** (hardcoded en `sample_extinction_by_type()`):

| Tipo | $\tau$ (mag) | $f_\text{zero}$ | Justificación física |
|---|---|---|---|
| **Ia** | 0.35 | 0.40 | ~40% en entornos sin polvo (galaxias elípticas, regiones limpias). Phillips+2013, Holwerda+2015, Pantheon+ (Scolnic+2022) |
| **II** | 0.25 | 0.20 | Progenitores RSG 8–25 $M_\odot$, regiones HII moderadas. Hatano+1998, Riello & Patat 2005 |
| **Ibc** | 0.50 | 0.20 | Progenitores WR >25 $M_\odot$, núcleos densos de formación estelar. Taddia+2015, Galbany+2018 |

**Justificación del modelo mixto**: Las distribuciones observadas de $E(B-V)$ en hosts (Jha+2007, Kessler+2009, Brout & Scolnic 2021) muestran un exceso de SNe con $E(B-V) \approx 0$ que una exponencial pura no puede reproducir. La componente "limpia" ($f_\text{zero}$) modela SNe en bordes de regiones HII o "burbujas" despejadas por vientos estelares de los progenitores. La componente exponencial modela SNe embebidas en polvo del medio interestelar local. Las SNe Ibc tienen $\tau$ mayor porque sus progenitores masivos viven y mueren en las regiones más densas de formación estelar.

> **Refs**: Burns+2014 (ApJ 789, 32); Folatelli+2010 (AJ 139, 120); Hatano+1998 (ApJ 502, 177); Riello & Patat 2005 (MNRAS 362, 671); Taddia+2015 (A&A 580, A131); Galbany+2018 (ApJ 855, 107); Hallgren+2023 (ApJ 949, 76); Pantheon+ (Scolnic+2022, ApJ 938, 113).

---

### §3. Extinción de la Vía Láctea

La extinción MW se describe mediante los mapas de polvo de Schlegel, Finkbeiner & Davis (1998, SFD98), consultados vía IRSA (`astroquery.irsa_dust.IrsaDust`) para coordenadas reales del footprint ZTF.

El $E(B-V)_\text{MW}$ varía entre ~0.01 mag (polos galácticos) y >0.15 mag (cerca del plano galáctico $|b| < 15°$).

En el pipeline actual, $E(B-V)_\text{MW}$ se fija a un valor representativo por simulación (actualmente $E(B-V)_\text{MW} = 0.02$ hardcoded en `run_per_field.py`). Las consultas dinámicas a SFD98 están implementadas en `tools/dust_maps.py` para uso futuro.

> **Refs**: Schlegel, Finkbeiner & Davis 1998 (ApJ 500, 525); Schlafly & Finkbeiner 2011 (ApJ 737, 103).

---

### §4. Ley de extinción y orden de correcciones espectrales

La extinción por polvo interestelar se aplica usando la ley paramétrica de Cardelli, Clayton & Mathis (1989), implementada en `core/correction.py` (`redden_spectrum_adjusted()`), basada en código IDL de G. Pignata (2004):

$$A_\lambda = E(B-V) \cdot \left[ a(x) \cdot R_V + b(x) \right]$$

donde $x = 1/\lambda$ (en $\mu\text{m}^{-1}$), $R_V = 3.1$ (estándar para la Vía Láctea), y $a(x)$, $b(x)$ son polinomios definidos por tramos para IR ($0.3 \leq x \leq 1.1$), óptico ($1.1 \leq x \leq 3.3$), UV ($3.3 \leq x \leq 8.0$) e IR lejano ($x \leq 0.3$).

La aplicación al espectro es:

$$F_\text{extinguido}(\lambda) = F_\text{intrínseco}(\lambda) \times 10^{-0.4 \, A_\lambda}$$

#### Orden de aplicación (físicamente crítico)

El pipeline aplica las correcciones en el orden que la luz recorre desde la fuente al telescopio (implementado en `correct_redeening()` con `reverse=True`):

$$\boxed{F_\text{intrínseco} \xrightarrow{\text{1. Host}} F_1 \xrightarrow{\text{2. Redshift}} F_2 \xrightarrow{\text{3. MW}} F_3 \xrightarrow{\text{4. } D_L} F_\text{observado}}$$

**Paso 1 — Extinción del host** (en rest-frame):

$$F_1(\lambda) = F_0(\lambda) \times 10^{-0.4 \, A_\lambda^\text{host}}$$

**Paso 2 — Redshift cosmológico**:

$$\lambda_\text{obs} = \lambda_\text{rest} \times (1+z)$$

$$F_2(\lambda_\text{obs}) = \frac{F_1(\lambda_\text{rest})}{1+z}$$

**Paso 3 — Extinción MW** (en frame observado):

$$F_3(\lambda_\text{obs}) = F_2(\lambda_\text{obs}) \times 10^{-0.4 \, A_{\lambda_\text{obs}}^\text{MW}}$$

**Paso 4 — Atenuación por distancia de luminosidad**:

$$F_\text{obs} = F_3 \times \left( \frac{10^{-5} \text{ Mpc}}{D_L(z)} \right)^2$$

donde la distancia de luminosidad se calcula numéricamente:

$$D_L(z) = \frac{c(1+z)}{H_0} \int_0^z \frac{dz'}{E(z')}$$

> **Nota**: $D_L$ se calcula en `DL_calculator()` (`core/utils.py`) usando $H_0 = 69.6$ km/s/Mpc, $\Omega_m = 0.286$, $\Omega_\Lambda = 0.714$ (Wright 2006 cosmology calculator). Ver [Known Issues](#known-issues) sobre la inconsistencia con los parámetros de `sample_cosmological_redshift()`.

**Justificación del orden**: (1) El host extingue primero porque la SN está embebida en la galaxia. (2) El redshift cosmológico estira las longitudes de onda durante el viaje. (3) La MW es el último obstáculo antes de llegar al telescopio. Invertir este orden produciría extinción MW evaluada en longitudes de onda incorrectas.

> **Refs**: Cardelli, Clayton & Mathis 1989 (ApJ 345, 245); Fitzpatrick 1999 (PASP 111, 63); G. Pignata 2004 (implementación IDL).

---

### §5. Fotometría sintética multi-banda

Para cada filtro $f \in \{g, r, i\}$ de ZTF, el flujo fotométrico se calcula convolucionando el espectro corregido con la respuesta del filtro (implementado en `Syntetic_photometry_v2()`, `core/utils.py`):

$$F_f = \frac{\int F_\lambda(\lambda) \cdot R_f(\lambda) \cdot \lambda \, d\lambda}{\int R_f(\lambda) \cdot \lambda \, d\lambda}$$

donde $R_f(\lambda)$ es la curva de respuesta total del sistema (filtro + detector + óptica + atmósfera), cargada desde archivos de spline precalculados (`spline_g.txt`, `spline_r.txt`, `spline_i.txt`).

La integral se evalúa numéricamente mediante trapecios (`np.trapz`) sobre la grilla de longitudes de onda del espectro, previa regridding de $R_f$ a la misma grilla.

#### Criterio de overlap espectral

Se calcula la fracción del área del filtro cubierta por el espectro disponible:

$$\text{overlap} = \frac{\int_{\lambda_\text{min,spec}}^{\lambda_\text{max,spec}} R_f(\lambda) \, d\lambda}{\int_{\lambda_\text{min,filt}}^{\lambda_\text{max,filt}} R_f(\lambda) \, d\lambda}$$

Se requiere **overlap > 95%** para considerar la fotometría confiable. Épocas con overlap insuficiente (típicamente a $z$ alto, donde el espectro redshifted ya no cubre el rango del filtro) se descartan.

**Conversión a magnitud AB**:

$$m_f = -2.5 \log_{10}\left( \frac{F_f}{C_f} \right)$$

donde $C_f$ son las constantes de calibración fotométrica por filtro (valores numéricos en `core/utils.py`: `cteg`, `cteR`, etc.).

> **Refs**: Bessell & Murphy 2012 (PASP 124, 140); Bellm+2019 (PASP 131, 018002) — ZTF system throughput.

---

### §6. Suavizado LOESS

Se aplica LOcally WEighted Scatterplot Smoothing (Cleveland 1979) para interpolar y suavizar la curva de luz sintética. Implementado en `Loess_fit()` (`core/utils.py`) usando Automated Loess Regression (ALR) vía `rpy2`.

**Parámetros**:

| Condición | $\alpha$ (bandwidth) |
|---|---|
| $N_\text{points} > 40$ | $[0.5, 0.5]$ (dos pasadas, antes y después del corte) |
| $N_\text{points} \leq 40$ | $[0.5]$ (una pasada) |

El parámetro `corte = 30` días define el punto de cambio del ancho de banda.

> **Nota**: En la implementación actual, el output suavizado del LOESS se calcula pero no se reasigna a la curva de luz. Los valores crudos de la fotometría sintética pasan directamente a calibración y ruido. Ver [Known Issues](#known-issues).

> **Refs**: Cleveland 1979 (JASA 74, 829); Cleveland & Devlin 1988 (JASA 83, 596).

---

### §7. Normalización de luminosidad

Se normaliza la magnitud peak del template para que corresponda a una magnitud absoluta muestreada de la distribución observada del tipo correspondiente. Implementado en `generate_synthetic_curves()` (`run_per_field.py` L144–167), configurado en `config.py: LUMINOSITY_CONFIG`.

**Distribuciones de $M_\text{peak}$**:

$$M_\text{peak} \sim \mathcal{N}(\mu_M, \sigma_M), \quad \text{clip} \in [-21.5, -13.0] \text{ mag}$$

| Tipo | $\mu_M$ (mag) | $\sigma_M$ (mag) | Referencia |
|---|---|---|---|
| **Ia** | $-19.3$ | 0.3 | Richardson+2014, Betoule+2014 |
| **II** | $-16.9$ | 1.1 | Richardson+2014, Anderson+2014 |
| **Ibc** | $-17.3$ | 0.9 | Richardson+2014, Drout+2011 |

**Cálculo del shift**:

1. Muestrear $M_\text{peak}$ de la distribución gaussiana truncada
2. Calcular módulo de distancia: $\mu(z) = 5 \log_{10}(D_L(z) \times 10^6) - 5$
3. Magnitud aparente target: $m_\text{target} = \mu(z) + M_\text{peak}$
4. Magnitud peak actual del template en filtro de referencia ($r$): $m_\text{current} = \min(\text{mag}_r)$
5. Shift uniforme: $\Delta m = m_\text{target} - m_\text{current}$
6. Aplicar a **todos los filtros**: $\text{mag}_f \leftarrow \text{mag}_f + \Delta m$

Este método **preserva los colores y la forma** de la curva de luz: solo ajusta el brillo absoluto.

> **Refs**: Richardson+2014 (AJ 147, 118); Betoule+2014 (A&A 568, A22); Anderson+2014 (ApJ 786, 67); Drout+2011 (ApJ 741, 97).

---

### §8. Ruido fotométrico

Se inyecta ruido realista simulando estadística de Poisson fotónica con un componente instrumental (implementado en `generate_synthetic_curves()`, `run_per_field.py` L127–140):

$$F_\text{noisy} = \mathcal{N}\left( F_\text{norm}, \; \sigma = \sqrt{|F_\text{norm}|} \times \epsilon \right) \times F_\text{min}$$

donde:
- $F_\text{norm} = F / F_\text{min}$: flujo normalizado al mínimo (punto más brillante)
- $\epsilon = 0.15$: factor de ruido (15%)
- $F_\text{min}$: flujo del punto más brillante (re-escaleo tras inyectar ruido)

La $\sigma$ proporcional a $\sqrt{F}$ reproduce la estadística de conteo de fotones: las fuentes más brillantes tienen mejor SNR relativo. El factor $\epsilon$ captura ruido instrumental adicional (read noise, sky background, etc.).

Resultado: $m_\text{noisy} = -2.5 \log_{10}(F_\text{noisy})$

---

### §9. Proyección temporal — Pivote determinístico

El corazón de la simulación: colocar la curva de luz sintética sobre la grilla de observaciones reales de un campo ZTF. Implementado en `multiband_field_projection()` (`core/multiband_projection.py`).

#### Grilla observacional

Para cada campo (OID), las observaciones reales definen una grilla temporal irregular de MJDs con magnitudes límite de detección (`maglimit`):

$$\text{grilla} = \{ (t_j, f_j, m_{\text{lim},j}) \}_{j=1}^{N_\text{obs}}$$

donde $t_j$ es MJD, $f_j \in \{g, r, i\}$ es el filtro, y $m_{\text{lim},j}$ es el `diffmaglim` de esa observación.

#### Anclaje temporal

Cada template tiene un "anchor" — el MJD de su máximo de brillo:
- **Ia**: leído de `maximum_Ia.txt` vía `maximo_lc()`
- **II**: calculado como el MJD de mínima magnitud (pico de flujo) en banda de referencia
- **Ibc**: leído de `maximum_Ibc.dat` vía `maximo_lc()`

#### Estrategia de pivote determinístico

La ventana temporal de la grilla $[t_\text{min}, t_\text{max}]$ se divide en $N = 10$ particiones iguales. Para cada partición $i \in \{0, 1, \ldots, 9\}$, el pivote se coloca en el **centro**:

$$\Delta t_\text{part} = \frac{t_\text{max} - t_\text{min}}{N}$$

$$t_\text{pivote}(i) = t_\text{min} + \Delta t_\text{part} \times (i + 0.5)$$

El desplazamiento temporal total es:

$$\delta = t_\text{pivote}(i) - t_\text{anchor}$$

Las fases del template se desplazan:

$$t_\text{template,adjusted} = t_\text{template} + \delta$$

#### Por qué determinístico

La estrategia determinística (vs. random offset) **garantiza cobertura uniforme** de toda la ventana temporal del campo. Esto es esencial para:
1. Evaluar la detectabilidad en función de la época del año (cadencia, gaps estacionales)
2. Reproducibilidad: misma semilla → mismos resultados
3. Cobertura completa: 10 posiciones barren la ventana temporal sin clustering aleatorio

#### Offset compartido multi-banda

El desplazamiento $\delta$ se aplica **simultáneamente a todos los filtros** (g, r, i). Esto refleja la realidad física: una SN aparece en la misma época en todos los filtros. Solo la magnitud/detectabilidad varía por banda.

#### Interpolación y detección

Para cada observación ZTF $(t_j, f_j, m_{\text{lim},j})$:

1. Se verifica que $t_j$ cae dentro del rango temporal del template ajustado
2. Se interpola linealmente la magnitud del modelo: $m_\text{modelo}(t_j) = \text{interp}(t_j, t_\text{adjusted}, \text{mag})$
3. Se evalúa la detección:

$$m_\text{proyectada} = \min(m_\text{modelo}, m_\text{lim})$$

$$\text{detected} = \begin{cases} \text{True} & \text{si } m_\text{modelo} < m_\text{lim} \text{ (más brillante que el límite)} \\ \text{False} & \text{si } m_\text{modelo} \geq m_\text{lim} \text{ (más débil: upper limit)} \end{cases}$$

---

### §10. Criterios de detección y upper limits

En cada época de la grilla:
- **Detección**: la SN es más brillante que el límite del survey → se registra $m_\text{proyectada} = m_\text{modelo}$
- **Upper limit**: la SN es más débil que el límite → se registra $m_\text{proyectada} = m_\text{lim}$, con flag `upperlimit = 'T'`

La **tasa de detección** de una simulación se define como:

$$\eta = \frac{N_\text{detected}}{N_\text{total}} \times 100\%$$

---

## Diseño de la Simulación

### 30 simulaciones por campo

Para cada campo (OID) se generan **30 simulaciones distintas**:

| Dimensión | Valores | Total |
|---|---|---|
| Tipo de SN | Ia, II, Ibc | 3 |
| Posición temporal (pivote) | Particiones 0–9 | ×10 |
| **Total** | | **30** |

### Selección de templates

Los templates se seleccionan **cíclicamente** del catálogo disponible por tipo:

```
template[i] = catálogo_tipo[i % len(catálogo_tipo)]
```

| Tipo | N templates | Origen |
|---|---|---|
| Ia | 13 | Espectros observados de SNe Ia |
| II | 31 | Espectros observados de SNe II |
| Ibc | 22 | Espectros observados de SNe Ibc |

Con 10 simulaciones por tipo, los templates se reciclan al agotarse (para Ia, el 11° pivote reutiliza el 1er template).

### Parámetros por simulación

Cada una de las 30 simulaciones tiene sus propios parámetros estocásticos **independientes**:
- $z$ muestreado volume-weighted (§1)
- $E(B-V)_\text{host}$ muestreado del modelo mixto del tipo (§2)
- $M_\text{peak}$ muestreado de la gaussiana del tipo (§7)
- $E(B-V)_\text{MW} = 0.02$ (fijo)

### Nota sobre fases temporales

Los templates espectrales difieren en su sistema temporal:
- **Ia**: MJD absoluto
- **II**: MJD absoluto (en los archivos `.dat`)
- **Ibc**: fases relativas al máximo (días)

Para los Ibc, el pipeline convierte automáticamente: $t_\text{MJD} = t_\text{fase} + t_\text{max}$ donde $t_\text{max}$ se lee de `maximum_Ibc.dat` (prioridad de filtros: V > R > r > g > i > I > B > U > u > z).

---

## Datos

### Observing log de ZTF

`data/ZTF_observing_log_complete.csv`:

| Estadística | Valor |
|---|---|
| Observaciones totales | 5,035,801 |
| Campos únicos (OIDs) | 67,827 |
| Rango MJD | ~58100–61000 |
| Filtros | g, r, i |
| Columnas | `oid`, `mjd`, `filter`, `maglimit` |

### Templates espectrales

Archivos `.dat` en `data/{Ia,II,Ibc}/`. Cada archivo contiene espectros multi-época con formato:

```
# time: -10.0        ← fase relativa al máximo (días) o MJD absoluto
# SPEC
# WAVE   FLUX
3000    1.234e-15    ← longitud de onda (Å), flujo (erg/s/cm²/Å)
3001    1.456e-15
...
```

---

## Formato de Output

Un archivo `.parquet` por campo en `outputs/per_field/<timestamp>/`:

| Columna | Tipo | Descripción |
|---|---|---|
| `mjd` | float | Modified Julian Date de la observación ZTF |
| `filter` | str | Filtro ZTF (g, r, i) |
| `maglimit` | float | Magnitud límite de detección del survey |
| `magnitud_modelo` | float | Magnitud del modelo interpolada en esa época |
| `magnitud_proyectada` | float | $\min(m_\text{modelo}, m_\text{lim})$ |
| `upperlimit` | str | `'T'` si es upper limit, `'F'` si es detección |
| `detected` | bool | True si es detección real |
| `sn_type` | str | Tipo de SN simulada (Ia, II, Ibc) |
| `template` | str | Nombre del template espectral usado |
| `oid` | str | Identificador del campo ZTF |
| `z` | float | Redshift cosmológico muestreado |
| `ebmv_host` | float | $E(B-V)$ del host muestreado |
| `ebmv_mw` | float | $E(B-V)$ de la Vía Láctea |
| `part_index` | int | Índice de partición temporal (0–9) |
| `n_divisions` | int | Número total de divisiones (10) |
| `offset_used` | int | Offset seleccionado (0 en modo determinístico) |
| `desplazamiento` | float | Desplazamiento temporal total aplicado (días) |

Adicionalmente se genera `run_summary.csv` con estadísticas agregadas y `run_metadata.json` con parámetros del run.

---

## Verificación

`verify_output.py` ejecuta 6 categorías de tests sobre los archivos de output:

| Test | Verifica |
|---|---|
| **structure** | Columnas requeridas, tipos de dato, NaN en columnas críticas |
| **completeness** | 30 sims por campo (3 tipos × 10 particiones), particiones faltantes |
| **physical_ranges** | MJD ∈ [58100, 62000], maglimit ∈ [14, 22], mag ∈ [10, 35], z ∈ [0.001, 1.0], E(B-V) ∈ [0, 2.0] |
| **consistency** | Integridad de upper limits ($m_\text{proy} = m_\text{lim}$), relaciones detected↔maglimit, unicidad de OID |
| **detection_physics** | Tasas de detección razonables (≥10% para z bajo), diferencias de brillo por tipo |
| **distributions** | z sigue distribución volume-weighted, patrones de E(B-V), balance de templates |

Uso:

```bash
# Verificar un run completo
python verify_output.py outputs/per_field/<run_dir>/

# Verificar un campo específico con detalle
python verify_output.py outputs/per_field/<run_dir>/ZTF18aaqeasu.parquet --verbose
```

---

## Known Issues

### 1. Inconsistencia en parámetros cosmológicos

`DL_calculator()` en `core/utils.py` usa $H_0 = 69.6$ km/s/Mpc, $\Omega_m = 0.286$, $\Omega_\Lambda = 0.714$ (Wright 2006 cosmology calculator), mientras que `sample_cosmological_redshift()` en `core/correction.py` usa $H_0 = 70$, $\Omega_m = 0.3$, $\Omega_\Lambda = 0.7$. La diferencia es <1% en $D_L$ para $z < 0.5$, pero debería unificarse en trabajos futuros.

### 2. LOESS suavizado pero no utilizado

`Loess_fit()` se ejecuta pero su output suavizado no se reasigna a la curva de luz. Los valores crudos de fotometría sintética pasan a calibración y ruido. Podría ser intencional (el LOESS era para visualización) o un bug del refactoring.

### 3. Fallos a redshift alto por overlap espectral

Templates con cobertura espectral limitada (típicamente $\lambda \lesssim 9000$ Å) no cubren los filtros rojos (i, r) cuando se aplica redshift alto ($z \gtrsim 0.3$). Estas simulaciones fallan el criterio de overlap > 95% y se descartan. Para el OID de test, ~6/30 simulaciones fallan por esta razón. Esto es una limitación física real, no un bug.

### 4. $E(B-V)_\text{MW}$ hardcoded

El pipeline actual usa $E(B-V)_\text{MW} = 0.02$ fijo para todas las simulaciones, en lugar de consultar los mapas SFD98 para las coordenadas del campo. La infraestructura para consultas reales existe en `tools/dust_maps.py`.

---

## Referencias

### Cosmología
- Hogg, D. W. 1999, arXiv:astro-ph/9905116 — Distance measures in cosmology
- Planck Collaboration 2020, A&A, 641, A6 — Parámetros cosmológicos
- Weinberg, S. 2008, *Cosmology*, Oxford University Press

### Extinción galáctica (MW)
- Schlegel, Finkbeiner & Davis 1998, ApJ, 500, 525 — Mapas SFD98
- Schlafly & Finkbeiner 2011, ApJ, 737, 103 — Recalibración SFD
- Green et al. 2019, ApJ, 887, 93 — Mapas 3D de polvo

### Ley de extinción
- Cardelli, Clayton & Mathis 1989, ApJ, 345, 245 — Ley $R_V = 3.1$
- Fitzpatrick 1999, PASP, 111, 63 — Parametrización de extinción

### Extinción del host — SNe Ia
- Burns et al. 2014, ApJ, 789, 32 — Carnegie SN Project: colores intrínsecos
- Folatelli et al. 2010, AJ, 139, 120 — Carnegie SN Project: análisis fotométrico
- Jha et al. 2007, ApJ, 659, 122 — Distribuciones de $A_V$ en Ia
- Kessler et al. 2009, ApJS, 185, 32 — SDSS-II SN Survey: extinción
- Brout & Scolnic 2021, ApJ, 909, 26 — Pantheon+: análisis de polvo
- Scolnic et al. 2022, ApJ, 938, 113 — Pantheon+: cosmología

### Extinción del host — SNe core-collapse
- Hatano et al. 1998, ApJ, 502, 177 — Extinción en progenitores CC
- Riello & Patat 2005, MNRAS, 362, 671 — Estadística de 200+ SNe CC
- Eldridge et al. 2013, MNRAS, 436, 774 — Progenitores masivos y polvo
- Taddia et al. 2015, A&A, 580, A131 — SNe stripped-envelope
- Galbany et al. 2018, ApJ, 855, 107 — 888 SNe CC del survey CALIFA
- Hallgren et al. 2023, ApJ, 949, 76 — Distribuciones en hosts

### Luminosidad (funciones de luminosidad de SNe)
- Richardson et al. 2014, AJ, 147, 118 — Distribuciones de magnitud absoluta por tipo
- Betoule et al. 2014, A&A, 568, A22 — JLA: SNe Ia como candelas estándar
- Anderson et al. 2014, ApJ, 786, 67 — Curvas de luz de SNe II
- Drout et al. 2011, ApJ, 741, 97 — Curvas de luz de SNe Ibc
- Li et al. 2011, MNRAS, 412, 1441 — Rates y distribuciones de luminosidad

### ZTF
- Bellm et al. 2019, PASP, 131, 018002 — Survey design + system throughput
- Dekany et al. 2020, PASP, 132, 038001 — Cámara ZTF
- Masci et al. 2019, PASP, 131, 018003 — IPAC pipeline
- Graham et al. 2019, PASP, 131, 078001 — Alert system

### Fotometría
- Bessell & Murphy 2012, PASP, 124, 140 — Fotometría sintética AB
- Cleveland 1979, JASA, 74, 829 — LOESS smoothing

### Completitud y rates
- Perrett et al. 2010, A&A, 515, A72 — Completitud de surveys
- Frohmaier et al. 2019, MNRAS, 486, 2308 — SN rates en volumen local
- Vincenzi et al. 2021, MNRAS, 505, 2819 — Simulaciones de surveys

### Software
- Astropy Collaboration 2013, A&A, 558, A33
- Astropy Collaboration 2018, AJ, 156, 123

---

## Documentación complementaria

La carpeta `docs/` contiene material de referencia detallado:

| Archivo | Contenido |
|---|---|
| [`PIPELINE_PROYECCION.tex`](docs/PIPELINE_PROYECCION.tex) | Pipeline paso a paso en LaTeX con ecuaciones y ejemplos numéricos |
| [`README_LATEX.tex`](docs/README_LATEX.tex) | Distribuciones científicas detalladas con derivaciones |
| [`Explicacion_Distribuciones_Extincion.ipynb`](docs/Explicacion_Distribuciones_Extincion.ipynb) | Notebook con derivaciones, código de validación y visualizaciones |
| [`FAILURES_20260125_231012.md`](docs/FAILURES_20260125_231012.md) | Análisis de fallos: reglas de z/extinción por tipo |

Ver [`docs/README.md`](docs/README.md) para descripción detallada de cada archivo.
