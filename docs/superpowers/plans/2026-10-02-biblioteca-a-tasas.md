# De la biblioteca congelada a las tasas: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Llevar las 78 series congeladas hasta tres productos de tesis. Primero, el clasificador validado en ZTF con la biblioteca final. Segundo, la clasificación de las SNe de SUDARE. Tercero, las tasas volumétricas de SUDARE con nuestra clasificación. Todo esto sin repetir corridas pesadas.

**Architecture:** La biblioteca se convierte una sola vez a matrices binarias locales (`~/thesis_store`). Un motor fotométrico vectorizado nuevo aplica host, redshift con dilatación temporal, Vía Láctea y distancia. Integra en curvas de filtro reales (ZTF y OmegaCAM) con punto cero AB.

Un runner paralelo y determinista escribe en disco local:
- la fotometría en el mismo esquema por campo que ya lee `run_parquet.py`;
- una tabla `_sims` con todas las inyecciones, incluidas las no detectadas, que es la base de la eficiencia y del control time.

La extracción de features y la receta del clasificador se reutilizan, con dos arreglos verificados. Las tasas son un módulo pequeño encima de la tabla `_sims`.

**Tech Stack:** Python 3.10 del env `/opt/anaconda3/envs/projection` con numpy 2.2.6, pandas, pyarrow, astropy 6.1.7, scikit-learn y emcee 3.1.6. Multiprocessing de la librería estándar. No hay dependencias nuevas, tampoco pytest: los tests son scripts con `assert` que se corren con `python`.

**Spec:**
- `proyeccion/HANDOFF_20260930_siguiente_paso.md`
- `paper2_ZTF/obslogsudare/PLAN_prueba_clasificador_sudare.md`
- Las secciones 0 a 3 de este documento, que fijan hallazgos, decisiones y contratos de datos.

## Global Constraints

- Python: `PY=/opt/anaconda3/envs/projection/bin/python`. Ningún paquete nuevo.
- La biblioteca `paper2_ZTF/series_aprobadas/` es de solo lectura. Nunca se recalcula ni se sobrescribe una serie congelada o un loess aprobado.
- Una sola corrida pesada a la vez, porque el Mac tiene 16 GB. Avisar antes de cualquier cosa que dure más de 10 minutos.
- **Ninguna corrida masiva sin orden explícita de Mauricio.** Cada una tiene su paso "ORDEN".
- Las corridas escriben en disco local (`~/thesis_store`, `~/thesis_runs`), nunca directo en Google Drive.
- El espejo en Drive es `cp`, verificación md5 y recién ahí listo. Nunca `mv` seguido de `rm -rf`.
- Antes de usar un archivo de Drive, verificar que esté hidratado: leerlo entero, con md5 estable en dos lecturas.
- Git:
  - Los commits locales de cada tarea quedan autorizados al aprobar este plan.
  - Push solo con orden.
  - **Jamás `Co-Authored-By`.**
  - Las corridas pesadas solo desde un commit limpio, y el manifiesto guarda el hash.
- Citas: solo bibcodes verificados en ADS o arXiv. Nunca inventar un valor de literatura.
- Tesis: inglés, registro A&A, sin punto y coma y sin em-dash en la prosa.
- Figuras: estilo publicable, en inglés, vectoriales, copiadas a `phd_thesis/tesis_template/figuras/`.
- Con Mauricio se habla en español chileno, directo. Los pasos se nombran con su descripción, no solo con el código del step.

## Review Focus

1. **Archivo de Drive a medio hidratar durante la conversión.** Se espera que la conversión aborte con un error claro y no deje una plantilla truncada. El test va en la Tarea 2 (`test_stable_md5_detects_change`).
2. **Plantilla con grilla de λ distinta entre bloques, o con flujos no finitos.** Se espera un error explícito con el nombre del archivo, no NaN silenciosos en la fotometría. El test va en la Tarea 2 (`test_parse_rejects_bad_grid`, `test_parse_rejects_nan`).
3. **Inyección sin cobertura de banda o sin épocas en la ventana**, por ejemplo z alto en SUDARE, un campo con una sola banda o una ventana corta. Se espera que la simulación igual quede en `_sims` con su `status`, porque si se pierde sesga la eficiencia. El test va en la Tarea 7 (`test_no_coverage_is_logged`).
4. **Corrida interrumpida a mitad de un campo.** Se espera que al relanzar no queden archivos a medias y que solo se rehagan las unidades incompletas. El test va en la Tarea 7 (`test_resume_skips_complete_units`).
5. **Relanzar sobre una carpeta con otra configuración, otra semilla o el repo con cambios sin commit.** Se espera que el runner se niegue en vez de mezclar resultados. El test va en la Tarea 7 (`test_refuses_config_mismatch`).

---

## 0. Lo que encontré revisando el código (cambia el orden de todo)

| # | Hallazgo | Dónde | Consecuencia |
|---|---|---|---|
| H1 | **No hay dilatación temporal.** Las fases del template nunca se multiplican por (1+z). | `core/correction.py:402`, `run_per_field.py`, `core/multiband_projection.py` | En ZTF (z ≤ 0.1) estira menos de 10 %. En SUDARE (z ~ 0.4 a 0.8) las curvas salen entre 1.4 y 1.8 veces más cortas de lo real. Con el código actual SUDARE es inválido. |
| H2 | **Las Ibc se anclan en su primera época, no en el máximo.** Las fechas ya están en MJD y el código les suma otra vez `maximo_lc`. | `run_per_field.py:325-330`, `multiband_projection.py:188-193` | En run_1000_v2 todas las Ibc quedaron con el pivote mal puesto. Además `maximo_lc` falla en IIb, IIn y SLSN-I. |
| H3 | **Las curvas de filtro son las SDSS g'r'i', no las de ZTF**, y no hay curvas de OmegaCAM. | `config.RESPONSE_FOLDER` (`Practica2/Splines_eachfilter_2`) | La fotometría sintética no está en el sistema de los datos reales. |
| H4 | **El clasificador cruza tipos al unir r con g.** El merge es por (oid, part_index) sin `sn_type`. Las 13 063 filas pasan a 26 082, y el color sale de la amplitud g de otro tipo. | `opt_clasificador/05_faseC_run1000_v2.py:22` y todos los `ab_*.py` | Todos los números del clasificador, incluido el 0.75, cargan este error. Hay que recalcularlos. |
| H5 | **La normalización de luminosidad cancela la extinción.** El brillo se fija sobre el r observado ya enrojecido, así que el polvo no atenúa. | `run_per_field.py:296-305` | La recalibración del polvo de las Ia del 6 de agosto (τ 0.65) no puede haber movido el M observado. Las Ia sintéticas quedan como clones en brillo. |

Además:
- La proyección no usa las 78: `TEMPLATE_DIRS` apunta a `data/*_new` del 15 de agosto.
- La lectura del texto de 70 MB por simulación es el cuello de botella de la proyección (unos 3.5 s por sim).
- La extracción de features cuesta unos 50 s por banda y por núcleo (emcee con 100 walkers y 5000 pasos). La corrida v2 tardó 43.5 h.
- Las features reales de ZTF son de enero, de antes de la regla "double bump", y salieron por otro camino (`main.py`, sin cascada de reintentos). Hay que recalcularlas con el mismo código que las sintéticas.
- `dm15_Ia.json` no tiene 5 de las 15 Ia congeladas.
- El capítulo 3 de la tesis describe un Random Forest y un ruido 0.15√F que ya no son los del código.

## 1. Decisiones que necesito de ti antes de empezar

Cada una dice qué tarea bloquea. La recomendación va primero.

- **D1. Clases.**
  - La propuesta es proyectar Ia, II, IIb, IIn e Ibc (con las Ic-BL). Para el clasificador de 3 clases, IIb e IIn van a II.
  - En ZTF, la propuesta es proyectar solo Ia, II, IIb e Ibc, porque la muestra real de ZTF no trae IIn.
  - Mezcla de II en el entrenamiento ZTF: 8 sims II y 2 IIb por campo, para que la clase II no quede con el doble de peso que Ia e Ibc.
  - SLSN-I quedan fuera.
  - *Bloquea las Tareas 3 y 7.*
- **D2. LOESS.** La propuesta es apagarlo para las 78. Son series diarias ya mangladas a la fotometría (0.012 mag de mediana), y volver a suavizarlas solo agrega una dependencia de R. *Bloquea la Tarea 4.*
- **D3. Filtros.** La propuesta es usar las curvas de ZTF g, r, i y OmegaCAM g, r, i del SVO, con punto cero AB calculado con la misma integral de fotones. *Bloquea la Tarea 1.*
- **D4. Luminosidad.**
  - La propuesta es sortear el M intrínseco (las distribuciones de literatura ya están corregidas por polvo). Se aplica al template en reposo antes del host, así que el polvo sí atenúa.
  - Junto con eso, devolver el polvo de las Ia a su valor de literatura, τ 0.35 y frac_zero 0.40 (Holwerda+2015), y verificar la dispersión del M observado en el piloto de la Tarea 8.
  - *Bloquea las Tareas 6 y 8.*
- **D5. Dilatación temporal y ancla.** Se aplica (1+z) a las fases. El ancla es el máximo en r de reposo del propio template, para todas las clases, sin tablas externas de máximos. No es una decisión, es la corrección de H1 y H2. Solo te lo aviso. *Tarea 4.*
- **D6. MCMC más barato.** La propuesta es aceptar 32 walkers, 1500 pasos y 300 de burn-in si el benchmark de la Tarea 10 muestra acuerdo con la configuración actual. Si no, se prueba con 50, 3000 y 500. Las features reales se recalculan siempre con la configuración elegida. *Bloquea las Tareas 11, 12 y 20.*
- **D7. Bandas de SUDARE para las features.**
  - La propuesta es r como banda principal, porque es la de búsqueda y la que más épocas tiene, e i como segunda, con color r−i.
  - Sobre z ≈ 0.33 la banda g de OmegaCAM cae bajo 3000 Å en reposo, fuera de la biblioteca. La Tarea 16 lo mide antes de cerrar la decisión.
  - *Bloquea las Tareas 19, 20 y 21.*
- **D8. Una sola corrida volumétrica para SUDARE.** La propuesta es usar la misma corrida para entrenar y para la eficiencia. En SUDARE la muestra real es la propia survey, así que la selección la reproduce la corrida volumétrica sin tener que emparejar z. *Bloquea la Tarea 18.*
- **D9. Detección en SUDARE.**
  - "Encontrada" se define como en el paper: probabilidad DE(m) de la ec. 1 en la banda r de búsqueda.
  - "Punto detectado" para las features se define como SNR ≥ 3 sobre el flujo con ruido, con la misma regla en sintéticas y en reales.
  - El resto de las épocas son límites a m50.
  - *Bloquea las Tareas 18 y 19.*
- **D10. Las tres congeladas con defecto visible** (2016coi, salto de 0.5 mag en u en +0 d; 2012fr, U y u dentadas de 0.2 a 0.4 mag; 2007gr, espectro de +106 d con continuo de host).
  - La propuesta es no reabrirlas y proyectarlas igual.
  - ZTF y SUDARE no observan en u ni en U, y la banda g de OmegaCAM cubre U en reposo solo a z bajo.
  - Lo de 2007gr pesa solo a +106 d, donde la SN ya está bajo el límite de SUDARE para z > 0.1.
  - Si prefieres sacarlas, basta con borrarlas de `catalog.csv` antes de la Tarea 7.
  - *Bloquea la Tarea 7.*

## 2. Etapas, cómputo y puertas

Las estimaciones marcadas con * se miden en el piloto anterior antes de lanzar.

| Etapa | Qué corre | Tamaño | Tiempo | Puerta antes de la siguiente |
|---|---|---|---|---|
| A. Biblioteca a almacén (T2) | 78 .dat → npy | 2.4 GB leídos de Drive, una vez | 30–60 min | Los 78 cargados, sin errores |
| B. Motor (T4) | prueba dorada contra el código viejo | 1 template | segundos | Δmag mediana < 0.003 |
| C. Piloto ZTF (T8) | 50 campos × 30 sims (Ia 10, II 8, IIb 2, Ibc 10) | 1 500 sims | < 5 min* | **G1: tú apruebas las figuras en el atlas** |
| D. Proyección ZTF (T9) | 1000 campos (los mismos de v2) | 30 000 sims | 30–60 min* | Reporte de la corrida |
| E. Benchmark MCMC (T10) | 100 tareas × 2 configuraciones | | ~30 min | **G2: tú eliges la configuración** |
| F. Features reales ZTF (T11) | ~1 450 SNe × g, r | ~2 900 ajustes | 1–5 h* | Faltantes explicados |
| G. Features ZTF (T12) | sims con ≥ 7 detecciones | ~28 000 ajustes | ~8 h con la configuración reducida, 49 h con la actual* | Tabla de errores |
| H. Clasificador ZTF (T13) | receta arreglada | minutos | minutos | **G3: validación primero, final una sola vez** |
| I. SUDARE: log, ruido, cobertura (T14–T17) | calibraciones | minutos | minutos | **G4: tú cierras D7 y el valor de las IIn** |
| J. Proyección SUDARE (T18) | 7 campo-temporadas × 5 clases × 3000 | 105 000 sims | ~1 h* | Piloto de 100 por campo aprobado |
| K. Features SUDARE (T20) | submuestra estratificada | ~20 000 sims × r, i | ~10 h reducido* | |
| L. Clasificación SUDARE (T21) | con z y sin z | minutos | minutos | **G5: resultados contra la herramienta de SUDARE** |
| M. Tasas (T22–T24) | control time, volumen, estimador | minutos | minutos | Se reproduce la Tabla 5 de SUDARE I con su clasificación |

Las horas de cómputo están en las etapas G y K. Todo lo anterior a ellas existe para que no haya que repetirlas.

## 3. Contratos de datos

**Almacén de plantillas.** `~/thesis_store/templates/<clase>/<SN>/` contiene:
- `time.npy`: MJD de reposo, float64;
- `wave.npy`: Å de reposo, float64;
- `flux.npy`: float32 de forma [época, λ], a 10 pc, en erg s⁻¹ cm⁻² Å⁻¹, leído con mmap, así que los procesos comparten memoria;
- `meta.json`: sn, clase, md5_src, n_epochs, t_first, t_last, wmin, wmax, n_neg, y además t_peak, peak_at_edge, M_ref, ref_band, dm15_B y clf_class.

`~/thesis_store/catalog.csv` tiene una fila por template con esos mismos campos y `store_path`.

**Logs de survey.**
- `~/thesis_store/ztf_obslog_best.parquet`: field (oid), mjd, band, maglim. Se queda la mejor época por día, campo y banda, como el código viejo.
- `~/thesis_store/sudare_obslog.parquet`: field (cdfs1–4, cosmos1–3), mjd, band, maglim (= m50), seeing, is_ref.

**Corrida** (`~/thesis_runs/<nombre>/`):
- `run_manifest.json`: configuración, semilla, hash de la configuración, commit de git y si había cambios sin commit.
- `<field>__<k0>.parquet`: la fotometría, **con el mismo esquema que lee `ztf_literature_features/parquet_reader.py`**:
  - identificación: oid, part_index (int32), sn_type, template, sim_id (int64);
  - fotometría: mjd (f8), filter ('g','r','i'), maglimit (f4), magnitud_modelo (f4), magnitud_proyectada (f4), magerr (f4, NaN en los límites), upperlimit ('T'/'F'), detected (bool), found (bool);
  - parámetros: z, ebmv_host, rv_host, ebmv_mw, m_peak_abs, dm15_used.
- `_sims/<field>__<k0>.parquet` y `_sims_all.parquet`: **una fila por inyección, con o sin detección.** Columnas: sim_id, field, part_index, sn_type, clf_class, template, z, ebmv_host, rv_host, ebmv_mw, m_peak_abs, dm15_used, t_anchor, status (ok / no_coverage / no_epochs), n_rows, n_det_g, n_det_r, n_det_i, found.
- Los archivos que empiezan con `_` los ignora el indexador de features.
- Escritura atómica: se escribe un `.tmp` y se hace `os.replace`.
- Una unidad está completa cuando existe su archivo en `_sims`. Por eso se escribe al final de la unidad.

**Bandas que se guardan.** Se proyectan siempre g, r, i.
- Features de ZTF: g y r, como la muestra real.
- Features de SUDARE: r e i (D7).

**Features.** `features/features.csv` de `run_parquet.py`, sin cambios de formato. La clave es (oid, part_index, sn_type, filter_band).

**Espejo.** Al cerrar cada etapa: `cp -R` de la carpeta a `paper2_ZTF/runs/<nombre>/`, comparación de md5 de todos los archivos y recién ahí se da por guardada.

## 4. Mapa de archivos

Repo `proyeccion` (rama nueva `pipeline78`):

| Archivo | Responsabilidad |
|---|---|
| `pipeline78/paths.py` | Rutas, con override por variable de entorno para tests |
| `pipeline78/bands.py` | Curvas, punto cero AB, fotometría sintética vectorizada |
| `pipeline78/store.py` | Parser de .dat, almacén npy, md5 estable |
| `pipeline78/catalog.py` | t_peak, M_ref, dm15 y mapeo de clases |
| `pipeline78/engine.py` | Espectro en reposo → magnitudes observadas por banda y tiempos con (1+z) |
| `pipeline78/sampling.py` | z, E(B−V) y M con un rng por simulación, y E(B−V) de la Vía Láctea por campo |
| `pipeline78/survey.py` | Logs de ZTF y SUDARE → esquema común |
| `pipeline78/project.py` | Ancla y cadencia: ruido, detección, límites |
| `pipeline78/runcfg.py` | Configuraciones con nombre de cada corrida |
| `pipeline78/run.py` | Runner paralelo, determinista, reanudable |
| `pipeline78/pilot_report.py` | Página de diagnóstico del piloto para el atlas |
| `pipeline78/compare_features.py` | Benchmark de configuraciones MCMC |
| `pipeline78/real_to_parquet.py` | Fotometría real ZTF y SUDARE → esquema de proyección |
| `pipeline78/recipe.py` | Receta del clasificador con el merge arreglado |
| `pipeline78/sudare_calib.py` | Ruido, cobertura por z y fronteras de temporada |
| `pipeline78/rates.py` | Control time, volumen y estimador |
| `pipeline78/rates_closure.py` | Prueba de cierre con catálogos simulados |
| `pipeline78/subsample.py` | Submuestra estratificada para las features de SUDARE |
| `tests/p78_fakes.py` | Template y log falsos para los tests |
| `tests/test_p78_*.py` | Un archivo de tests por módulo |

En `ztf_literature_features/config.py` solo cambian las líneas de `MCMC_CONFIG` (Tarea 10).

---

## 5. Tareas

### Parte A: base

### Task 0: Preparar el terreno

**Files:**
- Create: `pipeline78/__init__.py`, `pipeline78/paths.py`, `tests/test_p78_paths.py`

**Interfaces:**
- Produces: las constantes `GD, PHD, REPO, DATA, LIB, SUDARE_DIR, LEGACY_RESP, ZLF, OC, STORE, RUNS, FILTERS` (todas `pathlib.Path`). `STORE` y `RUNS` se pueden cambiar con `P78_STORE` y `P78_RUNS`.

- [x] **Step 1: Decisiones.** Confirmar D1 a D9 con Mauricio y anotar las respuestas al final de este archivo, en la sección "Decisiones tomadas".

- [x] **Step 2: Git.** El repo está en `feature/parallelization` con cambios sin commit del otro chat (config.py, core/*, run_per_field.py, HANDOFF*.md). Con el ok de Mauricio:

```bash
REPO="/Users/pulsar/Library/CloudStorage/GoogleDrive-mramirez.valenzuela@gmail.com/Mi unidad/Work/Universidad/Phd/paper2_ZTF/Codes/proyeccion"
cd "$REPO" && git status --short
git add -u && git commit -m "WIP previo a pipeline78: cambios pendientes de config, correction, multiband y runner"
git checkout -b pipeline78
mkdir -p ~/thesis_store ~/thesis_runs
```

- [ ] **Step 3: Escribir el test de rutas**

```python
# tests/test_p78_paths.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline78 import paths

def test_paths_exist():
    for p in (paths.REPO, paths.DATA, paths.LIB, paths.SUDARE_DIR, paths.LEGACY_RESP, paths.ZLF, paths.OC):
        assert p.exists(), p
    assert (paths.LIB / "MANIFIESTO.csv").exists()

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 4: Correrlo y ver que falla**

Run: `cd "$REPO" && $PY tests/test_p78_paths.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78'`

- [ ] **Step 5: Implementar**

```python
# pipeline78/__init__.py
```

```python
# pipeline78/paths.py
"""Rutas del pipeline v78. STORE y RUNS son locales (fuera de Drive) a proposito."""
import os
from pathlib import Path

GD = Path.home() / "Library/CloudStorage/GoogleDrive-mramirez.valenzuela@gmail.com/Mi unidad"
PHD = GD / "Work/Universidad/Phd"
REPO = PHD / "paper2_ZTF/Codes/proyeccion"
DATA = REPO / "data"
FILTERS = DATA / "filters"
LIB = PHD / "paper2_ZTF/series_aprobadas"
SUDARE_DIR = PHD / "paper2_ZTF/obslogsudare"
LEGACY_RESP = PHD / "Practica2/Splines_eachfilter_2"
ZLF = PHD / "paper2_ZTF/Codes/feature_extraction/ztf_literature_features"
OC = PHD / "paper2_ZTF/Codes/feature_extraction/opt_clasificador"
STORE = Path(os.environ.get("P78_STORE", Path.home() / "thesis_store"))
RUNS = Path(os.environ.get("P78_RUNS", Path.home() / "thesis_runs"))
```

- [ ] **Step 6: Correrlo y ver que pasa**

Run: `$PY tests/test_p78_paths.py`
Expected: `ok test_paths_exist`

- [ ] **Step 7: Commit**

```bash
git add pipeline78/__init__.py pipeline78/paths.py tests/test_p78_paths.py docs/superpowers/plans/2026-10-02-biblioteca-a-tasas.md
git commit -m "pipeline78: rutas y plan"
```

### Task 1: Bandas, punto cero AB y fotometría sintética

**Files:**
- Create: `data/filters/{ZTF_g,ZTF_r,ZTF_i,OmegaCAM_g,OmegaCAM_r,OmegaCAM_i}.dat`, `pipeline78/bands.py`, `tests/test_p78_bands.py`

**Interfaces:**
- Consumes: `paths.FILTERS`, `paths.LEGACY_RESP`, `core.utils.cte*`
- Produces:
  - `Band(name: str, wave: ndarray, resp: ndarray, f0: float)`
  - `synphot(wave: ndarray[n_w], flux2d: ndarray[n_ep, n_w], band) -> (ndarray[n_ep], float coverage)`
  - `survey_bands(survey: str, names=("g","r","i")) -> list[Band]`
  - `rest_bands() -> dict[str, Band]` con las claves `B_rest, V_rest, R_rest, r_rest`
  - `legacy_band(name) -> Band`
  - las constantes `C_AA`, `COVERAGE_MIN = 0.95`

- [ ] **Step 1: Bajar las curvas del SVO**

```bash
mkdir -p "$REPO/data/filters"
for id in Palomar/ZTF.g Palomar/ZTF.r Palomar/ZTF.i Paranal/OmegaCAM.g_SDSS Paranal/OmegaCAM.r_SDSS Paranal/OmegaCAM.i_SDSS; do
  n=$(echo "$id" | sed 's|.*/||; s|\.|_|; s|_SDSS||')
  /usr/bin/curl -s "http://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id=$id" -o "$REPO/data/filters/$n.dat"
done
$PY -c "
import numpy as np, glob
for f in sorted(glob.glob('$REPO/data/filters/*.dat')):
    d=np.loadtxt(f); print(f.split('/')[-1], d.shape, round(d[:,0].min()), round(d[:,0].max()), 'pico', round(d[d[:,1].argmax(),0]))"
```

Expected: seis archivos de dos columnas. ZTF g va de ~3700 a 5600 Å, r de ~5500 a 7400 y i de ~6800 a 9000. OmegaCAM g va de ~3900 a 5600, r de ~5300 a 7100 e i de ~6700 a 8600. Si alguno trae HTML en vez de números, el id cambió: buscarlo en `http://svo2.cab.inta-csic.es/theory/fps/` y repetir.

- [ ] **Step 2: Escribir los tests**

```python
# tests/test_p78_bands.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline78.bands import C_AA, synphot, survey_bands, legacy_band, rest_bands

def test_flat_ab_spectrum_is_zero_mag():
    w = np.arange(2500.0, 11000.0, 1.0)
    f = (3631e-23 * C_AA / w**2)[None, :]
    for b in survey_bands("ZTF") + survey_bands("SUDARE"):
        F, cov = synphot(w, f, b)
        assert cov > 0.999, b.name
        assert abs(-2.5 * np.log10(F[0] / b.f0)) < 1e-3, b.name

def test_coverage_drops_when_blue_edge_missing():
    g = survey_bands("ZTF", ("g",))[0]
    w = np.arange(4600.0, 9000.0, 1.0)
    _, cov = synphot(w, np.ones((1, w.size)), g)
    assert cov < 0.95

def test_legacy_band_keeps_historic_zero_point():
    from core.utils import cter
    assert legacy_band("r").f0 == cter

def test_rest_bands_present():
    rb = rest_bands()
    assert set(rb) == {"B_rest", "V_rest", "R_rest", "r_rest"}

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 3: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_bands.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78.bands'`

- [ ] **Step 4: Implementar**

```python
# pipeline78/bands.py
"""Bandas: curvas de transmision, punto cero AB y fotometria sintetica por fotones.

Convencion del pipeline (apendice C de la tesis):
    F = int F_lambda R lambda dlambda / int R lambda dlambda   (denominador sobre la banda completa)
    m = -2.5 log10(F / F0)
Con F0 calculado con la MISMA integral sobre el espectro AB (3631 Jy), m es una magnitud AB.
"""
from dataclasses import dataclass
import numpy as np
from pipeline78.paths import FILTERS, LEGACY_RESP

C_AA = 2.99792458e18          # c en A/s
COVERAGE_MIN = 0.95           # misma regla que la proyeccion historica (porcentaje > 0.95)
TRAPZ = np.trapezoid
SURVEY_FILES = {"ZTF": "ZTF_{}.dat", "SUDARE": "OmegaCAM_{}.dat"}


@dataclass(frozen=True, eq=False)
class Band:
    name: str
    wave: np.ndarray
    resp: np.ndarray
    f0: float


def read_curve(path):
    d = np.loadtxt(path, comments="#")
    d = d[np.argsort(d[:, 0])]
    return d[:, 0].astype(float), np.clip(d[:, 1].astype(float), 0.0, None)


def ab_f0(wave, resp):
    f_ab = 3631e-23 * C_AA / wave**2
    return float(TRAPZ(f_ab * resp * wave, wave) / TRAPZ(resp * wave, wave))


def make_band(name, path, f0=None):
    w, r = read_curve(path)
    return Band(name, w, r, ab_f0(w, r) if f0 is None else float(f0))


def synphot(wave, flux2d, band):
    """Flujo sintetico por epoca y fraccion de la banda cubierta por [wave[0], wave[-1]]."""
    total = TRAPZ(band.resp, band.wave)
    inside = (band.wave >= wave[0]) & (band.wave <= wave[-1])
    cov = float(TRAPZ(band.resp[inside], band.wave[inside]) / total) if inside.sum() > 1 else 0.0
    r = np.interp(wave, band.wave, band.resp, left=0.0, right=0.0)
    num = TRAPZ(np.asarray(flux2d) * (r * wave)[None, :], wave, axis=1)
    return num / TRAPZ(band.resp * band.wave, band.wave), cov


def survey_bands(survey, names=("g", "r", "i")):
    return [make_band(n, FILTERS / SURVEY_FILES[survey].format(n)) for n in names]


def rest_bands():
    """Bandas de reposo del catalogo con los F0 historicos (Vega para BVR, ~AB para r)."""
    from core.utils import cteB, cteV, cteR, cter
    files = {"B_rest": ("spline_B.txt", cteB), "V_rest": ("spline_V.txt", cteV),
             "R_rest": ("bessell_R_ph_lines.dat", cteR), "r_rest": ("spline_r'.txt", cter)}
    return {k: make_band(k, LEGACY_RESP / f, f0) for k, (f, f0) in files.items()}


def legacy_band(name):
    """Curva SDSS y F0 de la proyeccion historica. Solo para la prueba dorada."""
    from core import utils
    f = {"g": "spline_g'.txt", "r": "spline_r'.txt", "i": "spline_i'.txt"}[name]
    return make_band(name, LEGACY_RESP / f, getattr(utils, "cte" + name))
```

- [ ] **Step 5: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_bands.py`
Expected: cuatro `ok`.

- [ ] **Step 6: Commit**

```bash
git add data/filters pipeline78/bands.py tests/test_p78_bands.py
git commit -m "pipeline78: curvas ZTF y OmegaCAM del SVO, punto cero AB y fotometria sintetica vectorizada"
```

### Task 2: Almacén de plantillas

**Files:**
- Create: `pipeline78/store.py`, `tests/p78_fakes.py`, `tests/test_p78_store.py`

**Interfaces:**
- Consumes: `paths.LIB`, `paths.STORE`
- Produces:
  - `md5_file(path) -> str`
  - `stable_md5(path, reader=md5_file) -> str`, que lanza `IOError` si dos lecturas difieren
  - `parse_dat(path) -> (time ndarray, wave ndarray, flux ndarray[n_ep, n_w])`
  - `tdir(clase, sn) -> Path`
  - `save_template(d, time, wave, flux, meta)`
  - `load_template(d) -> dict`, con las claves de meta más `time`, `wave` y `flux` (mmap)
  - `build_store(force=False) -> DataFrame`, que escribe `STORE/build_report.csv`
  - en `tests/p78_fakes.py`: `fake_template(d, sn, clase, t_peak, n_ep)` y `write_dat(path, times, wave, fluxes)`

- [ ] **Step 1: Escribir los tests y los falsos**

```python
# tests/p78_fakes.py
import numpy as np
from pipeline78.store import save_template

def sed(w, T=10000.0):
    bb = 1.0 / (w**5 * (np.exp(1.4388e8 / (w * T)) - 1.0))
    return bb / bb.max() * 0.5

def fake_template(d, sn="FAKE1", clase="Ia", t_peak=55000.0, n_ep=120):
    w = np.arange(3005.0, 9195.0, 1.0)
    t = t_peak + np.arange(-20.0, n_ep - 20.0, 1.0)
    prof = np.exp(-0.5 * ((t - t_peak) / 15.0) ** 2) + 0.05
    save_template(d, t, w, prof[:, None] * sed(w)[None, :],
                  dict(sn=sn, clase=clase, md5_src="fake", n_epochs=int(t.size), t_first=float(t[0]),
                       t_last=float(t[-1]), wmin=float(w[0]), wmax=float(w[-1]), n_neg=0))

def write_dat(path, times, wave, fluxes):
    with open(path, "w") as fh:
        for t, f in zip(times, fluxes):
            fh.write(f"# time:\t{t}\n# SPEC\n#      WAVE   FLUX\n")
            for wi, fi in zip(wave, f):
                fh.write(f"{wi} {fi}\n")
```

```python
# tests/test_p78_store.py
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline78.store import parse_dat, save_template, load_template, stable_md5
from tests.p78_fakes import write_dat

def test_parse_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.dat"
        w = np.arange(3005.0, 3010.0)
        write_dat(p, [55001.0, 55000.0], w, [w * 2, w * 1])
        t, ww, f = parse_dat(p)
        assert list(t) == [55000.0, 55001.0]          # ordena por tiempo
        assert np.allclose(f[0], w) and np.allclose(f[1], 2 * w)
        save_template(pathlib.Path(td) / "s", t, ww, f, dict(sn="x"))
        tpl = load_template(pathlib.Path(td) / "s")
        assert tpl["flux"].dtype == np.float32 and tpl["flux"].shape == (2, 5)

def test_parse_rejects_bad_grid():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.dat"
        with open(p, "w") as fh:
            fh.write("# time: 1\n3005 1\n3006 1\n# time: 2\n3005 1\n3007 1\n")
        try:
            parse_dat(p); raise AssertionError("debio fallar")
        except ValueError as e:
            assert "grilla" in str(e)

def test_parse_rejects_nan():
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.dat"
        with open(p, "w") as fh:
            fh.write("# time: 1\n3005 nan\n3006 1\n")
        try:
            parse_dat(p); raise AssertionError("debio fallar")
        except ValueError as e:
            assert "no finito" in str(e)

def test_stable_md5_detects_change():
    vals = iter(["aaa", "bbb"])
    try:
        stable_md5("ignorado", reader=lambda p: next(vals)); raise AssertionError("debio fallar")
    except IOError:
        pass

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_store.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78.store'`

- [ ] **Step 3: Implementar**

```python
# pipeline78/store.py
"""Biblioteca congelada (.dat de texto, 5-270 MB) -> matrices npy locales, una vez."""
import hashlib, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import LIB, STORE


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stable_md5(path, reader=md5_file):
    """Dos lecturas completas: la primera hidrata el archivo de Drive, la segunda lo confirma."""
    a, b = reader(path), reader(path)
    if a != b:
        raise IOError(f"{path}: md5 distinto entre dos lecturas (archivo de Drive a medio hidratar)")
    return a


def parse_dat(path):
    times, counts, rows, n = [], [], [], 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("# time"):
                if times:
                    counts.append(n)
                times.append(float(line.split(":", 1)[1]))
                n = 0
            elif line.strip() and not line.lstrip().startswith("#"):
                rows.append(line)
                n += 1
    if not times:
        raise ValueError(f"{path}: sin bloques '# time'")
    counts.append(n)
    data = pd.read_csv(io.StringIO("".join(rows)), sep=r"\s+", header=None).to_numpy(dtype=float)
    if not np.all(np.isfinite(data[:, :2])):
        raise ValueError(f"{path}: flujo o lambda no finito")
    blocks = np.split(data[:, :2], np.cumsum(counts)[:-1])
    wave = blocks[0][:, 0]
    for t, b in zip(times, blocks):
        if b.shape[0] != wave.size or not np.allclose(b[:, 0], wave, rtol=0.0, atol=1e-4):
            raise ValueError(f"{path}: la grilla de lambda del bloque {t} difiere del primero")
    flux = np.vstack([b[:, 1] for b in blocks])
    order = np.argsort(times)
    return np.asarray(times, dtype=float)[order], wave, flux[order]


def tdir(clase, sn):
    return STORE / "templates" / clase / sn


def save_template(d, time, wave, flux, meta):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "time.npy", np.asarray(time, dtype=float))
    np.save(d / "wave.npy", np.asarray(wave, dtype=float))
    np.save(d / "flux.npy", np.asarray(flux, dtype=np.float32))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def load_template(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    return dict(meta, time=np.load(d / "time.npy"), wave=np.load(d / "wave.npy"),
                flux=np.load(d / "flux.npy", mmap_mode="r"))


def build_store(force=False):
    man = pd.read_csv(LIB / "MANIFIESTO.csv")
    report = []
    for r in man.itertuples():
        src = LIB / r.clase / "mangled" / f"{r.sn}.dat"
        d = tdir(r.clase, r.sn)
        md5 = stable_md5(src)
        if not force and (d / "meta.json").exists() and json.loads((d / "meta.json").read_text()).get("md5_src") == md5:
            report.append((r.sn, r.clase, "ya estaba"))
            continue
        t, w, f = parse_dat(src)
        save_template(d, t, w, f, dict(sn=r.sn, clase=r.clase, md5_src=md5, n_epochs=int(t.size),
                                       t_first=float(t[0]), t_last=float(t[-1]), wmin=float(w[0]),
                                       wmax=float(w[-1]), n_neg=int((f < 0).sum())))
        report.append((r.sn, r.clase, f"ok {t.size} epocas"))
        print(r.clase, r.sn, report[-1][2], flush=True)
    rep = pd.DataFrame(report, columns=["sn", "clase", "estado"])
    STORE.mkdir(parents=True, exist_ok=True)
    rep.to_csv(STORE / "build_report.csv", index=False)
    return rep


if __name__ == "__main__":
    build_store()
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_store.py`
Expected: cuatro `ok`.

- [ ] **Step 5: Construir el almacén real.** Lee 2.4 GB de Drive dos veces. Avisar a Mauricio, porque tarda de 30 a 60 minutos.

```bash
cd "$REPO" && nohup $PY -m pipeline78.store > ~/thesis_store/build.log 2>&1 &
# al terminar:
$PY -c "
import pandas as pd; r=pd.read_csv('$HOME/thesis_store/build_report.csv')
print(r.groupby('clase').size().to_dict()); print(r[~r.estado.str.startswith(('ok','ya'))])"
du -sh ~/thesis_store/templates
```

Expected: `{'II': 13, 'IIb': 10, 'IIn': 10, 'Ia': 15, 'Ibc': 30}`, sin filas de error y unos 0.6 a 1.2 GB en disco.

- [ ] **Step 6: Commit**

```bash
git add pipeline78/store.py tests/p78_fakes.py tests/test_p78_store.py
git commit -m "pipeline78: almacen npy de la biblioteca congelada con md5 estable"
```

### Task 3: Catálogo (máximo, magnitud de referencia y dm15)

> **Enmienda 2026-10-02 (decisión de Mauricio): ancla en el pico principal.** `argmin` toma el pico de enfriamiento en 2011fu (día 1), 2013df (día 2) y 2006aj (época 0), y deja el pico de Ni 0.2 a 0.6 mag más débil de lo que pide la LF. `t_peak` y `M_ref` se miden sobre el **pico principal**: el mínimo de r en reposo descartando un máximo local temprano de enfriamiento (pico en los primeros días seguido de un mínimo local de brillo y un segundo máximo). Agregar un test con una curva sintética de doble pico. Al correr el Step 5, listar las SNe cuyo t_peak cambia respecto de argmin y mostrárselas a Mauricio.

**Files:**
- Create: `pipeline78/catalog.py`, `tests/test_p78_catalog.py`

**Interfaces:**
- Consumes: `store.load_template`, `bands.rest_bands`, `bands.synphot`
- Produces:
  - `CLF_CLASS: dict`, `REF_BAND: dict`
  - `rest_mag(tpl, band) -> ndarray`
  - `peak_and_dm15(t, m) -> (t_peak, m_peak, at_edge, dm15)`
  - `build_catalog(store_dir=STORE) -> DataFrame`, que escribe `STORE/catalog.csv` y agrega a cada `meta.json` las claves t_peak, peak_at_edge, M_ref, ref_band, dm15_B y clf_class

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_p78_catalog.py
import sys, os, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

def test_peak_and_dm15_gaussian():
    from pipeline78.catalog import peak_and_dm15
    t = np.arange(0.0, 100.0)
    m = 0.01 * (t - 30.0) ** 2
    tp, mp, edge, dm15 = peak_and_dm15(t, m)
    assert tp == 30.0 and not edge and abs(dm15 - 2.25) < 1e-9

def test_build_catalog_on_fake_store():
    with tempfile.TemporaryDirectory() as td:
        os.environ["P78_STORE"] = td
        import importlib, pipeline78.paths, pipeline78.store, pipeline78.catalog
        for m in (pipeline78.paths, pipeline78.store, pipeline78.catalog): importlib.reload(m)
        from tests.p78_fakes import fake_template
        fake_template(pathlib.Path(td) / "templates/Ia/FAKE1", t_peak=55000.0)
        cat = pipeline78.catalog.build_catalog(pathlib.Path(td))
        r = cat.iloc[0]
        assert abs(r.t_peak - 55000.0) <= 1.0 and r.clf_class == "Ia" and np.isfinite(r.M_ref)
        assert np.isfinite(r.dm15_B)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_catalog.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78.catalog'`

- [ ] **Step 3: Implementar**

```python
# pipeline78/catalog.py
"""Una fila por template: maximo en r de reposo (ancla), M de referencia y dm15(B)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import STORE
from pipeline78.store import load_template
from pipeline78.bands import rest_bands, synphot, COVERAGE_MIN

CLF_CLASS = {"Ia": "Ia", "II": "II", "IIb": "II", "IIn": "II", "Ibc": "Ibc"}           # D1
REF_BAND = {"Ia": "R_rest", "II": "r_rest", "IIb": "r_rest", "IIn": "r_rest", "Ibc": "r_rest"}
# Ia: Prieto+2006 calibra en R (Bessell). IIb: Taddia+2018 en r. II/Ibc: valores adoptados, anclados en r.


def rest_mag(tpl, band):
    F, cov = synphot(tpl["wave"], tpl["flux"], band)
    if cov <= COVERAGE_MIN:
        raise ValueError(f"{tpl['sn']}: {band.name} no queda cubierta en reposo")
    return -2.5 * np.log10(np.clip(F, 1e-300, None) / band.f0)


def peak_and_dm15(t, m):
    i = int(np.argmin(m))
    tp = float(t[i])
    dm15 = float(np.interp(tp + 15.0, t, m) - m[i]) if tp + 15.0 <= t[-1] else float("nan")
    return tp, float(m[i]), bool(i == 0 or i == len(m) - 1), dm15


def build_catalog(store_dir=STORE):
    rb = rest_bands()
    rows = []
    for meta_p in sorted(Path(store_dir).glob("templates/*/*/meta.json")):
        tpl = load_template(meta_p.parent)
        cls = tpl["clase"]
        t_peak, _, edge, _ = peak_and_dm15(tpl["time"], rest_mag(tpl, rb["r_rest"]))
        M_ref = float(np.min(rest_mag(tpl, rb[REF_BAND[cls]])))
        dm15 = peak_and_dm15(tpl["time"], rest_mag(tpl, rb["B_rest"]))[3] if cls == "Ia" else float("nan")
        meta = json.loads(meta_p.read_text())
        meta.update(t_peak=t_peak, peak_at_edge=edge, M_ref=M_ref, ref_band=REF_BAND[cls],
                    dm15_B=None if np.isnan(dm15) else dm15, clf_class=CLF_CLASS[cls])
        meta_p.write_text(json.dumps(meta, indent=1))
        rows.append(dict(meta, store_path=str(meta_p.parent)))
    cat = pd.DataFrame(rows)
    cat.to_csv(Path(store_dir) / "catalog.csv", index=False)
    return cat


if __name__ == "__main__":
    c = build_catalog()
    print(c.groupby("clase").size().to_dict())
    print(c[["sn", "clase", "t_peak", "peak_at_edge", "M_ref", "dm15_B"]].to_string(index=False))
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_catalog.py`
Expected: dos `ok`.

- [ ] **Step 5: Catálogo real y revisión de sanidad**

Run: `cd "$REPO" && $PY -m pipeline78.catalog | tee ~/thesis_store/catalog.log`

Expected:
- 78 filas.
- SN2011fe con dm15_B entre 1.00 y 1.25 (la medición anterior fue 1.10) y t_peak entre 55810 y 55820.
- Ninguna Ia con peak_at_edge.
- Si alguna II, IIb, IIn o Ibc sale con peak_at_edge, anotarla y mostrársela a Mauricio. Una serie que empieza después del máximo queda anclada en su primera época.

- [ ] **Step 6: Commit**

```bash
git add pipeline78/catalog.py tests/test_p78_catalog.py
git commit -m "pipeline78: catalogo con ancla en el maximo de reposo, M de referencia y dm15 de las 15 Ia"
```

### Task 4: Motor fotométrico con dilatación temporal y prueba dorada

**Files:**
- Create: `pipeline78/engine.py`, `tests/test_p78_engine.py`

**Interfaces:**
- Consumes: `core.correction.redden_spectrum_adjusted`, `core.utils.DL_calculator`, `bands.synphot`, `bands.COVERAGE_MIN`
- Produces:
  - `extinction_factor(wave, rv, ebv) -> ndarray`
  - `observed_lightcurves(tpl, z, ebv_host, rv_host, ebv_mw, bands, dmag=0.0) -> (t_rel ndarray, dict band_name -> mag ndarray)`
  - `t_rel = (tpl["time"] - tpl["t_peak"]) * (1+z)`, en días observados respecto del máximo
  - una banda sin cobertura mayor que `COVERAGE_MIN` no aparece en el dict

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_p78_engine.py
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline78.engine import extinction_factor, observed_lightcurves
from pipeline78.bands import legacy_band, survey_bands

def _tpl():
    w = np.arange(3005.0, 9195.0, 1.0)
    t = np.arange(-20.0, 100.0)
    prof = np.exp(-0.5 * (t / 15.0) ** 2) + 0.05
    bb = 1.0 / (w**5 * (np.exp(1.4388e8 / (w * 10000.0)) - 1.0))
    return dict(sn="x", time=t, wave=w, flux=prof[:, None] * (bb / bb.max())[None, :], t_peak=0.0)

def test_extinction_factor_v_band():
    f = extinction_factor(np.array([5500.0]), 3.1, 0.1)[0]
    assert abs(f - 10 ** (-0.4 * 0.31)) < 0.02

def test_time_dilation():
    t_rel, _ = observed_lightcurves(_tpl(), 0.5, 0.0, 3.1, 0.0, survey_bands("ZTF", ("r",)))
    assert np.allclose(t_rel, np.arange(-20.0, 100.0) * 1.5)

def test_high_z_drops_blue_band():
    _, mags = observed_lightcurves(_tpl(), 0.6, 0.0, 3.1, 0.0, survey_bands("SUDARE"))
    assert "g" not in mags and "i" in mags

def test_dmag_shifts_all_bands_equally():
    b = survey_bands("ZTF")
    _, a = observed_lightcurves(_tpl(), 0.05, 0.1, 3.1, 0.02, b, dmag=0.0)
    _, c = observed_lightcurves(_tpl(), 0.05, 0.1, 3.1, 0.02, b, dmag=1.3)
    for k in a: assert np.allclose(c[k] - a[k], 1.3, atol=1e-6)

def test_golden_vs_legacy_chain():
    """Mismo template, z, polvo y curva SDSS r: el motor nuevo contra correct_redeening + Syntetic_photometry_v2."""
    from pipeline78.paths import LIB
    from pipeline78.store import parse_dat
    from core.utils import leer_spec, Syntetic_photometry_v2
    from core.correction import correct_redeening
    path = min(LIB.glob("Ia/mangled/*.dat"), key=lambda p: p.stat().st_size)
    t, w, f = parse_dat(path)
    tpl = dict(sn=path.stem, time=t[:5], wave=w, flux=f[:5], t_peak=float(t[0]))
    rb = legacy_band("r")
    _, mags = observed_lightcurves(tpl, 0.03, 0.10, 3.1, 0.03, [rb])
    esp, fases = leer_spec(str(path), ot=False, as_pandas=True)
    espc, _ = correct_redeening(sn=path.stem, ESPECTRO=esp[:5], fases=fases[:5], z=0.03, ebmv_host=0.10,
                                ebmv_mw=0.03, reverse=True, use_DL=True, rv_host=3.1)
    old = []
    for s in espc:
        F, _ = Syntetic_photometry_v2(s["wave"].values, s["flux"].values, rb.wave, rb.resp)
        old.append(-2.5 * np.log10(F / rb.f0))
    d = np.abs(np.asarray(old) - mags["r"])
    assert np.median(d) < 0.003 and d.max() < 0.01, d

def test_speed():
    from pipeline78.store import load_template
    from pipeline78.paths import STORE
    import pandas as pd
    p = pd.read_csv(STORE / "catalog.csv").query("sn == 'SN2011fe'").store_path.iloc[0]
    tpl = load_template(p)
    b = survey_bands("ZTF")
    t0 = time.time()
    for _ in range(20): observed_lightcurves(tpl, 0.05, 0.1, 3.1, 0.02, b)
    per = (time.time() - t0) / 20
    print(f"   {per*1000:.0f} ms por simulacion (SN2011fe, {tpl['n_epochs']} epocas)")
    assert per < 0.5

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_engine.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78.engine'`

- [ ] **Step 3: Implementar**

```python
# pipeline78/engine.py
"""Espectros de reposo a 10 pc -> magnitudes AB observadas por banda.

Orden fisico: brillo intrinseco (dmag) -> host (reposo, R_V del tipo) -> redshift (lambda*(1+z), F/(1+z))
-> Via Lactea (observado, R_V 3.1) -> distancia (10 pc / D_L)^2. Tiempos: (t - t_peak)*(1+z).
Sin LOESS (D2): las series congeladas son diarias y ya estan mangladas a la fotometria.
"""
import numpy as np
from core.correction import redden_spectrum_adjusted
from core.utils import DL_calculator
from pipeline78.bands import synphot, COVERAGE_MIN


def extinction_factor(wave, rv, ebv):
    wave = np.asarray(wave, dtype=float)
    if ebv <= 0:
        return np.ones_like(wave)
    return np.asarray(redden_spectrum_adjusted(wave, np.ones_like(wave), Rv=rv, ebmv=ebv), dtype=float)


def observed_lightcurves(tpl, z, ebv_host, rv_host, ebv_mw, bands, dmag=0.0):
    w = np.asarray(tpl["wave"], dtype=float)
    f = np.asarray(tpl["flux"], dtype=np.float64) * 10.0 ** (-0.4 * dmag)
    f = f * extinction_factor(w, rv_host, ebv_host)[None, :]
    wo = w * (1.0 + z)
    f = f / (1.0 + z)
    f = f * extinction_factor(wo, 3.1, ebv_mw)[None, :]
    f = f * (1e-5 / DL_calculator(z)) ** 2
    mags = {}
    for b in bands:
        F, cov = synphot(wo, f, b)
        if cov > COVERAGE_MIN:
            mags[b.name] = -2.5 * np.log10(np.clip(F, 1e-300, None) / b.f0)
    t_rel = (np.asarray(tpl["time"], dtype=float) - float(tpl["t_peak"])) * (1.0 + z)
    return t_rel, mags
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_engine.py`
Expected: seis `ok`. La prueba dorada imprime las diferencias si falla. Si falla por más de 0.01 mag, comparar paso a paso el orden de las operaciones y la regrilla a 1 Å del código viejo (`correction.py:386-389`). No aflojar el umbral sin mostrárselo a Mauricio.

- [ ] **Step 5: Commit**

```bash
git add pipeline78/engine.py tests/test_p78_engine.py
git commit -m "pipeline78: motor fotometrico vectorizado con dilatacion temporal, validado contra la cadena historica"
```

### Task 5: Logs de survey en esquema común

**Files:**
- Create: `pipeline78/survey.py`, `tests/test_p78_survey.py`

**Interfaces:**
- Consumes: `paths.DATA`, `paths.SUDARE_DIR`, `paths.STORE`
- Produces:
  - `build_ztf_log(csv_path, out) -> int`
  - `build_sudare_log(out, ref_epochs=None) -> DataFrame`
  - `load_log(path, fields=None) -> dict[field][band] = (mjd ndarray, maglim ndarray)`, ordenado por mjd
  - `SUDARE_SEASON_SPLIT`

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_p78_survey.py
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline78.survey import build_ztf_log, build_sudare_log, load_log

def test_ztf_best_per_day():
    with tempfile.TemporaryDirectory() as td:
        csv = pathlib.Path(td) / "log.csv"
        pd.DataFrame({"oid": ["A", "A", "A", "B"], "mjd": [100.1, 100.7, 101.2, 100.1], "fid": [2, 2, 1, 2],
                      "diffmaglim": [20.0, 20.5, 19.0, 21.0]}).to_csv(csv, index=False)
        out = pathlib.Path(td) / "best.parquet"
        build_ztf_log(csv, out)
        log = load_log(out, ["A"])
        assert list(log) == ["A"]
        assert list(log["A"]["r"][1]) == [20.5] and list(log["A"]["g"][0]) == [101.2]

def test_sudare_counts_match_paper_table1():
    with tempfile.TemporaryDirectory() as td:
        df = build_sudare_log(pathlib.Path(td) / "s.parquet")
        n = df.groupby(["field", "band"]).size().to_dict()
        assert (n[("cdfs1", "r")], n[("cdfs1", "g")], n[("cdfs1", "i")]) == (29, 7, 11)
        assert (n[("cdfs2", "r")], n[("cdfs2", "g")], n[("cdfs2", "i")]) == (23, 6, 4)
        assert (n[("cosmos1", "r")], n[("cosmos1", "g")], n[("cosmos1", "i")]) == (28, 7, 7)
        assert n[("cdfs3", "r")] == 30 and n[("cdfs4", "r")] == 29
        assert n[("cosmos2", "r")] == 24 and n[("cosmos3", "r")] == 13

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_survey.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78.survey'`

- [ ] **Step 3: Implementar**

```python
# pipeline78/survey.py
"""Logs de observacion -> esquema comun (field, mjd, band, maglim). Una epoca por dia, campo y banda."""
import numpy as np
import pandas as pd
from pipeline78.paths import SUDARE_DIR

SUDARE_SEASON_SPLIT = [(56300.0, "cosmos1"), (56900.0, "cosmos2"), (1e9, "cosmos3")]  # MJD de corte


def _best_per_day(df):
    df = df.dropna(subset=["band", "maglim"]).copy()
    df["mjd_day"] = np.floor(df["mjd"]).astype("int64")
    idx = df.groupby(["field", "mjd_day", "band"])["maglim"].idxmax()
    return df.loc[idx].drop(columns="mjd_day").sort_values(["field", "mjd"]).reset_index(drop=True)


def build_ztf_log(csv_path, out):
    df = pd.read_csv(csv_path, usecols=["oid", "mjd", "fid", "diffmaglim"])
    df = df.rename(columns={"oid": "field", "diffmaglim": "maglim"})
    df["band"] = df["fid"].map({1: "g", 2: "r", 3: "i"})
    best = _best_per_day(df[["field", "mjd", "band", "maglim"]])
    best.to_parquet(out, index=False)
    return len(best)


def build_sudare_log(out, ref_epochs=None):
    """obslog_I + obslog_II (las lineas con '#' quedan fuera). cosmos se parte en sus tres temporadas.
    ref_epochs: set de (field, band, mjd_day) que son imagen template, se marcan is_ref."""
    rows = []
    for fname in ("obslog_I.txt", "obslog_II.txt"):
        for line in open(SUDARE_DIR / fname):
            p = line.split()
            if len(p) < 6 or line.lstrip().startswith("#") or p[0] == "field":
                continue
            field = p[0].split("_")[0]
            mjd = float(p[3])
            if field == "cosmos":
                field = next(name for cut, name in SUDARE_SEASON_SPLIT if mjd < cut)
            rows.append((field, mjd, p[1], float(p[5]), float(p[4])))
    df = pd.DataFrame(rows, columns=["field", "mjd", "band", "maglim", "seeing"]).sort_values(["field", "mjd"])
    ref = ref_epochs or set()
    df["is_ref"] = [(f, b, int(np.floor(m))) in ref for f, b, m in zip(df.field, df.band, df.mjd)]
    df.reset_index(drop=True).to_parquet(out, index=False)
    return df


def load_log(path, fields=None):
    df = pd.read_parquet(path)
    if "is_ref" in df.columns:
        df = df[~df["is_ref"]]
    if fields is not None:
        df = df[df["field"].isin(set(fields))]
    out = {}
    for field, g in df.groupby("field", sort=False):
        out[field] = {b: (gb["mjd"].to_numpy(float), gb["maglim"].to_numpy(float))
                      for b, gb in g.sort_values("mjd").groupby("band")}
    return out
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_survey.py`
Expected: dos `ok`. Si los conteos de SUDARE no calzan, revisar el corte de temporadas contra las fechas de `estado_sudare/index.html`.

- [ ] **Step 5: Construir el log de ZTF.** Lee 355 MB de Drive y tarda unos minutos.

```bash
$PY -c "
from pipeline78.survey import build_ztf_log; from pipeline78.paths import DATA, STORE
print(build_ztf_log(DATA/'ZTF_observing_log_complete.csv', STORE/'ztf_obslog_best.parquet'))"
cat "$REPO"/data/oids_1000_v2_p{1,2,3,4}.txt > ~/thesis_store/ztf_fields_1000.txt && wc -l ~/thesis_store/ztf_fields_1000.txt
```

Expected: unos 4.4 a 5.0 millones de filas y 1000 campos.

- [ ] **Step 6: Commit**

```bash
git add pipeline78/survey.py tests/test_p78_survey.py
git commit -m "pipeline78: logs de ZTF y SUDARE en esquema comun, temporadas de cosmos separadas"
```

### Task 6: Muestreo con un rng por simulación

> **Enmienda 2026-10-02.** (1) `test_frac_zero_matches_config` del plan falla por construcción: con τ=0.25 la componente con polvo también cae bajo E=0.05 (P total ~0.57 para II). Se reemplaza por la CDF analítica de la mezcla en t=0.05, `f0·erf(t/(σ0√2)) + (1−f0)(1−exp(−t·R_V/τ))`, con tolerancia 0.02. (2) `EXT_KEY["IIn"] = "SNIIn"`: la entrada ya existe en `config.EXTINCTION_CONFIG` (copia provisional de SNII). (3) D4 aprobado: Step 5 se ejecuta.

**Files:**
- Modify: `config.py:349-351`. Ia vuelve a τ 0.35 y frac_zero 0.40, si se aprueba D4.
- Create: `pipeline78/sampling.py`, `tests/test_p78_sampling.py`

**Interfaces:**
- Consumes: `config.EXTINCTION_CONFIG`, `LUMINOSITY_CONFIG`, `PHILLIPS_CONFIG`, `paths.DATA`, `paths.STORE`
- Produces:
  - `sample_ebv_host(rng, cls) -> (ebv, rv)`
  - `sample_mpeak(rng, cls, dm15=None) -> float`
  - `zgrid_cdf(zmin, zmax) -> (z, cdf)`
  - `z_sampler(cfg) -> callable(rng, cls) -> float`
  - `load_mw(cfg) -> dict field -> ebv_mw`
  - `EXT_KEY`

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_p78_sampling.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline78.sampling import sample_ebv_host, sample_mpeak, zgrid_cdf, z_sampler
from config import EXTINCTION_CONFIG, PHILLIPS_CONFIG

def test_frac_zero_matches_config():
    rng = np.random.default_rng(1)
    e = np.array([sample_ebv_host(rng, "II")[0] for _ in range(20000)])
    assert abs((e < 0.05).mean() - EXTINCTION_CONFIG["SNII"]["frac_zero"]) < 0.05

def test_phillips_mean():
    rng = np.random.default_rng(2)
    m = np.array([sample_mpeak(rng, "Ia", PHILLIPS_CONFIG["dm15_ref"]) for _ in range(20000)])
    assert abs(m.mean() - PHILLIPS_CONFIG["M0"]) < 0.01

def test_same_seed_same_draws():
    a = sample_mpeak(np.random.default_rng([7, 1, 2]), "II"); b = sample_mpeak(np.random.default_rng([7, 1, 2]), "II")
    assert a == b

def test_volumetric_favours_high_z():
    z, c = zgrid_cdf(0.05, 1.0)
    assert np.interp(0.5, c, z) > 0.6

def test_fixed_z_sampler():
    assert z_sampler({"z_mode": "fixed", "z_fixed": 0.2})(np.random.default_rng(0), "Ia") == 0.2

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_sampling.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78.sampling'`

- [ ] **Step 3: Implementar**

```python
# pipeline78/sampling.py
"""Sorteos de una simulacion con SU propio rng (determinista e independiente del orden de ejecucion)."""
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from config import EXTINCTION_CONFIG, LUMINOSITY_CONFIG, PHILLIPS_CONFIG
from pipeline78.paths import DATA, STORE

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)          # la misma que core.utils.DL_calculator
EXT_KEY = {"Ia": "SNIa", "II": "SNII", "IIb": "SNIIb", "IIn": "SNII", "Ibc": "SNIbc"}   # IIn: ver Task 17


def sample_ebv_host(rng, cls):
    p = EXTINCTION_CONFIG[EXT_KEY[cls]]
    if rng.random() < p["frac_zero"]:
        return float(abs(rng.normal(0.0, p["sigma_zero"]))), float(p["Rv"])
    av = min(float(rng.exponential(p["tau"])), float(p["Av_max"]))
    return av / float(p["Rv"]), float(p["Rv"])


def sample_mpeak(rng, cls, dm15=None):
    """M intrinseco (libre de polvo) en la banda de referencia del catalogo (D4)."""
    if cls == "Ia" and PHILLIPS_CONFIG.get("enabled", False):
        d = dm15 if dm15 is not None and np.isfinite(dm15) else PHILLIPS_CONFIG["dm15_default"]
        m = (PHILLIPS_CONFIG["M0"] + PHILLIPS_CONFIG["slope"] * (d - PHILLIPS_CONFIG["dm15_ref"])
             + rng.normal(0.0, PHILLIPS_CONFIG["sigma_resid"]))
    else:
        p = LUMINOSITY_CONFIG["M_peak"][cls]
        m = rng.normal(p["mean"], p["sigma"])
    c = LUMINOSITY_CONFIG["clip"]
    return float(np.clip(m, c["min"], c["max"]))


def zgrid_cdf(zmin, zmax, n=4000):
    z = np.linspace(zmin, zmax, n)
    dv = COSMO.differential_comoving_volume(z).value
    c = np.concatenate([[0.0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(z))])
    return z, c / c[-1]


def z_sampler(cfg):
    mode = cfg["z_mode"]
    if mode == "fixed":
        return lambda rng, cls: float(cfg["z_fixed"])
    if mode == "empirical":
        vals = {c: np.loadtxt(DATA / f) for c, f in cfg["z_files"].items()}
        return lambda rng, cls: float(max(0.005, rng.choice(vals[cls]) + rng.normal(0.0, 0.003)))
    if mode == "volumetric":
        z, c = zgrid_cdf(cfg["zmin"], cfg["zmax"])
        return lambda rng, cls: float(np.interp(rng.random(), c, z))
    raise ValueError(mode)


def load_mw(cfg):
    mode = cfg["mw_mode"]
    if mode == "const":
        return {}
    if mode == "ztf_sfd":
        d = pd.read_parquet(DATA / "sfd98_cache.parquet")
        return dict(zip(d["oid"], d["ebmv_mw"].astype(float)))
    if mode == "sudare_fields":
        d = pd.read_csv(STORE / "sudare_fields.csv")
        return dict(zip(d["field"], d["ebmv_mw"].astype(float)))
    raise ValueError(mode)
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_sampling.py`
Expected: cinco `ok`.

- [ ] **Step 5: Polvo de las Ia (solo si D4 fue sí).** En `config.py:349-351` volver a `"tau": 0.35` y `"frac_zero": 0.40`. Agregar este comentario: "D4 2026-10: el M ahora es intrinseco y el polvo atenua; la recalibracion 0.65/0.25 del 2026-08-06 suponia que la normalizacion cancelaba la extincion (H5). Se re-verifica la dispersion en el piloto de la Tarea 8."

- [ ] **Step 6: Commit**

```bash
git add pipeline78/sampling.py tests/test_p78_sampling.py config.py
git commit -m "pipeline78: sorteos con rng por simulacion; M intrinseco; polvo Ia vuelve a literatura (D4)"
```

### Parte B: proyección ZTF

### Task 7: Proyector y runner (paralelo, determinista, reanudable)

> **Enmienda 2026-10-02 (D1 de Mauricio).** `ztf_v78` lleva además IIn 10 con su etiqueta: `classes=["Ia","II","IIb","IIn","Ibc"]`, `n_by_class={"Ia":10,"II":8,"IIb":2,"IIn":10,"Ibc":10}`, `z_files["IIn"]="z_empirical_Ia.txt"` (M medio de IIn ~ Ia, no hay IIn reales en ZTF). La ventana pre-explosión (`pre_ul_days`) queda como el plan: Mauricio advirtió que cambiarla afecta el fit de Villar (el extractor exige un UL previo a la primera detección); se mide en el piloto (Tarea 8) y se decide en G1.

**Files:**
- Create: `pipeline78/project.py`, `pipeline78/runcfg.py`, `pipeline78/run.py`, `tests/test_p78_run.py`

**Interfaces:**
- Consumes: todo lo de las Tareas 1 a 6
- Produces:
  - `project.anchor_time(cfg, rng, k, n, epochs) -> float`
  - `project.project_one(t_rel, mags, epochs, t_anchor, rng, cfg) -> DataFrame | None`, con las columnas mjd, filter, maglimit, magnitud_modelo, magnitud_proyectada, magerr, upperlimit, detected y found
  - `runcfg.RUNS_CFG: dict[str, dict]`, `runcfg.units(cfg, fields)`, `runcfg.log_path(cfg)`
  - `run.main(argv) -> Path` (la carpeta de la corrida), `run.sim_rng(seed, field, cls, k)`, `run.config_hash(cfg)`
  - la carpeta con el formato de la sección 3

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_p78_run.py
import sys, os, pathlib, tempfile, importlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

def _setup(td, n=6):
    os.environ["P78_STORE"] = td; os.environ["P78_RUNS"] = td
    import pipeline78.paths, pipeline78.store, pipeline78.catalog, pipeline78.runcfg, pipeline78.run
    for m in (pipeline78.paths, pipeline78.store, pipeline78.catalog, pipeline78.runcfg, pipeline78.run): importlib.reload(m)
    from tests.p78_fakes import fake_template
    fake_template(pathlib.Path(td) / "templates/Ia/FAKE1", "FAKE1", "Ia", 55000.0)
    fake_template(pathlib.Path(td) / "templates/Ia/FAKE2", "FAKE2", "Ia", 55000.0)
    pipeline78.catalog.build_catalog(pathlib.Path(td))
    mjd = np.arange(54900.0, 55300.0, 2.0)
    log = pd.DataFrame([("F1", m, b, 20.5) for m in mjd for b in "gr"], columns=["field", "mjd", "band", "maglim"])
    log.to_parquet(pathlib.Path(td) / "log.parquet", index=False)
    (pathlib.Path(td) / "fields.txt").write_text("F1\n")
    cfg = dict(survey="ZTF", bands=["g", "r", "i"], classes=["Ia"], n_by_class={"Ia": n}, chunk=None,
               anchor="pivot", z_mode="fixed", z_fixed=0.02, mw_mode="const", mw_const=0.02, rule="ztf",
               pre_ul_days=25.0, noise_k=5.0, sigma_floor=0.02, log_path=str(pathlib.Path(td) / "log.parquet"))
    pipeline78.runcfg.RUNS_CFG["_test"] = cfg
    return pipeline78.run, cfg

def _argv(td, out="r1", seed=1):
    return ["--run", "_test", "--out", str(pathlib.Path(td) / out), "--fields-file", str(pathlib.Path(td) / "fields.txt"),
            "--seed", str(seed), "--workers", "1", "--allow-dirty"]

def test_end_to_end_and_determinism():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        out1 = run.main(_argv(td, "r1")); out2 = run.main(_argv(td, "r2"))
        s = pd.read_parquet(out1 / "_sims_all.parquet")
        assert len(s) == 6 and set(s.status) == {"ok"}
        a = pd.read_parquet(next(out1.glob("F1__*.parquet"))); b = pd.read_parquet(next(out2.glob("F1__*.parquet")))
        pd.testing.assert_frame_equal(a, b)
        need = {"oid", "part_index", "sn_type", "mjd", "filter", "magnitud_proyectada", "magerr", "upperlimit", "sim_id"}
        assert need <= set(a.columns)
        assert (a.loc[a.upperlimit == "T", "magerr"].isna()).all()

def test_no_coverage_is_logged():
    with tempfile.TemporaryDirectory() as td:
        run, cfg = _setup(td, n=2)
        cfg["z_fixed"] = 3.0          # todas las bandas fuera de la biblioteca
        out = run.main(_argv(td, "r3"))
        s = pd.read_parquet(out / "_sims_all.parquet")
        assert len(s) == 2 and set(s.status) == {"no_coverage"}

def test_resume_skips_complete_units():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        out = run.main(_argv(td, "r4"))
        (out / "_sims_all.parquet").unlink()
        t0 = next((out / "_sims").glob("*.parquet")).stat().st_mtime
        run.main(_argv(td, "r4"))
        assert next((out / "_sims").glob("*.parquet")).stat().st_mtime == t0
        assert not list(out.glob("*.tmp"))

def test_refuses_config_mismatch():
    with tempfile.TemporaryDirectory() as td:
        run, _ = _setup(td)
        run.main(_argv(td, "r5", seed=1))
        try:
            run.main(_argv(td, "r5", seed=2)); raise AssertionError("debio negarse")
        except SystemExit as e:
            assert "otra configuracion" in str(e)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_run.py`
Expected: `ModuleNotFoundError: No module named 'pipeline78.runcfg'`

- [ ] **Step 3: Implementar el proyector**

```python
# pipeline78/project.py
"""Ancla en el cielo y cadencia del survey: interpolacion, ruido, deteccion y limites."""
import numpy as np
import pandas as pd


def _span(epochs):
    lo = min(m.min() for m, _ in epochs.values())
    hi = max(m.max() for m, _ in epochs.values())
    return float(lo), float(hi)


def anchor_time(cfg, rng, k, n, epochs):
    lo, hi = _span(epochs)
    if cfg["anchor"] == "pivot":                       # ZTF: pivote deterministico (tesis cap. 3)
        return lo + (hi - lo) / n * (k + 0.5)
    if cfg["anchor"] == "uniform_window":              # SUDARE: [t0 - 365, tK] como la ec. 3 de SUDARE I
        return float(rng.uniform(lo - cfg["window_pre"], hi))
    raise ValueError(cfg["anchor"])


def project_one(t_rel, mags, epochs, t_anchor, rng, cfg):
    t = t_anchor + t_rel
    t0, t1 = float(t[0]), float(t[-1])
    frames = []
    for b in cfg["bands"]:
        if b not in mags or b not in epochs:
            continue
        mjd, mlim = epochs[b]
        sel = (mjd >= t0 - cfg["pre_ul_days"]) & (mjd <= t1)
        if not sel.any():
            continue
        mj, ml = mjd[sel], mlim[sel]
        mm = np.interp(mj, t, mags[b])
        mm[mj < t0] = 99.0                               # antes de la explosion: no hay flujo
        if cfg["rule"] == "ztf":                         # identico a multiband_projection.py:356-372
            snr = cfg["noise_k"] * 10.0 ** (0.4 * (ml - mm))
            sig = np.clip(1.0857 / np.maximum(snr, 1e-6), cfg["sigma_floor"], None)
            mobs = mm + rng.normal(0.0, sig)
            det = mm < ml
            found = det
        else:
            raise ValueError(cfg["rule"])
        frames.append(pd.DataFrame({
            "mjd": mj, "filter": b, "maglimit": ml.astype(np.float32),
            "magnitud_modelo": mm.astype(np.float32),
            "magnitud_proyectada": np.where(det, mobs, ml).astype(np.float32),
            "magerr": np.where(det, sig, np.nan).astype(np.float32),
            "upperlimit": np.where(det, "F", "T"), "detected": det, "found": found}))
    return pd.concat(frames, ignore_index=True) if frames else None
```

- [ ] **Step 4: Implementar las configuraciones y el runner**

```python
# pipeline78/runcfg.py
"""Configuraciones con nombre. Cambiar una configuracion = nombre nuevo (el hash queda en el manifiesto)."""
from pipeline78.paths import STORE

RUNS_CFG = {
    "ztf_v78": dict(
        survey="ZTF", bands=["g", "r", "i"], classes=["Ia", "II", "IIb", "Ibc"],
        n_by_class={"Ia": 10, "II": 8, "IIb": 2, "Ibc": 10}, chunk=None,          # D1
        anchor="pivot", z_mode="empirical",
        z_files={"Ia": "z_empirical_Ia.txt", "II": "z_empirical_II.txt",
                 "IIb": "z_empirical_II.txt", "Ibc": "z_empirical_Ibc.txt"},
        mw_mode="ztf_sfd", mw_const=0.02, rule="ztf",
        pre_ul_days=25.0, noise_k=5.0, sigma_floor=0.02),
}


def units(cfg, fields):
    n = max(cfg["n_by_class"].values())
    step = cfg["chunk"] or n
    return [(f, a, min(a + step, n)) for f in fields for a in range(0, n, step)]


def log_path(cfg):
    if "log_path" in cfg:
        return cfg["log_path"]
    return str(STORE / {"ZTF": "ztf_obslog_best.parquet", "SUDARE": "sudare_obslog.parquet"}[cfg["survey"]])
```

```python
# pipeline78/run.py
"""Runner v78. Uso:
    $PY -m pipeline78.run --run ztf_v78 --out ~/thesis_runs/ztf_v78 --fields-file ~/thesis_store/ztf_fields_1000.txt --seed 20261002 --workers 4
"""
import argparse, hashlib, json, os, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import REPO, STORE, FILTERS
from pipeline78 import bands as B, engine, sampling, project, runcfg, survey
from pipeline78.store import load_template, md5_file

_W = {}


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def sim_rng(seed, field, cls, k):
    return np.random.default_rng([seed, _h(field) & 0xFFFFFFFF, _h(cls) & 0xFFFFFFFF, k])


def _init(cfg, seed, log_path, fields, out):
    cat = pd.read_csv(STORE / "catalog.csv")
    _W.update(cfg=cfg, seed=seed, out=Path(out), bands=B.survey_bands(cfg["survey"], cfg["bands"]),
              log=survey.load_log(log_path, fields), mw=sampling.load_mw(cfg), z=sampling.z_sampler(cfg),
              tpl={c: [load_template(p) for p in cat[cat.clase == c].sort_values("sn").store_path]
                   for c in cfg["classes"]})


def simulate(field, cls, k, epochs, mw):
    cfg, tpls = _W["cfg"], _W["tpl"][cls]
    rng = sim_rng(_W["seed"], field, cls, k)
    order = np.random.default_rng([_W["seed"], _h(field) & 0xFFFFFFFF, _h(cls) & 0xFFFFFFFF]).permutation(len(tpls))
    tpl = tpls[order[k % len(tpls)]]
    z = _W["z"](rng, cls)
    ebv, rv = sampling.sample_ebv_host(rng, cls)
    dm15 = tpl.get("dm15_B")
    M = sampling.sample_mpeak(rng, cls, dm15)
    t_rel, mags = engine.observed_lightcurves(tpl, z, ebv, rv, mw, _W["bands"], M - tpl["M_ref"])
    sim = dict(sim_id=_h(f"{field}|{cls}|{k}"), field=field, part_index=k, sn_type=cls,
               clf_class=tpl["clf_class"], template=tpl["sn"], z=z, ebmv_host=ebv, rv_host=rv,
               ebmv_mw=mw, m_peak_abs=M, dm15_used=np.nan if dm15 is None else dm15, t_anchor=np.nan,
               status="no_coverage", n_rows=0, found=False, **{f"n_det_{b}": 0 for b in cfg["bands"]})
    if not mags:
        return sim, None
    t_anchor = project.anchor_time(cfg, rng, k, cfg["n_by_class"][cls], epochs)
    df = project.project_one(t_rel, mags, epochs, t_anchor, rng, cfg)
    sim["t_anchor"] = t_anchor
    if df is None:
        sim["status"] = "no_epochs"
        return sim, None
    sim.update(status="ok", n_rows=len(df), found=bool(df["found"].any()),
               **{f"n_det_{b}": int(df.loc[df["filter"] == b, "detected"].sum()) for b in cfg["bands"]})
    for c in ("sim_id", "part_index", "sn_type", "template", "z", "ebmv_host", "rv_host", "ebmv_mw",
              "m_peak_abs", "dm15_used"):
        df[c] = sim[c]
    df["oid"] = field
    df["part_index"] = df["part_index"].astype(np.int32)
    return sim, df


def _atomic_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_unit(u):
    field, k0, k1 = u
    cfg, out = _W["cfg"], _W["out"]
    name = f"{field}__{k0:05d}.parquet"
    if (out / "_sims" / name).exists():
        return u, "skip"
    epochs = _W["log"].get(field)
    if not epochs:
        return u, "sin_log"
    mw = _W["mw"].get(field, cfg.get("mw_const", 0.02))
    sims, dfs = [], []
    for cls in cfg["classes"]:
        for k in range(k0, min(k1, cfg["n_by_class"][cls])):
            s, d = simulate(field, cls, k, epochs, mw)
            sims.append(s)
            if d is not None:
                dfs.append(d)
    if dfs:
        _atomic_parquet(pd.concat(dfs, ignore_index=True), out / name)
    _atomic_parquet(pd.DataFrame(sims), out / "_sims" / name)       # al final: marca la unidad como completa
    return u, f"{sum(s['status'] == 'ok' for s in sims)}/{len(sims)} ok"


def git_state():
    c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                       capture_output=True, text=True).stdout.strip()
    return c, bool(d)


def config_hash(cfg):
    filt = {n: md5_file(FILTERS / B.SURVEY_FILES[cfg["survey"]].format(n)) for n in cfg["bands"]}
    blob = json.dumps(dict(cfg=cfg, catalog=md5_file(STORE / "catalog.csv"), filters=filt), sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields-file", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    cfg = runcfg.RUNS_CFG[a.run]
    commit, dirty = git_state()
    if dirty and not a.allow_dirty:
        sys.exit("ERROR: el repo tiene cambios sin commit. Commitea (o --allow-dirty solo para pilotos).")
    out = Path(a.out).expanduser()
    man = out / "run_manifest.json"
    h = config_hash(cfg)
    if man.exists():
        old = json.loads(man.read_text())
        if old["config_hash"] != h or old["seed"] != a.seed:
            sys.exit(f"ERROR: {out} tiene otra configuracion o semilla. No se mezcla: usa otra carpeta.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps(dict(run=a.run, cfg=cfg, seed=a.seed, config_hash=h, git=commit, dirty=dirty,
                                       started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    fields = [l.strip() for l in open(a.fields_file) if l.strip()][: a.limit]
    us = runcfg.units(cfg, fields)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(cfg, a.seed, runcfg.log_path(cfg), fields, str(out))) as pool:
        for i, (u, st) in enumerate(pool.imap_unordered(run_unit, us), 1):
            print(f"[{i}/{len(us)}] {u[0]} {u[1]}-{u[2]} {st}  {time.time() - t0:.0f}s", flush=True)
    sims = pd.concat([pd.read_parquet(p) for p in sorted((out / "_sims").glob("*.parquet"))], ignore_index=True)
    sims.to_parquet(out / "_sims_all.parquet", index=False)
    print(sims.groupby(["sn_type", "status"]).size().to_string())
    return out


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_run.py`
Expected: cuatro `ok`.

- [ ] **Step 6: Commit**

```bash
git add pipeline78/project.py pipeline78/runcfg.py pipeline78/run.py tests/test_p78_run.py
git commit -m "pipeline78: runner paralelo, determinista y reanudable con tabla _sims completa"
```

### Task 8: Piloto ZTF y puerta G1

> **Enmienda 2026-10-02.** El reporte agrega un diagnóstico de la ventana pre-explosión por clase: fracción de sims con UL antes de la primera detección, separación entre el último UL y la primera detección, y cuántos UL caen entre la explosión estimada y la primera época del template (donde la SN real se vería). Con eso Mauricio decide en G1. Verificar antes que `real_val.parquet` tenga las columnas `label` y `M_peak_r`.

**Files:**
- Create: `pipeline78/pilot_report.py`
- Output: `paper2_ZTF/figures_templates/pipeline78_piloto_ztf/index.html`, que se enlaza sola en el atlas

**Interfaces:**
- Consumes: una carpeta de corrida, `OC/data/real_val.parquet` (columnas sn_name, label, z, M_peak_r)
- Produces: `report(run_dir, out_dir)`

- [ ] **Step 1: Piloto.** Son 50 campos y menos de 5 minutos. No necesita orden, porque es un piloto chico.

```bash
cd "$REPO" && $PY -m pipeline78.run --run ztf_v78 --out ~/thesis_runs/ztf_v78_piloto \
  --fields-file ~/thesis_store/ztf_fields_1000.txt --limit 50 --seed 20261002 --workers 4 --allow-dirty \
  | tee ~/thesis_runs/ztf_v78_piloto.log
```

Expected: 1500 sims. La gran mayoría `ok` (v2 tuvo 92 %). Anotar el tiempo total, que da la estimación para la Tarea 9.

- [ ] **Step 2: Implementar el reporte**

```python
# pipeline78/pilot_report.py
"""Diagnostico de un piloto: estado de las sims, M_peak observado contra la muestra real de validacion,
detecciones por banda y galeria de curvas. Escribe una pagina para el atlas."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.utils import DL_calculator
from pipeline78.paths import OC, PHD
from pipeline78.catalog import CLF_CLASS


def mu(z):
    return 5.0 * np.log10(DL_calculator(float(z)) * 1e6) - 5.0


def report(run_dir, out_dir):
    run_dir, out_dir = Path(run_dir).expanduser(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sims = pd.read_parquet(run_dir / "_sims_all.parquet")
    ph = pd.concat([pd.read_parquet(p) for p in run_dir.glob("*__*.parquet")], ignore_index=True)
    det = ph[ph.detected & (ph["filter"] == "r")]
    peak = det.groupby("sim_id").magnitud_proyectada.min().rename("m_peak_r").reset_index()
    s = sims.merge(peak, on="sim_id", how="left")
    s = s[(s.n_det_r >= 7) & s.m_peak_r.notna()].copy()
    s["M_obs_r"] = [m - mu(z) for m, z in zip(s.m_peak_r, s.z)]
    s["clase"] = s.sn_type.map(CLF_CLASS)
    real = pd.read_parquet(OC / "data/real_val.parquet")
    rows = []
    for c in ("Ia", "II", "Ibc"):
        a, b = s.loc[s.clase == c, "M_obs_r"], real.loc[real.label == c, "M_peak_r"].dropna()
        rows.append(dict(clase=c, n_sim=len(a), med_sim=a.median(), std_sim=a.std(), n_real=len(b),
                         med_real=b.median(), std_real=b.std(), delta_med=a.median() - b.median()))
    tab = pd.DataFrame(rows).round(3)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    sims.groupby(["sn_type", "status"]).size().unstack(fill_value=0).plot.bar(ax=ax[0], stacked=True)
    ax[0].set_title("status per projection class")
    for c, col in (("Ia", "C0"), ("II", "C2"), ("Ibc", "C3")):
        ax[1].hist(s.loc[s.clase == c, "M_obs_r"], bins=30, histtype="step", color=col, label=f"{c} sim", density=True)
        ax[1].hist(real.loc[real.label == c, "M_peak_r"].dropna(), bins=20, histtype="stepfilled", alpha=0.2,
                   color=col, label=f"{c} real", density=True)
    ax[1].invert_xaxis(); ax[1].set_xlabel("observed peak M_r"); ax[1].legend(fontsize=7)
    for c, col in (("Ia", "C0"), ("II", "C2"), ("Ibc", "C3")):
        ax[2].hist(sims.loc[sims.sn_type.map(CLF_CLASS) == c, "n_det_r"], bins=40, histtype="step", color=col, label=c)
    ax[2].set_xlabel("detections in r"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "resumen.png", dpi=110); plt.close(fig)
    pick = s.sample(min(12, len(s)), random_state=1).sim_id
    fig, axs = plt.subplots(3, 4, figsize=(13, 8), sharey=False)
    for ax_, sid in zip(axs.ravel(), pick):
        d = ph[ph.sim_id == sid]
        for b, col in (("g", "g"), ("r", "r")):
            x = d[(d["filter"] == b)]
            ax_.errorbar(x.mjd[x.detected], x.magnitud_proyectada[x.detected], x.magerr[x.detected], fmt="o", ms=2, color=col)
            ax_.plot(x.mjd[~x.detected], x.maglimit[~x.detected], "v", ms=2, color=col, alpha=0.4)
        r = sims[sims.sim_id == sid].iloc[0]
        ax_.set_title(f"{r.sn_type} {r.template} z={r.z:.3f}", fontsize=8); ax_.invert_yaxis()
    fig.tight_layout(); fig.savefig(out_dir / "galeria.png", dpi=100); plt.close(fig)
    (out_dir / "index.html").write_text(
        "<html><head><meta charset='utf-8'><title>Piloto ZTF v78</title></head><body style='font-family:sans-serif;max-width:1300px;margin:auto'>"
        f"<h1>Piloto de proyeccion ZTF con las 78 congeladas</h1><p>Corrida: {run_dir}</p>"
        f"<h2>M_peak observado en r: sims (n_det_r &ge; 7) contra la mitad de validacion real</h2>{tab.to_html(index=False)}"
        "<img src='resumen.png' width='100%'><h2>Galeria (g verde, r rojo; triangulos = limites)</h2>"
        "<img src='galeria.png' width='100%'></body></html>")
    return tab


if __name__ == "__main__":
    print(report(sys.argv[1], PHD / "paper2_ZTF/figures_templates/pipeline78_piloto_ztf").to_string(index=False))
```

- [ ] **Step 3: Generar el reporte y publicarlo en el atlas**

```bash
$PY -m pipeline78.pilot_report ~/thesis_runs/ztf_v78_piloto
zsh "$PHD/paper2_ZTF/Codes/spectral_series/run/sync_atlas_local.sh"
```

- [ ] **Step 4: Puerta G1 con Mauricio.** Se presenta la página y estos criterios:
  - (a) status `ok` por sobre el 85 % en cada clase;
  - (b) `delta_med` del M observado dentro de 0.3 mag en Ia, II e Ibc;
  - (c) `std_sim` de las Ia entre 0.4 y 0.7, porque la real robusta es 0.55;
  - (d) curvas de aspecto correcto en la galería, con límites antes de la subida y máximo dentro de la ventana.

  Si (b) o (c) fallan, la palanca es el polvo o la LF de esa clase en `config.py`. Se cambia, se usa un nombre de configuración nuevo y se repite el piloto. **No se pasa a la Tarea 9 sin el ok.**

- [ ] **Step 5: Commit**

```bash
git add pipeline78/pilot_report.py
git commit -m "pipeline78: reporte de piloto para el atlas"
```

### Task 9: Proyección ZTF completa

- [ ] **Step 1: ORDEN.** Pedirle a Mauricio la orden de lanzar. Darle el tiempo estimado, que es el del piloto multiplicado por 20.

- [ ] **Step 2: Repo limpio y lanzamiento**

```bash
cd "$REPO" && git status --short --untracked-files=no      # debe salir vacio
nohup $PY -m pipeline78.run --run ztf_v78 --out ~/thesis_runs/ztf_v78 \
  --fields-file ~/thesis_store/ztf_fields_1000.txt --seed 20261002 --workers 4 > ~/thesis_runs/ztf_v78.log 2>&1 &
```

- [ ] **Step 3: Verificar**

```bash
tail -20 ~/thesis_runs/ztf_v78.log
$PY -c "
import pandas as pd; s=pd.read_parquet('$HOME/thesis_runs/ztf_v78/_sims_all.parquet')
print(len(s), s.groupby(['sn_type','status']).size().to_dict())
print('con >=7 det en r:', (s.n_det_r>=7).sum())"
```

Expected: 30 000 sims (1000 campos × 30) y unas 15 000 con 7 o más detecciones en r.

- [ ] **Step 4: Espejo en Drive (copiar y verificar, sin mover)**

```bash
D="$PHD/paper2_ZTF/runs"; mkdir -p "$D"
cp -R ~/thesis_runs/ztf_v78 "$D/" && (cd ~/thesis_runs/ztf_v78 && find . -type f -exec md5 -r {} \; | sort) > /tmp/a.md5 \
 && (cd "$D/ztf_v78" && find . -type f -exec md5 -r {} \; | sort) > /tmp/b.md5 && diff /tmp/a.md5 /tmp/b.md5 && echo ESPEJO_OK
```

### Parte C: features y clasificador ZTF

### Task 10: Benchmark del MCMC y puerta G2

**Files:**
- Modify: `ZLF/config.py:50-56`
- Create: `pipeline78/compare_features.py`

**Interfaces:**
- Produces: `compare(csv_a, csv_b) -> DataFrame`, con una fila por parámetro y las columnas mediana y p90 de |Δ|/σ más la razón de tiempos

- [ ] **Step 1: Hacer configurable el MCMC.** En `ZLF/config.py` hay que agregar `import os` arriba y reemplazar el bloque:

```python
MCMC_CONFIG = {
    "n_walkers": int(os.environ.get("ZLF_MCMC_WALKERS", 100)),
    "n_steps": int(os.environ.get("ZLF_MCMC_STEPS", 5000)),
    "burn_in": int(os.environ.get("ZLF_MCMC_BURN", 500)),
    "n_threads": 1,
    "random_seed": 41,
}
```

- [ ] **Step 2: Implementar la comparación**

```python
# pipeline78/compare_features.py
"""Compara dos features.csv de run_parquet.py sobre las mismas tareas (benchmark del MCMC)."""
import sys
import numpy as np
import pandas as pd

KEYS = ["oid", "part_index", "sn_type", "filter_band"]
PARS = ["f", "t_rise", "t_fall", "gamma"]


def compare(csv_a, csv_b):
    a, b = pd.read_csv(csv_a), pd.read_csv(csv_b)
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    rows = []
    for p in PARS:
        sig = np.sqrt(m[f"{p}_err_a"] ** 2 + m[f"{p}_err_b"] ** 2).replace(0, np.nan)
        r = (m[f"{p}_a"] - m[f"{p}_b"]).abs() / sig
        rows.append(dict(par=p, n=len(m), mediana=r.median(), p90=r.quantile(0.9)))
    dA = (np.log10(m["A_a"]) - np.log10(m["A_b"])).abs()
    rows.append(dict(par="log10A", n=len(m), mediana=dA.median(), p90=dA.quantile(0.9)))
    rows.append(dict(par="tiempo_b/a", n=len(m), mediana=m.elapsed_s_b.median() / m.elapsed_s_a.median(), p90=np.nan))
    rows.append(dict(par="solo_en_a", n=len(a) - len(m), mediana=np.nan, p90=np.nan))
    rows.append(dict(par="solo_en_b", n=len(b) - len(m), mediana=np.nan, p90=np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(compare(sys.argv[1], sys.argv[2]).round(3).to_string(index=False))
```

- [ ] **Step 3: Correr el benchmark.** Son unos 30 minutos, hay que avisar. Las mismas 100 tareas salen con `--n_test 100 --seed 42`.

```bash
Z="$PHD/paper2_ZTF/Codes/feature_extraction/ztf_literature_features"
cd "$Z" && $PY run_parquet.py --parquet_dir ~/thesis_runs/ztf_v78 --output_dir ~/thesis_runs/bench_ref \
  --n_test 100 --seed 42 --filters g,r --workers 8
ZLF_MCMC_WALKERS=32 ZLF_MCMC_STEPS=1500 ZLF_MCMC_BURN=300 $PY run_parquet.py --parquet_dir ~/thesis_runs/ztf_v78 \
  --output_dir ~/thesis_runs/bench_fast --n_test 100 --seed 42 --filters g,r --workers 8
cd "$REPO" && $PY -m pipeline78.compare_features ~/thesis_runs/bench_ref/features/features.csv ~/thesis_runs/bench_fast/features/features.csv
```

**Criterio de aceptación:**
- mediana de |Δ|/σ menor que 0.3 y p90 menor que 1 en f, t_rise, t_fall y gamma;
- mediana de |Δ log A| menor que 0.02;
- `solo_en_a` y `solo_en_b` menores que el 3 %.

Si no se cumple, se repite con 50 walkers, 3000 pasos y 500 de burn-in.

- [ ] **Step 4: Puerta G2.** Mauricio elige la configuración y se anota en "Decisiones tomadas". Desde aquí, **todas** las features (reales y sintéticas, ZTF y SUDARE) se extraen con esa misma configuración.

- [ ] **Step 5: Commit en los dos repos**

```bash
cd "$Z" && git add config.py && git commit -m "MCMC configurable por variables de entorno (benchmark pipeline78)"
cd "$REPO" && git add pipeline78/compare_features.py && git commit -m "pipeline78: comparacion de features entre configuraciones MCMC"
```

### Task 11: Features reales de ZTF con el mismo código

**Files:**
- Create: `pipeline78/real_to_parquet.py`, `tests/test_p78_real.py`

**Interfaces:**
- Consumes: `OC/data/real_val.parquet`, `OC/data/real_final.parquet` (sn_name, label, z) y `paper2_ZTF/Photometry_ZTF_ST_Alerce/*/<sn>_photometry.dat`
- Produces:
  - `ztf_real_to_parquet(out_dir) -> DataFrame`, que escribe `<label>.parquet` con el esquema de proyección y además `meta_real_ztf.csv` (oid, part_index, sn_type, z, split)

- [ ] **Step 1: Escribir el test**

```python
# tests/test_p78_real.py
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline78.real_to_parquet import photometry_rows

def test_photometry_rows_schema():
    fd = {"g": pd.DataFrame({"MJD": [1.0, 2.0], "MAG": [19.0, 20.5], "MAGERR": [0.1, float("nan")], "Upperlimit": [False, True]})}
    r = photometry_rows("ZTFx", "Ia", fd)
    assert list(r.upperlimit) == ["F", "T"] and set(r.filter) == {"g"}
    assert {"oid", "part_index", "sn_type", "mjd", "magnitud_proyectada", "magerr"} <= set(r.columns)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlo y ver que falla**

Run: `$PY tests/test_p78_real.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

```python
# pipeline78/real_to_parquet.py
"""Fotometria real -> el mismo esquema que la proyeccion, para que real y sintetico pasen por el MISMO
run_parquet.py (misma cascada de reintentos, mismos filtros de calidad, mismo MCMC)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import OC, PHD, ZLF

sys.path.insert(0, str(ZLF))


def photometry_rows(sn, label, filters_data, bands=("g", "r")):
    out = []
    for b, d in filters_data.items():
        if b not in bands:
            continue
        out.append(pd.DataFrame({"oid": sn, "part_index": np.int32(0), "sn_type": label, "mjd": d["MJD"].astype(float),
                                 "filter": b, "magnitud_proyectada": d["MAG"].astype(float),
                                 "magerr": d["MAGERR"].astype(float), "upperlimit": np.where(d["Upperlimit"], "T", "F")}))
    return pd.concat(out, ignore_index=True) if out else None


def ztf_real_to_parquet(out_dir):
    from reader import parse_photometry_file
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.concat([pd.read_parquet(OC / "data/real_val.parquet").assign(split="val"),
                      pd.read_parquet(OC / "data/real_final.parquet").assign(split="final")], ignore_index=True)
    base = PHD / "paper2_ZTF/Photometry_ZTF_ST_Alerce"
    frames, missing = {}, []
    for r in meta.itertuples():
        hits = list(base.glob(f"*/{r.sn_name}_photometry.dat"))
        if not hits:
            missing.append(r.sn_name)
            continue
        fd, _ = parse_photometry_file(str(hits[0]))
        rows = photometry_rows(r.sn_name, r.label, fd)
        if rows is not None:
            frames.setdefault(r.label, []).append(rows)
    for label, fr in frames.items():
        pd.concat(fr, ignore_index=True).to_parquet(out_dir / f"{label}.parquet", index=False)
    m = meta.rename(columns={"sn_name": "oid", "label": "sn_type"})[["oid", "sn_type", "z", "split"]].assign(part_index=0)
    m.to_csv(out_dir / "meta_real_ztf.csv", index=False)
    print(f"{len(meta)} SNe, {len(missing)} sin archivo de fotometria: {missing[:10]}")
    return m


if __name__ == "__main__":
    ztf_real_to_parquet(Path.home() / "thesis_runs/real_ztf")
```

- [ ] **Step 4: Correr el test y convertir**

Run: `$PY tests/test_p78_real.py && $PY -m pipeline78.real_to_parquet`
Expected: `ok`, 675 SNe y una lista corta de faltantes, que hay que explicar.

- [ ] **Step 5: Extraer las features con la configuración de G2.** Tarda de 1 a 5 horas. Avisar antes de lanzar.

```bash
cd "$Z" && ZLF_MCMC_WALKERS=<G2> ZLF_MCMC_STEPS=<G2> ZLF_MCMC_BURN=<G2> nohup $PY run_parquet.py \
  --parquet_dir ~/thesis_runs/real_ztf --output_dir ~/thesis_runs/real_ztf_features --filters g,r --workers 8 \
  > ~/thesis_runs/real_ztf_features.log 2>&1 &
```

Los valores `<G2>` son los que quedaron anotados en la puerta G2. No es texto a completar a ciegas.

- [ ] **Step 6: Commit**

```bash
git add pipeline78/real_to_parquet.py tests/test_p78_real.py
git commit -m "pipeline78: fotometria real ZTF al esquema de proyeccion (mismo extractor que las sinteticas)"
```

### Task 12: Features ZTF completas

- [ ] **Step 1: ORDEN.** Pedirla con el tiempo estimado: tiempo por ajuste del benchmark × número de sims con 7 o más detecciones × 2, dividido por 8.

- [ ] **Step 2: Lanzar (reanudable)**

```bash
cd "$Z" && ZLF_MCMC_WALKERS=<G2> ZLF_MCMC_STEPS=<G2> ZLF_MCMC_BURN=<G2> nohup $PY run_parquet.py \
  --parquet_dir ~/thesis_runs/ztf_v78 --output_dir ~/thesis_runs/ztf_v78_features --filters g,r --workers 8 --resume \
  > ~/thesis_runs/ztf_v78_features.log 2>&1 &
```

Si se corta, se relanza la misma línea: `--resume` salta las tareas hechas.

- [ ] **Step 3: Verificar y espejar**

```bash
$PY -c "
import pandas as pd; f=pd.read_csv('$HOME/thesis_runs/ztf_v78_features/features/features.csv')
e=pd.read_csv('$HOME/thesis_runs/ztf_v78_features/features/errors.csv')
print(f.groupby(['sn_type','filter_band']).size().to_dict()); print(e.iloc[:,-1].str[:40].value_counts().head(8))"
```

Después, el espejo de la Tarea 9 Step 4 con `ztf_v78_features`.

### Task 13: Receta del clasificador con el merge arreglado y puerta G3

> **Enmienda 2026-10-02 (D1).** En ZTF, las sims `sn_type == "IIn"` quedan fuera del entrenamiento por defecto (la muestra real no tiene IIn). G3 decide si entran como II.

**Files:**
- Create: `pipeline78/recipe.py`, `tests/test_p78_recipe.py`

**Interfaces:**
- Consumes: features.csv (sintético y real), `_sims_all.parquet` (z y clase) y `meta_real_ztf.csv`
- Produces:
  - `build(features_csv, meta, b1, b2) -> DataFrame`, con una fila por (oid, part_index, sn_type) y las columnas f, t_fall, gamma, t_rise, color, M_peak, label
  - `fit_predict(train, test, seed, use_M=True) -> ndarray`
  - `evaluate(train, test, seeds, use_M, offset) -> dict`
  - `N1, N2`

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_p78_recipe.py
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pipeline78.recipe import build

def test_build_does_not_cross_types():
    rows = []
    for t in ("Ia", "II"):
        for b, A in (("r", 1e-8), ("g", 2e-8 if t == "Ia" else 5e-9)):
            rows.append(dict(oid="F", part_index=0, sn_type=t, filter_band=b, A=A, f=0.1, t0=0, t_rise=3, t_fall=30, gamma=20))
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "f.csv"; pd.DataFrame(rows).to_csv(p, index=False)
        meta = pd.DataFrame(dict(oid=["F", "F"], part_index=[0, 0], sn_type=["Ia", "II"], z=[0.05, 0.05]))
        m = build(p, meta, "r", "g")
        assert len(m) == 2
        assert np.isclose(m.set_index("sn_type").loc["Ia", "color"], -2.5 * np.log10(2.0))
        assert np.isclose(m.set_index("sn_type").loc["II", "color"], -2.5 * np.log10(0.5))

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlo y ver que falla**

Run: `$PY tests/test_p78_recipe.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implementar.** Es la receta congelada de `05_faseC_run1000_v2.py` con el merge arreglado y las bandas como parámetro.

```python
# pipeline78/recipe.py
"""Receta jerarquica congelada: N1 (II contra I) con forma; N2 (Ia contra Ibc) con forma + color + t_rise + M.
Arreglo H4: r y la segunda banda se unen por (oid, part_index, sn_type), validado uno a uno."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core.utils import DL_calculator
from pipeline78.catalog import CLF_CLASS

KEYS = ["oid", "part_index", "sn_type"]
N1 = ["f", "t_fall", "gamma"]
N2 = N1 + ["color", "t_rise", "M_peak"]


def _mu(z):
    return 5.0 * np.log10(DL_calculator(float(z)) * 1e6) - 5.0


def build(features_csv, meta, b1="r", b2="g"):
    f = pd.read_csv(features_csv)
    band = f["filter_band"].astype(str).str.lower().str[-1]
    one = f[band == b1][KEYS + ["A", "f", "t_rise", "t_fall", "gamma"]]
    two = f[band == b2][KEYS + ["A"]].rename(columns={"A": "A2"})
    m = one.merge(two, on=KEYS, how="left", validate="one_to_one")
    m = m.merge(meta[KEYS + ["z"]].drop_duplicates(KEYS), on=KEYS, how="left", validate="many_to_one")
    m["color"] = np.where((m.A > 0) & (m.A2 > 0), -2.5 * np.log10(m.A2 / m.A), np.nan)
    mp = np.where(m.A > 0, -2.5 * np.log10(m.A), np.nan)
    m["M_peak"] = [x - _mu(z) if np.isfinite(x) and z > 0 else np.nan for x, z in zip(mp, m.z)]
    m["label"] = m.sn_type.map(CLF_CLASS).fillna(m.sn_type)
    return m


def _mlp(seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800, random_state=seed))


def fit_predict(train, test, seed, use_M=True):
    n2 = N2 if use_M else [c for c in N2 if c != "M_peak"]
    m1 = _mlp(seed).fit(train[N1], np.where(train.label == "II", "II", "I"))
    p1 = m1.predict(test[N1])
    tI = train[train.label != "II"]
    if use_M:
        tI = tI.dropna(subset=["M_peak"])
    p2 = _mlp(seed).fit(tI[n2], tI.label).predict(test[n2])
    return np.where(p1 == "II", "II", p2)


def offset_by_type(train, ref):
    """Corrimiento de M por tipo: mediana sintetica menos mediana real (SOLO con la mitad de validacion)."""
    return float(np.mean([train.loc[train.label == c, "M_peak"].median() - ref.loc[ref.label == c, "M_peak"].median()
                          for c in ("Ia", "II", "Ibc")]))


def evaluate(train, test, seeds=range(10), use_M=True, offset=0.0):
    tr = train.copy()
    tr["M_peak"] = tr["M_peak"] - offset
    out = []
    for s in seeds:
        p = fit_predict(tr, test, s, use_M)
        rec = {c: float((p[test.label.values == c] == c).mean()) for c in ("Ia", "II", "Ibc")}
        out.append(dict(rec, acc=float((p == test.label.values).mean())))
    d = pd.DataFrame(out)
    return {k: (round(d[k].mean(), 3), round(d[k].std(), 3)) for k in d.columns}
```

- [ ] **Step 4: Correr el test y ver que pasa**

Run: `$PY tests/test_p78_recipe.py`
Expected: `ok test_build_does_not_cross_types`

- [ ] **Step 5: Medir con la mitad de validación.** Aquí se toman todas las decisiones, por ejemplo usar o no el offset de M.

```bash
$PY - <<'EOF'
import pandas as pd
from pathlib import Path
from pipeline78.recipe import build, evaluate, offset_by_type
H = Path.home() / "thesis_runs"
sims = pd.read_parquet(H / "ztf_v78/_sims_all.parquet").rename(columns={"field": "oid"})
syn = build(H / "ztf_v78_features/features/features.csv", sims, "r", "g")
meta = pd.read_csv(H / "real_ztf/meta_real_ztf.csv")
real = build(H / "real_ztf_features/features/features.csv", meta, "r", "g").merge(meta[["oid", "split"]], on="oid")
val = real[real.split == "val"]
off = offset_by_type(syn, val)
print("sinteticas", syn.label.value_counts().to_dict(), "offset M (val):", round(off, 3))
print("VAL sin offset:", evaluate(syn, val, offset=0.0))
print("VAL con offset:", evaluate(syn, val, offset=off))
print("VAL sin M     :", evaluate(syn, val, use_M=False))
EOF
```

- [ ] **Step 6: Puerta G3.** Mauricio elige la variante viendo solo la validación. Después se mide la final **una sola vez** con esa variante y se guarda:

```bash
$PY - <<'EOF'
import json, pandas as pd
from pathlib import Path
from pipeline78.recipe import build, evaluate, offset_by_type
H = Path.home() / "thesis_runs"
sims = pd.read_parquet(H / "ztf_v78/_sims_all.parquet").rename(columns={"field": "oid"})
syn = build(H / "ztf_v78_features/features/features.csv", sims, "r", "g")
meta = pd.read_csv(H / "real_ztf/meta_real_ztf.csv")
real = build(H / "real_ztf_features/features/features.csv", meta, "r", "g").merge(meta[["oid", "split"]], on="oid")
val, fin = real[real.split == "val"], real[real.split == "final"]
USE_OFFSET = True            # <- la variante elegida en G3
off = offset_by_type(syn, val) if USE_OFFSET else 0.0
res = dict(val=evaluate(syn, val, offset=off), final=evaluate(syn, fin, offset=off), offset=off,
           n_train=syn.label.value_counts().to_dict(), n_val=len(val), n_final=len(fin))
(H / "ztf_v78_resultados.json").write_text(json.dumps(res, indent=1)); print(res)
EOF
```

`USE_OFFSET` se fija con lo elegido en G3 y se anota en "Decisiones tomadas".

- [ ] **Step 7: Commit**

```bash
git add pipeline78/recipe.py tests/test_p78_recipe.py
git commit -m "pipeline78: receta del clasificador con merge por tipo (arregla el cruce r/g)"
```

### Parte D: SUDARE

### Task 14: Log de SUDARE, épocas template y E(B−V) de los campos

> **Enmienda 2026-10-02 (tasas).** `sudare_fields.csv` lleva `area_deg2` por campo-temporada. Las teselas de CDFS se solapan (SUDARE I: 1.15 deg² por pointing, 2.05 deg² entre las dos de CDFS); las temporadas de COSMOS repiten el mismo área. Verificar en el texto de SUDARE I (ADS) y confirmar si cdfs3 y cdfs4 son pointings nuevos o temporadas de cdfs1/cdfs2 antes de fijar los valores.

**Files:**
- Modify: `pipeline78/survey.py` (ya tiene `build_sudare_log`)
- Create: `pipeline78/sudare_calib.py`

**Interfaces:**
- Produces:
  - `ref_epochs_from_curves() -> set[(field, band, mjd_day)]`
  - `field_centers() -> DataFrame`
  - `build_sudare_inputs()`, que escribe `STORE/sudare_obslog.parquet` y `STORE/sudare_fields.csv` (field, ra, dec, ebmv_mw, window_days)

- [ ] **Step 1: Implementar**

```python
# pipeline78/sudare_calib.py
"""Insumos de SUDARE: epocas template, centros de campo, E(B-V) MW, ventana de control time,
calibracion del ruido y cobertura por z."""
import numpy as np
import pandas as pd
from pipeline78.paths import SUDARE_DIR, STORE
from pipeline78.survey import build_sudare_log, SUDARE_SEASON_SPLIT


def _season(field, mjd):
    if field != "cosmos":
        return field
    return next(name for cut, name in SUDARE_SEASON_SPLIT if mjd < cut)


def ref_epochs_from_curves(min_frac=0.5):
    """Una epoca es template si al menos min_frac de los candidatos medidos en ella la marcan REF."""
    lc = pd.read_csv(SUDARE_DIR / "curvas_sudare_parseadas.csv", dtype={"fit": str})
    lc["field"] = [_season(t.split("_")[0], m) for t, m in zip(lc.trid, lc.mjd)]
    lc["day"] = np.floor(lc.mjd).astype(int)
    g = lc.groupby(["field", "band", "day"]).fit.apply(lambda s: (s == "REF").mean())
    return {k for k, v in g.items() if v >= min_frac}


def field_centers():
    w = pd.read_csv(SUDARE_DIR / "web_master_sn_psn.csv", dtype=str)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(w.ra.values, w.dec.values, unit=(u.hourangle, u.deg))
    w["ra_deg"], w["dec_deg"] = c.ra.deg, c.dec.deg
    return w.groupby("campo")[["ra_deg", "dec_deg"]].median()


def build_sudare_inputs():
    log = build_sudare_log(STORE / "sudare_obslog.parquet", ref_epochs_from_curves())
    cen = field_centers()
    from tools.dust_maps import get_sfd98_extinction_real
    rows = []
    for f, g in log[~log.is_ref].groupby("field"):
        base = "cosmos" if f.startswith("cosmos") else f
        ra, dec = cen.loc[base, "ra_deg"], cen.loc[base, "dec_deg"]
        ebv, ok = get_sfd98_extinction_real(ra, dec)
        rows.append(dict(field=f, ra=ra, dec=dec, ebmv_mw=float(ebv) if ok else np.nan, sfd_ok=bool(ok),
                         t_first=g.mjd.min(), t_last=g.mjd.max(), window_days=g.mjd.max() - g.mjd.min() + 365.0))
    d = pd.DataFrame(rows)
    d.to_csv(STORE / "sudare_fields.csv", index=False)
    print(d.to_string(index=False)); print("epocas template excluidas:", int(log.is_ref.sum()))
    return d


if __name__ == "__main__":
    build_sudare_inputs()
```

- [ ] **Step 2: Correr y revisar**

Run: `cd "$REPO" && $PY -m pipeline78.sudare_calib`
Expected: 7 filas, con E(B−V) de CDFS cerca de 0.01 y de COSMOS cerca de 0.02, y ventanas de unos 520 a 870 días. Si `sfd_ok` es falso (sin red), mostrárselo a Mauricio antes de seguir.

- [ ] **Step 3: Commit**

```bash
git add pipeline78/sudare_calib.py
git commit -m "pipeline78: insumos SUDARE (epocas template, E(B-V) de campo, ventana de control time)"
```

### Task 15: Calibración del ruido de SUDARE con las curvas reales

**Files:**
- Modify: `pipeline78/sudare_calib.py`, agregando `noise_k()`

**Interfaces:**
- Produces: `noise_k() -> dict band -> k`, con σ_mag = 1.0857 / (k·10^{0.4(mlim − m)}), para la `RUNS_CFG["sudare_v78"]["noise_k"]` de la Tarea 18

- [ ] **Step 1: Implementar**

```python
def noise_k():
    """k tal que S/N = k 10^(0.4 (m50 - m)), medido en las detecciones reales (fitmag numerico, SNR>=3)."""
    lc = pd.read_csv(SUDARE_DIR / "curvas_sudare_parseadas.csv", dtype={"fit": str, "efit": str})
    ok = lc.fit.str.match(r"^-?\d") & lc.efit.str.match(r"^\d")
    d = lc[ok].copy()
    d["fit"], d["efit"] = d.fit.astype(float), d.efit.astype(float)
    d = d[(d.fit > 0) & (d.efit > 0) & (d.snr >= 3)]
    d["k"] = 1.0857 / (d.efit * 10 ** (0.4 * (d.mlim - d.fit)))
    out = d.groupby("band").k.agg(["median", lambda s: s.quantile(0.16), lambda s: s.quantile(0.84), "size"])
    print(out)
    return out["median"].to_dict()
```

- [ ] **Step 2: Correr**

Run: `$PY -c "from pipeline78.sudare_calib import noise_k; print(noise_k())"`
Expected: un k por banda, de unos pocos (m50 está al 50 % de eficiencia, no exactamente a 5σ). Anotarlo para la Tarea 18.

- [ ] **Step 3: Commit**

```bash
git add pipeline78/sudare_calib.py && git commit -m "pipeline78: ruido de SUDARE calibrado con las detecciones reales"
```

### Task 16: Cobertura por banda contra z y puerta G4 (D7)

**Files:**
- Modify: `pipeline78/sudare_calib.py`, agregando `coverage_table()`

- [ ] **Step 1: Implementar**

```python
def coverage_table(zs=np.arange(0.05, 1.01, 0.05)):
    """Por clase: fraccion de templates con cobertura > 0.95 en OmegaCAM g, r, i a cada z."""
    from pipeline78.bands import survey_bands, synphot
    from pipeline78.store import load_template
    cat = pd.read_csv(STORE / "catalog.csv")
    bands = survey_bands("SUDARE")
    rows = []
    for c, g in cat.groupby("clase"):
        tpls = [load_template(p) for p in g.store_path]
        for z in zs:
            for b in bands:
                ok = np.mean([synphot(t["wave"] * (1 + z), np.ones((1, t["wave"].size)), b)[1] > 0.95 for t in tpls])
                rows.append(dict(clase=c, z=round(float(z), 2), band=b.name, frac_cubierta=ok))
    tab = pd.DataFrame(rows).pivot_table(index=["clase", "z"], columns="band", values="frac_cubierta")
    tab.to_csv(STORE / "sudare_coverage.csv")
    return tab
```

- [ ] **Step 2: Correr y llevarlo a la puerta G4**

Run: `$PY -c "from pipeline78.sudare_calib import coverage_table; print(coverage_table().to_string())"`
Expected: g se pierde desde z ≈ 0.3, r se mantiene hasta z ≈ 0.8 e i hasta 1.0. Con esta tabla y la distribución de z de la muestra (mediana 0.38), Mauricio cierra D7.

### Task 17: Luminosidad y polvo de las IIn (literatura verificada)

> **Enmienda 2026-10-02.** Ya está en `config.py`: IIn M_r = −19.18 ± 1.32, Nyholm+2020 (2020A&A...637A..73N, verificado en ADS; banda r con K-corr; muestra PTF/iPTF limitada en magnitud, incluye IIn "superluminosas"). Sirve para el training de ZTF. Para D4 y las tasas falta una LF **intrínseca** (corregida por polvo y por Malmquist): revisar Kiewe+2012 (CCCP) y Richardson+2014 en el texto. Revisar también que la banda de cada LF calce con `REF_BAND` (II −16.9 e Ibc −17.3 vienen de calibraciones en B o V).

**Files:**
- Modify: `config.py` (`LUMINOSITY_CONFIG["M_peak"]`, y `EXTINCTION_CONFIG` si corresponde), `pipeline78/catalog.py` (`REF_BAND["IIn"]`), `pipeline78/sampling.py` (`EXT_KEY["IIn"]`)

- [ ] **Step 1: Buscar en ADS.** Hay que encontrar una distribución de M de máximo de IIn con media, dispersión y banda. Candidatos a revisar:
  - Richardson et al. 2014, 2014AJ....147..118R, ya está en la bib de la tesis y tiene M_B por tipo;
  - Kiewe et al. 2012;
  - Nyholm et al. 2020.

  Abrir el paper y anotar el bibcode verificado, la tabla, la página, la banda y si está corregido por extinción. **No se usa un número que no se haya leído en el paper.**

- [ ] **Step 2: Escribirlo.** En `config.py` va `"IIn": {"mean": <leido>, "sigma": <leido>}` con el bibcode en el comentario. `REF_BAND["IIn"]` toma la banda de esa calibración (`B_rest` si es M_B). Para el polvo, si no hay una medición de IIn, `EXT_KEY["IIn"] = "SNII"`, declarado en el comentario.

- [ ] **Step 3: Rehacer el catálogo y pasar los tests**

Run: `$PY -m pipeline78.catalog > /dev/null && $PY tests/test_p78_sampling.py && $PY tests/test_p78_catalog.py`

- [ ] **Step 4: Commit**

```bash
git add config.py pipeline78/catalog.py pipeline78/sampling.py
git commit -m "pipeline78: LF y polvo de IIn desde literatura verificada en ADS"
```

### Task 18: Proyección SUDARE (volumétrica, regla de detección de SUDARE)

> **Enmienda 2026-10-02 (tasas).** Antes de esta tarea, el catálogo lleva un `subtype` por template (Ia normal/91T/91bg/02cx; II IIP/IIL/87A/05cs; Ibc Ib/Ic/Ic-BL; ramas de IIn) y una tabla de fracciones intrínsecas por subtipo (Li+2011b, la mezcla que usó SUDARE I, verificada en el paper). Con 3000 sims por `sn_type`, sin pesos, IIn es el 25 % de CC e infla la η.

**Files:**
- Modify: `pipeline78/project.py` (regla `sudare`), `pipeline78/runcfg.py` (`sudare_v78_piloto`, `sudare_v78`), `tests/test_p78_run.py` (test de la regla)

**Interfaces:**
- Produces: corridas con `found` (búsqueda en r con DE de la ec. 1) y `detected` (SNR ≥ `snr_det`) separados

- [ ] **Step 1: Escribir el test de la regla**

```python
def test_sudare_rule_flags():
    import numpy as np
    from pipeline78.project import project_one
    cfg = dict(bands=["r", "i"], pre_ul_days=25.0, rule="sudare", noise_k={"r": 5.0, "i": 5.0}, snr_det=3.0,
               search_band="r", de_max=0.95, de_beta=4.0)
    t_rel = np.arange(-10.0, 60.0); mags = {"r": np.full(70, 21.0), "i": np.full(70, 21.0)}
    ep = {"r": (np.arange(-40.0, 60.0, 3.0), np.full(34, 24.0)), "i": (np.arange(-40.0, 60.0, 3.0), np.full(34, 24.0))}
    df = project_one(t_rel, mags, ep, 0.0, np.random.default_rng(0), cfg)
    pre = df[df.mjd < -10.0]
    assert (pre.upperlimit == "T").all() and not pre.found.any()
    post_r = df[(df.mjd >= -10.0) & (df["filter"] == "r")]
    assert post_r.detected.all() and post_r.found.mean() > 0.8
    assert not df[df["filter"] == "i"].found.any()
```

- [ ] **Step 2: Correrlo y ver que falla**

Run: `$PY tests/test_p78_run.py`
Expected: `ValueError: sudare` en `test_sudare_rule_flags`

- [ ] **Step 3: Implementar la regla.** En `project_one` se reemplaza el `else: raise ValueError(cfg["rule"])` por:

```python
        elif cfg["rule"] == "sudare":
            # fotometria forzada en flujo, ruido limitado por cielo calibrado (Task 15)
            fl = np.where(mj < t0, 0.0, 10.0 ** (-0.4 * mm))
            sigf = 10.0 ** (-0.4 * ml) / cfg["noise_k"][b]
            fobs = fl + rng.normal(0.0, sigf)
            snr = fobs / sigf
            det = snr >= cfg["snr_det"]                                      # D9: punto para features
            sig = 1.0857 / np.maximum(snr, 1e-6)
            mobs = -2.5 * np.log10(np.clip(fobs, 1e-300, None))
            u = rng.random(mj.size)                                          # siempre se sortea: determinismo
            de = cfg["de_max"] * (np.arctan(cfg["de_beta"] * (ml - mm)) / np.pi + 0.5)   # SUDARE I ec. 1
            found = (u < de) if b == cfg["search_band"] else np.zeros(mj.size, bool)
        else:
            raise ValueError(cfg["rule"])
```

En `runcfg.RUNS_CFG` se agregan las dos configuraciones (k medido en la Tarea 15, bandas de D7):

```python
    "sudare_v78_piloto": dict(
        survey="SUDARE", bands=["g", "r", "i"], classes=["Ia", "II", "IIb", "IIn", "Ibc"],
        n_by_class={"Ia": 100, "II": 100, "IIb": 100, "IIn": 100, "Ibc": 100}, chunk=100,
        anchor="uniform_window", window_pre=365.0, z_mode="volumetric", zmin=0.05, zmax=1.0,
        mw_mode="sudare_fields", mw_const=0.02, rule="sudare", pre_ul_days=25.0,
        noise_k={"g": K_G, "r": K_R, "i": K_I}, snr_det=3.0, search_band="r", de_max=0.95, de_beta=4.0),
```

`K_G`, `K_R` y `K_I` se definen arriba del dict como constantes con los valores medidos en la Tarea 15. `sudare_v78` es igual a la piloto, con `n_by_class` en 3000 por clase y `chunk=500`.

- [ ] **Step 4: Tests y piloto**

```bash
$PY tests/test_p78_run.py
printf "cdfs1\ncdfs2\ncdfs3\ncdfs4\ncosmos1\ncosmos2\ncosmos3\n" > ~/thesis_store/sudare_fields.txt
$PY -m pipeline78.run --run sudare_v78_piloto --out ~/thesis_runs/sudare_v78_piloto \
  --fields-file ~/thesis_store/sudare_fields.txt --seed 20261003 --workers 4 --allow-dirty
```

En el piloto se revisan la fracción `found` contra z por clase y la galería de curvas. Mauricio lo aprueba.

- [ ] **Step 5: ORDEN y corrida completa.** Son 105 000 sims y cerca de una hora. Se lanza desde el repo limpio, igual que en la Tarea 9. Después, el espejo en Drive.

```bash
nohup $PY -m pipeline78.run --run sudare_v78 --out ~/thesis_runs/sudare_v78 \
  --fields-file ~/thesis_store/sudare_fields.txt --seed 20261003 --workers 4 > ~/thesis_runs/sudare_v78.log 2>&1 &
```

- [ ] **Step 6: Commit**

```bash
git add pipeline78/project.py pipeline78/runcfg.py tests/test_p78_run.py
git commit -m "pipeline78: regla de deteccion SUDARE (busqueda DE en r, fotometria SNR>=3) y corridas SUDARE"
```

### Task 19: Fotometría real de SUDARE al mismo esquema

**Files:**
- Modify: `pipeline78/real_to_parquet.py`, agregando `sudare_real_to_parquet(out_dir)`

- [ ] **Step 1: Implementar**

```python
def sudare_real_to_parquet(out_dir, snr_det=3.0, bands=("g", "r", "i")):
    """Misma regla que las sinteticas (D9): SNR>=snr_det es punto, lo demas limite a mlim. REF y fit=0 fuera."""
    from pipeline78.paths import SUDARE_DIR
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    lc = pd.read_csv(SUDARE_DIR / "curvas_sudare_parseadas.csv", dtype={"fit": str, "efit": str})
    sm = pd.read_csv(SUDARE_DIR / "sudare_muestra_prueba_sn.csv")
    sm = sm[sm.usable]
    lab = {"Ia": "Ia", "II": "II", "IIn": "II", "Ibc": "Ibc"}
    lc = lc[lc.trid.isin(sm.trid) & (lc.fit != "REF") & lc.band.isin(bands)].copy()
    num = lc.fit.str.match(r"^-?\d")
    fit = pd.to_numeric(lc.fit, errors="coerce")
    det = num & (fit > 0) & (lc.snr >= snr_det)
    t2 = dict(zip(sm.trid, sm.tipo_sudare))
    out = pd.DataFrame({"oid": lc.trid, "part_index": np.int32(0), "sn_type": lc.trid.map(t2).map(lab),
                        "mjd": lc.mjd.astype(float), "filter": lc.band,
                        "magnitud_proyectada": np.where(det, fit, lc.mlim),
                        "magerr": np.where(det, pd.to_numeric(lc.efit, errors="coerce"), np.nan),
                        "upperlimit": np.where(det, "F", "T")})
    out.to_parquet(out_dir / "sudare_real.parquet", index=False)
    sm.rename(columns={"trid": "oid"}).assign(part_index=0, sn_type=sm.tipo_sudare.map(lab))[
        ["oid", "part_index", "sn_type", "z", "tipo_sudare"]].to_csv(out_dir / "meta_real_sudare.csv", index=False)
    print(out.groupby("sn_type").oid.nunique())
```

- [ ] **Step 2: Correr**

Run: `$PY -c "from pipeline78.real_to_parquet import sudare_real_to_parquet as f; f('~/thesis_runs/real_sudare')"`
Expected: 169 SNe (Ia 105, II 47 contando las IIn, Ibc 17).

- [ ] **Step 3: Commit**

```bash
git add pipeline78/real_to_parquet.py && git commit -m "pipeline78: fotometria real SUDARE con la misma regla de deteccion que las sinteticas"
```

### Task 20: Features de SUDARE (submuestra estratificada más reales)

**Files:**
- Create: `pipeline78/subsample.py`

- [ ] **Step 1: Implementar**

```python
# pipeline78/subsample.py
"""Submuestra para features: sims encontradas con >= min_det puntos en la banda principal,
hasta n_per_class por clase de proyeccion. La eficiencia usa TODAS las sims (_sims_all)."""
import sys
from pathlib import Path
import pandas as pd


def subsample(run_dir, out_dir, band="r", min_det=7, n_per_class=4000, seed=1):
    run_dir, out_dir = Path(run_dir).expanduser(), Path(out_dir).expanduser()
    s = pd.read_parquet(run_dir / "_sims_all.parquet")
    s = s[s.found & (s[f"n_det_{band}"] >= min_det)]
    pick = s.groupby("sn_type", group_keys=False).apply(lambda g: g.sample(min(len(g), n_per_class), random_state=seed))
    ids = set(pick.sim_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dir.glob("*__*.parquet"):
        d = pd.read_parquet(p)
        d = d[d.sim_id.isin(ids)]
        if len(d):
            d.to_parquet(out_dir / p.name, index=False)
    pick.to_parquet(out_dir / "_picked.parquet", index=False)
    print(pick.sn_type.value_counts().to_dict())


if __name__ == "__main__":
    subsample(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Submuestra, ORDEN y features.** Son unas 10 horas con la configuración reducida. Las bandas son las de D7.

```bash
$PY -m pipeline78.subsample ~/thesis_runs/sudare_v78 ~/thesis_runs/sudare_v78_sub
cd "$Z" && ZLF_MCMC_WALKERS=<G2> ZLF_MCMC_STEPS=<G2> ZLF_MCMC_BURN=<G2> nohup $PY run_parquet.py \
  --parquet_dir ~/thesis_runs/sudare_v78_sub --output_dir ~/thesis_runs/sudare_v78_features --filters r,i --workers 8 --resume \
  > ~/thesis_runs/sudare_v78_features.log 2>&1 &
# al terminar, las reales:
ZLF_MCMC_WALKERS=<G2> ZLF_MCMC_STEPS=<G2> ZLF_MCMC_BURN=<G2> $PY run_parquet.py --parquet_dir ~/thesis_runs/real_sudare \
  --output_dir ~/thesis_runs/real_sudare_features --filters r,i --workers 8
```

- [ ] **Step 3: Commit**

```bash
cd "$REPO" && git add pipeline78/subsample.py && git commit -m "pipeline78: submuestra estratificada para features SUDARE"
```

### Task 21: Clasificación de SUDARE, con z y sin z (puerta G5)

- [ ] **Step 1: Medir.** Es la misma receta de la Tarea 13 con las bandas de D7.

```bash
$PY - <<'EOF'
import json, pandas as pd
from pathlib import Path
from pipeline78.recipe import build, evaluate
H = Path.home() / "thesis_runs"
sims = pd.read_parquet(H / "sudare_v78/_sims_all.parquet").rename(columns={"field": "oid"})
syn = build(H / "sudare_v78_features/features/features.csv", sims, "r", "i")
meta = pd.read_csv(H / "real_sudare/meta_real_sudare.csv")
real = build(H / "real_sudare_features/features/features.csv", meta, "r", "i")
res = dict(con_z=evaluate(syn, real, use_M=True), sin_z=evaluate(syn, real, use_M=False),
           n_train=syn.label.value_counts().to_dict(), n_real=real.label.value_counts().to_dict(),
           n_real_sin_features=int(len(meta) - real.oid.nunique()))
(H / "sudare_v78_resultados.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
EOF
```

- [ ] **Step 2: Subconjunto con espectro.** Son las publicadas con nombre IAU en la columna 4 de `master_referee.list`. Las 9 de COSMOS renombradas en la base nueva no calzan por nombre y quedan fuera de este subconjunto, declarado.

```bash
$PY - <<'EOF'
import numpy as np, pandas as pd
from pathlib import Path
from pipeline78.paths import SUDARE_DIR
from pipeline78.recipe import build, fit_predict
H = Path.home() / "thesis_runs"
spec = {l.split()[0] for l in open(SUDARE_DIR / "master_referee.list")
        if l.strip() and not l.startswith("#") and l.split()[3] != "NULL"}
sims = pd.read_parquet(H / "sudare_v78/_sims_all.parquet").rename(columns={"field": "oid"})
syn = build(H / "sudare_v78_features/features/features.csv", sims, "r", "i")
meta = pd.read_csv(H / "real_sudare/meta_real_sudare.csv")
sub = build(H / "real_sudare_features/features/features.csv", meta, "r", "i").query("oid in @spec")
p = np.array([fit_predict(syn, sub, s, use_M=True) for s in range(10)])
acc = (p == sub.label.values).mean(axis=1)
print(f"con espectro: n={len(sub)}  acierto {acc.mean():.2f} +- {acc.std():.2f}")
print(pd.DataFrame({"oid": sub.oid.values, "tipo": sub.label.values, "pred_seed0": p[0]}).to_string(index=False))
EOF
```

- [ ] **Step 3: Puerta G5.** Mauricio revisa:
  - el acuerdo con la herramienta de SUDARE, que no es verdad espectroscópica, y aparte con el subconjunto con espectro;
  - cuánto se pierde sin z.

  Se declara que no hubo mitad de validación separada, salvo que algo se ajuste mirando SUDARE. En ese caso se parte en dos como en ZTF.

### Parte E: tasas

### Task 22: Módulo de tasas y prueba de cierre

> **Enmienda 2026-10-02 (tasas).** `response_matrix` pondera cada sim por `w = fracción intrínseca de su subtipo / fracción en la corrida` (Tarea 18) y usa `area_deg2` por campo de `sudare_fields.csv` (Tarea 14) en vez de un área única. El (1+z) del control time va por sim, no por el centro del bin. Agregar un test de cierre con dos subtipos de distinta eficiencia y pesos distintos de 1.

**Files:**
- Create: `pipeline78/rates.py`, `tests/test_p78_rates.py`

**Interfaces:**
- Consumes:
  - `_sims_all.parquet`: sim_id, field, clf_class, z, found, n_det_r;
  - predicciones de la submuestra: DataFrame sim_id, pred, donde pred ∈ {Ia, II, Ibc, none} y "none" significa que el sim fue a features y no salió;
  - las encontradas con menos de `min_det` puntos no fueron a features y cuentan como "none";
  - `sudare_fields.csv`;
  - el área por pointing.
- Produces:
  - `shell_volume(z1, z2, area_deg2) -> Mpc³`
  - `response_matrix(sims, preds, fields, zbins, classes, area_deg2, min_det=7) -> dict zbin -> DataFrame [pred × true]` en yr·Mpc³
  - `estimate(A, n_obs, n_boot=2000, seed=1) -> DataFrame` con rate, lo y hi por clase

- [ ] **Step 1: Escribir los tests**

```python
# tests/test_p78_rates.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pipeline78.rates import shell_volume, response_matrix, estimate

def _toy(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    sims = pd.DataFrame(dict(sim_id=np.arange(n), field="F1", clf_class=rng.choice(["Ia", "II"], n),
                             z=rng.uniform(0.2, 0.4, n), found=rng.random(n) < 0.5,
                             n_det_r=rng.choice([3, 10], n, p=[0.2, 0.8])))
    preds = sims.loc[sims.found & (sims.n_det_r >= 7), ["sim_id", "clf_class"]].rename(columns={"clf_class": "pred"})
    fields = pd.DataFrame(dict(field=["F1"], window_days=[365.25]))
    return sims, preds, fields

def test_shell_volume_positive_and_scales_with_area():
    v1, v2 = shell_volume(0.2, 0.4, 1.0), shell_volume(0.2, 0.4, 2.0)
    assert v1 > 0 and abs(v2 / v1 - 2.0) < 1e-9

def test_few_detections_count_as_unclassified():
    sims, preds, fields = _toy()
    A = response_matrix(sims, preds, fields, [(0.2, 0.4)], ["Ia", "II"], area_deg2=1.15)[(0.2, 0.4)]
    full = response_matrix(sims.assign(n_det_r=10), sims.loc[sims.found, ["sim_id", "clf_class"]].rename(
        columns={"clf_class": "pred"}), fields, [(0.2, 0.4)], ["Ia", "II"], area_deg2=1.15)[(0.2, 0.4)]
    assert abs(A.loc["Ia", "Ia"] / full.loc["Ia", "Ia"] - 0.8) < 0.03

def test_closure_exact():
    sims, preds, fields = _toy()
    A = response_matrix(sims, preds, fields, [(0.2, 0.4)], ["Ia", "II"], area_deg2=1.15)[(0.2, 0.4)]
    true = np.array([2e-5, 5e-5])
    n_obs = pd.Series(A.values @ true, index=A.index)
    r = estimate(A, n_obs, n_boot=0)
    assert np.allclose(r.rate.values, true, rtol=1e-6)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
```

- [ ] **Step 2: Correrlos y ver que fallan**

Run: `$PY tests/test_p78_rates.py`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

```python
# pipeline78/rates.py
"""Tasas volumetricas por control time Monte Carlo (SUDARE I ec. 3 y 5), con correccion de clasificacion.

Por bin de z, campo f, clase verdadera c y clase predicha p:
    CT_f(c->p) = T_f * eta_found(c, f, z) * P(p | c, found, z)          [dias observados]
    A[p, c]    = sum_f V_f(z) * CT_f(c->p) / 365.25 / (1 + z_mid)        [yr Mpc^3]
    N_obs[p]   = sum_c A[p, c] * r_c                                     -> r por NNLS, errores por bootstrap Poisson
eta_found sale de TODAS las inyecciones. P(p|c, found) = f_elig * P(p|c, con features) + (1 - f_elig) * [p = none],
con f_elig la fraccion de encontradas con >= min_det puntos en r (las demas nunca llegan a features).
"""
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import nnls

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
FULL_SKY_DEG2 = 41252.96


def shell_volume(z1, z2, area_deg2):
    return float((COSMO.comoving_volume(z2) - COSMO.comoving_volume(z1)).value * area_deg2 / FULL_SKY_DEG2)


def response_matrix(sims, preds, fields, zbins, classes, area_deg2, min_det=7):
    pred_cls = list(classes) + ["none"]
    s = sims.merge(preds, on="sim_id", how="left")
    out = {}
    for z1, z2 in zbins:
        zb = s[(s.z >= z1) & (s.z < z2)]
        zmid = 0.5 * (z1 + z2)
        V = shell_volume(z1, z2, area_deg2)
        A = pd.DataFrame(0.0, index=pred_cls, columns=list(classes))
        for c in classes:
            sc = zb[zb.clf_class == c]
            found = sc[sc.found]
            if len(found) == 0:
                continue
            elig = found[found.n_det_r >= min_det]
            featured = elig[elig.pred.notna()]
            f_elig = len(elig) / len(found)
            P = pd.Series(0.0, index=pred_cls)
            if len(featured):
                P = featured.pred.value_counts(normalize=True).reindex(pred_cls, fill_value=0.0) * f_elig
            P["none"] += 1.0 - f_elig
            for f in fields.itertuples():
                scf = sc[sc.field == f.field]
                if len(scf) == 0:
                    continue
                ct = f.window_days * scf.found.mean() * P
                A[c] += V * ct / 365.25 / (1.0 + zmid)
        out[(z1, z2)] = A.loc[list(classes)]
    return out


def estimate(A, n_obs, n_boot=2000, seed=1):
    a = A.values
    n = n_obs.reindex(A.index).fillna(0.0).values
    r, _ = nnls(a, n)
    rows = dict(cls=list(A.columns), rate=r)
    if n_boot:
        rng = np.random.default_rng(seed)
        boot = np.array([nnls(a, rng.poisson(n).astype(float))[0] for _ in range(n_boot)])
        rows.update(lo=np.percentile(boot, 16, axis=0), hi=np.percentile(boot, 84, axis=0))
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `$PY tests/test_p78_rates.py`
Expected: tres `ok`.

- [ ] **Step 5: Implementar la prueba de cierre sobre las sims reales.** Se corre en la Tarea 24, cuando ya existen las predicciones.

```python
# pipeline78/rates_closure.py
"""Cierre: la mitad A de las sims (sim_id par) construye la matriz; la mitad B fabrica 50 catalogos con tasas
conocidas (Poisson). Se mide sesgo y cobertura del intervalo 16-84 por clase y bin."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline78.paths import STORE
from pipeline78.rates import response_matrix, estimate, shell_volume

TRUE = {"Ia": 2.5e-5, "CC": 7.0e-5}                 # yr^-1 Mpc^-3, orden de SUDARE I (solo para el cierre)
ZBINS = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
AREA = 1.15


def mock_counts(B, zb, fields, rng, min_det=7):
    z1, z2 = zb
    zmid, V = 0.5 * (z1 + z2), shell_volume(z1, z2, AREA)
    Bz = B[(B.z >= z1) & (B.z < z2)]
    counts = pd.Series(0.0, index=list(TRUE) + ["none"])
    for c, r in TRUE.items():
        for f in fields.itertuples():
            s = Bz[(Bz.clf_class == c) & (Bz.field == f.field)]
            fs = s[s.found]
            if len(fs) == 0:
                continue
            n = rng.poisson(r * V * f.window_days * s.found.mean() / 365.25 / (1.0 + zmid))
            if n == 0:
                continue
            pick = fs.sample(n, replace=True, random_state=int(rng.integers(2**31)))
            elig = int((pick.n_det_r >= min_det).sum())
            counts["none"] += n - elig
            pool = fs[(fs.n_det_r >= min_det) & fs.pred.notna()].pred
            if elig and len(pool):
                vc = pool.sample(elig, replace=True, random_state=int(rng.integers(2**31))).value_counts()
                counts = counts.add(vc, fill_value=0.0)
    return counts.reindex(list(TRUE)).fillna(0.0)


def closure(sims, preds, n_mock=50, seed=3):
    fields = pd.read_csv(STORE / "sudare_fields.csv")
    A_half = sims[sims.sim_id % 2 == 0]
    B_half = sims[sims.sim_id % 2 != 0].merge(preds, on="sim_id", how="left")
    A = response_matrix(A_half, preds, fields, ZBINS, list(TRUE), AREA)
    rng = np.random.default_rng(seed)
    rows = []
    for zb in ZBINS:
        for _ in range(n_mock):
            est = estimate(A[zb], mock_counts(B_half, zb, fields, rng), n_boot=200)
            for e in est.itertuples():
                rows.append(dict(zbin=f"{zb[0]}-{zb[1]}", cls=e.cls, rate=e.rate, lo=e.lo, hi=e.hi, true=TRUE[e.cls]))
    d = pd.DataFrame(rows)
    d["dentro"] = (d.lo <= d.true) & (d.true <= d.hi)
    res = d.groupby(["zbin", "cls"]).agg(sesgo=("rate", lambda x: x.mean()), cobertura=("dentro", "mean")).reset_index()
    res["sesgo"] = res.sesgo / res.cls.map(TRUE) - 1.0
    return res


if __name__ == "__main__":
    H = Path.home() / "thesis_runs"
    sims = pd.read_parquet(H / "sudare_v78/_sims_all.parquet")
    sims["clf_class"] = sims.clf_class.replace({"II": "CC", "Ibc": "CC"})
    preds = pd.read_parquet(H / "sudare_v78_preds.parquet")
    preds["pred"] = preds.pred.replace({"II": "CC", "Ibc": "CC"})
    out = closure(sims, preds)
    (H / "rates").mkdir(exist_ok=True)
    out.to_csv(H / "rates/cierre.csv", index=False)
    print(out.round(3).to_string(index=False))
```

- [ ] **Step 6: Commit**

```bash
git add pipeline78/rates.py tests/test_p78_rates.py pipeline78/rates_closure.py
git commit -m "pipeline78: tasas por control time Monte Carlo con matriz de clasificacion y prueba de cierre"
```

### Task 23: Validación contra SUDARE I con su propia clasificación

> **Enmienda 2026-10-02.** `data/sudare_I_table5.csv` lleva columnas `z1, z2` (no `zbin`), que son las que usa el código.

- [ ] **Step 1: Transcribir la Tabla 5 de SUDARE I** a `data/sudare_I_table5.csv`, con las columnas tipo, zbin, n_sne, rate, stat_lo, stat_hi, sys_lo, sys_hi y página. Se lee del PDF (`papers_varios/Rate/Sudare_I.pdf`, página 14) y se verifica cada número dos veces.

- [ ] **Step 2: Medir con su clasificación.**
  - N_obs sale de la columna `fty` de `master_referee.list`, con peso 1 para `sn` y 0.5 para `psn`, y el z de la columna `zsn`.
  - Se usan solo los campos cdfs1, cdfs2 y cosmos1.
  - La clasificación es perfecta (P(p|c) = δ) y `min_det=0`, así que solo entra la eficiencia de búsqueda.
  - Mismos bins que la Tabla 5.
  - CC suma II, IIn e Ibc, como en SUDARE I §8.1.

```bash
$PY - <<'EOF'
import numpy as np, pandas as pd
from pathlib import Path
from pipeline78.paths import SUDARE_DIR, STORE, DATA
from pipeline78.rates import response_matrix, estimate
H = Path.home() / "thesis_runs"
CC = {"II": "CC", "Ibc": "CC"}
sims = pd.read_parquet(H / "sudare_v78/_sims_all.parquet").query("field in ['cdfs1','cdfs2','cosmos1']").copy()
sims["clf_class"] = sims.clf_class.replace(CC)
perfect = sims.loc[sims.found, ["sim_id", "clf_class"]].rename(columns={"clf_class": "pred"})
fields = pd.read_csv(STORE / "sudare_fields.csv")
ref = [l.split() for l in open(SUDARE_DIR / "master_referee.list") if l.strip() and not l.startswith("#")]
r = pd.DataFrame({"flag": [x[1] for x in ref], "z": pd.to_numeric([x[10] for x in ref], errors="coerce"),
                  "fty": [x[12] for x in ref]})
r["cls"] = r.fty.map({"Ia": "Ia", "II": "CC", "IIP": "CC", "IIn": "CC", "IIb": "CC", "Ib": "CC", "Ic": "CC", "Ibc": "CC"})
r["w"] = np.where(r.flag == "psn", 0.5, 1.0)
t5 = pd.read_csv(DATA / "sudare_I_table5.csv")
rows = []
for t in t5.itertuples():
    A = response_matrix(sims, perfect, fields, [(t.z1, t.z2)], ["Ia", "CC"], area_deg2=1.15, min_det=0)[(t.z1, t.z2)]
    n = r[(r.z >= t.z1) & (r.z < t.z2)].groupby("cls").w.sum().reindex(["Ia", "CC"]).fillna(0.0)
    e = estimate(A, n).set_index("cls").loc[t.tipo]
    rows.append(dict(tipo=t.tipo, zbin=f"{t.z1}-{t.z2}", n_nuestro=n[t.tipo], n_paper=t.n_sne,
                     nuestra=e.rate * 1e4, lo=e.lo * 1e4, hi=e.hi * 1e4, paper=t.rate,
                     err_paper=np.hypot(0.5 * (t.stat_lo + t.stat_hi), 0.5 * (t.sys_lo + t.sys_hi))))
d = pd.DataFrame(rows)
d["razon"] = d.nuestra / d.paper
(H / "rates").mkdir(exist_ok=True)
d.to_csv(H / "rates/validacion_sudareI.csv", index=False)
print(d.round(3).to_string(index=False))
EOF
```

- [ ] **Step 3: Criterio.** La razón nuestra/suya por bin tiene que ser compatible con 1 dentro de sus errores estadísticos más sistemáticos. Si no, se revisa en este orden:
  1. la ventana T_f (+365 d);
  2. DEmax y β;
  3. la LF y el polvo de cada clase.

  No se avanza a la Tarea 24 sin explicar las diferencias.

### Task 24: Tasas de SUDARE con nuestra clasificación

- [ ] **Step 1: Predecir.**
  - La submuestra sintética se predice con validación cruzada en dos mitades por paridad de `sim_id`. Cada sim lo predice el modelo entrenado con la otra mitad, así la matriz no sale optimista.
  - Las sims elegidas que no salieron del extractor quedan con pred "none".
  - Los reales se predicen por voto de mayoría sobre 10 semillas con el modelo entrenado en todo.

```bash
$PY - <<'EOF'
import numpy as np, pandas as pd
from pathlib import Path
from pipeline78.recipe import build, fit_predict
H = Path.home() / "thesis_runs"
USE_M = True                                   # <- la variante elegida en G5
sims = pd.read_parquet(H / "sudare_v78/_sims_all.parquet")
syn = build(H / "sudare_v78_features/features/features.csv", sims.rename(columns={"field": "oid"}), "r", "i")
syn = syn.merge(sims.rename(columns={"field": "oid"})[["oid", "part_index", "sn_type", "sim_id"]],
                on=["oid", "part_index", "sn_type"], validate="one_to_one")
even = syn.sim_id % 2 == 0
syn.loc[~even, "pred"] = fit_predict(syn[even], syn[~even], 0, USE_M)
syn.loc[even, "pred"] = fit_predict(syn[~even], syn[even], 0, USE_M)
picked = pd.read_parquet(H / "sudare_v78_sub/_picked.parquet")[["sim_id"]]
preds = picked.merge(syn[["sim_id", "pred"]], on="sim_id", how="left").fillna({"pred": "none"})
preds.to_parquet(H / "sudare_v78_preds.parquet", index=False)
meta = pd.read_csv(H / "real_sudare/meta_real_sudare.csv")
real = build(H / "real_sudare_features/features/features.csv", meta, "r", "i")
votes = np.array([fit_predict(syn, real, s, USE_M) for s in range(10)])
real["pred"] = [pd.Series(v).mode().iloc[0] for v in votes.T]
real[["oid", "label", "z", "pred"]].to_csv(H / "sudare_real_preds.csv", index=False)
print(preds.pred.value_counts().to_dict()); print(pd.crosstab(real.label, real.pred))
EOF
```

- [ ] **Step 2: Prueba de cierre**

Run: `$PY -m pipeline78.rates_closure`
Expected: en cada bin con al menos 10 objetos esperados, |sesgo| menor que 0.1 y cobertura entre 0.55 y 0.80. Si falla, no se reportan tasas. Primero se busca por qué.

- [ ] **Step 3: Estimar las tasas.** Ia y CC por bin con la matriz de todas las sims y nuestras predicciones.

```bash
$PY - <<'EOF'
import pandas as pd
from pathlib import Path
from pipeline78.paths import STORE, DATA
from pipeline78.rates import response_matrix, estimate
H = Path.home() / "thesis_runs"
CC = {"II": "CC", "Ibc": "CC"}
sims = pd.read_parquet(H / "sudare_v78/_sims_all.parquet"); sims["clf_class"] = sims.clf_class.replace(CC)
preds = pd.read_parquet(H / "sudare_v78_preds.parquet"); preds["pred"] = preds.pred.replace(CC)
real = pd.read_csv(H / "sudare_real_preds.csv"); real["pred"] = real.pred.replace(CC)
fields = pd.read_csv(STORE / "sudare_fields.csv")
t5 = pd.read_csv(DATA / "sudare_I_table5.csv")
rows = []
for (z1, z2), _ in t5.groupby(["z1", "z2"]):
    A = response_matrix(sims, preds, fields, [(z1, z2)], ["Ia", "CC"], area_deg2=1.15)[(z1, z2)]
    n = real[(real.z >= z1) & (real.z < z2)].pred.value_counts().reindex(["Ia", "CC"]).fillna(0.0)
    for e in estimate(A, n).itertuples():
        rows.append(dict(zbin=f"{z1}-{z2}", tipo=e.cls, n=n[e.cls], rate_1e4=e.rate * 1e4, lo=e.lo * 1e4, hi=e.hi * 1e4))
d = pd.DataFrame(rows); d.to_csv(H / "rates/tasas_sudare_v78.csv", index=False); print(d.round(3).to_string(index=False))
EOF
```

- [ ] **Step 4: Comparar y declarar.**
  - La comparación es contra `validacion_sudareI.csv` (Tarea 23) y la compilación de los apéndices de la tesis.
  - Se declara que faltan las `psn` reales mientras no lleguen sus curvas.
  - Cuando lleguen, solo se rehacen la Tarea 19 (reales), la extracción de features de los reales en la Tarea 20, la Tarea 21 y los Steps 1 a 3 de esta.

### Parte F: tesis

### Task 25: Capítulos al día

- [ ] **Step 1: Capítulo 3.** Usar la skill `escribir-seccion-paper` y después `verificar-citas`. Cambios:
  - sec:methods:proj:
    - curvas de filtro ZTF y OmegaCAM con punto cero AB;
    - dilatación (1+z) y ancla en el máximo de reposo;
    - M intrínseco aplicado antes del polvo;
    - modelo de ruido real, sky-limited con m_lim, y regla SUDARE;
    - tabla de polvo corregida;
    - fórmula de P(z) de la tabla de la l.2675.
  - sec:methods:features: emcee con la configuración de G2 y la misma extracción para reales y sintéticas.
  - sec:methods:classify: el MLP jerárquico y el protocolo de validación y final.
- [ ] **Step 2: Capítulo 4.**
  - sec:results:ztf: números de `ztf_v78_resultados.json`, sin "preliminary".
  - Sección nueva de clasificación SUDARE.
  - sec:results:rates: validación contra SUDARE I y nuestras tasas.
- [ ] **Step 3: Capítulo 5.** Conclusiones y trabajo futuro coherentes con lo anterior.
- [ ] **Step 4: Compilar sin errores, commit en el repo de la tesis.** Push a Overleaf solo con orden.

---

## Decisiones tomadas

2026-10-02, Mauricio:
- **Tarea 0:** commit WIP de lo pendiente, rama `pipeline78` y commit local por tarea. Push solo con orden.
- **D1:** ZTF con Ia 10, II 8, IIb 2, Ibc 10 y además IIn 10 con su etiqueta. IIn queda fuera del entrenamiento por defecto y G3 decide.
- **D2:** LOESS apagado.
- **D3:** curvas SVO de ZTF y OmegaCAM con punto cero AB.
- **D4:** M intrínseco antes del polvo, y polvo de las Ia a literatura (τ 0.35, frac_zero 0.40). Se verifica en G1.
- **D5:** dilatación (1+z) y ancla sin tablas externas.
- **D10:** las tres congeladas con defecto se proyectan igual.
- **Enmienda T3:** ancla en el pico principal, no en el de enfriamiento.
- **Enmienda T7:** la ventana pre-explosión queda como el plan. Mauricio advirtió que tocarla afecta el fit de Villar, así que se mide en el piloto y se decide en G1.
- **Enmiendas de tasas:** pesos por subtipo (Li+2011b) y área de CDFS con solape.
- **Pendientes:** D6 en G2, D7 en G4, D8 y D9 antes de la Tarea 18.

Avance y rulings del ejecutor: `.superpowers/sdd/2026-10-02-biblioteca-a-tasas/progress.md`.
