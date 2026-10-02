# Estudio de la proyección y preparación para las 78 congeladas (2026-10-02)

Para Mauricio y para el chat que arme el plan de trabajo. Complementa a
`HANDOFF_20260930_siguiente_paso.md`. Lo que dice "medido" se midió hoy. Lo que
dice "propuesta" no está implementado.

## 1. Cómo funciona hoy `run_per_field.py`, paso a paso

Por cada campo (OID de ZTF), por cada tipo y por cada pivote (`part_index` 0–9):

1. **Template.** Se toma de `data/<TEMPLATE_DIRS[tipo]>/`, barajado por (OID, tipo).
   `leer_spec` tiene `lru_cache`, así que cada proceso parsea cada template una sola vez.
2. **z.**
   - `z_mode='empirical'` (activo): remuestrea `data/z_empirical_<tipo>.txt` + N(0, 0.003).
     Ese archivo solo existe para Ia, II e Ibc.
   - Los demás tipos caen a dV/dz hasta `z_max_by_type`.
3. **Extinción.**
   - E(B−V)_host sale de una mezcla por tipo (`EXTINCTION_CONFIG`), con el R_V del tipo.
   - E(B−V)_MW sale del cache SFD98 por OID.
4. **Espectro observado** (`correct_redeening`, `reverse=True`, `use_DL=True`): host →
   ×(1+z) en λ y ÷(1+z) en flujo → MW → ×(10 pc/D_L)².
5. **Fotometría sintética** por banda, solo en las épocas con ≥95 % de cobertura del filtro.
   Las respuestas son las splines SDSS g'r'i' de `Practica2/Splines_eachfilter_2`, no
   curvas propias de ZTF. Los ceros están en `cteg/cter/ctei`.
6. **Normalización de luminosidad.** Se sortea M_peak y se desplaza toda la curva para que
   el mínimo de r observado sea igual a μ(z) + M_peak:
   - Ia: Phillips (`dm15_Ia.json`);
   - II, si `M_mode='empirical'`: M_empirical;
   - el resto: gaussiana.

   Consecuencia: el desplazamiento cancela el oscurecimiento por polvo en r y solo
   sobrevive el color. Además M queda definido en la banda r **observada**, sin
   K-corrección. Ver §4.
7. **Proyección** (`multiband_field_projection`, modo determinista):
   - La ventana del log se parte en 10 y el máximo del template cae en el centro de la
     partición `part_index`. Los `offset_range*` no se usan en este modo.
   - Detección si m_modelo < maglimit (5σ).
   - Ruido limitado por cielo, derivado del maglimit.
   - Las épocas hasta 25 d antes del inicio del template salen como upper limits.
8. **Salida.**
   - Un parquet por OID con todas las épocas en ventana (detecciones y upper limits).
   - Una simulación sin ninguna época en la ventana queda como FAIL y **no se guarda**.

## 2. Lo que se cambió hoy (sin commit)

| Cambio | Por qué | Evidencia |
|---|---|---|
| `TEMPLATE_DIRS` apunta a `data/{Ia,II,IIb,IIn,Ibc}_v78` (15/13/10/10/30). SLSN-I sale. | Antes apuntaba a `*_new` (EMPCA, ya descartado). | Copia con `cp` + `cmp` archivo por archivo. md5 en `data/templates_v78_md5.csv`. `test_v78.py` lo verifica. |
| LOESS apagado (`loess_smooth: False`) | Re-suavizaba curvas que ya aprobaste. | Medido: hasta 0.7–0.9 mag de deformación (2011fe y 2012aw). |
| Dilatación (1+z) del tiempo | Faltaba. Se ponían días de reposo como días observados. | `test_v78.py`: 2011fe dura 396 d en reposo y 415.8 d a z=0.05. |
| Ancla única: máximo de r de la curva sintética, para todos los tipos | Ver bug 1 de §3. Además IIb e IIn no estaban en las tablas de máximo y fallaban. | Smoke test: 50/50 con `min_mag_r`, desplazamientos sanos. |
| IIn agregada. M_r = −19.18 ± 1.32 (Nyholm+2020, 2020A&A...637A..73N, verificado en ADS). Extinción = la de II (provisional). z_max 0.08 (provisional). | Decisión del 2026-10-02: proyectar con la etiqueta real. | |
| Tipos por defecto = `TEMPLATE_DIRS`. `--types` se valida contra esa lista. Se falla al arrancar si falta M_peak o extinción. La metadata guarda tipos y templates. | Antes un tipo sin M_peak usaba en silencio la luminosidad propia del template. | |
| Solo bandas g y r en ZTF | La i es el 0.6 % del log (31 118 de 5.0 M) y el clasificador usa solo g y r. Ahorra ~1/3 del tiempo. | |
| `dm15_Ia.json` regenerado: 15/15, rango 0.76–1.46 | El json viejo tenía 10 valores y 1998aq y 2012ht caían al default 1.1. Respaldo en `dm15_Ia_Ia_new.json`. | Media de la población: −19.31, σ total 0.18. |

**Smoke test** (1 campo, 50 sims, semilla 7): 46 OK.
- 1 FAIL es un hueco estacional real.
- 3 FAIL son sims que solo tenían UL pre-explosión. Antes se contaban como OK.
- `verify_output`: 0 FAIL.
- Tiempo: 2.5 s por sim.
- RSS máximo: **3.0 GB por proceso**. El obslog completo en pandas pesa ~1.5 GB.
- Con 16 GB conviene correr como máximo 3 procesos en paralelo, no 4.

**Escala de flujo**, revisada en las 78: sin flujos negativos ni NaN, todas a 10 pc físicos.
- M_V de las Ia: mediana −19.35.
- Cuatro brillantes, que no afectan porque M se re-sortea: 2014L −19.76, 2005hg −19.22,
  ASASSN-14lp −19.97 y 2001V −19.96. Las tres con E(B−V)_host alto (0.33–0.58) apuntan a
  extinción sobrecorregida.

## 3. Bugs encontrados en lo que ya existía (afectan al baseline run_1000_v2, acc 0.747)

1. **Ibc con el tiempo duplicado.** `generate_synthetic_curves` sumaba el MJD de máximo
   de `maximum_Ibc.dat` a fases que ya eran MJD.
   - El ancla caía fuera del template y se "clipeaba" al inicio.
   - Resultado: en run_1000_v2 las Ibc se ubicaron con su **primera época** en el pivote,
     y las Ia y II con su máximo. Desplazamiento de las Ibc en el parquet: ≈ −50 000.
   - Arreglado hoy.
2. **Clasificador, merge r–g sin `sn_type`** (repo `feature_extraction`, no tocado).
   - En `opt_clasificador/05_faseC_run1000_v2.py`, `build(df, ['oid','part_index'])` une
     r con g solo por (oid, part_index). Ia, II e Ibc comparten `part_index` en el mismo OID.
   - Verificado: 13 063 filas r dan 24 644 al unir. **15 054 cruzan tipos**, con `color_gr`
     sacado del g de otra SN. Pasa lo mismo en los `ab_*.py`.
   - Arreglo de una línea: agregar `'sn_type'` a las llaves. Cambia el número del baseline.
3. **El clasificador es un MLP de dos niveles, no un RF.** La tesis dice RandomForest
   (cap. 2, cap. 3 §clasificador, cap. 5) y el código entrena `MLP(128,64)`.
   Hay que alinear el texto o el código.
4. **Etiquetas nuevas en el clasificador.** `CL=['Ia','II','Ibc']` está fijo. Una `IIb` o
   `IIn` sintética cae en silencio como "Tipo I" en el nivel 1. Antes de entrenar con el run
   nuevo hay que remapear y filtrar (`IIb→II`, IIn según decidas, `syn = syn[syn.label.isin(CL)]`).
5. **La muestra real de test (675) tiene solo** SN II, Ia, Ia-91T, Ia-91bg, Ib e Ic.
   No incluye IIb, IIn, Ic-BL ni SLSN. Las 7 Ic-BL de la clase Ibc no tienen contraparte
   real en el test.
6. **Cache del extractor.** `_tasks_index.parquet` solo se invalida si cambia el conjunto de
   paths. Hay que usar un `--output_dir` nuevo para el run nuevo.

## 4. Dos campañas distintas, no una

| | Clasificación (training set) | Tasas (Monte Carlo para η / CT) |
|---|---|---|
| Para qué | Entrenar y validar el clasificador | Eficiencia de detección más clasificación por tipo y z |
| Tiempo de explosión | Pivote determinista, el máximo cae dentro de la ventana | **Uniforme** en [t0 − 365 d, tK] con paso de 1 d (receta de SUDARE I) |
| Qué se guarda | Solo lo proyectado. El extractor exige ≥7 detecciones | **Todas** las inyectadas, también las no detectadas (son el denominador) |
| z | Empírico, igual a la muestra real | Grilla o dV/dz en 0.1–0.8 (SUDARE) |
| M_peak | Observado: M_empirical o gaussiana anclada en r observado | **Intrínseco**: LF de literatura (SUDARE usó Li+2011b) anclado en **reposo**, antes del polvo |
| Polvo | Se cancela en r (ver §1 paso 6) | Debe oscurecer: si no, η queda sobreestimada |
| Detección | m < m_lim(5σ) | Curva DE(m) = DEmax[arctan(β(m50 − m))/π + 1/2] en **r** (banda de búsqueda) |
| Mezcla de subtipos | La que salga de los templates | La de SUDARE: Ia 70/10/15/5 (normal/91T/91bg/02cx), II 60 IIP/10 05cs-like/10 87A-like/10 IIL/10 IIb, Ib/c 27/68/5 (Ib/Ic/98bw), IIn 45/45/10 (98S/10jl/05gj) |
| Semillas | Propias | Independientes del training |

La cadena es la misma (proyección → SPM → clasificador), pero la campaña de tasas
necesita código que hoy no existe:
- inyección uniforme en el tiempo;
- guardar las no detectadas;
- DE(m) con m50;
- anclaje en reposo;
- polvo que oscurezca.

Ojo con el PLAN de SUDARE, que dice que "la K-corrección sale sola". Eso vale para el
SED, pero no con la normalización actual, que fija M en r **observado**. A z = 0.4 esa
diferencia es la K-corrección entera. El código ya tiene la maquinaria para corregirlo
(`rest_anchor`, usado hoy solo con SLSN-I).

## 5. Qué proyectar y en qué bandas

**ZTF, clasificación (run siguiente):**
- Las 78, cada una con su etiqueta: Ia 15, II 13, IIb 10, IIn 10, Ibc 30.
- Bandas **g y r**.
- Mapeo al entrenar: Ia, II (+IIb, +IIn si así lo decides), Ibc (incluye Ic-BL).
- Pendiente de diseño:
  - IIb se sortea con dV/dz hasta 0.11 (no hay `z_empirical_IIb`). En el smoke test dio
    0/10 con detecciones: casi todas quedan cerca del límite.
  - Si IIb entra como II al entrenar, conviene darle el z empírico de II.
  - IIn está en el mismo caso con 0.08.

**SUDARE, training set:**
- Las mismas 78.
- Bandas **g, r e i**, con curvas de OmegaCAM (hoy no están, se usan las SDSS).
- z en 0.1–0.8.

**SUDARE, tasas:**
- Las mismas 78, con pesos por subtipo como en la tabla de §4.
- Hay que asignar un subtipo a cada template. No sé si las 15 Ia incluyen alguna 91bg o
  Iax: hay que revisarlo.
- La detección y el CT van en **r**. g e i solo sirven para clasificar.

**Cobertura en λ (templates de 3005 a 9193 Å en reposo):**
- g pierde el 95 % de cobertura sobre z ≈ 0.33.
- r roza el límite cerca de z ≈ 0.8.
- i no tiene problema.
- Hay que medirlo con las curvas reales de OmegaCAM antes de decidir entre r+i a alto z
  o la extensión UV.

## 6. Decisiones que siguen abiertas (tuyas)

1. Mapeo de clases al entrenar en ZTF: IIb→II (propuesto) e IIn→II o fuera.
2. z de IIb e IIn en el run de entrenamiento: ¿el empírico de II (o de Ia para IIn), o dV/dz?
3. La extinción de IIn queda como la de II mientras no se defina otra.
4. Arreglar el merge r–g del clasificador antes de reentrenar. Cambia el baseline 0.747.
5. RF o MLP: alinear la tesis con el código.
6. SUDARE: aceptar el diseño de la campaña de tasas de §4. Implica escribir el runner de SUDARE.
7. **El pico temprano se lleva el ancla y la normalización** (hallazgo de la revisión de
   código). `argmin(r)` toma el pico de enfriamiento en tres templates:
   - SN2011fu: pico en el día 1. El pico de Ni queda 0.60 mag más débil.
   - SN2013df: pico en el día 2. El de Ni queda a +0.22 mag.
   - SN2006aj: pico en la época 0. El segundo queda a +0.49 mag.

   El M de Taddia+2018 se refiere al pico principal. Opción: un json fijo con el t_peak
   principal de cada template, igual que el dm15.
8. **La ventana de upper limits pre-explosión cuenta desde el inicio del template, no desde
   la explosión** (viene de agosto). Las 15 Ia empiezan entre 6 y 18 d (reposo) antes del
   máximo de r, y algunas Ibc empiezan tarde (1994I, 2007ru, 2015ah).
   - Los 25 d de UL caen donde la SN real estaría a 0.5–2 mag del pico.
   - Resultado: subidas instantáneas, que pueden sesgar el t_rise del clasificador.
   - Opción: abrir la ventana desde un t_exp estimado por template.

Arreglados tras la revisión:
- z y M empíricos se buscan para todos los tipos, y al arrancar se imprime la fuente por tipo.
- Se aborta antes de cargar el obslog si un tipo no tiene templates o si a una Ia le falta dm15.
- Una sim que solo tiene UL pre-explosión cuenta como FAIL.
- El inicio de la SN es común a las bandas.
- `verify_output` ignora el 99 pre-explosión y toma los tipos de `run_metadata`.
- Sin duplicados en `--types`.

Hay otro plan en `docs/superpowers/plans/2026-10-02-biblioteca-a-tasas.md`, de otro chat.
Propone un motor nuevo, `pipeline78/`, y encontró por su cuenta lo mismo sobre la dilatación
y el ancla de las Ibc. Estos cambios dejan correcto el runner actual mientras tanto.

## 7. Cómo verificar antes de lanzar

- `python test_v78.py`: md5 de las 78, dilatación, LOESS apagado y ancla. Tarda ~10 s.
- `python run_per_field.py --oid ZTF17aaabmmr --seed 7 --m-mode empirical --output-dir <tmp>`:
  un campo, ~2 min.
- El run masivo lo lanza Mauricio:
  - con `--oids-file data/oids_1000_v2_pN.txt`;
  - como máximo 3 procesos (3 GB cada uno);
  - estimado: 1000 campos × 50 sims × 2.5 s ≈ 35 h de CPU, unas 12 h con 3 procesos.
