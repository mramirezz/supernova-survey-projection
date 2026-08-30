# HANDOFF — estado y contexto completo del proyecto (2026-08-24, madrugada)

Documento de continuación: si la sesión de Claude murió, esto es todo lo que hay que
saber para retomar sin releer nada. Escrito tras la maratón del 22-24 de agosto.

---

## 1. Objetivo grande y arquitectura

**Paper 2 / capítulo de tesis**: construir templates de series espectrales temporales
por Optimal Transport para **6 clases** (Ia, II, IIb, Ibc, IIn, SLSN-I), proyectarlas
sobre las condiciones reales de ZTF (cadencia, límites de mag, observing log),
entrenar un RandomForest sobre features de curva de luz (Villar+2019, ajuste MCMC) y
calcular tasas de SN (validar en ZTF, aplicar en SUDARE como análogo LSST).

**Cadena del pipeline** (repo `paper2_ZTF/Codes/spectral_series`, env
`/opt/anaconda3/envs/series/bin/python`; el env `projection` para astropy/astroquery):

```
espectros crudos → s0 staging → s0b Loess (GATE visual de Mauricio) →
s1 splines+K+MAD (GATE golden) → s1b extensión pre-OT (fusión UV + escalera) →
s2 OT pares → s3 calibración → s4 promedio ponderado → serie diaria →
s6 deredden → s8 mangling → dereddened/<clase> → proyección ZTF
```

**MAPA DE DATOS (unificación 2026-08-30, decisión delegada a Claude por Mauricio):**
- **Punto único de revisión: `paper2_ZTF/REVIEW.html`** (regenerable con
  `steps/make_review_index.py`): estado por clase + enlaces a cada gate Loess,
  QA, productos y tablas maestras. Mauricio revisa TODO desde ahí.
- **Productos canónicos** (todas las clases): `paper2_ZTF/dereddened/<clase>` y
  `paper2_ZTF/uv_dereddened/<clase>`. Tabla maestra de extinción (todas las
  clases): `nuevas_series_II/extinction_distance_table_ALL.csv`.
- **Staging crudo por SN** (todas las clases, pese al nombre histórico):
  `paper2_ZTF/nuevas_series_II/<SN>/`.
- **Workspaces por campaña** (Drive, rutas con espacios, SIEMPRE entre comillas),
  registrados en `config.WORKSPACE_BY_CLASS`, se activan con
  `SPECTRAL_SERIES_WORKSPACE` (RUTA COMPLETA, no la clave):
  - II: `Mi unidad/Spectral Time Series - Ramirez M II` (default)
  - Ia: `Mi unidad/Spectral Time Series - Ramirez M - Ia`
  - IIb: `Mi unidad/Spectral Time Series - Ramirez M IIb` (NUEVO 2026-08-30;
    el de Barbara queda INTACTO como fuente de solo lectura)
  - Ibc: `Mi unidad/Spectral Time Series - Ramirez M`
  - SLSN: `Mi unidad/Spectral Time Series - Ramirez M SLSN`
  - IIn: por crear cuando termine la descarga de datos.
`SPECTRAL_SERIES_PRE_EXT=1` redirige la cadena OT a directorios `_ext` (A/B
reversible, producción intacta). El pipeline vive en `Codes/spectral_series`
(REPO GIT PROPIO desde 2026-08-30; antes corría sin versionar).

---

## 2. La historia que explica el estado actual (léela, no es adorno)

### 2.1 El parpadeo azul de 2012aw y la refutación del orden viejo
Mauricio revisó visualmente la serie de SN2012aw y encontró que el borde azul
(3000-3400 Å) "parpadeaba": features que aparecen y desaparecen entre días vecinos.
El debug trazó la causa en cadena:
1. Las series se extendían DESPUÉS del OT (s7c rellenaba los ~119 días interpolados
   uno por uno, cada día eligiendo donantes por separado).
2. Primera corrección (insuficiente): regla propio-primero en `_extender_lado`
   (si la SN tiene UV real en la ventana, donantes ajenos no entran; etiqueta
   `transporte_propio`). Mejoró pero el zoom decisivo
   (`figures_templates/qa_mangled/II/SN2012aw_zoom_costura.png`) mostró que el
   relleno se salía de la envolvente de sus propios donantes: pozo 2-3x más hondo,
   bump más alto, dientes de sierra. Un baricentro NO puede salirse de la envolvente
   → la amplificación venía de la normalización de costura por segmento. Defecto
   estructural, no de parámetros.
3. Diagnóstico de Mauricio (textual): "donde tenemos espectros UV se pone el UV para
   que esa feature sea transportada en el OT". El error estaba antes de interpolar.

### 2.2 El orden nuevo (APROBADO por Mauricio el 2026-08-30: cascada y serie completa)
- El UV real de la propia SN (5 espectros Swift grism de 2012aw en +4.9/+6.9/+8.9/
  +11.9/+13.9, ya anclados por s6b con compuerta fotométrica UVOT ±0.2 mag) se
  **fusiona a los espectros observados ANTES del OT** (fusión coseno 500 Å,
  `_merge_uv_propio`, injertada en `s1b_extend_pre_ot.py`).
- El OT transporta la feature entre épocas reales: el día +5.9 es la geodésica entre
  los UV de +4.9 y +6.9. Acotado por la envolvente por construcción.
- Resultados de la compuerta (2012aw, serie en `dereddened/II_new_preot/`):
  envolvente 0.0% fuera en +5.9/+7.9/+9.9/+10.9 (2% en +12.9), oscilación
  3000-3200 ×1.86 (orden viejo ×2.51, sin golden ×3.53, bug original ×122),
  rectángulo único 3005 Å en 178 épocas.
- Figuras para la validación visual de Mauricio (LA compuerta dura):
  `SN2012aw_preot_cascada.png` y `SN2012aw_preot_serie.png`.

### 2.3 Golden sample: de métrica olvidada a regla
El port de s1 calculaba el MAD de los K entre bandas pero NUNCA filtró (el notebook
original del Paper 1 sí, umbral 0.1, celda 27). En 2012aw entraban 13/60 espectros
con MAD>0.10, ocho de ellos en +4..+25 d = exactamente la zona del valle reclamado.
Ahora es REGLA: `core/golden.py`, `config.GOLDEN_MAD_MAX=0.10`, filtro en la entrada
de s2 y en s1b, con traza de descartes. Sondeo: conserva 88.2% de épocas y no rompe
la cadencia de ninguna serie (II) ni de las Ia (89-100% por SN).

### 2.4 Bug de pares de la misma noche (cazado en la validación)
El OT emparejaba espectros separados por 43 minutos (56048.03/56048.06): cuantiles
degenerados → ratios ~1e10. El dedupe de staging (redondeo 0.1 d) falla cuando las
fases caen en lados opuestos del límite. Fix de raíz: `config.OT_MIN_DELTA_DAYS=0.5`
aplicado en `ot_grid_sn` (core/ot_interp.py) y `_keep` (s2). El bug existía también
en la cadena vieja de producción.

### 2.5 Diseño pendiente de implementar: decaimiento τ fuera de cobertura UV
Acordado con Mauricio (no codeado aún): fuera del rango con UV real, la feature no
muere de golpe. Relleno(fase) = baricentro OT entre P(fase) (forma del último UV
real re-anclada en nivel) y D(fase) (consenso de donantes construido UNA vez como
curva suave en fase, cuantiles regularizados en fase — su cachada: mezclar por época
contra donantes distintos rompe la suavidad aunque el peso sea suave). Peso w(Δfase)
rampa coseno con τ MEDIDO por LOO (esconder el último UV y predecirlo). NUNCA
promedio de flujos (fabrica features dobles), siempre transporte.

### 2.6 La lección transversal del proyecto (va al paper)
Seleccionar templates por calidad de datos sesga hacia objetos peculiares: en Ia,
13 de las 16 mejor observadas resultaron peculiares; en IIn, los 2 objetos mejor
muestreados de 204 son justamente los que DEJAN de ser IIn (espectroscopía de flash).
Se declara como criterio de construcción de muestra, no se esconde.

---

## 3. Estado por clase

### Ia — CERRADA por datos (15/15), lista para producción tras el desbloqueo
SN1994D, SN1998aq, SN1998dh, SN2001V, SN2003du, SN2004eo, SN2005cf, SN2006D,
SN2007af, SN2007le, SN2009ig, SN2011fe, SN2012fr, SN2012ht, ASASSN-14lp.
- Loess 15/15 aprobadas por Mauricio banda por banda (103 bandas; 18 uvw
  `rechazado` = "UV no entra a K", la extensión usa la fotometría cruda).
  Incidencias de su revisión: 1998dh U refit (alpha 0.98 se comía el máximo);
  2009ig V outlier mag 13.0 err 0.000 eliminado; 2012fr sin ULs pre-explosión,
  banda I solo LCO, corte >=56400.
- Notas suyas en `loess_review.csv`: 2006D sin pre-max (R incluida); 2004eo U sin
  pre-max. 2011fe: golden descarta -16,-15,+2 d (hueco resultante cae en nebular).
- Máximos: `maximum_perband.txt` (workspace Ia) 15/15, convención de Mauricio:
  **máximo en V, fallback r, luego R**. ASASSN-14lp derivado de su Loess (57016.0).
- OJO: ASASSN-14lp se había perdido por un glob `SN*.dat` — los inventarios deben
  cubrir nombres ASASSN/PTF/LSQ/OGLE/iPTF.
- Falta SOLO: s6b (montar espectros UV propios → `uv_dereddened/Ia`; 2011fe tiene
  HST+Swift de sobra) y el visto visual de 2012aw.

### II — lote nuevo OK, 3 históricas pendientes
13 del lote nuevo con todo (94 bandas Loess aprobadas sesión anterior, MAD, ceros).
Ceros de fase = EXPLOSIÓN, arreglados de raíz en `s7_empca.py::phase_zero()`
(la explosión manda sobre el fallback t_PT); épocas verificadas en
`nuevas_series_II/t_expl_II.txt` (1999gi 51517.8, 2005cs 53549.0, 2014cy 56899.5).
Faltan Loess+s1 de las 3 históricas de la whitelist: 1999gi, 2005cs, 2014cy.
REGLA de Mauricio: solo se revisa lo que será template — las ~27 II históricas
restantes son donantes y NO pasan por su ojo.

### IIn — definida en papel (13 templates, 5 ramas), CERO montaje en pipeline
Dossier completo (para el profesor): `nuevas_series_II/candidatas_IIn.html`
(autocontenido, curvas de luz en SVG, citas ADS verificadas). Construcción
ESTRATIFICADA (decisión de Mauricio, como Ib/Ic dentro de Ibc — etiquetas finas
que se agregan al clasificar y permiten atribuir fugas en la matriz de confusión):
- IIn-sostenida: SN2013L, SN2005kj, SN2005ip, SN2006jd, SN2005cp
- IIn-transicional: SN1998S, SN2007pk, SN2008fq
- IIn-luminosa: SN2010jl, SN2015da (corte SLSN adoptado: M≲-21 Gal-Yam 2012)
- IIn-subida lenta: SN2006aa
- IIn-2009ip-like: SN2009ip, LSQ13zm
- IIn-P: rama declarada SIN template (2011ht/2009kn/2006bo fallan por datos)
- Fuera: impostores LBV, Ia-CSM (SN1999E gemela de 1997cy), SN2010al→Ibn
  (Wolf-Rayet sin H, Pastorello+15), SN2012ab (sin espectros públicos).
Fracciones para pesar ramas en tasas (medidas en Nyholm+2020 untargeted, 40/42):
17.5% M<-20, 34% subida ≥30 d. Vincenzi+19 tiene 6 objetos en una sola etiqueta;
5 de esos 6 están acá repartidos en ramas.
FALTA: bajar espectros WISeREP + fotometría (espejo OSC en `Phd/DATA_OSC/IIn/`
tiene fotometría de 204 pero SUBCUENTA espectros — 1 de 32 para 2013L; usar
WISeREP para espectros), filas en tabla de extinción, y DECIDIR el cero de fase
de la clase (propuesta: máximo medido en la curva propia, como en el censo).

### Ibc (22) / IIb (12) / SLSN-I (14)
- Ibc: insumos crudos completos en `Phd/Practica2` (buscar case-insensitive).
- IIb: 8/12 con insumos; faltan datos de SN2008aq, SN2011ei, SN2011fu, SN2013df.
- SLSN-I (CORREGIDO 2026-08-30, verificado en el workspace): la muestra son 14, no
  7, y esta CASI CERRADA por datos desde las sesiones del 9-12 de agosto: 14/14 con
  espectros staged + fotometria, Loess 70/70 bandas aprobadas por Mauricio,
  maximos 14/14. Faltaban solo las filas de extincion de LSQ12dlf y LSQ14mo
  (en curso). La nota anterior "1/7 montada" era obsoleta.
Flujo para cada una: staging (espectros a `Data/spectra` del workspace de su
campaña, fotometría cruda a `nuevas_series_II/<SN>/`), s0b, revisión de Mauricio
banda por banda, s1, golden.

---

## 4. Convenciones y números congelados
- Golden: MAD(K)/K̃ ≤ 0.10 por época (criterio original Paper 1, ahora regla).
- Pares OT: 0.5 ≤ Δφ ≤ 50 d.
- Ceros de fase: II = explosión; Ia = máximo V (fallback r/R); IIb/Ibc = máximo;
  IIn = por decidir.
- Compuerta UVOT de espectros UV: |syn−obs| ≤ 0.2 mag (mata el grism de +25.9 d).
- Fusión UV: ventana coseno 500 Å (`FUSION_VENTANA_A`); mangling peso-cero 600 Å.
- Error del relleno azul (LOO, va en flags `relleno_A`/`err_esperado`):
  6.9% @400 Å, 9.3% @800 Å, 18% @1000 Å. Ablación: transporte 3.7-3.9% vs GP 10-13%.
- Rectángulo de clase II: 3000-10500 Å (objetivos en `config.S7C_*_TARGET`).

## 5. Tesis (`phd_thesis/tesis_template`, git → remoto `overleaf`, rama master)
- Escritos y pusheados (24537ac, 444b2c4): §3.1.3 muestra Ia (tabla generada desde
  archivos maestros), §3.1.4 IIn estratificada, cap 2 §2.2 actualizado al orden
  pre-OT (escalera con fusión UV, propio-primero, ventana escalada), subsección
  declarada del decaimiento τ, reencuadre §3.4 (EMPCA/WNMF = método evaluado),
  regla Δt mínimo en §3.3.1, seis clases en §3.1.1, subsecciones vacías con \pacha
  en cap 4 (IIb/IIn/SLSN-I). 18 entradas ADS verificadas en el .bib.
- Compilar local: BasicTeX SIN biber → swap temporal `backend=biber`→`bibtex` en
  tesisUNAB.cls, compilar, y RESTAURAR la clase byte a byte (Overleaf usa biber).
- `ESCRITURA_PLAN.md`: Fase 2.2-2.3 BLOQUEADA hasta validar 2012aw.
- El referee-positioning está en la memoria del agente y en §3.1.4: dato real >
  transportado > donante > fotométrico > continuo, todo con flag + error LOO,
  mangling ancla flujo en banda; punto débil declarado: SNe sin UV propio.

## 6. Bitácora Notion (Proyección `353d0d7d-848a-818b-94a5-df0ac0b34de9`)
Al día hasta "2026-08-23 (tarde/noche)". FALTA la entrada de la madrugada del 24:
validación numérica de 2012aw, bug misma-noche, pasada de tesis. Escribirla al
retomar. El resumen conceptual del comienzo sigue diciendo "Ia/II/Ibc, 3 tipos x 10
pivotes" — desactualizado, NO editarlo sin visto bueno de Mauricio.

## 7. Próximos pasos, en orden (REORDENADO 2026-08-30 con Mauricio)
El 2026-08-30 Mauricio APROBÓ visualmente las 2 figuras de 2012aw: el orden nuevo
es LA cadena de producción. Decisiones de la misma sesión:
- **No se lanza ninguna clase todavía**: primero se deja lista la DATA de todas
  las clases, después se lanza (orden pedido por él).
- **τ va antes del lanzamiento** (recomendación aceptada): D(fase) cambia el
  relleno de todas las SNe, lanzarlas antes obligaría a regenerar y re-revisar.
- **Fixes aplicados (sin commit aún)**: cero de fase Ia apunta al
  `maximum_perband.txt` del workspace (antes legacy de OT_unidos_2; resultó tener
  valores V idénticos para las 15, el fix es de fuente única, no de números) y
  `_pool_clase` de s1b quedó SOLO con donantes reales (`uv_dereddened/`), las
  series producidas no entran (circularidad template-de-template). Pool II
  real-only: 38 espectros UV. Pendiente conocido: s1b no persiste flags por
  época (agregarlo con τ).

1. Bitácora Notion de la madrugada del 24 + esta sesión (Notion pide re-auth).
2. Dejar data lista por clase: Loess de las 3 II históricas; s6b Ia
   (`uv_dereddened/Ia`); staging+Loess Ibc (22); datos IIb faltantes (4);
   descargas SLSN-I (6); montaje IIn (WISeREP + extinción + cero de fase).
3. Implementar D(fase) + mezcla τ con LOO (diseño §2.5) + flags por época en s1b
   → revalidar en 2012aw (compuertas + visto de Mauricio).
4. Lanzar las clases con data cerrada (Ia primero) por el orden nuevo.
5. Cadena 2012aw completa: mangling (s8) + compuerta de monotonía → comparación
   final vieja-vs-nueva.

## 8. Reglas de trabajo con Mauricio (no negociables)
- Figuras: MIRARLAS (Read) antes de entregar; entregar por SendUserFile con
  display render (Preview/open del Mac no le funciona).
- Revisión Loess: él aprueba banda por banda, se anota textual en loess_review.csv.
- Nada de atajos con datos de otros workspaces/épocas; nada de copias masivas a
  disco local; datos reales o declarar NO ENCONTRADO.
- Citas solo verificadas contra ADS/arXiv. Commits/push solo cuando él pida.
- Español chileno directo, sin rodeos; reportar errores propios sin maquillaje.
