# HANDOFF — estado y contexto completo del proyecto (2026-08-31, madrugada)

Documento de continuación: si la sesión de Claude murió, esto es todo lo que hay que
saber para retomar sin releer nada. Actualizado tras la maratón del 30-31 de agosto.

**ESTADO GLOBAL 2026-08-31: Fase A (data de las 6 clases) CERRADA. Fase B (τ)
IMPLEMENTADA, CALIBRADA y APROBADA por Mauricio (visto visual 2012aw v3,
"dale demosle con v3"). El sistema queda ESPERANDO SU ORDEN EXPLÍCITA de
lanzamiento (Ia primero). Nada se lanza sin esa orden.**

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

### 2.5 Decaimiento τ fuera de cobertura UV — IMPLEMENTADO y APROBADO (2026-08-31)
Relleno(fase) = baricentro OT entre P(fase) (forma del último UV real propio
re-anclada en nivel) y D(fase) (consenso suave de donantes REALES, CurvaD) con
peso w(Δfase) rampa coseno. NUNCA promedio de flujos, siempre transporte
(cuantiles W2 exactos, NQ=2048). Calibración MEDIDA, no supuesta: LOO con épocas
UV reales escondidas → τ_II=10 d (cruce en ~5 d, error mezcla 10.1% vs escalera
~13% vs persistencia 24.8%), Ia=None (persistencia pura, P gana todos los bines,
diversidad UV Foley+16). Licencia de span: τ solo cuando la costura queda a
≤ tgt_b+550 Å (rango medido del LOO), más allá manda la escalera. Código:
`core/tau_blend.py` (tests 7/7) + peldaño 1b en s1b + flags por época en
`Data/_s1b_flags.csv` (metodo/w/dfase_uv/err_esperado) + `config.TAU_BLEND` y
`config.ERR_RELLENO`. Compuertas de 2012aw v3: zona UV fusionado intacta
(2.2-3.6%), suavidad mediana 5.79% vs 6.17% de la aprobada, sin escalones,
rectángulo único 3005/10500. APROBADO por Mauricio tras comparar el zoom
aprobada-vs-tau ("dale demosle con v3"). Serie de referencia:
`dereddened/II_new_preot/SN2012aw.dat` (respaldos `_aprobada_20260830.dat` =
escalera, `_tau_aprobada_20260831.dat` = snapshot del v3 aprobado). Detalle
completo en `Codes/spectral_series/docs/design_tau.md`.

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

## 7. Próximos pasos (ACTUALIZADO 2026-08-31, madrugada)
Fase A cerrada en las 6 clases (ver §3 con las correcciones de abajo) y τ
aprobado (§2.5). Lo que queda, en orden:

1. **GATE DE CADENCIA GOLDEN por clase antes de cada lanzamiento** (requerimiento
   completo de Mauricio 2026-09-01: cobertura total + cadencia + gaps >50 d +
   subida pre-max para mapear el maximo; el sondeo de agosto solo cubria II/Ia).
   Inventario medido (rms_mad, mad<=0.10):
   - Ia (CORRIENDO, validada): subida intacta 15/15; solo colas (2009ig -58 d
     isla inemparejable, 2012fr -27 d) y el hoyo nebular declarado de 2011fe
     (81 d en +228/+309, decidir en QA si se rescata con relajacion por-SN).
   - GRAVES a decidir con Mauricio al lanzar cada clase: 1996cb (IIb) PIERDE LA
     SUBIDA + cobertura 95->57 + cadencia 3.5->21 d; 2004dk cobertura 282->38 y
     subida 3->1; 2013df -113 d; 2009jf -163 d; 2006T 35 de 121 d; 2015da
     -527 de 1448 d; hoyos interiores nuevos: 2005cs 61 d, 2005hg 63 d,
     2015bn 57 d, 2005cp 54 d, 2009ip 53+72 d, 2015da 52+86 d; subida
     debilitada 2->1: 2024hpj, 2011kg.
     [RESUELTO 2026-09-01, orden de Mauricio "modificar los cortes... podemos
     ser flexibles con el mad porq mangleamos": RESCATE ESTRUCTURAL
     implementado (steps/s1c_golden_rescue.py, commit f591d17). Epocas con
     MAD 0.10-0.20 entran SOLO con rol estructural (subida / hoyo>50d /
     cola encadenable), techo 0.20 porque el mangling corrige calibracion
     suave, no forma; entran con su MAD real al peso (~4x menos que una
     golden). Artefacto revisable: Data/golden_rescue.csv por workspace
     (Ia 3, II 3, IIb 6, Ibc 15, IIn 21, SLSN-I 12 epocas; el de Ibc
     incluye donantes fuera de muestra, inocuo). POST-RESCATE: ninguna SN
     sin subida; unico hoyo golden restante = 2011fe 81 d nebular
     +228/+309 (candidatas >0.20, declarado); coberturas truncadas
     incurables declaradas (2006T -86, 2013df -113, 2004dk -230, 2004gt
     -80, 2009jf -133, 2015da -512).
     RESUELTO tambien al relanzar la cadena Ia (2026-09-02): el crash de
     sesion obligo a re-correr s1b para las 15 CON el rescate ya activo, asi
     que no queda re-run pendiente de 2004eo/2012fr.]
   [FASE LENTA MEDIDA Y HABILITADA EN IIb/Ibc (2026-09-02, commit d3dc130,
   orden de Mauricio de cerrar el tema ahora). LOO de trios reales
   (tools/loo_fase_lenta.py): lo que degrada la interpolacion NO es el
   Delta_phi grande sino CRUZAR la transicion fotosferica->nebular
   (21.2% IIb / 14.6% Ibc contra 9.9%/8.6% de pares normales). Con ambas
   epocas nebulares (fase >= +40) el error vuelve al nivel normal: IIb
   50-80 d = 6.3%, Ibc 50-80 = 8.1%, 80-120 = 11.0%; se dispara despues
   (IIb 80-120 = 20.7%, 120-200 = 36.6%; Ibc 120-200 = 20.5%). Por eso el
   tope es POR CLASE: config.OT_SLOW_MAX_DELTA_BY_CLASS = IIb 80, Ibc 120.
   Habilita 82 pares nebulares nuevos en 13 SNe. Al lanzar IIb/Ibc hay que
   pasar --slow-extend <clase> --pair-parallel a s2.
   Fixes de camino: s2 resuelve la referencia de fase por clase (Ibc tiene
   maximum_perband.DAT, IIb no tiene archivo propio) y aborta si se pide
   --slow-extend sin --pair-parallel (antes se ignoraba EN SILENCIO).
   PENDIENTE DECLARADO: II y SLSN-I usan tope 200 sin haber pasado por este
   test. Medirlo antes de lanzarlas (cambia series ya aprobadas: consultar).]
2. **ESPERAR LA ORDEN DE MAURICIO para las clases restantes** (tras Ia: II,
   luego el resto), resolviendo el gate de cadencia + banderas golden
   (1999gi 69%, LSQ13zm 67%, 1996cb 27%, 2004gt 25%, 2018hti 35%).
2. Bitácora Notion del 30-31 (2 entradas grandes, autorizada; Notion pedía
   re-auth al cierre de la sesión).
3. Pendientes chicos declarados: número fino error→mag ZTF-g vs z para la
   defensa Ia; flag LSQ14mo host A_V=0.26 ("decidir"); grism Swift sin
   publicar (6 objetos, decisión uvotpy/pedir).
   [CERRADO 2026-08-31: τ/ERR_RELLENO de las clases del fondo, mismo LOO y
   métrica de II/Ia. IIn: τ=7 d MEDIDO (404 pares, cruce en bin 3-7,
   errores 0.05/0.055). SLSN-I: persistencia pura (P gana siempre, 6 pares,
   errores adoptados = análogo Ia 0.15/0.173, declarado). Ibc: sin LOO
   posible (solo 1994I), persistencia, error 0.25 = cota empírica interna de
   1994I a Δ24 d, declarado. Detalle en docs/design_tau.md, commit 49fdc93.]
   [CERRADO 2026-08-31: los 12 FITS CDS de 2007pk. 6 eran duplicados del
   staged, 1 grism IR sin banda para K, 2 nebulares (54713/54740) sin
   fotometría simultánea (curva termina en 54493) → descartes declarados.
   2 épocas NUEVAS staged y calibradas: 54417 (MAD 0.0005, golden, la mejor
   de la SN) y 54497 (no calibró, 4 d después del fin de la curva, fuera).
   2007pk queda 8 staged / 6 calibradas. Respaldo .pre_cds + ASCII de
   trazabilidad en nuevas_series_II/SN2007pk/spectra_raw/.]
4. Escritura de tesis de lo cerrado: clase IIn 11 SNe/ramas/regla sin-subida,
   datos por clase, tests y calibración de τ ("ya veremos cuando corresponda").
5. Mauricio envía el dossier IIn al profesor
   (`nuevas_series_II/curvas_IIn_espectros.html`, 11 SNe, nomenclatura inglesa).

**Correcciones al §3 (estado real 2026-08-31, manda esto sobre §3):**
- II: COMPLETA (13 lote nuevo + 3 históricas con Loess aprobado, s1 corrido).
- Ia: COMPLETA incluido s6b: 6/6 SNe con UV propio montado en `uv_dereddened/Ia`
  (119 épocas), gate U/u/B, grism cortado <2650 Å rest (aprobado).
- IIn: COMPLETA. Muestra final 11 = 9 históricas (regla sin-subida 13→9)
  + 2024hpj + 2021foa. Workspace propio "Ramirez M IIn", Loess 63 bandas
  aprobadas, cero de fase = máximo V (fallback r/R) 11/11, s1 11/11, extinción
  15 filas, UV montado (2009ip 20 ép, 1998S 4, 2010jl 4).
- IIb: workspace propio "Ramirez M IIb" (Barbara solo lectura), 12 SNe staged,
  fotometría LEGACY adoptada (incidente loess-sobre-loess: curvas legacy
  restauradas de .pre_s0b, guardia de crudeza instalada en s0b).
- Ibc: 22 staged, legacy adoptada, s1 previo válido.
- SLSN-I: COMPLETA (14; UV montado en 3: 2017egm, PTF12dam, 2011kg;
  Gaia16apd/2011ke sin ancla posible, declarado).
- Censos UV hasta 3000 Å: `censo_espectros_UV_Ia.md` y
  `censo_espectros_UV_IIn_SLSN.md` en `nuevas_series_II/`.
- Repo `Codes/spectral_series` es GIT PROPIO (~17 commits del arco 30-31).
- Punto único de revisión: `paper2_ZTF/REVIEW.html`.

## 8. Reglas de trabajo con Mauricio (no negociables)
- Figuras: MIRARLAS (Read) antes de entregar; entregar por SendUserFile con
  display render (Preview/open del Mac no le funciona).
- Revisión Loess: él aprueba banda por banda, se anota textual en loess_review.csv.
- Nada de atajos con datos de otros workspaces/épocas; nada de copias masivas a
  disco local; datos reales o declarar NO ENCONTRADO.
- Citas solo verificadas contra ADS/arXiv. Commits/push solo cuando él pida.
- Español chileno directo, sin rodeos; reportar errores propios sin maquillaje.
