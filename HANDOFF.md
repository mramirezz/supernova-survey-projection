---
name: estado-continuacion
description: "HANDOFF completo al 2026-08-24 madrugada — objetivo, estado por frente, pendientes y proximo paso, para retomar si la sesion murio"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1e3d437c-6d2b-4643-85e1-1c9e861c9b56
  modified: 2026-08-24T04:11:44.034Z
---

# HANDOFF — retomar aqui (escrito 2026-08-24 ~1 AM, antes de apagar el PC)

## Objetivo grande
Paper 2 / tesis: templates espectrales por OT para 6 clases (Ia, II, IIb, Ibc, IIn, SLSN-I) → proyeccion sobre ZTF → clasificador RF (features Villar) → tasas (SUDARE/LSST). La produccion de series esta BLOQUEADA por decision de Mauricio hasta que el valide visualmente 2012aw con el ORDEN NUEVO.

## Lo que quedo ESPERANDO SU RESPUESTA (primerisimo al retomar)
**Validacion visual de 2012aw orden nuevo.** Las 2 figuras estan mandadas al chat y en
`paper2_ZTF/figures_templates/qa_mangled/II/SN2012aw_preot_{cascada,serie}.png`.
Compuertas numericas YA superadas (envolvente 0%, oscilacion x1.86, sin espigas, rectangulo unico 3005 A — detalle en [[orden-extension-antes-de-ot]]). Si Mauricio da el OK visual:
1. Lanzar las 15 Ia por el orden nuevo (s1b→s2→s3→s4 con `SPECTRAL_SERIES_WORKSPACE` = workspace Ia y `SPECTRAL_SERIES_PRE_EXT=1`). ANTES: correr s6b para Ia (fuentes UV propias, no existe `uv_dereddened/Ia`).
2. Construir D(fase) + mezcla geodesica τ (diseño acordado, ver memoria orden-extension) con LOO.
3. Completar cadena 2012aw: mangling (s8) sobre la serie nueva + gate monotonia + comparacion final vieja-vs-nueva.

## Estado por frente
- **2012aw orden nuevo**: interior VALIDADO numericamente. Cadena corrio completa en dirs `_ext` + `dereddened/II_new_preot/`. Produccion vieja intacta.
- **Golden sample**: REGLA del pipeline (`core/golden.py`, `config.GOLDEN_MAD_MAX=0.10`, filtro en s2 y s1b). Bug pares misma-noche arreglado (`OT_MIN_DELTA_DAYS=0.5`).
- **Clase Ia**: 15/15 cerrada por datos — ver [[clase-ia-estado]]. Falta solo s6b (UV) y el desbloqueo.
- **Clase II**: lote nuevo 13 OK. Faltan Loess de las 3 historicas de la whitelist (1999gi, 2005cs, 2014cy) — SOLO templates se revisan, no las ~27 historicas donantes (regla de Mauricio).
- **Ibc (22)**: insumos completos en Practica2, falta staging+Loess+s1. **IIb (12)**: 8 con insumos, faltan datos de 2008aq/2011ei/2011fu/2013df. **SLSN-I (7)**: 1 con insumos, fuentes de las otras 6 ubicadas (busquedas del 22-23) sin descargar.
- **IIn (13 templates, 5 ramas)**: definida en papel (dossier `nuevas_series_II/candidatas_IIn.html`, mandado al profesor). CERO montaje en pipeline: falta bajar espectros WISeREP, fotometria, filas en tabla extincion, y DECIDIR convencion de cero de fase de la clase (propuesta: maximo medido en curva propia).
- **Tesis** (`phd_thesis/tesis_template`, repo git → remoto `overleaf`, rama master): caps 2-4 actualizados y pusheados (24537ac, 444b2c4). §3.1.3 Ia + §3.1.4 IIn escritos con tablas. Estructura declarada con `\pacha` donde falta. Compilar local: swap temporal a backend=bibtex (biber no instalado; RESTAURAR cls despues). ESCRITURA_PLAN.md tiene el bloqueo de la Fase 2.2-2.3.
- **Bitacora Notion (Proyeccion 353d0d7d-848a-818b-94a5-df0ac0b34de9)**: al dia hasta la entrada "2026-08-23 tarde/noche". FALTA la entrada de la madrugada (validacion 2012aw + bug misma-noche + pasada tesis) — escribirla al retomar.

## Procesos: NINGUNO vivo. Todo lo background termino antes de apagar. Nada que relanzar a ciegas.

## Recordatorios de forma (Mauricio)
Figuras: mirarlas yo (Read) antes de entregar + SendUserFile display render (no confiar en open/Preview). Revision Loess: el aprueba banda por banda, yo anoto en loess_review.csv. No inventar datos ni citas (ADS verificado). No copiar masivo a local (espacio). Espanol chileno directo; commits solo cuando pida.
