# HANDOFF 2026-09-30: de la biblioteca congelada a la proyección ZTF

Para el chat que retoma el trabajo. Léelo entero antes de tocar nada. Complementa la memoria del proyecto
(`estado-continuacion.md` y el resto del índice `MEMORY.md`), no la reemplaza.

## 1. Dónde estamos

**Las series espectrales están cerradas.** Son 78 templates aprobados por Mauricio uno a uno y congelados con md5.

| Clase | Congeladas | Cero de fase |
|---|---|---|
| Ia | 15 | máximo en V |
| II | 13 | explosión |
| IIb | 10 | máximo |
| IIn | 10 | máximo (2009ip: máximo del evento 2012b) |
| Ibc | 30 (Ib, Ic y 7 Ic-BL) | máximo |
| SLSN-I | 12 **pendientes, sin revisar** | máximo en r |

- Las SLSN-I quedaron aparcadas por decisión de Mauricio: SUDARE no tiene superluminosas.
- **Productos:** `paper2_ZTF/series_aprobadas/<clase>/{rest,dereddened,mangled}/<SN>.dat`, más `MANIFIESTO.csv` (sn, clase, md5, notas de aprobación). El producto final es `mangled`: reposo, desenrojecido, a 10 pc, con formato de bloques `# time:` / `# SPEC` / `WAVE FLUX`, el mismo que lee la proyección.
- **Validación:** fotometría sintética contra las curvas, mediana 0.012 mag por objeto (sin bandas vetadas). El 82 % de los 1686 días con espectro observado cumple las cuatro tolerancias de líneas.

**La tesis está al día y en Overleaf.** Commits `3eba856` y `c26fc76` en master, remoto `overleaf`.
- El capítulo 3 describe el método completo.
- El capítulo 4, en §4.1 y §4.2, tiene la biblioteca, dos ejemplos por clase, la deformación de líneas y la fotometría sintética.
- **La sección ZTF del capítulo 4 sigue "preliminar":** se hizo con templates de un run anterior (run_1000_v2, 2 de septiembre).

**El código de las series** está en la rama `cierre-series-Ibc-20260930` del repo `spectral_series`, ya pusheada, sin merge a master.

## 2. El siguiente paso: la proyección ZTF con la biblioteca final

Hoy la proyección NO usa las 78. En `config.py`, `TEMPLATE_DIRS` apunta a `data/{Ia,II,Ibc,IIb,SLSN-I}_new`. Son carpetas del 15 de agosto que salieron de la campaña EMPCA + mangling GP, un método que después se descartó. IIn no está como clase en la proyección.

Pasos:
1. **Cargar las 78 en carpetas nuevas de `data/`,** por ejemplo `data/<clase>_v78`, copiando desde `series_aprobadas/<clase>/mangled/`. No sobrescribir las carpetas `_new` y apuntar `TEMPLATE_DIRS` a las nuevas. Luego contar los .dat por clase y comparar md5 contra el MANIFIESTO.
2. **Recalcular lo que la proyección saca de cada template:**
   - `precompute_dm15_Ia.py` genera `data/dm15_Ia.json`, que es la relación de Phillips en `PHILLIPS_CONFIG`.
   - `precompute_Mrest_SLSN.py` solo aplica si entran las SLSN-I, y hoy no entran.
   - Revisar `LUMINOSITY_CONFIG`: IIn no tiene distribución de M_peak. Si IIn entra, necesita una de la literatura, con bibcode verificado en ADS.
3. **Revisar la escala de flujo.** Las congeladas están a 10 pc en erg s⁻¹ cm⁻² Å⁻¹, del orden de 5e-4 en 2011fe, mientras que `Ia_new` está del orden de 0.09. Verificar que la normalización de luminosidad de `run_per_field.py` reescala cada template al M_peak sorteado y no dependa de la escala de entrada.
4. **Correr la proyección ZTF, extraer los parámetros SPM de Villar y reentrenar el Random Forest.**
   - Esto es masivo. Mauricio decide cuándo se lanza, y no se lanza nada por iniciativa propia.
   - Protocolo del clasificador, en `feature_extraction/opt_clasificador/`: se entrena solo con sintéticas. Las 675 SNe reales de ZTF con clase de TNS se parten 50/50 con semilla 42: 337 de validación para las decisiones y el offset de M, y 338 finales que se miden una sola vez. La línea base es el run_1000_v2, con accuracy 0.747.
5. **Actualizar la sección ZTF del capítulo 4** con los números nuevos y sacarle "preliminar".

**Decisiones abiertas que son de Mauricio:**
- Cómo entran las clases al clasificador de tres tipos. La propuesta es IIb dentro de II, como hace SUDARE. Falta decidir si IIn va con II o queda fuera. Revisar además cómo mapea hoy `feature_extraction/classifier_config.py`.
- Si alguna de las tres congeladas con defectos se reabre:
  - 2016coi, con un salto de 0.5 mag en u en el día +0;
  - 2012fr, con U y u dentadas de 0.2 a 0.4 mag entre +30 y +95 d;
  - 2007gr, cuyo espectro observado de +106 d (MJD 54445) está dominado por un continuo tipo host.

## 3. Después: SUDARE

- El plan está en `paper2_ZTF/obslogsudare/PLAN_prueba_clasificador_sudare.md`. La muestra tiene 169 SNe utilizables (Ia 105, II 29, IIn 18, Ibc 17), con z mediano 0.38. Pignata pidió probar primero con z y después sin z.
- **Hay que proyectar la biblioteca sobre los logs de OmegaCAM** (g, r, i; z entre 0.1 y 0.8).
- **Problema físico por resolver antes:** los templates cubren 3000 a 9200 Å en reposo. Sobre z ≈ 0.33, el borde azul de g (cerca de 4000 Å observado) cae bajo los 3000 Å en reposo, y la proyección exige 95 % de cobertura del filtro. Con la mediana de z en 0.38, más de la mitad de la muestra quedaría sin g. Primero hay que medirlo con las curvas de filtro reales y después decidir: r e i solamente en alto z, o extender al UV los objetos con Swift.
- Después vienen la eficiencia η(z) y las tasas (capítulo 4, §rates, y capítulo 5).

## 4. Reglas que no se negocian

Todas están en la memoria. Estas son las que más se rompen:
- **Congeladas:** jamás recalcular ni sobrescribir una serie congelada o un loess aprobado sin orden explícita.
- **Runs:** nunca lanzar un run masivo por iniciativa propia. Una sola corrida pesada a la vez, porque el Mac tiene 16 GB, y avisar antes si va a durar más de 10 minutos. Hoy la swap está casi llena, con 13 de 14 GB.
- **Drive:**
  - copiar, verificar y recién ahí borrar, archivo por archivo;
  - nunca `mv` seguido de `rm -rf`;
  - un md5 sobre un archivo a medio hidratar da un hash falso.
- **Git:**
  - commit y push solo cuando Mauricio lo pide;
  - si estás en la rama por defecto, crear una rama antes;
  - jamás `Co-Authored-By`.
- **Citas:** solo bibcodes verificados en ADS o arXiv. Nunca inventar.
- **Estilo de la tesis:** inglés, registro A&A, sin punto y coma y sin em-dash en la prosa.
- **Figuras:** estilo publicable, en inglés, con fuente serif, vectorial en PDF y copiadas a `phd_thesis/tesis_template/figuras/`.
- **Notion:** el conector no está autorizado. Mauricio tiene que autorizarlo en la configuración de conectores de claude.ai, y mientras tanto no hay bitácora.
- **OT en la tesis, decisión de Mauricio del 2026-09-30:** el método se presenta como el baricentro OT del Paper 1, descrito por sus propiedades medidas: continuo tipo cuerpo negro, flujo positivo y ventaja sobre la interpolación lineal en huecos largos. No agregar al texto ni a las figuras comparaciones con promedios punto a punto ni caracterizaciones del interpolante como una media. Tampoco escribir que el OT desliza las líneas, porque es falso.
- **Idioma:** con Mauricio se habla en español chileno, directo.
