# Documentación Complementaria

Archivos de referencia técnica y derivaciones que complementan el [README principal](../README.md).

| Archivo | Contenido |
|---|---|
| `PIPELINE_PROYECCION.tex` | Documento LaTeX con el pipeline paso a paso, ecuaciones y ejemplos numéricos. Las ecuaciones de física son vigentes; los snippets de código pueden referenciar funciones legacy (ver README principal para la arquitectura actual). |
| `README_LATEX.tex` | Documento LaTeX con las distribuciones científicas detalladas: redshift volume-weighted, extinción host (modelo mixto), extinción MW (SFD98), y criterios de detección. Todos los parámetros coinciden con el código actual. |
| `Explicacion_Distribuciones_Extincion.ipynb` | Notebook Jupyter con derivaciones matemáticas, código de validación y visualizaciones de las distribuciones de extinción implementadas. Útil para verificar que los samplers producen las distribuciones esperadas. |
| `FAILURES_20260125_231012.md` | Análisis del batch de Enero 2026: reglas de fallo por redshift/extinción según tipo de SN. Las reglas de límites de z por tipo y los patrones de fallo por overlap espectral son referencia universal. |

## Documentos obsoletos

Los siguientes documentos están en `legacy/docs/` y referencian la arquitectura anterior (batch runners, single-band pipeline):

- `ARQUITECTURA_MULTIBAND.md` — estructura de outputs del sistema batch anterior
- `README_MAPEO_SISTEMA.md` — mapeo de funciones del pipeline antiguo
- `README_TECHNICAL.md` — decisiones de diseño del sistema batch
