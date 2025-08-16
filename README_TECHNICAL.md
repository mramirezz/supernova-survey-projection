# 🔧 DOCUMENTACIÓN TÉCNICA DEL CÓDIGO
# ====================================

## 📋 OVERVIEW DEL SISTEMA

Este documento explica la **implementación técnica** y **decisiones de diseño** del código de simulación de SNe. Complementa el README científico con detalles específicos del código.

---

## 🏗️ ARQUITECTURA DEL SISTEMA

### **Flujo principal de ejecución:**

```
simple_runner.py
    ↓
run_custom_batch() → create_simple_config()
    ↓
run_scientific_batch(config) → ProfessionalBatchRunner.run_batch()
    ↓
create_run_parameters() → execute_single_run()
    ↓
dust_maps.py (sample_realistic_mw_extinction)
    ↓
core/correction.py (apply_corrections)
    ↓
Archivos .dat de templates
```

### **Módulos principales:**

- **`simple_runner.py`**: CLI y configuración de parámetros
- **`batch_runner.py`**: Orquestador principal de simulaciones
- **`core/correction.py`**: Correcciones astronómicas (extinción, K-correction)
- **`dust_maps.py`**: Sistema de mapas de polvo MW (consultas SFD98 reales)
- **`core/utils.py`**: Utilidades generales

---

## 🔄 FLUJO DE EJECUCIÓN DETALLADO

### **1. Inicialización (simple_runner.py)**
```python
# Ejemplo: python simple_runner.py --runs 1000 --sn-types Ia --seed 123

# Se crea configuración:
config = create_simple_config(
    n_runs=1000,
    sn_types=['Ia'],
    seed=123,
    # ... otros parámetros
)
```

### **2. Simulación por lotes (batch_runner.py)**
```python
# IMPORTANTE: Cada run es UNA supernova individual
# Flujo completo en run_batch():

runner = ProfessionalBatchRunner()
runner.run_batch(config)

for run_index in range(config.n_runs):  # 1000 supernovas independientes
    
    # 2.1 Crear parámetros para esta SN específica
    iteration_params = create_run_parameters(batch_config, run_index)
    # Incluye: template, redshift, ebmv_host, coordenadas, etc.
    
    # 2.2 Ejecutar simulación de UNA supernova
    success, results = execute_single_run(iteration_params)
    
    # Dentro de execute_single_run():
    # 2.3 Extinción MW (CLAVE: siempre n_samples=1)
    ebmv_mw = sample_realistic_mw_extinction(
        n_samples=1,  # ← UNA consulta SFD98 por SN
        random_state=batch_config.base_seed + run_index
    )
    
    # 2.4 Correcciones astronómicas
    corrected_magnitudes = apply_corrections(...)
    
    # 2.5 Cálculo de detectabilidad
    detections = calculate_detections(...)
```

### **3. Sistema de extinción MW (dust_maps.py)**

**DECISIÓN DE DISEÑO CLAVE:**
```python
# ❌ NO hacemos:
extinctions = sample_realistic_mw_extinction(n_samples=1000)  # 1000 consultas de una vez

# ✅ SI hacemos:
for each_sn in range(1000):
    params = create_run_parameters(batch_config, run_index)  # Parámetros únicos por SN
    success, results = execute_single_run(params)           # Simula 1 SN completa
    # Dentro de execute_single_run():
    extinction = sample_realistic_mw_extinction(n_samples=1)  # 1 consulta por SN
```

**Razones técnicas:**
1. **Reproducibilidad**: Cada SN tiene su seed único
2. **Robustez**: Si falla una consulta, no afecta las demás
3. **Memoria**: No acumular arrays grandes
4. **Paralelización futura**: Fácil distribución

**Implicación técnica:**
```python
# En sample_realistic_mw_extinction():
if n_samples > 1:  # ← NUNCA se ejecuta en producción
    print("Estadísticas...")  # Solo para análisis científicos
```

---

## 🧩 DECISIONES DE IMPLEMENTACIÓN

### **1. ¿Por qué n_samples=1 siempre?**

**Contexto:**
- `simple_runner.py --runs 1000` = 1000 supernovas independientes
- Cada SN necesita 1 valor de extinción MW específico

**Implementación:**
```python
# En execute_single_run() para cada SN individual:
ebmv_mw_sample = sample_realistic_mw_extinction(
    n_samples=1,  # Retorna np.array([0.043])
    random_state=iteration_params['seed']
)

# ebmv_mw_sample.shape = (1,)  # Array de 1 elemento
# ebmv_mw_sample[0] = 0.043    # Valor escalar usado en correcciones
```

### **2. ¿Cuándo se usa n_samples > 1?**

**Solo en funciones de análisis:**
```python
# Validación científica:
validate_mw_extinction_distribution(n_test=1000)  # Estadísticas globales

# Visualizaciones:
create_extinction_map_visualization(n_samples=5000)  # Mapas para paper

# Histogramas:
create_extinction_histogram_publication(n_samples=10000)  # Distribuciones
```

### **3. Sistema de logging inteligente**

```python
# Para n_samples=1 (producción):
print("🗺️ Consultando mapas SFD98 reales para 1 campos...")
print("   ✅ Consultas completadas: 1/1 (100.0%)")
# NO imprime estadísticas (sin sentido para 1 valor)

# Para n_samples>1 (análisis):
print("🗺️ Consultando mapas SFD98 reales para 1000 campos...")
print("   ✅ Consultas completadas: 950/1000 (95.0%)")
print("   📊 E(B-V)_MW - Rango: 0.010 - 0.180 mag")
print("               Media: 0.055 ± 0.025 mag")
```

### **4. Gestión de errores con success flags**

**Problema original:**
```python
# ❌ Problema: E(B-V)=0.05 contado como error
if extinction_value == 0.05:  # ¡Podía ser valor real de SFD98!
    failed_queries += 1
```

**Solución implementada:**
```python
# ✅ Solución: Flag explícito de éxito
def get_sfd98_extinction_real(ra, dec):
    try:
        # Consulta real a IRSA
        ebv_mw = query_sfd98(ra, dec)
        return ebv_mw, True  # (valor, success=True)
    except:
        return 0.05, False   # (fallback, success=False)

# Uso:
ebmv, success = get_sfd98_extinction_real(ra, dec)
if not success:  # Solo cuenta errores reales
    failed_queries += 1
```

### **5. Semillas reproducibles**

**Implementación:**
```python
# En run_batch() - cada SN tiene semilla única pero determinista:
for run_index in range(n_runs):
    iteration_params = create_run_parameters(batch_config, run_index)
    # iteration_params['seed'] = batch_config.base_seed + run_index
    
    # SN #1: seed = 123 + 0 = 123
    # SN #2: seed = 123 + 1 = 124  
    # SN #3: seed = 123 + 2 = 125
    
    success, results = execute_single_run(iteration_params)
```

**Verificación reproducibilidad:**
```bash
# Mismo resultado siempre:
python simple_runner.py --runs 10 --seed 123
python simple_runner.py --runs 10 --seed 123
```

---

## 📁 ESTRUCTURA DE ARCHIVOS

### **Templates de SNe:**
```
data/Ia/
├── SN1994D.dat     # Template tipo Ia "normal"
├── SN2011fe.dat    # Template tipo Ia "rápido"
└── ...

data/IIn/
├── SN2006gy.dat    # Template tipo IIn
└── ...
```

### **Datos auxiliares:**
```
data/
├── grid_diffmaglim_ZTF.csv  # Límites de detección ZTF
├── obslog_I.csv             # Log de observaciones
└── ...
```

### **Outputs:**
```
outputs/
├── batch_YYYYMMDD_HHMMSS/   # Resultados por batch
│   ├── results.csv          # Datos principales
│   ├── metadata.json        # Configuración usada
│   └── extinction_maps/     # Mapas opcionales
└── ...
```

---

## 🐛 CASOS EDGE Y DEBUGGING

### **1. Consultas SFD98 fallidas**

**Síntomas:**
```
Warning: Error consultando IRSA para RA=123.456, Dec=45.678: Timeout
Usando valor promedio E(B-V)=0.05 mag
```

**Causas comunes:**
- Sin conexión a internet
- Servidor IRSA caído
- Coordenadas fuera de rango

**Manejo:**
- Fallback automático a E(B-V)=0.05 mag
- Contador de errores para diagnóstico
- Continúa simulación sin interrumpir

### **2. Templates faltantes**

**Error típico:**
```python
FileNotFoundError: Template 'SN2023abc.dat' not found
```

**Solución:**
- Verificar contenido de `SN_TEMPLATES` dict
- Comprobar paths relativos vs absolutos
- Validar formato de archivos .dat

### **3. Debugging y profiling**

**Logging levels:**
```python
# Producción (mínimo):
print("🗺️ Consultando mapas SFD98 reales para 1 campos...")

# Debug (detallado):
if DEBUG:
    print(f"RA={ra:.3f}, Dec={dec:.3f}, E(B-V)={ebv:.3f}")
```

**Performance:**
- **Cuello de botella**: Consultas IRSA online (~1-2 seg por consulta)
- **Escalabilidad**: `execute_single_run()` es independiente → fácil paralelización
- **Optimización futura**: Batch de consultas SFD98 para análisis (no implementado)
- **Alternativa**: Mapas SFD98 locales (requiere ~100MB descarga)

**Memory usage:**
```python
# ✅ Eficiente (producción):
for run_index in range(1000):
    params = create_run_parameters(batch_config, run_index)
    success, results = execute_single_run(params)  # Procesa 1 SN
    # Dentro: extinction = sample_realistic_mw_extinction(n_samples=1)  # Array(1,)
    
# ❌ Ineficiente (evitado):
all_extinctions = sample_realistic_mw_extinction(n_samples=1000)  # Array(1000,)
```

---

## 🚀 FUTURAS MEJORAS

### **1. Paralelización:**
```python
# Actual (secuencial):
for run_index in range(1000):
    params = create_run_parameters(batch_config, run_index)
    success, results = execute_single_run(params)

# Futuro (paralelo):
from multiprocessing import Pool

def run_single_sn(run_index):
    params = create_run_parameters(batch_config, run_index)
    return execute_single_run(params)

with Pool() as pool:
    results = pool.map(run_single_sn, range(1000))
```

### **2. Mapas locales:**
```python
# Actual: Consulta online
ebv = query_irsa_online(ra, dec)  # Lento

# Futuro: Lookup local
ebv = sfd98_local_map[ra_idx, dec_idx]  # Rápido
```

### **3. Caching inteligente:**
```python
# Cache consultas frecuentes
@lru_cache(maxsize=1000)
def get_sfd98_extinction_cached(ra_rounded, dec_rounded):
    return get_sfd98_extinction_real(ra_rounded, dec_rounded)
```

---

## 📚 REFERENCIAS TÉCNICAS

### **APIs utilizadas:**
- **astroquery.irsa_dust**: Consultas a mapas SFD98
- **numpy.random**: Generación de números aleatorios
- **pandas**: Manejo de datos tabulares

### **Estándares seguidos:**
- **PEP 8**: Estilo de código Python
- **Docstrings**: Documentación de funciones
- **Type hints**: Anotaciones de tipos (parcial)

### **Testing:**
```bash
# Validación básica:
python dust_maps.py  # Test incluido en __main__

# Validación completa:
python simple_runner.py --runs 5 --seed 42  # Smoke test
```

---

## 🎯 CONCLUSIÓN

Este sistema está diseñado para:

1. **Robustez**: Manejo de errores sin interrumpir simulaciones
2. **Reproducibilidad**: Semillas deterministas por SN
3. **Escalabilidad**: Fácil paralelización futura
4. **Mantenimiento**: Código modular y bien documentado

**Para nuevos desarrolladores:**
1. Leer README.md (contexto científico)
2. Leer este README_TECHNICAL.md (implementación)
3. Ejecutar smoke test: `python simple_runner.py --runs 5`
4. Revisar outputs en `outputs/` folder

**Para debugging:**
1. Verificar conectividad con `python dust_maps.py`
2. Comprobar templates disponibles en `SN_TEMPLATES`
3. Validar semillas con runs pequeños (`--runs 2`)

¡Happy coding! 🚀
