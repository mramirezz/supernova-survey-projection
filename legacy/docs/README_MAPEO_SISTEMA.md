# MAPEO COMPLETO DEL SISTEMA DE SIMULACIÓN
## Flujo de Ejecución End-to-End con Nombres de Funciones

> **Documentación Técnica Completa**  
> Sistema de simulación de detectabilidad de supernovas  
> Pipeline desde CLI hasta resultados científicos finales

---

## 📋 **RESUMEN EJECUTIVO - FLUJO COMPLETO**

### **🎯 Pipeline End-to-End en 5 Niveles:**

**1. ENTRADA CLI** → **2. CONFIGURACIÓN** → **3. MUESTREO CIENTÍFICO** → **4. PIPELINE FÍSICO** → **5. AGREGACIÓN ESTADÍSTICA**

### **🔍 Secuencia Detallada de Ejecución:**

1. **`simple_runner.py`** inicia el sistema:
   - **Importa funciones**: `create_simple_config()` desde el archivo `simple_config.py`
   - **Función `create_simple_config()`**: Retorna un objeto de clase `SimpleConfig` con parámetros validados
   - **Ejecuta `main()` [simple_runner.py]**: Entry point principal del sistema
     - **`parse_arguments()` [SE EJECUTA DENTRO DE main()]**: Parsea argumentos CLI
     - **`setup_environment()` [SE EJECUTA DENTRO DE main()]**: **CREA CARPETAS** para preparar las salidas en `outputs/` (single_runs/, batch_runs/, logs/, etc.)
     - **`run_custom_batch()` [SE EJECUTA DENTRO DE main()]**: Recibe inputs de los parámetros de línea de comandos parseados

2. **Transición a Batch Runner:**
   - **Dentro de `run_custom_batch()` [simple_runner.py]**: Se crea la config con `create_simple_config()` usando parámetros CLI
   - **Con la config creada [DENTRO DE run_custom_batch()]**: Se ejecuta `run_scientific_batch(config)` 
   - **Se transfiere control**: Del archivo `simple_runner.py` al archivo `batch_runner.py`
   - **`run_scientific_batch()` [batch_runner.py]**: Función wrapper que instancia `ProfessionalBatchRunner()` 
     - **`run_batch()` [SE EJECUTA DENTRO DE run_scientific_batch()]**: Método principal del batch runner

3. **Generación de Parámetros Científicos**:
   - **`run_batch()` [ProfessionalBatchRunner]**: Loop principal de iteraciones
     - **`create_run_parameters()` [SE EJECUTA DENTRO DE run_batch()]**: Muestreo científico por iteración
       - **`sample_cosmological_redshift()` [SE EJECUTA DENTRO DE create_run_parameters()]**: Muestreo cosmológico volume-weighted
       - **`sample_extinction_by_type()` [SE EJECUTA DENTRO DE create_run_parameters()]**: Distribuciones académicas por tipo SN
       - **`get_sn_templates()` [SE EJECUTA DENTRO DE create_run_parameters()]**: Obtiene templates disponibles
       - **`np.random.choice(available_templates)` [SE EJECUTA DENTRO DE create_run_parameters()]**: Selección **ALEATORIA** del template
       - **`sample_realistic_mw_extinction()` [SE EJECUTA DENTRO DE create_run_parameters()]**: Extinción MW que llama **INTERNAMENTE**:
         - Muestrear coordenadas de ZTF (actualmente sintéticas)
         - Consultar mapas de extinción realistas SFD98
         - **NOTA FUTURA**: No usa coordenadas reales del objeto ZTF pero sería bueno implementarlo
   - **Resultado**: Finalmente tenemos **TODOS** los parámetros guardados en `iteration_params`

4. **Ejecución Individual**:
   - **`execute_single_run()` [SE EJECUTA DENTRO DE run_batch()]**: Recibe `iteration_params`
     - **`update_config_for_run()` [SE EJECUTA DENTRO DE execute_single_run()]**: **ACTUALIZA** los parámetros de `config.py` global
     - **`load_and_validate_config()` [SE EJECUTA DENTRO DE execute_single_run()]**: Confirma que no haya errores en configuración
     - **`import main` [SE EJECUTA DENTRO DE execute_single_run()]**: Importa el módulo main.py
     - **`main.main(config=validated_config)` [SE EJECUTA DENTRO DE execute_single_run()]**: Llamada al pipeline científico

5. **Pipeline Científico Completo** (en `main.py` viene TODO el pipeline de proyección):
   - **`main()` [main.py]**: Función principal del pipeline científico
     - **PASO 1 [SE EJECUTA DENTRO DE main()]**: **LEE LOS ESPECTROS**
       - **`leer_spec()` [SE EJECUTA EN PASO 1]**: Lee archivo de template seleccionado
     - **PASO 2 [SE EJECUTA DENTRO DE main()]**: **APLICA CORRECCIONES COSMOLÓGICAS**
       - **`correct_redeening()` [SE EJECUTA EN PASO 2]**: Aplica redshift + extinción física
     - **PASO 3 [SE EJECUTA DENTRO DE main()]**: **CARGA RESPUESTA DE FILTRO**
       - **`pd.read_csv(path_response)` [SE EJECUTA EN PASO 3]**: Carga curva respuesta fotométrica
     - **PASO 4 [SE EJECUTA DENTRO DE main()]**: **FOTOMETRÍA SINTÉTICA**
       - **`Syntetic_photometry_v2()` [SE EJECUTA EN PASO 4]**: Overlap 95% entre espectro y filtro
     - **PASO 5 [SE EJECUTA DENTRO DE main()]**: **SUAVIZADO LOESS**
       - **`Loess_fit()` [SE EJECUTA EN PASO 5]**: Interpolación estadística de curva de luz
     - **PASO 6 [SE EJECUTA DENTRO DE main()]**: **CALIBRACIÓN FOTOMÉTRICA**
       - **Conversión flujo → magnitudes [SE EJECUTA EN PASO 6]**: Usando constantes de punto cero
     - **PASO 7 [SE EJECUTA DENTRO DE main()]**: **RUIDO FOTOMÉTRICO**
       - **`np.random.normal()` [SE EJECUTA EN PASO 7]**: Simula Poisson + Gaussiana
     - **PASO 7.5 [SE EJECUTA DENTRO DE main()]**: **CONVERSIÓN TEMPORAL CRÍTICA**
       - **`maximo_lc()` [SE EJECUTA EN PASO 7.5]**: Obtiene MJD del máximo
       - **Conversión fases → MJD [SE EJECUTA EN PASO 7.5]**: Para SNe core-collapse

6. **⚠️ Bug Temporal Resuelto** (Paso 7.5 - DETALLES COMPLETOS):
   ```python
   # DATOS DE ENTRADA (antes del PASO 7.5):
   fases = [-10, 0, +20]     # ← Fases relativas (días respecto al máximo)
   maximum = 53671           # ← MJD absoluto del máximo (función maximo_lc)
   mjd_pivote = 59000       # ← MJD de observaciones modernas ZTF

   # CÁLCULO ERRÓNEO:
   desplazamiento = 59000 - 53671 + offset ≈ 5329 días
   fases_ajustadas = [-10 + 5329, 0 + 5329, +20 + 5329]
                   = [5319, 5329, 5349]  # ← ¡Fechas imposibles año 1846!
   
   # SOLUCIÓN: Conversión explícita antes de proyección
   if tipo in ['Ibc', 'Ib', 'Ic']:
       mjd_absolute = maximum + lc_df['fase']  # Fases → MJD absoluto
       lc_df['fase'] = mjd_absolute           # Actualizar DataFrame
   ```

     - **PASO 8 [SE EJECUTA DENTRO DE main()]**: **PROYECCIÓN TEMPORAL**
       - **`field_projection()` [SE EJECUTA EN PASO 8]**: Proyección sobre observaciones reales del survey
         - **`maximo_lc()` [SE EJECUTA DENTRO DE field_projection()]**: Calcula máximo de la SN
         - **`interpolate.interp1d()` [SE EJECUTA DENTRO DE field_projection()]**: Interpola magnitudes en tiempos observación
     - **PASO 9 [SE EJECUTA DENTRO DE main()]**: **RESUMEN CON MÉTRICAS**
       - **Cálculo detecciones/upper limits [SE EJECUTA EN PASO 9]**: Análisis de detectabilidad
     - **PASO 10 [SE EJECUTA DENTRO DE main()]**: **GUARDAR RESULTADOS**
       - **`save_projection_results()` [SE EJECUTA EN PASO 10]**: Persistencia individual
     - **PASO 11 [SE EJECUTA DENTRO DE main()]**: **ACTUALIZAR ÍNDICE**
       - **`create_master_index()` [SE EJECUTA EN PASO 11]**: Actualización índice maestro

7. **Lo que pasa DESPUÉS de main** (retorno y extracción de resultados):
   - **RETORNO A `execute_single_run()` [batch_runner.py]**: Después de completar main.main()
     - **`extract_run_statistics()` [SE EJECUTA DENTRO DE execute_single_run()]**: Busca CSV generado por `save_projection_results()`
       - **`glob.glob()` [SE EJECUTA DENTRO DE extract_run_statistics()]**: Busca archivos summary más recientes
       - **`pd.read_csv()` [SE EJECUTA DENTRO DE extract_run_statistics()]**: Lee métricas reales del CSV
     - **Retorna estadísticas confirmadas [DESDE execute_single_run()]**: No estimadas, sino valores reales
     - **Timing de ejecución [EN execute_single_run()]**: Se calcula tiempo total de la iteración

8. **Agregación por Batch** (retorno a `run_batch()`):
   - **RETORNO A `run_batch()` [ProfessionalBatchRunner]**: Después de execute_single_run()
     - **`add_successful_run()` [SE EJECUTA DENTRO DE run_batch()]**: Acumula estadísticas globales
       - **Suma detecciones totales [DENTRO DE add_successful_run()]**: Acumulación estadística
       - **Actualiza distribuciones por tipo [DENTRO DE add_successful_run()]**: Para análisis científico
     - **Registro en `run_registry` [DENTRO DE run_batch()]**: Lista completa con TODAS las iteraciones

9. **Persistencia Final**:
   - **AL FINAL DE `run_batch()` [ProfessionalBatchRunner]**: Después del loop de iteraciones
     - **`save_batch_results()` [SE EJECUTA DENTRO DE run_batch()]**: Guardado agregado final
       - **`json.dump(batch_metadata)` [SE EJECUTA DENTRO DE save_batch_results()]**: Guarda configuración completa
       - **`df_runs.to_csv()` [SE EJECUTA DENTRO DE save_batch_results()]**: CSV con todas las iteraciones
       - **Generación de reportes [DENTRO DE save_batch_results()]**: Archivos de resumen estadístico

### **🎯 Resultado Final:**
- **Simulaciones individuales** con reproducibilidad total
- **Estadísticas agregadas** publication-ready
- **Trazabilidad completa** desde CLI hasta métricas científicas
- **Sistema robusto** con fallbacks y validación en cada nivel

---

## � **MAPEO JERÁRQUICO COMPLETO DE FUNCIONES**

### **📋 Mapeo Específico de Funciones por Archivo**

#### **`simple_runner.py`** - Entry Point y Control CLI
```python
main()                           # Entry point principal del sistema
├── parse_arguments()            # [EJECUTA DENTRO DE main()]
├── setup_environment()          # [EJECUTA DENTRO DE main()] - Crea estructura de carpetas
└── run_custom_batch()           # [EJECUTA DENTRO DE main()] - Coordina ejecución batch
    ├── create_simple_config()   # [EJECUTA DENTRO DE run_custom_batch()]
    └── run_scientific_batch()   # [EJECUTA DENTRO DE run_custom_batch()]
```

#### **`batch_runner.py`** - Coordinación de Batch y Muestreo Científico
```python
run_scientific_batch(config)                           # Wrapper function
└── ProfessionalBatchRunner.run_batch()                # [EJECUTA DENTRO DE run_scientific_batch()]
    ├── create_run_parameters()                        # [EJECUTA DENTRO DE run_batch()]
    │   ├── sample_cosmological_redshift()             # [EJECUTA DENTRO DE create_run_parameters()]
    │   ├── sample_extinction_by_type()                # [EJECUTA DENTRO DE create_run_parameters()]  
    │   ├── get_sn_templates()                         # [EJECUTA DENTRO DE create_run_parameters()]
    │   ├── np.random.choice(available_templates)      # [EJECUTA DENTRO DE create_run_parameters()]
    │   └── sample_realistic_mw_extinction()           # [EJECUTA DENTRO DE create_run_parameters()]
    │       ├── sample_ztf_coordinates()               # [EJECUTA DENTRO DE sample_realistic_mw_extinction()]
    │       └── query_sfd98_extinction_maps()          # [EJECUTA DENTRO DE sample_realistic_mw_extinction()]
    ├── execute_single_run(iteration_params)           # [EJECUTA DENTRO DE run_batch()]
    │   ├── update_config_for_run()                    # [EJECUTA DENTRO DE execute_single_run()]
    │   ├── load_and_validate_config()                 # [EJECUTA DENTRO DE execute_single_run()]
    │   ├── main.main(config=validated_config)         # [EJECUTA DENTRO DE execute_single_run()]
    │   └── extract_run_statistics()                   # [EJECUTA DENTRO DE execute_single_run()]
    │       ├── glob.glob()                            # [EJECUTA DENTRO DE extract_run_statistics()]
    │       └── pd.read_csv()                          # [EJECUTA DENTRO DE extract_run_statistics()]
    ├── add_successful_run()                           # [EJECUTA DENTRO DE run_batch()]
    │   ├── accumulate_detection_stats()               # [EJECUTA DENTRO DE add_successful_run()]
    │   └── update_type_distributions()                # [EJECUTA DENTRO DE add_successful_run()]
    └── save_batch_results()                           # [EJECUTA DENTRO DE run_batch()]
        ├── json.dump(batch_metadata)                  # [EJECUTA DENTRO DE save_batch_results()]
        ├── df_runs.to_csv()                           # [EJECUTA DENTRO DE save_batch_results()]
        └── generate_summary_reports()                 # [EJECUTA DENTRO DE save_batch_results()]
```

#### **`main.py`** - Pipeline Científico (11 Pasos)
```python
main(config)                                          # Pipeline científico principal
├── PASO 1: leer_spec()                               # [EJECUTA DENTRO DE main()]
├── PASO 2: correct_redeening()                       # [EJECUTA DENTRO DE main()]
├── PASO 3: pd.read_csv(path_response)                # [EJECUTA DENTRO DE main()]
├── PASO 4: Syntetic_photometry_v2()                  # [EJECUTA DENTRO DE main()]
├── PASO 5: Loess_fit()                               # [EJECUTA DENTRO DE main()]
├── PASO 6: flujo_to_magnitudes()                     # [EJECUTA DENTRO DE main()]
├── PASO 7: np.random.normal()                        # [EJECUTA DENTRO DE main()]
├── PASO 7.5: temporal_conversion_critical()          # [EJECUTA DENTRO DE main()]
│   ├── maximo_lc()                                   # [EJECUTA DENTRO DE temporal_conversion_critical()]
│   └── phase_to_mjd_conversion()                     # [EJECUTA DENTRO DE temporal_conversion_critical()]
├── PASO 8: field_projection()                        # [EJECUTA DENTRO DE main()]
│   ├── maximo_lc()                                   # [EJECUTA DENTRO DE field_projection()]
│   └── interpolate.interp1d()                        # [EJECUTA DENTRO DE field_projection()]
├── PASO 9: calculate_detection_metrics()             # [EJECUTA DENTRO DE main()]
├── PASO 10: save_projection_results()                # [EJECUTA DENTRO DE main()]
└── PASO 11: create_master_index()                    # [EJECUTA DENTRO DE main()]
```

#### **`core/correction.py`** - Funciones de Corrección Física
```python
correct_redeening()                                   # Correcciones cosmológicas principales
├── apply_redshift_correction()                       # [EJECUTA DENTRO DE correct_redeening()]
├── apply_host_extinction()                           # [EJECUTA DENTRO DE correct_redeening()]
└── apply_mw_extinction()                             # [EJECUTA DENTRO DE correct_redeening()]
```

#### **`core/utils.py`** - Utilidades y Proyección
```python
field_projection()                                    # Proyección temporal sobre observaciones
├── load_ztf_observations()                          # [EJECUTA DENTRO DE field_projection()]
├── maximo_lc()                                       # [EJECUTA DENTRO DE field_projection()]
├── interpolate_magnitudes()                          # [EJECUTA DENTRO DE field_projection()]
└── calculate_detectability()                         # [EJECUTA DENTRO DE field_projection()]

save_projection_results()                             # Persistencia de resultados
├── generate_csv_summary()                            # [EJECUTA DENTRO DE save_projection_results()]
├── save_lightcurve_data()                            # [EJECUTA DENTRO DE save_projection_results()]
└── update_run_metadata()                             # [EJECUTA DENTRO DE save_projection_results()]
```

---

## �🚀 **PUNTO DE ENTRADA: simple_runner.py**

### **Función Principal: `main()`**
```python
def main():
    args = parse_arguments()  # Parsear CLI arguments
    setup_environment()       # Crear directorios de output
    run_custom_batch(...)     # Ejecutar batch con parámetros CLI
```

### **CLI Arguments → SimpleConfig**
```python
def run_custom_batch(n_runs, redshift_max, sn_types, survey, filter_band, seed):
    # 1. CREAR CONFIGURACIÓN desde CLI
    config = create_simple_config(
        n_runs=n_runs,           # --runs 50
        redshift_max=redshift_max, # --redshift-max 0.3
        sn_types=sn_types,       # --sn-types Ia Ibc
        survey=survey,           # --survey ZTF
        filter_band=filter_band, # --filter r
        base_seed=seed           # --seed 123
    )
    
    # 2. EJECUTAR BATCH CIENTÍFICO
    results = run_scientific_batch(config)  # → batch_runner.py
```

**Archivos Involucrados:**
- `simple_runner.py`: Entry point y CLI parsing
- `simple_config.py`: Clase `SimpleConfig` y función `create_simple_config()`

---

## 🔄 **BATCH RUNNER: batch_runner.py**

### **Función de Alto Nivel: `run_scientific_batch(config)`**
```python
def run_scientific_batch(batch_config) -> Dict:
    runner = ProfessionalBatchRunner()
    return runner.run_batch(batch_config)  # ← Método principal
```

### **Clase Principal: `ProfessionalBatchRunner`**

#### **Método Central: `run_batch(batch_config)`**
```python
def run_batch(self, batch_config) -> Dict:
    self.stats.start_batch()  # Iniciar cronómetro
    
    for i in range(batch_config.n_runs):  # Loop principal de iteraciones
        # 1. GENERAR PARÁMETROS de la iteración
        iteration_params = self.create_run_parameters(batch_config, i, batch_config.n_runs)
        
        # 2. EJECUTAR iteración individual
        success, iteration_results = self.execute_single_run(iteration_params)
        
        # 3. REGISTRAR resultados
        iteration_record = {**iteration_params, **iteration_results, 'success': success}
        self.run_registry.append(iteration_record)
        
        # 4. ACTUALIZAR estadísticas agregadas
        if success:
            self.stats.add_successful_run(...)
        else:
            self.stats.add_failed_run(...)
    
    # 5. GUARDAR resultados del batch
    self.save_batch_results(batch_config)
    return summary
```

---

## 📊 **GENERACIÓN DE PARÁMETROS: `create_run_parameters()`**

### **Muestreo Científico por Iteración**
```python
def create_run_parameters(self, batch_config, run_index: int, total_runs: int) -> Dict:
    np.random.seed(batch_config.base_seed + run_index)  # Reproducibilidad
    
    # 1. SELECCIONAR tipo de SN según distribución
    sn_type = np.random.choice(
        list(batch_config.sn_type_distribution.keys()),
        p=list(batch_config.sn_type_distribution.values())
    )  # → "Ia", "Ibc", "II"
    
    # 2. MUESTREO COSMOLÓGICO (volume-weighted)
    redshift_sample = sample_cosmological_redshift(
        n_samples=1,
        z_min=z_min, z_max=z_max,
        H0=cosmology.get('H0', 70),
        Om=cosmology.get('Om', 0.3),
        OL=cosmology.get('OL', 0.7)
    )[0]  # → 0.0249
    
    # 3. EXTINCIÓN DEL HOST (distribuciones por tipo)
    ebmv_host_sample = sample_extinction_by_type(
        sn_type=sn_type, 
        n_samples=1, 
        random_state=batch_config.base_seed + run_index
    )  # → 0.043 (distribución exponencial/mixta según tipo)
    
    # 4. SELECCIONAR TEMPLATE aleatoriamente
    available_templates = get_sn_templates()[sn_type]
    template = np.random.choice(available_templates)  # → "SN2006ep.dat"
    
    # 5. EXTINCIÓN MW (mapas realistas ZTF)
    ebmv_mw_sample = sample_realistic_mw_extinction(
        sn_name=template.replace('.dat', ''), 
        n_samples=1, 
        random_state=batch_config.base_seed + run_index
    )  # → 0.067 (consulta mapas SFD98)
    
    # 6. SELECCIONAR SURVEY
    survey = np.random.choice(
        list(batch_config.survey_distribution.keys()),
        p=list(batch_config.survey_distribution.values())
    )  # → "ZTF"
    
    return iteration_params  # Diccionario con todos los parámetros
```

**Funciones Científicas Llamadas:**
- `sample_cosmological_redshift()` → `core/correction.py`
- `sample_extinction_by_type()` → `core/correction.py`
- `get_sn_templates()` → `simple_config.py`
- `sample_realistic_mw_extinction()` → `dust_maps.py`

---

## ⚙️ **EJECUCIÓN INDIVIDUAL: `execute_single_run()`**

### **Sistema Robusto con Fallback**
```python
def execute_single_run(self, iteration_params: Dict) -> Tuple[bool, Dict]:
    iteration_start_time = time.time()
    
    try:
        # 1. ACTUALIZAR configuración global
        self.update_config_for_run(iteration_params)
        
        # 2. VALIDAR configuración
        validated_config = load_and_validate_config()
        
        # 3. MÉTODO PRIMARIO: Ejecución directa
        try:
            import main
            main.main(config=validated_config)  # ← LLAMADA A MAIN.PY
            
            # Extraer estadísticas reales del CSV generado
            real_stats = self.extract_run_statistics(iteration_params)
            return True, iteration_stats
            
        except Exception as direct_error:
            # 4. MÉTODO FALLBACK: Subprocess
            result = subprocess.run([sys.executable, 'main.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                real_stats = self.extract_run_statistics(iteration_params)
                return True, iteration_stats
            else:
                return False, error_details
    
    except Exception as e:
        return False, {'error': str(e)}
```

### **Actualización de Configuración: `update_config_for_run()`**
```python
def update_config_for_run(self, iteration_params: Dict) -> None:
    # MODIFICAR variables globales en config.py
    config.SN_CONFIG['sn_name'] = iteration_params['sn_name']        # "SN2006ep"
    config.SN_CONFIG['tipo'] = iteration_params['sn_type']           # "Ibc"
    config.SN_CONFIG['z_proy'] = iteration_params['redshift']        # 0.0249
    config.SN_CONFIG['ebmv_host'] = iteration_params['ebmv_host']    # 0.043
    config.SN_CONFIG['ebmv_mw'] = iteration_params['ebmv_mw']        # 0.067
    config.PATHS['spec_file'] = iteration_params['template_file']    # "Ibc/SN2006ep.dat"
    config.SN_CONFIG['selected_filter'] = iteration_params['filter_band']  # "r"
    config.SURVEY = iteration_params['survey']                       # "ZTF"
```

---

## 🔬 **PIPELINE CIENTÍFICO: main.py**

### **Función Principal: `main(config=None)`**
```python
def main(config=None):
    # CONFIGURACIÓN (si no viene del batch)
    if config is None:
        config = load_and_validate_config()
    
    # Extraer información específica
    survey_info = get_survey_info(config)
    sn_info = get_sn_info(config)
    processing_config = config['processing']
    extinction_config = config['extinction']
```

### **PASO 1: Lectura de Espectro**
```python
print(f"\nPASO 1: Lectura de espectro")
ESPECTRO, fases = leer_spec(path_spec, ot=False, as_pandas=True)
# → Lista de DataFrames con espectros por fase
# → Lista de fases temporales (MJD o días relativos)
```

### **PASO 2: Correcciones Cosmológicas y Extinción**
```python
print(f"\nPASO 2: Correcciones cosmológicas y extinción")
ESPECTRO_corr, fases_corr = correct_redeening(
    sn=sn_name, ESPECTRO=ESPECTRO, fases=fases,
    z=z_proy,                    # Redshift de la iteración
    ebmv_host=ebmv_host_final,   # Extinción del host
    ebmv_mw=ebmv_mw,            # Extinción MW
    reverse=True, use_DL=True
)
# → Aplica redshift cosmológico + extinción en orden físico correcto
```

### **PASO 3: Curva de Respuesta del Filtro**
```python
print(f"\nPASO 3: Curva de respuesta del filtro {selected_filter}")
response_df = pd.read_csv(path_response, sep='\s+', comment='#', header=None)
response_df.columns = ['wave', 'response']
# → Carga curva de respuesta del filtro fotométrico (r, g, i, etc.)
```

### **PASO 4: Fotometría Sintética**
```python
print(f"\nPASO 4: Fotometría sintética")
for spec, fase in zip(ESPECTRO_corr, fases_corr):
    flux, porcentaje = Syntetic_photometry_v2(
        spec['wave'].values, spec['flux'].values,
        response_df['wave'].values, response_df['response'].values
    )
    if porcentaje > processing_config['overlap_threshold']:  # 0.95
        fases_lc.append(fase)
        fluxes_lc.append(flux)
# → Convoluciona espectros con filtro según overlap mínimo
```

### **PASO 5: Suavizado LOESS**
```python
print(f"\nPASO 5: Suavizado LOESS")
df_loess = Loess_fit(LC_df, selected_filter, 
                    alpha=alpha_usado, plot=False)
# → Suavizado estadístico para interpolar curva de luz
```

### **PASO 6: Calibración Fotométrica**
```python
print(f"\nPASO 6: Calibración fotométrica")
constante = 'cte' + selected_filter  # cter, cteg, cteV, etc.
mul = arr_val_ctes[jj]              # Constante de punto cero
flux_calibrado = np.array(lc_df['flux']) / mul
mag = -2.5 * np.log10(np.clip(flux_calibrado, 1e-20, None))
# → Convierte flujos a magnitudes calibradas
```

### **PASO 7: Ruido Fotométrico**
```python
print(f"\nPASO 7: Aplicación de ruido fotométrico")
noise_level = processing_config['noise_level']  # 0.15 (15%)
flux_noisy_norm = np.random.normal(
    loc=flux_norm, 
    scale=np.sqrt(np.abs(flux_norm)) * noise_level
)
mag_noisy = -2.5 * np.log10(flux_noisy)
# → Simula ruido de Poisson con distribución gaussiana
```

### **PASO 7.5: Conversión de Unidades Temporales (CRÍTICO)**
```python
maximum = maximo_lc(tipo, sn_name)  # MJD del máximo
if tipo in ['Ibc', 'Ib', 'Ic']:     # SNe core-collapse
    mjd_absolute = maximum + lc_df['fase']  # Fases relativas → MJD absoluto
    lc_df['fase'] = mjd_absolute            # Actualizar DataFrame
```

**⚠️ BUG CRÍTICO RESUELTO:**
```python
# ANTES (INCORRECTO):
# fases = [-10, 0, +20]        # ← Fases relativas
# maximum = 53671              # ← MJD absoluto  
# mjd_pivote = 59000          # ← MJD modernas ZTF
# desplazamiento = 59000 - 53671 + offset ≈ 5329 días
# fases_ajustadas = [5319, 5329, 5349]  # ← ¡Fechas imposibles año 1846!

# DESPUÉS (CORRECTO):
# fases_convertidas = [53659, 53671, 53691]  # ← MJD absoluto
# fases_ajustadas = [58988, 59000, 59089]    # ← Fechas modernas válidas
```

### **PASO 8: Proyección sobre Observaciones Reales**
```python
print(f"\nPASO 8: Proyección sobre observaciones reales ({SURVEY})")
df_obslog_survey = pd.read_csv(path_obslog)

# Selección de target específica
if SURVEY == "ZTF":
    available_targets = df_obslog_survey[target_column].unique()
    selected_target = np.random.choice(available_targets)  # OID aleatorio

df_projected = field_projection(
    fases=lc_df['fase'].values,          # MJD absoluto (ya convertido)
    flux_y=mag_noisy,                    # Magnitudes con ruido
    df_obslog=df_obslog_survey,          # Observaciones reales del survey
    tipo=tipo,                           # Tipo SN para calcular máximo
    selected_filter=projection_filter,   # Filtro específico del survey
    selected_field=selected_target,      # Target seleccionado
    offset=np.arange(offset_range[0], offset_range[1], offset_step),
    sn=sn_name,
    plot=show_debug_plots
)
```

### **Función `field_projection()` (core/projection.py):**
```python
def field_projection(fases, flux_y, df_obslog, tipo, selected_filter, 
                    offset, sn, selected_field=None, plot=False):
    # 1. FILTRAR observaciones por campo/OID y filtro
    df_filtered = obs_log[
        (obs_log['field/oid'] == selected_field) &
        (obs_log['filter'] == selected_filter)
    ]
    
    # 2. CALCULAR máximo de la SN
    maximum = maximo_lc(tipo, sn)
    
    # 3. SELECCIONAR offset aleatorio y calcular desplazamiento
    select_offset = np.random.choice(offset)
    mjd_pivote = df_filtered.iloc[0]['mjd']
    desplazamiento = mjd_pivote - maximum + select_offset
    
    # 4. AJUSTAR fases de la SN al tiempo de observación
    fases_ajustadas = [fecha + desplazamiento for fecha in fases]
    
    # 5. FILTRAR observaciones que coinciden temporalmente
    df_filtered_cut = df_filtered[
        (df_filtered['mjd'] >= min(fases_ajustadas)) &
        (df_filtered['mjd'] <= max(fases_ajustadas))
    ]
    
    # 6. INTERPOLAR magnitudes de la SN en tiempos de observación
    interpolation_function = interpolate.interp1d(
        fases_ajustadas, flux_y, kind='linear', fill_value='extrapolate'
    )
    df_filtered_cut['magnitud_proyectada'] = interpolation_function(df_filtered_cut['mjd'])
    
    # 7. CLASIFICAR detecciones vs upper limits
    df_filtered_cut['upperlimit'] = (
        df_filtered_cut['maglimit'] == df_filtered_cut['magnitud_proyectada']
    ).map({True: 'T', False: 'F'})
    
    return df_filtered_cut  # DataFrame con proyecciones finales
```

### **PASO 9: Resultados Finales**
```python
print(f"\nPASO 9: Resultados finales ({SURVEY})")
if len(df_projected) > 0:
    detecciones = len(df_projected[df_projected['upperlimit'] == 'F'])
    upper_limits = len(df_projected[df_projected['upperlimit'] == 'T'])
    tasa_deteccion = detecciones/len(df_projected)*100
    
    print(f"   • Detecciones: {detecciones:,}")
    print(f"   • Upper limits: {upper_limits:,}")
    print(f"   • Tasa de detección: {tasa_deteccion:.1f}%")
```

### **PASO 10: Guardar Resultados**
```python
print(f"\nPASO 10: Guardar Resultados")
saved_files = save_projection_results(
    df_projected=df_projected,      # Proyecciones finales
    lc_df=lc_df,                   # Curva de luz sintética
    mag=mag,                       # Magnitudes limpias
    mag_noisy=mag_noisy,           # Magnitudes con ruido
    survey_params=survey_params,    # Metadatos del survey
    sn_params=sn_params,           # Parámetros físicos SN
    projection_params=projection_params,  # Config técnica
    ruido_promedio=ruido_promedio,
    alpha_usado=alpha_usado,
    maximum=maximum
)
```

### **PASO 11: Actualizar Índice Maestro**
```python
print(f"\nPASO 11: Actualizar Índice Maestro")
create_master_index()  # Actualiza índice global de simulaciones
```

---

## 📊 **RETORNO Y AGREGACIÓN**

### **Extracción de Estadísticas: `extract_run_statistics()`**
```python
def extract_run_statistics(self, iteration_params: Dict) -> Dict:
    # 1. BUSCAR archivo summary más reciente
    search_pattern = f"outputs/**/summary_*{sn_name}*.csv"
    summary_files = glob.glob(search_pattern, recursive=True)
    
    # 2. LEER CSV generado por main.py
    df_summary = pd.read_csv(summary_file)
    row = df_summary.iloc[0]
    
    # 3. EXTRAER métricas reales
    return {
        'n_detections': int(row.get('detections', 0)),      # 29
        'n_observations': int(row.get('total_points', 0)),  # 29  
        'detection_rate_percent': float(row.get('detection_rate_percent', 0.0)),  # 100.0
        'ebmv_host': float(row.get('ebmv_host', 0.0)),      # 0.043
        'ebmv_mw': float(row.get('ebmv_mw', 0.0)),          # 0.067
        'status': row.get('status', 'UNKNOWN')              # "SUCCESS"
    }
```

### **Agregación de Estadísticas: `add_successful_run()`**
```python
def add_successful_run(self, execution_time: float, n_detections: int, 
                      n_observations: int, sn_type: str):
    self.runs_completed += 1
    self.execution_times.append(execution_time)        # Para promedios
    self.total_detections += n_detections              # Suma global
    self.total_observations += n_observations          # Suma global
    
    # Estadísticas por tipo de SN
    if sn_type not in self.detection_rates_by_type:
        self.detection_rates_by_type[sn_type] = []
    
    detection_rate = n_detections / max(1, n_observations)
    self.detection_rates_by_type[sn_type].append(detection_rate)
```

---

## 📁 **ESTRUCTURA DE OUTPUTS GENERADA**

### **Archivos Individuales (main.py):**
```
outputs/single_runs/run_20250816_143052_SN2006ep/
├── projection_results.csv      ← Detecciones/upper limits por observación
├── metadata.json               ← Parámetros completos de la simulación
├── lightcurve_plot.png         ← Gráfico científico curva de luz
├── projection_plot.png         ← Gráfico temporal de proyección
└── summary_SN2006ep_Ibc.csv    ← Resumen de 1 fila con métricas clave
```

### **Archivos Agregados (batch_runner.py):**
```
outputs/batch_runs/batch_20250816_142830_a1b2c3d4/
├── batch_metadata.json         ← Configuración + estadísticas agregadas
├── run_summary.csv             ← Todas las iteraciones en 1 DataFrame
├── statistical_summary.txt     ← Resumen científico legible
└── logs/batch_[ID].log         ← Log completo del batch
```

---

## 🔄 **DIAGRAMA DE FLUJO COMPLETO**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI INPUTS    │───▶│ SIMPLE_RUNNER   │───▶│ SIMPLE_CONFIG   │
│ --runs 50       │    │ parse_arguments │    │ create_simple_  │
│ --filter r      │    │ setup_env       │    │ config()        │
│ --sn-types Ibc  │    │ run_custom_batch│    │ SimpleConfig    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ BATCH_RUNNER    │◀───│ run_scientific_ │◀───│ CONFIG OBJECT   │
│ Professional    │    │ batch()         │    │ n_runs=50       │
│ BatchRunner     │    │                 │    │ sn_types=["Ibc"]│
│                 │    │                 │    │ filter_band="r" │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PARAMETER       │───▶│ SINGLE RUN      │───▶│ MAIN.PY         │
│ GENERATION      │    │ EXECUTION       │    │ PIPELINE        │
│ create_run_     │    │ execute_single_ │    │ main()          │
│ parameters()    │    │ run()           │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ SCIENTIFIC      │    │ CONFIG UPDATE   │    │ 11 PASOS        │
│ SAMPLING        │    │ update_config_  │    │ CIENTÍFICOS     │
│ • Cosmology     │    │ for_run()       │    │ • Espectros     │
│ • Extinction    │    │ • sn_name       │    │ • Correcciones  │
│ • Templates     │    │ • redshift      │    │ • Fotometría    │
│ • Coordinates   │    │ • extinctions   │    │ • Proyección    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ RESULTS         │◀───│ STATISTICS      │◀───│ OUTPUTS         │
│ AGGREGATION     │    │ EXTRACTION      │    │ • CSV files     │
│ • Total detect  │    │ extract_run_    │    │ • JSON metadata │
│ • By SN type    │    │ statistics()    │    │ • Plots         │
│ • Success rate  │    │                 │    │ • Logs          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🔧 **FUNCIONES CLAVE POR ARCHIVO**

### **simple_runner.py**
- `main()`: Entry point principal
- `parse_arguments()`: Parseo CLI
- `setup_environment()`: Creación directorios
- `run_custom_batch()`: Configuración y ejecución

### **simple_config.py**
- `create_simple_config()`: Factory de configuraciones
- `get_sn_templates()`: Obtener templates por tipo
- `SimpleConfig`: Clase de configuración

### **batch_runner.py**
- `ProfessionalBatchRunner.run_batch()`: Loop principal
- `create_run_parameters()`: Muestreo científico
- `execute_single_run()`: Ejecución individual
- `update_config_for_run()`: Actualización config
- `extract_run_statistics()`: Extracción métricas
- `save_batch_results()`: Persistencia agregada

### **main.py**
- `main()`: Pipeline científico 11 pasos
- Funciones de cada paso detalladas arriba

### **core/projection.py**
- `field_projection()`: Proyección temporal core

### **core/correction.py**
- `sample_cosmological_redshift()`: Muestreo cosmológico
- `sample_extinction_by_type()`: Distribuciones extinción
- `correct_redeening()`: Aplicación correcciones

### **core/utils.py**
- `leer_spec()`: Lectura espectros
- `Syntetic_photometry_v2()`: Fotometría sintética
- `Loess_fit()`: Suavizado estadístico
- `maximo_lc()`: Fecha del máximo

---

## ⚠️ **PUNTOS CRÍTICOS DOCUMENTADOS**

### **1. Bug de Unidades Temporales (RESUELTO)**
- **Problema**: Fases relativas vs MJD absoluto en proyección
- **Solución**: PASO 7.5 conversión explícita
- **Ubicación**: `main.py` líneas 200-210

### **2. Sistema de Fallback**
- **Método primario**: Import directo `main.main()`
- **Método fallback**: Subprocess `python main.py`
- **Razón**: Aislamiento de estado entre iteraciones

### **3. Configuración Global**
- **Actualización**: `update_config_for_run()` modifica `config.py`
- **Validación**: `load_and_validate_config()` lee valores actualizados
- **Consistencia**: Ambos métodos (directo/subprocess) ven misma config

### **4. Extracción de Estadísticas**
- **Fuente**: CSV generado por `save_projection_results()`
- **Timing**: Después de completar `main.main()`
- **Validación**: Confirma ejecución exitosa vs fallida

---

## 📈 **MÉTRICAS CIENTÍFICAS GENERADAS**

### **Por Iteración Individual:**
- `n_detections`: Número de detecciones
- `n_observations`: Observaciones totales  
- `detection_rate_percent`: Tasa de detección
- `execution_time`: Tiempo de ejecución
- `ebmv_host`, `ebmv_mw`: Extinciones aplicadas

### **Agregadas por Batch:**
- `total_detections`: Suma de todas las detecciones
- `global_detection_rate`: Tasa global
- `detection_rates_by_type`: Distribuciones por tipo SN
- `success_rate`: Tasa de éxito de simulaciones
- `average_execution_time`: Tiempo promedio

### **Distribuciones Estadísticas:**
- Media, desviación estándar, mediana por tipo SN
- Rangos de redshift, extinción, magnitudes
- Cobertura temporal y espacial

---

##  **DISEÑO PARA INVESTIGACIÓN **

### **Reproducibilidad Científica:**
- Seeds controladas para cada iteración
- Configuraciones completas guardadas
- Trazabilidad end-to-end documentada

### **Escalabilidad:**
- Paralelización futura posible
- Manejo robusto de errores
- Logging detallado para debugging

### **Validación Académica:**
- Referencias científicas en distribuciones
- Métodos observacionalmente validados
- Estadísticas publication-ready

---

**SISTEMA COMPLETO MAPEADO** ✅  
*Documentación técnica completa para referencia permanente*  
*Actualizado: Agosto 2025*
