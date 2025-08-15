# BATCH RUNNER PROFESIONAL - TESIS DOCTORAL
# ============================================
# Sistema robusto para ejecución de múltiples simulaciones de supernovas
# Diseñado para investigación científica de alto nivel

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import asdict
import traceback

# Importar configuración simplificada
try:
    from simple_config import (
        SimpleConfig, SNType, Survey, 
        create_cosmological_sample, get_sn_templates
    )
except ImportError:
    SimpleConfig = None

# Importar funciones científicas de extinción (sistema unificado)
from core.correction import (
    sample_host_extinction_SNIa,
    sample_host_extinction_core_collapse
)

# Importar configuración base y ejecutor principal
import config
from config_loader import load_and_validate_config
# NO importar main aquí para evitar problemas de ejecución inmediata

class BatchLogger:
    """Sistema de logging profesional para batch runs"""
    
    def __init__(self, log_dir: str, batch_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger principal
        self.logger = logging.getLogger(f'batch_{batch_id}')
        self.logger.setLevel(logging.INFO)
        
        # Archivo de log
        log_file = self.log_dir / f"batch_{batch_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)

class BatchStatistics:
    """Estadísticas y métricas del batch run"""
    
    def __init__(self):
        self.runs_completed = 0
        self.runs_failed = 0
        self.total_detections = 0
        self.total_observations = 0
        self.execution_times = []
        self.failed_runs = []
        self.detection_rates_by_type = {}
        self.start_time = None
        self.end_time = None
    
    def start_batch(self):
        self.start_time = datetime.now()
    
    def end_batch(self):
        self.end_time = datetime.now()
    
    def add_successful_run(self, execution_time: float, n_detections: int, 
                          n_observations: int, sn_type: str):
        self.runs_completed += 1
        self.execution_times.append(execution_time)
        self.total_detections += n_detections
        self.total_observations += n_observations
        
        if sn_type not in self.detection_rates_by_type:
            self.detection_rates_by_type[sn_type] = []
        
        detection_rate = n_detections / max(1, n_observations)
        self.detection_rates_by_type[sn_type].append(detection_rate)
    
    def add_failed_run(self, run_id: str, error: str):
        self.runs_failed += 1
        self.failed_runs.append({'run_id': run_id, 'error': error})
    
    def get_summary(self) -> Dict:
        """Resumen estadístico completo"""
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            'batch_metadata': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'total_duration_seconds': total_time,
                'total_duration_formatted': f"{total_time/60:.1f} minutes"
            },
            'run_statistics': {
                'runs_completed': self.runs_completed,
                'runs_failed': self.runs_failed,
                'success_rate': self.runs_completed / max(1, self.runs_completed + self.runs_failed),
                'average_execution_time': np.mean(self.execution_times) if self.execution_times else 0,
                'std_execution_time': np.std(self.execution_times) if self.execution_times else 0
            },
            'detection_statistics': {
                'total_detections': self.total_detections,
                'total_observations': self.total_observations,
                'global_detection_rate': self.total_detections / max(1, self.total_observations),
                'detection_rates_by_type': {
                    sn_type: {
                        'mean': np.mean(rates),
                        'std': np.std(rates),
                        'median': np.median(rates),
                        'n_runs': len(rates)
                    } for sn_type, rates in self.detection_rates_by_type.items()
                }
            },
            'failed_runs': self.failed_runs
        }

class ProfessionalBatchRunner:
    """
    Sistema profesional de batch runs para investigación de tesis doctoral.
    
    Características:
    - Muestreo estadísticamente riguroso de parámetros
    - Logging completo y trazabilidad
    - Manejo robusto de errores
    - Estadísticas científicas detalladas
    - Reproducibilidad garantizada
    - Escalabilidad para grandes simulaciones
    """
    
    def __init__(self, base_config_path: str = "config.py"):
        self.base_config_path = base_config_path
        self.batch_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Directorios
        self.batch_dir = Path(config.PATHS['output_dir']) / "batch_runs" / self.batch_id
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = BatchLogger(str(self.batch_dir / "logs"), self.batch_id)
        
        # Estadísticas
        self.stats = BatchStatistics()
        
        # Lista de runs ejecutados
        self.run_registry = []
        
        self.logger.info(f"Iniciando Professional Batch Runner")
        self.logger.info(f"Batch ID: {self.batch_id}")
        self.logger.info(f"Directorio: {self.batch_dir}")
    
    def create_run_parameters(self, batch_config, 
                            run_index: int, total_runs: int) -> Dict:
        """
        Genera parámetros para un run individual usando muestreo científico
        """
        np.random.seed(batch_config.base_seed + run_index)  # Reproducibilidad
        
        # Seleccionar tipo de SN según distribución
        sn_type = np.random.choice(
            list(batch_config.sn_type_distribution.keys()),
            p=list(batch_config.sn_type_distribution.values())
        )
        
        # Muestreo cosmológico (compatible con ambas configuraciones)
        if hasattr(batch_config, 'cosmology'):
            # Configuración completa
            redshift_sample = create_cosmological_sample(
                batch_config.redshift_range, 1, 
                batch_config.cosmology, batch_config.volume_weighted
            )[0]
        else:
            # Configuración simplificada
            from simple_config import create_cosmological_sample as simple_cosmo
            redshift_sample = simple_cosmo(
                batch_config.redshift_range, 1, 
                batch_config.volume_weighted
            )[0]
        
        # Debug: imprimir información del muestreo cada pocos runs
        if run_index < 3:  # Solo para los primeros runs
            self.logger.info(f"DEBUG - Run {run_index+1}: volume_weighted={batch_config.volume_weighted}, z_sampled={redshift_sample:.6f}")
        
        # SISTEMA UNIFICADO: Muestreo de extinción académicamente correcto
        # Extinción del host usando distribuciones exponenciales unificadas (Holwerda et al. 2014)
        if sn_type == 'Ia':
            ebmv_host_sample = sample_host_extinction_SNIa(n_samples=1, tau=0.4, Av_max=3.0, Rv=3.1)
        else:  # Core-collapse (II, Ibc) - AHORA UNIFICADO a exponencial
            ebmv_host_sample = sample_host_extinction_core_collapse(
                n_samples=1, sn_type=sn_type, tau=0.4, Av_max=3.0, Rv=3.1
            )
        
        # Seleccionar template aleatoriamente del tipo (configuración unificada)
        try:
            # Usar siempre la función de simple_config
            templates_dict = get_sn_templates()
            if isinstance(list(templates_dict.keys())[0], SNType):
                # Si las claves son SNType
                available_templates = templates_dict[SNType(sn_type.upper())]
            else:
                # Si las claves son strings
                available_templates = templates_dict[sn_type]
        except:
            # Fallback directo a SN_TEMPLATES  
            from simple_config import SN_TEMPLATES
            available_templates = SN_TEMPLATES[sn_type]
        template = np.random.choice(available_templates)
        
        # EXTINCIÓN MW: Basada en mapas de polvo realistas para campos ZTF
        # Importar sistema de mapas de polvo
        from dust_maps import sample_realistic_mw_extinction
        
        # Muestrear extinción MW realista basada en footprint ZTF
        ebmv_mw_sample = sample_realistic_mw_extinction(
            sn_name=template.replace('.dat', ''), 
            n_samples=1, 
            random_state=batch_config.base_seed + run_index
        )
        
        # Guardar coordenadas y extinción para futuro análisis (opcional)
        if not hasattr(self, 'extinction_data'):
            self.extinction_data = {'run_id': [], 'ebmv_mw': []}
        
        # Almacenar para estadísticas básicas
        self.extinction_data['run_id'].append(f"iter_{run_index+1:04d}")
        self.extinction_data['ebmv_mw'].append(ebmv_mw_sample[0])
        
        # Seleccionar survey
        survey = np.random.choice(
            list(batch_config.survey_distribution.keys()),
            p=list(batch_config.survey_distribution.values())
        )
        
        # Parámetros de la iteración (cada supernova)
        iteration_params = {
            'iteration_id': f"iter_{run_index+1:04d}_of_{total_runs:04d}",
            'sn_name': template.replace('.dat', ''),
            'sn_type': sn_type,
            'template_file': f"{sn_type}/{template}",
            'redshift': float(redshift_sample),
            'ebmv_host': float(ebmv_host_sample[0]),    # Extinción del host
            'ebmv_mw': float(ebmv_mw_sample[0]),        # Extinción MW
            'extinction_total': float(ebmv_host_sample[0] + ebmv_mw_sample[0]),  # Para compatibilidad
            'survey': survey,
            'seed': batch_config.base_seed + run_index,
            'batch_id': self.batch_id,
            'iteration_index': run_index,
            'total_iterations': total_runs
        }
        
        return iteration_params
    
    def update_config_for_run(self, iteration_params: Dict) -> None:
        """
        Actualiza la configuración global para la iteración específica
        """
        # Actualizar configuración de SN
        config.SN_CONFIG['sn_name'] = iteration_params['sn_name']
        config.SN_CONFIG['tipo'] = iteration_params['sn_type']
        config.SN_CONFIG['z_proy'] = iteration_params['redshift']
        
        # Actualizar extinción con valores separados (físicamente correcto)
        config.SN_CONFIG['ebmv_host'] = iteration_params['ebmv_host']
        config.SN_CONFIG['ebmv_mw'] = iteration_params['ebmv_mw']
        # Mantener compatibilidad con código anterior
        config.SN_CONFIG['ebmv_proy'] = iteration_params['extinction_total']
        
        # Actualizar archivo de espectro
        config.PATHS['spec_file'] = iteration_params['template_file']
        
        # Actualizar survey
        config.SURVEY = iteration_params['survey']
        
        # Añadir metadatos del batch
        if not hasattr(config, 'BATCH_METADATA'):
            config.BATCH_METADATA = {}
        
        config.BATCH_METADATA.update({
            'batch_id': iteration_params['batch_id'],
            'iteration_id': iteration_params['iteration_id'],
            'iteration_index': iteration_params['iteration_index'],
            'total_iterations': iteration_params['total_iterations'],
            'seed': iteration_params['seed']
        })
    
    def extract_run_statistics(self, iteration_params: Dict) -> Dict:
        """
        Extrae estadísticas reales del CSV generado por la simulación
        """
        try:
            # Construir el nombre del archivo summary basado en los parámetros
            sn_name = iteration_params['sn_name']
            survey = iteration_params['survey']
            redshift = iteration_params['redshift']
            
            # Buscar el archivo summary más reciente en el directorio outputs
            import glob
            from pathlib import Path
            
            # Patrón de búsqueda basado en la estructura de archivos
            search_pattern = f"outputs/**/summary_*{sn_name}*.csv"
            summary_files = glob.glob(search_pattern, recursive=True)
            
            if not summary_files:
                # Intentar patrón más general
                search_pattern = f"outputs/**/summary_*.csv"
                summary_files = glob.glob(search_pattern, recursive=True)
                
                # Tomar el más reciente
                if summary_files:
                    summary_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                    summary_file = summary_files[0]
                else:
                    self.logger.warning("No se encontró archivo summary CSV")
                    return {
                        'n_detections': 0,
                        'n_observations': 0,
                        'ebmv_host': iteration_params.get('ebmv_host', 0.0),
                        'ebmv_mw': iteration_params.get('ebmv_mw', 0.0),
                        'ebmv_total': iteration_params.get('extinction_total', 0.0),
                        'detection_rate_percent': 0.0
                    }
            else:
                # Tomar el más reciente
                summary_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                summary_file = summary_files[0]
            
            # Leer el CSV
            import pandas as pd
            df_summary = pd.read_csv(summary_file)
            
            if len(df_summary) > 0:
                row = df_summary.iloc[0]  # Tomar la primera (y única) fila
                
                return {
                    'n_detections': int(row.get('detections', 0)),
                    'n_observations': int(row.get('total_points', 0)),
                    'ebmv_host': float(row.get('ebmv_host', iteration_params.get('ebmv_host', 0.0))),
                    'ebmv_mw': float(row.get('ebmv_mw', iteration_params.get('ebmv_mw', 0.0))),
                    'ebmv_total': float(row.get('ebmv_total', iteration_params.get('extinction_total', 0.0))),
                    'detection_rate_percent': float(row.get('detection_rate_percent', 0.0)),
                    'sn_type': row.get('sn_type', iteration_params.get('sn_type', 'Unknown')),
                    'redshift': float(row.get('redshift', iteration_params.get('redshift', 0.0))),
                    'survey': row.get('survey', iteration_params.get('survey', 'Unknown')),
                    'status': row.get('status', 'UNKNOWN')
                }
            else:
                self.logger.warning("Archivo summary CSV está vacío")
                return {
                    'n_detections': 0,
                    'n_observations': 0,
                    'ebmv_host': iteration_params.get('ebmv_host', 0.0),
                    'ebmv_mw': iteration_params.get('ebmv_mw', 0.0),
                    'ebmv_total': iteration_params.get('extinction_total', 0.0),
                    'detection_rate_percent': 0.0
                }
                
        except Exception as e:
            self.logger.warning(f"Error extrayendo estadísticas del CSV: {str(e)}")
            return {
                'n_detections': 0,
                'n_observations': 0,
                'ebmv_host': iteration_params.get('ebmv_host', 0.0),
                'ebmv_mw': iteration_params.get('ebmv_mw', 0.0),
                'ebmv_total': iteration_params.get('extinction_total', 0.0),
                'detection_rate_percent': 0.0
            }
    
    def execute_single_run(self, iteration_params: Dict) -> Tuple[bool, Dict]:
        """
        Ejecuta una iteración individual (una supernova) con manejo robusto de errores
        """
        iteration_start_time = time.time()
        
        # Separador visual para cada iteración
        self.logger.info("="*60)
        self.logger.info(f"ITERACIÓN {iteration_params['iteration_id']} ({iteration_params['iteration_index']+1}/{iteration_params['total_iterations']})")
        self.logger.info(f"SN: {iteration_params['sn_type']} {iteration_params['sn_name']}")
        self.logger.info(f"z={iteration_params['redshift']:.4f}, E(B-V)_host={iteration_params['ebmv_host']:.3f}, E(B-V)_MW={iteration_params['ebmv_mw']:.3f}")
        self.logger.info(f"Survey: {iteration_params['survey']}, Seed: {iteration_params['seed']}")
        self.logger.info("-"*60)
        
        try:
            # Actualizar configuración
            self.update_config_for_run(iteration_params)
            
            # Validar configuración
            validated_config = load_and_validate_config()
            
            # Intentar ejecución directa primero
            try:
                # Importar y ejecutar main directamente
                import main
                
                # Ejecutar la función main directamente
                main.main()
                
                iteration_time = time.time() - iteration_start_time
                
                # Extraer estadísticas reales del CSV generado
                real_stats = self.extract_run_statistics(iteration_params)
                
                iteration_stats = {
                    'execution_time': iteration_time,
                    'method': 'direct_import',
                    **real_stats  # Incluir estadísticas reales
                }
                
                self.logger.info(f"COMPLETADA ITERACIÓN {iteration_params['iteration_id']} en {iteration_time:.2f}s")
                self.logger.info("="*60)
                return True, iteration_stats
                
            except Exception as direct_error:
                self.logger.warning(f"Método directo falló: {str(direct_error)}, intentando subprocess...")
                
                # Método subprocess como respaldo
                import subprocess
                import sys
                
                # Ejecutar main.py directamente con manejo robusto de encoding
                result = subprocess.run([sys.executable, 'main.py'], 
                                      capture_output=True, 
                                      text=True,
                                      encoding='utf-8',
                                      errors='replace')
                
                iteration_time = time.time() - iteration_start_time
                
                # Verificar si la ejecución fue exitosa
                if result.returncode == 0:
                    # Extraer estadísticas reales del CSV generado
                    real_stats = self.extract_run_statistics(iteration_params)
                    
                    iteration_stats = {
                        'execution_time': iteration_time,
                        'stdout': result.stdout if result.stdout else "",
                        'stderr': result.stderr if result.stderr else "",
                        'method': 'subprocess',
                        **real_stats  # Incluir estadísticas reales
                    }
                    
                    self.logger.info(f"COMPLETADA ITERACIÓN {iteration_params['iteration_id']} en {iteration_time:.2f}s")
                    self.logger.info("="*60)
                    return True, iteration_stats
                else:
                    # Error en la ejecución
                    error_msg = f"Error en subprocess (código {result.returncode}): {result.stderr if result.stderr else 'Sin mensaje de error'}"
                    self.logger.error(f"FALLÓ ITERACIÓN {iteration_params['iteration_id']}: {error_msg}")
                    self.logger.info("="*60)
                    return False, {
                        'execution_time': iteration_time,
                        'error': error_msg,
                        'stdout': result.stdout if result.stdout else "",
                        'stderr': result.stderr if result.stderr else "",
                        'returncode': result.returncode,
                        'method': 'subprocess_failed'
                    }
            
        except Exception as e:
            iteration_time = time.time() - iteration_start_time
            error_msg = f"Error en {iteration_params['iteration_id']}: {str(e)}"
            self.logger.error(f"FALLÓ ITERACIÓN {iteration_params['iteration_id']}: {error_msg}")
            self.logger.error(traceback.format_exc())
            self.logger.info("="*60)
            
            return False, {
                'execution_time': iteration_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def save_batch_results(self, batch_config) -> None:
        """
        Guarda todos los resultados del batch de manera organizada
        """
        # Resumen estadístico
        summary = self.stats.get_summary()
        
        # Metadata del batch (compatible con ambas configuraciones)
        batch_metadata = {
            'batch_id': self.batch_id,
            'batch_config': batch_config.__dict__ if hasattr(batch_config, '__dict__') else str(batch_config),
            'statistics': summary,
            'run_registry': self.run_registry,
            'generation_info': {
                'code_version': '1.0.0',  # Versión del código
                'python_version': sys.version,
                'timestamp': datetime.now().isoformat(),
                'working_directory': str(Path.cwd()),
                'git_commit': None  # Placeholder para hash de git
            }
        }
        
        # Guardar metadata
        metadata_file = self.batch_dir / "batch_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # Crear resumen en CSV para análisis fácil
        if self.run_registry:
            df_runs = pd.DataFrame(self.run_registry)
            df_runs.to_csv(self.batch_dir / "run_summary.csv", index=False)
        
        # Generar visualización de configuración automáticamente
        try:
            print("Generando visualización de configuración del batch...")
            # config_visualizer eliminado - skip visualización avanzada
            self.logger.info("Visualización de configuración omitida (módulo no disponible)")
        except Exception as e:
            self.logger.warning(f"No se pudo generar visualización de configuración: {str(e)}")
        
        # Resumen estadístico en texto plano (sin emojis para Windows)
        with open(self.batch_dir / "statistical_summary.txt", 'w', encoding='utf-8') as f:
            f.write("RESUMEN ESTADISTICO DEL BATCH RUN\n")
            f.write("="*50 + "\n\n")
            f.write(f"Batch ID: {self.batch_id}\n")
            f.write(f"Duracion total: {summary['batch_metadata']['total_duration_formatted']}\n")
            f.write(f"Runs completados: {summary['run_statistics']['runs_completed']}\n")
            f.write(f"Runs fallidos: {summary['run_statistics']['runs_failed']}\n")
            f.write(f"Tasa de exito: {summary['run_statistics']['success_rate']:.1%}\n")
            f.write(f"Tiempo promedio por run: {summary['run_statistics']['average_execution_time']:.2f}s\n")
            f.write(f"Detecciones totales: {summary['detection_statistics']['total_detections']}\n")
            f.write(f"Observaciones totales: {summary['detection_statistics']['total_observations']}\n")
            f.write(f"Tasa de deteccion global: {summary['detection_statistics']['global_detection_rate']:.3f}\n\n")
            
            f.write("Tasas de deteccion por tipo de SN:\n")
            for sn_type, stats in summary['detection_statistics']['detection_rates_by_type'].items():
                f.write(f"  {sn_type}: {stats['mean']:.3f} +/- {stats['std']:.3f} "
                       f"(mediana: {stats['median']:.3f}, n={stats['n_runs']})\n")
        
        self.logger.info(f"Resultados guardados en: {self.batch_dir}")
        self.logger.info(f"Metadata: {metadata_file}")
    
    def run_batch(self, batch_config) -> Dict:
        """
        Ejecuta un batch completo de simulaciones
        """
        self.logger.info("INICIANDO BATCH RUN PROFESIONAL")
        self.logger.info("="*80)
        self.logger.info(f"Configuración del Batch:")
        self.logger.info(f"   • Número de runs: {batch_config.n_runs}")
        self.logger.info(f"   • Tipos de SN: {list(batch_config.sn_type_distribution.keys())}")
        self.logger.info(f"   • Distribución SN: {batch_config.sn_type_distribution}")
        self.logger.info(f"   • Rango redshift: {batch_config.redshift_range}")
        self.logger.info(f"   • Rango extinción: {getattr(batch_config, 'extinction_range', 'N/A')}")
        self.logger.info(f"   • Volume weighted: {batch_config.volume_weighted}")
        self.logger.info(f"   • Surveys: {list(batch_config.survey_distribution.keys())}")
        self.logger.info(f"   • Semilla base: {batch_config.base_seed}")
        self.logger.info(f"   • Batch ID: {self.batch_id}")
        self.logger.info("="*80)
        
        self.stats.start_batch()
        
        for i in range(batch_config.n_runs):
            # Generar parámetros de la iteración
            iteration_params = self.create_run_parameters(batch_config, i, batch_config.n_runs)
            
            # Ejecutar iteración (una supernova)
            success, iteration_results = self.execute_single_run(iteration_params)
            
            # Registrar resultados
            iteration_record = {**iteration_params, **iteration_results, 'success': success}
            self.run_registry.append(iteration_record)
            
            if success:
                self.stats.add_successful_run(
                    iteration_results['execution_time'],
                    iteration_results.get('n_detections', 0),
                    iteration_results.get('n_observations', 0),
                    iteration_params['sn_type']
                )
            else:
                self.stats.add_failed_run(iteration_params['iteration_id'], iteration_results.get('error', 'Unknown error'))
            
            # Progreso intermedio cada 10 iteraciones
            if (i + 1) % 10 == 0 or (i + 1) == batch_config.n_runs:
                completed = self.stats.runs_completed
                failed = self.stats.runs_failed
                total_processed = completed + failed
                progress_pct = (total_processed / batch_config.n_runs) * 100
                
                self.logger.info("PROGRESO INTERMEDIO")
                self.logger.info(f"   Procesados: {total_processed}/{batch_config.n_runs} ({progress_pct:.1f}%)")
                self.logger.info(f"   Exitosos: {completed} | Fallidos: {failed}")
                if self.stats.execution_times:
                    avg_time = np.mean(self.stats.execution_times)
                    remaining = batch_config.n_runs - total_processed
                    eta_minutes = (remaining * avg_time) / 60
                    self.logger.info(f"   Tiempo promedio: {avg_time:.2f}s | ETA: {eta_minutes:.1f}min")
                self.logger.info("-"*40)
            
            # Pausa entre runs si está configurada
            pause_time = getattr(batch_config, 'pause_between_runs', 0.0)
            if pause_time > 0:
                time.sleep(pause_time)
        
        self.stats.end_batch()
        
        # Guardar resultados
        self.save_batch_results(batch_config)
        
        # Resumen final
        summary = self.stats.get_summary()
        self.logger.info("BATCH COMPLETADO")
        self.logger.info("="*80)
        self.logger.info(f"RESUMEN FINAL:")
        self.logger.info(f"   • Runs exitosos: {summary['run_statistics']['runs_completed']}")
        self.logger.info(f"   • Runs fallidos: {summary['run_statistics']['runs_failed']}")
        self.logger.info(f"   • Tasa de éxito: {summary['run_statistics']['success_rate']:.1%}")
        self.logger.info(f"   • Duración total: {summary['batch_metadata']['total_duration_formatted']}")
        self.logger.info(f"   • Tiempo promedio por run: {summary['run_statistics']['average_execution_time']:.2f}s")
        self.logger.info(f"   • Resultados guardados en: {self.batch_dir}")
        self.logger.info("="*80)
        
        return summary

# Función de conveniencia para uso directo
def run_scientific_batch(batch_config) -> Dict:
    """
    Función de alto nivel para ejecutar un batch científico
    """
    runner = ProfessionalBatchRunner()
    return runner.run_batch(batch_config)

if __name__ == "__main__":
    # Ejemplo de uso (deshabilitado - usar simple_runner.py)
    print("Use simple_runner.py para ejecutar batches")
    # from academic_batch_config import create_thesis_level_config
    
    # # Crear configuración para tesis doctoral
    # thesis_config = create_thesis_level_config(
    #     n_runs=50,  # 50 runs para demostración
    #     redshift_max=0.3,
    #     include_all_types=True
    # )
    
    # print("Ejecutando batch de tesis doctoral...")
    # results = run_scientific_batch(thesis_config)
    
    # print("\nBatch completado!")
    # print(f"Tasa de éxito: {results['run_statistics']['success_rate']:.1%}")
    # print(f"Duración: {results['batch_metadata']['total_duration_formatted']}")
