"""
BATCH RUNNER MULTI-BANDA
========================
Sistema para ejecutar simulaciones masivas con proyección multi-banda simultánea.
"""

import sys
import os

# Agregar flag para indicar que estamos en modo multi-banda
os.environ['MULTIBAND_MODE'] = '1'

# Importar el batch_runner normal
from batch_runner import ProfessionalBatchRunner, run_scientific_batch

# La única diferencia es que usaremos main_multiband en lugar de main
class MultibandBatchRunner(ProfessionalBatchRunner):
    """
    Versión del BatchRunner que usa proyección multi-banda simultánea
    """
    
    def execute_single_run(self, iteration_params: dict) -> tuple:
        """
        Sobrescribe el método para usar main_multiband en lugar de main
        """
        import time
        iteration_start_time = time.time()
        
        # Separador visual
        self.logger.info("="*60)
        self.logger.info(f"ITERACIÓN {iteration_params['iteration_id']} ({iteration_params['iteration_index']+1}/{iteration_params['total_iterations']})")
        self.logger.info(f"SN: {iteration_params['sn_type']} {iteration_params['sn_name']}")
        self.logger.info(f"z={iteration_params['redshift']:.4f}, E(B-V)_host={iteration_params['ebmv_host']:.3f}, E(B-V)_MW={iteration_params['ebmv_mw']:.3f}")
        self.logger.info(f"Survey: {iteration_params['survey']}, Seed: {iteration_params['seed']}")
        self.logger.info("-"*60)
        
        try:
            # Formatear iteration_label
            iter_idx = iteration_params['iteration_index']
            attempt = int(iteration_params.get('attempt', 1))
            iteration_label = f"iter_{iter_idx:04d}" if attempt <= 1 else f"iter_{iter_idx:04d}_try_{attempt:02d}"
            
            # Agregar info de batch al config
            iteration_params['batch_id'] = self.batch_id
            iteration_params['iteration_label'] = iteration_label
            
            # Actualizar configuración
            self.update_config_for_run(iteration_params)
            
            # Validar configuración
            from config_loader import load_and_validate_config
            validated_config = load_and_validate_config()
            
            # Pasar batch_id, iteration_label y datos de la SN al config validado
            validated_config['batch_id'] = self.batch_id
            validated_config['iteration_label'] = iteration_label
            validated_config['sn_name'] = iteration_params['sn_name']
            validated_config['tipo'] = iteration_params['sn_type']
            validated_config['sn_type'] = iteration_params['sn_type']

            # Control de guardado (para reintentos: no guardar intentos intermedios)
            if 'processing' in validated_config and isinstance(validated_config['processing'], dict):
                validated_config['processing']['save_outputs'] = bool(iteration_params.get('save_outputs', True))
                # Parámetros opcionales para criterio mínimo de detecciones (runner por lista)
                if iteration_params.get('required_filter') is not None:
                    validated_config['processing']['required_filter'] = str(iteration_params['required_filter'])
                if iteration_params.get('min_detections_required') is not None:
                    validated_config['processing']['min_detections_required'] = int(iteration_params['min_detections_required'])
                if iteration_params.get('offset_search_mode') is not None:
                    validated_config['processing']['offset_search_mode'] = str(iteration_params['offset_search_mode'])
                if iteration_params.get('force_brighten_to_min_detections') is not None:
                    validated_config['processing']['force_brighten_to_min_detections'] = bool(iteration_params['force_brighten_to_min_detections'])
                if iteration_params.get('max_force_brightening_mag') is not None:
                    validated_config['processing']['max_force_brightening_mag'] = float(iteration_params['max_force_brightening_mag'])

            # Fijar semilla local para la normalización de luminosidad (sin tocar RNG global)
            if 'luminosity' in validated_config and isinstance(validated_config['luminosity'], dict):
                lum_seed = iteration_params.get('luminosity_random_seed', None)
                if lum_seed is not None:
                    validated_config['luminosity']['use_reproducible_sampling'] = True
                    validated_config['luminosity']['random_seed'] = int(lum_seed)
            
            # Importar y ejecutar main_multiband
            import main_multiband
            
            # Ejecutar la función main_multiband
            df_projected = main_multiband.main_multiband(config=validated_config)
            
            iteration_time = time.time() - iteration_start_time
            
            # Estadísticas directas desde el DataFrame retornado
            n_obs = int(len(df_projected)) if df_projected is not None else 0
            n_det = 0
            det_by_filter = {}
            obs_by_filter = {}
            if df_projected is not None and n_obs > 0:
                if 'filter' in df_projected.columns:
                    for f, df_f in df_projected.groupby('filter'):
                        obs_by_filter[str(f)] = int(len(df_f))
                        if 'upperlimit' in df_f.columns:
                            det_by_filter[str(f)] = int((df_f['upperlimit'] == 'F').sum())
                        elif 'detected' in df_f.columns:
                            det_by_filter[str(f)] = int(df_f['detected'].sum())
                        else:
                            det_by_filter[str(f)] = 0
                # Totales
                n_det = int(sum(det_by_filter.values())) if det_by_filter else 0
            
            iteration_stats = {
                'execution_time': iteration_time,
                'method': 'multiband_direct_import',
                'n_observations': n_obs,
                'n_detections': n_det,
                'detections_by_filter': det_by_filter,
                'observations_by_filter': obs_by_filter,
                # compat (algunos logs/plots usan estas claves)
                'observations': n_obs,
                'detections': n_det,
            }
            
            self.logger.info(f"COMPLETADA ITERACIÓN {iteration_params['iteration_id']} en {iteration_time:.2f}s")
            # Liberación explícita de memoria (útil en corridas largas)
            try:
                import gc
                del df_projected
                gc.collect()
            except Exception:
                pass
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass

            return (True, iteration_stats)
            
        except Exception as e:
            iteration_time = time.time() - iteration_start_time
            error_msg = str(e)
            self.logger.error(f"ERROR en iteración {iteration_params['iteration_id']}: {error_msg}")
            
            import traceback
            self.logger.error(traceback.format_exc())
            
            return (False, {
                'execution_time': iteration_time,
                'method': 'multiband_direct_import',
                'n_detections': 0,
                'n_observations': 0,
                'detections': 0,
                'observations': 0,
                'error': error_msg
            })

def run_multiband_batch(batch_config) -> dict:
    """
    Ejecuta un batch científico con proyección multi-banda simultánea
    
    Parameters:
    -----------
    batch_config : SimpleConfig
        Configuración del batch
    
    Returns:
    --------
    dict : Resultados del batch con estadísticas
    """
    runner = MultibandBatchRunner()
    return runner.run_batch(batch_config)

if __name__ == "__main__":
    print("="*60)
    print("BATCH RUNNER MULTI-BANDA")
    print("="*60)
    print("\nUse simple_runner_multiband.py para ejecutar batches")
    print("="*60)
