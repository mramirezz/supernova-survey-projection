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
            iteration_label = f"iter_{iter_idx:04d}"
            
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
            
            # Importar y ejecutar main_multiband
            import main_multiband
            
            # Ejecutar la función main_multiband
            main_multiband.main_multiband(config=validated_config)
            
            iteration_time = time.time() - iteration_start_time
            
            # Extraer estadísticas reales del CSV generado
            real_stats = self.extract_run_statistics(iteration_params)
            
            iteration_stats = {
                'execution_time': iteration_time,
                'method': 'multiband_direct_import',
                **real_stats
            }
            
            self.logger.info(f"COMPLETADA ITERACIÓN {iteration_params['iteration_id']} en {iteration_time:.2f}s")
            
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
