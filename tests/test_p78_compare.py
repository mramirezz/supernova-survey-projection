"""Test para compare_features.py"""
import tempfile
from pathlib import Path
import pandas as pd
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline78.compare_features import compare


def test_compare_basic():
    """Test básico: dos features.csv con medida consistente de |Δ|/σ."""

    # Crear CSVs temporales con datos de prueba
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # CSV A: parámetros de referencia
        csv_a_data = {
            'oid': ['oid1', 'oid2', 'oid3'],
            'part_index': [0, 0, 0],
            'sn_type': ['Ia', 'II', 'Ibc'],
            'filter_band': ['g', 'r', 'g'],
            'A': [1e-6, 2e-6, 1.5e-6],
            'f': [0.3, 0.5, 0.4],
            'f_err': [0.01, 0.01, 0.01],
            't_rise': [20.0, 25.0, 22.0],
            't_rise_err': [1.0, 1.0, 1.0],
            't_fall': [50.0, 55.0, 52.0],
            't_fall_err': [2.0, 2.0, 2.0],
            'gamma': [100.0, 110.0, 105.0],
            'gamma_err': [5.0, 5.0, 5.0],
            'elapsed_s': [1.5, 1.6, 1.7],
        }
        df_a = pd.DataFrame(csv_a_data)
        csv_a_path = tmpdir / 'features_a.csv'
        df_a.to_csv(csv_a_path, index=False)

        # CSV B: parámetros ligeramente diferentes
        csv_b_data = {
            'oid': ['oid1', 'oid2', 'oid3'],
            'part_index': [0, 0, 0],
            'sn_type': ['Ia', 'II', 'Ibc'],
            'filter_band': ['g', 'r', 'g'],
            'A': [1.05e-6, 2.05e-6, 1.55e-6],  # pequeñas diferencias
            'f': [0.305, 0.505, 0.405],
            'f_err': [0.01, 0.01, 0.01],
            't_rise': [20.5, 25.5, 22.5],
            't_rise_err': [1.0, 1.0, 1.0],
            't_fall': [50.5, 55.5, 52.5],
            't_fall_err': [2.0, 2.0, 2.0],
            'gamma': [101.0, 111.0, 106.0],
            'gamma_err': [5.0, 5.0, 5.0],
            'elapsed_s': [1.4, 1.5, 1.6],
        }
        df_b = pd.DataFrame(csv_b_data)
        csv_b_path = tmpdir / 'features_b.csv'
        df_b.to_csv(csv_b_path, index=False)

        # Llamar a compare
        result = compare(str(csv_a_path), str(csv_b_path))

        # Verificar que tenemos 8 filas esperadas (f, t_rise, t_fall, gamma, log10A, tiempo_b/a, solo_en_a, solo_en_b)
        assert len(result) == 8, f"Expected 8 rows, got {len(result)}"

        # Verificar que "f" tiene una mediana razonable
        f_row = result[result['par'] == 'f'].iloc[0]
        assert f_row['n'] == 3, f"Expected n=3 for f, got {f_row['n']}"
        assert f_row['mediana'] < 1.0, f"Median |Δf|/σ should be < 1, got {f_row['mediana']}"
        assert not np.isnan(f_row['p90']) and f_row['p90'] >= f_row['mediana'], f"p90 should be >= median"

        # Verificar que "solo_en_a" y "solo_en_b" son 0
        solo_a = result[result['par'] == 'solo_en_a'].iloc[0]
        solo_b = result[result['par'] == 'solo_en_b'].iloc[0]
        assert solo_a['n'] == 0, f"Expected 0 rows only in A, got {solo_a['n']}"
        assert solo_b['n'] == 0, f"Expected 0 rows only in B, got {solo_b['n']}"

        # Verificar que tiempo_b/a es cercano a 1
        tiempo = result[result['par'] == 'tiempo_b/a'].iloc[0]
        assert 0.8 < tiempo['mediana'] < 1.2, f"Time ratio should be ~1, got {tiempo['mediana']}"

        print("✓ test_compare_basic passed")


def test_zero_error_case():
    """Test con errores cero: debe dar NaN, no inf."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # CSV A y B idénticos pero con errores cero
        csv_data = {
            'oid': ['oid1', 'oid2'],
            'part_index': [0, 0],
            'sn_type': ['Ia', 'II'],
            'filter_band': ['g', 'r'],
            'A': [1e-6, 2e-6],
            'f': [0.3, 0.5],
            'f_err': [0.0, 0.0],  # Error cero
            't_rise': [20.0, 25.0],
            't_rise_err': [0.0, 0.0],  # Error cero
            't_fall': [50.0, 55.0],
            't_fall_err': [0.0, 0.0],  # Error cero
            'gamma': [100.0, 110.0],
            'gamma_err': [0.0, 0.0],  # Error cero
            'elapsed_s': [1.5, 1.6],
        }

        df_a = pd.DataFrame(csv_data)
        csv_a_path = tmpdir / 'features_a.csv'
        df_a.to_csv(csv_a_path, index=False)

        df_b = pd.DataFrame(csv_data)
        csv_b_path = tmpdir / 'features_b.csv'
        df_b.to_csv(csv_b_path, index=False)

        # Llamar a compare
        result = compare(str(csv_a_path), str(csv_b_path))

        # Verificar que los NaN se reemplazan (no inf) incluso con error cero
        f_row = result[result['par'] == 'f'].iloc[0]
        assert f_row['n'] == 2, f"Expected n=2 for f, got {f_row['n']}"
        # Con error cero y diferencias cero, debería ser NaN (después del .replace(0, np.nan))
        assert np.isnan(f_row['mediana']), f"Expected NaN with zero errors, got {f_row['mediana']}"

        print("✓ test_zero_error_case passed")


def test_mismatches():
    """Test con registros únicos en A y B."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # CSV A: tiene oid1, oid2, oid3
        csv_a_data = {
            'oid': ['oid1', 'oid2', 'oid3'],
            'part_index': [0, 0, 0],
            'sn_type': ['Ia', 'II', 'Ibc'],
            'filter_band': ['g', 'r', 'g'],
            'A': [1e-6, 2e-6, 1.5e-6],
            'f': [0.3, 0.5, 0.4],
            'f_err': [0.01, 0.01, 0.01],
            't_rise': [20.0, 25.0, 22.0],
            't_rise_err': [1.0, 1.0, 1.0],
            't_fall': [50.0, 55.0, 52.0],
            't_fall_err': [2.0, 2.0, 2.0],
            'gamma': [100.0, 110.0, 105.0],
            'gamma_err': [5.0, 5.0, 5.0],
            'elapsed_s': [1.5, 1.6, 1.7],
        }
        df_a = pd.DataFrame(csv_a_data)
        csv_a_path = tmpdir / 'features_a.csv'
        df_a.to_csv(csv_a_path, index=False)

        # CSV B: tiene oid1, oid2, oid4 (falta oid3, añade oid4)
        csv_b_data = {
            'oid': ['oid1', 'oid2', 'oid4'],
            'part_index': [0, 0, 0],
            'sn_type': ['Ia', 'II', 'Ibc'],
            'filter_band': ['g', 'r', 'g'],
            'A': [1.05e-6, 2.05e-6, 1.5e-6],
            'f': [0.305, 0.505, 0.4],
            'f_err': [0.01, 0.01, 0.01],
            't_rise': [20.5, 25.5, 22.0],
            't_rise_err': [1.0, 1.0, 1.0],
            't_fall': [50.5, 55.5, 52.0],
            't_fall_err': [2.0, 2.0, 2.0],
            'gamma': [101.0, 111.0, 105.0],
            'gamma_err': [5.0, 5.0, 5.0],
            'elapsed_s': [1.4, 1.5, 1.7],
        }
        df_b = pd.DataFrame(csv_b_data)
        csv_b_path = tmpdir / 'features_b.csv'
        df_b.to_csv(csv_b_path, index=False)

        # Llamar a compare
        result = compare(str(csv_a_path), str(csv_b_path))

        # Verificar que solo_en_a es 1 (oid3)
        solo_a = result[result['par'] == 'solo_en_a'].iloc[0]
        assert solo_a['n'] == 1, f"Expected 1 row only in A, got {solo_a['n']}"

        # Verificar que solo_en_b es 1 (oid4)
        solo_b = result[result['par'] == 'solo_en_b'].iloc[0]
        assert solo_b['n'] == 1, f"Expected 1 row only in B, got {solo_b['n']}"

        # Verificar que la merge contiene 2 registros (oid1, oid2)
        f_row = result[result['par'] == 'f'].iloc[0]
        assert f_row['n'] == 2, f"Expected n=2 for f, got {f_row['n']}"

        print("✓ test_mismatches passed")


def test_duplicates():
    """Test con duplicados en las keys: drop_duplicates debe limpiarlo."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # CSV A: tiene oid1 dos veces (duplicado)
        csv_a_data = {
            'oid': ['oid1', 'oid1', 'oid2'],
            'part_index': [0, 0, 0],
            'sn_type': ['Ia', 'Ia', 'II'],
            'filter_band': ['g', 'g', 'r'],
            'A': [1e-6, 1.01e-6, 2e-6],  # Segundo oid1 ligeramente distinto
            'f': [0.3, 0.305, 0.5],
            'f_err': [0.01, 0.01, 0.01],
            't_rise': [20.0, 20.1, 25.0],
            't_rise_err': [1.0, 1.0, 1.0],
            't_fall': [50.0, 50.1, 55.0],
            't_fall_err': [2.0, 2.0, 2.0],
            'gamma': [100.0, 100.5, 110.0],
            'gamma_err': [5.0, 5.0, 5.0],
            'elapsed_s': [1.5, 1.55, 1.6],
        }
        df_a = pd.DataFrame(csv_a_data)
        csv_a_path = tmpdir / 'features_a.csv'
        df_a.to_csv(csv_a_path, index=False)

        # CSV B: limpio sin duplicados
        csv_b_data = {
            'oid': ['oid1', 'oid2'],
            'part_index': [0, 0],
            'sn_type': ['Ia', 'II'],
            'filter_band': ['g', 'r'],
            'A': [1.05e-6, 2.05e-6],
            'f': [0.305, 0.505],
            'f_err': [0.01, 0.01],
            't_rise': [20.5, 25.5],
            't_rise_err': [1.0, 1.0],
            't_fall': [50.5, 55.5],
            't_fall_err': [2.0, 2.0],
            'gamma': [101.0, 111.0],
            'gamma_err': [5.0, 5.0],
            'elapsed_s': [1.4, 1.5],
        }
        df_b = pd.DataFrame(csv_b_data)
        csv_b_path = tmpdir / 'features_b.csv'
        df_b.to_csv(csv_b_path, index=False)

        # Llamar a compare
        result = compare(str(csv_a_path), str(csv_b_path))

        # Después de drop_duplicates, A debería tener 2 registros únicos
        # y merge debería tener 2 registros (oid1, oid2)
        f_row = result[result['par'] == 'f'].iloc[0]
        assert f_row['n'] == 2, f"Expected n=2 after drop_duplicates, got {f_row['n']}"

        print("✓ test_duplicates passed")


if __name__ == '__main__':
    test_compare_basic()
    test_zero_error_case()
    test_mismatches()
    test_duplicates()
    print("\n✓ All tests passed!")
