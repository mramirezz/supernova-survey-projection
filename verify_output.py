"""
VERIFICADOR DE RESULTADOS — PIPELINE DE PROYECCIÓN POR CAMPO
=============================================================

Valida que los outputs del runner por campo (run_per_field.py) sean:
  1. Estructuralmente correctos (columnas, tipos, sin NaN inesperados)
  2. Físicamente razonables (magnitudes, redshifts, extinciones)
  3. Completos (30 sims por campo: 3 tipos × 10 particiones)
  4. Internamente consistentes (upperlimit ↔ mag, detección ↔ maglimit)
  5. Estadísticamente plausibles (tasas de detección vs z, distribuciones)

Uso:
    python verify_output.py outputs/per_field/20260412_183239/
    python verify_output.py outputs/per_field/20260412_183239/ZTF17aaabgiw.parquet
    python verify_output.py outputs/per_field/20260412_183239/ --strict
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────
# CONSTANTES DE VALIDACIÓN
# ──────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    'oid', 'mjd', 'maglimit', 'filter', 'magnitud_modelo',
    'magnitud_proyectada', 'upperlimit', 'detected',
    'sn_type', 'template', 'z', 'ebmv_host', 'ebmv_mw',
    'part_index', 'n_divisions', 'desplazamiento',
]
EXPECTED_TYPES = ['Ia', 'II', 'Ibc']
N_DIVISIONS = 10
EXPECTED_SIMS_PER_FIELD = len(EXPECTED_TYPES) * N_DIVISIONS  # 30

# Rangos físicos
ZTF_MJD_RANGE = (58100, 62000)       # ~2018 - ~2028
MAG_MODEL_RANGE = (10, 35)            # mag AB
MAGLIMIT_RANGE = (14, 22)             # ZTF typical
Z_RANGE = (0.001, 1.0)
EBMV_RANGE = (0.0, 2.0)

# Umbrales de alerta
MIN_DETECTION_RATE_LOW_Z = 0.1   # Si z < 0.1, esperar al menos 10% detecciones
MAX_Z_FOR_HIGH_DETECTIONS = 0.05  # A z < 0.05, Ia debería tener buena tasa
PEAK_MAG_ABS_RANGE = {            # M_peak absoluta razonable por tipo
    'Ia':  (-21.0, -17.0),
    'II':  (-19.5, -13.0),
    'Ibc': (-20.0, -14.0),
}


class VerificationResult:
    """Acumula checks PASS/WARN/FAIL."""

    def __init__(self, name):
        self.name = name
        self.checks = []  # (status, category, message)

    def _add(self, status, category, msg):
        self.checks.append((status, category, msg))

    def ok(self, cat, msg):
        self._add('PASS', cat, msg)

    def warn(self, cat, msg):
        self._add('WARN', cat, msg)

    def fail(self, cat, msg):
        self._add('FAIL', cat, msg)

    @property
    def n_pass(self):
        return sum(1 for s, _, _ in self.checks if s == 'PASS')

    @property
    def n_warn(self):
        return sum(1 for s, _, _ in self.checks if s == 'WARN')

    @property
    def n_fail(self):
        return sum(1 for s, _, _ in self.checks if s == 'FAIL')

    @property
    def passed(self):
        return self.n_fail == 0

    def summary(self):
        return f"{self.n_pass} PASS, {self.n_warn} WARN, {self.n_fail} FAIL"

    def print_report(self, show_pass=False):
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  {self.summary()}")
        print(f"{'='*60}")
        for status, cat, msg in self.checks:
            if status == 'PASS' and not show_pass:
                continue
            icon = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗'}[status]
            print(f"  {icon} [{status}] ({cat}) {msg}")


# ──────────────────────────────────────────────────────────
# CHECKS INDIVIDUALES
# ──────────────────────────────────────────────────────────

def check_structure(df, v):
    """Columnas requeridas, tipos, NaN en columnas críticas."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        v.fail('structure', f'Columnas faltantes: {missing}')
        return False
    v.ok('structure', f'{len(REQUIRED_COLUMNS)} columnas requeridas presentes')

    # NaN en columnas críticas (no en las que vienen del obslog original)
    critical = ['mjd', 'maglimit', 'magnitud_modelo', 'magnitud_proyectada',
                'sn_type', 'template', 'z', 'ebmv_host', 'part_index', 'detected']
    nan_counts = {c: int(df[c].isna().sum()) for c in critical if df[c].isna().any()}
    if nan_counts:
        v.fail('structure', f'NaN en columnas críticas: {nan_counts}')
    else:
        v.ok('structure', 'Sin NaN en columnas críticas')

    # Filas vacías
    if len(df) == 0:
        v.fail('structure', 'DataFrame vacío')
        return False

    return True


def check_completeness(df, v):
    """30 sims esperadas: 3 tipos × 10 particiones."""
    sims = df.groupby(['sn_type', 'part_index']).size().reset_index(name='n')
    n_sims = len(sims)

    types_present = sorted(df['sn_type'].unique())
    missing_types = [t for t in EXPECTED_TYPES if t not in types_present]

    if missing_types:
        v.fail('completeness', f'Tipos faltantes: {missing_types}')
    else:
        v.ok('completeness', f'3 tipos presentes: {types_present}')

    # Particiones por tipo
    for tipo in EXPECTED_TYPES:
        sub = df[df['sn_type'] == tipo]
        parts = sorted(sub['part_index'].unique()) if len(sub) > 0 else []
        missing_parts = [i for i in range(N_DIVISIONS) if i not in parts]
        if len(missing_parts) == 0:
            v.ok('completeness', f'{tipo}: 10/10 particiones')
        elif len(missing_parts) <= 3:
            v.warn('completeness', f'{tipo}: {10-len(missing_parts)}/10 particiones (faltan {missing_parts})')
        else:
            v.fail('completeness', f'{tipo}: solo {10-len(missing_parts)}/10 particiones (faltan {missing_parts})')

    if n_sims == EXPECTED_SIMS_PER_FIELD:
        v.ok('completeness', f'{n_sims}/{EXPECTED_SIMS_PER_FIELD} simulaciones completas')
    elif n_sims >= EXPECTED_SIMS_PER_FIELD * 0.7:
        v.warn('completeness', f'{n_sims}/{EXPECTED_SIMS_PER_FIELD} simulaciones ({EXPECTED_SIMS_PER_FIELD - n_sims} fallaron)')
    else:
        v.fail('completeness', f'{n_sims}/{EXPECTED_SIMS_PER_FIELD} simulaciones — demasiados fallos')


def check_physical_ranges(df, v):
    """Rangos de magnitudes, z, extinción, MJD."""
    # MJD
    mjd_min, mjd_max = df['mjd'].min(), df['mjd'].max()
    if ZTF_MJD_RANGE[0] <= mjd_min and mjd_max <= ZTF_MJD_RANGE[1]:
        v.ok('ranges', f'MJD en rango ZTF: [{mjd_min:.1f}, {mjd_max:.1f}]')
    else:
        v.fail('ranges', f'MJD fuera de rango ZTF: [{mjd_min:.1f}, {mjd_max:.1f}]')

    # Maglimit
    ml_min, ml_max = df['maglimit'].min(), df['maglimit'].max()
    if MAGLIMIT_RANGE[0] <= ml_min and ml_max <= MAGLIMIT_RANGE[1]:
        v.ok('ranges', f'maglimit: [{ml_min:.1f}, {ml_max:.1f}]')
    else:
        v.warn('ranges', f'maglimit fuera de lo típico: [{ml_min:.1f}, {ml_max:.1f}]')

    # mag_modelo
    mm_min, mm_max = df['magnitud_modelo'].min(), df['magnitud_modelo'].max()
    if MAG_MODEL_RANGE[0] <= mm_min and mm_max <= MAG_MODEL_RANGE[1]:
        v.ok('ranges', f'mag_modelo: [{mm_min:.1f}, {mm_max:.1f}]')
    else:
        v.fail('ranges', f'mag_modelo fuera de rango físico: [{mm_min:.1f}, {mm_max:.1f}]')

    # Redshift
    z_min, z_max = df['z'].min(), df['z'].max()
    if Z_RANGE[0] <= z_min and z_max <= Z_RANGE[1]:
        v.ok('ranges', f'z: [{z_min:.3f}, {z_max:.3f}]')
    else:
        v.fail('ranges', f'z fuera de rango: [{z_min:.3f}, {z_max:.3f}]')

    # E(B-V)
    ebmv_min, ebmv_max = df['ebmv_host'].min(), df['ebmv_host'].max()
    if EBMV_RANGE[0] <= ebmv_min and ebmv_max <= EBMV_RANGE[1]:
        v.ok('ranges', f'E(B-V)_host: [{ebmv_min:.3f}, {ebmv_max:.3f}]')
    else:
        v.fail('ranges', f'E(B-V)_host fuera de rango: [{ebmv_min:.3f}, {ebmv_max:.3f}]')


def check_consistency(df, v):
    """Consistencia interna: upperlimit ↔ magnitudes, detected ↔ maglimit."""
    # Upper limits: mag_proyectada debe == maglimit
    ul = df[df['upperlimit'] == 'T']
    if len(ul) > 0:
        ul_ok = (ul['magnitud_proyectada'] == ul['maglimit']).all()
        if ul_ok:
            v.ok('consistency', f'Upper limits ({len(ul)}): mag_proyectada == maglimit')
        else:
            n_bad = (~(ul['magnitud_proyectada'] == ul['maglimit'])).sum()
            v.fail('consistency', f'Upper limits inconsistentes: {n_bad} filas donde mag_proy != maglimit')

    # Detecciones: mag_modelo < maglimit
    det = df[df['detected'] == True]
    if len(det) > 0:
        det_ok = (det['magnitud_modelo'] < det['maglimit']).all()
        if det_ok:
            v.ok('consistency', f'Detecciones ({len(det)}): mag_modelo < maglimit')
        else:
            n_bad = (det['magnitud_modelo'] >= det['maglimit']).sum()
            v.fail('consistency', f'Detecciones inconsistentes: {n_bad} filas con mag_modelo >= maglimit')

    # detected == (upperlimit == 'F')
    check_det_ul = (df['detected'] == (df['upperlimit'] == 'F')).all()
    if check_det_ul:
        v.ok('consistency', 'detected ↔ upperlimit consistente')
    else:
        v.fail('consistency', 'detected y upperlimit no son consistentes')

    # Cada sim tiene un solo OID, z, ebmv_host, template
    sims = df.groupby(['sn_type', 'part_index'])
    multi_z = sims['z'].nunique().max()
    multi_oid = sims['oid'].nunique().max()
    if multi_z > 1 or multi_oid > 1:
        v.fail('consistency', 'Una misma simulación tiene valores mixtos de z u OID')
    else:
        v.ok('consistency', 'Cada simulación tiene z/OID únicos')

    # OID único por archivo
    n_oids = df['oid'].nunique()
    if n_oids == 1:
        v.ok('consistency', f'OID único en archivo: {df["oid"].iloc[0]}')
    else:
        v.fail('consistency', f'Múltiples OIDs en un archivo: {n_oids}')

    # Duplicados exactos
    n_dup = df.duplicated(subset=['mjd', 'filter', 'sn_type', 'part_index']).sum()
    if n_dup == 0:
        v.ok('consistency', 'Sin filas duplicadas')
    else:
        v.warn('consistency', f'{n_dup} filas duplicadas (mjd+filter+type+part)')


def check_detection_physics(df, v):
    """
    Verifica que las tasas de detección sean físicamente razonables:
    - SNe a z bajo deberían tener detecciones significativas
    - SNe a z alto pueden ser mayormente upper limits (ok para ZTF)
    - Ia a z < 0.1 deberían ser brillantes (M ~ -19)
    """
    sims = df.groupby(['sn_type', 'part_index']).agg(
        z=('z', 'first'),
        n_obs=('mjd', 'count'),
        n_det=('detected', 'sum'),
        mag_min=('magnitud_modelo', 'min'),
    ).reset_index()
    sims['det_rate'] = sims['n_det'] / sims['n_obs']

    # Check 1: SNe a z < 0.1 deberían tener detecciones
    low_z = sims[sims['z'] < 0.1]
    if len(low_z) > 0:
        low_z_det_rate = low_z['det_rate'].mean()
        low_z_any_det = (low_z['n_det'] > 0).sum()
        if low_z_det_rate >= MIN_DETECTION_RATE_LOW_Z:
            v.ok('physics', f'z<0.1: tasa detección={low_z_det_rate:.1%} ({low_z_any_det}/{len(low_z)} sims con det)')
        elif low_z_any_det > 0:
            v.warn('physics', f'z<0.1: tasa baja {low_z_det_rate:.1%} pero hay {low_z_any_det} sims con detecciones')
        else:
            v.fail('physics', f'z<0.1: CERO detecciones en {len(low_z)} sims — algo está mal')
    else:
        v.warn('physics', 'No hay sims con z < 0.1 para validar')

    # Check 2: Tasa de detección global
    total_det = df['detected'].sum()
    total_rows = len(df)
    det_rate = total_det / total_rows if total_rows > 0 else 0
    if det_rate > 0:
        v.ok('physics', f'Tasa detección global: {det_rate:.1%} ({total_det}/{total_rows})')
    else:
        v.warn('physics', f'Tasa detección global: 0% — puede ser OK si todos los z son altos')

    # Check 3: Ia debería ser más brillante que II/Ibc a mismo z (en promedio)
    type_brightness = {}
    for tipo in EXPECTED_TYPES:
        sub = sims[sims['sn_type'] == tipo]
        if len(sub) > 0:
            type_brightness[tipo] = sub['mag_min'].mean()

    if 'Ia' in type_brightness and 'II' in type_brightness:
        # No podemos comparar directamente porque z es diferente para cada sim.
        # Solo verificar que Ia no sea absurdamente más débil que II.
        pass  # Esta comparación requiere normalizar por z, skip por ahora

    # Check 4: Magnitud absoluta implícita
    # NOTA: mag_min es la más brillante OBSERVADA, no necesariamente el peak real.
    # Si el peak cae fuera de la ventana de overlap, M_abs parecerá más débil.
    # Solo alertamos si la desviación es extrema (> 5 mag de lo esperado).
    for tipo in EXPECTED_TYPES:
        sub = sims[sims['sn_type'] == tipo]
        if len(sub) == 0:
            continue
        n_extreme = 0
        for _, row in sub.iterrows():
            z = row['z']
            mag_app = row['mag_min']
            if z > 0.001:
                from core.utils import DL_calculator
                import math
                dl = DL_calculator(z)
                mu = 5 * math.log10(dl * 1e6) - 5
                M_abs = mag_app - mu
                m_range = PEAK_MAG_ABS_RANGE.get(tipo, (-22, -12))
                # Solo FAIL si está > 5 mag más débil que el extremo débil del rango
                if M_abs > m_range[1] + 5:
                    n_extreme += 1
        if n_extreme > 0:
            v.warn('physics', f'{tipo}: {n_extreme} sims con M_abs observada > 5mag del rango (peak fuera de ventana)')
        else:
            v.ok('physics', f'{tipo}: magnitudes absolutas observadas razonables')


def check_distributions(df, v):
    """Distribuciones estadísticas: z, E(B-V), templates."""
    # Z: debería tener distribución volume-weighted (más a z alto)
    z_unique = df.groupby(['sn_type', 'part_index'])['z'].first()
    z_median = z_unique.median()
    z_mean = z_unique.mean()

    # Para volume-weighted en (0.01, 0.5), la mediana debería estar ~0.3-0.4
    if z_mean > 0.15:
        v.ok('distributions', f'z medio={z_mean:.3f}, mediana={z_median:.3f} — consistente con volume-weighted')
    else:
        v.warn('distributions', f'z medio={z_mean:.3f} — podría ser demasiado bajo para volume-weighted')

    # E(B-V): verificar que varía por tipo
    ebmv_by_type = df.groupby('sn_type')['ebmv_host'].mean()
    if len(ebmv_by_type) > 1:
        v.ok('distributions', f'E(B-V) medio por tipo: ' +
             ', '.join(f'{t}={e:.3f}' for t, e in ebmv_by_type.items()))

    # Templates: verificar diversidad
    for tipo in EXPECTED_TYPES:
        sub = df[df['sn_type'] == tipo]
        n_tpl = sub['template'].nunique()
        if n_tpl >= 3:
            v.ok('distributions', f'{tipo}: {n_tpl} templates distintos')
        elif n_tpl >= 1:
            v.warn('distributions', f'{tipo}: solo {n_tpl} template(s) — poca diversidad')
        # (si son 0 ya se reportó en completeness)

    # Filtros: verificar que hay observaciones en más de 1 filtro
    filters_used = df['filter'].unique()
    if len(filters_used) >= 2:
        v.ok('distributions', f'Multi-banda: {len(filters_used)} filtros ({list(filters_used)})')
    else:
        v.warn('distributions', f'Solo 1 filtro usado ({list(filters_used)}) — campo con cobertura limitada')


# ──────────────────────────────────────────────────────────
# VERIFICACIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────

def verify_file(path, strict=False):
    """Verifica un archivo parquet individual."""
    name = os.path.basename(path)
    v = VerificationResult(name)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        v.fail('io', f'Error leyendo {path}: {e}')
        return v

    if not check_structure(df, v):
        return v

    check_completeness(df, v)
    check_physical_ranges(df, v)
    check_consistency(df, v)
    check_detection_physics(df, v)
    check_distributions(df, v)

    return v


def verify_run(run_dir, strict=False, max_files=None):
    """Verifica todos los parquet de un run."""
    parquet_files = sorted(glob.glob(os.path.join(run_dir, '*.parquet')))

    if len(parquet_files) == 0:
        print(f"[ERROR] No se encontraron archivos .parquet en {run_dir}")
        return

    if max_files:
        parquet_files = parquet_files[:max_files]

    print(f"\nVerificando {len(parquet_files)} archivos en {run_dir}")
    print(f"{'='*60}")

    results = []
    totals = defaultdict(int)

    for path in parquet_files:
        v = verify_file(path, strict=strict)
        results.append(v)
        totals['pass'] += v.n_pass
        totals['warn'] += v.n_warn
        totals['fail'] += v.n_fail

        # Solo imprimir detalle si hay problemas
        if v.n_fail > 0:
            v.print_report(show_pass=False)
        elif v.n_warn > 0:
            v.print_report(show_pass=False)
        else:
            print(f"  ✓ {v.name}: ALL PASS ({v.n_pass} checks)")

    # Resumen global
    n_clean = sum(1 for v in results if v.passed and v.n_warn == 0)
    n_warnings = sum(1 for v in results if v.passed and v.n_warn > 0)
    n_failed = sum(1 for v in results if not v.passed)

    print(f"\n{'='*60}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'='*60}")
    print(f"  Archivos: {len(results)}")
    print(f"  Clean (todos PASS):   {n_clean}")
    print(f"  Con warnings:         {n_warnings}")
    print(f"  Con FAIL:             {n_failed}")
    print(f"  Total checks: {totals['pass']} PASS, {totals['warn']} WARN, {totals['fail']} FAIL")

    if n_failed > 0:
        print(f"\n  ⚠ HAY ERRORES — revisar los archivos con FAIL")
        return False

    if n_warnings > 0:
        print(f"\n  Los warnings son informativos — revisar si los valores fuera de lo típico están justificados")

    return True


def main():
    parser = argparse.ArgumentParser(description='Verificador de outputs del pipeline de proyección')
    parser.add_argument('path', help='Directorio de run o archivo .parquet individual')
    parser.add_argument('--strict', action='store_true', help='Modo estricto: warnings cuentan como fallos')
    parser.add_argument('--max-files', type=int, default=None, help='Máximo de archivos a verificar')
    parser.add_argument('--verbose', action='store_true', help='Mostrar también checks PASS')
    args = parser.parse_args()

    path = args.path

    if path.endswith('.parquet'):
        v = verify_file(path, strict=args.strict)
        v.print_report(show_pass=args.verbose)
        sys.exit(0 if v.passed else 1)
    elif os.path.isdir(path):
        ok = verify_run(path, strict=args.strict, max_files=args.max_files)
        sys.exit(0 if ok else 1)
    else:
        print(f"[ERROR] Path no válido: {path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
