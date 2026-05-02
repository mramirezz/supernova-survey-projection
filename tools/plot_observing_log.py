"""
PLOT OBSERVING LOG POR OID
===========================

Grafica el footprint observacional de cada OID: MJD vs diffmaglim
separado por filtro. Ayuda a inspeccionar cadencia + profundidad
antes de lanzar simulaciones.

Uso:
    python tools/plot_observing_log.py --oids ZTF18aazucuo ZTF18aaiytox
    python tools/plot_observing_log.py --oids-file path/to/oids.txt
    python tools/plot_observing_log.py --oids-file top10.txt --output-dir outputs/plots_top10

Output:
    <output-dir>/<oid>.png          (1 plot por OID)
    <output-dir>/overview_grid.png  (resumen 2x5)
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSLOG = ROOT / "data" / "ZTF_observing_log_complete.csv"

FILTER_COLOR = {'g': '#2ca02c', 'r': '#d62728', 'i': '#9467bd'}
FID_TO_FILT = {1: 'g', 2: 'r', 3: 'i'}


def load_obslog(path, oids):
    """Carga solo las filas de los OIDs pedidos."""
    cols = ['oid', 'mjd', 'fid', 'diffmaglim', 'is_detection']
    df = pd.read_csv(path, usecols=cols)
    df = df[df['oid'].isin(oids)].copy()
    df['filter'] = df['fid'].map(FID_TO_FILT)
    return df


def plot_single_oid(df_oid, ax, title=None, legend=True):
    """Dibuja MJD vs diffmaglim coloreado por filtro en un Axes dado."""
    for filt in ['g', 'r', 'i']:
        sub = df_oid[df_oid['filter'] == filt]
        if sub.empty:
            continue
        color = FILTER_COLOR[filt]
        ax.scatter(sub['mjd'], sub['diffmaglim'],
                   c=color, s=6, alpha=0.55,
                   label=f'{filt} ({len(sub):,})')

    ax.invert_yaxis()  # magnitudes: valores menores = más brillante arriba
    ax.set_xlabel('MJD')
    ax.set_ylabel('diffmaglim (mag)')
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.25)


def oid_stats(df_oid):
    """Estadísticas breves para el título."""
    n = len(df_oid)
    span = df_oid['mjd'].max() - df_oid['mjd'].min()
    counts = df_oid['filter'].value_counts()
    g = int(counts.get('g', 0))
    r = int(counts.get('r', 0))
    i = int(counts.get('i', 0))
    maglim_p50 = df_oid['diffmaglim'].median()
    return f"n={n:,} (g={g}, r={r}, i={i}) · span={int(span)}d · maglim₅₀={maglim_p50:.2f}"


def main():
    parser = argparse.ArgumentParser(description='Plot observing log por OID')
    parser.add_argument('--oids', nargs='+', default=None,
                        help='Lista de OIDs separados por espacio')
    parser.add_argument('--oids-file', type=str, default=None,
                        help='Archivo de texto con un OID por línea')
    parser.add_argument('--obslog', type=str, default=str(DEFAULT_OBSLOG),
                        help=f'Path al observing log CSV (default: {DEFAULT_OBSLOG})')
    parser.add_argument('--output-dir', type=str, default='outputs/obslog_plots',
                        help='Directorio de salida')
    parser.add_argument('--no-grid', action='store_true',
                        help='No generar la figura overview grid')
    args = parser.parse_args()

    # Resolver OIDs
    oids = []
    if args.oids:
        oids.extend(args.oids)
    if args.oids_file:
        with open(args.oids_file) as f:
            oids.extend([line.strip() for line in f if line.strip()])
    if not oids:
        print('ERROR: proveer --oids o --oids-file', file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Cargando obslog y filtrando {len(oids)} OIDs...')
    df = load_obslog(args.obslog, oids)
    print(f'  {len(df):,} filas cargadas')

    # Plots individuales
    for oid in oids:
        df_oid = df[df['oid'] == oid]
        if df_oid.empty:
            print(f'  [WARN] {oid}: sin observaciones en obslog')
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        title = f'{oid}\n{oid_stats(df_oid)}'
        plot_single_oid(df_oid, ax, title=title)
        fig.tight_layout()
        out = os.path.join(args.output_dir, f'{oid}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'  {oid}: {out}')

    # Figura overview 2x5
    if not args.no_grid and len(oids) > 1:
        ncols = 5 if len(oids) >= 5 else len(oids)
        nrows = int(np.ceil(len(oids) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                                 squeeze=False)
        for idx, oid in enumerate(oids):
            ax = axes[idx // ncols, idx % ncols]
            df_oid = df[df['oid'] == oid]
            if df_oid.empty:
                ax.text(0.5, 0.5, f'{oid}\n(sin datos)', ha='center', va='center')
                ax.axis('off')
                continue
            plot_single_oid(df_oid, ax, title=oid, legend=(idx == 0))
            ax.tick_params(labelsize=8)
        # ocultar axes sobrantes
        for k in range(len(oids), nrows * ncols):
            axes[k // ncols, k % ncols].axis('off')
        fig.suptitle(f'Observing log — {len(oids)} OIDs (MJD vs diffmaglim)', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_grid = os.path.join(args.output_dir, 'overview_grid.png')
        fig.savefig(out_grid, dpi=110)
        plt.close(fig)
        print(f'  overview: {out_grid}')


if __name__ == '__main__':
    main()
