"""
Genera PNGs de las curvas de luz simuladas del piloto.
Para cada OID: 1 figura con 3 subplots (Ia, II, Ibc), cada uno muestra las 10
simulaciones en g/r/i. Detecciones = puntos, upper limits = triángulos ↓.

Uso:
    python tools/plot_pilot_lcs.py <pilot_base_dir> [--max-oids N]

Ejemplo:
    python tools/plot_pilot_lcs.py outputs/pilot_20260501
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILTER_COLOR = {'g': '#2ca02c', 'r': '#d62728', 'i': '#9467bd'}
TYPE_ORDER = ['Ia', 'II', 'Ibc']


def plot_oid(parquet_path, out_path):
    df = pd.read_parquet(parquet_path)
    oid = df['oid'].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Piloto · OID {oid} · 30 sims (3 tipos × 10 pivotes)', fontsize=13)

    for ax, tipo in zip(axes, TYPE_ORDER):
        df_t = df[df['sn_type'] == tipo]
        if df_t.empty:
            ax.set_title(f'{tipo}: sin simulaciones')
            continue

        for (part_idx, filt), grp in df_t.groupby(['part_index', 'filter']):
            color = FILTER_COLOR.get(filt, 'gray')
            det = grp[grp['detected']]
            upl = grp[~grp['detected']]
            ax.scatter(det['mjd'], det['magnitud_proyectada'],
                       c=color, s=14, alpha=0.7,
                       label=f'{filt} det' if part_idx == 0 else None)
            ax.scatter(upl['mjd'], upl['magnitud_proyectada'],
                       c=color, marker='v', s=10, alpha=0.25,
                       label=f'{filt} uplim' if part_idx == 0 else None)

        n_sims = df_t[['part_index', 'template']].drop_duplicates().shape[0]
        n_det = int(df_t['detected'].sum())
        n_tot = len(df_t)
        eta = 100 * n_det / n_tot if n_tot else 0
        ax.set_title(f'{tipo}: {n_sims} sims · {n_det}/{n_tot} det ({eta:.1f}%)',
                     fontsize=11)
        ax.invert_yaxis()
        ax.set_ylabel('mag (proyectada)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right', ncol=3)

    axes[-1].set_xlabel('MJD')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pilot_base', help='outputs/pilot_YYYYMMDD (contiene subdirs timestamped)')
    ap.add_argument('--max-oids', type=int, default=10)
    args = ap.parse_args()

    pq_paths = sorted(
        glob.glob(os.path.join(args.pilot_base, '*.parquet'))
        + glob.glob(os.path.join(args.pilot_base, '*', '*.parquet'))
    )
    pq_paths = [p for p in pq_paths if 'run_summary' not in os.path.basename(p)]
    if not pq_paths:
        print(f'[ERROR] No parquets encontrados en {args.pilot_base}/*/')
        sys.exit(1)

    out_dir = os.path.join(args.pilot_base, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    for pq in pq_paths[: args.max_oids]:
        oid = os.path.basename(pq).replace('.parquet', '')
        out_path = os.path.join(out_dir, f'{oid}.png')
        print(f'  Plot: {oid} → {out_path}')
        plot_oid(pq, out_path)

    print(f'\n[OK] {min(len(pq_paths), args.max_oids)} PNGs en {out_dir}/')


if __name__ == '__main__':
    main()
