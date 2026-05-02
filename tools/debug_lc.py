"""
Debug visual de curvas de luz proyectadas por campo.

Genera UN PNG por OID con:
    - Arriba: grilla observacional real del campo (todas las obs ZTF en g/r/i,
      con maglim vs MJD, gaps estacionales visibles).
    - Abajo: mosaico 3x10 con las 30 simulaciones (tipos x part_index). Cada
      celda muestra la LC proyectada sobre la grilla del campo recortada al
      rango de la SN, con detecciones y upper limits, y subtitulo con el
      template y params físicos usados.

Uso:
    python tools/debug_lc.py <parquet_path> [<parquet_path> ...]
        [--out-dir DIR] [--grid-only]

Ejemplos:
    # Un OID
    python tools/debug_lc.py outputs/pilot_20260501_v3/20260501_230540/ZTF21acjcarx.parquet

    # Todos los OIDs de un run (glob)
    python tools/debug_lc.py outputs/pilot_20260501_v3/consolidated/*.parquet

    # Solo la grilla, sin simulaciones
    python tools/debug_lc.py <parquet> --grid-only
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILTER_COLOR = {'g': '#2ca02c', 'r': '#d62728', 'i': '#8c564b'}
TYPE_ORDER = ['Ia', 'II', 'Ibc']
N_DIVISIONS = 10


def load_field_obslog(oid, obslog_path='data/ZTF_observing_log_complete.csv'):
    """Carga las observaciones reales del campo desde el CSV global."""
    if not os.path.exists(obslog_path):
        return None
    df = pd.read_csv(obslog_path)
    df = df[df['oid'] == oid].copy()
    if 'filter' not in df.columns and 'fid' in df.columns:
        df['filter'] = df['fid'].map({1: 'g', 2: 'r', 3: 'i'})
    if 'maglimit' not in df.columns and 'diffmaglim' in df.columns:
        df['maglimit'] = df['diffmaglim']
    return df


def plot_field_grid(ax, df_obs, title=''):
    """Dibuja la grilla observacional del campo (maglim vs MJD por filtro)."""
    for filt in ['g', 'r', 'i']:
        sub = df_obs[df_obs['filter'] == filt]
        if len(sub) == 0:
            continue
        ax.scatter(sub['mjd'], sub['maglimit'],
                   c=FILTER_COLOR[filt], s=6, alpha=0.45,
                   label=f'{filt} ({len(sub)})')
    ax.invert_yaxis()
    ax.set_xlabel('MJD')
    ax.set_ylabel('maglimit')
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=9, ncol=3)


def plot_sim_cell(ax, df_sim, df_obs, tipo, part_idx):
    """Dibuja UNA simulación en una celda del mosaico.

    df_sim: subset de filas del parquet correspondientes a esta (tipo, part_idx).
    df_obs: grilla observacional completa del campo.
    """
    if df_sim.empty:
        ax.set_title(f'{tipo} p{part_idx}\n[FAIL]', fontsize=8, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    template = df_sim['template'].iloc[0]
    z = df_sim['z'].iloc[0]
    ebmv_host = df_sim['ebmv_host'].iloc[0]
    ebmv_mw = df_sim['ebmv_mw'].iloc[0]
    desplazamiento = df_sim['desplazamiento'].iloc[0]
    n_det = int(df_sim['detected'].sum())
    n_tot = len(df_sim)

    # Ventana temporal de la SN
    t_min, t_max = df_sim['mjd'].min(), df_sim['mjd'].max()
    pad = max(5, 0.05 * (t_max - t_min))

    # Grilla del campo recortada al rango de la SN (contexto en fondo claro)
    obs_window = df_obs[(df_obs['mjd'] >= t_min - pad) & (df_obs['mjd'] <= t_max + pad)]
    for filt in ['g', 'r', 'i']:
        sub = obs_window[obs_window['filter'] == filt]
        if len(sub) == 0:
            continue
        ax.scatter(sub['mjd'], sub['maglimit'],
                   c=FILTER_COLOR[filt], marker='_', s=30,
                   alpha=0.35, linewidth=0.8)

    # Curva proyectada: detecciones y upper limits
    for filt in ['g', 'r', 'i']:
        sub = df_sim[df_sim['filter'] == filt]
        if len(sub) == 0:
            continue
        det = sub[sub['detected']]
        upl = sub[~sub['detected']]
        if len(det):
            ax.scatter(det['mjd'], det['magnitud_proyectada'],
                       c=FILTER_COLOR[filt], s=18, alpha=0.9, zorder=3,
                       edgecolors='black', linewidths=0.3)
        if len(upl):
            ax.scatter(upl['mjd'], upl['magnitud_proyectada'],
                       c=FILTER_COLOR[filt], marker='v', s=12, alpha=0.4,
                       zorder=2)

    ax.invert_yaxis()
    # Subtítulo compacto con todos los parámetros
    subtitle = (f'{tipo} p{part_idx}  tpl={template}\n'
                f'z={z:.3f}  E_h={ebmv_host:.2f}  E_mw={ebmv_mw:.3f}\n'
                f'det={n_det}/{n_tot}  δ={desplazamiento:.0f}d')
    ax.set_title(subtitle, fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.grid(alpha=0.2)


def debug_oid(parquet_path, out_dir=None, grid_only=False,
              obslog_path='data/ZTF_observing_log_complete.csv'):
    """Genera el PNG de debug para un OID."""
    df = pd.read_parquet(parquet_path)
    if df.empty:
        print(f'[WARN] {parquet_path} vacío, skip')
        return None

    oid = df['oid'].iloc[0]
    df_obs = load_field_obslog(oid, obslog_path)
    if df_obs is None or df_obs.empty:
        print(f'[WARN] No pude cargar obslog para {oid}; uso grilla derivada del parquet')
        df_obs = df[['mjd', 'filter', 'maglimit']].drop_duplicates()

    out_dir = out_dir or os.path.join(os.path.dirname(parquet_path), 'debug')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{oid}.png')

    if grid_only:
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_field_grid(ax, df_obs,
                        title=f'OID {oid} · grilla ZTF · {len(df_obs)} obs · '
                              f'MJD [{df_obs["mjd"].min():.0f}, {df_obs["mjd"].max():.0f}] '
                              f'(span {df_obs["mjd"].max()-df_obs["mjd"].min():.0f}d)')
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f'  Grid only: {out_path}')
        return out_path

    # Figura con grilla arriba + mosaico 3x10 abajo
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(
        nrows=4, ncols=10,
        height_ratios=[1.3, 1, 1, 1],
        hspace=0.6, wspace=0.3
    )

    # Row 0: grilla completa del campo
    ax_grid = fig.add_subplot(gs[0, :])
    span = df_obs['mjd'].max() - df_obs['mjd'].min()
    plot_field_grid(ax_grid, df_obs,
                    title=f'OID {oid} · {len(df_obs)} obs · '
                          f'span {span:.0f} d · '
                          f'sims OK: {df.groupby(["sn_type","part_index"]).ngroups}/30')

    # Marcar divisiones del pivote determinístico
    t_min, t_max = df_obs['mjd'].min(), df_obs['mjd'].max()
    part_size = (t_max - t_min) / N_DIVISIONS
    for i in range(N_DIVISIONS + 1):
        ax_grid.axvline(t_min + i * part_size, color='gray',
                        ls=':', alpha=0.4, lw=0.8)
    for i in range(N_DIVISIONS):
        center = t_min + part_size * (i + 0.5)
        ax_grid.text(center, ax_grid.get_ylim()[1], f'p{i}',
                     ha='center', va='top', fontsize=8, color='gray')

    # Rows 1-3: 3x10 mosaico de simulaciones
    for row, tipo in enumerate(TYPE_ORDER, start=1):
        for col in range(N_DIVISIONS):
            ax = fig.add_subplot(gs[row, col])
            df_sim = df[(df['sn_type'] == tipo) & (df['part_index'] == col)]
            plot_sim_cell(ax, df_sim, df_obs, tipo, col)

    fig.suptitle(f'Debug per-field · {os.path.basename(parquet_path)}',
                 fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  {oid} → {out_path}')
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('parquets', nargs='+',
                    help='Uno o más paths a parquets (acepta globs expandidos por shell)')
    ap.add_argument('--out-dir', default=None,
                    help='Directorio de salida (default: <parquet_dir>/debug/)')
    ap.add_argument('--grid-only', action='store_true',
                    help='Solo dibujar grilla del campo, sin sims')
    ap.add_argument('--obslog', default='data/ZTF_observing_log_complete.csv',
                    help='Path al observing log ZTF (para grilla completa)')
    args = ap.parse_args()

    # Expandir globs en caso que el shell no lo haya hecho
    paths = []
    for p in args.parquets:
        matches = sorted(glob.glob(p)) or [p]
        paths.extend(matches)
    paths = [p for p in paths if p.endswith('.parquet') and os.path.exists(p)]

    if not paths:
        print('[ERROR] No se encontraron parquets')
        sys.exit(1)

    print(f'Procesando {len(paths)} parquet(s)...')
    for p in paths:
        debug_oid(p, out_dir=args.out_dir, grid_only=args.grid_only,
                  obslog_path=args.obslog)
    print(f'[OK] {len(paths)} PNGs generados')


if __name__ == '__main__':
    main()
