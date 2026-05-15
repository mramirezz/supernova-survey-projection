"""
ENCUENTRA CAMPOS ZTF CON MEJOR COBERTURA FOTOMÉTRICA
=====================================================
Selecciona los N objetos transitorios del catálogo ZTF con mayor número
total de observaciones y exporta la lista de forma reproducible.

Criterio de selección:
  - Se calcula cuántas observaciones tiene cada objeto en total
    (sumando todas las noches en todos los filtros ópticos: g, r, i).
  - Se ordenan de mayor a menor número de observaciones.
  - Se toman los primeros N (por defecto 1000).
  - No se aplica ningún filtro adicional: se priorizan los objetos
    con mayor cobertura observacional independientemente del filtro.

Uso:
  python tools/find_best_fields.py                  # top 1000 (default)
  python tools/find_best_fields.py --n-fields 500   # top 500
  python tools/find_best_fields.py --no-plots       # sin generar figuras

Salidas (en outputs/field_analysis/):
  oids_selected.txt    — lista de OIDs seleccionados (uno por línea)
  oids_with_stats.csv  — tabla con estadísticas de cada objeto
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Selecciona los N objetos ZTF con más observaciones."
    )
    parser.add_argument(
        "--n-fields", type=int, default=1000,
        help="Cuántos objetos seleccionar (default: 1000)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/field_analysis"),
        help="Directorio donde guardar los archivos de salida."
    )
    parser.add_argument(
        "--obslog", type=Path, default=Path("data/ZTF_observing_log_complete.csv"),
        help="Ruta al log de observaciones de ZTF."
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Omitir la generación de figuras."
    )
    return parser.parse_args()


def compute_stats(df):
    """
    Para cada objeto (OID), calcula:
      - n_total   : total de noches de observación en todos los filtros
      - n_g       : noches en filtro verde  (fid=1, banda g, ~480 nm)
      - n_r       : noches en filtro rojo   (fid=2, banda r, ~640 nm)
      - n_i       : noches en infrarrojo cercano (fid=3, banda i, ~750 nm)
      - span_days : duración del seguimiento en días (última obs - primera obs)

    En ZTF el filtro se codifica como entero (fid): 1=g, 2=r, 3=i.
    """
    def oid_stats(g):
        return pd.Series({
            "n_total":    len(g),
            "n_g":        (g["fid"] == 1).sum(),
            "n_r":        (g["fid"] == 2).sum(),
            "n_i":        (g["fid"] == 3).sum(),
            "span_days":  g["mjd"].max() - g["mjd"].min(),
        })

    stats = (
        df.groupby("oid")
        .apply(oid_stats, include_groups=False)
        .sort_values("n_total", ascending=False)
        .reset_index()
    )
    return stats


def export_selected(stats, n_fields, output_dir, obslog_path):
    """Exporta oids_selected.txt y oids_with_stats.csv con encabezados descriptivos."""
    selected = stats.head(n_fields)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- oids_selected.txt ---
    txt_path = output_dir / "oids_selected.txt"
    header_lines = [
        f"# OIDs seleccionados para simulación de supernovas — generado {generated_at}",
        f"# Fuente: {obslog_path}",
        f"# Criterio: top {n_fields} objetos con mayor número total de observaciones ZTF",
        f"# Total seleccionados: {len(selected)}",
        f"# Rango de observaciones: {int(selected['n_total'].max())} (1°) "
        f"a {int(selected['n_total'].min())} ({len(selected)}°)",
        "#",
        "# Cada línea es el identificador único del objeto en ZTF (OID).",
        "# Este archivo se usa como entrada para run_per_field.py --oids-file.",
    ]
    with open(txt_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        for oid in selected["oid"]:
            f.write(oid + "\n")
    print(f"✓ Exportado: {txt_path}  ({len(selected)} OIDs)")

    # --- oids_with_stats.csv ---
    csv_path = output_dir / "oids_with_stats.csv"
    csv_header = (
        f"# Estadísticas de cobertura fotométrica — generado {generated_at}\n"
        f"# Fuente: {obslog_path}\n"
        f"# Criterio de selección: top {n_fields} por n_total (mayor a menor)\n"
        "#\n"
        "# Columnas:\n"
        "#   oid        — identificador único del objeto transitorio en ZTF\n"
        "#   n_total    — total de noches de observación (suma de todos los filtros)\n"
        "#   n_g        — noches en filtro verde (banda g, ~480 nm)\n"
        "#   n_r        — noches en filtro rojo (banda r, ~640 nm)\n"
        "#   n_i        — noches en infrarrojo cercano (banda i, ~750 nm)\n"
        "#   span_days  — duración del seguimiento en días (primera a última observación)\n"
    )
    with open(csv_path, "w") as f:
        f.write(csv_header)
        selected.to_csv(f, index=False)
    print(f"✓ Exportado: {csv_path}")

    return selected


def print_summary(stats, selected, n_fields):
    print("=" * 70)
    print(f"ANÁLISIS DE CAMPOS ZTF — TOP {n_fields} POR COBERTURA OBSERVACIONAL")
    print("=" * 70)
    print(f"\nTotal objetos únicos en el catálogo: {len(stats):,}")
    print(f"Total observaciones en el catálogo:  {int(stats['n_total'].sum()):,}")
    print(f"Promedio de obs por objeto:          {stats['n_total'].mean():.1f}")
    print(f"\nTop {n_fields} seleccionados:")
    print(f"  Más observado:  {selected.iloc[0]['oid']}  "
          f"({int(selected.iloc[0]['n_total'])} noches, "
          f"{selected.iloc[0]['span_days']:.0f} días de seguimiento)")
    print(f"  El número {n_fields}: {selected.iloc[-1]['oid']}  "
          f"({int(selected.iloc[-1]['n_total'])} noches, "
          f"{selected.iloc[-1]['span_days']:.0f} días de seguimiento)")

    print("\n" + "=" * 70)
    print("TOP 20 OBJETOS MÁS OBSERVADOS")
    print("=" * 70)
    print(f"{'#':<5} {'OID':<20} {'Total obs':<12} "
          f"{'Verde (g)':<12} {'Rojo (r)':<12} {'Infrarrojo (i)':<16} {'Seguimiento (d)'}")
    print("-" * 85)
    for rank, row in enumerate(selected.head(20).itertuples(), 1):
        print(f"{rank:<5} {row.oid:<20} {int(row.n_total):<12} "
              f"{int(row.n_g):<12} {int(row.n_r):<12} {int(row.n_i):<16} "
              f"{row.span_days:.0f}")


def make_plots(df, selected, output_dir):
    # fid en ZTF: 1=g (verde), 2=r (rojo), 3=i (infrarrojo)
    fid_meta = {1: ("g", "green"), 2: ("r", "red"), 3: ("i", "purple")}
    maglim_col = "diffmaglim" if "diffmaglim" in df.columns else "maglimit"

    # --- Top 5 más observados ---
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))
    fig.suptitle("TOP 5 OBJETOS ZTF MÁS OBSERVADOS", fontsize=16, fontweight="bold")

    for idx, row in enumerate(selected.head(5).itertuples()):
        ax = axes[idx]
        df_field = df[df["oid"] == row.oid]
        for fid, (label, color) in fid_meta.items():
            d = df_field[df_field["fid"] == fid]
            if len(d) > 0:
                ax.scatter(d["mjd"], d[maglim_col], c=color,
                           label=f"{label} ({len(d)} noches)", alpha=0.6, s=30)
        ax.invert_yaxis()
        ax.set_ylabel("Mag límite", fontsize=10, fontweight="bold")
        ax.set_title(f"#{idx+1}: {row.oid}  ({int(row.n_total)} obs, "
                     f"{row.span_days:.0f} días)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        if idx == 4:
            ax.set_xlabel("MJD (Tiempo Juliano Modificado)", fontsize=10)

    plt.tight_layout()
    path = output_dir / "top5_most_observed.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Figura: {path}")

    # --- Histograma de distribución de cobertura ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DISTRIBUCIÓN DE COBERTURA FOTOMÉTRICA — OBJETOS SELECCIONADOS",
                 fontsize=13, fontweight="bold")

    axes[0].hist(selected["n_total"], bins=50, color="steelblue", edgecolor="white", lw=0.5)
    axes[0].set_xlabel("Total de noches de observación", fontsize=11)
    axes[0].set_ylabel("Número de objetos", fontsize=11)
    axes[0].set_title("Distribución por total de observaciones", fontsize=11)
    axes[0].axvline(selected["n_total"].median(), color="orange", linestyle="--",
                    label=f"Mediana: {selected['n_total'].median():.0f}")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(selected["span_days"], bins=50, color="salmon", edgecolor="white", lw=0.5)
    axes[1].set_xlabel("Duración del seguimiento (días)", fontsize=11)
    axes[1].set_ylabel("Número de objetos", fontsize=11)
    axes[1].set_title("Distribución por duración del seguimiento", fontsize=11)
    axes[1].axvline(selected["span_days"].median(), color="orange", linestyle="--",
                    label=f"Mediana: {selected['span_days'].median():.0f} d")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "coverage_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Figura: {path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando log de observaciones: {args.obslog}")
    df = pd.read_csv(args.obslog)
    print(f"  {len(df):,} filas | {df['oid'].nunique():,} objetos únicos")

    print("Calculando estadísticas por objeto...")
    stats = compute_stats(df)

    selected = export_selected(stats, args.n_fields, args.output_dir, args.obslog)
    print_summary(stats, selected, args.n_fields)

    if not args.no_plots:
        print("\nGenerando figuras...")
        make_plots(df, selected, args.output_dir)

    print("\n" + "=" * 70)
    print(f"Archivos guardados en: {args.output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

