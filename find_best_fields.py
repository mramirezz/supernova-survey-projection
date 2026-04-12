"""
ENCUENTRA CAMPOS ZTF CON MEJOR COBERTURA
=========================================
Script para encontrar los campos ZTF más observados
para usar en pruebas con fixed_field
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cargar obslog ZTF
obslog_path = "data/ZTF_observing_log_complete.csv"
df = pd.read_csv(obslog_path)

print("=" * 70)
print("ANÁLISIS DE CAMPOS ZTF - MEJORES PARA PRUEBAS")
print("=" * 70)

# Verificar columnas
print(f"\nColumnas disponibles: {df.columns.tolist()}")
print(f"Primeras filas:\n{df.head()}\n")

# Contar observaciones por OID
obs_per_oid = df.groupby('oid').size().sort_values(ascending=False)

print(f"\nTotal de campos únicos: {len(obs_per_oid):,}")
print(f"Total de observaciones: {len(df):,}")
print(f"Promedio obs/campo: {len(df)/len(obs_per_oid):.1f}")

print("\n" + "=" * 70)
print("TOP 20 CAMPOS MÁS OBSERVADOS")
print("=" * 70)
print(f"{'Rank':<6} {'OID':<20} {'Total Obs':<12} {'g':<8} {'r':<8} {'i':<8}")
print("-" * 70)

for rank, (oid, total) in enumerate(obs_per_oid.head(20).items(), 1):
    df_field = df[df['oid'] == oid]
    
    # Contar por filtro
    g_count = len(df_field[df_field['filter'] == 'g'])
    r_count = len(df_field[df_field['filter'] == 'r'])
    i_count = len(df_field[df_field['filter'] == 'i'])
    
    print(f"{rank:<6} {oid:<20} {total:<12} {g_count:<8} {r_count:<8} {i_count:<8}")

# Análisis temporal del mejor campo
best_oid = obs_per_oid.index[0]
df_best = df[df['oid'] == best_oid]

print("\n" + "=" * 70)
print(f"ANÁLISIS DETALLADO DEL MEJOR CAMPO: {best_oid}")
print("=" * 70)

print(f"\nTotal observaciones: {len(df_best)}")
print(f"Rango temporal: MJD {df_best['mjd'].min():.1f} - {df_best['mjd'].max():.1f}")
print(f"Duración: {df_best['mjd'].max() - df_best['mjd'].min():.1f} días")

print("\nDistribución por filtro:")
for filt in ['g', 'r', 'i']:
    df_filt = df_best[df_best['filter'] == filt]
    if len(df_filt) > 0:
        print(f"  {filt}: {len(df_filt):4d} obs | "
              f"MJD {df_filt['mjd'].min():.1f} - {df_filt['mjd'].max():.1f} | "
              f"maglim med: {df_filt['maglimit'].median():.2f}")

print("\n" + "=" * 70)
print("CAMPOS CON COBERTURA MULTI-BANDA BALANCEADA")
print("=" * 70)
print(f"{'OID':<20} {'Total':<8} {'g':<8} {'r':<8} {'i':<8} {'Balance':<10}")
print("-" * 70)

# Buscar campos con buena cobertura en los 3 filtros
balanced_fields = []
for oid in obs_per_oid.head(100).index:
    df_field = df[df['oid'] == oid]
    g_count = len(df_field[df_field['filter'] == 'g'])
    r_count = len(df_field[df_field['filter'] == 'r'])
    i_count = len(df_field[df_field['filter'] == 'i'])
    
    total = g_count + r_count + i_count
    
    # Calcular balance (desviación estándar normalizada)
    if i_count > 0:  # Solo campos con las 3 bandas
        counts = np.array([g_count, r_count, i_count])
        balance = np.std(counts) / np.mean(counts)
        balanced_fields.append({
            'oid': oid,
            'total': total,
            'g': g_count,
            'r': r_count,
            'i': i_count,
            'balance': balance
        })

# Ordenar por menor desbalance
balanced_fields_sorted = sorted(balanced_fields, key=lambda x: x['balance'])

for field in balanced_fields_sorted[:10]:
    print(f"{field['oid']:<20} {field['total']:<8} "
          f"{field['g']:<8} {field['r']:<8} {field['i']:<8} "
          f"{field['balance']:<10.3f}")

print("\n" + "=" * 70)
print("RECOMENDACIONES PARA config.py")
print("=" * 70)
print("\nPara usar un campo fijo, edita config.py:")
print(f"\n    'fixed_field': '{best_oid}',  # Campo más observado")
if balanced_fields_sorted:
    print(f"    # O mejor balanceado: '{balanced_fields_sorted[0]['oid']}'")
print("\nPara volver a modo aleatorio:")
print("    'fixed_field': None,")

# =========================================================================
# PLOTS DE VISUALIZACIÓN
# =========================================================================

print("\n" + "=" * 70)
print("GENERANDO PLOTS DE LOS MEJORES CAMPOS...")
print("=" * 70)

# Crear directorio para plots
output_dir = Path("outputs/field_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot para top 5 campos más observados
fig, axes = plt.subplots(5, 1, figsize=(14, 16))
fig.suptitle('TOP 5 CAMPOS ZTF MÁS OBSERVADOS', fontsize=16, fontweight='bold')

for idx, (oid, total) in enumerate(obs_per_oid.head(5).items()):
    ax = axes[idx]
    df_field = df[df['oid'] == oid]
    
    # Plot observaciones por filtro
    colors = {'g': 'green', 'r': 'red', 'i': 'purple'}
    for filt in ['g', 'r', 'i']:
        df_filt = df_field[df_field['filter'] == filt]
        if len(df_filt) > 0:
            ax.scatter(df_filt['mjd'], df_filt['maglimit'], 
                      c=colors[filt], label=f'{filt} ({len(df_filt)} obs)',
                      alpha=0.6, s=30, marker='o')
    
    ax.invert_yaxis()
    ax.set_ylabel('Limiting Mag', fontsize=10, fontweight='bold')
    ax.set_title(f"#{idx+1}: {oid} ({total} obs total)", 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if idx == 4:
        ax.set_xlabel('MJD', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_path_top5 = output_dir / "top5_most_observed_fields.png"
plt.savefig(plot_path_top5, dpi=150, bbox_inches='tight')
print(f"\n✓ Guardado: {plot_path_top5}")

# Plot para top 5 campos balanceados (multi-banda)
if len(balanced_fields_sorted) >= 5:
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))
    fig.suptitle('TOP 5 CAMPOS CON MEJOR BALANCE MULTI-BANDA (g+r+i)', 
                fontsize=16, fontweight='bold')
    
    for idx, field_info in enumerate(balanced_fields_sorted[:5]):
        ax = axes[idx]
        oid = field_info['oid']
        df_field = df[df['oid'] == oid]
        
        # Plot observaciones por filtro
        colors = {'g': 'green', 'r': 'red', 'i': 'purple'}
        for filt in ['g', 'r', 'i']:
            df_filt = df_field[df_field['filter'] == filt]
            if len(df_filt) > 0:
                ax.scatter(df_filt['mjd'], df_filt['maglimit'], 
                          c=colors[filt], label=f'{filt} ({len(df_filt)} obs)',
                          alpha=0.6, s=30, marker='o')
        
        ax.invert_yaxis()
        ax.set_ylabel('Limiting Mag', fontsize=10, fontweight='bold')
        title = (f"#{idx+1}: {oid} ({field_info['total']} obs) | "
                f"Balance={field_info['balance']:.3f}")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if idx == 4:
            ax.set_xlabel('MJD', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path_balanced = output_dir / "top5_balanced_multiband_fields.png"
    plt.savefig(plot_path_balanced, dpi=150, bbox_inches='tight')
    print(f"✓ Guardado: {plot_path_balanced}")

# Plot comparativo: Histograma de observaciones por filtro
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ANÁLISIS COMPARATIVO: MEJOR CAMPO vs MEJOR BALANCEADO', 
            fontsize=14, fontweight='bold')

# Campo más observado
oid_most = best_oid
df_most = df[df['oid'] == oid_most]

ax = axes[0, 0]
for filt, color in [('g', 'green'), ('r', 'red'), ('i', 'purple')]:
    df_filt = df_most[df_most['filter'] == filt]
    if len(df_filt) > 0:
        ax.scatter(df_filt['mjd'], df_filt['maglimit'], 
                  c=color, label=f'{filt} ({len(df_filt)})',
                  alpha=0.6, s=25)
ax.invert_yaxis()
ax.set_title(f'Más Observado: {oid_most}', fontsize=11, fontweight='bold')
ax.set_xlabel('MJD', fontsize=10)
ax.set_ylabel('Limiting Mag', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Mejor balanceado
if balanced_fields_sorted:
    oid_bal = balanced_fields_sorted[0]['oid']
    df_bal = df[df['oid'] == oid_bal]
    
    ax = axes[0, 1]
    for filt, color in [('g', 'green'), ('r', 'red'), ('i', 'purple')]:
        df_filt = df_bal[df_bal['filter'] == filt]
        if len(df_filt) > 0:
            ax.scatter(df_filt['mjd'], df_filt['maglimit'], 
                      c=color, label=f'{filt} ({len(df_filt)})',
                      alpha=0.6, s=25)
    ax.invert_yaxis()
    ax.set_title(f'Mejor Balance: {oid_bal}', fontsize=11, fontweight='bold')
    ax.set_xlabel('MJD', fontsize=10)
    ax.set_ylabel('Limiting Mag', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Histogramas de distribución temporal
ax = axes[1, 0]
for filt, color in [('g', 'green'), ('r', 'red'), ('i', 'purple')]:
    df_filt = df_most[df_most['filter'] == filt]
    if len(df_filt) > 0:
        ax.hist(df_filt['mjd'], bins=30, alpha=0.5, color=color, 
               label=f'{filt}', edgecolor='black', linewidth=0.5)
ax.set_xlabel('MJD', fontsize=10)
ax.set_ylabel('Observaciones', fontsize=10)
ax.set_title('Distribución Temporal (Más Observado)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

if balanced_fields_sorted:
    ax = axes[1, 1]
    for filt, color in [('g', 'green'), ('r', 'red'), ('i', 'purple')]:
        df_filt = df_bal[df_bal['filter'] == filt]
        if len(df_filt) > 0:
            ax.hist(df_filt['mjd'], bins=30, alpha=0.5, color=color, 
                   label=f'{filt}', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('MJD', fontsize=10)
    ax.set_ylabel('Observaciones', fontsize=10)
    ax.set_title('Distribución Temporal (Mejor Balance)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path_comparison = output_dir / "field_comparison.png"
plt.savefig(plot_path_comparison, dpi=150, bbox_inches='tight')
print(f"✓ Guardado: {plot_path_comparison}")

print("\n" + "=" * 70)
print(f"Plots guardados en: {output_dir.absolute()}")
print("=" * 70)

