from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# ============================================================================
# CONFIGURACIÓN PROFESIONAL ESTILO IEEE/NATURE
# ============================================================================

# Usar Computer Modern (LaTeX font) - la más elegante para papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',  # Computer Modern para matemáticas
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Paleta de colores elegante tipo Science/Nature
COLORS = {
    'primary': '#1f2937',      # Gris carbón (elegante)
    'secondary': '#dc2626',    # Rojo científico
    'accent': '#059669',       # Verde esmeralda
    'velocity': '#7c3aed',     # Púrpura profundo (para KPH)
    'neutral': '#6b7280',      # Gris medio
    'light': '#d1d5db',        # Gris claro
    'highlight': '#0891b2',    # Cyan profesional
    'warning': '#ea580c',      # Naranja terracota
}

# Paleta para múltiples sensores (diferenciables y elegantes)
SENSOR_PALETTE = [
    '#1f2937',  # Gris carbón
    '#dc2626',  # Rojo
    '#059669',  # Verde
    '#0891b2',  # Cyan
    '#ea580c',  # Naranja
    '#7c3aed',  # Púrpura
    '#0d9488',  # Teal
    '#4338ca',  # Índigo
]


def plot_sensor_timeseries(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Datos de Sensores",
    figsize: tuple = (14, 7)
) -> plt.Figure:
    """
    Series de tiempo con fuente elegante y colores profesionales.
    """
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    
    # Sensores en eje izquierdo
    sensor_cols = [
        col for col in df.columns 
        if col not in ['Fecha', 'KPH']
    ]
    
    for i, column in enumerate(sensor_cols):
        color = SENSOR_PALETTE[i % len(SENSOR_PALETTE)]
        ax1.plot(
            df['Fecha'], 
            df[column], 
            color=color, 
            label=column,
            linewidth=1.3, 
            alpha=0.85
        )
    
    # Configuración eje izquierdo
    ax1.set_xlabel('Tiempo', fontsize=11, fontweight='bold')
    ax1.set_ylabel(
        'Desplazamiento (μm pk-pk)', 
        fontsize=11, 
        fontweight='bold',
        color=COLORS['primary']
    )
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    
    # Formato de fechas
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(8))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Eje derecho para velocidad (púrpura profesional)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLORS['velocity'])
    ax2.spines['right'].set_linewidth(1.5)
    
    ax2.plot(
        df['Fecha'], 
        df['KPH'], 
        color=COLORS['velocity'],
        label='Velocidad', 
        linewidth=2.2, 
        alpha=0.95, 
        linestyle='-'
    )
    
    ax2.set_ylabel(
        'Velocidad (KPH)', 
        fontsize=11, 
        fontweight='bold',
        color=COLORS['velocity']
    )
    ax2.tick_params(axis='y', labelcolor=COLORS['velocity'], width=1.5)
    
    # Título
    ax1.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Leyenda elegante debajo de la gráfica
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(
        lines1 + lines2, 
        labels1 + labels2,
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(5, len(labels1) + len(labels2)),
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='#d1d5db',
        framealpha=0.98,
        columnspacing=1.5
    )
    
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path, 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
    
    return fig


def plot_before_after_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    sensor_col: str,
    output_path: Optional[Path] = None,
    title: str = "Comparación Antes/Después"
) -> plt.Figure:
    """
    Comparación elegante con estadísticas.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(14, 9), 
        sharex=True,
        dpi=100
    )
    
    # Antes (rojo elegante)
    ax1.plot(
        df_before['Fecha'], 
        df_before[sensor_col],
        color=COLORS['secondary'], 
        alpha=0.85, 
        linewidth=1.3,
        label='Original'
    )
    
    ax1.set_ylabel(
        'Desplazamiento (μm pk-pk)', 
        fontsize=11, 
        fontweight='bold'
    )
    ax1.set_title(
        '(a) Datos Originales', 
        fontsize=11, 
        fontweight='bold',
        loc='left'
    )
    
    # Estadísticas en caja elegante
    stats_text = (
        f'$N$ = {len(df_before):,}\n'
        f'$\\mu$ = {df_before[sensor_col].mean():.2f}\n'
        f'$\\sigma$ = {df_before[sensor_col].std():.2f}'
    )
    
    ax1.text(
        0.98, 0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='white',
            edgecolor='#d1d5db',
            alpha=0.95,
            linewidth=1
        )
    )
    
    # Después (verde elegante)
    ax2.plot(
        df_after['Fecha'], 
        df_after[sensor_col],
        color=COLORS['accent'], 
        alpha=0.85, 
        linewidth=1.3,
        label='Procesado'
    )
    
    ax2.set_ylabel(
        'Desplazamiento (μm pk-pk)', 
        fontsize=11,
        fontweight='bold'
    )
    ax2.set_xlabel('Tiempo', fontsize=11, fontweight='bold')
    ax2.set_title(
        '(b) Datos Procesados', 
        fontsize=11,
        fontweight='bold', 
        loc='left'
    )
    
    # Estadísticas
    reduction_pct = (
        (len(df_before) - len(df_after)) / len(df_before) * 100
    )
    
    stats_text = (
        f'$N$ = {len(df_after):,}\n'
        f'$\\mu$ = {df_after[sensor_col].mean():.2f}\n'
        f'$\\sigma$ = {df_after[sensor_col].std():.2f}\n'
        f'Reducción = {reduction_pct:.1f}%'
    )
    
    ax2.text(
        0.98, 0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='white',
            edgecolor='#d1d5db',
            alpha=0.95,
            linewidth=1
        )
    )
    
    # Formato de fechas
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(8))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Título general
    fig.suptitle(
        f'{title}\nSensor: {sensor_col}',
        fontsize=12,
        fontweight='bold',
        y=0.995
    )
    
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
    
    return fig


def plot_reduction_summary(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Barras profesionales con variable usada correctamente.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    
    success_df = summary_df[summary_df['status'] == 'success'].copy()
    success_df = success_df.sort_values('reduction_pct', ascending=False)
    
    # Colores elegantes por categoría
    colors = [
        COLORS['primary'] if cat == 'imbalance' else COLORS['warning']
        for cat in success_df['category']
    ]
    
    # Crear barras
    ax.bar(
        range(len(success_df)),
        success_df['reduction_pct'],
        color=colors,
        alpha=0.85,
        edgecolor='#6b7280',
        linewidth=0.8
    )
    
    # Configuración
    ax.set_xlabel('Archivo de Datos', fontsize=11, fontweight='bold')
    ax.set_ylabel('Reducción (%)', fontsize=11, fontweight='bold')
    ax.set_title(
        'Porcentaje de Reducción de Datos por Archivo',
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    
    ax.set_xticks(range(len(success_df)))
    ax.set_xticklabels(
        success_df['filename'],
        rotation=45,
        ha='right',
        fontsize=8
    )
    
    # Promedio
    avg_reduction = success_df['reduction_pct'].mean()
    ax.axhline(
        y=avg_reduction,
        color=COLORS['secondary'],
        linestyle='--',
        linewidth=1.8,
        alpha=0.8,
        label=f'Promedio: {avg_reduction:.1f}%'
    )
    
    # Leyenda elegante
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(
            facecolor=COLORS['primary'],
            alpha=0.85,
            label='Imbalance',
            edgecolor='#6b7280',
            linewidth=0.8
        ),
        Patch(
            facecolor=COLORS['warning'],
            alpha=0.85,
            label='Misalignment',
            edgecolor='#6b7280',
            linewidth=0.8
        ),
        plt.Line2D(
            [0], [0],
            color=COLORS['secondary'],
            linestyle='--',
            linewidth=1.8,
            label=f'Promedio: {avg_reduction:.1f}%'
        )
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='#d1d5db',
        framealpha=0.95
    )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
    
    return fig


def plot_category_comparison(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Box plots sin warnings de ticklabels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    
    success_df = summary_df[summary_df['status'] == 'success']
    
    # Paleta elegante
    palette = [COLORS['primary'], COLORS['warning']]
    
    # Reducción - FIX: usar set_xticks antes de set_xticklabels
    sns.boxplot(
        data=success_df,
        x='category',
        y='reduction_pct',
        ax=axes[0],
        palette=palette,
        hue='category',
        legend=False,
        linewidth=1.5,
        fliersize=4
    )
    
    axes[0].set_title(
        '(a) Distribución de Reducción',
        fontsize=11,
        fontweight='bold',
        loc='left'
    )
    axes[0].set_xlabel('Categoría', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Reducción (%)', fontsize=10, fontweight='bold')
    
    # FIX: establecer ticks antes de labels
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Imbalance', 'Misalignment'])
    
    # Filas procesadas - FIX igual
    sns.boxplot(
        data=success_df,
        x='category',
        y='processed_rows',
        ax=axes[1],
        palette=palette,
        hue='category',
        legend=False,
        linewidth=1.5,
        fliersize=4
    )
    
    axes[1].set_title(
        '(b) Distribución de Filas Procesadas',
        fontsize=11,
        fontweight='bold',
        loc='left'
    )
    axes[1].set_xlabel('Categoría', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Filas Procesadas', fontsize=10, fontweight='bold')
    
    # FIX: establecer ticks antes de labels
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Imbalance', 'Misalignment'])
    
    # Spines
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle(
        'Comparación Estadística: Imbalance vs Misalignment',
        fontsize=12,
        fontweight='bold',
        y=0.98
    )
    
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
    
    return fig


def plot_all_sensors_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Comparación Multi-Sensor"
) -> plt.Figure:
    """
    Todos los sensores en subplots elegantes.
    """
    sensor_cols = [
        col for col in df_before.columns 
        if col not in ['Fecha', 'KPH']
    ]
    n_sensors = len(sensor_cols)
    
    n_cols = 2
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 3.5*n_rows),
        sharex=True,
        dpi=100
    )
    
    if n_sensors == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, sensor_col in enumerate(sensor_cols):
        ax = axes[i]
        
        # Original (rojo elegante)
        ax.plot(
            df_before['Fecha'],
            df_before[sensor_col],
            color=COLORS['secondary'],
            alpha=0.75,
            linewidth=1.1,
            label='Original',
            zorder=1
        )
        
        # Procesado (verde elegante)
        ax.plot(
            df_after['Fecha'],
            df_after[sensor_col],
            color=COLORS['accent'],
            alpha=0.95,
            linewidth=1.4,
            label='Procesado',
            zorder=2
        )
        
        # Título con letra
        letter = chr(97 + i)
        ax.set_title(
            f'({letter}) {sensor_col}',
            fontsize=10,
            fontweight='bold',
            loc='left'
        )
        ax.set_ylabel(
            'Desplazamiento\n(μm pk-pk)',
            fontsize=9,
            fontweight='bold'
        )
        
        # Leyenda compacta
        ax.legend(
            loc='upper right',
            fontsize=8,
            frameon=True,
            fancybox=False,
            edgecolor='#d1d5db',
            framealpha=0.95
        )
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Ocultar vacíos
    for i in range(n_sensors, len(axes)):
        axes[i].axis('off')
    
    # Formato fechas última fila
    for idx in range(max(0, len(axes) - n_cols), len(axes)):
        if idx < n_sensors:
            axes[idx].xaxis.set_major_formatter(
                mdates.DateFormatter('%Y-%m-%d\n%H:%M')
            )
            axes[idx].xaxis.set_major_locator(ticker.MaxNLocator(5))
            axes[idx].set_xlabel('Tiempo', fontsize=9, fontweight='bold')
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
    
    return fig
