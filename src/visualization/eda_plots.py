from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Usar configuración profesional existente
from src.visualization.plots import COLORS, SENSOR_PALETTE


def plot_distribution_analysis(
    df: pd.DataFrame,
    sensor_cols: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Análisis de distribuciones univariables con histogramas y KDE.
    """
    n_sensors = len(sensor_cols)
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15, 4*n_rows),
        dpi=100
    )
    
    if n_sensors == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, sensor in enumerate(sensor_cols):
        ax = axes[i]
        data = df[sensor].dropna()
        
        # Histograma + KDE
        ax.hist(
            data,
            bins=50,
            density=True,
            alpha=0.6,
            color=SENSOR_PALETTE[i % len(SENSOR_PALETTE)],
            edgecolor='black',
            linewidth=0.5
        )
        
        # KDE
        try:
            data.plot.kde(
                ax=ax,
                color=COLORS['secondary'],
                linewidth=2,
                label='KDE'
            )
        except Exception:
            pass
        
        # Línea de media
        mean_val = data.mean()
        ax.axvline(
            mean_val,
            color=COLORS['primary'],
            linestyle='--',
            linewidth=1.5,
            label=f'Media: {mean_val:.2f}'
        )
        
        # Línea de mediana
        median_val = data.median()
        ax.axvline(
            median_val,
            color=COLORS['accent'],
            linestyle=':',
            linewidth=1.5,
            label=f'Mediana: {median_val:.2f}'
        )
        
        # Configuración
        letter = chr(97 + i)
        ax.set_title(
            f'({letter}) {sensor}',
            fontsize=11,
            fontweight='bold',
            loc='left'
        )
        ax.set_xlabel('Valor (μm pk-pk)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Densidad', fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Ocultar subplots vacíos
    for i in range(n_sensors, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(
        'Análisis de Distribuciones Univariables',
        fontsize=14,
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
            facecolor='white'
        )
        plt.close(fig)
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    sensor_cols: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Matriz de correlación profesional.
    """
    # Calcular correlación
    corr_data = df[sensor_cols + ['KPH_abs']].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Heatmap
    mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
    
    sns.heatmap(
        corr_data,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={
            'shrink': 0.8,
            'label': 'Coeficiente de Correlación'
        },
        ax=ax,
        vmin=-1,
        vmax=1
    )
    
    ax.set_title(
        'Matriz de Correlación de Pearson',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close(fig)
    
    return fig


def plot_boxplots_outliers(
    df: pd.DataFrame,
    sensor_cols: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Box plots para detección de outliers.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Preparar datos para boxplot
    data_to_plot = [df[col].dropna() for col in sensor_cols]
    
    bp = ax.boxplot(
        data_to_plot,
        labels=sensor_cols,
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops=dict(
            marker='D',
            markerfacecolor=COLORS['secondary'],
            markersize=6
        ),
        flierprops=dict(
            marker='o',
            markerfacecolor=COLORS['warning'],
            markersize=4,
            alpha=0.5
        )
    )
    
    # Colorear cajas
    for patch, color in zip(bp['boxes'], SENSOR_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(
        'Análisis de Dispersión y Detección de Outliers',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Sensor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Desplazamiento (μm pk-pk)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close(fig)
    
    return fig


def plot_velocity_analysis(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Análisis detallado de velocidad (con signo y absoluta).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    
    # (a) Serie temporal con signo
    axes[0, 0].plot(
        df['Fecha'],
        df['KPH'],
        color=COLORS['velocity'],
        linewidth=1,
        alpha=0.8
    )
    axes[0, 0].axhline(
        y=0,
        color='black',
        linestyle='--',
        linewidth=1,
        alpha=0.5
    )
    axes[0, 0].set_title(
        '(a) Velocidad con Signo',
        fontsize=11,
        fontweight='bold',
        loc='left'
    )
    axes[0, 0].set_ylabel('Velocidad (KPH)', fontsize=10, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, linestyle=':')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    
    # (b) Histograma con signo
    axes[0, 1].hist(
        df['KPH'],
        bins=50,
        color=COLORS['velocity'],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    axes[0, 1].axvline(
        df['KPH'].mean(),
        color=COLORS['secondary'],
        linestyle='--',
        linewidth=2,
        label=f"Media: {df['KPH'].mean():.2f}"
    )
    axes[0, 1].set_title(
        '(b) Distribución con Signo',
        fontsize=11,
        fontweight='bold',
        loc='left'
    )
    axes[0, 1].set_xlabel('Velocidad (KPH)', fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    
    # (c) Serie temporal absoluta
    axes[1, 0].plot(
        df['Fecha'],
        df['KPH_abs'],
        color=COLORS['accent'],
        linewidth=1,
        alpha=0.8
    )
    axes[1, 0].set_title(
        '(c) Velocidad Absoluta',
        fontsize=11,
        fontweight='bold',
        loc='left'
    )
    axes[1, 0].set_ylabel('|Velocidad| (KPH)', fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel('Tiempo', fontsize=10, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, linestyle=':')
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    
    # (d) Histograma absoluto
    axes[1, 1].hist(
        df['KPH_abs'],
        bins=50,
        color=COLORS['accent'],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    axes[1, 1].axvline(
        df['KPH_abs'].mean(),
        color=COLORS['secondary'],
        linestyle='--',
        linewidth=2,
        label=f"Media: {df['KPH_abs'].mean():.2f}"
    )
    axes[1, 1].set_title(
        '(d) Distribución Absoluta',
        fontsize=11,
        fontweight='bold',
        loc='left'
    )
    axes[1, 1].set_xlabel('|Velocidad| (KPH)', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    
    fig.suptitle(
        'Análisis Completo de Velocidad',
        fontsize=14,
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
            facecolor='white'
        )
        plt.close(fig)
    
    return fig
