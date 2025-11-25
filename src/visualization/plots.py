from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# Configuración global de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_sensor_timeseries(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Datos de Sensores",
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Grafica series de tiempo de sensores de desplazamiento y velocidad.
    
    Args:
        df: DataFrame con columnas 'Fecha', 'KPH' y sensores
        output_path: Ruta para guardar la imagen (opcional)
        title: Título de la gráfica
        figsize: Tamaño de la figura
    
    Returns:
        Figure de matplotlib
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Configurar eje X para fechas
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.xticks(rotation=45, ha='right')
    ax1.set_xlabel('Fecha', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Desplazamientos (μm pk-pk)', color='tab:red', fontsize=12, fontweight='bold')
    
    # Seleccionar columnas de sensores
    sensor_cols = [col for col in df.columns if col not in ['Fecha', 'KPH']]
    colors = plt.cm.tab10(range(len(sensor_cols)))
    
    # Graficar sensores
    for i, column in enumerate(sensor_cols):
        ax1.plot(df['Fecha'], df[column], color=colors[i], label=column, linewidth=1.5, alpha=0.8)
    
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=10)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Eje derecho para velocidad
    ax2 = ax1.twinx()
    ax2.set_ylabel('Velocidad (KPH)', color='tab:blue', fontsize=12, fontweight='bold')
    ax2.plot(
        df['Fecha'],
        df['KPH'],
        color='tab:blue',
        label='Velocidad (KPH)',
        linewidth=2,
        alpha=0.9
    )
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=10)
    
    # Título y leyenda
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), fontsize=9, framealpha=0.9)
    
    fig.tight_layout()
    
    # Guardar si se especifica ruta
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_before_after_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    sensor_col: str,
    output_path: Optional[Path] = None,
    title: str = "Comparación Antes/Después del Preprocesamiento"
) -> plt.Figure:
    """
    Compara datos antes y después del preprocesamiento para un sensor específico.
    
    Args:
        df_before: DataFrame original
        df_after: DataFrame procesado
        sensor_col: Nombre de la columna del sensor a comparar
        output_path: Ruta para guardar
        title: Título de la gráfica
    
    Returns:
        Figure de matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Antes del preprocesamiento
    ax1.plot(df_before['Fecha'], df_before[sensor_col], color='tab:red', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Desplazamiento (μm pk-pk)', fontsize=11, fontweight='bold')
    ax1.set_title('Antes del Preprocesamiento', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.text(
        0.02,
        0.95,
        f'Filas: {len(df_before)}',
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.5
        )
    )
    
    # Después del preprocesamiento
    ax2.plot(df_after['Fecha'], df_after[sensor_col], color='tab:green', alpha=0.7, linewidth=1)
    ax2.set_ylabel('Desplazamiento (μm pk-pk)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Fecha', fontsize=11, fontweight='bold')
    ax2.set_title('Después del Preprocesamiento', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    reduction_pct = ((len(df_before) - len(df_after)) / len(df_before) * 100)
    ax2.text(
        0.02,
        0.95,
        f'Filas: {len(df_after)} (Reducción: {reduction_pct:.1f}%)',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightgreen',
            alpha=0.5
        )
    )
    
    # Formato de fechas
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.xticks(rotation=45, ha='right')
    
    fig.suptitle(f'{title} - {sensor_col}', fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_reduction_summary(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Gráfica de barras con porcentaje de reducción por archivo.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filtrar solo archivos exitosos
    success_df = summary_df[summary_df['status'] == 'success'].copy()
    success_df = success_df.sort_values('reduction_pct', ascending=False)
    
    colors = ['tab:blue' if cat == 'imbalance' else 'tab:orange' 
              for cat in success_df['category']]
    
    ax.bar(range(len(success_df)), success_df['reduction_pct'], color=colors, alpha=0.7)
    
    ax.set_xlabel('Archivo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reducción (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Porcentaje de Reducción de Datos por Archivo',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xticks(range(len(success_df)))
    ax.set_xticklabels(success_df['filename'], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', alpha=0.7, label='Imbalance'),
        Patch(facecolor='tab:orange', alpha=0.7, label='Misalignment')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Línea de promedio
    avg_reduction = success_df['reduction_pct'].mean()
    ax.axhline(y=avg_reduction, color='red', linestyle='--', linewidth=2, 
               label=f'Promedio: {avg_reduction:.1f}%')
    
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_category_comparison(
    summary_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Box plots comparando Imbalance vs Misalignment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    success_df = summary_df[summary_df['status'] == 'success']
    
    # Reducción por categoría
    sns.boxplot(data=success_df, x='category', y='reduction_pct', ax=axes[0], palette='Set2')
    axes[0].set_title('Distribución de Reducción por Categoría', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Categoría', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Reducción (%)', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Filas procesadas por categoría
    sns.boxplot(data=success_df, x='category', y='processed_rows', ax=axes[1], palette='Set2')
    axes[1].set_title(
        'Distribución de Filas Procesadas por Categoría',
        fontsize=12,
        fontweight='bold'
    )
    axes[1].set_xlabel('Categoría', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Filas Procesadas', fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    fig.suptitle('Comparación: Imbalance vs Misalignment', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_all_sensors_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Comparación Completa Antes/Después"
) -> plt.Figure:
    """
    Compara TODOS los sensores antes y después en una sola figura.
    """
    sensor_cols = [col for col in df_before.columns if col not in ['Fecha', 'KPH']]
    n_sensors = len(sensor_cols)
    
    # Calcular layout de subplots
    n_cols = 2
    n_rows = (n_sensors + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows), sharex=True)
    axes = axes.flatten() if n_sensors > 1 else [axes]
    
    for i, sensor_col in enumerate(sensor_cols):
        ax = axes[i]
        
        # Antes (rojo)
        ax.plot(df_before['Fecha'], df_before[sensor_col], 
                color='tab:red', alpha=0.6, linewidth=1, label='Antes')
        
        # Después (verde)
        ax.plot(df_after['Fecha'], df_after[sensor_col], 
                color='tab:green', alpha=0.8, linewidth=1.5, label='Después')
        
        ax.set_title(sensor_col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Desplazamiento (μm pk-pk)', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
    
    # Ocultar subplots vacíos
    for i in range(n_sensors, len(axes)):
        axes[i].axis('off')
    
    # Formato de fechas solo en la fila inferior
    for ax in axes[-n_cols:]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig

