"""
Visualizaciones con Plotly para Streamlit.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_residuals_by_sensor(residuals: np.ndarray, sensors: List[str]) -> plt.Figure:
    """Gráfica: Residuos por sensor."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    mean_residuals = np.abs(residuals).mean(axis=0)
    colors = ['#059669' if r < 0.5 else '#D97706' if r < 1.0 else '#DC2626' for r in mean_residuals]
    
    ax.bar(sensors, mean_residuals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Sensor', fontweight='bold')
    ax.set_ylabel('Residuo Medio Absoluto', fontweight='bold')
    ax.set_title('Residuos por Sensor', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_anomaly_timeline(anomaly_scores: np.ndarray) -> plt.Figure:
    """Gráfica: Timeline de anomalías."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(anomaly_scores, linewidth=1.5, color='#2563EB', label='Anomaly Score')
    ax.axhline(y=0.1, color='#DC2626', linestyle='--', linewidth=2, label='Umbral')
    ax.fill_between(range(len(anomaly_scores)), 0, anomaly_scores, alpha=0.3, color='#2563EB')
    
    ax.set_xlabel('Muestra', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Timeline de Anomalías', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_feature_distribution(residuals: np.ndarray) -> plt.Figure:
    """Gráfica: Distribución de residuos."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    flat_residuals = np.abs(residuals).flatten()
    ax.hist(flat_residuals, bins=50, color='#2563EB', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(flat_residuals), color='#DC2626', linestyle='--', linewidth=2, label='Media')
    
    ax.set_xlabel('Valor Absoluto del Residuo', fontweight='bold')
    ax.set_ylabel('Frecuencia', fontweight='bold')
    ax.set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig
