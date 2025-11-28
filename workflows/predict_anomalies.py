"""
Workflow: Predicción en nuevos datos.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import settings
from src.models.anomaly_detector import AnomalyDetector
from src.visualization.plots import COLORS


def create_prediction_report(
    detector_result: dict,
    output_dir: Path
):
    """Crea reporte visual de predicción."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gráfica 1: Probabilidades por muestra
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    
    x = np.arange(len(detector_result['p_desbalanceo_por_muestra']))
    ax.fill_between(
        x,
        0,
        detector_result['p_desbalanceo_por_muestra'],
        alpha=0.6,
        color=COLORS['primary'],
        label='P(Desbalanceo)'
    )
    ax.fill_between(
        x,
        detector_result['p_desbalanceo_por_muestra'],
        1,
        alpha=0.6,
        color=COLORS['accent'],
        label='P(Desalineación)'
    )
    
    ax.set_title(
        'Probabilidades de Anomalía por Muestra',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel('Muestra', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probabilidad', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "probabilities.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfica 2: Anomaly scores vs threshold
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    
    ax.scatter(
        x,
        detector_result['anomaly_scores'],
        alpha=0.6,
        s=30,
        color=COLORS['secondary']
    )
    ax.axhline(
        y=0.1,  # Threshold ejemplo
        color=COLORS['warning'],
        linestyle='--',
        linewidth=2,
        label='Umbral anomalía'
    )
    
    ax.set_title(
        'Anomaly Scores vs Umbral',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel('Muestra', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_scores.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("\n" + "="*70)
    print("🔍 PREDICCIÓN DE ANOMALÍAS EN NUEVOS DATOS")
    print("="*70 + "\n")
    
    # === Cargar detector entrenado ===
    project_models_dir = settings.ROOT_DIR / "models" / "trained"
    
    model_path = sorted(project_models_dir.glob("residuals_*.pkl"))[-1]
    classifier_path = sorted(project_models_dir.glob("classifier_*.pkl"))[-1]
    
    print("📦 Cargando componentes...")
    print(f"  Modelo: {model_path.name}")
    print(f"  Clasificador: {classifier_path.name}\n")
    
    detector = AnomalyDetector.load(str(model_path), str(classifier_path))
    
    # === Cargar datos de prueba ===
    test_file = settings.DATA_DIR / "processed" / "imbalance" / "12_paradaDesblnCSLCSPCTP.csv"
    
    print(f"📥 Cargando datos de prueba: {test_file.name}\n")
    
    df_test = pd.read_csv(test_file)
    df_test['Fecha'] = pd.to_datetime(df_test['Fecha'], errors='coerce')
    
    # === Predicción ===
    print("🔮 Realizando predicción...\n")
    
    result = detector.predict(df_test)
    
    # === Mostrar resultados ===
    print("="*70)
    print("📊 RESULTADOS DE PREDICCIÓN")
    print("="*70 + "\n")
    
    print(f"Clasificación: {result['classification']}")
    print(f"P(Desbalanceo): {result['p_desbalanceo_global']:.4f}")
    print(f"P(Desalineación): {result['p_desalineacion_global']:.4f}")
    print(f"Anomalías detectadas: {result['n_anomalies']}\n")
    
    # === Guardar reporte ===
    output_dir = project_models_dir.parent / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_prediction_report(result, output_dir)
    
    # Exportar JSON
    result_json = {
        'classification': result['classification'],
        'p_desbalanceo': float(result['p_desbalanceo_global']),
        'p_desalineacion': float(result['p_desalineacion_global']),
        'n_anomalies': result['n_anomalies'],
        'sensors': result['sensors']
    }
    
    with open(output_dir / "prediction_result.json", 'w') as f:
        json.dump(result_json, f, indent=2)
    
    print(f"✓ Reporte guardado: {output_dir}\n")


if __name__ == "__main__":
    main()
