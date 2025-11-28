"""
Workflow: Predicción de anomalías usando el mejor clasificador entrenado.
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
    output_dir: Path,
    best_method: str
):
    """Crea reporte visual de predicción."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gráfica 1: Anomaly scores vs threshold
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    
    anomaly_scores = detector_result['anomaly_scores']
    x = np.arange(len(anomaly_scores))
    
    ax.scatter(
        x,
        anomaly_scores,
        alpha=0.6,
        s=30,
        color=COLORS['secondary'],
        label='Anomaly Score'
    )
    ax.axhline(
        y=0.1,
        color=COLORS['warning'],
        linestyle='--',
        linewidth=2,
        label='Umbral anomalía'
    )
    
    ax.set_title(
        f'Anomaly Scores vs Umbral (Clasificador: {best_method.upper()})',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel('Muestra', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_anomaly_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfica 2: Clasificación por sensor
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    sensors = detector_result['sensors']
    if len(detector_result['residuals'].shape) == 2:
        sensor_scores = np.abs(detector_result['residuals']).mean(axis=0)
    else:
        sensor_scores = np.array([np.abs(detector_result['residuals']).mean()])
    
    colors_sensors = [COLORS['accent'] if s > 0.1 else COLORS['primary'] for s in sensor_scores]
    bars = ax.bar(
        range(len(sensors)), sensor_scores,
        color=colors_sensors, alpha=0.8, edgecolor='black'
    )
    
    ax.set_xlabel('Sensor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Absolute Residual', fontsize=11, fontweight='bold')
    ax.set_title('Residuos por Sensor', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(sensors)))
    ax.set_xticklabels(sensors)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, sensor_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_residuals_by_sensor.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Gráficas de predicción generadas (2 gráficas)")


def main():
    print("\n" + "="*70)
    print("🔍 PREDICCIÓN DE ANOMALÍAS EN NUEVOS DATOS")
    print("="*70 + "\n")
    
    project_models_dir = settings.ROOT_DIR / "models" / "trained"
    
    # === Cargar modelo de residuos ===
    model_path = sorted(project_models_dir.glob("residuals_*.pkl"))[-1]
    
    # === Cargar MEJOR clasificador ===
    best_classifier_path = project_models_dir / "classifier_best.pkl"
    
    if not best_classifier_path.exists():
        print("⚠️ classifier_best.pkl no encontrado")
        print("   Disponibles:")
        for f in project_models_dir.glob("classifier_*.pkl"):
            print(f"   - {f.name}")
        print("\n   Usando GMM por defecto\n")
        best_classifier_path = project_models_dir / "classifier_gmm.pkl"
    
    # === Cargar metadata ===
    metadata_path = project_models_dir / "best_classifier_metadata.json"
    metadata = {}
    best_method = "desconocido"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        best_method = metadata['best_method']
        
        print("📊 CONFIGURACIÓN DEL CLASIFICADOR")
        print(f"{'='*70}")
        print(f"Método: {best_method.upper()}")
        print(f"Test Accuracy: {metadata['test_accuracy']:.4f}")
        print(f"Test AUC-ROC: {metadata['test_auc']:.4f}")
        print(f"Overfit Gap: {metadata['overfit_gap']:.4f}")
        print(f"{'='*70}\n")
    
    print("📦 Cargando componentes...")
    print(f"  Modelo residuos: {model_path.name}")
    print(f"  Clasificador: {best_classifier_path.name}\n")
    
    detector = AnomalyDetector.load(str(model_path), str(best_classifier_path))
    
    # === Cargar datos de prueba ===
    # Opción 1: Un archivo específico
    test_file = settings.DATA_DIR / "processed" / "imbalance" / "12_paradaDesblnCSLCSPCTP.csv"
    
    # Opción 2: Si no existe, buscar cualquier archivo
    if not test_file.exists():
        imbalance_files = list((settings.DATA_DIR / "processed" / "imbalance").glob("*.csv"))
        if imbalance_files:
            test_file = imbalance_files[0]
        else:
            print("❌ No se encontraron archivos de prueba")
            return
    
    print(f"📥 Cargando datos de prueba: {test_file.name}\n")
    
    try:
        df_test = pd.read_csv(test_file)
        df_test['Fecha'] = pd.to_datetime(df_test['Fecha'], errors='coerce')
    except Exception as e:
        print(f"❌ Error cargando archivo: {e}")
        return
    
    # === Predicción ===
    print("🔮 Realizando predicción...\n")
    
    try:
        result = detector.predict(df_test)
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return
    
    # === Mostrar resultados ===
    print("="*70)
    print("📊 RESULTADOS DE PREDICCIÓN")
    print("="*70 + "\n")
    
    print(f"Archivo: {test_file.name}")
    print(f"Muestras: {len(df_test):,}")
    print(f"Sensores: {result['sensors']}\n")
    
    print(f"Clasificación Global: {result['classification']}")
    print(f"P(Desbalanceo): {result['p_desbalanceo_global']:.4f}")
    print(f"P(Desalineación): {result['p_desalineacion_global']:.4f}")
    print(f"Anomalías detectadas: {result['n_anomalies']:,}\n")
    
    # === Guardar reporte ===
    output_dir = project_models_dir.parent / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_prediction_report(result, output_dir, best_method)
    
    # === Exportar resultados JSON ===
    result_json = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'file': test_file.name,
        'classifier_method': best_method,
        'classifier_test_accuracy': float(metadata.get('test_accuracy', 0)),
        'classifier_test_auc': float(metadata.get('test_auc', 0)),
        'classification': result['classification'],
        'p_desbalanceo': float(result['p_desbalanceo_global']),
        'p_desalineacion': float(result['p_desalineacion_global']),
        'n_anomalies': result['n_anomalies'],
        'sensors': result['sensors'],
        'n_samples': len(df_test),
        'residuals_mean': float(np.mean(result['anomaly_scores'])),
        'residuals_max': float(np.max(result['anomaly_scores'])),
        'residuals_std': float(np.std(result['anomaly_scores']))
    }
    
    result_json_path = output_dir / "latest_prediction.json"
    with open(result_json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    
    print("="*70)
    print("✅ PREDICCIÓN COMPLETADA")
    print("="*70 + "\n")
    
    print("📁 Resultados guardados en:")
    print(f"   {output_dir}\n")
    
    print("   - 01_probabilities.png (Probabilidades por muestra)")
    print("   - 02_anomaly_scores.png (Anomaly scores)")
    print("   - 03_residuals_by_sensor.png (Residuos por sensor)")
    print("   - latest_prediction.json (Resultados numéricos)\n")


if __name__ == "__main__":
    main()
