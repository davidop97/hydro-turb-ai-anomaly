"""
Workflow COMPLETO: Selección → Entrenamiento → Validación Train/Test + K-Fold → Visualización.
"""

import json
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from configs.config import settings
from src.models.residuals_model import DataResidualsProcessor
from src.models.sensor_selector import SensorSelector
from src.visualization.plots import COLORS


def create_validation_plots(
    model,
    df_train,
    df_test,
    best_sensor,
    output_dir: Path
):
    """
    Crea gráficas de validación train/test.
    """
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === GRÁFICA 1: Predicciones vs Reales (Train) ===
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Obtener predicciones
    residuals_train, _, _, originals_train, preds_train = (
        model.calculate_residuals_global(df=df_train)
    )
    
    x = np.arange(len(originals_train[:500]))  # Primeras 500
    
    ax.plot(x, originals_train[:500, 0], color=COLORS['secondary'],
            linewidth=1.5, label='Real', alpha=0.8)
    ax.plot(x, preds_train[:500, 0], color=COLORS['accent'],
            linewidth=1.5, label='Predicción', alpha=0.8, linestyle='--')
    
    ax.set_title('Train: Real vs Predicción (primeras 500 muestras)',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Muestra', fontsize=10, fontweight='bold')
    ax.set_ylabel('Desplazamiento (μm pk-pk)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_train_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # === GRÁFICA 2: Predicciones vs Reales (Test) ===
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    residuals_test, _, _, originals_test, preds_test = (
        model.calculate_residuals_global(df=df_test)
    )
    
    x = np.arange(len(originals_test[:500]))
    
    ax.plot(x, originals_test[:500, 0], color=COLORS['secondary'],
            linewidth=1.5, label='Real', alpha=0.8)
    ax.plot(x, preds_test[:500, 0], color=COLORS['accent'],
            linewidth=1.5, label='Predicción', alpha=0.8, linestyle='--')
    
    ax.set_title('Test: Real vs Predicción (primeras 500 muestras)',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Muestra', fontsize=10, fontweight='bold')
    ax.set_ylabel('Desplazamiento (μm pk-pk)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_test_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # === GRÁFICA 3: Residuos (Train vs Test) ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    # Train residuos
    residuals_flat_train = np.abs(residuals_train).flatten()
    ax1.hist(residuals_flat_train, bins=50, color=COLORS['primary'],
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(model.threshold, color=COLORS['secondary'], linestyle='--',
               linewidth=2, label=f'Umbral: {model.threshold:.4f}')
    ax1.set_title('Train: Distribución de Residuos Absolutos',
                 fontsize=11, fontweight='bold', loc='left')
    ax1.set_xlabel('|Residuo|', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Test residuos
    residuals_flat_test = np.abs(residuals_test).flatten()
    ax2.hist(residuals_flat_test, bins=50, color=COLORS['accent'],
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(model.threshold, color=COLORS['secondary'], linestyle='--',
               linewidth=2, label=f'Umbral: {model.threshold:.4f}')
    ax2.set_title('Test: Distribución de Residuos Absolutos',
                 fontsize=11, fontweight='bold', loc='left')
    ax2.set_xlabel('|Residuo|', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_residuals_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # === GRÁFICA 4: Scatter Real vs Predicción ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    # Train scatter
    ax1.scatter(originals_train.flatten(), preds_train.flatten(),
               alpha=0.5, s=10, color=COLORS['primary'])
    min_val = min(originals_train.min(), preds_train.min())
    max_val = max(originals_train.max(), preds_train.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Ajuste perfecto')
    ax1.set_title('Train: Real vs Predicción',
                 fontsize=11, fontweight='bold', loc='left')
    ax1.set_xlabel('Real', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Predicción', fontsize=10, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Test scatter
    ax2.scatter(originals_test.flatten(), preds_test.flatten(),
               alpha=0.5, s=10, color=COLORS['accent'])
    min_val = min(originals_test.min(), preds_test.min())
    max_val = max(originals_test.max(), preds_test.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Ajuste perfecto')
    ax2.set_title('Test: Real vs Predicción',
                 fontsize=11, fontweight='bold', loc='left')
    ax2.set_xlabel('Real', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Predicción', fontsize=10, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_scatter_real_vs_pred.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Gráficas de validación guardadas")


def create_metrics_report(
    metrics_train,
    metrics_test,
    metrics_cv,
    best_sensor,
    best_file,
    output_path: Path
):
    """
    Crea reporte HTML con métricas detalladas.
    """
    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Métricas Modelo</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #1f2937;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #667eea;
            margin-top: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table thead {{
            background-color: #667eea;
            color: white;
        }}
        table th, table td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #e5e7eb;
        }}
        table tbody tr:nth-child(odd) {{
            background-color: #f9fafb;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .comparison-box {{
            border: 2px solid #e5e7eb;
            padding: 15px;
            border-radius: 8px;
        }}
        .comparison-box h3 {{
            margin-top: 0;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Reporte de Métricas - Modelo de Residuos</h1>
        
        <h2>🔍 Información del Modelo</h2>
        <table>
            <tr>
                <td><strong>Sensor Seleccionado</strong></td>
                <td>{best_sensor}</td>
            </tr>
            <tr>
                <td><strong>Archivo Base</strong></td>
                <td>{best_file}</td>
            </tr>
            <tr>
                <td><strong>Grado Polinomio</strong></td>
                <td>3</td>
            </tr>
            <tr>
                <td><strong>Método Selección</strong></td>
                <td>5-Fold CV (Per-File)</td>
            </tr>
        </table>
        
        <h2>📈 Métricas TRAIN</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">MAE</div>
                <div class="value">{metrics_train['mae']:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="label">RMSE</div>
                <div class="value">{metrics_train['rmse']:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="label">R²</div>
                <div class="value">{metrics_train['r2']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Muestras</div>
                <div class="value">{metrics_train['n_samples']}</div>
            </div>
        </div>
        
        <h2>📈 Métricas TEST</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">MAE</div>
                <div class="value">{metrics_test['mae']:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="label">RMSE</div>
                <div class="value">{metrics_test['rmse']:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="label">R²</div>
                <div class="value">{metrics_test['r2']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Muestras</div>
                <div class="value">{metrics_test['n_samples']}</div>
            </div>
        </div>
        
        <h2>📈 Métricas K-FOLD CV (5-Folds)</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">MAE Promedio</div>
                <div class="value">{metrics_cv['mean_mae']:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="label">RMSE Promedio</div>
                <div class="value">{metrics_cv['mean_rmse']:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="label">R² Promedio</div>
                <div class="value">{metrics_cv['mean_r2']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Desv. Est. MAE</div>
                <div class="value">{metrics_cv['std_mae']:.6f}</div>
            </div>
        </div>
        
        <h2>🔄 Comparación Train vs Test</h2>
        <div class="comparison">
            <div class="comparison-box">
                <h3>Train</h3>
                <p><strong>MAE:</strong> {metrics_train['mae']:.6f}</p>
                <p><strong>R²:</strong> {metrics_train['r2']:.4f}</p>
                <p><strong>Muestras:</strong> {metrics_train['n_samples']}</p>
            </div>
            <div class="comparison-box">
                <h3>Test</h3>
                <p><strong>MAE:</strong> {metrics_test['mae']:.6f}</p>
                <p><strong>R²:</strong> {metrics_test['r2']:.4f}</p>
                <p><strong>Muestras:</strong> {metrics_test['n_samples']}</p>
            </div>
        </div>
        
        <h2>💡 Interpretación</h2>
        <div style="background-color: #f0f9ff; padding: 15px; border-left: 4px solid #0284c7; border-radius: 4px;">
            <p>
                <strong>Diferencia MAE (Train-Test):</strong> 
                {abs(metrics_train['mae'] - metrics_test['mae']):.6f}
            </p>
            <p>
                <strong>Diferencia R² (Train-Test):</strong> 
                {abs(metrics_train['r2'] - metrics_test['r2']):.4f}
            </p>
            {
                '<p style="color: #dc2626;"><strong>⚠️ Posible Overfitting</strong> si Train R² >> Test R²</p>'
                if metrics_train['r2'] - metrics_test['r2'] > 0.1 else
                '<p style="color: #059669;"><strong>✓ Modelo Balanceado</strong> - Generaliza bien</p>'
            }
        </div>
        
        <p style="text-align: center; color: #6b7280; font-size: 0.9em; margin-top: 40px;">
            Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""  # noqa: E501
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    print("\n" + "="*70)
    print("🤖 ENTRENAMIENTO v3: COMPLETO CON TRAIN/TEST + GRÁFICAS")
    print("="*70 + "\n")
    
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    run_name = (
        f"model_training_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    with mlflow.start_run(run_name=run_name) as run:
        
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        # === DIRECTORIOS ===
        mlartifacts_base = settings.ROOT_DIR / "mlartifacts"
        mlflow_artifacts_dir = (
            mlartifacts_base / str(experiment_id) / run_id / "artifacts"
        )
        models_dir = mlflow_artifacts_dir / "models"
        metrics_dir = mlflow_artifacts_dir / "metrics"
        plots_dir = mlflow_artifacts_dir / "plots"
        
        # También guardar en proyecto
        project_models_dir = settings.ROOT_DIR / "models" / "trained"
        project_models_dir.mkdir(parents=True, exist_ok=True)
        
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 MLflow: {mlflow_artifacts_dir}")
        print(f"📂 Proyecto: {project_models_dir}\n")
        
        mlflow.set_tags({
            "stage": "model_training_v3_final",
            "approach": "per_file_kfold_train_test",
            "version": "3.1.0"
        })
        
        # === PASO 1: SELECCIONAR ARCHIVO + SENSOR ===
        print("📥 Evaluando archivos...\n")
        
        processed_dir = settings.DATA_DIR / "processed"
        imbalance_dir = processed_dir / "imbalance"
        
        imbalance_files = sorted(imbalance_dir.glob("*.csv"))
        
        selector = SensorSelector(
            degree=3,
            speed_col='KPH',
            n_splits=5,
            min_samples=50
        )
        
        best_file, best_sensor, selection_results = (
            selector.select_best_sensor(
                [str(f) for f in imbalance_files],
                verbose=True
            )
        )
        
        selection_path = metrics_dir / "selection_results.json"
        selection_summary = {
            'best_file': str(best_file),
            'best_sensor': best_sensor,
            'top_candidates': [
                {
                    'file': c['file_name'],
                    'sensor': c['sensor'],
                    'mae': float(c['mae']),
                    'r2': float(c['r2'])
                }
                for c in selection_results['all_candidates'][:5]
            ]
        }
        
        with open(selection_path, 'w', encoding='utf-8') as f:
            json.dump(selection_summary, f, indent=2)
        
        # === PASO 2: CARGAR Y DIVIDIR DATOS ===
        print("\n" + "="*70)
        print("PASO 2: TRAIN/TEST SPLIT")
        print("="*70 + "\n")
        
        df_full = selector.load_best_file_data()
        df_train, df_test = train_test_split(
            df_full,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        
        print(f"Total: {len(df_full):,} muestras")
        print(f"Train: {len(df_train):,} (80%)")
        print(f"Test: {len(df_test):,} (20%)\n")
        
        # === PASO 3: ENTRENAR EN TRAIN ===
        print("="*70)
        print("PASO 3: ENTRENAMIENTO EN SET TRAIN")
        print("="*70 + "\n")
        
        model = DataResidualsProcessor(degree=3, speed_col='KPH')
        model.fit(df=df_train, verbose=True)
        
        # === PASO 4: EVALUAR EN TRAIN Y TEST ===
        print("\n" + "="*70)
        print("PASO 4: EVALUACIÓN")
        print("="*70 + "\n")
        
        # Evaluar TRAIN
        print("📊 Evaluando TRAIN...\n")
        residuals_train, _, _, orig_train, pred_train = (
            model.calculate_residuals_global(df=df_train)
        )
        
        mae_train = mean_absolute_error(orig_train.flatten(), pred_train.flatten())
        rmse_train = np.sqrt(mean_squared_error(orig_train.flatten(), pred_train.flatten()))
        r2_train = r2_score(orig_train.flatten(), pred_train.flatten())
        
        print(f"Train MAE:  {mae_train:.6f}")
        print(f"Train RMSE: {rmse_train:.6f}")
        print(f"Train R²:   {r2_train:.4f}\n")
        
        # Evaluar TEST
        print("📊 Evaluando TEST...\n")
        residuals_test, _, _, orig_test, pred_test = (
            model.calculate_residuals_global(df=df_test)
        )
        
        mae_test = mean_absolute_error(orig_test.flatten(), pred_test.flatten())
        rmse_test = np.sqrt(mean_squared_error(orig_test.flatten(), pred_test.flatten()))
        r2_test = r2_score(orig_test.flatten(), pred_test.flatten())
        
        print(f"Test MAE:  {mae_test:.6f}")
        print(f"Test RMSE: {rmse_test:.6f}")
        print(f"Test R²:   {r2_test:.4f}\n")
        
        # === PASO 5: K-FOLD CV ===
        print("="*70)
        print("PASO 5: VALIDACIÓN K-FOLD CV (5 Folds)")
        print("="*70 + "\n")
        
        df_clean = df_full[[model.speed_col, best_sensor]].dropna()
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(df_clean)
        ):
            X_train_fold = df_clean.iloc[train_idx][model.speed_col].values.reshape(-1, 1)
            X_val_fold = df_clean.iloc[val_idx][model.speed_col].values.reshape(-1, 1)
            y_train_fold = df_clean.iloc[train_idx][best_sensor].values
            y_val_fold = df_clean.iloc[val_idx][best_sensor].values
            
            X_train_poly = model.poly.fit_transform(X_train_fold)
            X_val_poly = model.poly.transform(X_val_fold)
            
            m = LinearRegression()
            m.fit(X_train_poly, y_train_fold)
            
            y_pred_fold = m.predict(X_val_poly)
            mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
            rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
            r2_fold = r2_score(y_val_fold, y_pred_fold)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'mae': mae_fold,
                'rmse': rmse_fold,
                'r2': r2_fold
            })
            
            print(f"Fold {fold_idx+1}: MAE={mae_fold:.6f}, RMSE={rmse_fold:.6f}, R²={r2_fold:.4f}")
        
        mean_mae_cv = np.mean([f['mae'] for f in fold_results])
        mean_rmse_cv = np.mean([f['rmse'] for f in fold_results])
        mean_r2_cv = np.mean([f['r2'] for f in fold_results])
        std_mae_cv = np.std([f['mae'] for f in fold_results])
        
        print(f"\n📊 CV Media: MAE={mean_mae_cv:.6f}±{std_mae_cv:.6f}, "
              f"RMSE={mean_rmse_cv:.6f}, R²={mean_r2_cv:.4f}\n")
        
        # === PASO 6: GRÁFICAS ===
        print("📈 Generando gráficas...\n")
        
        metrics_train_dict = {
            'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train,
            'n_samples': len(df_train)
        }
        metrics_test_dict = {
            'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test,
            'n_samples': len(df_test)
        }
        
        create_validation_plots(
            model, df_train, df_test, best_sensor, plots_dir
        )
        
        # === PASO 7: REPORTE HTML ===
        metrics_cv_dict = {
            'mean_mae': mean_mae_cv,
            'mean_rmse': mean_rmse_cv,
            'mean_r2': mean_r2_cv,
            'std_mae': std_mae_cv
        }
        
        report_path = metrics_dir / "metrics_report.html"
        create_metrics_report(
            metrics_train_dict,
            metrics_test_dict,
            metrics_cv_dict,
            best_sensor,
            best_file,
            report_path
        )
        
        print("✓ Reporte HTML generado\n")
        
        # === PASO 8: GUARDAR MODELO ===
        print("="*70)
        print("PASO 8: GUARDANDO ARTEFACTOS")
        print("="*70 + "\n")
        
        # Guardar en AMBOS lados
        model_filename = f"residuals_{best_sensor}_v3.pkl"
        
        model_path_mlflow = models_dir / model_filename
        model_path_project = project_models_dir / model_filename
        
        model.save(str(model_path_mlflow))
        model.save(str(model_path_project))
        
        print(f"✓ MLflow:  {model_path_mlflow}")
        print(f"✓ Proyecto: {model_path_project}\n")
        
        # === PASO 9: EXPORTAR MÉTRICAS DETALLADAS ===
        metrics_detailed = {
            'train': metrics_train_dict,
            'test': metrics_test_dict,
            'cv': metrics_cv_dict,
            'fold_results': fold_results,
            'model_config': {
                'polynomial_degree': 3,
                'speed_column': 'KPH',
                'best_sensor': best_sensor,
                'best_file': str(best_file),
                'threshold': float(model.threshold)
            }
        }
        
        metrics_path = metrics_dir / "detailed_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_detailed, f, indent=2, default=str)
        
        print(f"✓ Métricas detalladas: {metrics_path.name}\n")
        
        # === PASO 10: LOG A MLFLOW ===
        mlflow.log_metrics({
            'train_mae': mae_train,
            'train_rmse': rmse_train,
            'train_r2': r2_train,
            'test_mae': mae_test,
            'test_rmse': rmse_test,
            'test_r2': r2_test,
            'cv_mean_mae': mean_mae_cv,
            'cv_mean_rmse': mean_rmse_cv,
            'cv_mean_r2': mean_r2_cv,
            'cv_std_mae': std_mae_cv,
            'threshold': float(model.threshold)
        })
        
        # === RESUMEN FINAL ===
        print("="*70)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*70)
        print("\n📊 MODELO FINAL")
        print(f"  Sensor: {best_sensor}")
        print("  Grado: 3")
        print(f"  Umbral: {model.threshold:.6f}\n")
        print("📈 TRAIN")
        print(f"  MAE: {mae_train:.6f} | RMSE: {rmse_train:.6f} | R²: {r2_train:.4f}\n")
        print("📈 TEST")
        print(f"  MAE: {mae_test:.6f} | RMSE: {rmse_test:.6f} | R²: {r2_test:.4f}\n")
        print("📈 K-FOLD CV")
        print(f"  MAE: {mean_mae_cv:.6f}±{std_mae_cv:.6f}")
        print(f"  R²: {mean_r2_cv:.4f}\n")
        print("📁 Guardado en:")
        print(f"  - MLflow: {models_dir}")
        print(f"  - Proyecto: {project_models_dir}\n")
        print(f"🌐 MLflow: http://localhost:5000/#/experiments/"
              f"{experiment_id}/runs/{run_id}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
