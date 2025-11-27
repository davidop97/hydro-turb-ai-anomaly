import json
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from configs.config import settings
from src.preprocessing.eda_loader import EDADataLoader
from src.visualization.eda_plots import (
    plot_boxplots_outliers,
    plot_correlation_matrix,
    plot_distribution_analysis,
    plot_velocity_analysis,
)
from src.visualization.plots import plot_sensor_timeseries

warnings.filterwarnings("ignore")


def generate_univariate_stats(
    df: pd.DataFrame,
    sensor_cols: list
) -> dict:
    """
    Análisis estadístico univariable completo.
    Incluye tests de normalidad, outliers, etc.
    """
    stats_dict = {'sensors': {}, 'interpretation': {}}
    
    for sensor in sensor_cols:
        data = df[sensor].dropna()
        
        if len(data) == 0:
            continue
        
        # Estadísticas descriptivas
        sensor_stats = {
            'n_observations': int(len(data)),
            'n_missing': int(df[sensor].isna().sum()),
            'pct_missing': float(
                (df[sensor].isna().sum() / len(df)) * 100
            ),
            'data_type': str(df[sensor].dtype),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'var': float(data.var()),
            'min': float(data.min()),
            'max': float(data.max()),
            'range': float(data.max() - data.min()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
            'skewness': float(sp_stats.skew(data)),
            'kurtosis': float(sp_stats.kurtosis(data)),
        }
        
        # Test de normalidad (Shapiro-Wilk)
        if len(data) > 5000:
            # Para muestras grandes, usar subsample
            test_data = np.random.choice(data, 5000, replace=False)
        else:
            test_data = data
        
        try:
            shapiro_stat, shapiro_p = sp_stats.shapiro(test_data)
            sensor_stats['normality_test'] = {
                'test_name': 'Shapiro-Wilk',
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': bool(shapiro_p > 0.05)
            }
        except Exception:
            sensor_stats['normality_test'] = None
        
        # Detección de outliers (método IQR)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        sensor_stats['outliers'] = {
            'count': int(len(outliers)),
            'pct': float((len(outliers) / len(data)) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        
        # Z-score outliers (|z| > 3)
        z_scores = np.abs(sp_stats.zscore(data))
        z_outliers = np.sum(z_scores > 3)
        sensor_stats['z_score_outliers'] = int(z_outliers)
        
        stats_dict['sensors'][sensor] = sensor_stats
        
        # Interpretación texual
        interpretation = []
        
        if sensor_stats['normality_test'] and not sensor_stats['normality_test']['is_normal']:
            interpretation.append(
                f"• Distribución NO normal (p={sensor_stats['normality_test']['p_value']:.4f})"
            )
        
        if sensor_stats['skewness'] > 1:
            interpretation.append(
                f"• Asimetría positiva pronunciada (skewness={sensor_stats['skewness']:.2f})"
            )
        elif sensor_stats['skewness'] < -1:
            interpretation.append(
                f"• Asimetría negativa pronunciada (skewness={sensor_stats['skewness']:.2f})"
            )
        
        if sensor_stats['outliers']['pct'] > 5:
            interpretation.append(
                f"• {sensor_stats['outliers']['pct']:.1f}% de outliers detectados (método IQR)"
            )
        
        if sensor_stats['kurtosis'] > 3:
            interpretation.append("• Distribución leptocúrtica (colas pesadas)")
        
        stats_dict['interpretation'][sensor] = interpretation
    
    return stats_dict


def generate_bivariate_stats(
    df: pd.DataFrame,
    sensor_cols: list
) -> dict:
    """
    Análisis bivariable: correlaciones, covariancias, etc.
    """
    stats_dict = {
        'correlations': {},
        'sensor_velocity_corr': {},
        'interpretation': []
    }
    
    # Crear matriz de correlación
    all_cols = sensor_cols + ['KPH_abs']
    corr_matrix = df[all_cols].corr()
    
    # Guardar matriz
    stats_dict['correlations'] = corr_matrix.to_dict()
    
    # Correlación de cada sensor con velocidad absoluta
    for sensor in sensor_cols:
        corr = df[sensor].corr(df['KPH_abs'])
        stats_dict['sensor_velocity_corr'][sensor] = float(corr)
        
        if abs(corr) > 0.7:
            stats_dict['interpretation'].append(
                f"• {sensor} tiene correlación FUERTE con velocidad (r={corr:.3f})"
            )
        elif abs(corr) > 0.4:
            stats_dict['interpretation'].append(
                f"• {sensor} tiene correlación MODERADA con velocidad (r={corr:.3f})"
            )
    
    # Autocorrelación entre sensores
    top_correlations = []
    for i, sensor1 in enumerate(sensor_cols):
        for sensor2 in sensor_cols[i+1:]:
            corr = df[sensor1].corr(df[sensor2])
            top_correlations.append((sensor1, sensor2, corr))
    
    top_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    stats_dict['top_sensor_correlations'] = [
        {
            'sensor1': s1,
            'sensor2': s2,
            'correlation': float(c)
        }
        for s1, s2, c in top_correlations[:5]
    ]
    
    return stats_dict


def create_enhanced_eda_report(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    univariate_stats: dict,
    bivariate_stats: dict,
    category: str,
    output_path: Path
) -> None:
    """
    Reporte HTML profesional tipo paper científico.
    """
    sensor_cols = [
        col for col in df_raw.columns 
        if col not in ['Fecha', 'KPH', 'KPH_abs']
    ]
    
    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - {category.capitalize()}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', 'Times New Roman', serif;
            background-color: #f5f7fa;
            color: #1f2937;
            line-height: 1.7;
            font-size: 11pt;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}
        
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.95;
        }}
        
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        
        .section h2 {{
            font-size: 1.6em;
            margin-bottom: 15px;
            color: #1f2937;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        
        .section h3 {{
            font-size: 1.2em;
            margin-top: 20px;
            margin-bottom: 12px;
            color: #667eea;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px;
            border-radius: 6px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-card .label {{
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 6px;
        }}
        
        .stat-card .value {{
            font-size: 1.6em;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        
        table thead {{
            background-color: #667eea;
            color: white;
        }}
        
        table th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border: 1px solid #e5e7eb;
        }}
        
        table td {{
            padding: 10px 12px;
            border: 1px solid #e5e7eb;
        }}
        
        table tbody tr:nth-child(odd) {{
            background-color: #f9fafb;
        }}
        
        table tbody tr:hover {{
            background-color: #f3f4f6;
        }}
        
        .interpretation {{
            background-color: #f0f9ff;
            border-left: 4px solid #0284c7;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            font-size: 0.95em;
        }}
        
        .interpretation p {{
            margin: 5px 0;
        }}
        
        .warning {{
            background-color: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        
        .footer {{
            text-align: center;
            color: #6b7280;
            font-size: 0.85em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
        }}
        
        .page-break {{
            page-break-after: always;
            margin: 40px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        
        <!-- HEADER -->
        <div class="header">
            <h1>📊 Análisis Exploratorio de Datos (EDA)</h1>
            <div class="subtitle">
                Categoría: <strong>{category.upper()}</strong> | 
                Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <!-- RESUMEN GENERAL -->
        <div class="section">
            <h2>1. Resumen General</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">Registros (Raw)</div>
                    <div class="value">{len(df_raw):,}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Registros (Procesado)</div>
                    <div class="value">{len(df_processed):,}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Reducción</div>
                    <div class="value">{((len(df_raw) - len(df_processed)) / len(df_raw) * 100):.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Sensores</div>
                    <div class="value">{len(sensor_cols)}</div>
                </div>
            </div>
            
            <h3>Rango Temporal</h3>
            <table>
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th>Inicio</th>
                        <th>Fin</th>
                        <th>Duración</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Raw</strong></td>
                        <td>{df_raw['Fecha'].min()}</td>
                        <td>{df_raw['Fecha'].max()}</td>
                        <td>{(df_raw['Fecha'].max() - df_raw['Fecha'].min()).days} días</td>
                    </tr>
                    <tr>
                        <td><strong>Procesado</strong></td>
                        <td>{df_processed['Fecha'].min()}</td>
                        <td>{df_processed['Fecha'].max()}</td>
                        <td>{(df_processed['Fecha'].max() - df_processed['Fecha'].min()).days} días</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- ANÁLISIS UNIVARIABLE -->
        <div class="section">
            <h2>2. Análisis Univariable</h2>
            <p>
                Análisis estadístico individual de cada variable. 
                Se incluyen tests de normalidad (Shapiro-Wilk) y detección de outliers.
            </p>
            
            <h3>Estadísticas Descriptivas por Sensor</h3>
            <table>
                <thead>
                    <tr>
                        <th>Sensor</th>
                        <th>n</th>
                        <th>Media</th>
                        <th>Desv. Est.</th>
                        <th>Mín</th>
                        <th>Máx</th>
                        <th>Asimetría</th>
                        <th>Outliers (IQR)</th>
                    </tr>
                </thead>
                <tbody>
"""  # noqa: E501
    
    for sensor in sensor_cols:
        s = univariate_stats['sensors'][sensor]
        html += f"""
                    <tr>
                        <td><strong>{sensor}</strong></td>
                        <td>{s['n_observations']}</td>
                        <td>{s['mean']:.4f}</td>
                        <td>{s['std']:.4f}</td>
                        <td>{s['min']:.4f}</td>
                        <td>{s['max']:.4f}</td>
                        <td>{s['skewness']:.4f}</td>
                        <td>{s['outliers']['pct']:.2f}%</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
            
            <h3>Interpretación Univariable</h3>
"""
    
    for sensor, interp in univariate_stats['interpretation'].items():
        if interp:
            html += f"""
            <div class="interpretation">
                <p><strong>{sensor}:</strong></p>
"""
            for line in interp:
                html += f"                <p>{line}</p>\n"
            html += "            </div>\n"
    
    html += """
        </div>
        
        <!-- ANÁLISIS BIVARIABLE -->
        <div class="section">
            <h2>3. Análisis Bivariable</h2>
            <p>
                Estudio de relaciones entre variables mediante correlación de Pearson.
            </p>
            
            <h3>Correlación Sensor - Velocidad Absoluta</h3>
            <table>
                <thead>
                    <tr>
                        <th>Sensor</th>
                        <th>Correlación (r)</th>
                        <th>Fuerza</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for sensor, corr in bivariate_stats['sensor_velocity_corr'].items():
        if abs(corr) > 0.7:
            strength = "FUERTE"
        elif abs(corr) > 0.4:
            strength = "MODERADA"
        else:
            strength = "DÉBIL"
        
        html += f"""
                    <tr>
                        <td><strong>{sensor}</strong></td>
                        <td>{corr:.4f}</td>
                        <td>{strength}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
            
            <h3>Correlaciones entre Sensores (Top 5)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Sensor 1</th>
                        <th>Sensor 2</th>
                        <th>Correlación</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for item in bivariate_stats['top_sensor_correlations']:
        html += f"""
                    <tr>
                        <td>{item['sensor1']}</td>
                        <td>{item['sensor2']}</td>
                        <td>{item['correlation']:.4f}</td>
                    </tr>
"""
    
    html += f"""
                </tbody>
            </table>
            
            <h3>Interpretación Bivariable</h3>
            <div class="interpretation">
"""  # noqa: F541
    
    for line in bivariate_stats['interpretation']:
        html += f"                <p>{line}</p>\n"
    
    html += f"""
            </div>
        </div>
        
        <!-- ANÁLISIS DE VELOCIDAD -->
        <div class="section">
            <h2>4. Análisis de Velocidad</h2>
            <p>
                Nota: Los datos de velocidad se presentan en dos formas:
                <br/><strong>Con signo:</strong> Incluye dirección (positiva: avance, negativa: retroceso)
                <br/><strong>Absoluta:</strong> Magnitud sin considerar dirección
            </p>
            
            <h3>Estadísticas de Velocidad</h3>
            <table>
                <thead>
                    <tr>
                        <th>Métrica</th>
                        <th>Con Signo (KPH)</th>
                        <th>Absoluta (|KPH|)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Media</strong></td>
                        <td>{df_raw['KPH'].mean():.4f}</td>
                        <td>{df_raw['KPH_abs'].mean():.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Mediana</strong></td>
                        <td>{df_raw['KPH'].median():.4f}</td>
                        <td>{df_raw['KPH_abs'].median():.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Desv. Est.</strong></td>
                        <td>{df_raw['KPH'].std():.4f}</td>
                        <td>{df_raw['KPH_abs'].std():.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Mín</strong></td>
                        <td>{df_raw['KPH'].min():.4f}</td>
                        <td>{df_raw['KPH_abs'].min():.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Máx</strong></td>
                        <td>{df_raw['KPH'].max():.4f}</td>
                        <td>{df_raw['KPH_abs'].max():.4f}</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="warning">
                <strong>⚠️ Nota Importante:</strong> 
                La media de velocidad con signo es {df_raw['KPH'].mean():.2f} KPH debido a que 
                incluye fases de aceleración (positiva) y frenado (negativa). 
                Para análisis de magnitud, usar velocidad absoluta: {df_raw['KPH_abs'].mean():.2f} KPH.
            </div>
        </div>
        
        <!-- PIE DE PÁGINA -->
        <div class="footer">
            <p>
                Reporte generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                <br/>Categoría: {category.upper()} | Pipeline EDA v1.0
            </p>
        </div>
        
    </div>
</body>
</html>
"""  # noqa: E501
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    print("\n" + "="*70)
    print("📊 ANÁLISIS EXPLORATORIO DE DATOS AVANZADO (EDA)")
    print("="*70 + "\n")
    
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    run_name = f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        mlartifacts_base = settings.ROOT_DIR / "mlartifacts"
        mlflow_artifacts_dir = (
            mlartifacts_base / str(experiment_id) / run_id / "artifacts"
        )
        mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        eda_reports_dir = mlflow_artifacts_dir / "eda" / "reports"
        eda_plots_dir = mlflow_artifacts_dir / "eda" / "plots"
        eda_stats_dir = mlflow_artifacts_dir / "eda" / "statistics"
        
        eda_reports_dir.mkdir(parents=True, exist_ok=True)
        eda_plots_dir.mkdir(parents=True, exist_ok=True)
        eda_stats_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Artefactos: {mlflow_artifacts_dir}\n")
        
        mlflow.set_tags({
            "stage": "eda_advanced",
            "author": "David Oliva Patiño",
            "version": "2.0.0",
            "description": "EDA avanzado con análisis univariable y bivariable"
        })
        
        # Cargar resumen
        processed_dir = settings.DATA_DIR / "processed"
        summary_files = sorted(processed_dir.glob("summary_*.csv"))
        
        if not summary_files:
            print("❌ No se encontró resumen de preprocesamiento")
            return
        
        summary_df = pd.read_csv(summary_files[-1])
        eda_loader = EDADataLoader()
        
        # Procesar cada categoría
        total_plots = 0
        
        for category in ['imbalance', 'misalignment']:
            print(f"\n{'='*70}")
            print(f"📁 Análisis: {category.upper()}")
            print(f"{'='*70}\n")
            
            cat_files = summary_df[
                (summary_df['category'] == category)
                & (summary_df['status'] == 'success')
            ].copy().reset_index(drop=True)
            
            if len(cat_files) == 0:
                print(f"⚠️ No hay archivos para {category}")
                continue
            
            avg_reduction = cat_files['reduction_pct'].mean()
            sample_idx = (
                (cat_files['reduction_pct'] - avg_reduction)
                .abs()
                .argsort()
                .iloc[0]
            )
            sample_file = cat_files.loc[sample_idx, 'filename']
            
            print(f"📄 Archivo analizado: {sample_file}")
            
            raw_path = settings.DATA_DIR / "raw" / category / sample_file
            processed_path = (
                settings.DATA_DIR / "processed" / category / sample_file
            )
            
            try:
                df_raw, df_processed = eda_loader.load_before_after(
                    str(raw_path),
                    str(processed_path)
                )
            except Exception as e:
                print(f"❌ Error cargando datos: {e}")
                continue
            
            sensor_cols = [
                col for col in df_raw.columns
                if col not in ['Fecha', 'KPH', 'KPH_abs']
            ]
            
            # ===== ESTADÍSTICAS =====
            print("📊 Calculando estadísticas univariables...")
            univariate = generate_univariate_stats(df_raw, sensor_cols)
            
            print("📊 Calculando estadísticas bivariables...")
            bivariate = generate_bivariate_stats(df_raw, sensor_cols)
            
            # ===== GRÁFICAS =====
            print("📈 Generando gráficas...")
            
            # 1. Distribuciones
            plot_path = eda_plots_dir / f"distributions_{category}.png"
            plot_distribution_analysis(
                df_raw,
                sensor_cols,
                output_path=plot_path
            )
            print(f"  ✓ Distribuciones: {plot_path.name}")
            total_plots += 1
            
            # 2. Correlaciones
            plot_path = eda_plots_dir / f"correlation_{category}.png"
            plot_correlation_matrix(
                df_raw,
                sensor_cols,
                output_path=plot_path
            )
            print(f"  ✓ Correlaciones: {plot_path.name}")
            total_plots += 1
            
            # 3. Box plots (outliers)
            plot_path = eda_plots_dir / f"boxplots_{category}.png"
            plot_boxplots_outliers(
                df_raw,
                sensor_cols,
                output_path=plot_path
            )
            print(f"  ✓ Box plots: {plot_path.name}")
            total_plots += 1
            
            # 4. Velocidad
            plot_path = eda_plots_dir / f"velocity_{category}.png"
            plot_velocity_analysis(
                df_raw,
                output_path=plot_path
            )
            print(f"  ✓ Velocidad: {plot_path.name}")
            total_plots += 1
            
            # 5. Series temporales
            plot_path = eda_plots_dir / f"timeseries_{category}_raw.png"
            plot_sensor_timeseries(
                df_raw,
                output_path=plot_path,
                title=f"Serie de Tiempo - {category.capitalize()} (Raw)",
                figsize=(14, 7)
            )
            print(f"  ✓ Series temporales (raw): {plot_path.name}")
            total_plots += 1
            
            # ===== REPORTE HTML =====
            print("📄 Generando reporte HTML...")
            report_path = eda_reports_dir / f"eda_report_{category}.html"
            create_enhanced_eda_report(
                df_raw,
                df_processed,
                univariate,
                bivariate,
                category,
                report_path
            )
            print(f"  ✓ Reporte HTML: {report_path.name}")
            
            # ===== EXPORTAR ESTADÍSTICAS A JSON =====
            print("💾 Guardando estadísticas...")
            stats_path = (
                eda_stats_dir / f"univariate_{category}.json"
            )
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(
                    univariate,
                    f,
                    indent=2,
                    default=str
                )
            print(f"  ✓ Stats univariable: {stats_path.name}")
            
            bivariate_path = (
                eda_stats_dir / f"bivariate_{category}.json"
            )
            with open(bivariate_path, 'w', encoding='utf-8') as f:
                json.dump(
                    bivariate,
                    f,
                    indent=2,
                    default=str
                )
            print(f"  ✓ Stats bivariable: {bivariate_path.name}")
            
            # ===== LOG A MLFLOW =====
            mlflow.log_metrics({
                f"{category}_n_observations": len(df_raw),
                f"{category}_n_sensors": len(sensor_cols),
                f"{category}_mean_velocity_abs": float(df_raw['KPH_abs'].mean()),
                f"{category}_mean_velocity_signed": float(df_raw['KPH'].mean()),
            })
        
        # ===== RESUMEN FINAL =====
        print(f"\n{'='*70}")
        print("✅ EDA AVANZADO COMPLETADO")
        print(f"{'='*70}")
        print(f"\n📊 Gráficas generadas: {total_plots}")
        print("📄 Reportes HTML: 2")
        print("📈 Archivos estadísticos: 4")
        print(f"\n📂 Ubicación: {mlflow_artifacts_dir / 'eda'}")
        print(f"\n🌐 MLflow: http://localhost:5000/#/experiments/"
              f"{experiment_id}/runs/{run_id}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
