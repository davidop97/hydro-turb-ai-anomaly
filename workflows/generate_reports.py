import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

from configs.config import settings
from src.preprocessing.pipeline import TurbineDataPipeline
from src.visualization.plots import (
    plot_all_sensors_comparison,
    plot_before_after_comparison,
    plot_category_comparison,
    plot_reduction_summary,
    plot_sensor_timeseries,
)

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning)


def generate_comparison_plots(category: str, sample_file: str):
    """
    Genera gráficas de comparación antes/después para un archivo de muestra.
    """
    raw_path = settings.DATA_DIR / "raw" / category / sample_file
    processed_path = settings.DATA_DIR / "processed" / category / sample_file
    
    if not raw_path.exists():
        print(f"  ⚠️ Archivo raw no encontrado: {sample_file}")
        return []
    
    if not processed_path.exists():
        print(f"  ⚠️ Archivo procesado no encontrado: {sample_file}")
        return []
    
    artifacts = []
    
    try:
        pipeline = TurbineDataPipeline()
        df_raw = pipeline.pipeline.named_steps['loader'].fit_transform(str(raw_path))
        
        if df_raw is None or len(df_raw) == 0:
            print(f"  ⚠️ No se pudo cargar archivo raw: {sample_file}")
            return []
        
        df_processed = pd.read_csv(processed_path)
        df_processed['Fecha'] = pd.to_datetime(df_processed['Fecha'], errors='coerce')
        df_processed = df_processed.dropna(subset=['Fecha'])
        
        if len(df_processed) == 0:
            print(f"  ⚠️ Archivo procesado vacío: {sample_file}")
            return []
        
    except Exception as e:
        print(f"  ❌ Error cargando datos para {sample_file}: {e}")
        return []
    
    output_dir = settings.ROOT_DIR / "outputs" / "plots" / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Serie de tiempo completa ANTES
        output_path = output_dir / f"{Path(sample_file).stem}_before.png"
        plot_sensor_timeseries(
            df_raw,
            output_path=output_path,
            title=f"Antes del Preprocesamiento - {Path(sample_file).stem}",
            figsize=(16, 8)
        )
        artifacts.append(str(output_path))
        print(f"  ✓ Antes: {output_path.name}")
        
        # 2. Serie de tiempo completa DESPUÉS
        output_path = output_dir / f"{Path(sample_file).stem}_after.png"
        plot_sensor_timeseries(
            df_processed,
            output_path=output_path,
            title=f"Después del Preprocesamiento - {Path(sample_file).stem}",
            figsize=(16, 8)
        )
        artifacts.append(str(output_path))
        print(f"  ✓ Después: {output_path.name}")
        
        # 3. Comparaciones para múltiples sensores (primeros 3)
        sensor_cols = [col for col in df_raw.columns if col not in ['Fecha', 'KPH']]
        sensors_to_plot = sensor_cols[:min(3, len(sensor_cols))]
        
        for i, sensor_col in enumerate(sensors_to_plot, 1):
            filename = f"{Path(sample_file).stem}_comparison_sensor{i}_{sensor_col}.png"
            output_path = output_dir / filename
            plot_before_after_comparison(
                df_raw,
                df_processed,
                sensor_col,
                output_path=output_path,
                title=f"Comparación {category.capitalize()} - Sensor {sensor_col}"
            )
            artifacts.append(str(output_path))
            print(f"  ✓ Comparación sensor {i} ({sensor_col}): {output_path.name}")
        
        # 4. Comparación de todos los sensores
        output_path = output_dir / f"{Path(sample_file).stem}_comparison_all_sensors.png"
        plot_all_sensors_comparison(
            df_raw,
            df_processed,
            output_path=output_path,
            title=f"Todos los Sensores - {category.capitalize()} - {Path(sample_file).stem}"
        )
        artifacts.append(str(output_path))
        print(f"  ✓ Comparación todos los sensores: {output_path.name}")
        
        print(f"  📊 Total: {len(artifacts)} gráficas generadas")
    
    except Exception as e:
        print(f"  ❌ Error generando gráficas para {sample_file}: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return artifacts


def main():
    print("\n" + "="*70)
    print("📊 GENERACIÓN DE REPORTES VISUALES")
    print("="*70 + "\n")
    
    # Configurar MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"visualization_{run_timestamp}"
    
    with mlflow.start_run(run_name=run_name) as run:
        
        mlflow.set_tags({
            "stage": "visualization",
            "author": "David Oliva Patiño",
            "version": "1.0.0",
            "description": "Generación de visualizaciones para documentación de tesis"
        })
        
        # Cargar resumen
        processed_dir = settings.DATA_DIR / "processed"
        summary_files = sorted(processed_dir.glob("summary_*.csv"))
        
        if not summary_files:
            print("❌ No se encontró archivo summary. Ejecuta preprocess.py primero.")
            return
        
        latest_summary = summary_files[-1]
        summary_df = pd.read_csv(latest_summary)
        
        print(f"📄 Usando resumen: {latest_summary.name}\n")
        
        # ===== GRÁFICAS AGREGADAS =====
        print("📈 Generando gráficas agregadas...\n")
        
        output_dir = settings.ROOT_DIR / "outputs" / "plots" / "summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Reducción por archivo
            output_path = output_dir / "reduction_summary.png"
            plot_reduction_summary(summary_df, output_path=output_path)
            
            if output_path.exists():
                print(f"  ✓ Reducción de datos: {output_path.name}")
            else:
                print(f"  ⚠️ Archivo no encontrado: {output_path}")
            
            # 2. Comparación de categorías
            output_path = output_dir / "category_comparison.png"
            plot_category_comparison(summary_df, output_path=output_path)
            
            if output_path.exists():
                print(f"  ✓ Comparación categorías: {output_path.name}")
            else:
                print(f"  ⚠️ Archivo no encontrado: {output_path}")
        
        except Exception as e:
            print(f"  ❌ Error generando gráficas agregadas: {e}")
            import traceback
            traceback.print_exc()
        
        # ===== GRÁFICAS DETALLADAS =====
        print("\n📊 Generando comparaciones detalladas antes/después...\n")
        
        total_artifacts = 0
        
        for category in ['imbalance', 'misalignment']:
            cat_files = summary_df[
                (summary_df['category'] == category) & 
                (summary_df['status'] == 'success')
            ].copy().reset_index(drop=True)
            
            if len(cat_files) == 0:
                print(f"⚠️ No hay archivos exitosos para {category}\n")
                continue
            
            print(f"{'='*70}")
            print(f"📁 Categoría: {category.upper()}")
            print(f"{'='*70}")
            
            avg_reduction = cat_files['reduction_pct'].mean()
            sample_idx = (cat_files['reduction_pct'] - avg_reduction).abs().argsort().iloc[0]
            sample_file = cat_files.loc[sample_idx, 'filename']
            sample_reduction = cat_files.loc[sample_idx, 'reduction_pct']
            
            print(f"📄 Archivo seleccionado: {sample_file}")
            print(f"   Reducción: {sample_reduction:.1f}% (promedio: {avg_reduction:.1f}%)")
            
            artifacts = generate_comparison_plots(category, sample_file)
            
            total_artifacts += len(artifacts)
            
            print()
        
        # ===== SUBIR TODOS LOS ARTEFACTOS JUNTOS AL FINAL =====
        print("📤 Subiendo artefactos a MLflow...\n")
        
        try:
            # Subir carpeta completa de plots
            plots_dir = settings.ROOT_DIR / "outputs" / "plots"
            
            if plots_dir.exists():
                # Cambiar al directorio para rutas relativas
                import os
                original_cwd = os.getcwd()
                os.chdir(plots_dir.parent)  # Cambiar a outputs
                
                try:
                    mlflow.log_artifacts(str(plots_dir), artifact_path="visualizations")
                    print(f"✓ Todos los artefactos subidos a MLflow desde: {plots_dir}")
                except Exception as e:
                    print(f"⚠️ Error subiendo carpeta: {e}")
                    # Intentar subir archivo por archivo
                    print("  → Intentando subir archivos individualmente...\n")
                    for png_file in plots_dir.rglob("*.png"):
                        try:
                            relative_path = png_file.relative_to(plots_dir.parent)
                            artifact_dir = str(relative_path.parent)
                            mlflow.log_artifact(str(png_file), artifact_path=artifact_dir)
                            print(f"    ✓ {png_file.name}")
                        except Exception as e2:
                            print(f"    ⚠️ {png_file.name}: {e2}")
                finally:
                    os.chdir(original_cwd)
            else:
                print(f"⚠️ Directorio de plots no encontrado: {plots_dir}")
        
        except Exception as e:
            print(f"❌ Error subiendo artefactos: {e}")
            import traceback
            traceback.print_exc()
        
        # ===== RESUMEN FINAL =====
        print("\n" + "="*70)
        print("✅ REPORTES VISUALES COMPLETADOS")
        print("="*70)
        print("\n📊 Estadísticas:")
        print(f"   - Gráficas generadas: {total_artifacts + 2}")
        print(f"   - Ubicación local: {settings.ROOT_DIR / 'outputs' / 'plots'}")
        print(f"\n🔗 MLflow Run ID: {run.info.run_id}")
        print(f"🌐 Ver en MLflow: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
