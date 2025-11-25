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


def generate_comparison_plots(category: str, sample_file: str, mlflow_artifacts_dir: Path):
    """
    Genera gráficas directamente en la carpeta de artifacts de MLflow.
    """
    raw_path = settings.DATA_DIR / "raw" / category / sample_file
    processed_path = settings.DATA_DIR / "processed" / category / sample_file
    
    if not raw_path.exists() or not processed_path.exists():
        print(f"  ⚠️ Archivos no encontrados para {sample_file}")
        return 0
    
    try:
        pipeline = TurbineDataPipeline()
        df_raw = pipeline.pipeline.named_steps['loader'].fit_transform(str(raw_path))
        
        if df_raw is None or len(df_raw) == 0:
            print(f"  ⚠️ No se pudo cargar archivo raw: {sample_file}")
            return 0
        
        df_processed = pd.read_csv(processed_path)
        df_processed['Fecha'] = pd.to_datetime(df_processed['Fecha'], errors='coerce')
        df_processed = df_processed.dropna(subset=['Fecha'])
        
        if len(df_processed) == 0:
            print(f"  ⚠️ Archivo procesado vacío: {sample_file}")
            return 0
        
    except Exception as e:
        print(f"  ❌ Error cargando datos: {e}")
        return 0
    
    # Crear carpeta para esta categoría
    category_dir = mlflow_artifacts_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    try:
        # 1. Antes
        output_path = category_dir / f"{Path(sample_file).stem}_before.png"
        plot_sensor_timeseries(df_raw, output_path=output_path,
                             title=f"Antes - {Path(sample_file).stem}", figsize=(16, 8))
        print(f"  ✓ {output_path.name}")
        count += 1
        
        # 2. Después
        output_path = category_dir / f"{Path(sample_file).stem}_after.png"
        plot_sensor_timeseries(df_processed, output_path=output_path,
                             title=f"Después - {Path(sample_file).stem}", figsize=(16, 8))
        print(f"  ✓ {output_path.name}")
        count += 1
        
        # 3. Sensores individuales (hasta 3)
        sensor_cols = [col for col in df_raw.columns if col not in ['Fecha', 'KPH']]
        for i, sensor_col in enumerate(sensor_cols[:3], 1):
            output_path = category_dir / f"{Path(sample_file).stem}_sensor{i}_{sensor_col}.png"
            plot_before_after_comparison(df_raw, df_processed, sensor_col, output_path=output_path,
                                        title=f"Sensor {i}: {sensor_col}")
            print(f"  ✓ {output_path.name}")
            count += 1
        
        # 4. Todos los sensores
        output_path = category_dir / f"{Path(sample_file).stem}_all_sensors.png"
        plot_all_sensors_comparison(df_raw, df_processed, output_path=output_path,
                                   title=f"Todos los sensores - {category.capitalize()}")
        print(f"  ✓ {output_path.name}")
        count += 1
    
    except Exception as e:
        print(f"  ❌ Error generando gráficas: {e}")
        import traceback
        traceback.print_exc()
    
    return count


def main():
    print("\n" + "="*70)
    print("📊 GENERACIÓN DE REPORTES VISUALES")
    print("="*70 + "\n")
    
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    run_name = f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        
        mlflow.set_tags({
            "stage": "visualization",
            "version": "1.0.0",
            "description": "Visualizaciones para tesis"
        })
        
        # ⭐ CLAVE: Usar mlartifacts/ directamente (para servidor Docker)
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        # Ruta de artifacts para MLflow server en Docker
        mlartifacts_base = settings.ROOT_DIR / "mlartifacts"
        mlflow_artifacts_dir = mlartifacts_base / str(experiment_id) / run_id / "artifacts"
        
        print(f"📂 Guardando en: {mlflow_artifacts_dir}")
        print("   (MLflow server usará esta carpeta)\n")
        
        mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # ===== GRÁFICAS AGREGADAS =====
        print("📈 Gráficas agregadas...\n")
        
        processed_dir = settings.DATA_DIR / "processed"
        summary_files = sorted(processed_dir.glob("summary_*.csv"))
        
        if not summary_files:
            print("❌ No se encontró archivo summary.")
            return
        
        summary_df = pd.read_csv(summary_files[-1])
        
        summary_dir = mlflow_artifacts_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            output_path = summary_dir / "reduction_summary.png"
            plot_reduction_summary(summary_df, output_path=output_path)
            print(f"  ✓ {output_path.name}")
            
            output_path = summary_dir / "category_comparison.png"
            plot_category_comparison(summary_df, output_path=output_path)
            print(f"  ✓ {output_path.name}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # ===== GRÁFICAS DETALLADAS =====
        print("\n📊 Comparaciones antes/después...\n")
        
        total = 0
        
        for category in ['imbalance', 'misalignment']:
            cat_files = summary_df[
                (summary_df['category'] == category) & 
                (summary_df['status'] == 'success')
            ].copy().reset_index(drop=True)
            
            if len(cat_files) == 0:
                continue
            
            print(f"{'='*70}")
            print(f"📁 {category.upper()}")
            print(f"{'='*70}")
            
            avg_reduction = cat_files['reduction_pct'].mean()
            sample_idx = (cat_files['reduction_pct'] - avg_reduction).abs().argsort().iloc[0]
            sample_file = cat_files.loc[sample_idx, 'filename']
            
            print(f"📄 {sample_file}")
            
            count = generate_comparison_plots(category, sample_file, mlflow_artifacts_dir)
            total += count
            print()
        
        # ===== VERIFICACIÓN =====
        print("="*70)
        print(f"✅ COMPLETADO - {total + 2} gráficas generadas")
        print("="*70)
        
        # Verificar que los archivos existen
        all_pngs = list(mlflow_artifacts_dir.rglob("*.png"))
        print("\n📊 Archivos en mlartifacts/:")
        print(f"   Total: {len(all_pngs)} imágenes PNG")
        
        if len(all_pngs) > 0:
            print("\n✓ Archivos creados correctamente en:")
            print(f"  {mlflow_artifacts_dir}")
            print("\n⏳ Espera 5-10 segundos y recarga MLflow UI...")
        else:
            print("\n⚠️ No se encontraron archivos PNG")
        
        print(f"\n🌐 MLflow UI: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
