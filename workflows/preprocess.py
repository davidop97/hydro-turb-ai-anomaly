from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from tqdm import tqdm

from configs.config import settings
from src.preprocessing.pipeline import TurbineDataPipeline


def process_directory(
    input_dir: Path,
    output_dir: Path,
    category: str
) -> pd.DataFrame:
    """Procesa todos los CSV de un directorio."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(input_dir.glob("*.csv"))
    
    print(f"\n{'='*70}")
    print(f"📁 Procesando: {category}")
    print(f"   Archivos encontrados: {len(csv_files)}")
    print(f"{'='*70}\n")
    
    pipeline = TurbineDataPipeline()
    results = []
    
    for csv_file in tqdm(csv_files, desc=category):
        try:
            df_processed, nominal_speed, max_values = pipeline.fit_transform(str(csv_file))
            df_original = pipeline.get_original_data()
            
            # Guardar procesado
            output_path = output_dir / csv_file.name
            df_processed.to_csv(output_path, index=False)
            
            # Métricas
            results.append({
                'filename': csv_file.name,
                'category': category,
                'original_rows': len(df_original),
                'processed_rows': len(df_processed),
                'reduction_pct': ((len(df_original) - len(df_processed)) / len(df_original)) * 100,
                'nominal_speed': nominal_speed,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Error en {csv_file.name}: {e}")
            results.append({
                'filename': csv_file.name,
                'category': category,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def main():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        all_results = []
        
        for category in ['imbalance', 'misalignment']:
            input_dir = settings.DATA_DIR / "raw" / category
            output_dir = settings.DATA_DIR / "processed" / category
            
            df_results = process_directory(input_dir, output_dir, category)
            all_results.append(df_results)
            
            # Log a MLflow
            mlflow.log_metrics({
                f"{category}_files_processed": len(df_results),
                f"{category}_avg_reduction": df_results['reduction_pct'].mean()
            })
        
        # Guardar resumen
        summary_df = pd.concat(all_results, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = settings.DATA_DIR / "processed" / f"summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))
        
        print("\n✅ Procesamiento completado")
        print(f"📊 Resumen guardado en: {summary_path}")
        print(f"\n{summary_df.groupby('category')[['processed_rows', 'reduction_pct']].mean()}")


if __name__ == "__main__":
    main()
