import json
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
            df_processed, nominal_speed, max_values = (
                pipeline.fit_transform(str(csv_file))
            )
            df_original = pipeline.get_original_data()
            
            # Guardar procesado
            output_path = output_dir / csv_file.name
            df_processed.to_csv(output_path, index=False)
            
            # Métricas detalladas
            results.append({
                'filename': csv_file.name,
                'category': category,
                'original_rows': len(df_original),
                'processed_rows': len(df_processed),
                'reduction_pct': (
                    (len(df_original) - len(df_processed)) 
                    / len(df_original)
                ) * 100,
                'nominal_speed': nominal_speed,
                'columns': len(df_processed.columns),
                'file_size_kb': csv_file.stat().st_size / 1024,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Error en {csv_file.name}: {e}")
            results.append({
                'filename': csv_file.name,
                'category': category,
                'original_rows': 0,
                'processed_rows': 0,
                'reduction_pct': 0,
                'nominal_speed': 0,
                'columns': 0,
                'file_size_kb': csv_file.stat().st_size / 1024,
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    return pd.DataFrame(results)


def main():
    # Configurar MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    # Configurar experimento con descripción
    experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
    if experiment:
        mlflow.set_experiment_tag(
            "mlflow.note.content",
            "Pipeline de preprocesamiento de datos de turbinas hidroeléctricas. "
            "Limpia y normaliza datos de sensores para detección de anomalías."
        )
    
    run_name = f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        
        # ===== CONFIGURAR DIRECTORIO DE ARTEFACTOS =====
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        # Ruta de artifacts para MLflow Docker
        mlartifacts_base = settings.ROOT_DIR / "mlartifacts"
        mlflow_artifacts_dir = (
            mlartifacts_base / str(experiment_id) / run_id / "artifacts"
        )
        mlflow_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 Artefactos se guardarán en: {mlflow_artifacts_dir}\n")
        
        # ===== TAGS =====
        mlflow.set_tags({
            "stage": "preprocessing",
            "author": "David Oliva Patiño",
            "version": "1.0.0",
            "description": "Preprocesamiento completo de datos raw a processed"
        })
        
        # ===== PARÁMETROS =====
        mlflow.log_params({
            "categories": "imbalance,misalignment",
            "pipeline_version": "1.0.0",
            "raw_data_path": str(settings.DATA_DIR / "raw"),
            "processed_data_path": str(settings.DATA_DIR / "processed"),
            "timestamp": datetime.now().isoformat()
        })
        
        all_results = []
        total_files_processed = 0
        total_rows_original = 0
        total_rows_processed = 0
        
        # ===== PROCESAR CADA CATEGORÍA =====
        for category in ['imbalance', 'misalignment']:
            input_dir = settings.DATA_DIR / "raw" / category
            output_dir = settings.DATA_DIR / "processed" / category
            
            df_results = process_directory(input_dir, output_dir, category)
            all_results.append(df_results)
            
            # Filtrar archivos exitosos
            files_ok = df_results[df_results['status'] == 'success']
            files_error = df_results[df_results['status'] == 'error']
            
            if len(files_ok) > 0:
                mlflow.log_metrics({
                    f"{category}_files_total": len(df_results),
                    f"{category}_files_success": len(files_ok),
                    f"{category}_files_error": len(files_error),
                    f"{category}_avg_reduction_pct": (
                        files_ok['reduction_pct'].mean()
                    ),
                    f"{category}_avg_processed_rows": (
                        files_ok['processed_rows'].mean()
                    ),
                    f"{category}_total_original_rows": (
                        files_ok['original_rows'].sum()
                    ),
                    f"{category}_total_processed_rows": (
                        files_ok['processed_rows'].sum()
                    ),
                })
                
                total_files_processed += len(files_ok)
                total_rows_original += files_ok['original_rows'].sum()
                total_rows_processed += files_ok['processed_rows'].sum()
        
        # ===== MÉTRICAS GLOBALES =====
        if total_rows_original > 0:
            mlflow.log_metrics({
                "total_files_processed": total_files_processed,
                "total_rows_original": total_rows_original,
                "total_rows_processed": total_rows_processed,
                "global_reduction_pct": (
                    (total_rows_original - total_rows_processed)
                    / total_rows_original
                ) * 100
            })
        
        # ===== GUARDAR RESUMEN =====
        summary_df = pd.concat(all_results, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar también en data/processed (para uso local)
        summary_path_local = (
            settings.DATA_DIR / "processed" / f"summary_{timestamp}.csv"
        )
        summary_df.to_csv(summary_path_local, index=False)
        
        # ===== GUARDAR ARTEFACTOS EN MLARTIFACTS =====
        reports_dir = mlflow_artifacts_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        configs_dir = mlflow_artifacts_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Resumen CSV
        summary_path = reports_dir / f"summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Guardado: {summary_path.name}")
        
        # 2. Estadísticas detalladas (JSON)
        stats = {}
        for category in ['imbalance', 'misalignment']:
            cat_data = summary_df[
                (summary_df['category'] == category)
                & (summary_df['status'] == 'success')
            ]
            if len(cat_data) > 0:
                stats[category] = {
                    "files_processed": int(len(cat_data)),
                    "avg_rows_original": float(
                        cat_data['original_rows'].mean()
                    ),
                    "avg_rows_processed": float(
                        cat_data['processed_rows'].mean()
                    ),
                    "avg_reduction_pct": float(
                        cat_data['reduction_pct'].mean()
                    ),
                    "total_rows_processed": int(
                        cat_data['processed_rows'].sum()
                    )
                }
        
        stats_path = reports_dir / f"stats_{timestamp}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Guardado: {stats_path.name}")
        
        # 3. Configuración del pipeline
        pipeline_config = {
            "raw_data_dir": str(settings.DATA_DIR / "raw"),
            "processed_data_dir": str(settings.DATA_DIR / "processed"),
            "categories": ["imbalance", "misalignment"],
            "execution_date": datetime.now().isoformat()
        }
        
        config_path = configs_dir / f"config_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_config, f, indent=2)
        print(f"✓ Guardado: {config_path.name}")
        
        # ===== REGISTRAR DATASETS =====
        try:
            # Dataset de entrada
            mlflow.log_input(
                mlflow.data.from_pandas(
                    summary_df[[
                        'filename',
                        'category',
                        'original_rows',
                        'status'
                    ]],
                    source=str(settings.DATA_DIR / "raw"),
                    name="raw_turbine_data"
                ),
                context="training"
            )
            
            # Dataset de salida
            mlflow.log_input(
                mlflow.data.from_pandas(
                    summary_df[[
                        'filename',
                        'category',
                        'processed_rows',
                        'reduction_pct',
                        'status'
                    ]],
                    source=str(settings.DATA_DIR / "processed"),
                    name="processed_turbine_data"
                ),
                context="training"
            )
        except Exception as e:
            print(f"⚠️ No se pudieron registrar datasets: {e}")
        
        # ===== RESUMEN FINAL =====
        print(f"\n{'='*70}")
        print("✅ PROCESAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"\n📊 Resumen local: {summary_path_local}")
        print(f"📦 Artefactos MLflow: {mlflow_artifacts_dir}")
        print("\n📈 Estadísticas por categoría:")
        
        # Mostrar estadísticas
        success_df = summary_df[summary_df['status'] == 'success']
        if len(success_df) > 0:
            print(
                success_df.groupby('category')[[
                    'processed_rows',
                    'reduction_pct'
                ]].mean()
            )
        
        # Mostrar errores
        error_df = summary_df[summary_df['status'] == 'error']
        if len(error_df) > 0:
            print(f"\n⚠️ Archivos con errores: {len(error_df)}")
            print(error_df[['filename', 'category', 'error_message']])
        
        print(f"\n🔗 MLflow Run ID: {run.info.run_id}")
        print(
            f"🌐 Ver en: http://localhost:5000/#/experiments/"
            f"{run.info.experiment_id}/runs/{run.info.run_id}"
        )
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
