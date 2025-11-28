"""
Workflow: Entrenar clasificador usando FEATURES estadísticas.
Maneja archivos heterogéneos correctamente.
"""
import mlflow
import numpy as np

from configs.config import settings
from src.models.classifier import AnomalyClassifier
from src.models.residuals_model import DataResidualsProcessor


def main():
    print("\n" + "="*70)
    print("🤖 ENTRENAMIENTO CLASIFICADOR (FEATURES ESTADÍSTICAS)")
    print("="*70 + "\n")
    
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        
        # 1. Cargar modelo de residuos BASE
        # (Usamos el mejor modelo de residuos para generar los residuos)
        project_models_dir = settings.ROOT_DIR / "models" / "trained"
        model_path = sorted(project_models_dir.glob("residuals_*.pkl"))[-1]
        residuals_model = DataResidualsProcessor.load(str(model_path))
        
        print(f"✓ Modelo residuos base: {model_path.name}\n")
        
        # 2. Preparar datos (Features X, Labels y)
        X_features = []
        y_labels = []
        
        processed_dir = settings.DATA_DIR / "processed"
        
        # --- PROCESAR DESBALANCEO (Clase 0) ---
        files_0 = list((processed_dir / "imbalance").glob("*.csv"))
        print(f"📂 Procesando {len(files_0)} archivos de Desbalanceo...")
        
        for f in files_0:
            try:
                # calculate_residuals_global maneja internamente:
                # - Diferentes nombres de sensores en cada archivo
                # - NaN en datos de velocidad o sensores
                # - Filtrado de columnas no numéricas
                res_matrix, _, _, _, _ = (
                    residuals_model.calculate_residuals_global(
                        ruta_archivo=str(f)
                    )
                )
                
                # EXTRAER FEATURE (Media absoluta)
                # Esto convierte (n_samples, n_sensors) -> (1, 1)
                mean_res = np.mean(np.abs(res_matrix))
                
                X_features.append([mean_res])
                y_labels.append(0)
            except Exception as e:
                print(f"  ⚠️ Error {f.name}: {e}")
        
        # --- PROCESAR DESALINEACIÓN (Clase 1) ---
        files_1 = list((processed_dir / "misalignment").glob("*.csv"))
        print(f"📂 Procesando {len(files_1)} archivos de Desalineación...")
        
        for f in files_1:
            try:
                res_matrix, _, _, _, _ = (
                    residuals_model.calculate_residuals_global(
                        ruta_archivo=str(f)
                    )
                )
                
                # Feature
                mean_res = np.mean(np.abs(res_matrix))
                
                X_features.append([mean_res])
                y_labels.append(1)
            except Exception as e:
                print(f"  ⚠️ Error {f.name}: {e}")
        
        # Convertir a arrays
        X = np.array(X_features)
        y = np.array(y_labels)
        
        print("\n📊 Dataset final:")
        print(f"   X shape: {X.shape} (Archivos, Features)")
        print(f"   y shape: {y.shape}")
        print(f"   Desbalanceo (0): {np.sum(y==0)}")
        print(f"   Desalineación (1): {np.sum(y==1)}\n")
        
        # 3. Entrenar Clasificador
        classifier = AnomalyClassifier(method='logistic')
        classifier.fit(X, y, verbose=True)
        
        # 4. Evaluar
        metrics = classifier.evaluate(X, y)
        print("\n📈 Métricas:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # 5. Guardar
        clf_path = project_models_dir / "classifier_logistic.pkl"
        classifier.save(str(clf_path))
        print(f"\n💾 Guardado en: {clf_path}")


if __name__ == "__main__":
    main()
