"""
AnomalyDetector: Calcula residuos + devuelve datos para graficar.
"""

import numpy as np
import pandas as pd

from src.models.classifier import AnomalyClassifier
from src.models.residuals_model import DataResidualsProcessor


class AnomalyDetector:
    def __init__(self, residuals_model: DataResidualsProcessor, classifier: AnomalyClassifier):
        self.residuals_model = residuals_model
        self.classifier = classifier

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predicción completa con datos para graficar.
        """
        # 1. Calcular residuos
        residuals, sensors, kph, originals, predictions = (
            self.residuals_model.calculate_residuals_global(df=df)
        )
        
        # 2. Feature global
        from workflows.train_classifier import extract_statistical_features
        features = extract_statistical_features(residuals)
        
        # 3. Clasificación
        p_desalineacion = self.classifier.predict_proba(features)[0]
        p_desbalanceo = 1.0 - p_desalineacion
        label_global = int(self.classifier.predict(features)[0])
        
        # 4. Anomaly scores por muestra
        anomaly_score = np.mean(np.abs(residuals), axis=1)
        n_anomalies = int(np.sum(anomaly_score > 0.1))
        
        # 5. ✅ NUEVO: Datos por sensor para graficar
        sensor_data = {}
        for i, sensor in enumerate(sensors):
            sensor_data[sensor] = {
                "original": originals[:, i],
                "predicted": predictions[:, i],
                "residual": residuals[:, i],
                "abs_residual": np.abs(residuals[:, i]),
                "mean_residual": np.mean(np.abs(residuals[:, i]))
            }
        
        return {
            "residuals": residuals,
            "predictions": predictions,
            "originals": originals,
            "sensors": sensors,
            "kph": kph.flatten(),
            "p_desbalanceo_global": float(p_desbalanceo),
            "p_desalineacion_global": float(p_desalineacion),
            "label_global": label_global,
            "anomaly_scores": anomaly_score,
            "n_anomalies": n_anomalies,
            "classification": "DESALINEACIÓN" if label_global == 1 else "DESBALANCEO",
            "sensor_data": sensor_data  # ✅ NUEVO
        }

    def save(self, model_path: str, classifier_path: str) -> None:
        self.residuals_model.save(model_path)
        self.classifier.save(classifier_path)

    @classmethod
    def load(cls, model_path: str, classifier_path: str) -> "AnomalyDetector":
        model = DataResidualsProcessor.load(model_path)
        classifier = AnomalyClassifier.load(classifier_path)
        return cls(model, classifier)
