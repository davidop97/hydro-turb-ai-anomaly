from pathlib import Path

from src.models.anomaly_detector import AnomalyDetector
from src.models.vibration_severity_checker import check_vibration_severity
from src.preprocessing.pipeline import TurbineDataPipeline


class TurbinePredictor:
    def __init__(self):
        models_dir = Path(__file__).parent.parent.parent / "models" / "trained"
        
        residuals_model_path = sorted(models_dir.glob("residuals_*.pkl"))[-1]
        classifier_path = models_dir / "classifier_best.pkl"
        
        self.detector = AnomalyDetector.load(str(residuals_model_path), str(classifier_path))

    def predict(self, file_path: str, machine_type: str = "Francis horizontal") -> dict:
        # 1. Preprocesar
        pipeline = TurbineDataPipeline()
        df_processed, nominal_speed, max_values = pipeline.fit_transform(file_path)
        
        # 2. Detectar anomalías (AHORA con sensor_data)
        detection_result = self.detector.predict(df_processed)
        
        # 3. Evaluar severidad
        severity_results = check_vibration_severity(
            max_values=max_values,
            machine_type=machine_type,
            severity_level=None
        )
        
        # 4. Respuesta
        prediction_label = detection_result["classification"]
        p_desalineacion = detection_result["p_desalineacion_global"]
        confidence = max(p_desalineacion, 1 - p_desalineacion)
        
        return {
            "prediction": prediction_label,
            "confidence": float(confidence),
            "probabilities": {
                "desbalanceo": float(1 - p_desalineacion),
                "desalineacion": float(p_desalineacion)
            },
            "metadata": {
                "nominal_speed": float(nominal_speed),
                "samples_analyzed": len(df_processed),
                "n_anomalies": detection_result["n_anomalies"],
                "sensors": detection_result["sensors"]
            },
            "severity": severity_results,
            "max_values": max_values,
            "sensor_data": detection_result["sensor_data"],  # ✅ NUEVO
            "kph": detection_result["kph"]  # ✅ NUEVO
        }
