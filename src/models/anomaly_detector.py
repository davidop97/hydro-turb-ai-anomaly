"""
Detector de anomalías integrado.
Combina modelo de residuos + clasificador.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.models.classifier import AnomalyClassifier
from src.models.residuals_model import DataResidualsProcessor


class AnomalyDetector:
    """
    Pipeline completo: residuos → clasificación → predicción.
    """
    
    def __init__(
        self,
        residuals_model: DataResidualsProcessor,
        classifier: AnomalyClassifier
    ):
        """
        Args:
            residuals_model: Modelo de residuos entrenado
            classifier: Clasificador entrenado
        """
        self.residuals_model = residuals_model
        self.classifier = classifier
    
    
    def predict(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Predicción completa en nuevos datos.
        
        Args:
            df: DataFrame con datos nuevos
            
        Returns:
            Dict con predicciones y probabilidades
        """
        # Calcular residuos
        residuals, sensors, kph, originals, predictions = (
            self.residuals_model.calculate_residuals_global(df=df)
        )
        
        # Clasificar
        p_desbalanceo, p_desalineacion = (
            self.classifier.predict_probability(residuals)
        )
        
        # Etiquetas
        labels = self.classifier.predict_label(residuals)
        
        # Estadísticas
        p_desbalanceo_global = p_desbalanceo.mean()
        p_desalineacion_global = p_desalineacion.mean()
        
        anomaly_score = np.abs(residuals).mean(axis=1)
        n_anomalies = (anomaly_score > self.residuals_model.threshold).sum()
        
        return {
            'residuals': residuals,
            'predictions': predictions,
            'originals': originals,
            'sensors': sensors,
            'kph': kph.flatten(),
            'p_desbalanceo_por_muestra': p_desbalanceo,
            'p_desalineacion_por_muestra': p_desalineacion,
            'p_desbalanceo_global': float(p_desbalanceo_global),
            'p_desalineacion_global': float(p_desalineacion_global),
            'labels_por_muestra': labels,
            'anomaly_scores': anomaly_score,
            'n_anomalies': int(n_anomalies),
            'classification': (
                'DESALINEACIÓN' if p_desalineacion_global > 0.5
                else 'DESBALANCEO'
            )
        }
    
    
    def save(self, model_path: str, classifier_path: str) -> None:
        """Guarda ambos componentes."""
        self.residuals_model.save(model_path)
        self.classifier.save(classifier_path)
    
    
    @classmethod
    def load(cls, model_path: str, classifier_path: str) -> 'AnomalyDetector':
        """Carga ambos componentes."""
        model = DataResidualsProcessor.load(model_path)
        classifier = AnomalyClassifier.load(classifier_path)
        return cls(model, classifier)
