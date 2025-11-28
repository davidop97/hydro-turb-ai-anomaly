"""
Clasificador basado en FEATURES estadísticas de residuos.
Resuelve el problema de dimensiones variables.
"""

from typing import Dict, Literal, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.mixture import GaussianMixture


class AnomalyClassifier:
    """
    Clasificador que usa ESTADÍSTICAS de residuos (features fijas).
    Independiente del número de sensores.
    """
    
    def __init__(
        self,
        method: Literal['linear', 'logistic', 'gmm'] = 'logistic',
        lower_percentile: float = 25.0,
        upper_percentile: float = 75.0
    ):
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        self.lower_threshold: Optional[float] = None
        self.upper_threshold: Optional[float] = None
        self.model: Optional[object] = None
        self.metrics: Dict = {}
    
    
    def extract_features(self, residuals: np.ndarray) -> np.ndarray:
        """
        Convierte residuos crudos (n_muestras, n_sensores)
        a vector de características FIJO (1, 1).
        
        Feature principal: Media absoluta de los residuos.
        (Podríamos añadir max, std, etc., pero empezamos simple).
        """
        # Aplanar y tomar valor absoluto
        abs_resid = np.abs(residuals)
        
        # Feature: Media global de todos los residuos en el archivo
        # Esto da un solo escalar que representa "cuánto residuo hay"
        mean_residual = np.mean(abs_resid)
        
        # Retornar como matriz (1, 1)
        return np.array([[mean_residual]])
    
    
    def fit(
        self,
        X_features: np.ndarray,
        y_labels: np.ndarray,
        verbose: bool = True
    ) -> 'AnomalyClassifier':
        """
        Entrena usando features extraídas (NO residuos crudos).
        
        Args:
            X_features: Matriz (n_archivos, n_features)
            y_labels: Etiquetas (n_archivos,)
        """
        if self.method == 'linear':
            # Para lineal, usamos percentiles de las features
            feature_values = X_features.flatten()
            self.lower_threshold = np.percentile(
                feature_values[y_labels == 0],
                self.lower_percentile
            )
            self.upper_threshold = np.percentile(
                feature_values[y_labels == 1],
                self.upper_percentile
            )
            
            if verbose:
                print("🔧 UMBRALES LINEALES")
                print(f"   Bajo (Desbalanceo): {self.lower_threshold:.6f}")
                print(f"   Alto (Desalineación): {self.upper_threshold:.6f}")
        
        elif self.method == 'logistic':
            self.model = LogisticRegression(random_state=42)
            self.model.fit(X_features, y_labels)
            
            if verbose:
                print("🔧 LOGISTIC REGRESSION")
                print(f"   Coef: {self.model.coef_[0][0]:.6f}")
                print(f"   Intercept: {self.model.intercept_[0]:.6f}")
        
        elif self.method == 'gmm':
            self.model = GaussianMixture(n_components=2, random_state=42)
            self.model.fit(X_features)
            
            means = self.model.means_.flatten()
            means_sorted = np.sort(means)
            self.lower_threshold = means_sorted[0]
            self.upper_threshold = means_sorted[-1]
            
            if verbose:
                print("🔧 GMM")
                print(f"   Componente 1 (Bajo): {means_sorted[0]:.6f}")
                print(f"   Componente 2 (Alto): {means_sorted[1]:.6f}")
        
        return self
    
    
    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        """Predice probabilidad de Desalineación (Clase 1)."""
        if self.method == 'linear':
            vals = X_features.flatten()
            probs = np.zeros_like(vals)
            
            # Interpolación
            mask = (vals > self.lower_threshold) & (vals < self.upper_threshold)
            probs[mask] = (
                (vals[mask] - self.lower_threshold) /
                (self.upper_threshold - self.lower_threshold)
            )
            probs[vals >= self.upper_threshold] = 1.0
            return probs
            
        elif self.method == 'logistic':
            return self.model.predict_proba(X_features)[:, 1]
            
        elif self.method == 'gmm':
            # En GMM, asumimos la componente con mayor media es Clase 1
            means = self.model.means_.flatten()
            idx_class1 = np.argmax(means)
            return self.model.predict_proba(X_features)[:, idx_class1]
    
    
    def predict(self, X_features: np.ndarray) -> np.ndarray:
        """Predice etiqueta binaria."""
        probs = self.predict_proba(X_features)
        return (probs >= 0.5).astype(int)
    
    
    def evaluate(self, X_features: np.ndarray, y_labels: np.ndarray) -> Dict:
        """Evalúa métricas."""
        y_pred = self.predict(X_features)
        y_proba = self.predict_proba(X_features)
        
        cm = confusion_matrix(y_labels, y_pred)
        try:
            auc = roc_auc_score(y_labels, y_proba)
        except ValueError:
            auc = 0.0  # Si solo hay una clase en test
            
        report = classification_report(
            y_labels, y_pred, output_dict=True, zero_division=0
        )
        
        return {
            'confusion_matrix': cm.tolist(),
            'auc_roc': float(auc),
            'accuracy': report['accuracy'],
            'report': report
        }
    
    def save(self, path: str):
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
