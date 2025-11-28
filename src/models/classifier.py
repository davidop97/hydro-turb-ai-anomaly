"""
Clasificador basado en FEATURES estadísticas de residuos.
Resuelve el problema de dimensiones variables.
"""

from typing import Dict, Literal, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.mixture import GaussianMixture


class AnomalyClassifier:
    """
    Clasificador que usa ESTADÍSTICAS de residuos (features fijas).
    Independiente del número de sensores.
    """

    def __init__(
        self,
        method: Literal["linear", "logistic", "gmm"] = "logistic",
        lower_percentile: float = 25.0,
        upper_percentile: float = 75.0,
    ):
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        self.lower_threshold: Optional[float] = None
        self.upper_threshold: Optional[float] = None
        self.model: Optional[object] = None
        self.metrics: Dict = {}

    def fit(
        self,
        X_features: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "AnomalyClassifier":
        """
        Entrena usando features extraídas.

        Args:
            X_features: Matriz (n_archivos, n_features)
            y: Etiquetas (n_archivos,)
        """
        assert X_features.shape[0] == len(y), (
            f"X shape {X_features.shape} != y shape {y.shape}"
        )

        if self.method == "linear":
            # Usar SOLO la primera feature (media absoluta) para umbrales lineales
            feature_values = X_features[:, 0]

            desbalanceo_values = feature_values[y == 0]
            desalineacion_values = feature_values[y == 1]

            self.lower_threshold = float(
                np.percentile(desbalanceo_values, self.lower_percentile)
            )
            self.upper_threshold = float(
                np.percentile(desalineacion_values, self.upper_percentile)
            )

            if verbose:
                print("🔧 UMBRALES LINEALES (feature: Media)")
                print(f"   Bajo (Desbalanceo): {self.lower_threshold:.6f}")
                print(f"   Alto (Desalineación): {self.upper_threshold:.6f}")

        elif self.method == "logistic":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver="lbfgs",
            )
            self.model.fit(X_features, y)

            if verbose:
                print("🔧 LOGISTIC REGRESSION")
                print(
                    f"   Coefs (primeras 3): "
                    f"{self.model.coef_[0][:3]}..."
                )
                print(f"   Intercept: {self.model.intercept_[0]:.6f}")

        elif self.method == "gmm":
            # IMPORTANTE: reducir complejidad del modelo
            self.model = GaussianMixture(
                n_components=2,
                random_state=42,
                covariance_type="diag",  # antes 'full'
                reg_covar=1e-6,
            )
            self.model.fit(X_features)

            means_sum = self.model.means_.sum(axis=1)
            order = np.argsort(means_sum)

            if verbose:
                print("🔧 GAUSSIAN MIXTURE MODEL (diag)")
                print(f"   Componente 1 (Bajo): media_sum={means_sum[order[0]]:.6f}")
                print(f"   Componente 2 (Alto): media_sum={means_sum[order[1]]:.6f}")

        return self

    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        """Predice probabilidad de Desalineación (Clase 1)."""
        if self.method == "linear":
            vals = X_features[:, 0]
            probs = np.zeros(len(vals), dtype=float)

            mask_low = vals <= self.lower_threshold
            mask_high = vals >= self.upper_threshold
            mask_mid = (vals > self.lower_threshold) & (
                vals < self.upper_threshold
            )

            probs[mask_low] = 0.0
            probs[mask_high] = 1.0
            probs[mask_mid] = (
                (vals[mask_mid] - self.lower_threshold)
                / (self.upper_threshold - self.lower_threshold)
            )

            return probs

        elif self.method == "logistic":
            return self.model.predict_proba(X_features)[:, 1]

        elif self.method == "gmm":
            means_sum = self.model.means_.sum(axis=1)
            idx_class1 = int(np.argmax(means_sum))
            return self.model.predict_proba(X_features)[:, idx_class1]

    def predict(self, X_features: np.ndarray) -> np.ndarray:
        """Predice etiqueta binaria."""
        probs = self.predict_proba(X_features)
        return (probs >= 0.5).astype(int)

    def evaluate(self, X_features: np.ndarray, y: np.ndarray) -> Dict:
        """Evalúa métricas."""
        y_pred = self.predict(X_features)
        y_proba = self.predict_proba(X_features)

        cm = confusion_matrix(y, y_pred)

        try:
            auc = roc_auc_score(y, y_proba)
        except ValueError:
            auc = 0.0

        accuracy = float((y_pred == y).mean())

        return {
            "confusion_matrix": cm.tolist(),
            "auc_roc": float(auc),
            "accuracy": accuracy,
        }

    def save(self, path: str):
        """Guarda clasificador."""
        joblib.dump(self, path)
        print(f"✓ Guardado: {path}")

    @classmethod
    def load(cls, path: str):
        """Carga clasificador."""
        return joblib.load(path)
