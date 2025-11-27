"""
Módulo de validación y evaluación de modelos de residuos.
Incluye train/test split, métricas y visualización.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.models.residuals_model import DataResidualsProcessor


class ModelValidator:
    """
    Validador para modelos de residuos.
    
    Realiza:
    - Train/test split
    - Validación cruzada
    - Cálculo de métricas
    - Comparación de modelos
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Inicializa validador.
        
        Args:
            test_size: Proporción test (0.2 = 20%)
            random_state: Seed para reproducibilidad
        """
        self.test_size = test_size
        self.random_state = random_state
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.validation_results: dict = {}
    
    
    def split_data(
        self,
        df: pd.DataFrame,
        stratify_col: Optional[str] = None,
        shuffle: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide datos en train/test.
        
        Args:
            df: DataFrame completo
            stratify_col: Columna para stratificación (opcional)
            shuffle: Mezclar datos
            
        Returns:
            Tupla (df_train, df_test)
        """
        stratify = (
            df[stratify_col] if stratify_col and stratify_col in df.columns
            else None
        )
        
        self.train_data, self.test_data = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=shuffle,
            stratify=stratify
        )
        
        print("\n📊 TRAIN/TEST SPLIT")
        print(f"{'='*60}")
        print(f"Total: {len(df)} muestras")
        print(f"Train: {len(self.train_data)} ({100*(1-self.test_size):.0f}%)")
        print(f"Test: {len(self.test_data)} ({100*self.test_size:.0f}%)")
        print(f"{'='*60}\n")
        
        return self.train_data, self.test_data
    
    
    def train_and_evaluate(
        self,
        degree: int = 3,
        speed_col: str = 'KPH',
        threshold_percentile: float = 90.0
    ) -> dict:
        """
        Entrena modelo y evalúa en train/test.
        
        Args:
            degree: Grado polinomio
            speed_col: Columna velocidad
            threshold_percentile: Percentil para umbral
            
        Returns:
            Dict con resultados de validación
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError(
                "Ejecutar split_data() primero"
            )
        
        # Entrenar
        print("🔧 ENTRENANDO MODELO...")
        model = DataResidualsProcessor(
            degree=degree,
            speed_col=speed_col,
            threshold_percentile=threshold_percentile
        )
        model.fit(df=self.train_data, verbose=True)
        
        # Evaluar en TRAIN
        print("\n📈 EVALUANDO EN TRAIN...")
        train_residuals, _, _, train_originals, train_preds = (
            model.calculate_residuals_global(df=self.train_data)
        )
        
        train_mae = mean_absolute_error(
            train_originals.flatten(),
            train_preds.flatten()
        )
        train_rmse = np.sqrt(
            mean_squared_error(
                train_originals.flatten(),
                train_preds.flatten()
            )
        )
        train_r2 = r2_score(
            train_originals.flatten(),
            train_preds.flatten()
        )
        
        print(f"Train MAE: {train_mae:.6f}")
        print(f"Train RMSE: {train_rmse:.6f}")
        print(f"Train R²: {train_r2:.4f}")
        
        # Evaluar en TEST
        print("\n📈 EVALUANDO EN TEST...")
        test_residuals, _, _, test_originals, test_preds = (
            model.calculate_residuals_global(df=self.test_data)
        )
        
        test_mae = mean_absolute_error(
            test_originals.flatten(),
            test_preds.flatten()
        )
        test_rmse = np.sqrt(
            mean_squared_error(
                test_originals.flatten(),
                test_preds.flatten()
            )
        )
        test_r2 = r2_score(
            test_originals.flatten(),
            test_preds.flatten()
        )
        
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Detección de anomalías
        train_anomalies = model.predict_anomaly(df=self.train_data)
        test_anomalies = model.predict_anomaly(df=self.test_data)
        
        print("\n🚨 DETECCIÓN DE ANOMALÍAS")
        print(f"{'='*60}")
        print(f"Train: {train_anomalies['n_anomalies']} "
              f"({train_anomalies['pct_anomalies']:.1f}%)")
        print(f"Test: {test_anomalies['n_anomalies']} "
              f"({test_anomalies['pct_anomalies']:.1f}%)")
        print(f"{'='*60}\n")
        
        # Compilar resultados
        self.validation_results = {
            'model': model,
            'train': {
                'mae': float(train_mae),
                'rmse': float(train_rmse),
                'r2': float(train_r2),
                'n_samples': len(self.train_data),
                'n_anomalies': train_anomalies['n_anomalies'],
                'pct_anomalies': train_anomalies['pct_anomalies']
            },
            'test': {
                'mae': float(test_mae),
                'rmse': float(test_rmse),
                'r2': float(test_r2),
                'n_samples': len(self.test_data),
                'n_anomalies': test_anomalies['n_anomalies'],
                'pct_anomalies': test_anomalies['pct_anomalies']
            },
            'hyperparameters': {
                'polynomial_degree': degree,
                'speed_column': speed_col,
                'threshold_percentile': threshold_percentile,
                'test_size': self.test_size
            }
        }
        
        return self.validation_results
    
    
    def get_summary(self) -> dict:
        """Retorna resumen de validación."""
        if not self.validation_results:
            raise ValueError("Ejecutar train_and_evaluate() primero")
        
        return self.validation_results
