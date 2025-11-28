"""
Módulo de modelado de residuos para detección de anomalías.

Este módulo implementa un enfoque de regresión polinómica para ajustar
el comportamiento esperado de los sensores como función de la velocidad.
Los residuos resultantes se usan para identificar anomalías (desbalanceo/desalineación).

Versión: 2.0.0
"""

from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class  DataResidualsProcessor:
    """
    Procesador de residuos basado en regresión polinómica.
    
    Flujo:
    1. fit() - Entrena modelos polinómicos usando datos de desbalanceo
    2. calculate_residuals_global() - Calcula residuos en datos nuevos
    3. get_anomaly_score() - Calcula puntuación de anomalía
    4. save/load() - Persistencia del modelo
    
    Atributos:
        degree: Grado del polinomio (recomendado: 2-4)
        speed_col: Nombre columna velocidad (default: 'KPH')
        threshold_percentile: Percentil para umbral (default: 90)
        poly: Transformador PolynomialFeatures entrenado
        models: Dict de modelos LinearRegression por sensor
        displacement_cols: Lista de sensores de desplazamiento
        df_train: DataFrame de entrenamiento
        threshold: Umbral calculado en fit()
        metrics: Dict con métricas de validación
    """
    
    def __init__(
        self,
        degree: int = 3,
        speed_col: str = 'KPH',
        threshold_percentile: float = 90.0
    ):
        """
        Inicializa el procesador.
        
        Args:
            degree: Grado polinomio (2-4 recomendado)
            speed_col: Nombre columna velocidad
            threshold_percentile: Percentil para umbral de anomalía
        """
        self.degree = degree
        self.speed_col = speed_col
        self.threshold_percentile = threshold_percentile
        
        # Inicializar componentes
        self.poly = PolynomialFeatures(degree=self.degree)
        self.models: dict[str, LinearRegression] = {}
        self.displacement_cols: List[str] = []
        self.df_train: Optional[pd.DataFrame] = None
        self.threshold: Optional[float] = None
        
        # Métricas de validación
        self.metrics: dict = {
            'train': {},
            'test': {},
            'cv': {}
        }
        
        # Metadata
        self.best_sensor: Optional[str] = None
        self.best_model: Optional[LinearRegression] = None
    
    
    def _load_data(
        self,
        ruta_archivo: Optional[str] = None,
        df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Carga datos desde DataFrame o archivo CSV.
        
        Args:
            ruta_archivo: Ruta al archivo CSV
            df: DataFrame directamente
            
        Returns:
            DataFrame cargado y validado
            
        Raises:
            ValueError: Si no se proporciona ruta ni DataFrame
        """
        if df is not None:
            return df.copy()
        
        elif ruta_archivo is not None:
            try:
                df_loaded = pd.read_csv(ruta_archivo, delimiter=",")
                return df_loaded
            except Exception as e:
                raise ValueError(
                    f"No se pudo leer archivo {ruta_archivo}: {e}"
                )
        
        else:
            raise ValueError(
                "Proporcionar DataFrame o ruta de archivo"
            )
    
    
    def _get_displacement_cols(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        """
        Extrae columnas de sensores (excluye Fecha y velocidad).
        
        Args:
            df: DataFrame
            
        Returns:
            Lista de nombres de sensores
            
        Raises:
            ValueError: Si no hay sensores válidos
        """
        columnas = [
            col for col in df.columns
            if col not in ['Fecha', self.speed_col, 'KPH_abs']
        ]
        
        if not columnas:
            raise ValueError(
                "No se encontraron columnas de sensores válidas"
            )
        
        return columnas
    
    
    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        ruta_archivo: Optional[str] = None,
        verbose: bool = True
    ) -> 'DataResidualsProcessor':
        """
        Ajusta modelos polinómicos usando datos de desbalanceo.
        
        Cambio: Ahora maneja NaNs automáticamente.
        """
        # Cargar datos
        self.df_train = self._load_data(ruta_archivo, df)
        self.displacement_cols = self._get_displacement_cols(self.df_train)
        
        # LIMPIEZA: Eliminar NaN
        cols_to_check = [self.speed_col] + self.displacement_cols
        initial_len = len(self.df_train)
        self.df_train = self.df_train.dropna(subset=cols_to_check)
        final_len = len(self.df_train)
        
        if verbose:
            print("\n🔧 ENTRENANDO MODELO DE RESIDUOS")
            print(f"{'='*60}")
            print(f"Grado polinomio: {self.degree}")
            print(f"Sensores: {self.displacement_cols}")
            print(f"Muestras originales: {initial_len:,}")
            print(f"Muestras después de limpiar NaNs: {final_len:,} "
                f"(-{initial_len - final_len})")
            print(f"{'='*60}\n")
        
        if final_len < 10:
            raise ValueError(
                f"Muestras insuficientes después de limpiar: {final_len}"
            )
        
        # Preparar datos
        kph = self.df_train[self.speed_col].to_numpy().reshape(-1, 1)
        X_poly = self.poly.fit_transform(kph)
        
        if verbose:
            print(f"Características polinómicas: {X_poly.shape[1]}\n")
        
        # Entrenar modelo por sensor
        if verbose:
            print("Entrenando modelos por sensor:")
        
        for col in self.displacement_cols:
            y = self.df_train[col].values
            model = LinearRegression()
            model.fit(X_poly, y)
            self.models[col] = model
            
            if verbose:
                score = model.score(X_poly, y)
                print(f"  ✓ {col}: R² = {score:.4f}")
        
        # Calcular residuos
        residuals = []
        for col in self.displacement_cols:
            y = self.df_train[col].values
            y_pred = self.models[col].predict(X_poly)
            resid = y - y_pred
            residuals.append(resid)
        
        residuals_matrix = np.column_stack(residuals)
        mean_abs_resid = np.mean(np.abs(residuals_matrix), axis=1)
        
        # Umbral
        self.threshold = float(
            np.percentile(mean_abs_resid, self.threshold_percentile)
        )
        
        if verbose:
            print("\n📊 UMBRAL CALCULADO")
            print(f"{'='*60}")
            print(f"Percentil: {self.threshold_percentile}")
            print(f"Umbral (MAE): {self.threshold:.6f}")
            print(f"Media (MAE): {mean_abs_resid.mean():.6f}")
            print(f"Std (MAE): {mean_abs_resid.std():.6f}")
            print(f"{'='*60}\n")
        
        # Referencia
        self.best_sensor = self.displacement_cols[0]
        self.best_model = self.models[self.best_sensor]
        
        # Métricas
        self.metrics['train'] = {
            'n_samples': len(self.df_train),
            'n_sensors': len(self.displacement_cols),
            'mean_abs_residuals': float(mean_abs_resid.mean()),
            'std_abs_residuals': float(mean_abs_resid.std()),
            'threshold': self.threshold,
            'polynomial_degree': self.degree,
            'n_features': X_poly.shape[1]
        }
        
        return self

    
    
    def calculate_residuals_global(
        self,
        df: Optional[pd.DataFrame] = None,
        ruta_archivo: Optional[str] = None,
        return_predictions: bool = True,
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Calcula residuos en datos nuevos usando modelo entrenado.
        
        Maneja internamente archivos heterogéneos:
        - Usa el mejor modelo entrenado (primer sensor) como referencia
        - Lo aplica a TODOS los sensores nuevos (incluso si tienen nombres diferentes)
        - Filtra columnas no numéricas automáticamente
        - Maneja NaN de forma segura
        
        Args:
            df: DataFrame con datos nuevos
            ruta_archivo: Ruta a archivo CSV
            return_predictions: Incluir predicciones
            verbose: Mostrar información del modelo utilizado
            
        Returns:
            Tupla (residuals, sensor_names, velocities, originals, predictions)
            
        Raises:
            ValueError: Si el modelo no está entrenado o no hay sensores válidos
        """
        if self.best_model is None:
            raise ValueError(
                "Modelo no entrenado. Ejecutar fit() primero."
            )
        
        # Cargar datos nuevos
        data = self._load_data(ruta_archivo, df)
        
        # Verificar que existe la columna de velocidad
        if self.speed_col not in data.columns:
            raise ValueError(
                f"Columna de velocidad '{self.speed_col}' no encontrada en datos"
            )
        
        # Sensores nuevos (columnas numéricas excluyendo metadata)
        excluded_cols = {'Fecha', self.speed_col, 'KPH_abs'}
        new_sensor_cols = [
            col for col in data.columns
            if col not in excluded_cols and pd.api.types.is_numeric_dtype(data[col])
        ]
        
        if not new_sensor_cols:
            raise ValueError(
                "No se encontraron columnas de sensores válidas en los datos"
            )
        
        if verbose:
            print(f"Usando modelo de referencia del sensor: {self.best_sensor}")
            print(f"Aplicando a {len(new_sensor_cols)} sensores: {new_sensor_cols}")
        
        # Identificar filas válidas (sin NaN en velocidad y sensores)
        cols_to_check = [self.speed_col] + new_sensor_cols
        valid_mask = data[cols_to_check].notna().all(axis=1)
        data_clean = data.loc[valid_mask].copy()
        
        if len(data_clean) == 0:
            raise ValueError(
                "No hay filas válidas después de eliminar NaN"
            )
        
        # Transformar velocidad
        kph = data_clean[self.speed_col].to_numpy().reshape(-1, 1)
        X_poly = self.poly.transform(kph)
        
        # Aplicar modelo de referencia a cada sensor
        residuals_list = []
        predictions_list = []
        original_list = []
        
        for col in new_sensor_cols:
            y_real = data_clean[col].values
            y_pred = self.best_model.predict(X_poly)
            resid = y_real - y_pred
            
            residuals_list.append(resid)
            predictions_list.append(y_pred)
            original_list.append(y_real)
        
        # Armar matrices
        residuals_matrix = np.column_stack(residuals_list)
        predictions_matrix = np.column_stack(predictions_list)
        original_matrix = np.column_stack(original_list)
        
        if return_predictions:
            return (
                residuals_matrix,
                new_sensor_cols,
                kph,
                original_matrix,
                predictions_matrix
            )
        else:
            return (
                residuals_matrix,
                new_sensor_cols,
                kph,
                original_matrix,
                None
            )
    
    
    def get_anomaly_score(
        self,
        residuals: np.ndarray
    ) -> np.ndarray:
        """
        Calcula puntuación de anomalía como MAE por muestra.
        
        Args:
            residuals: Matriz (n_samples, n_sensors)
            
        Returns:
            Array 1D con puntuación por muestra
        """
        if len(residuals.shape) != 2:
            raise ValueError("Residuos debe ser matriz 2D")
        
        # MAE por muestra
        anomaly_score = np.mean(np.abs(residuals), axis=1)
        
        return anomaly_score
    
    
    def predict_anomaly(
        self,
        df: Optional[pd.DataFrame] = None,
        ruta_archivo: Optional[str] = None
    ) -> dict:
        """
        Predice si datos son normales o anómalos.
        
        Args:
            df: DataFrame con datos nuevos
            ruta_archivo: Ruta a archivo
            
        Returns:
            Dict con predicciones y scores
        """
        residuals, sensors, kph, originals, predictions = (
            self.calculate_residuals_global(df, ruta_archivo)
        )
        
        anomaly_scores = self.get_anomaly_score(residuals)
        is_anomaly = anomaly_scores > self.threshold
        
        return {
            'anomaly_scores': anomaly_scores,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold,
            'n_anomalies': int(np.sum(is_anomaly)),
            'pct_anomalies': float(
                (np.sum(is_anomaly) / len(anomaly_scores)) * 100
            ),
            'residuals': residuals,
            'sensors': sensors,
            'velocities': kph.flatten()
        }
    
    
    def stack_residuals(
        self,
        residuals: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Apila residuos 2D a vector 1D.
        
        Args:
            residuals: Matriz o lista de matrices
            
        Returns:
            Array 1D
        """
        if isinstance(residuals, list):
            if not residuals:
                raise ValueError("Lista vacía")
            return np.concatenate([r.flatten() for r in residuals])
        
        elif isinstance(residuals, np.ndarray):
            if len(residuals.shape) != 2:
                raise ValueError("Array debe ser 2D")
            return residuals.flatten()
        
        else:
            raise ValueError("Entrada debe ser array o lista")
    
    
    def save(self, file_path: str) -> None:
        """
        Guarda modelo entrenado.
        
        Args:
            file_path: Ruta de destino (.pkl)
        """
        joblib.dump(self, file_path)
        print(f"✓ Modelo guardado: {file_path}")
    
    
    @classmethod
    def load(cls, file_path: str) -> 'DataResidualsProcessor':
        """
        Carga modelo entrenado.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Instancia cargada
        """
        model = joblib.load(file_path)
        print(f"✓ Modelo cargado: {file_path}")
        return model
    
    
    def get_model_info(self) -> dict:
        """
        Retorna información del modelo para documentación.
        
        Returns:
            Dict con parámetros y métricas
        """
        return {
            'polynomial_degree': self.degree,
            'speed_column': self.speed_col,
            'threshold_percentile': self.threshold_percentile,
            'threshold_value': self.threshold,
            'displacement_columns': self.displacement_cols,
            'best_sensor': self.best_sensor,
            'n_training_samples': (
                len(self.df_train) if self.df_train is not None else None
            ),
            'metrics': self.metrics,
            'n_polynomial_features': (
                self.poly.n_features_in_ 
                if hasattr(self.poly, 'n_features_in_') else None
            )
        }
