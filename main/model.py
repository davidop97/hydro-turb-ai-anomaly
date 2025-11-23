from typing import List, Optional, Tuple, Union

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class DataResidualsProcessor:
    """
    Clase para procesar datos, ajustar modelos polinómicos y calcular residuos.

    #calc-res: Versión 1.3
    - Se entrena sobre un DataFrame de entrenamiento para obtener un ajuste que represente
      el comportamiento de desbalanceo.
    - El objeto resultante guarda los modelos, la transformación polinómica, las columnas de
      desplazamiento y un umbral basado en el percentil del error medio absoluto.
    - Luego se utiliza este ajuste para calcular los residuos en nuevos datos sin reentrenar.
    - Incluye métodos para guardar y cargar el objeto entrenado (el "mejor ajuste").
    """

    def __init__(self, degree: int = 3, speed_col: str = 'KPH', threshold_percentile: float = 90.0):
        """
        Inicializa el procesador.

        Parámetros:
          - degree: Grado del polinomio para el modelo.
          - speed_col: Nombre de la columna de velocidad.
          - threshold_percentile: Percentil para definir el umbral basado en el error medio absoluto.
        """
        self.degree = degree
        self.speed_col = speed_col
        self.threshold_percentile = threshold_percentile
        self.poly = PolynomialFeatures(degree=self.degree)
        self.models: dict[str, LinearRegression] = {}  # Diccionario para almacenar el modelo de cada columna de desplazamiento.
        self.displacement_cols: List[str] = []  # Inicializado como lista vacía para evitar problemas de tipo.
        self.df_train: Optional[pd.DataFrame] = None  # DataFrame de entrenamiento.
        self.threshold: Optional[float] = None  # Umbral calculado a partir de los residuos de entrenamiento.

    def _load_data(self, ruta_archivo: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Carga los datos desde un DataFrame o desde una ruta de archivo.
        Se asume que el archivo ya está preprocesado.
        """
        if df is not None:
            return df
        elif ruta_archivo is not None:
            try:
                df_loaded = pd.read_csv(ruta_archivo, delimiter=",")
                return df_loaded
            except Exception as e:
                raise ValueError(f"No se pudo leer el archivo {ruta_archivo}: {e}")
        else:
            raise ValueError("Se debe proporcionar un DataFrame o una ruta de archivo.")

    def _get_displacement_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las columnas de desplazamiento a partir del DataFrame,
        descartando 'Fecha' y la columna de velocidad.
        """
        columnas = [col for col in df.columns if col not in ['Fecha', self.speed_col]]
        if not columnas:
            raise ValueError("No se encontraron columnas de desplazamiento válidas.")
        return columnas

    def fit(self, df: Optional[pd.DataFrame] = None, ruta_archivo: Optional[str] = None) -> 'DataResidualsProcessor':
        """
        Ajusta los modelos de regresión polinómica usando los datos de entrenamiento (desbalanceo)
        y calcula un umbral basado en el percentil del error medio absoluto.

        Se recomienda pasar un DataFrame (que puede ser la concatenación de varios archivos) o
        un archivo preprocesado para obtener un ajuste global.
        """
        # Cargar datos de entrenamiento
        self.df_train = self._load_data(ruta_archivo, df)
        # Extraer columnas de desplazamiento
        self.displacement_cols = self._get_displacement_cols(self.df_train)
        # Transformar la velocidad en funciones polinómicas
        kph = self.df_train[self.speed_col].to_numpy().reshape(-1, 1)
        X_poly = self.poly.fit_transform(kph)

        # Ajustar un modelo para cada columna de desplazamiento
        for col in self.displacement_cols:
            y = self.df_train[col].values
            model = LinearRegression()
            model.fit(X_poly, y)
            self.models[col] = model

        # Calcular los residuos en el conjunto de entrenamiento
        residuals = []
        for col in self.displacement_cols:
            y = self.df_train[col].values
            y_pred = self.models[col].predict(X_poly)
            resid = y - y_pred
            residuals.append(resid)
        # Combinar los residuos en una matriz (n_muestras x n_ejes)
        residuals_matrix = np.column_stack(residuals)
        # Calcular el error medio absoluto por muestra
        mean_abs_resid = np.mean(np.abs(residuals_matrix), axis=1)
        # Definir el umbral como el percentil especificado
        self.threshold = float(np.percentile(mean_abs_resid, self.threshold_percentile))
        print(f"Threshold (percentil {self.threshold_percentile}): {self.threshold}")
        return self

    def calculate_residuals_global(self, df: Optional[pd.DataFrame] = None, ruta_archivo: Optional[str] = None) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los residuos en nuevos datos utilizando el modelo de referencia (el "mejor modelo")
        guardado en el objeto. Esta función ignora las diferencias en nombres o cantidad de columnas;
        para cada sensor (columna) en el archivo nuevo (todas las columnas que no sean 'Fecha' ni la velocidad),
        se aplica el mismo modelo (el mejor modelo entrenado) para calcular la predicción y el residuo.

        Retorna:
          - residuals_matrix: Matriz de residuos (n_muestras, n_sensores) calculada usando el mejor modelo para cada sensor.
          - new_sensor_cols: Lista de nombres de los sensores (columnas) del nuevo archivo.
          - kph: Arreglo de la velocidad (KPH) en forma (n_muestras, 1).
          - original_matrix: Matriz con los datos originales de cada sensor.
          - predictions_matrix: Matriz con las predicciones del modelo de referencia (aplicada a cada sensor).
        """
        # Cargar los datos del archivo nuevo
        data = self._load_data(ruta_archivo, df)
        # Extraer la columna de velocidad y transformarla
        kph = data[self.speed_col].to_numpy().reshape(-1, 1)
        X_poly = self.poly.transform(kph)

        # Obtener todas las columnas de sensor en el nuevo archivo (ignorando 'Fecha' y la velocidad)
        new_sensor_cols = [col for col in data.columns if col not in ['Fecha', self.speed_col]]

        # Seleccionamos el mejor modelo de referencia.
        # Se asume que el objeto guardado (por ejemplo, de select_best_fit) ya contiene
        # un conjunto de modelos; usaremos el del primer sensor entrenado como modelo de referencia.
        best_sensor = self.displacement_cols[0]
        best_model = self.models[best_sensor]
        print(f"Usando el modelo de referencia del sensor: {best_sensor}")

        # Para cada sensor nuevo, aplicamos el mismo modelo de referencia
        residuals_list = []
        predictions_list = []
        original_list = []

        for col in new_sensor_cols:
            y_real = data[col].values
            # Usar el modelo de referencia para predecir la señal esperada
            y_pred = best_model.predict(X_poly)
            resid = y_real - y_pred
            residuals_list.append(resid)
            predictions_list.append(y_pred)
            original_list.append(y_real)

        # Armar las matrices de salida
        residuals_matrix = np.column_stack(residuals_list)
        predictions_matrix = np.column_stack(predictions_list)
        original_matrix = np.column_stack([np.asarray(arr) for arr in original_list])

        return residuals_matrix, new_sensor_cols, kph, original_matrix, predictions_matrix


    def stack_residuals(self, residuals: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Apila todas las columnas de residuos (o una lista de arreglos) en un vector 1D.

        Args:
          - residuals: Matriz de residuos (n_muestras, n_columnas) o lista de matrices.

        Retorna:
          - np.ndarray: Arreglo 1D con todos los residuos apilados.

        #calc-res: Método auxiliar para combinar residuos.
        """
        if isinstance(residuals, list):
            if not residuals:
                raise ValueError("La lista de residuos está vacía.")
            return np.concatenate([r.flatten() for r in residuals])
        elif isinstance(residuals, np.ndarray):
            if len(residuals.shape) != 2:
                raise ValueError("El arreglo de residuos debe ser 2D (muestras, columnas).")
            return residuals.flatten()
        else:
            raise ValueError("La entrada debe ser un arreglo NumPy o una lista de arreglos.")

    def save(self, file_path: str) -> None:
        """
        Guarda el objeto entrenado (incluyendo poly, modelos, columnas y threshold) en el archivo especificado.
        """
        joblib.dump(self, file_path)
        print(f"Modelo guardado en: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'DataResidualsProcessor':
        """
        Carga el objeto entrenado desde un archivo.

        Gracias a @classmethod, se puede llamar sin necesidad de una instancia previa:
            processor = DataResidualsProcessor.load("ruta_al_modelo.pkl")
        """
        return joblib.load(file_path)