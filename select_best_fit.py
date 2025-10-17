import numpy as np
from typing import List
from main.model import DataResidualsProcessor

    
    
def select_best_fit(file_paths: List[str], degree: int = 3, speed_col: str = 'KPH', threshold_percentile: float = 90.0) -> DataResidualsProcessor:
    """
    Para cada archivo en file_paths, entrena un modelo candidato (DataResidualsProcessor) y calcula el error promedio
    (media del error absoluto) para cada sensor (columna de desplazamiento). Se selecciona el sensor con el error
    promedio mínimo entre todos los archivos. Se crea un objeto DataResidualsProcessor que se basa únicamente en ese sensor.

    Retorna:
      - Una instancia de DataResidualsProcessor que contiene el modelo (regresión lineal) entrenado para el mejor sensor,
        la lista de columnas (una sola, el sensor seleccionado), el objeto polinómico y el umbral (calculado sobre ese sensor).
    """
    best_candidate = None
    best_error = float('inf')
    best_sensor = None
    best_df_train = None
    best_poly = None
    best_model_for_sensor = None

    for path in file_paths:
        print(f"Entrenando ajuste en {path}...")
        # Entrena un candidato usando todos los sensores disponibles en el archivo
        candidate = DataResidualsProcessor(degree=degree, speed_col=speed_col, threshold_percentile=threshold_percentile)
        candidate.fit(ruta_archivo=path)
        # Preparar datos para calcular el error en candidate.df_train
        if candidate.df_train is None:
            raise ValueError("candidate.df_train is None. Ensure that the fit method populates df_train correctly.")
        if candidate.df_train[candidate.speed_col] is None:
            raise ValueError(f"Column '{candidate.speed_col}' in candidate.df_train is None. Ensure the data is loaded correctly.")
        kph = np.asarray(candidate.df_train[candidate.speed_col].values).reshape(-1, 1)
        X_poly = candidate.poly.transform(kph)
        for sensor in candidate.displacement_cols:
            y = candidate.df_train[sensor].values
            y_pred = candidate.models[sensor].predict(X_poly)
            error = np.mean(np.abs(y - y_pred))
            print(f"Archivo: {path}, Sensor: {sensor} - Error promedio: {error:.4f}")
            if error < best_error:
                best_error = error
                best_sensor = sensor
                best_df_train = candidate.df_train.copy()
                if candidate.poly is None:
                    raise ValueError("candidate.poly is None. Ensure that the fit method initializes the polynomial transformer correctly.")
                if candidate.poly is None:
                    raise ValueError("candidate.poly is None. Ensure that the fit method initializes the polynomial transformer correctly.")
                best_poly = candidate.poly  # Se conserva el objeto de transformación polinómica
                best_model_for_sensor = candidate.models[sensor]
    print(f"\nMejor sensor seleccionado: {best_sensor} con error promedio: {best_error:.4f}")

    # Construir un nuevo DataResidualsProcessor basado únicamente en el sensor seleccionado
    best_candidate = DataResidualsProcessor(degree=degree, speed_col=speed_col, threshold_percentile=threshold_percentile)
    best_candidate.df_train = best_df_train
    if best_poly is None:
        raise ValueError("best_poly is None. Ensure a valid polynomial transformer is selected.")
    if best_poly is not None:
        best_candidate.poly = best_poly
    else:
        raise ValueError("best_poly is None. Ensure a valid polynomial transformer is selected.")
    if best_sensor is None:
        raise ValueError("best_sensor is None. Ensure a valid sensor is selected.")
    best_candidate.displacement_cols = [best_sensor]  # Solo el mejor sensor
    if best_model_for_sensor is None:
        raise ValueError("best_model_for_sensor is None. Ensure a valid model is selected for the best sensor.")
    else:
        best_candidate.models = {best_sensor: best_model_for_sensor}

    # Recalcular el umbral usando únicamente los residuos de ese sensor
    if best_candidate.df_train is None:
        raise ValueError("best_candidate.df_train is None. Ensure that the DataFrame is properly initialized.")
    kph = np.array(best_candidate.df_train[best_candidate.speed_col].values).reshape(-1, 1)
    X_poly = best_candidate.poly.transform(kph)
    y = best_candidate.df_train[best_sensor].values
    y_pred = best_candidate.models[best_sensor].predict(X_poly)
    resid = np.abs(y - y_pred)
    best_candidate.threshold = np.percentile(resid, threshold_percentile)
    print(f"Umbral recalculado para el sensor {best_sensor} (percentil {threshold_percentile}%): {best_candidate.threshold:.4f}")

    return best_candidate
