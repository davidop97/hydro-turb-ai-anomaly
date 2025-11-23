from typing import Dict

import pandas as pd

from src.preprocessing.pipeline import TurbineDataPipeline
from src.visualization.plotter import plot_data


def preprocess_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, float]]:
    """
    Genera los dataframes utilizando una nueva instancia del pipeline de procesamiento.
    
    Args:
        file_path (str): Ruta al archivo CSV de datos
        
    Returns:
        tuple: (df_original, df_processed, nominal_speed, max_values)
    """
    # Crear una nueva instancia del pipeline
    pipeline = TurbineDataPipeline()
    
    # Procesar el archivo y obtener los DataFrames
    df_processed, nominal_speed, max_values = pipeline.fit_transform(file_path)
    df_original = pipeline.get_original_data()
    return df_original, df_processed, nominal_speed, max_values


if __name__ == "__main__":
    # Procesar un archivo
    file_path = "Histórico, Tendencia_ UNIDAD 1 (1).csv"
    df_original, df_processed, nominal_speed, max_values = preprocess_data(file_path)

    # Graficar los datos procesados
    plot_data(df_processed, parent_window=None)
    plot_data(df_original, parent_window=None)

    # Mostrar el DataFrame procesado y la velocidad nominal
    print('DataFrame procesado:')
    print(df_processed.head())
    print('DataFrame original:')
    print(df_original.head())
    print('Velocidad nominal:')
    print(nominal_speed)
    print('Valores máximos:')
    print(max_values)