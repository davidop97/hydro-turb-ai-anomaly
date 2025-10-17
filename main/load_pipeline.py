from typing import Dict
import joblib  # type: ignore
import pandas as pd
from main.plotter import plot_data
from turbine_pipeline import TurbineDataPipeline


def _load_pipeline(pipeline_path: str) -> TurbineDataPipeline:
    """
    Carga el pipeline de procesamiento desde un archivo.
    """
    pipeline = joblib.load(pipeline_path)
    return pipeline


def preprocess_data(file_path: str, pipeline_path: str) -> tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, float]]:
    """
    Genera los dataframes a partir del pipeline de procesamiento.
    Los retorna
    """
    pipeline = _load_pipeline(pipeline_path)
    # Procesar el archivo y obtener los DataFrames  
    df_processed, nominal_speed, max_values = pipeline.fit_transform(file_path)
    df_original = pipeline.get_original_data()
    return df_original, df_processed, nominal_speed, max_values


if __name__ == "__main__":
    # Procesar un archivo
    file_path = "Histórico, Tendencia_ UNIDAD 1 (1).csv"
    pipeline_path = "turbine_pipeline.joblib"
    df_original, df_processed, nominal_speed, max_values = preprocess_data(file_path, pipeline_path)

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