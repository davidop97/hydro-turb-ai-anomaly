import joblib  # type: ignore
import pandas as pd
from plotter import plot_data

pipeline = joblib.load("turbine_pipeline.joblib")

# Procesar un archivo
file_path = "Histórico, Tendencia_ UNIDAD 1 (1).csv"
df_processed, nominal_speed = pipeline.fit_transform(file_path)
df_original = pipeline.get_original_data()

def preprocess_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera los dataframes a partir del pipeline de procesamiento.
    Los retorna
    """
    df_original = pipeline.get_original_data()
    df_processed = pipeline.fit_transform(file_path)
    return df_original, df_processed

# Graficar los datos procesados
plot_data(df_original)
plot_data(df_processed)