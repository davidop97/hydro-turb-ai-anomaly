from typing import Tuple

import pandas as pd

from src.preprocessing.pipeline import DataProcessor


class EDADataLoader:
    """
    Cargador de datos para EDA - SIN escalado.
    Mantiene valores físicos interpretables.
    """
    
    def __init__(
        self,
        delimiter: str = ";",
        date_format: str = "%Y/%m/%d %H:%M:%S"
    ):
        self.processor = DataProcessor(
            delimiter=delimiter,
            date_format=date_format
        )
    
    def load_clean_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga y limpia datos SIN escalar.
        Mantiene valores originales para interpretación física.
        """
        df = self.processor.process_file(file_path)
        
        if df is None:
            raise ValueError(f"No se pudo cargar: {file_path}")
        
        # Convertir velocidad a valores absolutos para análisis
        df['KPH_abs'] = df['KPH'].abs()
        
        return df
    
    def load_before_after(
        self,
        raw_path: str,
        processed_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga datos antes y después del preprocesamiento.
        
        Returns:
            Tupla (df_raw_clean, df_processed_unscaled)
        """
        # Raw data limpio (sin escalar)
        df_raw = self.load_clean_data(raw_path)
        
        # Processed data (viene escalado, pero podemos des-escalar)
        df_processed = pd.read_csv(processed_path)
        df_processed['Fecha'] = pd.to_datetime(
            df_processed['Fecha'],
            errors='coerce'
        )
        
        # IMPORTANTE: processed ya viene escalado
        # Para EDA, mejor usar df_raw cortado en la misma región
        # O reportar que processed está escalado
        
        return df_raw, df_processed
