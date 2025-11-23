from typing import Dict, List, Optional, Tuple, Union

import joblib  # type: ignore
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Clase DataProcessor
class DataProcessor:
    def __init__(self, delimiter: str = ";", date_format: str = "%Y/%m/%d %H:%M:%S") -> None:
        self.delimiter: str = delimiter
        self.date_format: str = date_format

    def read_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Reads a CSV file with a specified delimiter, skipping non-data header lines."""
        try:
            df: pd.DataFrame = pd.read_csv(file_path, delimiter=self.delimiter, skiprows=2)
            return df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def clean_and_reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans columns by removing 'AXI' if it exists, reordering columns, and renaming them."""
        if "AXI" in df.columns:
            df = df.drop("AXI", axis=1)
        date_col: str = df.columns[0]
        other_cols: List[str] = [col for col in df.columns if col != date_col]
        df = df[[date_col] + other_cols]
        df.columns = pd.Index(["Fecha"] + [df.columns[i] for i in range(1, len(df.columns))])
        return df

    def convert_to_float_and_date_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts columns to the appropriate data types."""
        for col in df.columns[1:]:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        df["Fecha"] = pd.to_datetime(
            df["Fecha"], format=self.date_format, errors="coerce"
        )
        return df

    def process_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Processes a single file by reading, cleaning, and converting data types."""
        df: Optional[pd.DataFrame] = self.read_file(file_path)
        if df is not None:
            df = self.clean_and_reorder_columns(df)
            df = self.convert_to_float_and_date_time(df)
        return df

# Transformador para cargar datos
class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, speed_col: str = 'KPH', date_col: str = 'Fecha',
                 delimiter: str = ";", date_format: str = "%Y/%m/%d %H:%M:%S") -> None:
        self.speed_col: str = speed_col
        self.date_col: str = date_col
        self.delimiter: str = delimiter
        self.date_format: str = date_format
        self.data_processor: DataProcessor = DataProcessor(
            delimiter=delimiter,
            date_format=date_format
        )
        self.df_original: Optional[pd.DataFrame] = None

    def fit(self, X: Union[str, pd.DataFrame], y: None = None) -> "DataLoader":
        return self

    def transform(self, X: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(X, str):
            df: Optional[pd.DataFrame] = self.data_processor.process_file(X)
            if df is None:
                raise ValueError(f"No se pudo cargar el archivo: {X}")
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise ValueError("X debe ser una ruta de archivo (str) o un DataFrame.")
        self.df_original = df.copy()
        return df

    def get_original_data(self) -> pd.DataFrame:
        if self.df_original is None:
            raise ValueError("No se han cargado datos todavía.")
        return self.df_original

# Transformador para identificar la velocidad nominal
class NominalSpeedIdentifier(BaseEstimator, TransformerMixin):
    def __init__(self, speed_col: str = 'KPH', max_speed_diff: float = 1.0,
                 min_stable_points: int = 10, top_n_blocks: int = 3) -> None:
        self.speed_col: str = speed_col
        self.max_speed_diff: float = max_speed_diff
        self.min_stable_points: int = min_stable_points
        self.top_n_blocks: int = top_n_blocks
        self.nominal_speed: Optional[float] = None
        self.max_values: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: None = None) -> "NominalSpeedIdentifier":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Identifica la velocidad nominal y calcula los valores máximos de variables
        en bloques estables."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame.")

        df: pd.DataFrame = X.copy()
        df['speed_diff'] = df[self.speed_col].diff().abs()
        df['is_stable'] = df['speed_diff'] < self.max_speed_diff
        df['stable_block'] = (df['is_stable'] != df['is_stable'].shift()).cumsum() * df['is_stable']
        stable_blocks: pd.Series = df[df['stable_block'] > 0].groupby('stable_block').size()

        if stable_blocks.empty or stable_blocks.max() < self.min_stable_points:
            raise ValueError("No se encontró una región estable suficientemente larga.")

        stable_blocks_sorted: pd.Series = stable_blocks[
            stable_blocks >= self.min_stable_points
        ].sort_values(ascending=False)
        top_blocks: pd.Index = stable_blocks_sorted.head(
            min(self.top_n_blocks, len(stable_blocks_sorted))
        ).index

        if len(top_blocks) == 0:
            raise ValueError(
                "No se encontraron bloques estables que cumplan con el mínimo de puntos."
            )

        # Seleccionar el bloque con mayor velocidad promedio
        block_means: pd.Series = (
            df[df['stable_block'].isin(top_blocks)]
            .groupby('stable_block')[self.speed_col]
            .mean()
        )
        selected_block: int = int(block_means.idxmax())
        stable_region: pd.DataFrame = df[df['stable_block'] == selected_block]
        self.nominal_speed = stable_region[self.speed_col].mean()

        # Calcular valores máximos de columnas (excepto speed_col y temporales) en top_blocks
        columns_to_analyze: list[str] = [
            col for col in df.columns 
            if col not in [self.speed_col, 'speed_diff', 'is_stable', 'stable_block']
        ]
        stable_data: pd.DataFrame = df[df['stable_block'].isin(top_blocks)][columns_to_analyze]
        self.max_values = stable_data.max(numeric_only=True).to_dict()

        df.drop(columns=['speed_diff', 'is_stable', 'stable_block'], inplace=True)
        return df

    def get_nominal_speed(self) -> float:
        """Devuelve la velocidad nominal calculada."""
        if self.nominal_speed is None:
            raise ValueError("No se ha calculado la velocidad nominal todavía.")
        return self.nominal_speed

    def get_max_values(self) -> Dict[str, float]:
        """Devuelve los valores máximos de las variables en los bloques estables."""
        if self.max_values is None:
            raise ValueError("No se han calculado los valores máximos todavía.")
        return self.max_values

# Transformador para cortar datos
class DataCutter(BaseEstimator, TransformerMixin):
    def __init__(self, speed_col: str = 'KPH', fall_threshold: float = 0.95,
                 min_consecutive: int = 3) -> None:
        self.speed_col: str = speed_col
        self.fall_threshold: float = fall_threshold
        self.min_consecutive: int = min_consecutive
        self.nominal_speed: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: None = None) -> "DataCutter":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame.")
        if self.nominal_speed is None:
            raise ValueError(
                "La velocidad nominal debe estar definida "
                "(asegúrate de ejecutar NominalSpeedIdentifier primero)."
            )
        df: pd.DataFrame = X.copy()
        below_threshold: pd.Series = df[self.speed_col] < self.fall_threshold * self.nominal_speed
        for i in range(len(df) - self.min_consecutive + 1):
            if below_threshold.iloc[i]:
                if all(df[self.speed_col].iloc[i+j] < df[self.speed_col].iloc[i+j-1]
                       for j in range(1, self.min_consecutive)):
                    inicio_caida_idx: int = df.index[i]
                    return df.loc[inicio_caida_idx:].copy()
        return df.iloc[-1:].copy()

    def set_nominal_speed(self, nominal_speed: float) -> "DataCutter":
        self.nominal_speed = nominal_speed
        return self

# Transformador para escalar datos y eliminar ruido
class DataScaler(BaseEstimator, TransformerMixin):
    def __init__(self, speed_col: str = 'KPH', date_col: str = 'Fecha',
                 method: str = 'standard', minmax_range: Tuple[float, float] = (0, 1),
                 include_speed: bool = True, trim_percentage: float = 0.0) -> None:
        self.speed_col: str = speed_col
        self.date_col: str = date_col
        self.method: str = method
        self.minmax_range: Tuple[float, float] = minmax_range
        self.include_speed: bool = include_speed
        self.trim_percentage: float = trim_percentage
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None

    def fit(self, X: pd.DataFrame, y: None = None) -> "DataScaler":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame.")
        df: pd.DataFrame = X.copy()
        displacement_cols: List[str] = [col for col in df.columns if col not in [self.date_col]]
        columns_to_scale: List[str] = displacement_cols.copy()
        if self.include_speed and self.speed_col in df.columns:
            columns_to_scale.append(self.speed_col)
        for col in columns_to_scale:
            if col not in df.columns:
                raise ValueError(f"La columna {col} no está en el DataFrame.")
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler(
                feature_range=(
                    int(self.minmax_range[0]),
                    int(self.minmax_range[1])
                )
            )
        else:
            raise ValueError("El método debe ser 'standard' o 'minmax'.")
        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        if self.trim_percentage > 0 and self.trim_percentage <= 1:
            if self.method == 'standard':
                speed_scaled_min: float = df[self.speed_col].min()
                threshold: float = speed_scaled_min + self.trim_percentage * (0 - speed_scaled_min)
            elif self.method == 'minmax':
                threshold = self.trim_percentage * self.minmax_range[1]
            df = df[df[self.speed_col] > threshold].copy()
        return df

# Pipeline completo
class TurbineDataPipeline:
    def __init__(self, speed_col: str = 'KPH', date_col: str = 'Fecha',
                 delimiter: str = ";", date_format: str = "%Y/%m/%d %H:%M:%S",
                 max_speed_diff: float = 1.0, min_stable_points: int = 10,
                 top_n_blocks: int = 3, fall_threshold: float = 0.95,
                 min_consecutive: int = 3, scale_method: str = 'standard',
                 minmax_range: Tuple[float, float] = (0, 1),
                 include_speed: bool = True, trim_percentage: float = 0.0) -> None:
        """Inicializa el pipeline de preprocesamiento para datos de turbinas."""
        self.speed_col: str = speed_col
        self.date_col: str = date_col
        self.pipeline: Pipeline = Pipeline([
            ('loader', DataLoader(
                speed_col=speed_col, date_col=date_col,
                delimiter=delimiter, date_format=date_format
            )),
            ('nominal_speed', NominalSpeedIdentifier(
                speed_col=speed_col, max_speed_diff=max_speed_diff,
                min_stable_points=min_stable_points, top_n_blocks=top_n_blocks
            )),
            ('cutter', DataCutter(
                speed_col=speed_col, fall_threshold=fall_threshold,
                min_consecutive=min_consecutive
            )),
            ('scaler', DataScaler(
                speed_col=speed_col,
                date_col=date_col,
                method=scale_method,
                minmax_range=minmax_range,
                include_speed=include_speed,
                trim_percentage=trim_percentage
            ))
        ])

    def fit_transform(
        self,
        X: Union[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
        """Procesa los datos y devuelve el DataFrame preprocesado, la velocidad nominal
        y los valores máximos de los sensores."""
        # Paso 1: Cargar los datos
        df: pd.DataFrame = self.pipeline.named_steps['loader'].fit_transform(X)
        
        # Paso 2: Identificar la velocidad nominal y los valores máximos de los sensores
        df = self.pipeline.named_steps['nominal_speed'].fit_transform(df)
        nominal_speed: float = self.pipeline.named_steps['nominal_speed'].get_nominal_speed()
        max_values: Dict[str, float] = self.pipeline.named_steps['nominal_speed'].get_max_values()
        
        # Paso 3: Pasar la velocidad nominal al cutter
        self.pipeline.named_steps['cutter'].set_nominal_speed(nominal_speed)
        
        # Paso 4: Cortar los datos
        df = self.pipeline.named_steps['cutter'].fit_transform(df)
        
        # Paso 5: Escalar los datos y eliminar ruido
        df_processed: pd.DataFrame = self.pipeline.named_steps['scaler'].fit_transform(df)
        
        return df_processed, nominal_speed, max_values

    def get_original_data(self) -> pd.DataFrame:
        """Devuelve los datos originales para visualización."""
        return self.pipeline.named_steps['loader'].get_original_data()

    def save(self, filepath: str) -> None:
        """Guarda el pipeline en un archivo para usarlo en el frontend."""
        joblib.dump(self, filepath)
        print(f"Pipeline guardado en {filepath}")

    @staticmethod
    def load(filepath: str) -> "TurbineDataPipeline":
        """Carga un pipeline guardado desde un archivo."""
        pipeline: "TurbineDataPipeline" = joblib.load(filepath)
        print(f"Pipeline cargado desde {filepath}")
        return pipeline