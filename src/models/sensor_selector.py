"""
Selector de sensor óptimo por ARCHIVO (no por concatenación).
Cada archivo se evalúa independientemente.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures


class SensorSelector:
    """
    Selecciona el mejor (ARCHIVO, SENSOR) usando K-Fold CV.
    
    Estrategia CORRECTA:
    1. Evalúa CADA archivo INDEPENDIENTEMENTE
    2. En cada archivo, evalúa cada sensor con K-Fold
    3. Retorna el mejor sensor del mejor archivo
    4. IMPORTANTE: No concatena archivos (evita NaN)
    """
    
    def __init__(
        self,
        degree: int = 3,
        speed_col: str = 'KPH',
        n_splits: int = 5,
        random_state: int = 42,
        min_samples: int = 50
    ):
        """
        Args:
            degree: Grado polinomio
            speed_col: Columna velocidad
            n_splits: K-folds por archivo
            random_state: Reproducibilidad
            min_samples: Mínimas muestras válidas para evaluar
        """
        self.degree = degree
        self.speed_col = speed_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.min_samples = min_samples
        self.poly = PolynomialFeatures(degree=degree)
        
        # Resultados por archivo
        self.file_results: Dict = {}
        self.file_rankings: Dict = {}  # sensor ranking POR archivo
        
        # Mejor resultado global
        self.best_file: Optional[str] = None
        self.best_sensor: Optional[str] = None
        self.best_file_data: Optional[pd.DataFrame] = None
        self.best_metrics: Optional[Dict] = None
    
    
    @staticmethod
    def get_sensor_cols(
        df: pd.DataFrame,
        speed_col: str = 'KPH'
    ) -> List[str]:
        """
        Extrae sensores válidos de un DataFrame.
        
        Args:
            df: DataFrame
            speed_col: Columna a excluir
            
        Returns:
            Lista de nombres de sensores
        """
        exclude = ['Fecha', speed_col, 'KPH_abs']
        sensors = [
            col for col in df.columns
            if col not in exclude 
            and df[col].dtype in ['float64', 'float32', 'int64']
        ]
        return sensors
    
    
    def evaluate_sensor_kfold(
        self,
        df: pd.DataFrame,
        sensor: str,
        file_name: str = ""
    ) -> Dict:
        """
        Evalúa UN sensor en UN archivo con K-Fold CV.
        
        Args:
            df: DataFrame del archivo (solo con columnas necesarias)
            sensor: Nombre del sensor
            file_name: Para logs
            
        Returns:
            Dict con métricas K-Fold
        """
        # Seleccionar solo las columnas necesarias y limpiar
        df_sensor = df[[self.speed_col, sensor]].dropna()
        
        # Validación de muestras
        if len(df_sensor) < self.min_samples:
            return {
                'valid': False,
                'error': f'Muestras insuficientes ({len(df_sensor)} < {self.min_samples})'
            }
        
        # K-Fold
        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(df_sensor)
        ):
            # Split
            X_train = (
                df_sensor.iloc[train_idx][self.speed_col]
                .values.reshape(-1, 1)
            )
            X_val = (
                df_sensor.iloc[val_idx][self.speed_col]
                .values.reshape(-1, 1)
            )
            y_train = df_sensor.iloc[train_idx][sensor].values
            y_val = df_sensor.iloc[val_idx][sensor].values
            
            # Transformar polinómicamente
            X_train_poly = self.poly.fit_transform(X_train)
            X_val_poly = self.poly.transform(X_val)
            
            # Entrenar modelo
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Evaluar
            y_pred = model.predict(X_val_poly)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            fold_metrics.append({
                'fold': fold_idx,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'n_val_samples': len(y_val)
            })
        
        # Promediar
        mean_mae = np.mean([f['mae'] for f in fold_metrics])
        std_mae = np.std([f['mae'] for f in fold_metrics])
        mean_r2 = np.mean([f['r2'] for f in fold_metrics])
        
        return {
            'valid': True,
            'mean_mae': float(mean_mae),
            'std_mae': float(std_mae),
            'mean_r2': float(mean_r2),
            'n_samples': len(df_sensor),
            'folds': fold_metrics
        }
    
    
    def evaluate_file(
        self,
        file_path: str,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Evalúa TODOS los sensores en UN archivo.
        
        Args:
            file_path: Ruta del archivo
            verbose: Mostrar detalles
            
        Returns:
            Tupla (éxito, resultados)
        """
        file_name = file_path.split('/')[-1]
        
        try:
            df = pd.read_csv(file_path)
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        except Exception as e:
            if verbose:
                print(f"❌ {file_name}: Error cargando - {e}")
            return False, {}
        
        # Limpiar NaN en velocidad
        df = df.dropna(subset=[self.speed_col])
        
        if len(df) < self.min_samples:
            if verbose:
                print(f"⚠️ {file_name}: Muestras insuficientes ({len(df)})")
            return False, {}
        
        # Obtener sensores
        sensors = self.get_sensor_cols(df, self.speed_col)
        
        if not sensors:
            if verbose:
                print(f"⚠️ {file_name}: Sin sensores válidos")
            return False, {}
        
        if verbose:
            print(f"\n📄 {file_name}")
            print(f"   Sensores: {sensors}")
            print(f"   Muestras: {len(df):,}")
        
        # Evaluar cada sensor
        sensor_results = {}
        for sensor in sensors:
            result = self.evaluate_sensor_kfold(df, sensor, file_name)
            sensor_results[sensor] = result
            
            if result['valid'] and verbose:
                print(f"      {sensor:6s} → MAE={result['mean_mae']:.6f}±"
                      f"{result['std_mae']:.6f}, R²={result['mean_r2']:.4f}")
            elif not result['valid'] and verbose:
                print(f"      {sensor:6s} → {result['error']}")
        
        # Ranking de sensores válidos
        valid_sensors = [
            (s, r['mean_mae'], r['std_mae'], r['mean_r2'])
            for s, r in sensor_results.items()
            if r['valid']
        ]
        valid_sensors.sort(key=lambda x: x[1])  # Sort by MAE
        
        file_result = {
            'file_path': file_path,
            'file_name': file_name,
            'n_samples': len(df),
            'sensors': sensors,
            'sensor_results': sensor_results,
            'ranking': valid_sensors,
            'best_sensor': valid_sensors[0][0] if valid_sensors else None,
            'best_mae': valid_sensors[0][1] if valid_sensors else None
        }
        
        return True, file_result
    
    
    def select_best_sensor(
        self,
        file_paths: List[str],
        verbose: bool = True
    ) -> Tuple[str, str, Dict]:
        """
        Selecciona el mejor (ARCHIVO, SENSOR) globalmente.
        
        Estrategia:
        1. Evalúa CADA archivo por separado (sin concatenación)
        2. En cada archivo, ranking de sensores
        3. Compara todos los archivos
        4. Retorna archivo + sensor con MEJOR MAE global
        
        Args:
            file_paths: Lista de rutas
            verbose: Mostrar detalles
            
        Returns:
            Tupla (best_file_path, best_sensor, results_dict)
        """
        if verbose:
            print(f"\n{'='*70}")
            print("🔍 SELECCIÓN ÓPTIMA: MEJOR ARCHIVO + MEJOR SENSOR")
            print(f"{'='*70}")
            print("Estrategia: K-Fold CV por ARCHIVO (sin concatenación)")
            print(f"Archivos a evaluar: {len(file_paths)}")
            print(f"K-Folds por archivo: {self.n_splits}")
            print(f"Polinomio: grado {self.degree}\n")
        
        # Evaluar cada archivo
        all_candidates = []  # Lista de (file, sensor, mae)
        
        for file_path in file_paths:
            success, file_result = self.evaluate_file(
                file_path,
                verbose=verbose
            )
            
            if success and file_result['ranking']:
                self.file_results[file_path] = file_result
                
                # Agregar a candidatos
                for sensor, mae, std_mae, r2 in file_result['ranking']:
                    all_candidates.append({
                        'file_path': file_path,
                        'file_name': file_result['file_name'],
                        'sensor': sensor,
                        'mae': mae,
                        'std_mae': std_mae,
                        'r2': r2,
                        'n_samples': file_result['n_samples']
                    })
        
        if not all_candidates:
            raise ValueError(
                "No se encontraron combinaciones (archivo, sensor) válidas"
            )
        
        # Ranking global
        all_candidates.sort(key=lambda x: x['mae'])
        
        if verbose:
            print(f"\n{'='*70}")
            print("🏆 RANKING GLOBAL (Top 10)")
            print(f"{'='*70}\n")
            print(f"{'Archivo':<25} {'Sensor':<8} "
                  f"{'MAE':<12} {'R²':<8} {'Muestras':<10}")
            print("-" * 70)
            
            for i, cand in enumerate(all_candidates[:10], 1):
                print(
                    f"{cand['file_name']:<25} {cand['sensor']:<8} "
                    f"{cand['mae']:<12.6f} {cand['r2']:<8.4f} "
                    f"{cand['n_samples']:<10}"
                )
        
        # Mejor candidato
        best_candidate = all_candidates[0]
        self.best_file = best_candidate['file_path']
        self.best_sensor = best_candidate['sensor']
        self.best_metrics = best_candidate
        
        if verbose:
            print(f"\n{'='*70}")
            print("✅ SELECCIÓN FINAL")
            print(f"{'='*70}")
            print(f"Archivo: {best_candidate['file_name']}")
            print(f"Sensor: {best_candidate['sensor']}")
            print(f"MAE: {best_candidate['mae']:.6f}±"
                  f"{best_candidate['std_mae']:.6f}")
            print(f"R²: {best_candidate['r2']:.4f}")
            print(f"Muestras: {best_candidate['n_samples']:,}")
            print(f"{'='*70}\n")
        
        return (
            best_candidate['file_path'],
            best_candidate['sensor'],
            {
                'best_candidate': best_candidate,
                'all_candidates': all_candidates,
                'file_results': self.file_results
            }
        )
    
    
    def load_best_file_data(self) -> pd.DataFrame:
        """
        Carga el DataFrame del archivo seleccionado.
        
        Returns:
            DataFrame del mejor archivo
        """
        if self.best_file is None:
            raise ValueError(
                "Ejecutar select_best_sensor() primero"
            )
        
        df = pd.read_csv(self.best_file)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        
        self.best_file_data = df
        return df
