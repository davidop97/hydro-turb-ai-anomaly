import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict
from show_graphs import mostrar_graficas_en_scroll
from model import DataResidualsProcessor
from compute_probabilities import compute_desalineacion_probability

def _process_residuals(df: pd.DataFrame, model: DataResidualsProcessor):
    """Procesa los datos para obtener residuos, KPH, datos originales y predicciones."""
    residuos, columnas, kph, datos_originales, predicciones = model.calculate_residuals_global(df)
    if kph.ndim != 2 or kph.shape[1] != 1:
        raise ValueError("KPH debe ser un arreglo (n, 1).")
    kph = kph.flatten()
    order = np.argsort(kph)[::-1]
    return residuos, columnas, kph[order], datos_originales[order, :], predicciones[order, :]

def _calculate_probabilities(residuos_abs: np.ndarray, columnas: list, lower_threshold: float, upper_threshold: float):
    """Calcula las probabilidades de desalineación y desbalanceo por columna."""
    p_desalineacion_cols = {}
    p_desbalanceo_cols = {}
    for i, col in enumerate(columnas):
        col_res = residuos_abs[:, i]
        p_d = compute_desalineacion_probability(col_res, lower_threshold, upper_threshold)
        p_desalineacion_cols[col] = p_d.mean()
        p_desbalanceo_cols[col] = 1.0 - p_desalineacion_cols[col]
    return p_desalineacion_cols, p_desbalanceo_cols

def _generate_report_df(kph: np.ndarray, datos: np.ndarray, predicciones: np.ndarray, columnas: list, p_desalineacion_cols: dict, p_desbalanceo_cols: dict, residuos_abs: np.ndarray):
    """Genera un DataFrame con el reporte de los datos."""
    data_dict = {'KPH': kph}
    for i, col in enumerate(columnas):
        data_dict[f'Amplitud ({col})'] = datos[:, i]
        data_dict[f'Predicción ({col})'] = predicciones[:, i]
        data_dict[f'|Residuo| ({col})'] = np.abs(datos[:, i] - predicciones[:, i])
        data_dict[f'P(Desalineación) Global ({col})'] = p_desalineacion_cols[col]
        data_dict[f'P(Desbalanceo) Global ({col})'] = p_desbalanceo_cols[col]
    df_report = pd.DataFrame(data_dict)
    df_report['ResiduoMedioArchivo'] = residuos_abs.mean()
    return df_report

def _plot_adjustment(kph: np.ndarray, datos: np.ndarray, predicciones: np.ndarray, columnas: list, p_desalineacion_cols: dict, n_columns: int):
    """Genera las gráficas de ajuste para cada columna."""
    fig = plt.figure(figsize=(12, 6 * n_columns))
    for i, col in enumerate(columnas):
        plt.subplot(n_columns, 1, i + 1)
        diferencia = np.abs(datos[:, i] - predicciones[:, i])
        scatter = plt.scatter(kph, datos[:, i], c=diferencia, cmap='viridis', alpha=0.7, s=50, label=f"Datos reales ({col})")
        plt.colorbar(scatter, label="|Residuo|")
        plt.plot(kph, predicciones[:, i], color='red', label='Ajuste polinómico (grado 3)', linewidth=2)
        plt.fill_between(kph, predicciones[:, i], datos[:, i], color='gray', alpha=0.3, label='Diferencia (ruido)')
        plt.xlabel('KPH (velocidad)')
        plt.ylabel(f'Amplitud ({col})')
        plt.title(f'{col} - P(Desalineación) = {p_desalineacion_cols[col]:.4f}')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
    return fig

def graficar_ajuste_con_percentiles(
    lower_threshold: float,
    upper_threshold: float,
    df: pd.DataFrame,
    model: DataResidualsProcessor,
    report_file: Optional[str] = None,
    pdf_file: Optional[str] = None,
    machine_type: str = "Francis horizontal",
    severity_level: Optional[str] = None,
    max_values: Optional[dict] = None,
    custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None
):
    """
    Grafica el ajuste polinómico usando un modelo preentrenado, calcula probabilidades de desalineación,
    genera un reporte y muestra los resultados de severidad en un formato organizado.

    Args:
        lower_threshold (float): Umbral inferior para la interpolación.
        upper_threshold (float): Umbral superior para la interpolación.
        df (pd.DataFrame): DataFrame con los datos a procesar.
        model (DataResidualsProcessor): Modelo preentrenado.
        report_file (str, optional): Ruta para guardar el reporte en CSV.
        pdf_file (str, optional): Ruta para guardar las gráficas en un archivo PDF.
        machine_type (str): Tipo de máquina para evaluar severidad.
        severity_level (str, optional): Nivel de severidad ('verde', 'amarillo', 'rojo') o None.
        max_values (dict, optional): Diccionario con valores máximos para evaluar severidad.
        custom_thresholds (dict, optional): Umbrales personalizados para la evaluación de severidad.

    Returns:
        tuple: (df_report, severity_results, p_desalineacion_cols, p_desbalanceo_cols) - DataFrame del reporte,
               resultados de severidad, y diccionarios de probabilidades.
    """
    # Procesar residuos y ordenar datos
    residuos, columnas, kph, datos, predicciones = _process_residuals(df, model)
    residuos_abs = np.abs(residuos)
    n_columns = len(columnas)

    # Calcular probabilidades de desalineación y desbalanceo
    p_desalineacion_cols, p_desbalanceo_cols = _calculate_probabilities(residuos_abs, columnas, lower_threshold, upper_threshold)

    # Generar reporte
    df_report = _generate_report_df(kph, datos, predicciones, columnas, p_desalineacion_cols, p_desbalanceo_cols, residuos_abs)
    if report_file:
        df_report.to_csv(report_file, index=False)

    # Generar gráficas
    fig = _plot_adjustment(kph, datos, predicciones, columnas, p_desalineacion_cols, n_columns)
    if pdf_file:
        plt.savefig(pdf_file)
        plt.close()
    mostrar_graficas_en_scroll(fig)

    # Evaluar severidad si se proporcionan max_values
    severity_results = {}
    if max_values is not None:
        from vibration_severity_checker import check_vibration_severity
        severity_results = check_vibration_severity(max_values, machine_type, severity_level, custom_thresholds=custom_thresholds)

    # Devolver el DataFrame del reporte, resultados de severidad y probabilidades
    return df_report, severity_results, p_desalineacion_cols, p_desbalanceo_cols