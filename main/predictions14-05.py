import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
from show_graphs import mostrar_graficas_en_scroll
from model import DataResidualsProcessor
from compute_probabilities import compute_desalineacion_probability


def graficar_ajuste_con_percentiles(
    lower_threshold: float,
    upper_threshold: float,
    df: pd.DataFrame,
    model: DataResidualsProcessor,
    report_file: Optional[str] = None,
    pdf_file: Optional[str] = None
):
    """
    Grafica el ajuste polinómico usando un modelo preentrenado y, para cada columna de desplazamiento,
    calcula la probabilidad de desalineación de cada punto. Genera un reporte y opcionalmente guarda
    las gráficas en un archivo PDF.

    Args:
        lower_threshold (float): Umbral inferior para la interpolación.
        upper_threshold (float): Umbral superior para la interpolación.
        df (pd.DataFrame): DataFrame con los datos a procesar.
        model (DataResidualsProcessor): Modelo preentrenado.
        report_file (str, optional): Ruta para guardar el reporte en CSV.
        pdf_file (str, optional): Ruta para guardar las gráficas en un archivo PDF.
    """
    # Calcular residuos usando el ajuste preentrenado.
    residuos, columnas, kph, datos_originales, predicciones = model.calculate_residuals_global(df=df)
    print(f"Forma de residuos calculados: {residuos.shape}")
    print(f"Columnas: {columnas}")

    # Convertir kph a vector 1D
    if kph.ndim != 2 or kph.shape[1] != 1:
        raise ValueError("KPH debe ser un arreglo (n, 1).")
    kph = kph.flatten()

    # Ordenar los datos por KPH (de mayor a menor) para graficar de forma ordenada
    order = np.argsort(kph)[::-1]
    kph_sorted = kph[order]
    datos_sorted = datos_originales[order, :]
    predicciones_sorted = predicciones[order, :]

    # Calcular la matriz de residuos absolutos
    residuos_abs = np.abs(residuos)
    n_columns = residuos_abs.shape[1]
    n_columns = residuos.shape[1]

    # Calcular probabilidades de desalineación y desbalanceo
    p_desalineacion_cols = {}
    p_desbalanceo_cols = {}
    for i, col in enumerate(columnas):
        # col_res = residuos[:, i]
        col_res = residuos_abs[:, i]
        p_d = compute_desalineacion_probability(col_res, lower_threshold, upper_threshold)
        p_desalineacion_cols[col] = p_d.mean()
        p_desbalanceo_cols[col] = 1.0 - p_desalineacion_cols[col]
        print(f"Columna {col}: P(Desalineación) = {p_desalineacion_cols[col]:.4f}, P(Desbalanceo) = {p_desbalanceo_cols[col]:.4f}")

    # Probabilidades globales
    # p_desalineacion_global = np.mean(list(p_desalineacion_cols.values()))
    # p_desbalanceo_global = 1.0 - p_desalineacion_global
    # print(f"\nProbabilidad global de desalineación: {p_desalineacion_global:.4f}")
    # print(f"Probabilidad global de desbalanceo: {p_desbalanceo_global:.4f}")

    # Crear DataFrame de reporte
    data_dict = {'KPH': kph_sorted}
    for i, col in enumerate(columnas):
        data_dict[f'Amplitud ({col})'] = datos_sorted[:, i]
        data_dict[f'Predicción ({col})'] = predicciones_sorted[:, i]
        data_dict[f'|Residuo| ({col})'] = np.abs(datos_sorted[:, i] - predicciones_sorted[:, i])
        data_dict[f'P(Desalineación) Global ({col})'] = p_desalineacion_cols[col]
        data_dict[f'P(Desbalanceo) Global ({col})'] = p_desbalanceo_cols[col]

    df_report = pd.DataFrame(data_dict)
    # df_report['ResiduoMedioArchivo'] = residuos.mean()
    df_report['ResiduoMedioArchivo'] = residuos_abs.mean()

    if report_file:
        df_report.to_csv(report_file, index=False)
        print(f"Reporte completo guardado en: {report_file}")

    # Crear la figura con las gráficas
    fig = plt.figure(figsize=(12, 6 * n_columns))
    for i, col in enumerate(columnas):
        plt.subplot(n_columns, 1, i + 1)
        diferencia = np.abs(datos_sorted[:, i] - predicciones_sorted[:, i])

        scatter = plt.scatter(
            kph_sorted,
            datos_sorted[:, i],
            c=diferencia,
            cmap='viridis',
            alpha=0.7,
            s=50,
            label=f"Datos reales ({col})"
        )
        plt.colorbar(scatter, label="|Residuo|")
        plt.plot(
            kph_sorted,
            predicciones_sorted[:, i],
            color='red',
            label='Ajuste polinómico (grado 3)',
            linewidth=2
        )
        plt.fill_between(
            kph_sorted,
            predicciones_sorted[:, i],
            datos_sorted[:, i],
            color='gray',
            alpha=0.3,
            label='Diferencia (ruido)'
        )
        plt.xlabel('KPH (velocidad)')
        plt.ylabel(f'Amplitud ({col})')
        plt.title(f'{col} - P(Desalineación) = {p_desalineacion_cols[col]:.4f}')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

    # Mostrar las gráficas en una ventana con scroll
    mostrar_graficas_en_scroll(fig)
