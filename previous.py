from tkinter import Button, Entry, Label, Tk, filedialog

import pandas as pd
from load_pipeline import preprocess_data
from model import DataResidualsProcessor
from predictions import graficar_ajuste_con_percentiles


def load_model(model_path: str) -> DataResidualsProcessor:
    """Carga el modelo desde un archivo."""
    model = DataResidualsProcessor.load(model_path)
    print("Modelo cargado con éxito.")
    return model


def load_thresholds(thresholds_path: str) -> tuple[float, float]:
    """Carga los umbrales desde un archivo CSV."""
    lower_threshold, upper_threshold = pd.read_csv(thresholds_path).values[0]
    print(f"Umbral inferior: {lower_threshold}, Umbral superior: {upper_threshold}")
    return lower_threshold, upper_threshold


def process_and_plot(
    file_path: str,
    pipeline_path: str,
    model: DataResidualsProcessor,
    lower_threshold: float,
    upper_threshold: float,
    report_file: str,
    report_pdf: str
):
    """Procesa los datos y genera el gráfico con los percentiles."""
    df_original, df_preprocessed, _ = preprocess_data(file_path, pipeline_path)
    print("Datos originales y preprocesados cargados con éxito.")
    graficar_ajuste_con_percentiles(
        lower_threshold=lower_threshold,
        model=model,
        df=df_preprocessed,
        upper_threshold=upper_threshold,
        report_file=report_file,
        pdf_file=report_pdf,
    )


def open_gui():
    """Abre una ventana GUI para ingresar parámetros y ejecutar el procesamiento."""
    def browse_file(entry_field):
        """Permite seleccionar un archivo y coloca la ruta en el campo correspondiente."""
        file_path = filedialog.askopenfilename()
        entry_field.delete(0, "end")
        entry_field.insert(0, file_path)

    def execute():
        """Ejecuta el procesamiento con los parámetros ingresados."""
        model_path = model_path_entry.get()
        thresholds_path = thresholds_path_entry.get()
        file_path = file_path_entry.get()
        pipeline_path = pipeline_path_entry.get()
        report_file = report_file_entry.get()
        report_pdf = report_pdf_entry.get()

        # Cargar modelo y umbrales
        model = load_model(model_path)
        lower_threshold, upper_threshold = load_thresholds(thresholds_path)

        # Procesar datos y graficar
        process_and_plot(
            file_path=file_path,
            pipeline_path=pipeline_path,
            model=model,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            report_file=report_file,
            report_pdf=report_pdf,
        )
        print("Procesamiento completado.")

    # Crear ventana principal
    root = Tk()
    root.title("Parámetros de Procesamiento")

    # Crear campos de entrada
    Label(root, text="Ruta del modelo:").grid(row=0, column=0, sticky="w")
    model_path_entry = Entry(root, width=50)
    model_path_entry.grid(row=0, column=1)
    Button(root, text="Buscar", command=lambda: browse_file(model_path_entry)).grid(row=0, column=2)

    Label(root, text="Ruta de umbrales:").grid(row=1, column=0, sticky="w")
    thresholds_path_entry = Entry(root, width=50)
    thresholds_path_entry.grid(row=1, column=1)
    Button(root, text="Buscar", command=lambda: browse_file(thresholds_path_entry)).grid(row=1, column=2)

    Label(root, text="Ruta del archivo de datos:").grid(row=2, column=0, sticky="w")
    file_path_entry = Entry(root, width=50)
    file_path_entry.grid(row=2, column=1)
    Button(root, text="Buscar", command=lambda: browse_file(file_path_entry)).grid(row=2, column=2)

    Label(root, text="Ruta del pipeline:").grid(row=3, column=0, sticky="w")
    pipeline_path_entry = Entry(root, width=50)
    pipeline_path_entry.grid(row=3, column=1)
    Button(root, text="Buscar", command=lambda: browse_file(pipeline_path_entry)).grid(row=3, column=2)

    Label(root, text="Archivo de reporte (CSV):").grid(row=4, column=0, sticky="w")
    report_file_entry = Entry(root, width=50)
    report_file_entry.grid(row=4, column=1)

    Label(root, text="Archivo de reporte (PDF):").grid(row=5, column=0, sticky="w")
    report_pdf_entry = Entry(root, width=50)
    report_pdf_entry.grid(row=5, column=1)

    # Botón para ejecutar
    Button(root, text="Ejecutar", command=execute).grid(row=6, column=1, pady=10)

    # Iniciar loop de la ventana
    root.mainloop()


if __name__ == "__main__":
    open_gui()

#set PYTHONPATH=c:\Users\Usuario\Desktop\tests