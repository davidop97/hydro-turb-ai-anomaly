from tkinter import (Button, Canvas, Entry, Frame, IntVar, Label, OptionMenu,
                     Radiobutton, Scrollbar, StringVar, Tk, Toplevel,
                     filedialog, messagebox)
from typing import Dict, Optional

import pandas as pd
from load_pipeline import preprocess_data
from model import DataResidualsProcessor
from predictions import graficar_ajuste_con_percentiles
from show_report_table import show_report_table
from vibration_severity_checker import (DEFAULT_ACTION_LIMITS,
                                        check_vibration_severity)

from src.visualization.plotter import plot_data

# ===========================================================================
# Constantes
# ===========================================================================
DEFAULT_MODEL_PATH = "modelo_mejor_fit.pkl"
DEFAULT_THRESHOLDS_PATH = "thresholds.csv"
DEFAULT_PIPELINE_PATH = "turbine_pipeline.joblib"
DEFAULT_REPORT_FILE = "reporte.csv"
DEFAULT_REPORT_PDF = "reporte.pdf"

# ===========================================================================
# Funciones de carga
# ===========================================================================
def load_model(model_path: str) -> DataResidualsProcessor:
    model = DataResidualsProcessor.load(model_path)
    print("Modelo cargado con éxito.")
    return model

def load_thresholds(thresholds_path: str) -> tuple[float, float]:
    lower_threshold, upper_threshold = pd.read_csv(thresholds_path).values[0]
    print(f"Umbral inferior: {lower_threshold}, Umbral superior: {upper_threshold}")
    return lower_threshold, upper_threshold

# ===========================================================================
# Funciones de procesamiento
# ===========================================================================
def process_and_plot(
    file_path: str,
    pipeline_path: str,
    model: DataResidualsProcessor,
    lower_threshold: float,
    upper_threshold: float,
    report_file: str,
    report_pdf: str,
    machine_type: str,
    severity_level: Optional[str],
    custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None
):
    df_original, df_preprocessed, _, max_values = preprocess_data(file_path, pipeline_path)
    print("Datos originales y preprocesados cargados con éxito.")
    messagebox.showinfo("Éxito", "Datos procesados exitosamente.")
    show_processed_data(df_original, df_preprocessed)

    _, severity_results, p_desalineacion_cols, p_desbalanceo_cols = graficar_ajuste_con_percentiles(
        lower_threshold=lower_threshold,
        model=model,
        df=df_preprocessed,
        upper_threshold=upper_threshold,
        report_file=report_file,
        pdf_file=report_pdf,
    )
    
    # Evaluar severidad (manteniendo la lógica original si aplica)
    severity_results = check_vibration_severity(max_values, machine_type, severity_level, custom_thresholds=custom_thresholds)
    
    # Mostrar el reporte en una tabla
    show_report_table(max_values, machine_type, severity_results, p_desalineacion_cols, p_desbalanceo_cols)

def show_processed_data(df_original, df_processed):
    window = Toplevel()
    window.title("Datos Procesados")
    window.state('zoomed')
    window.geometry("1200x800")
    window.configure(bg="#F5F6FA")
    canvas = Canvas(window, bg="#F5F6FA")
    scrollbar = Scrollbar(window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollable_frame = Frame(canvas, bg="#F5F6FA")
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    Label(scrollable_frame, text="Datos Originales", font=("Helvetica", 16, "bold"), bg="#F5F6FA", fg="#1E3A8A").pack(pady=10)
    plot_data(df_original, scrollable_frame)
    Label(scrollable_frame, text="", bg="#F5F6FA").pack(pady=10)
    Label(scrollable_frame, text="Datos Procesados", font=("Helvetica", 16, "bold"), bg="#F5F6FA", fg="#1E3A8A").pack(pady=10)
    plot_data(df_processed, scrollable_frame)

# ===========================================================================
# Interfaz Gráfica
# ===========================================================================
def open_gui():
    def browse_file(entry_field):
        file_path = filedialog.askopenfilename()
        entry_field.delete(0, "end")
        entry_field.insert(0, file_path)

    def update_threshold_fields(*args):
        selected_machine = machine_type_var.get()
        default_limits = DEFAULT_ACTION_LIMITS[selected_machine]
        severity_value = severity_var.get()
        for widget in thresholds_frame.winfo_children():
            widget.destroy()

        Label(thresholds_frame, text="Ajustar Umbrales:", font=("Helvetica", 12, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=0, column=0, columnspan=6, sticky="w", padx=5, pady=5)
        colors_to_show = ["verde", "amarillo", "rojo"] if severity_value == 0 else [["verde", "amarillo", "rojo"][severity_value - 1]]
        for i, direction in enumerate(["GE-DE", "GE-NDE", "T"]):
            Label(thresholds_frame, text=direction, font=("Helvetica", 10, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=1, column=i*2, columnspan=2, padx=5, pady=2)
            for j, color in enumerate(colors_to_show):
                color_label = color.capitalize()
                color_fg = {"verde": "#2ECC71", "amarillo": "#F1C40F", "rojo": "#E74C3C"}[color]
                Label(thresholds_frame, text=f"{color_label}:", font=("Helvetica", 10), bg="#F5F6FA", fg=color_fg).grid(row=j+2, column=i*2, sticky="w", padx=5)
                entry = Entry(thresholds_frame, width=10, font=("Helvetica", 10), fg=color_fg)
                entry.insert(0, str(default_limits[direction][color]))
                entry.grid(row=j+2, column=i*2+1, padx=5, pady=2)
                threshold_entries[direction][color] = entry

    def execute():
        model_path = model_path_entry.get() or DEFAULT_MODEL_PATH
        thresholds_path = thresholds_path_entry.get() or DEFAULT_THRESHOLDS_PATH
        file_path = file_path_entry.get()
        pipeline_path = pipeline_path_entry.get() or DEFAULT_PIPELINE_PATH
        report_file = report_file_entry.get() or DEFAULT_REPORT_FILE
        report_pdf = report_pdf_entry.get() or DEFAULT_REPORT_PDF
        machine_type = machine_type_var.get()
        severity_value = severity_var.get()

        severity_level = None
        if severity_value == 1:
            severity_level = 'verde'
        elif severity_value == 2:
            severity_level = 'amarillo'
        elif severity_value == 3:
            severity_level = 'rojo'

        custom_thresholds = {"GE-DE": {}, "GE-NDE": {}, "T": {}}
        colors_to_collect = ["verde", "amarillo", "rojo"] if severity_value == 0 else [["verde", "amarillo", "rojo"][severity_value - 1]]
        for direction in ["GE-DE", "GE-NDE", "T"]:
            for color in colors_to_collect:
                try:
                    custom_thresholds[direction][color] = float(threshold_entries[direction][color].get())
                except ValueError:
                    messagebox.showerror("Error", f"Umbral inválido para {direction} - {color.capitalize()}. Debe ser un número.")
                    return

        model = load_model(model_path)
        lower_threshold, upper_threshold = load_thresholds(thresholds_path)
        process_and_plot(
            file_path=file_path,
            pipeline_path=pipeline_path,
            model=model,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            report_file=report_file,
            report_pdf=report_pdf,
            machine_type=machine_type,
            severity_level=severity_level,
            custom_thresholds=custom_thresholds
        )

    root = Tk()
    root.title("Parámetros de Procesamiento")
    root.geometry("700x700")  # Aumentamos el ancho para acomodar la nueva columna
    root.configure(bg="#F5F6FA")
    root.protocol("WM_DELETE_WINDOW", root.quit)

    main_frame = Frame(root, bg="#F5F6FA")
    main_frame.pack(padx=15, pady=15, fill="both", expand=True)

    inputs_frame = Frame(main_frame, bg="#F5F6FA", bd=1, relief="groove", padx=10, pady=10)
    inputs_frame.pack(pady=5, fill="x")

    Label(inputs_frame, text="Ruta del modelo:", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    model_path_entry = Entry(inputs_frame, width=40, font=("Helvetica", 10))
    model_path_entry.insert(0, DEFAULT_MODEL_PATH)
    model_path_entry.grid(row=0, column=1, padx=5, pady=5)
    Button(inputs_frame, text="Buscar", command=lambda: browse_file(model_path_entry), bg="#60A5FA", fg="white", font=("Helvetica", 10)).grid(row=0, column=2, padx=5, pady=5)

    Label(inputs_frame, text="Ruta de umbrales:", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    thresholds_path_entry = Entry(inputs_frame, width=40, font=("Helvetica", 10))
    thresholds_path_entry.insert(0, DEFAULT_THRESHOLDS_PATH)
    thresholds_path_entry.grid(row=1, column=1, padx=5, pady=5)
    Button(inputs_frame, text="Buscar", command=lambda: browse_file(thresholds_path_entry), bg="#60A5FA", fg="white", font=("Helvetica", 10)).grid(row=1, column=2, padx=5, pady=5)

    Label(inputs_frame, text="Ruta del archivo de datos:", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    file_path_entry = Entry(inputs_frame, width=40, font=("Helvetica", 10))
    file_path_entry.grid(row=2, column=1, padx=5, pady=5)
    Button(inputs_frame, text="Buscar", command=lambda: browse_file(file_path_entry), bg="#60A5FA", fg="white", font=("Helvetica", 10)).grid(row=2, column=2, padx=5, pady=5)

    Label(inputs_frame, text="Ruta del pipeline:", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    pipeline_path_entry = Entry(inputs_frame, width=40, font=("Helvetica", 10))
    pipeline_path_entry.insert(0, DEFAULT_PIPELINE_PATH)
    pipeline_path_entry.grid(row=3, column=1, padx=5, pady=5)
    Button(inputs_frame, text="Buscar", command=lambda: browse_file(pipeline_path_entry), bg="#60A5FA", fg="white", font=("Helvetica", 10)).grid(row=3, column=2, padx=5, pady=5)

    Label(inputs_frame, text="Archivo de reporte (CSV):", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=4, column=0, sticky="w", padx=5, pady=5)
    report_file_entry = Entry(inputs_frame, width=40, font=("Helvetica", 10))
    report_file_entry.insert(0, DEFAULT_REPORT_FILE)
    report_file_entry.grid(row=4, column=1, padx=5, pady=5)

    Label(inputs_frame, text="Archivo de reporte (PDF):", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=5, column=0, sticky="w", padx=5, pady=5)
    report_pdf_entry = Entry(inputs_frame, width=40, font=("Helvetica", 10))
    report_pdf_entry.insert(0, DEFAULT_REPORT_PDF)
    report_pdf_entry.grid(row=5, column=1, padx=5, pady=5)

    machine_frame = Frame(main_frame, bg="#F5F6FA", bd=1, relief="groove", padx=10, pady=5)
    machine_frame.pack(pady=10, fill="x")
    Label(machine_frame, text="Tipo de máquina:", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=0, column=0, sticky="w", padx=5)
    machine_type_var = StringVar(value=list(DEFAULT_ACTION_LIMITS.keys())[0])
    OptionMenu(machine_frame, machine_type_var, *DEFAULT_ACTION_LIMITS.keys(), command=update_threshold_fields).grid(row=0, column=1, padx=5)

    severity_frame = Frame(main_frame, bg="#F5F6FA", bd=1, relief="groove", padx=10, pady=5)
    severity_frame.pack(pady=10, fill="x")
    Label(severity_frame, text="Nivel de severidad:", font=("Helvetica", 11, "bold"), bg="#F5F6FA", fg="#1E3A8A").grid(row=0, column=0, sticky="w", padx=5)
    severity_var = IntVar(value=0)
    Radiobutton(severity_frame, text="Clasificación general", variable=severity_var, value=0, bg="#F5F6FA", font=("Helvetica", 10), command=update_threshold_fields).grid(row=0, column=1, padx=5)
    Radiobutton(severity_frame, text="Verde", variable=severity_var, value=1, bg="#F5F6FA", fg="#2ECC71", font=("Helvetica", 10), command=update_threshold_fields).grid(row=0, column=2, padx=5)
    Radiobutton(severity_frame, text="Amarillo", variable=severity_var, value=2, bg="#F5F6FA", fg="#F1C40F", font=("Helvetica", 10), command=update_threshold_fields).grid(row=0, column=3, padx=5)
    Radiobutton(severity_frame, text="Rojo", variable=severity_var, value=3, bg="#F5F6FA", fg="#E74C3C", font=("Helvetica", 10), command=update_threshold_fields).grid(row=0, column=4, padx=5)

    thresholds_frame = Frame(main_frame, bg="#F5F6FA", bd=1, relief="groove", padx=10, pady=5)
    thresholds_frame.pack(pady=10, fill="x")
    threshold_entries = {"GE-DE": {}, "GE-NDE": {}, "T": {}}  # Añadimos "T"
    update_threshold_fields()

    Button(main_frame, text="Ejecutar", command=execute, bg="#60A5FA", fg="white", font=("Helvetica", 12, "bold"), relief="raised", padx=10, pady=5).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    open_gui()