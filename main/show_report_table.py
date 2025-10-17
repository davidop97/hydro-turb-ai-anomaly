from tkinter import Toplevel, Frame, Label
from tkinter import ttk

def show_report_table(max_values, machine_type, severity_results, p_desalineacion_cols, p_desbalanceo_cols):
    # Crear ventana emergente
    report_window = Toplevel()
    report_window.title("Reporte Detallado")
    report_window.geometry("800x500")
    report_window.configure(bg="#F5F6FA")

    # Frame principal
    table_frame = Frame(report_window, bg="#F5F6FA")
    table_frame.pack(padx=20, pady=20, fill="both", expand=True)

    # Título del reporte
    Label(table_frame, text="Reporte de Análisis", font=("Helvetica", 16, "bold"), bg="#F5F6FA", fg="#1E3A8A").pack(pady=10)

    # Crear tabla con ttk.Treeview
    tree = ttk.Treeview(table_frame, show="headings", height=15)
    tree["columns"] = ("Category", "Value")
    tree.heading("Category", text="Categoría")
    tree.heading("Value", text="Valor")
    tree.column("Category", width=400, anchor="w")
    tree.column("Value", width=200, anchor="center")

    # Estilo para la tabla
    style = ttk.Style()
    style.configure("Treeview", font=("Helvetica", 11), rowheight=30)
    style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"), background="#60A5FA", foreground="white")

    # Sección: Tipo de máquina
    tree.insert("", "end", values=("", ""), tags=("separator",))
    tree.insert("", "end", values=("Tipo de Máquina", machine_type), tags=("section_header",))

    # Sección: Valores máximos
    tree.insert("", "end", values=("", ""), tags=("separator",))
    tree.insert("", "end", values=("Valores Máximos", ""), tags=("section_header",))
    for col, val in max_values.items():
        tree.insert("", "end", values=(f"  {col}", f"{val:.2f}"))

    # Sección: Probabilidades
    tree.insert("", "end", values=("", ""), tags=("separator",))
    tree.insert("", "end", values=("Probabilidades", ""), tags=("section_header",))
    for col in p_desalineacion_cols.keys():
        tree.insert("", "end", values=(f"  P(Desalineación) ({col})", f"{p_desalineacion_cols[col]:.4f}"))
        tree.insert("", "end", values=(f"  P(Desbalanceo) ({col})", f"{p_desbalanceo_cols[col]:.4f}"))

    # Sección: Severidad
    tree.insert("", "end", values=("", ""), tags=("separator",))
    tree.insert("", "end", values=("Severidad", ""), tags=("section_header",))
    for col, status in severity_results.items():
        tree.insert("", "end", values=(f"  {col}", status), tags=(status.lower(),))

    # Configurar tags para estilos
    tree.tag_configure("section_header", font=("Helvetica", 12, "bold"), background="#E0E7FF")
    tree.tag_configure("separator", background="#F5F6FA")
    tree.tag_configure("verde", foreground="#2ECC71", font=("Helvetica", 11, "bold"))
    tree.tag_configure("amarillo", foreground="#F1C40F", font=("Helvetica", 11, "bold"))
    tree.tag_configure("rojo", foreground="#E74C3C", font=("Helvetica", 11, "bold"))

    # Empaquetar la tabla
    tree.pack(fill="both", expand=True)

    report_window.mainloop()