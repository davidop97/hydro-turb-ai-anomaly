import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def mostrar_graficas_en_scroll(fig):
    """
    Muestra las gráficas en una ventana con scroll usando Tkinter.
    Args:
        fig (matplotlib.figure.Figure): Figura de matplotlib a mostrar.
    """
    # Crear ventana principal
    root = tk.Tk()
    root.title("Gráficas con Scroll")

    # Configurar el tamaño inicial de la ventana (más grande)
    root.geometry("1400x900")  # Ancho x Alto

    # Configurar el cierre de la ventana para detener el programa
    def on_closing():
        root.quit()  # Detener el loop principal
        root.destroy()  # Cerrar la ventana

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Crear un frame con scroll
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    canvas_scroll = tk.Canvas(frame, bg="white")
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas_scroll.yview)
    scrollable_frame = tk.Frame(canvas_scroll, bg="white")

    # Ajustar el ancho del canvas para centrar el contenido
    def ajustar_ancho(event):
        canvas_scroll.itemconfig(window_id, width=event.width)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
    )

    window_id = canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="n")
    canvas_scroll.bind("<Configure>", ajustar_ancho)
    canvas_scroll.configure(yscrollcommand=scrollbar.set)

    canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Agregar la figura de matplotlib al frame con scroll
    canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True, pady=5)  # Espaciado reducido entre gráficas

    # Reducir el espacio superior entre el contenido y el borde
    canvas_scroll.yview_moveto(0.01)

    # Mejorar el comportamiento del scroll
    def _on_mouse_wheel(event):
        canvas_scroll.yview_scroll(-1 * int(event.delta / 120), "units")

    canvas_scroll.bind_all("<MouseWheel>", _on_mouse_wheel)

    # Iniciar el loop de la ventana
    root.mainloop()