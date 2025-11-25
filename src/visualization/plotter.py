import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_data(df, parent_window):
    """
    Función para graficar las variables de desplazamiento y velocidad en una ventana de Tkinter.
    """
    # Crear la figura de Matplotlib
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Configurar el eje X para mostrar fechas
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(10))
    plt.xticks(rotation=45, ha='right')
    ax1.set_xlabel('Fecha', fontsize=10)
    ax1.set_ylabel('Desplazamientos (um pk-pk)', color='tab:red', fontsize=10)

    # Seleccionar columnas para graficar (excluyendo 'Fecha' y 'KPH')
    columns_to_plot = [col for col in df.columns if col not in ['Fecha', 'KPH']]
    colors = [
        'tab:red', 'tab:green', 'tab:orange', 'tab:gray', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan'
    ]

    # Graficar cada columna
    for i, column in enumerate(columns_to_plot):
        ax1.plot(df['Fecha'], df[column], color=colors[i % len(colors)], label=column)

    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=8)
    ax1.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

    # Eje derecho para la velocidad (KPH)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Velocidad (KPH)', color='tab:blue', fontsize=10)
    ax2.plot(df['Fecha'], df['KPH'], color='tab:blue', label='Velocidad (KPH)')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=8)

    # Leyenda
    fig.legend(loc="upper left", bbox_to_anchor=(1.07, 0.75), fontsize=9)

    # Título dinámico basado en nombre del DataFrame y columnas
    # title = (
    #     f"Gráfico de sensores de desplazamiento {columns_to_plot} del eje "
    #     "respecto a un punto de referencia y Velocidad (KPH) Normalizadas "
    #     "de una Turbina Hidráulica vs Tiempo (En el instante de parada)"
    # )
    # plt.title(title, fontsize=14, fontweight='bold', pad=15, wrap=True)

    # Ajustamos el formato de la gráfica
    fig.tight_layout()

    # Integrar la figura en la ventana de Tkinter
    canvas = FigureCanvasTkAgg(fig, master=parent_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)
    canvas.draw()