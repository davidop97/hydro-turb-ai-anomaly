import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Turbina Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Detector de Anomalías en Turbinas Hidráulicas")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    uploaded_file = st.file_uploader("Carga archivo CSV:", type=["csv"])

if uploaded_file is not None:
    
    with st.spinner("🔄 Procesando..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI no está corriendo en puerto 8000")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()
    
    # === TABS ===
    tab1, tab2, tab3 = st.tabs(["📊 Predicción", "📈 Gráficas por Sensor", "🎯 Severidad"])
    
    # === TAB 1: PREDICCIÓN GLOBAL ===
    with tab1:
        st.subheader("Resultado de Predicción Global")
        
        probs = result["probabilities"]
        desbal_pct = probs["desbalanceo"] * 100
        desalin_pct = probs["desalineacion"] * 100
        
        # Mostrar predicción grande y clara
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prediction = result["prediction"]
            confidence = result["confidence"] * 100
            
            if "DESALINEACIÓN" in prediction:
                st.error(f"🔴 **{prediction}**\nConfianza: {confidence:.1f}%", icon="⚠️")
            else:
                st.success(f"🟢 **{prediction}**\nConfianza: {confidence:.1f}%", icon="✅")
        
        with col2:
            st.metric("Anomalías", result["metadata"]["n_anomalies"])
        
        st.markdown("---")
        
        # Gráfica Pie Chart
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        sizes = [desbal_pct, desalin_pct]
        labels = [f"Desbalanceo\n{desbal_pct:.1f}%", f"Desalineación\n{desalin_pct:.1f}%"]
        colors = ["#059669", "#DC2626"]
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors, 
            autopct="%1.1f%%",
            shadow=True, 
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"}
        )
        
        ax.set_title("Distribución de Probabilidad", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Detalles
        st.subheader("📋 Información del Análisis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info(f"**Muestras:** {result['metadata']['samples_analyzed']}")
        with col2:
            st.info(f"**Velocidad:** {result['metadata']['nominal_speed']:.2f} KPH")
        with col3:
            st.info(f"**Sensores:** {len(result['metadata']['sensors'])}")
        with col4:
            st.info(f"**Anomalías Detectadas:** {result['metadata']['n_anomalies']}")
    
    # === TAB 2: GRÁFICAS POR SENSOR ===
    with tab2:
        st.subheader("📈 Análisis Detallado por Sensor")
        
        # ✅ Obtener datos de metadata
        sensor_data = result["metadata"].get("sensor_data", {})
        sensors = result["metadata"]["sensors"]
        kph = result["metadata"].get("kph", [])
        severity = result["severity"]
        max_values = result["metadata"].get("max_values", {})
        
        # Color por severidad
        severity_colors = {
            "verde": "#059669",
            "amarillo": "#F59E0B",
            "rojo": "#DC2626"
        }
        
        for sensor in sensors:
            st.markdown(f"### {sensor}")
            
            # Obtener severidad de este sensor
            sensor_severity = severity.get(sensor, "desconocido").lower()
            severity_color = severity_colors.get(sensor_severity, "#666666")
            max_val = max_values.get(sensor, 0)
            
            # Mostrar severidad en la parte superior
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"**Severidad:** <span style='color: {severity_color}; font-size: 18px; font-weight: bold;'>{sensor_severity.upper()}</span>",  # noqa: E501
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(f"**Valor Máx:** {max_val:.2f}")
            with col3:
                mean_residual = sensor_data.get(sensor, {}).get("mean_residual", 0)
                st.markdown(f"**Residuo Medio:** {mean_residual:.4f}")
            
            # Gráfica: Original vs Predicción
            if sensor in sensor_data:
                fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
                
                original = sensor_data[sensor]["original"]
                predicted = sensor_data[sensor]["predicted"]
                abs_residual = sensor_data[sensor]["abs_residual"]
                
                ax.scatter(
                    kph, original,
                    c=abs_residual, cmap="RdYlGn_r",
                    alpha=0.7, s=50, label=f"Datos Reales ({sensor})",
                    edgecolors="black", linewidth=0.5
                )
                
                ax.plot(kph, predicted, color="red", label="Ajuste Polinómico", linewidth=2.5)
                
                ax.fill_between(kph, predicted, original, color="gray", alpha=0.2, label="Residuo")
                
                ax.set_xlabel("KPH (Velocidad)", fontsize=11, fontweight="bold")
                ax.set_ylabel(f"Amplitud ({sensor})", fontsize=11, fontweight="bold")
                ax.set_title(f"{sensor} - Datos vs Predicción", fontsize=12, fontweight="bold")
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Colorbar para residuos
                scatter = ax.collections[0]
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("|Residuo|", fontsize=10, fontweight="bold")
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("---")
    
    # === TAB 3: SEVERIDAD DETALLADA ===
    with tab3:
        st.subheader("🎯 Reporte de Severidad por Sensor")
        
        severity = result["severity"]
        max_vals = result["metadata"].get("max_values", {})
        
        # Crear tabla con nombres visibles
        severity_data = []
        for sensor in result["metadata"]["sensors"]:
            level = severity.get(sensor, "desconocido")
            max_val = max_vals.get(sensor, 0)
            
            severity_data.append({
                "🔹 SENSOR": f"**{sensor}**",
                "📊 VALOR MÁX": f"{max_val:.2f}",
                "🎯 SEVERIDAD": f"**{level.upper()}**",
                "ESTADO": "✅ OK" if "verde" in level.lower() else ("⚠️ ALERTA" if "amarillo" in level.lower() else "❌ CRÍTICO")  # noqa: E501
            })
        
        df_sev = pd.DataFrame(severity_data)
        
        # Mostrar tabla HTML bonita
        html_table = "<table style='width: 100%; border-collapse: collapse;'>"
        html_table += "<tr style='background-color: #1E3A8A; color: white;'>"
        for col in df_sev.columns:
            html_table += f"<th style='padding: 12px; text-align: left; border: 1px solid #ccc;'>{col}</th>"  # noqa: E501
        html_table += "</tr>"
        
        for idx, row in df_sev.iterrows():
            severity_val = row["🎯 SEVERIDAD"].lower()
            
            if "verde" in severity_val:
                bg_color = "#DCFCE7"
                text_color = "#059669"
            elif "amarillo" in severity_val:
                bg_color = "#FEF3C7"
                text_color = "#F59E0B"
            else:
                bg_color = "#FEE2E2"
                text_color = "#DC2626"
            
            html_table += f"<tr style='background-color: {bg_color};'>"
            for col in df_sev.columns:
                html_table += f"<td style='padding: 12px; border: 1px solid #ccc; color: {text_color}; font-weight: bold;'>{row[col]}</td>"  # noqa: E501
            html_table += "</tr>"
        
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Resumen
        verde = sum(1 for s in severity.values() if "verde" in s.lower())
        amarillo = sum(1 for s in severity.values() if "amarillo" in s.lower())
        rojo = sum(1 for s in severity.values() if "rojo" in s.lower())
        
        st.subheader("📋 Resumen de Estados")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 VERDE (Normal)", verde, delta=f"{verde}/{len(severity)}")
        with col2:
            st.metric("🟡 AMARILLO (Alerta)", amarillo, delta=f"{amarillo}/{len(severity)}")
        with col3:
            st.metric("🔴 ROJO (Crítico)", rojo, delta=f"{rojo}/{len(severity)}")
        
        st.markdown("---")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones")
        
        if rojo > 0:
            sensores_rojos = [s for s, level in severity.items() if "rojo" in level.lower()]
            st.error(f"🔴 **CRÍTICO**: Sensores en rojo: {', '.join(sensores_rojos)}. Requiere atención INMEDIATA.")  # noqa: E501
        
        if amarillo > 0:
            sensores_amarillos = [s for s, level in severity.items() if "amarillo" in level.lower()]
            st.warning(f"🟡 **ALERTA**: Sensores en alerta: {', '.join(sensores_amarillos)}. Monitorear continuamente.")  # noqa: E501
        
        if verde == len(severity):
            st.success("🟢 **NORMAL**: Todos los sensores dentro de los límites normales.")

else:
    st.info("👈 Carga un archivo CSV para comenzar el análisis")
