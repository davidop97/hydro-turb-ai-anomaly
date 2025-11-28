import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.models.turb_predictor import TurbinePredictor  # noqa: E402

# === CONFIG ===
st.set_page_config(
    page_title="Turbina Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Detector de Anomalías en Turbinas Hidráulicas")
st.markdown("---")

# === CACHE: Cargar predictor una sola vez ===
@st.cache_resource
def load_predictor():
    return TurbinePredictor()

predictor = load_predictor()

# === FUNCIÓN AUXILIAR: Convertir numpy (igual que en el endpoint) ===
def convert_numpy(obj):
    """Convierte numpy arrays a listas/floats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    return obj

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Configuración")
    uploaded_file = st.file_uploader("Carga archivo CSV:", type=["csv"])

if uploaded_file is not None:
    
    with st.spinner("🔄 Procesando..."):
        try:
            # === LÓGICA DEL ENDPOINT PREDICT ===
            temp_path = None
            
            # Validar extensión
            if not uploaded_file.name.endswith('.csv'):
                st.error("❌ Solo archivos CSV permitidos")
                st.stop()
            
            # Guardar temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            # Predicción
            result = predictor.predict(temp_path)
            
            # ✅ Convertir numpy (igual que en el endpoint)
            sensor_data_clean = {}
            for sensor, data in result.get("sensor_data", {}).items():
                sensor_data_clean[sensor] = {
                    "original": convert_numpy(data["original"]),
                    "predicted": convert_numpy(data["predicted"]),
                    "residual": convert_numpy(data["residual"]),
                    "abs_residual": convert_numpy(data["abs_residual"]),
                    "mean_residual": float(data["mean_residual"])
                }
            
            # ✅ Estructura final (igual que endpoint)
            result = {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "metadata": {
                    **result["metadata"],
                    "sensor_data": sensor_data_clean,
                    "kph": convert_numpy(result.get("kph", [])),
                    "max_values": convert_numpy(result.get("max_values", {}))
                },
                "severity": result["severity"]
            }
            
            # Limpiar
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()
    
    # === TABS ===
    tab1, tab2, tab3 = st.tabs(["📊 Predicción", "📈 Gráficas por Sensor", "🎯 Severidad"])
    
    # === TAB 1: PREDICCIÓN GLOBAL ===
    with tab1:
        st.subheader("Resultado de Predicción Global")
        
        probs = result["probabilities"]
        desbal_pct = probs["desbalanceo"] * 100
        desalin_pct = probs["desalineacion"] * 100
        
        prediction = result["prediction"]
        confidence = result["confidence"] * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if "DESALINEACIÓN" in prediction:
                st.error(f"🔴 **{prediction}**\nConfianza: {confidence:.1f}%", icon="⚠️")
            else:
                st.success(f"🟢 **{prediction}**\nConfianza: {confidence:.1f}%", icon="✅")
        
        with col2:
            total_points = result["metadata"]["samples_analyzed"]
            st.metric("Total Muestras", total_points)
        
        st.markdown("---")
        
        desbal_points = int(total_points * probs["desbalanceo"])
        desalin_points = int(total_points * probs["desalineacion"])
        
        st.subheader("📊 Distribución de Fenómenos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "🟢 Desbalanceo",
                f"{desbal_points} puntos",
                delta=f"{desbal_pct:.1f}%"
            )
        
        with col2:
            st.metric(
                "🔴 Desalineación",
                f"{desalin_points} puntos",
                delta=f"{desalin_pct:.1f}%"
            )
        
        st.markdown("---")
        
        st.subheader("📋 Información del Análisis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info(f"**Velocidad Nominal:** {result['metadata']['nominal_speed']:.2f} KPH")
        with col2:
            st.info(f"**Cantidad de Sensores:** {len(result['metadata']['sensors'])}")
        with col3:
            sensors_list = ", ".join(result['metadata']['sensors'])
            st.info(f"**Sensores:** {sensors_list}")
        with col4:
            st.info(f"**Confianza del Modelo:** {confidence:.1f}%")
    
    # === TAB 2: GRÁFICAS POR SENSOR ===
    with tab2:
        st.subheader("📈 Análisis Detallado por Sensor")
        
        sensor_data = result["metadata"].get("sensor_data", {})
        sensors = result["metadata"]["sensors"]
        kph = result["metadata"].get("kph", [])
        severity = result["severity"]
        max_values = result["metadata"].get("max_values", {})
        
        severity_colors = {
            "verde": "#059669",
            "amarillo": "#F59E0B",
            "rojo": "#DC2626"
        }
        
        for sensor in sensors:
            st.markdown(f"#### {sensor}")
            
            sensor_severity = severity.get(sensor, "desconocido").lower()
            severity_color = severity_colors.get(sensor_severity, "#666666")
            max_val = max_values.get(sensor, 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"**Severidad:** "
                    f"<span style='color: {severity_color}; font-size: 16px; font-weight: bold;'>"
                    f"{sensor_severity.upper()}</span>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(f"**Valor Máx:** {max_val:.2f}")
            with col3:
                mean_residual = sensor_data.get(sensor, {}).get("mean_residual", 0)
                st.markdown(f"**Residuo Medio:** {mean_residual:.4f}")
            
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
        
        severity_data = []
        for sensor in result["metadata"]["sensors"]:
            level = severity.get(sensor, "desconocido")
            max_val = max_vals.get(sensor, 0)
            
            severity_data.append({
                "Sensor": sensor,
                "Valor Máx": f"{max_val:.2f}",
                "Severidad": level.upper(),
                "Estado": (
                    "✅ OK" if "verde" in level.lower()
                    else ("⚠️ ALERTA" if "amarillo" in level.lower()
                          else "❌ CRÍTICO")
                )
            })
        
        df_sev = pd.DataFrame(severity_data)
        
        html_table = "<table style='width: 100%; border-collapse: collapse;'>"
        html_table += "<tr style='background-color: #1E3A8A; color: white;'>"
        for col in df_sev.columns:
            html_table += (
                f"<th style='padding: 12px; text-align: left; border: 1px solid #ccc;'>"
                f"{col}</th>"
            )
        html_table += "</tr>"
        
        for idx, row in df_sev.iterrows():
            severity_val = row["Severidad"].lower()
            
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
            html_table += (
                f"<td style='padding: 12px; border: 1px solid #ccc; color: {text_color}; "
                f"font-weight: bold;'>{row['Sensor']}</td>"
            )
            html_table += (
                f"<td style='padding: 12px; border: 1px solid #ccc; color: {text_color}; "
                f"font-weight: bold;'>{row['Valor Máx']}</td>"
            )
            html_table += (
                f"<td style='padding: 12px; border: 1px solid #ccc; color: {text_color}; "
                f"font-weight: bold;'>{row['Severidad']}</td>"
            )
            html_table += (
                f"<td style='padding: 12px; border: 1px solid #ccc; color: {text_color}; "
                f"font-weight: bold;'>{row['Estado']}</td>"
            )
            html_table += "</tr>"
        
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
        
        st.markdown("---")
        
        verde = sum(1 for s in severity.values() if "verde" in s.lower())
        amarillo = sum(1 for s in severity.values() if "amarillo" in s.lower())
        rojo = sum(1 for s in severity.values() if "rojo" in s.lower())
        
        st.subheader("📋 Resumen de Estados")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Verde", verde, delta=f"{verde}/{len(severity)}")
        with col2:
            st.metric("🟡 Amarillo", amarillo, delta=f"{amarillo}/{len(severity)}")
        with col3:
            st.metric("🔴 Rojo", rojo, delta=f"{rojo}/{len(severity)}")
        
        st.markdown("---")
        
        st.subheader("💡 Recomendaciones")
        
        if rojo > 0:
            sensores_rojos = [s for s, level in severity.items() if "rojo" in level.lower()]
            st.error(f"🔴 CRÍTICO: {', '.join(sensores_rojos)} - Atención INMEDIATA")
        
        if amarillo > 0:
            sensores_amarillos = [s for s, level in severity.items() if "amarillo" in level.lower()]
            st.warning(f"🟡 ALERTA: {', '.join(sensores_amarillos)} - Monitoreo continuo")
        
        if verde == len(severity):
            st.success("🟢 NORMAL: Todos los sensores dentro de límites")

else:
    st.info("👈 Carga un archivo CSV para comenzar el análisis")
