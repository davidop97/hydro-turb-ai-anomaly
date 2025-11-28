# ⚡ Detector de Anomalías en Turbinas Hidráulicas

**Sistema de ML para detección y clasificación de anomalías (desbalanceo vs desalineación) en turbinas hidráulicas Francis usando análisis de residuos y clasificadores probabilísticos.**

---

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Entrenar el Modelo](#entrenar-el-modelo)
  - [Entrenar Clasificadores](#entrenar-clasificadores)
  - [Hacer Predicciones](#hacer-predicciones)
  - [Ejecutar la Interfaz Web](#ejecutar-la-interfaz-web)
- [API REST (Legado)](#api-rest-legado)
- [Workflows de CI/CD](#workflows-de-cicd)
- [Deployment](#deployment)
- [Documentación Técnica](#documentación-técnica)

---

## 🎯 Descripción General

Este proyecto implementa un **sistema completo de machine learning** para la detección de anomalías en turbinas hidráulicas. El sistema:

1. **Procesa datos de sensores** de vibración (CSP, CSL, CTP, CTL) en diferentes velocidades (KPH)
2. **Entrena un modelo de residuos** usando polinomios cúbicos para capturar la vibración base
3. **Entrena clasificadores probabilísticos** (Linear, Logistic, GMM) para diferenciar:
   - **Desbalanceo**: Desequilibrio de masa rotacional
   - **Desalineación**: Desalineación del eje
4. **Calcula severidad** en tres niveles (Verde, Amarillo, Rojo) por sensor
5. **Proporciona visualización interactiva** mediante Streamlit

---

## ✨ Características

✅ **Modelo de Residuos Robusto**: Ajuste polinómico por sensor para capturar patrones base  
✅ **3 Clasificadores Probabilísticos**: Linear, Logistic, GMM - todos con validación train/test  
✅ **Severidad Multinivel**: Evaluación por sensor con umbrales configurables  
✅ **Interfaz Web Interactiva**: Streamlit con 3 tabs (Predicción, Gráficas, Severidad)  
✅ **Tracking de Experimentos**: MLflow para reproducibilidad  
✅ **Dockerizado**: docker-compose con MLflow integrado  
✅ **Deployed**: Streamlit Cloud en producción  

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Datos CSV (Sensores)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐              ┌──────────▼────┐
    │ EDA       │              │ Preprocessing │
    │ (eda.py)  │              │ (pipeline.py) │
    └────┬─────┘              └──────────┬────┘
         │                               │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Modelo de Residuos (train)   │
         │  - Polinomios cúbicos         │
         │  - Por sensor                 │
         │  - residuals_CSP_v3.pkl       │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Extracción de Features       │
         │  - 12 features estadísticos   │
         │  - Por archivo entrenamiento  │
         └───────────────┬───────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───▼────┐          ┌────▼────┐          ┌───▼────┐
│ Linear │          │ Logistic │          │  GMM   │
│ (best) │          │          │          │        │
└───┬────┘          └────┬────┘          └───┬────┘
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Predicción en Datos Nuevos   │
         │  - Residuos por muestra       │
         │  - Probabilidades             │
         │  - Severidad por sensor       │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Streamlit UI                 │
         │  ├─ Tab 1: Predicción Global  │
         │  ├─ Tab 2: Gráficas Sensores  │
         │  └─ Tab 3: Severidad Detalle  │
         └───────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
hydro-turb-ai-anomaly/
│
├── 📂 src/
│   ├── 📂 models/
│   │   ├── anomaly_detector.py          # Clase principal de detección
│   │   ├── classifier.py                # Clasificadores (Linear/Logistic/GMM)
│   │   ├── residuals_model.py           # Modelo de residuos base
│   │   ├── sensor_selector.py           # Selección de sensores
│   │   ├── turb_predictor.py            # Predictor integrado
│   │   └── vibration_severity_checker.py # Evaluación de severidad
│   │
│   ├── 📂 preprocessing/
│   │   ├── eda_loader.py                # Carga y EDA inicial
│   │   ├── pipeline.py                  # Pipeline de preprocesamiento
│   │   └── utils.py                     # Utilidades
│   │
│   └── 📂 visualization/
│       ├── charts.py                    # Gráficos matplotlib
│       ├── eda_plots.py                 # Plots exploratorios
│       ├── plots.py                     # Plots adicionales
│       └── config.py                    # Configuración de estilos
│
├── 📂 workflows/
│   ├── eda.py                           # Análisis exploratorio de datos
│   ├── preprocess.py                    # Preprocesamiento de datos
│   ├── train_model.py                   # Entrenamiento modelo residuos
│   ├── train_classifier.py              # Entrenamiento clasificadores
│   ├── predict_anomalies.py             # Predicción en datos nuevos
│   ├── generate_reports.py              # Generación de reportes
│   └── __init__.py
│
├── 📂 configs/
│   ├── config.py                        # Configuración global
│   └── settings.py                      # Parámetros ajustables
│
├── 📂 app/
│   └── main.py                          # Interfaz Streamlit
│
├── 📂 data/
│   ├── raw/                             # Datos originales
│   ├── processed/
│   │   ├── imbalance/                   # Datos desbalanceo
│   │   └── misalignment/                # Datos desalineación
│   └── reports/                         # Reportes generados
│
├── 📂 models/
│   └── trained/                         # Modelos entrenados
│       ├── residuals_CSP_v3.pkl         # Modelo residuos
│       ├── classifier_linear.pkl
│       ├── classifier_logistic.pkl
│       ├── classifier_gmm.pkl
│       ├── classifier_best.pkl
│       └── best_classifier_metadata.json
│
├── 📂 mlruns/                           # MLflow experiments
├── 📂 mlartifacts/                      # MLflow artifacts
│
├── 📂 .github/workflows/
│   ├── preprocess_on_data_change.yml    # Trigger preprocesamiento
│   └── preprocess_on_pipeline_change.yml # Trigger por cambios
│
├── Dockerfile                           # Docker image
├── docker-compose.yml                   # Services (MLflow + API)
├── requirements.txt                     # Dependencias Python
├── .env                                 # Variables de entorno
├── .gitignore
└── README.md                            # Este archivo
```

---

## 📦 Requisitos

### Sistema
- **Python**: 3.11+
- **Docker**: 24.0+ (opcional, para servicios)
- **RAM**: 4GB+ (entrenamiento)
- **CPU**: 2+ núcleos

### Dependencias Python

```txt
# Core ML/Data
pandas==2.2.0
numpy==1.24.3
scikit-learn==1.3.2
scipy==1.11.4

# Modelos
scikit-learn==1.3.2

# Visualización
matplotlib==3.8.2
seaborn==0.13.0

# Web/API
streamlit==1.28.1
fastapi==0.104.1
uvicorn==0.24.0

# MLflow (Tracking)
mlflow==2.9.0

# Utilities
python-dotenv==1.0.0
pydantic==2.4.2
joblib==1.3.2

# Desarrollo
pytest==7.4.3
black==23.12.0
flake8==6.1.0
```

---

## ⚙️ Instalación

### 1. Clonar Repositorio

```bash
git clone https://github.com/tu-usuario/hydro-turb-ai-anomaly.git
cd hydro-turb-ai-anomaly
```

### 2. Crear Entorno Virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# o
.venv\Scripts\activate  # Windows
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tus valores
```

### 5. Descargar Datos (si aplica)

```bash
# Colocar archivos CSV en data/raw/
# Estructura esperada:
# data/
# ├── raw/
# │   ├── desbalanceo_archivo1.csv
# │   ├── desbalanceo_archivo2.csv
# │   ├── desalineacion_archivo1.csv
# │   └── ...
```

---

## 🚀 Uso

### Ejecutar como Módulo

Todos los scripts deben ejecutarse como **módulos** desde la raíz del proyecto:

```bash
python -m workflows.nombre_script
```

### 1️⃣ Análisis Exploratorio (EDA)

```bash
python -m workflows.eda
```

**Salida:**
- Estadísticas descriptivas
- Distribuciones por sensor
- Gráficas en `data/reports/eda/`
- Perfiles de cada archivo

---

### 2️⃣ Preprocesamiento

```bash
python -m workflows.preprocess
```

**Salida:**
- `data/processed/imbalance/` - Datos desbalanceo
- `data/processed/misalignment/` - Datos desalineación
- Estadísticas de normalización
- Detección de valores atípicos

---

### 3️⃣ Entrenar Modelo de Residuos

```bash
python -m workflows.train_model
```

**Parámetros (en `configs/config.py`):**
```python
POLYNOMIAL_DEGREE = 3  # Grado del polinomio
TEST_SIZE = 0.2        # Proporción test
RANDOM_STATE = 42
```

**Salida:**
- `models/trained/residuals_CSP_v3.pkl` - Modelo serializado
- Métricas de ajuste por sensor
- Gráficas de residuos en `mlartifacts/`
- Experimento registrado en MLflow

---

### 4️⃣ Entrenar Clasificadores

```bash
python -m workflows.train_classifier
```

**Métodos entrenados:**
1. **Linear** - Interpolación basada en percentiles
2. **Logistic** - Regresión logística (sklearn)
3. **GMM** - Gaussian Mixture Model

**Salida:**
- `models/trained/classifier_*.pkl` - 3 clasificadores
- `models/trained/classifier_best.pkl` - Mejor modelo (por test accuracy)
- `models/trained/best_classifier_metadata.json` - Metadata del mejor
- Comparativas en MLflow (train vs test, ROC curves, etc.)

**Selección del mejor:**
```
Si test_accuracy igual: Logistic > GMM > Linear
Detecta overfitting automáticamente (gap > 0.15)
```

---

### 5️⃣ Hacer Predicciones

```bash
python -m workflows.predict_anomalies
```

**Entrada:** Archivo CSV en `data/processed/imbalance/`

**Salida:**
- Clasificación global (Desbalanceo/Desalineación)
- Probabilidades (P(Desbal), P(Desalin))
- Severidad por sensor (Verde/Amarillo/Rojo)
- Gráficas en `models/predictions/`
- Reporte JSON con resultados

**Ejemplo salida:**
```json
{
  "prediction": "DESBALANCEO",
  "confidence": 0.98,
  "probabilities": {
    "desbalanceo": 0.98,
    "desalineacion": 0.02
  },
  "severity": {
    "CSP": "VERDE",
    "CSL": "AMARILLO",
    "CTP": "VERDE",
    "CTL": "ROJO"
  }
}
```

---

### 6️⃣ Ejecutar Interfaz Web (Streamlit)

```bash
streamlit run app/main.py
```

**URL Local:** `http://localhost:8501`

**Tabs:**
1. **Predicción Global**
   - Clasificación y confianza
   - Distribución de fenómenos (puntos de desbalanceo vs desalineación)
   - Información del análisis

2. **Gráficas por Sensor**
   - Datos reales vs predicción (scatter plot)
   - Ajuste polinómico (línea roja)
   - Residuos (relleno gris)
   - Colorbar con magnitud de residuos

3. **Severidad Detallada**
   - Tabla por sensor con valoración
   - Resumen de estados (Verde/Amarillo/Rojo)
   - Recomendaciones automáticas

**Uso:**
1. Cargar archivo CSV desde sidebar
2. Esperar procesamiento
3. Ver análisis en los tabs

---

## 🐳 Docker & MLflow

### Iniciar Servicios (Dev)

```bash
docker-compose up -d
```

**Servicios:**
- **MLflow**: `http://localhost:5000` - Tracking de experimentos
- **API**: `http://localhost:8000` - (Legado, no en uso actualmente)

**Volumes:**
```
./mlruns -> /mlflow/mlruns              (Backend store)
./mlartifacts -> /mlflow/mlartifacts    (Artifact store)
./ -> /app                               (Código)
./data -> /app/data                      (Datos)
```

### Detener Servicios

```bash
docker-compose down
```

### Ver Logs

```bash
docker-compose logs -f mlflow
docker-compose logs -f api
```

---

## 🤖 API REST (Legado)

> **Nota:** La API FastAPI ya no está en uso. Toda la lógica está en Streamlit.
> Se mantiene aquí para referencia histórica.

### Endpoint: POST `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/processed/imbalance/archivo.csv"
```

**Response:**
```json
{
  "prediction": "DESBALANCEO",
  "confidence": 0.95,
  "probabilities": {
    "desbalanceo": 0.95,
    "desalineacion": 0.05
  },
  "metadata": {
    "samples_analyzed": 1135,
    "nominal_speed": 279.18,
    "sensors": ["CSP", "CSL", "CTP", "CTL"],
    "sensor_data": {
      "CSP": {
        "original": [...],
        "predicted": [...],
        "mean_residual": 1.3029
      }
    }
  },
  "severity": {
    "CSP": "VERDE",
    "CSL": "AMARILLO",
    "CTP": "VERDE",
    "CTL": "ROJO"
  }
}
```

---

## 📊 Workflows de CI/CD

### Workflows Actuales

**`.github/workflows/`:**

- `preprocess_on_data_change.yml` - Dispara preprocesamiento al cambiar datos
- `preprocess_on_pipeline_change.yml` - Dispara al cambiar pipeline.py

### Workflows Pendientes (TODO)

Los siguientes workflows necesitan completarse:

```yaml
# 1. test_on_pr.yml
# Ejecuta pytest cuando hay PR
# - Validar sintaxis
# - Pruebas unitarias
# - Lint (flake8, black)

# 2. train_model_scheduled.yml
# Entrenamiento automático semanal
# - Trigger: cron (semanal)
# - Entrenar modelo residuos
# - Entrenar clasificadores
# - Comparar con anterior
# - Notificar resultados

# 3. deploy_streamlit.yml
# Deploy automático a Streamlit Cloud
# - Trigger: push a main
# - Verificar tests
# - Deploy a producción
# - Verificar salud

# 4. data_validation.yml
# Validación de datos nuevos
# - Trigger: nuevos CSVs en data/raw
# - Validar formato
# - Detectar anomalías
# - Alertar si hay problemas

# 5. model_registry.yml
# Registro de modelos
# - Trigger: nuevo best classifier
# - Guardar en model registry
# - Versionado (MLflow)
# - Tracking de performance
```

---

## 🌐 Deployment

### Streamlit Cloud (Producción)

**URL:** [Turbina Anomaly Detector](https://hydro-turb-ai-anomaly-hpzpvsmfjrv4gdlcxyxvjg.streamlit.app/)

**Pasos para Deploying:**

1. **Conectar GitHub a Streamlit Cloud**
   ```
   https://share.streamlit.io/ -> "New app" -> Seleccionar repo
   ```

2. **Configurar**
   ```
   - Repository: tu-usuario/hydro-turb-ai-anomaly
   - Branch: main
   - Main file path: app/main.py
   - Python version: 3.11
   ```

3. **Environment (Secrets)**
   ```
   # .streamlit/secrets.toml
   MLFLOW_TRACKING_URI = "http://localhost:5000"
   ```

4. **Deploy**
   - Automático con cada push a `main`
   - Logs en Streamlit dashboard

---

## 📚 Documentación Técnica

### Modelo de Residuos

**Clase:** `DataResidualsProcessor` (`src/models/residuals_model.py`)

```python
# Entrada: DataFrame con sensores + KPH
# Proceso:
# 1. Por cada sensor:
#    - Ajuste polinomio cúbico (KPH vs amplitud)
#    - Predicción = polinomio(KPH)
#    - Residuo = amplitud real - predicción
# 2. Retorna matriz de residuos (n_samples, n_sensores)

# Salida: Residuos, Columnas, KPH, Datos, Predicciones
```

**Uso:**
```python
from src.models.residuals_model import DataResidualsProcessor

model = DataResidualsProcessor.load("models/trained/residuals_CSP_v3.pkl")
residuals, cols, kph, data, pred = model.calculate_residuals_global(df)
```

---

### Clasificadores

**Clase:** `AnomalyClassifier` (`src/models/classifier.py`)

**Métodos:**

| Método | Parámetro | Descripción |
|--------|-----------|-------------|
| Linear | N/A | Umbrales percentil (p25/p75) |
| Logistic | C=1.0 | Regresión logística sklearn |
| GMM | n_components=2 | Gaussian Mixture Model |

**Probabilidades:**
```python
# Todos retornan P(Desalineación)
# P(Desbalanceo) = 1 - P(Desalineación)

y_proba = classifier.predict_proba(X_test)  # shape: (n, 1)
```

---

### Severidad

**Clase:** `VibrationSeverityChecker` (`src/models/vibration_severity_checker.py`)

**Umbrales por Sensor (Francis Horizontal):**

| Sensor | Verde | Amarillo | Rojo |
|--------|-------|----------|------|
| CSP    | ≤60   | 60-100   | >100 |
| CSL    | ≤70   | 70-110   | >110 |
| CTP    | ≤80   | 80-120   | >120 |
| CTL    | ≤2.5  | 2.5-5    | >5   |

**Configurables en `configs/config.py`:**
```python
SEVERITY_THRESHOLDS = {
    "Francis horizontal": {
        "CSP": {"verde": 60, "amarillo": 100},
        "CSL": {"verde": 70, "amarillo": 110},
        # ...
    }
}
```

---

### Estructura de Datos

**Entrada CSV (raw):**
```csv
Fecha,KPH,CSP,CSL,CTP,CTL
2024-01-15 10:30:00,100.5,65.2,72.1,85.3,2.1
2024-01-15 10:31:00,100.6,65.4,72.3,85.5,2.0
...
```

**Salida Predicción:**
```python
{
    "prediction": "DESBALANCEO",               # Clasificación global
    "confidence": 0.95,                        # Confianza del mejor model
    "probabilities": {
        "desbalanceo": 0.95,
        "desalineacion": 0.05
    },
    "metadata": {
        "samples_analyzed": 1135,
        "nominal_speed": 279.18,
        "sensors": ["CSP", "CSL", "CTP", "CTL"],
        "n_anomalies": 835,
        "sensor_data": {
            "CSP": {
                "original": [65.2, 65.4, ...],
                "predicted": [64.1, 64.3, ...],
                "residual": [1.1, 1.1, ...],
                "abs_residual": [1.1, 1.1, ...],
                "mean_residual": 1.3029
            }
        },
        "kph": [100.5, 100.6, ...]
    },
    "severity": {
        "CSP": "VERDE",
        "CSL": "AMARILLO",
        "CTP": "VERDE",
        "CTL": "ROJO"
    }
}
```

---

## 🔍 Comandos Útiles

### Development

```bash
# Linting
flake8 src/ workflows/ app/

# Format
black src/ workflows/ app/

# Tests (cuando estén implementados)
pytest tests/ -v

# Ver estructura
tree -L 3 -I '__pycache__|*.pyc|.venv'
```

### MLflow

```bash
# Abrir dashboard
mlflow ui --backend-store-uri file:./mlruns

# Ver experimentos
mlflow experiments list

# Ver runs de un experimento
mlflow runs list --experiment-name "classifier_training"
```

### Streamlit

```bash
# Dev local
streamlit run app/main.py

# Deploy (si está conectado)
streamlit deploy

# Clear cache
streamlit cache clear
```

---

## 🚨 Troubleshooting

### Problema: "No hay datos gráficos para sensor X"

**Causa:** `sensor_data` no está siendo poblado correctamente

**Solución:**
```python
# Verificar que TurbinePredictor devuelve sensor_data
result = predictor.predict(temp_path)
assert "sensor_data" in result["metadata"]
```

### Problema: Severidad muestra 0.00

**Causa:** `max_values` no está en el nivel correcto

**Solución:**
```python
# max_values debe estar en result, no en metadata
max_vals = result.get("max_values", {})  # Correcto
# NO
max_vals = result["metadata"].get("max_values", {})  # Incorrecto
```

### Problema: MLflow no conecta desde Docker

**Causa:** URL de MLflow incorrecta

**Solución:**
```python
# Dentro del container, usar nombre del service
import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
```

### Problema: Port 8000/5000 ya en uso

**Solución:**
```bash
# Linux/macOS
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 📝 Próximos Pasos

- [ ] Implementar workflows de CI/CD completos
- [ ] Agregar pruebas unitarias (`tests/`)
- [ ] Documentación de API OpenAPI (si se reactiva)
- [ ] Dashboard adicional con histórico de predicciones
- [ ] Alerts automáticos por correo si Rojo
- [ ] Versionado de modelos en Production
- [ ] Monitoreo de data drift

---

## 👥 Contribuciones

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nombre`)
3. Commit cambios (`git commit -am 'Agregar feature'`)
4. Push a rama (`git push origin feature/nombre`)
5. Abrir Pull Request

---

## 📄 Licencia

MIT License - Ver `LICENSE` para detalles

---

## 📧 Contacto

Para preguntas o issues:
- Abrir GitHub Issue
- Contactar equipo de desarrollo

---

**Última actualización:** Noviembre 2025  
**Versión:** 1.0.0
