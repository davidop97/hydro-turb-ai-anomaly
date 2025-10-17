# create_pipeline.py
from turbine_pipeline import TurbineDataPipeline

# Crear el pipeline
pipeline = TurbineDataPipeline(
    speed_col='KPH',
    date_col='Fecha',
    max_speed_diff=1.0,
    min_stable_points=10,
    top_n_blocks=3,
    fall_threshold=0.95,
    min_consecutive=20,
    scale_method='minmax',
    trim_percentage=0.08
)

# Guardar el pipeline
pipeline.save("turbine_pipeline.joblib")

