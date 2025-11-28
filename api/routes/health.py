import mlflow
from fastapi import APIRouter

from configs.config import settings
from src.models.turb_predictor import TurbinePredictor

router = APIRouter()
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
predictor = TurbinePredictor()

@router.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": predictor.is_loaded()}
