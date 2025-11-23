import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from src.models.predict import TurbinePredictor

from src.preprocessing.pipeline import TurbineDataPipeline

router = APIRouter()
predictor = TurbinePredictor()

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    metadata: dict

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo archivos CSV")
    try:
        df = pd.read_csv(file.file)
        pipeline = TurbineDataPipeline()
        df_processed, nominal_speed, max_values = pipeline.fit_transform_dataframe(df)
        result = predictor.predict(df_processed)
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            metadata={"nominal_speed": nominal_speed, "samples_analyzed": len(df_processed)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            df = pd.read_csv(file.file)
            pipeline = TurbineDataPipeline()
            df_processed, _, _ = pipeline.fit_transform_dataframe(df)
            result = predictor.predict(df_processed)
            results.append({
                "filename": file.filename,
                "prediction": result['prediction'],
                "confidence": result['confidence']
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return {"results": results, "total": len(files)}
