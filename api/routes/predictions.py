import os
import tempfile

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.models.turb_predictor import TurbinePredictor

router = APIRouter()
predictor = TurbinePredictor()


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    metadata: dict
    severity: dict


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


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo archivos CSV")
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        
        result = predictor.predict(temp_path)
        
        # ✅ Convertir numpy a tipos serializables
        sensor_data_clean = {}
        for sensor, data in result.get("sensor_data", {}).items():
            sensor_data_clean[sensor] = {
                "original": convert_numpy(data["original"]),
                "predicted": convert_numpy(data["predicted"]),
                "residual": convert_numpy(data["residual"]),
                "abs_residual": convert_numpy(data["abs_residual"]),
                "mean_residual": float(data["mean_residual"])
            }
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            metadata={
                **result["metadata"],
                "sensor_data": sensor_data_clean,
                "kph": convert_numpy(result.get("kph", [])),
                "max_values": convert_numpy(result.get("max_values", {}))
            },
            severity=result["severity"]
        )
    
    except Exception as e:
        import traceback
        print("=" * 70)
        print("ERROR EN PREDICCIÓN:")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    results = []
    
    for file in files:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_path = temp_file.name
            
            result = predictor.predict(temp_path)
            
            results.append({
                "filename": file.filename,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "severity": convert_numpy(result["severity"])
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    
    return {"results": results, "total": len(files)}
