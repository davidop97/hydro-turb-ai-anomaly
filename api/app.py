import sys
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from api.routes.health import router as health_router
from api.routes.prediction import router as prediction_router

app = FastAPI(
    title="Turbine Diagnostics API",
    description="API para diagnóstico de turbinas con ML",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Turbine Diagnostics API", "version": "1.0.0", "status": "running"}

app.include_router(health_router)
app.include_router(prediction_router)

# Entry point Uvicorn dev only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
