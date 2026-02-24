from fastapi import FastAPI
import uvicorn
from loguru import logger
from src.pipeline import InferencePipeline
import numpy as np
import os
import random

app = FastAPI()

class DummyModel:
    def predict(self, obs, deterministic=True):
        return random.choice([0, 1, 2]), None

model = DummyModel()

pipeline = InferencePipeline(
    data_addr=os.environ.get("DATA_ADDR", "localhost:50051"),
    chart_addr=os.environ.get("CHART_ADDR", "localhost:50052"),
    vision_addr=os.environ.get("VISION_ADDR", "localhost:50053")
)

action_map = {
    0: "SAME",
    1: "DOWN",
    2: "UP"
}

@app.get("/api/v1/predict/current")
async def predict_current(symbol: str):
    logger.info(f"Predicting for {symbol}")
    
    features = pipeline.get_current_features(symbol)
    
    if not np.any(features):
        return {"prediction": "UNKNOWN", "confidence": 0.0, "status": "Pipeline services unavailable"}
        
    action, _states = model.predict(features, deterministic=True)
    predicted_action = action_map.get(int(action), "UNKNOWN")
    
    return {
        "prediction": predicted_action,
        "confidence": 0.85, 
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
