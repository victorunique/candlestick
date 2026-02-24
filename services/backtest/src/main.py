from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from loguru import logger

app = FastAPI()

class BacktestRequest(BaseModel):
    symbol: str
    start_time: str
    end_time: str

@app.post("/api/v1/backtest/run")
async def run_backtest(req: BacktestRequest):
    logger.info(f"Running backtest for {req.symbol} from {req.start_time} to {req.end_time}")
    
    # In practice: iteratively query Inference service and build returns array
    # Mock return
    return {
        "status": "success",
        "symbol": req.symbol,
        "metrics": {
            "accuracy": 0.55,
            "sharpe": 1.2,
            "max_drawdown": 0.15,
            "total_return": 0.05
        }
    }

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8003, reload=True)
