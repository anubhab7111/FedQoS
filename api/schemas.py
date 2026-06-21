from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    window: List[List[float]] # Expected shape: (window_len, input_size)

class PredictionResponse(BaseModel):
    throughput: List[float] # Expected shape: (future_win_len)
    delay: List[float]      # Expected shape: (future_win_len)
