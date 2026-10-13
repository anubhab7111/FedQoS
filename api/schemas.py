from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    # Raw (unsmoothed) measurements, oldest first, one row per time step with the
    # columns of `data.feature_cols`. The API applies the rolling mean itself.
    # Send `window_len` rows, or up to `window_len + filter_len - 1` rows so the
    # rolling mean of the first rows has its full history.
    window: List[List[float]]

class PredictionResponse(BaseModel):
    throughput: List[float] # Expected shape: (future_win_len)
    delay: List[float]      # Expected shape: (future_win_len)
