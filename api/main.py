import torch
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
from src.config import load_config
from src.data.pipeline import get_smooth_cols
from src.data.preprocessing import smooth_features
from src.models.lstm import build_model
import joblib # for loading scalers

app = FastAPI(title="NR-V2X QoS Prediction API")

# Global variables for model and scalers
model = None
feature_scaler = None
target_scaler = None
config = None

@app.on_event("startup")
def load_resources():
    global model, feature_scaler, target_scaler, config
    
    config = load_config()

    checkpoint_path = os.path.join(config["deployment"]["checkpoint_dir"], config["deployment"].get("inference_checkpoint", "global_model_best.pt"))
    if os.path.exists(checkpoint_path):
        model = build_model(config)
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
    
    scaler_path = os.path.join(config["deployment"]["checkpoint_dir"], "scalers.joblib")
    if os.path.exists(scaler_path):
        scalers = joblib.load(scaler_path)
        feature_scaler = scalers["feature"]
        target_scaler = scalers["target"]

@app.get("/health")
def health_check():
    ready = model is not None and feature_scaler is not None and target_scaler is not None
    return {
        "status": "healthy" if ready else "not_ready",
        "model_loaded": model is not None,
        "scalers_loaded": feature_scaler is not None and target_scaler is not None,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=503, detail="Model or scalers not loaded")
    
    # Validate before the try block so client errors are not turned into 500s.
    # The model uses the last `window_len` rows. Up to `filter_len - 1` earlier rows
    # may be sent as well, so the rolling mean of the first rows has its full history.
    data_cfg = config["data"]
    window_len, input_size = data_cfg["window_len"], config["model"]["input_size"]
    max_rows = window_len + data_cfg["filter_len"] - 1
    try:
        window_data = np.array(request.window, dtype=float)
    except ValueError:
        window_data = None
    if (window_data is None or window_data.ndim != 2 or window_data.shape[1] != input_size
            or not window_len <= window_data.shape[0] <= max_rows):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window shape. Expected {window_len} to {max_rows} rows of {input_size} values"
        )
    if not np.isfinite(window_data).all():
        raise HTTPException(status_code=400, detail="Window contains NaN or infinite values")

    try:
        # Smooth the raw measurements exactly as the training pipeline does
        frame = pd.DataFrame(window_data, columns=data_cfg["feature_cols"])
        frame = smooth_features(frame, get_smooth_cols(data_cfg), data_cfg["filter_len"])
        window_data = frame.values[-window_len:]

        # Scale input
        window_scaled = feature_scaler.transform(window_data)
        window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            prediction_scaled = model(window_tensor).squeeze(0).numpy()
        
        # Inverse scale output
        prediction_unscaled = target_scaler.inverse_transform(prediction_scaled)
        # Throughput and delay cannot be negative; the unconstrained model output can be
        prediction_unscaled = np.clip(prediction_unscaled, 0, None)
        
        target_cols = config["data"]["target_cols"]
        return PredictionResponse(
            throughput=prediction_unscaled[:, target_cols.index("throughput")].tolist(),
            delay=prediction_unscaled[:, target_cols.index("delay")].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
