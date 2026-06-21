import torch
import yaml
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
from src.models.lstm import LSTM
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
    
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = os.path.join(config["deployment"]["checkpoint_dir"], "global_model.pt")
    if os.path.exists(checkpoint_path):
        model = LSTM(
            input_size=config["model"]["input_size"],
            num_target_features=len(config["data"]["target_cols"]),
            future_window_len=config["data"]["future_win_len"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"]
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
    
    scaler_path = os.path.join(config["deployment"]["checkpoint_dir"], "scalers.joblib")
    if os.path.exists(scaler_path):
        scalers = joblib.load(scaler_path)
        feature_scaler = scalers["feature"]
        target_scaler = scalers["target"]

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=503, detail="Model or scalers not loaded")
    
    try:
        window_data = np.array(request.window)
        if window_data.shape != (config["data"]["window_len"], config["model"]["input_size"]):
             raise HTTPException(status_code=400, detail=f"Invalid window shape. Expected ({config['data']['window_len']}, {config['model']['input_size']})")
        
        # Scale input
        window_scaled = feature_scaler.transform(window_data)
        window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            prediction_scaled = model(window_tensor).squeeze(0).numpy()
        
        # Inverse scale output
        prediction_unscaled = target_scaler.inverse_transform(prediction_scaled)
        
        return PredictionResponse(
            throughput=prediction_unscaled[:, 0].tolist(),
            delay=prediction_unscaled[:, 1].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
