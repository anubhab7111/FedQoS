import torch
import yaml
import os
from src.models.lstm import LSTM

def export_to_onnx():
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = os.path.join(config["deployment"]["checkpoint_dir"], "global_model.pt")
    onnx_path = os.path.join(config["deployment"]["checkpoint_dir"], "global_model.onnx")
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    model = LSTM(
        input_size=config["model"]["input_size"],
        num_target_features=len(config["data"]["target_cols"]),
        future_window_len=config["data"]["future_win_len"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"]
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    # Dummy input for ONNX export
    dummy_input = torch.randn(1, config["data"]["window_len"], config["model"]["input_size"])
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
