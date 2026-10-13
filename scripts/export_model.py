import os
import sys

# Ensure project root is on sys.path so `from src...` imports work
# when running this script directly (e.g. `python scripts/export_model.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from src.config import load_config
from src.models.lstm import build_model

def export_to_onnx():
    config = load_config()
    
    checkpoint_path = os.path.join(config["deployment"]["checkpoint_dir"], config["deployment"].get("inference_checkpoint", "global_model_best.pt"))
    onnx_path = os.path.splitext(checkpoint_path)[0] + ".onnx"
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    model = build_model(config)
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
