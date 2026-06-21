import os
import yaml
import torch
import torch.nn as nn
import flwr as fl
import joblib
from src.data.preprocessing import load_dataset, impute_missing, smooth_features, select_features
from src.data.dataset import create_autoregressive_dataset
from src.data.partitioning import create_federated_data
from src.models.lstm import LSTM
from src.fl.utils import get_parameters
from src.fl.server import create_strategy, get_evaluate_fn, get_on_fit_config_fn

def main():
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["deployment"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["deployment"]["results_dir"], exist_ok=True)

    # Load and preprocess
    df = load_dataset(config["data"]["dataset_path"])
    df = impute_missing(df)
    df = smooth_features(df, config["data"]["feature_cols"], config["data"]["filter_len"])
    
    # Create windows
    target_indices = [config["data"]["feature_cols"].index(name) for name in config["data"]["target_cols"]]
    windows, targets = create_autoregressive_dataset(
        df, config["data"]["feature_cols"], target_indices,
        future_window_len=config["data"]["future_win_len"],
        window_len=config["data"]["window_len"],
        stride=config["data"]["stride"]
    )
    
    # Partition for global evaluation
    client_loaders, global_test_loader = create_federated_data(
        windows, targets, config["federation"]["num_clients"], config["training"]["batch_size"]
    )
    
    # Save scalers from first client for inference (representative)
    scalers = {
        "feature": client_loaders[0]["feature_scaler"],
        "target": client_loaders[0]["target_scaler"]
    }
    joblib.dump(scalers, os.path.join(config["deployment"]["checkpoint_dir"], "scalers.joblib"))

    # Initialize model
    model = LSTM(
        input_size=config["model"]["input_size"],
        num_target_features=len(config["data"]["target_cols"]),
        future_window_len=config["data"]["future_win_len"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"]
    )
    
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model))
    
    # Server functions
    evaluate_fn = get_evaluate_fn(
        model, global_test_loader, nn.MSELoss(), client_loaders[0]["target_scaler"], device
    )
    on_fit_config_fn = get_on_fit_config_fn(config["training"]["local_epochs"])
    
    # Strategy
    strategy = create_strategy(
        initial_parameters, evaluate_fn, on_fit_config_fn,
        min_fit_clients=config["federation"]["min_fit_clients"],
        min_eval_clients=config["federation"]["num_clients"],
        fraction_fit=config["federation"]["fraction_fit"]
    )
    
    # Start server
    server_address = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")
    print(f"Starting server at {server_address}...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["federation"]["num_rounds"]),
        strategy=strategy,
    )
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config["deployment"]["checkpoint_dir"], "global_model.pt"))
    print("Model saved.")

if __name__ == "__main__":
    main()
