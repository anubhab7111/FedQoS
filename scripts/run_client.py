import os
import yaml
import torch
import flwr as fl
from src.data.preprocessing import load_dataset, impute_missing, smooth_features
from src.data.dataset import create_autoregressive_dataset
from src.data.partitioning import create_federated_data
from src.models.lstm import LSTM
from src.fl.client import QosClient

def main():
    cid = os.getenv("CLIENT_ID", "0")
    server_address = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["deployment"]["results_dir"], exist_ok=True)

    # Load and preprocess (each client currently loads full dataset and partitions)
    # In a real scenario, the client would have its own data file.
    df = load_dataset(config["data"]["dataset_path"])
    df = impute_missing(df)
    df = smooth_features(df, config["data"]["feature_cols"], config["data"]["filter_len"])
    
    target_indices = [config["data"]["feature_cols"].index(name) for name in config["data"]["target_cols"]]
    windows, targets = create_autoregressive_dataset(
        df, config["data"]["feature_cols"], target_indices,
        future_window_len=config["data"]["future_win_len"],
        window_len=config["data"]["window_len"],
        stride=config["data"]["stride"]
    )
    
    client_loaders, _ = create_federated_data(
        windows, targets, config["federation"]["num_clients"], config["training"]["batch_size"]
    )
    
    client_data = client_loaders[int(cid)]
    
    # Initialize model
    model = LSTM(
        input_size=config["model"]["input_size"],
        num_target_features=len(config["data"]["target_cols"]),
        future_window_len=config["data"]["future_win_len"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"]
    )
    
    # Start client
    client = QosClient(
        cid=cid,
        model=model,
        trainloader=client_data["train"],
        testloader=client_data["test"],
        target_scaler=client_data["target_scaler"],
        target_feature_names=config["data"]["target_cols"],
        learning_rate=config["training"]["learning_rate"],
        device=device,
        results_dir=config["deployment"]["results_dir"]
    )
    
    print(f"Starting client {cid} connecting to {server_address}...")
    fl.client.start_client(server_address=server_address, client=client.to_client())

if __name__ == "__main__":
    main()
