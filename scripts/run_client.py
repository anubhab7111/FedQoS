import os
import sys

# Ensure project root is on sys.path so `from src...` imports work
# when running this script directly (e.g. `python scripts/run_client.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import socket
import time
import torch
import flwr as fl
from src.config import load_config
from src.data.pipeline import load_federated_data
from src.models.lstm import build_model
from src.fl.client import QosClient
from src.fl.utils import set_seed

def wait_for_server(server_address, timeout=600):
    """Block until the server accepts connections; a client started first would otherwise crash."""
    host, port = server_address.rsplit(":", 1)
    deadline = time.time() + timeout
    while True:
        try:
            socket.create_connection((host, int(port)), timeout=2).close()
            return
        except OSError:
            if time.time() > deadline:
                raise TimeoutError(f"No server reachable at {server_address} after {timeout}s")
            time.sleep(2)

def main():
    cid = os.getenv("CLIENT_ID", "0")
    server_address = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["deployment"]["results_dir"], exist_ok=True)

    # Load and preprocess (each client currently loads full dataset and partitions)
    # In a real scenario, the client would have its own data file.
    client_loaders, _, scalers = load_federated_data(config)
    
    if not 0 <= int(cid) < len(client_loaders):
        raise ValueError(f"CLIENT_ID={cid} is out of range for num_clients={len(client_loaders)}")
    client_data = client_loaders[int(cid)]
    
    # Seed after data loading so each client shuffles its batches differently
    if config.get("seed") is not None:
        set_seed(config["seed"] + int(cid))

    # Initialize model
    model = build_model(config)
    
    # Start client
    client = QosClient(
        cid=cid,
        model=model,
        trainloader=client_data["train"],
        valloader=client_data["val"],
        testloader=client_data["test"],
        target_scaler=scalers["target"],
        target_feature_names=config["data"]["target_cols"],
        learning_rate=config["training"]["learning_rate"],
        device=device,
        results_dir=config["deployment"]["results_dir"]
    )
    
    print(f"Client {cid} waiting for server at {server_address}...")
    wait_for_server(server_address)
    print(f"Starting client {cid} connecting to {server_address}...")
    fl.client.start_client(server_address=server_address, client=client.to_client())

if __name__ == "__main__":
    main()
