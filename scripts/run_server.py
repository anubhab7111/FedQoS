import os
import sys

# Ensure project root is on sys.path so `from src...` imports work
# when running this script directly (e.g. `python scripts/run_server.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import flwr as fl
import joblib
from src.config import load_config
from src.data.pipeline import load_federated_data
from src.models.lstm import build_model
from src.fl.utils import get_parameters, set_seed
from src.fl.server import GlobalEvaluator, create_strategy, get_on_fit_config_fn, BEST_CHECKPOINT, FINAL_CHECKPOINT

def main():
    config = load_config()
    set_seed(config.get("seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["deployment"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["deployment"]["results_dir"], exist_ok=True)

    # Load, preprocess and partition (only the global val/test sets are used here)
    _, global_loaders, scalers = load_federated_data(config)
    
    # Save the shared scalers for inference
    joblib.dump(scalers, os.path.join(config["deployment"]["checkpoint_dir"], "scalers.joblib"))

    # Initialize model
    model = build_model(config)
    
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model))
    
    # Server functions. The evaluator saves the best round (lowest validation
    # loss) and the final round as it goes.
    checkpoint_dir = config["deployment"]["checkpoint_dir"]
    evaluate_fn = GlobalEvaluator(
        model, global_loaders["val"], global_loaders["test"], nn.MSELoss(), scalers["target"], device,
        config["data"]["target_cols"], checkpoint_dir=checkpoint_dir,
        history_path=os.path.join(config["deployment"]["results_dir"], "server_history.csv")
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
    
    summary = evaluate_fn.summary()
    if summary["best"] is None:
        print("No training round completed; no model saved.")
        return
    print(f"Best round {summary['best']['round']} saved to {os.path.join(checkpoint_dir, BEST_CHECKPOINT)}: {summary['best']}")
    print(f"Final round {summary['final']['round']} saved to {os.path.join(checkpoint_dir, FINAL_CHECKPOINT)}: {summary['final']}")

if __name__ == "__main__":
    main()
