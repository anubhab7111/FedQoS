import os
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.pipeline import load_federated_data
from src.models.lstm import build_model
from src.fl.server import GlobalEvaluator
from src.fl.utils import get_parameters, set_parameters, set_seed
from src.training.train import train

def run_simulation(config, device=None, checkpoint_dir=None, history_path=None, verbose=True, patience=None):
    """Run FedAvg in a single process, one client after another.

    Uses the same data pipeline, local training loop and server-side evaluator
    as the gRPC scripts, without the networking. Intended for experiments and
    sweeps. If `patience` is set, training stops once the validation loss has
    not improved for that many rounds. Returns (summary, history) from the evaluator.
    """
    seed = config.get("seed")
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)

    client_loaders, global_loaders, scalers = load_federated_data(config)
    num_clients = len(client_loaders)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        joblib.dump(scalers, os.path.join(checkpoint_dir, "scalers.joblib"))
    fed_cfg = config["federation"]
    # Same sampling rule as Flower's FedAvg
    num_fit = max(int(num_clients * fed_cfg["fraction_fit"]), fed_cfg["min_fit_clients"])

    global_model = build_model(config)
    local_model = build_model(config).to(device)
    criterion = nn.MSELoss()
    evaluator = GlobalEvaluator(
        global_model, global_loaders["val"], global_loaders["test"], criterion, scalers["target"], device,
        config["data"]["target_cols"], checkpoint_dir=checkpoint_dir, history_path=history_path
    )

    parameters = get_parameters(global_model)
    evaluator(0, parameters, {})

    for server_round in range(1, fed_cfg["num_rounds"] + 1):
        updates, sizes = [], []
        for cid in rng.sample(range(num_clients), num_fit):
            set_parameters(local_model, parameters)
            optimizer = optim.Adam(local_model.parameters(), lr=config["training"]["learning_rate"])
            trainloader = client_loaders[cid]["train"]
            train(local_model, trainloader, criterion, optimizer, config["training"]["local_epochs"], device)
            updates.append(get_parameters(local_model))
            sizes.append(len(trainloader.dataset))

        # Weighted average of client parameters (FedAvg)
        weights = np.array(sizes) / sum(sizes)
        parameters = [sum(w * layer for w, layer in zip(weights, layers)).astype(layers[0].dtype) for layers in zip(*updates)]

        val_loss, metrics = evaluator(server_round, parameters, {})
        if verbose:
            r2 = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("val_r2"))
            print(f"Round {server_round}: val_loss={val_loss:.5f}, {r2}", flush=True)
        if patience is not None and server_round - evaluator.best["round"] >= patience:
            break

    return evaluator.summary(), evaluator.history
