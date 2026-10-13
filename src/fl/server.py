import os
import json
import torch
import pandas as pd
import flwr as fl
from typing import Dict, Optional, Tuple
from src.fl.utils import set_parameters
from src.training.train import validate

BEST_CHECKPOINT = "global_model_best.pt"
FINAL_CHECKPOINT = "global_model_final.pt"

class GlobalEvaluator:
    """Server-side evaluation that tracks and saves the best and the final global model.

    The best round is the one with the lowest loss on the global validation set.
    Test metrics are recorded every round for reporting but never used for selection.
    Call it like a Flower `evaluate_fn`.
    """
    def __init__(self, model, val_loader, test_loader, criterion, target_scaler, device,
                 target_feature_names, checkpoint_dir=None, history_path=None):
        self.model = model
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.target_scaler = target_scaler
        self.device = device
        self.target_feature_names = target_feature_names
        self.checkpoint_dir = checkpoint_dir
        self.history_path = history_path
        self.history = []
        self.best = None

    def __call__(self, server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        set_parameters(self.model, parameters)
        val_loss, val_r2 = validate(self.model, self.val_loader, self.criterion, self.target_scaler, self.device)
        test_loss, test_r2 = validate(self.model, self.test_loader, self.criterion, self.target_scaler, self.device)

        row = {"round": server_round, "val_loss": float(val_loss), "test_loss": float(test_loss)}
        for i, name in enumerate(self.target_feature_names):
            row[f"val_r2_{name}"] = float(val_r2[f"r2_feature_{i}"])
            row[f"test_r2_{name}"] = float(test_r2[f"r2_feature_{i}"])
        self.history.append(row)

        # Round 0 is the untrained initial model, so it is never a candidate.
        if server_round > 0:
            if self.best is None or row["val_loss"] < self.best["val_loss"]:
                self.best = row
                self._save(BEST_CHECKPOINT)
            self._save(FINAL_CHECKPOINT)
        if self.history_path is not None:
            pd.DataFrame(self.history).to_csv(self.history_path, index=False)
        if self.checkpoint_dir is not None:
            with open(os.path.join(self.checkpoint_dir, "training_summary.json"), "w") as f:
                json.dump(self.summary(), f, indent=2)

        metrics = {k: v for k, v in row.items() if k not in ("round", "val_loss")}
        return row["val_loss"], metrics

    def _save(self, filename):
        if self.checkpoint_dir is not None:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, filename))

    def summary(self):
        """Metrics of the best (by validation loss) and the final round."""
        trained = [row for row in self.history if row["round"] > 0]
        return {"best": self.best, "final": trained[-1] if trained else None}

def get_on_fit_config_fn(local_epochs: int):
    """Return a function that prepares config for fit."""
    def fit_config(server_round: int) -> Dict[str, str]:
        return {"server_round": str(server_round), "local_epochs": str(local_epochs)}
    return fit_config

class FedAvgWaitForClients(fl.server.strategy.FedAvg):
    """FedAvg that waits for `min_available_clients` before sizing a round.

    Plain FedAvg works out how many clients to sample from the number connected
    at that instant and only then waits for the rest, so the first round can
    train fewer clients than `fraction_fit` asks for.
    """
    def configure_fit(self, server_round, parameters, client_manager):
        client_manager.wait_for(self.min_available_clients)
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        client_manager.wait_for(self.min_available_clients)
        return super().configure_evaluate(server_round, parameters, client_manager)

def create_strategy(initial_parameters, evaluate_fn, on_fit_config_fn, min_fit_clients, min_eval_clients, fraction_fit):
    """Create a FedAvg strategy."""
    return FedAvgWaitForClients(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_eval_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_fit_config_fn,
    )
