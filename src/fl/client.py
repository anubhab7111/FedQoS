import os
from datetime import datetime
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.fl.utils import get_parameters, set_parameters
from src.training.train import train, validate

class QosClient(fl.client.NumPyClient):
    """Flower client for QoS prediction."""
    def __init__(self, cid, model, trainloader, valloader, testloader, target_scaler, target_feature_names, learning_rate, device, results_dir):
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.target_scaler = target_scaler
        self.target_feature_names = target_feature_names
        self.device = device
        self.lr = learning_rate

        self.criterion = nn.MSELoss()

        self.last_train_loss = float("nan")
        self.last_fit_round = None
        # One log file per client run, so repeated runs never interleave.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(results_dir, f"client_{self.cid}_log_{run_id}.csv")
        self._initialize_log_file()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        # Fresh optimizer each round: Adam state from a previous local model
        # does not apply to the newly received global weights.
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        avg_loss = train(self.model, self.trainloader, self.criterion, optimizer, local_epochs, self.device)
        self.last_train_loss = avg_loss
        self.last_fit_round = int(config.get("server_round", 0))
        return get_parameters(self.model), len(self.trainloader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluate the global model on the local validation split; the test split is only logged."""
        set_parameters(self.model, parameters)
        val_loss, val_r2 = validate(self.model, self.valloader, self.criterion, self.target_scaler, self.device)
        test_loss, test_r2 = validate(self.model, self.testloader, self.criterion, self.target_scaler, self.device)
        server_round = int(config.get("server_round", 0))
        # Only report a train loss if this client actually trained this round.
        train_loss = self.last_train_loss if self.last_fit_round == server_round else float("nan")
        self._log_metrics_to_csv(server_round, train_loss, val_loss, val_r2, test_loss, test_r2)
        return float(val_loss), len(self.valloader.dataset), val_r2

    def _log_columns(self):
        columns = ["round", "train_loss"]
        for split in ("val", "test"):
            columns += [f"{split}_loss"] + [f"{split}_r2_{name}" for name in self.target_feature_names]
        return columns

    def _initialize_log_file(self):
        pd.DataFrame(columns=self._log_columns()).to_csv(self.log_file_path, index=False)

    def _log_metrics_to_csv(self, server_round, train_loss, val_loss, val_r2, test_loss, test_r2):
        row = {"round": server_round, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}
        for i, name in enumerate(self.target_feature_names):
            row[f"val_r2_{name}"] = val_r2.get(f'r2_feature_{i}', float("nan"))
            row[f"test_r2_{name}"] = test_r2.get(f'r2_feature_{i}', float("nan"))
        pd.DataFrame([row], columns=self._log_columns()).to_csv(self.log_file_path, mode='a', header=False, index=False)
