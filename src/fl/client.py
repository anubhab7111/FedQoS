import os
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
    def __init__(self, cid, model, trainloader, testloader, target_scaler, target_feature_names, learning_rate, device, results_dir):
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.target_scaler = target_scaler
        self.target_feature_names = target_feature_names
        self.device = device
        self.lr = learning_rate
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.last_train_loss = -1.0
        self.log_file_path = os.path.join(results_dir, f"client_{self.cid}_log.csv")
        self._initialize_log_file()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        avg_loss = train(self.model, self.trainloader, self.criterion, self.optimizer, local_epochs, self.device)
        self.last_train_loss = avg_loss
        return get_parameters(self.model), len(self.trainloader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        val_loss, r2_scores = validate(self.model, self.testloader, self.criterion, self.target_scaler, self.device)
        server_round = int(config.get("server_round", 0))
        self._log_metrics_to_csv(server_round, self.last_train_loss, val_loss, r2_scores)
        return float(val_loss), len(self.testloader.dataset), r2_scores

    def _initialize_log_file(self):
        if not os.path.exists(self.log_file_path):
            headers = ["round", "train_loss", "val_loss"] + [f"r2_{name}" for name in self.target_feature_names]
            pd.DataFrame(columns=headers).to_csv(self.log_file_path, index=False)

    def _log_metrics_to_csv(self, server_round, train_loss, val_loss, r2_scores):
        row = {"round": server_round, "train_loss": train_loss, "val_loss": val_loss}
        for i, name in enumerate(self.target_feature_names):
            row[f"r2_{name}"] = r2_scores.get(f'r2_feature_{i}', -1.0)
        pd.DataFrame([row]).to_csv(self.log_file_path, mode='a', header=False, index=False)
