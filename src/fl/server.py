import flwr as fl
from typing import Dict, Optional, Tuple
from src.fl.utils import set_parameters
from src.training.train import validate

def get_on_fit_config_fn(local_epochs: int):
    """Return a function that prepares config for fit."""
    def fit_config(server_round: int) -> Dict[str, str]:
        return {"server_round": str(server_round), "local_epochs": str(local_epochs)}
    return fit_config

def get_evaluate_fn(model, testloader, criterion, target_scaler, device):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        set_parameters(model, parameters)
        loss, r2_scores = validate(model, testloader, criterion, target_scaler, device)
        return loss, r2_scores
    return evaluate

def create_strategy(initial_parameters, evaluate_fn, on_fit_config_fn, min_fit_clients, min_eval_clients, fraction_fit):
    """Create a FedAvg strategy."""
    return fl.server.strategy.FedAvg(
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
