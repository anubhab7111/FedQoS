import copy
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from src.config import validate_config
from src.fl.server import FedAvgWaitForClients
from src.fl.simulation import run_simulation
from src.training.train import validate

def test_validation_loss_is_the_per_sample_mean():
    # 10 samples in batches of 4: the last batch has 2 samples and must not count as a full one
    torch.manual_seed(0)
    X, y = torch.rand(10, 3, 2), torch.rand(10, 1, 2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(6, 2), nn.Unflatten(1, (1, 2)))
    scaler = MinMaxScaler().fit(y.reshape(-1, 2).numpy())
    loss, _ = validate(model, DataLoader(TensorDataset(X, y), batch_size=4), nn.MSELoss(), scaler, torch.device("cpu"))
    with torch.no_grad():
        expected = ((model(X) - y) ** 2).mean().item()
    assert loss == pytest.approx(expected, rel=1e-5)

def test_simulation_runs_and_is_repeatable(tiny_config):
    device = torch.device("cpu")
    summary, history = run_simulation(copy.deepcopy(tiny_config), device=device, verbose=False)
    assert [row["round"] for row in history] == [0, 1, 2]
    assert summary["final"]["round"] == 2
    assert summary["best"]["val_loss"] == min(row["val_loss"] for row in history[1:])

    _, repeat = run_simulation(copy.deepcopy(tiny_config), device=device, verbose=False)
    assert repeat == history

def test_config_validation_rejects_impossible_settings(tiny_config):
    validate_config(tiny_config)
    for section, name, value in [
        ("federation", "min_fit_clients", 4),   # more than num_clients
        ("federation", "fraction_fit", 0.0),
        ("model", "input_size", 5),             # does not match feature_cols
        ("data", "val_ratio", 0.9),             # val + test >= 1
    ]:
        bad = copy.deepcopy(tiny_config)
        bad[section][name] = value
        with pytest.raises(ValueError):
            validate_config(bad)

def test_strategy_waits_for_all_clients_before_sampling():
    class FakeClientManager:
        def __init__(self):
            self.connected, self.calls = 2, []
        def num_available(self):
            return self.connected
        def wait_for(self, num_clients, timeout=86400):
            self.calls.append("wait")
            self.connected = max(self.connected, num_clients)  # the remaining clients join
            return True
        def sample(self, num_clients, min_num_clients=None, criterion=None):
            self.calls.append(("sample", num_clients))
            return []

    strategy = FedAvgWaitForClients(fraction_fit=1.0, min_fit_clients=2, min_available_clients=5)
    manager = FakeClientManager()
    strategy.configure_fit(1, None, manager)
    assert manager.calls == ["wait", ("sample", 5)]
