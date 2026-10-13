import copy
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from src.fl.simulation import run_simulation

@pytest.fixture
def client(tiny_config, monkeypatch):
    run_simulation(copy.deepcopy(tiny_config), device=torch.device("cpu"),
                   checkpoint_dir=tiny_config["deployment"]["checkpoint_dir"], verbose=False)
    monkeypatch.setenv("CONFIG_PATH", tiny_config["_path"])
    import api.main
    with TestClient(api.main.app) as test_client:
        yield test_client

def _rows(n):
    return np.random.default_rng(0).random((n, 3)).tolist()

def test_health_and_prediction_shape(client):
    assert client.get("/health").json()["status"] == "healthy"
    response = client.post("/predict", json={"window": _rows(8)})
    assert response.status_code == 200
    body = response.json()
    assert len(body["throughput"]) == 2 and len(body["delay"]) == 2

def test_accepts_extra_history_rows_for_smoothing(client):
    # window_len = 8 and filter_len = 4, so 8 to 11 rows are valid
    assert client.post("/predict", json={"window": _rows(11)}).status_code == 200
    assert client.post("/predict", json={"window": _rows(12)}).status_code == 400
    assert client.post("/predict", json={"window": _rows(7)}).status_code == 400
    assert client.post("/predict", json={"window": [[0.0, 0.0]] * 8}).status_code == 400

def test_predictions_are_never_negative(client, monkeypatch):
    # A model whose scaled output is -1 maps to values below zero in original units
    import api.main
    monkeypatch.setattr(api.main, "model", lambda x: torch.full((1, 2, 2), -1.0))
    response = client.post("/predict", json={"window": _rows(8)})
    assert response.status_code == 200
    assert response.json() == {"throughput": [0.0, 0.0], "delay": [0.0, 0.0]}
