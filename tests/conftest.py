import numpy as np
import pandas as pd
import pytest
import yaml

@pytest.fixture
def tiny_config(tmp_path):
    """A small synthetic dataset and a matching config that trains in a second on the CPU."""
    rng = np.random.default_rng(0)
    frames = []
    for setting in range(6):
        n = 120
        frames.append(pd.DataFrame({
            "Unnamed: 0": np.arange(n), "X": np.arange(n) * 10.0, "Y": 0.0,
            "throughput": setting + rng.random(n), "delay": 5 + rng.random(n) * 10, "sinr": rng.normal(0, 5, n),
            "mcs": setting,
        }))
    dataset_path = tmp_path / "dataset.csv"
    pd.concat(frames).to_csv(dataset_path, index=False)

    config = {
        "seed": 0,
        "data": {
            "dataset_path": str(dataset_path), "feature_cols": ["throughput", "sinr", "delay"],
            "target_cols": ["throughput", "delay"], "smooth_cols": ["sinr"], "filter_len": 4,
            "run_cols": ["mcs"], "max_position_step": 60, "drop_zero_runs": True,
            "window_len": 8, "future_win_len": 2, "stride": 2, "val_ratio": 0.2, "test_ratio": 0.2, "noise_level": 0.0,
        },
        "model": {"input_size": 3, "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "bidirectional": False},
        "training": {"batch_size": 16, "learning_rate": 0.01, "local_epochs": 1},
        "federation": {"num_clients": 3, "num_rounds": 2, "fraction_fit": 1.0, "min_fit_clients": 2},
        "deployment": {
            "checkpoint_dir": str(tmp_path / "checkpoints"), "results_dir": str(tmp_path / "results"),
            "inference_checkpoint": "global_model_best.pt",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config["_path"] = str(config_path)
    return config
