import os
import yaml

def validate_config(config):
    """Raise ValueError listing every setting that would make a run crash or hang later."""
    data, model, training, fed = config["data"], config["model"], config["training"], config["federation"]
    problems = []

    if model["input_size"] != len(data["feature_cols"]):
        problems.append(f"model.input_size ({model['input_size']}) must equal the number of data.feature_cols ({len(data['feature_cols'])})")
    missing = [c for c in data["target_cols"] if c not in data["feature_cols"]]
    if missing:
        problems.append(f"data.target_cols {missing} must also be listed in data.feature_cols")
    for name in ("filter_len", "window_len", "future_win_len", "stride"):
        if data[name] < 1:
            problems.append(f"data.{name} must be at least 1")
    val_ratio, test_ratio = data.get("val_ratio", 0.2), data.get("test_ratio", 0.2)
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        problems.append("data.val_ratio and data.test_ratio must be positive and sum to less than 1")

    for name in ("batch_size", "local_epochs"):
        if training[name] < 1:
            problems.append(f"training.{name} must be at least 1")
    if training["learning_rate"] <= 0:
        problems.append("training.learning_rate must be positive")

    if fed["num_clients"] < 1 or fed["num_rounds"] < 1:
        problems.append("federation.num_clients and federation.num_rounds must be at least 1")
    if not 1 <= fed["min_fit_clients"] <= fed["num_clients"]:
        problems.append(f"federation.min_fit_clients ({fed['min_fit_clients']}) must be between 1 and "
                        f"federation.num_clients ({fed['num_clients']}): the server would wait forever for clients that do not exist")
    if not 0 < fed["fraction_fit"] <= 1:
        problems.append("federation.fraction_fit must be in (0, 1]")

    if problems:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(problems))
    return config

def load_config(path=None):
    """Load and validate the YAML config. Defaults to $CONFIG_PATH, then configs/default.yaml."""
    path = path or os.getenv("CONFIG_PATH", "configs/default.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return validate_config(config)
