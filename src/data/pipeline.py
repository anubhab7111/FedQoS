from src.data.preprocessing import (
    load_dataset, assign_run_ids, drop_end_markers, drop_zero_runs, label_splits, impute_missing, smooth_features
)
from src.data.dataset import create_run_datasets
from src.data.partitioning import create_federated_data

def get_smooth_cols(data_cfg):
    """Columns that get the rolling mean. Targets are not smoothed by default: a rolling
    mean over the targets makes each future value largely a function of values already
    in the input window."""
    return data_cfg.get("smooth_cols", [c for c in data_cfg["feature_cols"] if c not in data_cfg["target_cols"]])

def load_federated_data(config):
    """Run the full data pipeline from the config.

    Shared by the server and the clients so both build identical partitions.
    Returns (client_loaders, global_loaders, scalers).
    """
    data_cfg = config["data"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]

    df = load_dataset(data_cfg["dataset_path"])
    # The file is many simulation runs back to back; everything below works run by run.
    df = assign_run_ids(df, data_cfg.get("run_cols", []))
    df = drop_end_markers(df, data_cfg.get("max_position_step"))
    if data_cfg.get("drop_zero_runs", True):
        df = drop_zero_runs(df, target_cols)
    df = label_splits(df, data_cfg.get("val_ratio", 0.2), data_cfg.get("test_ratio", 0.2))
    df = impute_missing(df, fit_rows=df["split"] == "train")
    df = smooth_features(df, get_smooth_cols(data_cfg), data_cfg["filter_len"])

    target_indices = [feature_cols.index(name) for name in target_cols]
    windows, targets, splits, runs = create_run_datasets(
        df, feature_cols, target_indices,
        future_window_len=data_cfg["future_win_len"],
        window_len=data_cfg["window_len"],
        stride=data_cfg["stride"]
    )

    return create_federated_data(
        windows, targets, splits, runs, config["federation"]["num_clients"], config["training"]["batch_size"],
        target_indices,
        noise_level=data_cfg.get("noise_level", 0.02),
        seed=config.get("seed")
    )
