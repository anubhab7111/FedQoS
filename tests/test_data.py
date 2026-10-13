import numpy as np
import pandas as pd
from src.data.preprocessing import (
    load_dataset, assign_run_ids, drop_end_markers, drop_zero_runs, label_splits, impute_missing, smooth_features
)
from src.data.dataset import create_autoregressive_dataset, create_run_datasets
from src.data.partitioning import create_federated_data

WINDOW_LEN, FUTURE_WIN_LEN, STRIDE = 15, 5, 5
RUN_LEN = 200

def _make_frame(n_runs=10):
    # Column "a" holds the row number, so every value identifies the row it came from.
    n_rows = n_runs * RUN_LEN
    df = pd.DataFrame({
        "a": np.arange(n_rows, dtype=float), "b": np.random.rand(n_rows),
        "setting": np.repeat(np.arange(n_runs), RUN_LEN),
    })
    return label_splits(assign_run_ids(df, ["setting"]), val_ratio=0.2, test_ratio=0.2)

def _make_samples(n_runs=10):
    return create_run_datasets(
        _make_frame(n_runs), ["a", "b"], [0], future_window_len=FUTURE_WIN_LEN, window_len=WINDOW_LEN, stride=STRIDE
    )

def test_window_count_and_alignment():
    df = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.random.rand(100)})
    windows, targets = create_autoregressive_dataset(
        df, ["a", "b"], [0], future_window_len=FUTURE_WIN_LEN, window_len=WINDOW_LEN, stride=STRIDE
    )
    assert len(windows) == (100 - WINDOW_LEN - FUTURE_WIN_LEN) // STRIDE + 1
    assert windows[-1, 0, 0] == (len(windows) - 1) * STRIDE
    # Targets start right after the window and the last one uses the final row
    assert targets[0, 0, 0] == WINDOW_LEN
    assert targets[-1, -1, 0] == 99

def test_duplicate_rows_are_kept(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"Unnamed: 0": [0, 1, 2, 0], "x": [1.0, 1.0, 1.0, 1.0]}).to_csv(path, index=False)
    df = load_dataset(str(path))
    assert len(df) == 4
    assert df["source_row"].tolist() == [0, 1, 2, 0]

def test_run_ids_and_zero_runs():
    df = pd.DataFrame({"setting": [1, 1, 2, 2, 1, 1], "y": [1.0, 2.0, 0.0, 0.0, 3.0, 0.0]})
    df = assign_run_ids(df, ["setting"])
    assert df["run_id"].tolist() == [0, 0, 1, 1, 2, 2]
    assert drop_zero_runs(df, ["y"])["run_id"].tolist() == [0, 0, 2, 2]

def test_new_source_file_starts_a_new_run():
    # Same settings throughout, but the per-file row counter restarts
    df = pd.DataFrame({"setting": [1] * 6, "source_row": [0, 1, 2, 0, 1, 2]})
    assert assign_run_ids(df, ["setting"])["run_id"].tolist() == [0, 0, 0, 1, 1, 1]

def test_end_marker_rows_are_dropped():
    df = pd.DataFrame({
        "setting": [1, 1, 1, 2, 2, 2],
        "X": [0.0, 10.0, 900.0, 500.0, 510.0, 520.0], "Y": [0.0] * 6,
    })
    df = drop_end_markers(assign_run_ids(df, ["setting"]), max_step=60)
    # The jump at the end of run 0 goes; the jump into run 1 is a new run, not a marker
    assert df["X"].tolist() == [0.0, 10.0, 500.0, 510.0, 520.0]

def test_split_labels_are_chronological_within_each_run():
    df = _make_frame(n_runs=2)
    for _, run in df.groupby("run_id"):
        assert run["split"].tolist() == ["train"] * 120 + ["val"] * 40 + ["test"] * 40

def test_imputation_is_fitted_on_training_rows_only():
    # jitter = 2 * delay on the training rows, 100 * delay elsewhere
    df = pd.DataFrame({
        "delay": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "jitter": [2.0, 4.0, np.nan, 400.0, 500.0, np.nan],
    })
    is_train = pd.Series([True, True, True, False, False, False])
    filled = impute_missing(df.copy(), fit_rows=is_train)
    assert np.allclose(filled["jitter"], [2.0, 4.0, 6.0, 400.0, 500.0, 12.0])

def test_smoothing_restarts_at_each_run():
    df = assign_run_ids(pd.DataFrame({"setting": [1, 1, 1, 2, 2, 2], "x": [0.0, 0.0, 0.0, 9.0, 9.0, 9.0]}), ["setting"])
    assert smooth_features(df, ["x"], 3)["x"].tolist() == [0.0, 0.0, 0.0, 9.0, 9.0, 9.0]

def test_samples_stay_inside_one_run():
    windows, targets, splits, runs = _make_samples()
    rows = np.concatenate([windows[:, :, 0], targets[:, :, 0]], axis=1)
    assert np.array_equal(rows.min(axis=1) // RUN_LEN, runs)
    assert np.array_equal(rows.max(axis=1) // RUN_LEN, runs)
    assert {"train", "val", "test"} == set(splits)

def test_no_val_or_test_rows_in_training():
    windows, targets, splits, runs = _make_samples()

    def rows(arr, name):
        return set(arr[splits == name][:, :, 0].ravel().astype(int))

    train_rows = rows(windows, "train") | rows(targets, "train")
    val_targets, test_targets = rows(targets, "val"), rows(targets, "test")
    assert not train_rows & val_targets
    assert not train_rows & test_targets
    assert not val_targets & test_targets
    # Validation inputs never include a test target row either
    assert not rows(windows, "val") & test_targets

def test_clients_get_whole_runs_and_noise_only_augments_training():
    windows, targets, splits, runs = _make_samples()
    kwargs = dict(num_clients=4, batch_size=16, target_indices=[0], seed=0)
    clean, clean_global, scalers = create_federated_data(windows, targets, splits, runs, noise_level=0.0, **kwargs)
    noisy, noisy_global, _ = create_federated_data(windows, targets, splits, runs, noise_level=0.02, **kwargs)

    seen_runs = set()
    for c, n in zip(clean, noisy):
        assert len(n["train"].dataset) == 2 * len(c["train"].dataset)
        for split in ("val", "test"):
            assert np.array_equal(n[split].dataset.X.numpy(), c[split].dataset.X.numpy())
        client_runs = set()
        for split in ("train", "val", "test"):
            a = scalers["feature"].inverse_transform(c[split].dataset.X.numpy().reshape(-1, 2))[:, 0]
            client_runs |= set(np.rint(a).astype(int) // RUN_LEN)
        assert not client_runs & seen_runs
        seen_runs |= client_runs
    assert seen_runs == set(range(10))

    assert np.array_equal(noisy_global["test"].dataset.y.numpy(), clean_global["test"].dataset.y.numpy())
    for split in ("val", "test"):
        assert len(clean_global[split].dataset) == sum(len(c[split].dataset) for c in clean)
