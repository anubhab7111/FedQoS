import torch
import numpy as np
from torch.utils.data import Dataset

class QosDataset(Dataset):
    """PyTorch Dataset for QoS data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_autoregressive_dataset(data, cols, indices, future_window_len=1, window_len=100, stride=10):
    """Extract sliding windows and the future target values that follow each one.

    Returns clean windows only. Noise augmentation must be applied after the
    train/test split (see `augment_with_noise`), otherwise noisy copies of test
    windows end up in the training data.

    `data` must be a single simulation run; use `create_run_datasets` for the
    full dataset.
    """
    X = data[cols].values
    T, D = X.shape

    n_windows = max(0, (T - window_len - future_window_len) // stride + 1)
    windows = np.zeros((n_windows, window_len, D))
    targets = np.zeros((n_windows, future_window_len, len(indices)))

    for i in range(n_windows):
        start = i * stride
        windows[i] = X[start : start + window_len]
        targets[i] = X[start + window_len : start + window_len + future_window_len, indices]

    return windows, targets

def augment_with_noise(windows, targets, indices, noise_level=0.02, rng=None):
    """Return a noisy copy of the given windows and targets.

    Noise is Gaussian, scaled by the per-feature std of the given windows.
    """
    if rng is None:
        rng = np.random.default_rng()
    col_stds = windows.reshape(-1, windows.shape[-1]).std(axis=0)
    windows_noisy = windows + rng.standard_normal(windows.shape) * (col_stds * noise_level)
    targets_noisy = targets + rng.standard_normal(targets.shape) * (col_stds[indices] * noise_level)
    return windows_noisy, targets_noisy

def create_run_datasets(data, cols, indices, future_window_len=1, window_len=100, stride=10):
    """Extract windows run by run and label each sample train / val / test.

    `data` needs a `run_id` column and a row-level `split` column (see
    `label_splits`). Windows never cross a run boundary. A sample belongs to the
    split that holds all of its target rows; samples whose targets straddle a
    split boundary are dropped. Because the splits are chronological, a training
    sample lies entirely in the training rows, so no validation or test row is
    ever seen in training. The input window of a validation or test sample may
    reach back into earlier rows, as it would in deployment.

    Returns (windows, targets, splits, runs), where `splits` holds the split
    name and `runs` the run id of every sample.
    """
    windows, targets, splits, runs = [], [], [], []
    for run_id, run in data.groupby('run_id', sort=True):
        X, y = create_autoregressive_dataset(run, cols, indices, future_window_len, window_len, stride)
        if len(X) == 0:
            continue  # run shorter than one sample

        row_split = run['split'].values
        target_start = np.arange(len(X)) * stride + window_len
        first, last = row_split[target_start], row_split[target_start + future_window_len - 1]
        keep = first == last

        windows.append(X[keep]); targets.append(y[keep]); splits.append(first[keep])
        runs.append(np.full(keep.sum(), run_id))

    return np.concatenate(windows), np.concatenate(targets), np.concatenate(splits), np.concatenate(runs)
