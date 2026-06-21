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

def sliding_window_augmentation(df, cols, indices, future_window_len=1, L=100, s=10, add_noise=True, noise_level=0.02):
    """Extract sliding windows and optionally add noise for augmentation."""
    X = df[cols].values
    T, D = X.shape
    
    n_windows = max(0, (T - L - future_window_len) // s)
    if n_windows <= 0:
        return np.array([]), np.array([])

    windows = np.zeros((n_windows, L, D))
    targets = np.zeros((n_windows, future_window_len, len(indices)))
    
    col_stds = df[cols].std().values

    for i in range(n_windows):
        start = i * s
        windows[i] = X[start : start + L].copy()
        targets[i] = X[start + L : start + L + future_window_len, indices].copy()

        if add_noise:
            # Note: In the notebook, noise was added to a separate copy. 
            # This function can be called twice or modified. 
            # For simplicity, we'll let the orchestrator handle the noise copy.
            pass

    return windows, targets

def create_autoregressive_dataset(data, cols, indices, future_window_len=1, window_len=100, stride=10, add_noise=True, noise_level=0.02):
    """Orchestrate window creation and augmentation."""
    windows, targets = sliding_window_augmentation(
        data, cols, indices, future_window_len=future_window_len, L=window_len, s=stride, add_noise=False
    )
    
    if add_noise:
        windows_noisy, targets_noisy = sliding_window_augmentation(
            data, cols, indices, future_window_len=future_window_len, L=window_len, s=stride, add_noise=False
        )
        # Add noise to the copies
        col_stds = data[cols].std().values
        for i in range(len(windows_noisy)):
            noise_X = np.random.randn(*windows_noisy[i].shape) * (col_stds * noise_level)
            windows_noisy[i] += noise_X
            noise_y = np.random.randn(*targets_noisy[i].shape) * (col_stds[indices] * noise_level)
            targets_noisy[i] += noise_y
            
        windows = np.concatenate([windows, windows_noisy], axis=0)
        targets = np.concatenate([targets, targets_noisy], axis=0)
        
    return windows, targets
