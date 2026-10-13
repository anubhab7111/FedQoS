import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from src.data.dataset import QosDataset, augment_with_noise

def _scale(scaler, arr):
    """Apply a fitted scaler to a (samples, timesteps, features) array."""
    return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)

def assign_runs_to_clients(runs, splits, num_clients):
    """Give each client a contiguous group of whole runs with a similar number of training samples.

    Returns a dict mapping run id to client index.
    """
    run_ids = np.unique(runs)
    if len(run_ids) < num_clients:
        raise ValueError(f"Only {len(run_ids)} runs for {num_clients} clients. Reduce num_clients.")
    counts = np.array([np.sum((runs == r) & (splits == 'train')) for r in run_ids])
    # Position of each run's midpoint in the cumulative training-sample count
    midpoints = (np.cumsum(counts) - counts / 2) / counts.sum()
    clients = np.minimum((midpoints * num_clients).astype(int), num_clients - 1)
    return dict(zip(run_ids, clients))

def create_federated_data(all_windows, all_targets, splits, runs, num_clients, batch_size, target_indices,
                          noise_level=0.02, seed=None):
    """Split data across clients and prepare dataloaders.

    `splits` and `runs` come from `create_run_datasets`: every sample already
    carries its train/val/test label and the simulation run it was cut from.
    Each client gets a contiguous group of whole runs, so every client holds
    train, validation and test samples from the same runs. Noise augmentation
    is applied to the training windows only. All clients share one feature
    scaler and one target scaler, fitted on the clean training windows of all
    clients.

    The validation split is for model selection (e.g. picking the best round);
    the test split must only be used for reporting.

    Returns (client_loaders, global_loaders, scalers).
    """
    rng = np.random.default_rng(seed)
    run_to_client = assign_runs_to_clients(runs, splits, num_clients)
    sample_client = np.array([run_to_client[r] for r in runs])

    client_splits = []
    for i in range(num_clients):
        split = {}
        for name in ('train', 'val', 'test'):
            mask = (sample_client == i) & (splits == name)
            split[name] = (all_windows[mask], all_targets[mask])
        if any(len(X) == 0 for X, _ in split.values()):
            raise ValueError(
                f"Client {i} has an empty train, val or test split. "
                f"Reduce num_clients, stride or window sizes."
            )
        client_splits.append(split)

    # Shared scalers, fitted on clean training data only
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(np.concatenate([s['train'][0] for s in client_splits]).reshape(-1, all_windows.shape[-1]))
    target_scaler = MinMaxScaler()
    target_scaler.fit(np.concatenate([s['train'][1] for s in client_splits]).reshape(-1, all_targets.shape[-1]))

    client_loaders = []
    global_parts = {'val': [], 'test': []}

    for split in client_splits:
        X_train, y_train = split['train']
        if noise_level:
            X_noisy, y_noisy = augment_with_noise(X_train, y_train, target_indices, noise_level, rng)
            X_train = np.concatenate([X_train, X_noisy], axis=0)
            y_train = np.concatenate([y_train, y_noisy], axis=0)

        train_dataset = QosDataset(_scale(feature_scaler, X_train), _scale(target_scaler, y_train))
        loaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True)}

        for name in ('val', 'test'):
            X_scaled = _scale(feature_scaler, split[name][0])
            y_scaled = _scale(target_scaler, split[name][1])
            loaders[name] = DataLoader(QosDataset(X_scaled, y_scaled), batch_size=batch_size, shuffle=False)
            global_parts[name].append((X_scaled, y_scaled))

        client_loaders.append(loaders)

    # Create global val and test sets
    global_loaders = {}
    for name, parts in global_parts.items():
        dataset = QosDataset(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        global_loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scalers = {"feature": feature_scaler, "target": target_scaler}
    return client_loaders, global_loaders, scalers
