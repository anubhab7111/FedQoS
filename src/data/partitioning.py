import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from src.data.dataset import QosDataset

def create_federated_data(all_windows, all_targets, num_clients, batch_size, train_test_split_ratio=0.8):
    """Split data across clients and prepare dataloaders."""
    client_loaders = []
    global_test_X_parts = []
    global_test_y_parts = []
    
    num_samples = len(all_windows)
    samples_per_client = num_samples // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples

        client_windows = all_windows[start_idx:end_idx]
        client_targets = all_targets[start_idx:end_idx]

        if len(client_windows) < 2:
            continue

        # Chronological split
        X_train, X_test, y_train, y_test = train_test_split(
            client_windows, client_targets, train_size=train_test_split_ratio, shuffle=False
        )

        # Scale input features
        scaler = MinMaxScaler()
        n_train, n_timesteps, n_features = X_train.shape
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, n_timesteps, n_features)
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], n_timesteps, n_features)

        # Scale targets
        target_scaler = MinMaxScaler()
        n_train_y, n_timesteps_y, n_features_y = y_train.shape
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, n_features_y)).reshape(n_train_y, n_timesteps_y, n_features_y)
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, n_features_y)).reshape(y_test.shape[0], n_timesteps_y, n_features_y)

        train_dataset = QosDataset(X_train_scaled, y_train_scaled)
        test_dataset = QosDataset(X_test_scaled, y_test_scaled)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        client_loaders.append({
            'train': train_loader,
            'test': test_loader,
            'target_scaler': target_scaler,
            'feature_scaler': scaler
        })

        global_test_X_parts.append(X_test_scaled)
        global_test_y_parts.append(y_test_scaled)

    # Create global test set
    global_test_X = np.concatenate(global_test_X_parts, axis=0)
    global_test_y = np.concatenate(global_test_y_parts, axis=0)
    global_test_dataset = QosDataset(global_test_X, global_test_y)
    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)

    return client_loaders, global_test_loader
