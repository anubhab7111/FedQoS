import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

def train(model, trainloader, criterion, optimizer, epochs, device, scheduler=None):
    """Local training loop. Returns the mean per-sample loss of the last epoch."""
    model.to(device)
    model.train()
    epoch_loss = float("nan")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            # Weight by batch size: the last batch is usually smaller
            epoch_loss += loss.item() * len(X)

        if scheduler is not None:
            scheduler.step()
    
    return epoch_loss / len(trainloader.dataset)

def validate(model, valloader, criterion, target_scaler, device):
    """Validation loop. Returns the mean per-sample loss and the per-target R² in original units."""
    model.to(device)
    model.eval()
    epoch_loss = 0
    all_pred_unscaled = []
    all_targets_unscaled = []

    with torch.no_grad():
        for X, y in valloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            # Weight by batch size: the last batch is usually smaller
            epoch_loss += loss.item() * len(X)

            pred_cpu = pred.cpu().numpy()
            y_cpu = y.cpu().numpy()
            
            num_target_features = pred_cpu.shape[-1]
            pred_reshaped = pred_cpu.reshape(-1, num_target_features)
            y_reshaped = y_cpu.reshape(-1, num_target_features)

            all_pred_unscaled.append(target_scaler.inverse_transform(pred_reshaped))
            all_targets_unscaled.append(target_scaler.inverse_transform(y_reshaped))

    avg_loss = epoch_loss / len(valloader.dataset)
    final_preds = np.concatenate(all_pred_unscaled)
    final_targets = np.concatenate(all_targets_unscaled)

    r2_scores = {}
    for i in range(final_targets.shape[1]):
        r2_scores[f'r2_feature_{i}'] = r2_score(final_targets[:, i], final_preds[:, i])

    return avg_loss, r2_scores
