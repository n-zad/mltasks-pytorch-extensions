"""
Time Series Level 2: Autoregressive Model (AR with Autograd)

AR(p) model: x_t = sum_{i=1}^{p} w_i * x_{t-i} + bias
  = w^T * [x_{t-p}, ..., x_{t-1}] + b

Implemented via nn.Linear(p, 1) with Adam optimizer.
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def get_task_metadata():
    """Return task metadata for pytorch_task_v1."""
    return {
        "task_id": "ts_lvl2_ar_autograd",
        "series": "Time Series Forecasting",
        "level": 2,
        "algorithm": "Autoregressive Model (AR with Autograd)",
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return available device (CUDA/CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_sliding_windows(series: np.ndarray, p: int):
    """Create [x_{t-p}, ..., x_{t-1}] -> x_t pairs."""
    X, y = [], []
    for i in range(p, len(series)):
        X.append(series[i - p : i])
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_dataloaders(cfg=None):
    """
    Create synthetic AR process or use classic dataset.
    Sliding window: [x_{t-p}, ..., x_{t-1}] -> x_t.
    """
    cfg = cfg or {}
    p = cfg.get("p", 2)
    n_points = cfg.get("n_points", 400)
    train_ratio = cfg.get("train_ratio", 0.8)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Synthetic AR(2) process: x_t = 0.6*x_{t-1} + 0.3*x_{t-2} + noise
    series = np.zeros(n_points)
    series[:2] = np.random.randn(2)
    for t in range(2, n_points):
        series[t] = 0.6 * series[t - 1] + 0.3 * series[t - 2] + 0.05 * np.random.randn()
    series = series.astype(np.float32)

    X, y = _create_sliding_windows(series, p)
    split_idx = int(len(X) * train_ratio)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))

    batch_size = cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, {"p": p}


def build_model(cfg=None):
    """AR(p) model: nn.Linear(p, 1)."""
    cfg = cfg or {}
    p = cfg.get("p", 2)
    device = get_device()
    model = nn.Sequential(nn.Linear(p, 1))
    return model.to(device)


def train(model, train_loader, cfg=None):
    """Train AR model with Adam."""
    cfg = cfg or {}
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.01))
    loss_fn = nn.MSELoss()

    epochs = cfg.get("epochs", 500)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(train_loader))

    return {"loss_history": loss_history}


def evaluate(model, train_result, val_loader, cfg=None):
    """Compute MSE, MAE, R2 on validation."""
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu())
            all_targets.append(y_batch)
    preds = torch.cat(all_preds).squeeze()
    targets = torch.cat(all_targets).squeeze()

    mse = torch.mean((targets - preds) ** 2).item()
    mae = torch.mean(torch.abs(targets - preds)).item()
    var = torch.var(targets).item()
    r2 = 1 - (mse / var) if var > 1e-8 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "preds": preds,
        "targets": targets,
    }


def predict(model, data_loader, cfg=None):
    """Return predictions for given data."""
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in data_loader:
            X = batch[0].to(device)
            pred = model(X)
            preds.append(pred.cpu())
    return torch.cat(preds).squeeze()


def save_artifacts(model, train_result, val_metrics, output_dir, cfg=None):
    """Save model, forecast plot, and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "ts_lvl2_model.pt"))

    fig, ax = plt.subplots(figsize=(10, 4))
    n = len(val_metrics["targets"])
    ax.plot(range(n), val_metrics["targets"].numpy(), label="Actual", alpha=0.8)
    ax.plot(range(n), val_metrics["preds"].numpy(), label="Predicted", alpha=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("AR Model: Actual vs Predicted (Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ts_lvl2_forecast.png"), dpi=100)
    plt.close()

    metrics = {
        "loss_history": train_result["loss_history"],
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
    }
    with open(os.path.join(output_dir, "ts_lvl2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    set_seed(42)
    cfg = {
        "p": 2,
        "n_points": 400,
        "train_ratio": 0.8,
        "epochs": 500,
        "lr": 0.01,
        "output_dir": "outputs/ts_lvl2_ar_autograd",
    }

    train_loader, val_loader, data_info = make_dataloaders(cfg)
    model = build_model({**cfg, **data_info})

    train_result = train(model, train_loader, cfg)
    val_metrics = evaluate(model, train_result, val_loader, cfg)

    print("=" * 50)
    print("Time Series Lvl 2: AR(p) with Autograd")
    print("=" * 50)
    print(f"Val MSE: {val_metrics['mse']:.6f}")
    print(f"Val MAE: {val_metrics['mae']:.6f}")
    print(f"Val R2:  {val_metrics['r2']:.6f}")
    print("=" * 50)

    assert val_metrics["r2"] > 0.6, f"R2 {val_metrics['r2']:.4f} should be > 0.6"
    print("Assertion passed: R2 > 0.6")

    save_artifacts(model, train_result, val_metrics, cfg["output_dir"], cfg)
    print(f"Artifacts saved to {cfg['output_dir']}")
    exit(0)
