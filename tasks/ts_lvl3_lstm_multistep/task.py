"""
Time Series Level 3: LSTM Multi-Step Forecasting

LSTM cell equations (simplified):
  f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)  # forget gate
  i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)  # input gate
  c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)
  c_t = f_t * c_{t-1} + i_t * c_tilde
  o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
  h_t = o_t * tanh(c_t)

Multi-step: sequence [x_1..x_T] -> predict [x_{T+1}, ..., x_{T+H}]
Teacher forcing: use true targets during training.
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
        "task_id": "ts_lvl3_lstm_multistep",
        "series": "Time Series Forecasting",
        "level": 3,
        "algorithm": "LSTM Multi-Step Forecasting",
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_sequences(series: np.ndarray, seq_len: int, horizon: int):
    """Create (input_seq, target_seq) pairs. Target is next 'horizon' values."""
    X, y = [], []
    for i in range(len(series) - seq_len - horizon + 1):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len : i + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_dataloaders(cfg=None):
    """Synthetic: sum of sinusoids + noise. seq_len -> horizon."""
    cfg = cfg or {}
    seq_len = cfg.get("seq_len", 24)
    horizon = cfg.get("horizon", 5)
    n_points = cfg.get("n_points", 600)
    train_ratio = cfg.get("train_ratio", 0.8)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    t = np.linspace(0, 12 * np.pi, n_points)
    series = np.sin(t) + 0.5 * np.sin(2 * t) + np.random.randn(n_points) * 0.2
    series = series.astype(np.float32)

    mean, std = series.mean(), series.std()
    if std < 1e-8:
        std = 1.0
    series = (series - mean) / std

    X, y = _create_sequences(series, seq_len, horizon)
    split_idx = int(len(X) * train_ratio)
    X_train, y_train = X[:, :, np.newaxis], y  # (N, seq_len, 1), (N, horizon)
    X_val, y_val = X[split_idx:], y[split_idx:]
    X_val = X_val[:, :, np.newaxis]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    batch_size = cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, {
        "seq_len": seq_len,
        "horizon": horizon,
        "mean": mean,
        "std": std,
    }


class LSTMForecaster(nn.Module):
    """LSTM that predicts horizon steps. Teacher forcing during training."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=1, horizon=5, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


def build_model(cfg=None):
    """Build LSTM forecaster."""
    cfg = cfg or {}
    model = LSTMForecaster(
        input_size=1,
        hidden_size=cfg.get("hidden_size", 64),
        num_layers=cfg.get("num_layers", 1),
        horizon=cfg.get("horizon", 5),
        dropout=cfg.get("dropout", 0.1),
    )
    return model.to(get_device())


def train(model, train_loader, cfg=None):
    """Train with Adam, LR scheduler, gradient clipping, early stopping."""
    cfg = cfg or {}
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    loss_fn = nn.MSELoss()

    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 15)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    train_loss_history = []
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        # Validation for early stopping (if val_loader provided in cfg)
        val_loader = cfg.get("val_loader")
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch)
                    val_loss += loss_fn(pred, y_batch).item()
            val_loss /= len(val_loader)
        else:
            val_loss = avg_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    return {"loss_history": train_loss_history}


def evaluate(model, train_result, val_loader, cfg=None):
    """MSE per horizon; compare to naive (last-seen) baseline."""
    device = next(model.parameters()).device
    model.eval()
    horizon = model.horizon
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu())
            all_targets.append(y_batch)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse_per_horizon = []
    for h in range(horizon):
        mse_h = torch.mean((targets[:, h] - preds[:, h]) ** 2).item()
        mse_per_horizon.append(mse_h)

    # Naive baseline: predict last value of input sequence for all horizons
    naive_preds = []
    for X_batch, y_batch in val_loader:
        last_vals = X_batch[:, -1, 0]  # (B,)
        naive_preds.append(last_vals.unsqueeze(1).expand(-1, horizon))
    naive_preds = torch.cat(naive_preds)
    naive_mse_h1 = torch.mean((targets[:, 0] - naive_preds[:, 0]) ** 2).item()
    model_mse_h1 = mse_per_horizon[0]

    return {
        "mse_per_horizon": mse_per_horizon,
        "mse": mse_per_horizon[0],
        "naive_mse_h1": naive_mse_h1,
        "preds": preds,
        "targets": targets,
    }


def predict(model, data_loader, cfg=None):
    """Return multi-step predictions."""
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.append(pred.cpu())
    return torch.cat(preds)


def save_artifacts(model, train_result, val_metrics, output_dir, cfg=None):
    """Save ts_lvl3_multistep.png (true vs predicted for horizons 1, 3, 5)."""
    os.makedirs(output_dir, exist_ok=True)

    preds = val_metrics["preds"]
    targets = val_metrics["targets"]
    horizon = preds.shape[1]
    plot_horizons = [0, min(2, horizon - 1), min(4, horizon - 1)]  # 1, 3, 5 (0-indexed)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, h in enumerate(plot_horizons):
        if h >= horizon:
            continue
        ax = axes[idx]
        ax.scatter(targets[:, h].numpy(), preds[:, h].numpy(), alpha=0.5, s=10)
        min_val = min(targets[:, h].min().item(), preds[:, h].min().item())
        max_val = max(targets[:, h].max().item(), preds[:, h].max().item())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Horizon {h + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("LSTM Multi-Step: Actual vs Predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ts_lvl3_multistep.png"), dpi=100)
    plt.close()

    torch.save(model.state_dict(), os.path.join(output_dir, "ts_lvl3_model.pt"))

    metrics = {
        "loss_history": train_result["loss_history"],
        "mse_per_horizon": val_metrics["mse_per_horizon"],
        "naive_mse_h1": val_metrics["naive_mse_h1"],
    }
    with open(os.path.join(output_dir, "ts_lvl3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    set_seed(42)
    cfg = {
        "seq_len": 24,
        "horizon": 5,
        "n_points": 600,
        "epochs": 100,
        "lr": 0.001,
        "patience": 15,
        "output_dir": "outputs/ts_lvl3_lstm_multistep",
    }

    train_loader, val_loader, data_info = make_dataloaders(cfg)
    model = build_model({**cfg, **data_info})
    cfg["val_loader"] = val_loader

    train_result = train(model, train_loader, cfg)
    val_metrics = evaluate(model, train_result, val_loader, cfg)

    print("=" * 50)
    print("Time Series Lvl 3: LSTM Multi-Step")
    print("=" * 50)
    for h, mse in enumerate(val_metrics["mse_per_horizon"]):
        print(f"Horizon {h + 1} MSE: {mse:.6f}")
    print(f"Naive (last-seen) MSE h1: {val_metrics['naive_mse_h1']:.6f}")
    print("=" * 50)

    assert val_metrics["mse"] < val_metrics["naive_mse_h1"], (
        f"LSTM horizon-1 MSE {val_metrics['mse']:.4f} should improve over naive {val_metrics['naive_mse_h1']:.4f}"
    )
    print("Assertion passed: LSTM horizon-1 MSE improves over naive baseline")

    save_artifacts(model, train_result, val_metrics, cfg["output_dir"], cfg)
    print(f"Artifacts saved to {cfg['output_dir']}")
    exit(0)
