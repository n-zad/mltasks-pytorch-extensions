"""
Time Series Level 1: Exponential Smoothing Baseline

Exponential smoothing recurrence:
    s_t = alpha * x_t + (1 - alpha) * s_{t-1}

Single exponential: one-step-ahead forecast = s_{t-1}
Double exponential (Holt): adds trend component b_t = beta * (s_t - s_{t-1}) + (1 - beta) * b_{t-1}
"""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_task_metadata():
    """Return task metadata for pytorch_task_v1."""
    return {
        "task_id": "ts_lvl1_baseline_smoothing",
        "series": "Time Series Forecasting",
        "level": 1,
        "algorithm": "Time Series (Exponential Smoothing Baseline)",
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return available device (CPU for this task - no gradient-based training)."""
    return torch.device("cpu")


def make_dataloaders(cfg=None):
    """
    Create synthetic time series: sinusoid + trend + Gaussian noise.
    80% train / 20% validation split.
    """
    cfg = cfg or {}
    n_points = cfg.get("n_points", 500)
    train_ratio = cfg.get("train_ratio", 0.8)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    t = np.linspace(0, 4 * np.pi, n_points)
    series = np.sin(t) + 0.1 * t + np.random.randn(n_points) * 0.3
    series = series.astype(np.float32)

    split_idx = int(n_points * train_ratio)
    train_series = torch.tensor(series[:split_idx])
    val_series = torch.tensor(series[split_idx:])

    train_ds = TensorDataset(train_series.unsqueeze(1))
    val_ds = TensorDataset(val_series.unsqueeze(1))

    batch_size = cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, {"train_series": train_series, "val_series": val_series}


def build_model(cfg=None):
    """Return parametric smoothing config. train() selects best alpha/beta via grid search."""
    cfg = cfg or {}
    return {
        "alpha": cfg.get("alpha", 0.3),
        "beta": cfg.get("beta", 0.1),
        "use_double": cfg.get("use_double", True),
    }


def _exponential_smoothing(series: torch.Tensor, alpha: float, beta: float = None, use_double: bool = False):
    """
    Exponential smoothing: s_t = alpha * x_t + (1-alpha) * s_{t-1}
    Returns smoothed series and one-step-ahead forecasts.
    """
    x = series.squeeze().numpy()
    n = len(x)
    s = np.zeros(n)
    s[0] = x[0]

    for t in range(1, n):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]

    forecasts = np.roll(s, 1)
    forecasts[0] = x[0]

    if use_double and beta is not None:
        b = np.zeros(n)
        b[0] = 0
        for t in range(1, n):
            b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]
        # One-step-ahead: forecast[t] = s[t-1] + b[t-1]
        f = np.zeros(n)
        f[0] = x[0]
        for t in range(1, n):
            f[t] = s[t - 1] + b[t - 1]
        forecasts = f

    return torch.tensor(s, dtype=torch.float32), torch.tensor(forecasts, dtype=torch.float32)


def train(model, train_loader, cfg=None):
    """Grid search over alpha (and beta) to minimize train MSE; update model with best params."""
    cfg = cfg or {}
    data = train_loader.dataset.tensors[0].squeeze()
    use_double = model.get("use_double", True)

    alpha_grid = cfg.get("alpha_grid", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    beta_grid = cfg.get("beta_grid", [0.05, 0.1, 0.2, 0.3]) if use_double else [0.1]

    best_mse = float("inf")
    best_alpha, best_beta = model["alpha"], model.get("beta", 0.1)

    for alpha in alpha_grid:
        for beta in beta_grid:
            _, forecasts = _exponential_smoothing(data, alpha, beta, use_double)
            mse = torch.mean((data[1:] - forecasts[1:]) ** 2).item()
            if mse < best_mse:
                best_mse = mse
                best_alpha, best_beta = alpha, beta

    model["alpha"] = best_alpha
    model["beta"] = best_beta

    smoothed, forecasts = _exponential_smoothing(data, best_alpha, best_beta, use_double)
    mae = torch.mean(torch.abs(data[1:] - forecasts[1:])).item()

    return {
        "smoothed_history": smoothed,
        "forecasts": forecasts,
        "train_mse": best_mse,
        "train_mae": mae,
    }


def evaluate(model, train_result, val_loader, cfg=None):
    """Compute MSE and MAE on validation one-step forecasts."""
    val_series = val_loader.dataset.tensors[0].squeeze()
    alpha = model["alpha"]
    beta = model.get("beta", 0.1)
    use_double = model.get("use_double", True)

    smoothed, forecasts = _exponential_smoothing(val_series, alpha, beta, use_double)
    targets = val_series[1:]
    preds = forecasts[1:]

    mse = torch.mean((targets - preds) ** 2).item()
    mae = torch.mean(torch.abs(targets - preds)).item()
    var = torch.var(val_series).item()
    r2 = 1 - (mse / var) if var > 1e-8 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "smoothed_history": smoothed,
        "forecasts": forecasts,
    }


def predict(model, data_loader, cfg=None):
    """Return one-step-ahead forecasts for given data."""
    data = data_loader.dataset.tensors[0].squeeze()
    alpha = model["alpha"]
    beta = model.get("beta", 0.1)
    use_double = model.get("use_double", True)
    _, forecasts = _exponential_smoothing(data, alpha, beta, use_double)
    return forecasts


def save_artifacts(model, train_result, val_metrics, output_dir, cfg=None):
    """Save learned parameters (alpha, beta) with torch.save(); metrics to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    params = {"alpha": model["alpha"], "beta": model["beta"], "use_double": model["use_double"]}
    torch.save(params, os.path.join(output_dir, "ts_lvl1_params.pt"))

    metrics = {
        "train_mse": train_result["train_mse"],
        "train_mae": train_result["train_mae"],
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
    }
    with open(os.path.join(output_dir, "ts_lvl1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return {"smoothed_history": train_result["smoothed_history"].tolist(), "val_metrics": val_metrics}


if __name__ == "__main__":
    set_seed(42)
    cfg = {
        "n_points": 500,
        "train_ratio": 0.8,
        "use_double": True,
        "output_dir": "outputs/ts_lvl1_baseline_smoothing",
    }

    train_loader, val_loader, data_info = make_dataloaders(cfg)
    model = build_model(cfg)

    train_result = train(model, train_loader, cfg)
    val_metrics = evaluate(model, train_result, val_loader, cfg)

    print("=" * 50)
    print("Time Series Lvl 1: Exponential Smoothing")
    print("=" * 50)
    print(f"Learned params: alpha={model['alpha']:.2f}, beta={model['beta']:.2f}")
    print(f"Train MSE: {train_result['train_mse']:.6f}")
    print(f"Train MAE: {train_result['train_mae']:.6f}")
    print(f"Val MSE:   {val_metrics['mse']:.6f}")
    print(f"Val MAE:   {val_metrics['mae']:.6f}")
    print(f"Val R2:    {val_metrics['r2']:.6f}")
    print("=" * 50)

    series_var = torch.var(data_info["val_series"]).item()
    assert val_metrics["mse"] < series_var, f"MSE {val_metrics['mse']:.4f} should be < variance {series_var:.4f}"
    print("Assertion passed: MSE < variance of series")

    output_dir = cfg["output_dir"]
    save_artifacts(model, train_result, val_metrics, output_dir, cfg)
    print(f"Artifacts saved to {output_dir}")
    exit(0)
