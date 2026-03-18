# MLtasks PyTorch Extensions

Four new PyTorch training and evaluation tasks extending the CoderGym MLtasks framework. This repository adds a **Time Series Forecasting** series with tasks ranging from baseline smoothing to uncertainty-aware production-style forecasting.

This repository is coursework for CMPE 258 Deep Learning at San Jose State University (Spring 2026).

## Overview

This repository adds four tasks that follow the specification defined in the original MLtasks project (https://github.com/lkk688/CoderGym/tree/main/MLtasks). Each task is implemented as a self-contained PyTorch script and is fully self-verifiable via exit status.

Each task:

* Trains and evaluates a model
* Prints training and validation metrics
* Asserts quality thresholds
* Returns `exit(0)` on success and non-zero on failure

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib (for visualizations)

## Task Details

#### `ts_lvl1_baseline_smoothing` (Level 1)

Exponential smoothing from scratch using PyTorch tensors. Grid search over α and β to minimize train MSE. Uses synthetic data (sinusoid + trend + noise). Outputs: `ts_lvl1_params.pt`, `ts_lvl1_metrics.json`.

#### `ts_lvl2_ar_autograd` (Level 2)

Autoregressive model predicting next value from past p values. Synthetic AR(2) process. Adam optimizer, device-agnostic. Outputs: `ts_lvl2_model.pt`, `ts_lvl2_metrics.json`, `ts_lvl2_forecast.png`.

#### `ts_lvl3_lstm_multistep` (Level 3)

LSTM sequence-to-sequence forecaster for multi-step ahead predictions. Synthetic sum of sinusoids. LR scheduler, gradient clipping, early stopping. Outputs: `ts_lvl3_model.pt`, `ts_lvl3_metrics.json`, `ts_lvl3_multistep.png`.

#### `ts_lvl4_uncertainty_production` (Level 4)

LSTM with Monte Carlo dropout for prediction intervals. EDA (series plot, lag correlation), residual-based interval calibration. Outputs: `ts_lvl4_model.pt`, `ts_lvl4_report.json`, `ts_lvl4_eda.png`, `forecast_with_intervals.png`.

## Running a Task

```bash
python tasks/<task_id>/task.py
```

Example:

```bash
python tasks/ts_lvl1_baseline_smoothing/task.py
python tasks/ts_lvl2_ar_autograd/task.py
python tasks/ts_lvl3_lstm_multistep/task.py
python tasks/ts_lvl4_uncertainty_production/task.py
```

Artifacts (models, metrics, plots) are written to `outputs/<task_id>/`.

# CUDA GEMM Benchmark

The cuda-benchmark folder contains code to benchmark a simple fully connected (FC) layer using:

- **PyTorch** (FP32 matmul with TF32 disabled vs enabled)
- **CUDA C++ / cuBLAS** (baseline `cublasSgemm` vs TF32 Tensor Core GEMM using `cublasGemmEx`)

Benchmarks sweep FC sizes, run warmup iterations, time with GPU events, and write results to disk for plotting.

Default benchmark configuration:

- **Batch size**: 256
- **Input features**: 4096
- **Output features**: 4096
- **Hidden sizes swept**: 512, 1024, 2048, 4096 (configurable via `--sizes`)
- **Warm-up iterations**: 10
- **Timed iterations**: 50 (average latency and TFLOP/s reported over these forwards)
