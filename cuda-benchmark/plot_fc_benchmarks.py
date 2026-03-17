import argparse
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_pytorch_results(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cublas_results(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["batch_size"] = int(row["batch_size"])
            row["in_features"] = int(row["in_features"])
            row["hidden_features"] = int(row["hidden_features"])
            row["out_features"] = int(row["out_features"])
            row["avg_ms"] = float(row["avg_ms"])
            row["tflops"] = float(row["tflops"])
            row["tf32"] = bool(int(row["tf32"]))
            rows.append(row)
    return rows


def plot_metric_vs_hidden(results, backend, metric, out_path: Path, ylabel: str):
    hidden_sizes = sorted({r["hidden_features"] for r in results if r["backend"] == backend})
    if not hidden_sizes:
        return

    modes = ["cuda_core_fp32", "tensor_core_tf32"]
    plt.figure(figsize=(7, 4))

    for mode in modes:
        xs = []
        ys = []
        for h in hidden_sizes:
            matches = [
                r
                for r in results
                if r["backend"] == backend
                and r["mode"] == mode
                and r["hidden_features"] == h
            ]
            if not matches:
                continue
            xs.append(h)
            ys.append(matches[0][metric])
        if xs:
            label = f"{backend} - {mode}"
            plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Hidden size")
    plt.ylabel(ylabel)
    plt.title(f"{backend}: {ylabel} vs hidden size")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot FC GEMM TF32 vs FP32 benchmarks.")
    parser.add_argument(
        "--pytorch_json",
        type=str,
        default="pytorch_fc_benchmark.json",
        help="Path to PyTorch benchmark JSON.",
    )
    parser.add_argument(
        "--cublas_csv",
        type=str,
        default="cublas_fc_benchmark.csv",
        help="Path to cuBLAS benchmark CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Directory to write plots into.",
    )
    args = parser.parse_args()

    pytorch_results = load_pytorch_results(Path(args.pytorch_json))
    cublas_results = load_cublas_results(Path(args.cublas_csv))

    # Normalize field names to a common schema
    all_results = []
    for r in pytorch_results:
        all_results.append(
            {
                "backend": r["backend"],
                "mode": r["mode"],
                "batch_size": r["batch_size"],
                "in_features": r["in_features"],
                "hidden_features": r["hidden_features"],
                "out_features": r["out_features"],
                "avg_ms": r["avg_ms"],
                "tflops": r["tflops"],
                "tf32": bool(r["tf32"]),
            }
        )
    for r in cublas_results:
        all_results.append(r)

    out_dir = Path(args.out_dir)

    # Latency plots
    plot_metric_vs_hidden(
        all_results,
        backend="pytorch",
        metric="avg_ms",
        out_path=out_dir / "pytorch_latency_vs_hidden.png",
        ylabel="Avg latency per forward (ms)",
    )
    plot_metric_vs_hidden(
        all_results,
        backend="cublas",
        metric="avg_ms",
        out_path=out_dir / "cublas_latency_vs_hidden.png",
        ylabel="Avg latency per forward (ms)",
    )

    # Throughput plots
    plot_metric_vs_hidden(
        all_results,
        backend="pytorch",
        metric="tflops",
        out_path=out_dir / "pytorch_tflops_vs_hidden.png",
        ylabel="Throughput (TFLOP/s)",
    )
    plot_metric_vs_hidden(
        all_results,
        backend="cublas",
        metric="tflops",
        out_path=out_dir / "cublas_tflops_vs_hidden.png",
        ylabel="Throughput (TFLOP/s)",
    )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()

