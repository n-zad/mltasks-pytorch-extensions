import argparse
import json
import os
from typing import List, Dict

import torch
import torch.nn as nn


def run_fc_benchmark(
    sizes: List[int],
    batch_size: int,
    in_features: int,
    out_features: int,
    warmup_iters: int,
    timed_iters: int,
    device: str,
) -> List[Dict]:
    assert torch.cuda.is_available(), "CUDA GPU is required for this benchmark."
    torch.manual_seed(0)

    results = []

    # Save original global flags so we can restore
    orig_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_tf32_cudnn = torch.backends.cudnn.allow_tf32

    try:
        for mode_name, tf32_flag in [
            ("cuda_core_fp32", False),  # TF32 disabled, pure FP32 path
            ("tensor_core_tf32", True),  # TF32 enabled (Tensor Core path)
        ]:
            torch.backends.cuda.matmul.allow_tf32 = tf32_flag
            torch.backends.cudnn.allow_tf32 = tf32_flag

            for hidden_size in sizes:
                model = nn.Sequential(
                    nn.Linear(in_features, hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(hidden_size, out_features, bias=True),
                ).to(device)

                x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)

                # Warmup
                for _ in range(warmup_iters):
                    _ = model(x)

                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(timed_iters):
                    _ = model(x)
                end_event.record()

                torch.cuda.synchronize()
                total_ms = start_event.elapsed_time(end_event)
                avg_ms = total_ms / timed_iters

                # Compute FLOPs: two GEMMs (input->hidden, hidden->output)
                # GEMM FLOPs ≈ 2 * M * K * N (mul + add). Here, M=batch, K=in, N=hidden/out.
                m = batch_size
                k1 = in_features
                n1 = hidden_size
                k2 = hidden_size
                n2 = out_features
                flops_per_forward = 2.0 * m * k1 * n1 + 2.0 * m * k2 * n2
                t_sec = avg_ms / 1e3
                tflops = (flops_per_forward / t_sec) / 1e12

                results.append(
                    {
                        "backend": "pytorch",
                        "mode": mode_name,
                        "batch_size": batch_size,
                        "in_features": in_features,
                        "hidden_features": hidden_size,
                        "out_features": out_features,
                        "avg_ms": avg_ms,
                        "tflops": tflops,
                        "tf32": tf32_flag,
                    }
                )

    finally:
        # Restore global flags
        torch.backends.cuda.matmul.allow_tf32 = orig_tf32_matmul
        torch.backends.cudnn.allow_tf32 = orig_tf32_cudnn

    return results


def main():
    parser = argparse.ArgumentParser(description="PyTorch FC TF32 vs FP32 benchmark.")
    parser.add_argument(
        "--sizes",
        type=str,
        default="64,128,256,512,1024,2048,4096,8192",
        help="Comma-separated list of hidden layer sizes to benchmark.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--in_features", type=int, default=4096, help="Input feature dimension.")
    parser.add_argument("--out_features", type=int, default=4096, help="Output feature dimension.")
    parser.add_argument("--warmup_iters", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--timed_iters", type=int, default=100, help="Timed iterations per config.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (default: cuda)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("results", "pytorch_benchmark.json"),
        help="Output JSON file to write results.",
    )

    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s]

    results = run_fc_benchmark(
        sizes=sizes,
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        device=args.device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()

