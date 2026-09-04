"""Benchmark: SSM-PINN training speed, NumPy (CPU finite-difference) vs
PyTorch (GPU autograd), on the real MODIS UCSB + SLUM inversion dataset.

Runs the exact dataset builder used by the validation study, trains the
SSM-PINN with each backend for the configured iterations, and reports wall-clock
time, speedup, device, and final test RMSE.  Use a smaller ``--iters`` for a
quick check; the manuscript run uses 300.

Usage
-----
    python -m experiment_system.data.benchmark_gpu --iters 300
    python -m experiment_system.data.benchmark_gpu --iters 50 --skip-numpy
    python -m experiment_system.data.benchmark_gpu --iters 300 --json bench.json
"""

import argparse
import json
import os

# Permit NumPy(MKL) + PyTorch OpenMP runtimes to coexist (see inversion_torch);
# must be set before torch is imported anywhere in the process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time

import numpy as np

from experiment_system.inversion import InversionConfig, create_ssm_pinn_model
from experiment_system.evaluation import compute_inversion_metrics
from experiment_system.data.public_dataset_validation import (
    _build_real_emissivity_dataset,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    m = compute_inversion_metrics(
        y_test[:, 0], y_test[:, 1], pred[:, 0], pred[:, 1])
    ss_res = float(np.sum((y_test[:, 0] - pred[:, 0]) ** 2))
    ss_tot = float(np.sum((y_test[:, 0] - y_test[:, 0].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(m.emissivity_rmse), float(r2), float(m.constraint_violation_rate)


def main():
    ap = argparse.ArgumentParser(description="SSM-PINN CPU vs GPU benchmark.")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip-numpy", action="store_true",
                    help="Skip the slow NumPy finite-difference baseline.")
    ap.add_argument("--json", default=None,
                    help="Optional path to write the benchmark result as JSON.")
    args = ap.parse_args()

    ds = _build_real_emissivity_dataset(seed=42)
    X_tr, y_tr = ds["X_train"], ds["y_train"]
    X_te, y_te = ds["X_test"], ds["y_test"]
    print("dataset: n=%d train=%d test=%d dim=%d"
          % (ds["n"], ds["n_train"], ds["n_test"], X_tr.shape[1]))

    results = {}

    # --- GPU / torch backend ---------------------------------------------- #
    try:
        import torch
        from experiment_system.inversion_torch import (
            create_ssm_pinn_torch, gpu_available)
        cfg = InversionConfig(max_iterations=args.iters, learning_rate=args.lr,
                              enforce_hard_constraint=True)
        dev = "cuda" if gpu_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if gpu_available() else "n/a"
        print("\n[torch] device=%s (%s) torch=%s"
              % (dev, gpu_name, torch.__version__))
        model = create_ssm_pinn_torch(cfg)
        model.set_feature_stats(ds["feat_mean"], ds["feat_std"])
        if gpu_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        model._train(X_tr, y_tr)
        if gpu_available():
            torch.cuda.synchronize()
        t_torch = time.time() - t0
        rmse_torch, r2_torch, viol_torch = _metrics(model, X_te, y_te)
        mem = (torch.cuda.max_memory_allocated() / 1e6) if gpu_available() else 0.0
        results["torch"] = {"time_s": round(t_torch, 3), "rmse": round(rmse_torch, 4),
                            "r2": round(r2_torch, 4), "viol": round(viol_torch, 4),
                            "device": dev, "gpu_name": gpu_name,
                            "gpu_mem_mb": round(mem, 1)}
        print("[torch] time=%.2fs rmse=%.4f r2=%.4f viol=%.3f gpu_mem=%.1fMB"
              % (t_torch, rmse_torch, r2_torch, viol_torch, mem))
    except ImportError as e:
        print("[torch] unavailable: %s" % e)

    # --- NumPy CPU baseline ----------------------------------------------- #
    if not args.skip_numpy:
        cfg = InversionConfig(max_iterations=args.iters, learning_rate=args.lr,
                              enforce_hard_constraint=True)
        print("\n[numpy] CPU finite-difference baseline ...")
        model = create_ssm_pinn_model(cfg)
        t0 = time.time()
        model._train(X_tr, y_tr)
        t_np = time.time() - t0
        rmse_np, r2_np, viol_np = _metrics(model, X_te, y_te)
        results["numpy"] = {"time_s": round(t_np, 3), "rmse": round(rmse_np, 4),
                            "r2": round(r2_np, 4), "viol": round(viol_np, 4),
                            "device": "cpu"}
        print("[numpy] time=%.2fs rmse=%.4f r2=%.4f viol=%.3f"
              % (t_np, rmse_np, r2_np, viol_np))

    # --- summary ---------------------------------------------------------- #
    print("\n==== SSM-PINN benchmark (iters=%d) ====" % args.iters)
    for k, v in results.items():
        print("  %-6s %8.2fs  rmse=%.4f  r2=%.4f  (%s)"
              % (k, v["time_s"], v["rmse"], v["r2"], v.get("device", "")))
    speedup = None
    if "numpy" in results and "torch" in results:
        speedup = results["numpy"]["time_s"] / max(results["torch"]["time_s"], 1e-9)
        print("  speedup (numpy CPU / torch GPU): %.1fx" % speedup)

    if args.json:
        out = {"iters": args.iters, "lr": args.lr,
               "n_train": ds["n_train"], "n_test": ds["n_test"],
               "results": results, "speedup_cpu_over_gpu": speedup}
        dest = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=2)
        print("Wrote %s" % dest)


if __name__ == "__main__":
    main()

