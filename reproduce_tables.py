"""reproduce_tables.py — 复现论文 Table 5 / Table 6 绝对数值的运行入口

复现路径 (与论文章节对应):
  Table 5 (tab:filtering_rmse, §5.4.1, CAL0827.tex L891-L918):
      NS-ARKF 协方差每步解析求解 (§5.1.1 L737), 调用
      run_paper_experiments.run_filtering_experiment 产出 7 方法 × 4 噪声 RMSE.
  Table 6 (tab:inversion_accuracy, §5.4.2, L920-L941):
      (a) 闭式 ε/ρ 最小二乘 (§5.1.1 L734-L737 实际方法, np.linalg.lstsq)
          — torch_training.fit_closed_form_eps_rho / evaluate_closed_form
      (b) PyTorch SSM-PINN Algorithm 3 训练 (§4.5 + Algorithm 3 L688-L721)
          — torch_training.SSMPINNTrainer + evaluate_inversion

用法 (包内):
    python -m experiment_system.reproduce_tables --samples 2000 --epochs 60
用法 (直接):
    python reproduce_tables.py --samples 2000 --epochs 60

说明:
  - --samples 控制数据集规模 (论文全量 125000; 小值用于快速验证).
  - --epochs 控制 PyTorch 训练轮数.
  - --skip-table5 / --skip-train 可跳过耗时步骤.
  - 输出会与论文 Table 5/6 的绝对数值并列对比.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Dict

import numpy as np

# 包内/直接运行双模式
try:
    from .data_generator import DatasetConfig, FullDatasetGenerator
    from .init_database import MATERIAL_DATA
    from . import torch_training as TT
except Exception:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_generator import DatasetConfig, FullDatasetGenerator
    from init_database import MATERIAL_DATA
    import torch_training as TT


# 论文 Table 5 (tab:filtering_rmse) 绝对数值 (CAL0827.tex L900-L906)
PAPER_TABLE5 = {
    "EKF":      [0.823, 1.452, 2.187, 1.634],
    "UKF":      [0.756, 1.298, 1.945, 1.487],
    "CKF":      [0.741, 1.245, 1.876, 1.423],
    "AEKF":     [0.698, 1.156, 1.654, 1.298],
    "RUKF":     [0.712, 1.089, 1.432, 1.187],
    "DeepKF":   [0.654, 0.987, 1.298, 1.098],
    "NS-ARKF":  [0.512, 0.723, 0.876, 0.745],
}
NOISE_LABELS = ["Gaussian", "Mixture", "Impulsive", "Time-Varying"]


def generate_samples(n_samples: int, seed: int = 42):
    """生成数据集 (§5.1.2)."""
    ds_cfg = DatasetConfig(total_samples=n_samples, seed=seed)
    gen = FullDatasetGenerator(ds_cfg)
    return gen.generate_full_dataset(MATERIAL_DATA)


# ----------------------------------------------------------------------------
# Table 6 复现
# ----------------------------------------------------------------------------
def reproduce_table6(samples, cfg: TT.TorchTrainingConfig, skip_train: bool = False) -> Dict:
    """复现 Table 6 (tab:inversion_accuracy)."""
    print("\n" + "=" * 92)
    print(f"{'Table 6: Material Property Inversion Accuracy (tab:inversion_accuracy)':^92}")
    print(f"{'(Paper §5.4.2, CAL0827.tex L920-L941)':^92}")
    print("=" * 92)

    # ---- 划分数据 (两条路径使用同一测试集, 保证公平比较) ----
    TT.set_global_seed(cfg.seed)
    train_loader, val_loader, test_loader, _, splits = TT.build_dataloaders(samples, cfg)
    test_samples = splits["test"]

    # ---- (a) 闭式 ε/ρ 最小二乘 (§5.1.1 实际方法) ----
    print("\n[a] Closed-form ε/ρ least-squares (paper §5.1.1, np.linalg.lstsq)...")
    cf = TT.evaluate_closed_form(test_samples)

    # ---- (b) PyTorch SSM-PINN Algorithm 3 训练 ----
    if skip_train:
        print("[b] PyTorch SSM-PINN training SKIPPED (--skip-train)")
        torch_metrics = None
    else:
        print(f"[b] PyTorch SSM-PINN training (Algorithm 3, epochs={cfg.epochs}, "
              f"batch={cfg.batch_size}, AdamW lr={cfg.learning_rate})...")
        trainer = TT.SSMPINNTrainer(cfg)
        import time as _time
        t0 = _time.time()
        trainer.fit(train_loader, val_loader)
        dt = _time.time() - t0
        best_val = min(trainer.history["val"]) if trainer.history["val"] else float("nan")
        print(f"    training done in {dt:.1f}s | best_val={best_val:.5f}")
        print("    evaluating on test set...")
        pred = TT.predict_torch(trainer, test_loader)
        torch_metrics = TT.evaluate_inversion(pred)

    # ---- 对比表 ----
    paper = TT.PAPER_TABLE6_SSM_PINN
    hdr = f"{'Metric':<26}{'Paper(SSM-PINN)':>18}{'Closed-form':>16}{'PyTorch':>16}"
    print("\n" + "-" * 92)
    print(hdr)
    print("-" * 92)
    metric_keys = [
        ("Emissivity RMSE",        "emissivity_rmse",    "{:.4f}"),
        ("Reflectivity RMSE",      "reflectivity_rmse",  "{:.4f}"),
        ("Classification Acc (%)", "classification_acc", "{:.1f}"),
        ("F1-Score",               "f1_score",           "{:.3f}"),
    ]
    for name, key, fmt in metric_keys:
        p = paper[key]; c = cf[key]
        t = torch_metrics[key] if torch_metrics else None
        cstr = fmt.format(c)
        tstr = fmt.format(t) if t is not None else "—"
        print(f"{name:<26}{p:>18}{cstr:>16}{tstr:>16}")
    print("-" * 92)
    print("Closed-form = §5.1.1 deterministic path; PyTorch = Algorithm 3 gradient path.")
    return {"closed_form": cf, "torch": torch_metrics}


# ----------------------------------------------------------------------------
# Table 5 复现
# ----------------------------------------------------------------------------
def reproduce_table5(seed: int = 42, num_samples: int = 1000) -> Dict:
    """复现 Table 5 (tab:filtering_rmse) — 调用既有 NS-ARKF 闭式滤波对比.

    注意:
      - run_filtering_experiment 期望 comparison_experiments.ExperimentConfig
        (含 seed / num_samples / dim_x / dim_z / noise_level 等字段), 而非
        config.SystemConfig. 此前误用 config.DEFAULT_CONFIG (SystemConfig)
        导致 AttributeError: 'seed' 缺失.
      - comparison_experiments.py 使用裸相对导入 (from .filtering import ...),
        在脚本直接运行 (顶层模块) 模式下会触发
        "attempted relative import with no known parent package".
        因此这里不直接 import comparison_experiments, 而是用 SimpleNamespace
        构造等价配置 (FilteringComparison 仅读取属性, 不做 isinstance 校验),
        并通过 run_paper_experiments (有 try/except 回退) 获取入口函数.
    """
    from types import SimpleNamespace

    print("\n" + "=" * 92)
    print(f"{'Table 5: State Estimation RMSE (tab:filtering_rmse)':^92}")
    print(f"{'(Paper §5.4.1, CAL0827.tex L891-L918; NS-ARKF closed-form, §5.1.1 L737)':^92}")
    print("=" * 92)
    try:
        # run_paper_experiments.py 有 try/except 回退 (包内/直接运行双模式),
        # 其命名空间已绑定 ExperimentConfig; 直接 import 它的入口函数即可.
        from run_paper_experiments import run_filtering_experiment
        # 等价于 comparison_experiments.ExperimentConfig 的字段集
        fcfg = SimpleNamespace(
            num_samples=num_samples,
            dim_x=6,            # 状态维: [x, y, vx, vy, ax, ay] (论文§5.1.2)
            dim_z=4,            # 量测维: [x, y, vx, vy]
            noise_level=0.1,    # 论文§5.1.3 基准噪声水平 σ=0.1
            seed=seed,
            output_dir=tempfile.mkdtemp(prefix="table5_"),
            max_iterations=500,
            learning_rate=0.01,
        )
        all_results = run_filtering_experiment(fcfg, fcfg.output_dir)
    except Exception as e:
        print(f"\n[Table 5] 既有滤波对比入口调用失败: {e!r}")
        print("可单独运行: python -m experiment_system.run_paper_experiments")
        print("\n论文 Table 5 参考值 (NS-ARKF 闭式解析):")
        _print_paper_table5()
        return {"reproduced": False}

    # 与论文对比
    methods = ['EKF', 'UKF', 'CKF', 'AEKF', 'RUKF', 'DeepKF', 'NS-ARKF']
    noise_types = ['gaussian', 'mixture', 'impulsive', 'time_varying']
    print("\n" + "-" * 92)
    print(f"{'Method':<12}" + "".join(f"{n:>14}" for n in NOISE_LABELS) + f"{'Avg':>10}")
    print("-" * 92)
    for m in methods:
        got = [all_results[nt][m].overall_rmse for nt in noise_types]
        avg = float(np.mean(got))
        row = f"{m:<12}" + "".join(f"{v:>14.3f}" for v in got) + f"{avg:>10.3f}"
        if m == "NS-ARKF":
            row += "  *"
        print(row)
    print("-" * 92)
    print("* proposed method (NS-ARKF).  论文参考值见上方 PAPER_TABLE5 区块或 CAL0827.tex L900-L906.")
    return {"reproduced": True, "results": all_results}


def _print_paper_table5():
    print(f"{'Method':<12}" + "".join(f"{n:>14}" for n in NOISE_LABELS) + f"{'Avg':>10}")
    print("-" * 80)
    for m, vals in PAPER_TABLE5.items():
        avg = float(np.mean(vals))
        row = f"{m:<12}" + "".join(f"{v:>14.3f}" for v in vals) + f"{avg:>10.3f}"
        if m == "NS-ARKF":
            row += "  *"
        print(row)


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Reproduce paper Table 5 & 6")
    ap.add_argument("--samples", type=int, default=2000,
                    help="数据集规模 (论文全量 125000; 默认 2000 用于快速验证)")
    ap.add_argument("--epochs", type=int, default=60, help="PyTorch 训练轮数")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-table5", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--full", action="store_true", help="使用论文全量 125000 样本")
    args = ap.parse_args()

    n_samples = 125000 if args.full else args.samples
    print(f"Reproduce Tables | samples={n_samples} epochs={args.epochs} "
          f"batch={args.batch_size} device={args.device} seed={args.seed}")

    # ---- Table 6 ----
    samples = generate_samples(n_samples, seed=args.seed)
    cfg = TT.TorchTrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, device=args.device, seed=args.seed,
    )
    reproduce_table6(samples, cfg, skip_train=args.skip_train)

    # ---- Table 5 ----
    # Table 5 使用独立的合成状态序列 (num_samples 较小即可收敛, 默认 1000)
    if not args.skip_table5:
        reproduce_table5(seed=args.seed, num_samples=min(n_samples, 1000))
    else:
        print("\n[Table 5] skipped (--skip-table5)")


if __name__ == "__main__":
    main()
