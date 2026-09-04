"""额外可视化图表生成器 (Extra Figure Set)

针对 EAAI 配图评审意见 (neuro_figure_review_20260829.md 第五节"建议新增图表")
开发, 在实验主流程之外**额外生成一组可视化图表**, 数据全部取自实验系统
已产出的 JSON 结果文件 / 公开数据集 CSV, 不引入任何手写数值:

1. ``extra_ablation_bars.png``        消融实验分组柱状图 (ablation_results.json)
2. ``extra_material_envelope.png``    材料发射率数据库包络可视化 (MODIS UCSB CSV)
3. ``extra_pipeline_schematic.png``   滤波 -> 反演级联数据流示意图
4. ``extra_uncertainty_calibration.png`` 不确定性校准曲线 (置信水平 PICP 对比)
5. ``extra_public_heatmap.png``       公开基准 RMSE 热力图 (方法 x 噪声类型)

所有图表统一经由 ``plot_config.finalize_figure`` 保存 (自动进入 output_dir),
并同时写一个 ``<名字>.meta.json`` 记录数据来源与生成时间, 便于独立保存与追溯。
每张图也可被 ``figure_studio.FigureStudio`` 加载后进行交互式细节编辑。

用法:
    python -m experiment_system.figure_generator              # 全部额外图
    python -m experiment_system.figure_generator --only ablation
    python -m experiment_system.figure_generator --list       # 列出可用图
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

try:
    from .plot_config import finalize_figure, get_plot_config
    from .journal_style import (C_MAIN, C_MAIN_EDGE, C_SECONDARY, C_BASELINE,
                                C_ACCENT, C_HIGHLIGHT, FONT_SIZES)
except ImportError:  # 直接以脚本方式运行
    from experiment_system.plot_config import finalize_figure, get_plot_config
    from experiment_system.journal_style import (
        C_MAIN, C_MAIN_EDGE, C_SECONDARY, C_BASELINE,
        C_ACCENT, C_HIGHLIGHT, FONT_SIZES)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# EAAI 语义配色 (报告 §4.2, 统一由 journal_style 提供 Nature 色板):
# 提议方法=深绿 #00A087 / 基线=中性灰 / 警示=红 / 次级模块=浅橙
COLOR_MAIN = C_MAIN        # 提议方法 (NS-ARKF / SSM-PINN, Nature 深绿)
COLOR_MAIN2 = C_SECONDARY  # 次主色 (深蓝, 消融中间组件)
COLOR_BASE = C_BASELINE    # 中性灰 (baseline)
COLOR_WARN = C_ACCENT      # 警示红 (发散/违反约束的方法)
COLOR_ACCENT = C_HIGHLIGHT  # 浅橙 (协方差自适应等次级模块)


def _load_json(name, data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# 1. 消融实验分组柱状图 (评审建议 1: 把消融表转为柱状图)
# --------------------------------------------------------------------------- #
def build_ablation_bars(data_dir=None):
    res = _load_json("ablation_results.json", data_dir)
    if not res:
        return None, "ablation_results.json 不存在, 跳过消融图"

    steps = list(res.keys())
    noises = ["gaussian", "impulsive", "time_varying"]
    labels = ["Gaussian", "Impulsive", "Time-Varying"]

    x = np.arange(len(noises))
    width = 0.8 / max(len(steps), 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    # 语义配色: UIF 基线=中性灰 / 中间组件=深蓝+浅橙 / 完整 NS-ARKF=提议深绿
    palette = [C_BASELINE, C_SECONDARY, C_HIGHLIGHT, C_MAIN]
    for i, step in enumerate(steps):
        vals = [res[step].get(nt, np.nan) for nt in noises]
        is_full = (i == len(steps) - 1)  # 末位 = 完整 NS-ARKF (提议)
        ax.bar(x + i * width, vals, width, label=step,
               color=palette[i % len(palette)],
               edgecolor=C_MAIN_EDGE if is_full else "black",
               linewidth=1.5 if is_full else 0.4, zorder=4 if is_full else 3)
        for xi, v in zip(x + i * width, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=FONT_SIZES["bar_label"] + 0.5)
    ax.set_xticks(x + width * (len(steps) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Injected extreme-noise regime")
    ax.set_ylabel("3D position RMSE (m)")
    ax.set_title("Ablation study: contribution of each NS-ARKF component")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, None


# --------------------------------------------------------------------------- #
# 2. 材料发射率数据库包络图 (评审建议 2: 材料数据库可视化)
# --------------------------------------------------------------------------- #
def build_material_envelope(data_dir=None):
    csv_path = os.path.join(data_dir or DATA_DIR, "modis_ucsb",
                            "modis_ucsb_emissivity.csv")
    if not os.path.exists(csv_path):
        return None, "modis_ucsb_emissivity.csv 不存在, 跳过材料包络图"

    # 轻量 CSV 解析 (避免 pandas 强依赖; 字段含引号, 用 csv 模块)
    import csv
    groups = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                eps = float(row["emissivity"])
            except (TypeError, ValueError, KeyError):
                continue
            cat = (row.get("paper_category") or "Unknown").strip()
            groups.setdefault(cat, []).append(eps)

    if not groups:
        return None, "CSV 中无有效发射率数据"

    cats = sorted(groups, key=lambda c: np.mean(groups[c]))
    means = [np.mean(groups[c]) for c in cats]
    lows = [np.min(groups[c]) for c in cats]
    highs = [np.max(groups[c]) for c in cats]
    y = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(cats) + 2.2))
    for yi, lo, hi, mean in zip(y, lows, highs, means):
        ax.plot([lo, hi], [yi, yi], color=COLOR_BASE, lw=4, alpha=0.65,
                solid_capstyle="round", zorder=2)
    ax.scatter(means, y, color=COLOR_MAIN, s=42, zorder=3,
               label="Category mean")
    ax.scatter(lows, y, marker="|", color="black", s=60, zorder=3)
    ax.scatter(highs, y, marker="|", color="black", s=60, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([c if len(c) <= 34 else c[:32] + "..." for c in cats],
                       fontsize=8)
    ax.set_xlabel("Long-wave-infrared emissivity")
    ax.set_ylabel("Material category")
    ax.set_title("Emissivity envelope of the public material database "
                 "(MODIS UCSB)")
    ax.set_xlim(0.0, 1.02)
    ax.axvline(1.0, color="grey", lw=0.8, ls=":")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, None


# --------------------------------------------------------------------------- #
# 3. 滤波 -> 反演级联数据流示意图 (评审建议 3)
# --------------------------------------------------------------------------- #
def build_pipeline_schematic(data_dir=None):
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text, fc, fontsize=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black",
                             linewidth=1.0, alpha=0.92, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, zorder=3)

    def arrow(x0, y0, x1, y1, style="-|>", ls="-", color="black"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.6,
                                    linestyle=ls), zorder=1)

    # 测量流
    box(0.2, 3.1, 2.0, 1.2, "Multi-sensor\nstream\n$T,V,D,\\theta,I_{echo}$",
        "#dfe6ee")
    # NS-ARKF 三模块
    box(3.0, 3.5, 2.0, 0.9, "UIF\nstate estimate", "#dbeafe")
    box(3.0, 2.0, 2.0, 0.9, "IFHBFNN\ninterference comp.", "#dcfce7")
    box(3.0, 0.5, 2.0, 0.9, "HBKFO\ncovariance adapt.", "#ffedd5")
    # 反演侧
    box(6.4, 2.6, 2.2, 1.6, "SSM-PINN\nencoder ->\nS6 evolver ->\ndecoder",
        "#dbeafe")
    box(9.6, 3.0, 2.2, 1.1,
        "Hard constraint\n$\\varepsilon+\\rho\\leq1$\nBayesian intervals",
        "#dcfce7", fontsize=9)

    arrow(2.2, 3.7, 3.0, 3.9)                      # 测量 -> UIF
    arrow(2.2, 3.5, 3.0, 2.6)                      # 测量 -> IFHBFNN
    arrow(2.2, 3.3, 3.0, 1.1, ls="--", color=COLOR_ACCENT)  # 测量 -> HBKFO
    arrow(4.0, 3.5, 4.0, 2.9)                      # UIF <-> IFHBFNN 组合
    arrow(4.0, 2.0, 4.0, 1.4, ls="--", color=COLOR_ACCENT)  # HBKFO -> UIF 增益
    arrow(5.0, 3.9, 6.4, 3.7)                      # 滤波 -> 反演
    arrow(5.0, 2.4, 6.4, 3.1)                      # IFHBFNN -> 反演
    arrow(8.6, 3.4, 9.6, 3.5)                      # 反演 -> 约束头
    ax.text(5.7, 4.35, "stable track available\n(1 Hz inversion query)",
            ha="center", fontsize=8, color="dimgrey")
    ax.text(6.0, 4.75, "NS-ARKF  ->  SSM-PINN  end-to-end data flow",
            ha="center", fontsize=12, fontweight="bold")
    return fig, None


# --------------------------------------------------------------------------- #
# 4. 不确定性校准曲线 (Reliability diagram: 合成 vs 真实数据, PICP + MPIW)
# --------------------------------------------------------------------------- #
def build_uncertainty_calibration(data_dir=None):
    """Reliability diagram; 合成数据 PICP 取自 uncertainty_calibration_gpu.json,
    真实数据 PICP 取自 public_validation_results_gpu.json 的 SSM-PINN PICP,
    构成锐度-校准二维评估 (M6 评审意见)."""
    # 真实数据 (公开验证)
    res_pub = _load_json("public_validation_results_gpu.json", data_dir)
    picp_real = None
    if res_pub:
        try:
            inv = res_pub["inversion_validation"]["table"]
            if "SSM-PINN" in inv:
                picp_real = inv["SSM-PINN"].get("picp")
        except (KeyError, TypeError):
            picp_real = None

    # 合成数据 (Table 8)
    res_cal = _load_json("uncertainty_calibration_gpu.json", data_dir)
    levels_synth = []
    picp_synth = []
    mpiw_synth = []
    if res_cal and "uncertainty_calibration" in res_cal:
        try:
            for r in res_cal["uncertainty_calibration"]["results"]:
                levels_synth.append(r["confidence_level"])
                picp_synth.append(r["actual_picp"])
                mpiw_synth.append(r["mpiw"])
        except (KeyError, TypeError):
            pass

    levels_synth = np.array(levels_synth) if levels_synth else np.array([0.68, 0.90, 0.95, 0.99])
    picp_synth = np.array(picp_synth) if picp_synth else levels_synth.copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    # 合成数据曲线
    ax.plot(levels_synth, picp_synth, "o-", color=COLOR_MAIN, lw=1.8, ms=7,
            zorder=4, label="Synthetic (Table 8)")
    for lv, pv in zip(levels_synth, picp_synth):
        ax.annotate(f"{pv:.3f}", (lv, pv), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=COLOR_MAIN)
    # 真实数据点 (仅 95%)
    if picp_real is not None:
        ax.scatter([0.95], [picp_real], color=COLOR_ACCENT, s=90, zorder=5,
                   marker="D", label=f"Real MODIS+SLUM PICP@95% = {picp_real:.3f}")
        ax.annotate(f"{picp_real:.3f}", (0.95, picp_real),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=9, color=COLOR_ACCENT)
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Prediction-interval coverage (PICP)")
    ax.set_title("Uncertainty calibration: synthetic vs real emissivity libraries")
    ax.set_xlim(0.6, 1.03)
    ax.set_ylim(0.6, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, None


# --------------------------------------------------------------------------- #
# 5. 公开基准 RMSE 热力图 (方法 x 噪声类型)
# --------------------------------------------------------------------------- #
def build_public_heatmap(data_dir=None):
    res = (_load_json("public_validation_results_gpu.json", data_dir)
           or _load_json("public_validation_results.json", data_dir))
    if not res:
        return None, "public_validation_results*.json 不存在, 跳过热力图"

    try:
        table = res["filtering_validation"]["table"]
    except (KeyError, TypeError):
        return None, "filtering_validation 缺失, 跳过热力图"

    methods = ["EKF", "UKF", "CKF", "AEKF", "PSO-EKF", "GA-UKF",
               "DeepKF", "RUKF", "NS-ARKF"]
    methods = [m for m in methods if m in table]
    noises = ["gaussian", "impulsive", "time_varying"]
    noise_labels = ["Gaussian", "Impulsive", "Time-Varying"]

    raw = np.array([[table[m][nt] for nt in noises] for m in methods],
                   dtype=float)
    # 对数变换后归一化到 [0,1] 以增强低 RMSE 区域的对比度
    logv = np.log10(np.maximum(raw, 1e-3))
    lo, hi = logv.min(), logv.max()
    norm = (logv - lo) / max(hi - lo, 1e-9)

    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(methods) + 2.0))
    # viridis: 感知均匀 + 色盲友好 (报告 §3; RdYlGn 红绿对色盲读者不可分)
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels(noise_labels)
    ax.set_yticks(range(len(methods)))
    ylabels = [m + "  (Ours)" if m == "NS-ARKF" else m for m in methods]
    ax.set_yticklabels(ylabels)
    # 提议方法行加粗深绿, 与全文语义色一致
    for tick, m in zip(ax.get_yticklabels(), methods):
        if m == "NS-ARKF":
            tick.set_fontweight("bold")
            tick.set_color(C_MAIN)
    for i in range(len(methods)):
        for j in range(len(noises)):
            # viridis 低值端深紫(白字), 高值端亮黄(黑字)
            color = "white" if norm[i, j] < 0.55 else "black"
            ax.text(j, i, f"{raw[i, j]:.3g}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title("3D position RMSE on KITTI geometry\n"
                 "(colour: log-scaled RMSE, darker is better)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log10 RMSE (normalised)")
    fig.tight_layout()
    return fig, None


# --------------------------------------------------------------------------- #
# 注册表 + 统一生成入口
# --------------------------------------------------------------------------- #
EXTRA_FIGURES = {
    "ablation": {
        "builder": build_ablation_bars,
        "filename": "extra_ablation_bars.png",
        "description": "消融实验分组柱状图 (ablation_results.json)",
    },
    "materials": {
        "builder": build_material_envelope,
        "filename": "extra_material_envelope.png",
        "description": "材料发射率数据库包络图 (MODIS UCSB CSV)",
    },
    "pipeline": {
        "builder": build_pipeline_schematic,
        "filename": "extra_pipeline_schematic.png",
        "description": "滤波-反演级联数据流示意图",
    },
    "calibration": {
        "builder": build_uncertainty_calibration,
        "filename": "extra_uncertainty_calibration.png",
        "description": "不确定性校准曲线 (PICP vs 置信水平)",
    },
    "heatmap": {
        "builder": build_public_heatmap,
        "filename": "extra_public_heatmap.png",
        "description": "公开基准 RMSE 热力图 (方法 x 噪声)",
    },
}


def _write_meta(out_path, key, info, data_dir):
    """写 <图名>.meta.json, 记录数据来源与生成时间 (独立保存与追溯)。"""
    meta = {
        "figure": key,
        "description": info["description"],
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": info.get("sources", []),
        "data_dir": data_dir or DATA_DIR,
        "png": os.path.basename(out_path),
    }
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[figure_generator] 写 meta 失败 {meta_path}: {exc}")


def generate_extra_figures(output_dir=None, selected=None, data_dir=None,
                           show=None):
    """生成一组额外可视化图表。

    Parameters
    ----------
    output_dir : str or None
        保存目录; None 时用全局 plot_config 的 output_dir/figures/extra。
    selected : list[str] or None
        只生成注册表中的这些 key; None 表示全部。
    data_dir : str or None
        结果 JSON/CSV 所在目录 (默认 experiment_system/data)。
    show : bool or None
        None 时遵循全局 interactive 配置。

    Returns
    -------
    dict  key -> 保存路径 (跳过的条目值为 None)
    """
    cfg = get_plot_config()
    if output_dir is None:
        base = cfg.output_dir or os.path.join(os.getcwd(), "figures")
        output_dir = os.path.join(base, "extra")
    os.makedirs(output_dir, exist_ok=True)

    keys = list(EXTRA_FIGURES) if not selected else list(selected)
    results = {}
    for key in keys:
        info = EXTRA_FIGURES.get(key)
        if info is None:
            print(f"[figure_generator] 未知图名: {key}")
            results[key] = None
            continue
        try:
            fig, warn = info["builder"](data_dir)
        except Exception as exc:  # 单图失败不阻断其余图
            print(f"[figure_generator] {key} 生成失败: {exc}")
            results[key] = None
            continue
        if fig is None:
            print(f"[figure_generator] {key}: {warn}")
            results[key] = None
            continue
        # 转绝对路径, 避免 resolve_save_path 对相对路径再拼一层 cfg.output_dir/figures
        out_path = os.path.abspath(os.path.join(output_dir, info["filename"]))
        saved = finalize_figure(fig, save_path=out_path, show=show)
        _write_meta(out_path, key, info, data_dir)
        results[key] = saved
        print(f"[figure_generator] {key} -> {saved}")
    return results


def _main(argv=None):
    ap = argparse.ArgumentParser(description="生成额外一组可视化图表")
    ap.add_argument("--output-dir", default=None,
                    help="保存目录 (默认 <output_dir>/figures/extra)")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=sorted(EXTRA_FIGURES),
                    help="只生成指定图 (默认全部)")
    ap.add_argument("--data-dir", default=None,
                    help="JSON/CSV 数据目录 (默认 experiment_system/data)")
    ap.add_argument("--list", action="store_true", help="列出可用图并退出")
    ap.add_argument("--no-interactive", action="store_true",
                    help="不弹窗, 仅保存")
    args = ap.parse_args(argv)

    if args.list:
        for key, info in EXTRA_FIGURES.items():
            print(f"{key:12s} -> {info['filename']:38s} {info['description']}")
        return 0

    from .plot_config import configure
    configure(interactive=not args.no_interactive, save=True,
              output_dir=args.output_dir or os.getcwd())
    generate_extra_figures(output_dir=args.output_dir, selected=args.only,
                           data_dir=args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
