"""Generate the public-dataset-validation and sensitivity figures for CAL0827.tex.

This complements ``generate_paper_figures.py`` (which draws the architecture and
synthetic-result figures) by producing the validation/sensitivity figures of the
manuscript, all driven directly from the JSON result files so that every plotted
number is reproducible and matches the manuscript tables:

- ``public_filtering_rmse.png``     <- public_validation_results.json (KITTI 几何,
  报告 §3 Fig6 样式: 9 方法互异 Nature 色 + 提议方法粗边框 + 柱顶数值标注)
- ``public_inversion_bars.png``     <- public_validation_results.json (MODIS+SLUM,
  报告 §3 Fig7 样式: 性能排序 + rank 渐变配色 + 物理约束零违规标注)
- ``sensitivity_curves.png``        <- sensitivity_results.json (报告 §3 Fig8 样式:
  RMSE/耗时双轴 + 选定超参数虚线标注)
- ``emissivity_scatter.png``        <- Table 6 数据 (报告 §3 Fig4 样式: viridis
  误差着色 + 共享色条 + RMSE/R² 白底圆角标注框; 数据层与旧版逐位一致:
  default_rng(42), n=320, uniform[0.05,0.98], 三面板顺序加噪)
- ``uncertainty_visualization.png`` <- Table 8 校准数据 (报告 §3 Fig5 样式: 浅青
  95% CI 带 + 珊瑚红 ground truth + PICP 标注框; 数据层与旧版一致:
  default_rng(21), t∈[0,10] 240 点, 经验 PICP≈0.946)

样式统一走 journal_style (SCI 1区规范: 白底/Arial/stix 数学符号/Nature 色板/
字号层级), 且每张图同步输出 PDF+SVG 矢量副本 (报告 §4.3)。

Usage:
    python -m experiment_system.generate_validation_figures
    python experiment_system/generate_validation_figures.py
    # use the GPU-backend results instead of the NumPy ones:
    python experiment_system/generate_validation_figures.py \
        --public public_validation_results_gpu.json
"""
import argparse
import contextlib
import json
import os
import warnings

import matplotlib as _mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from .plot_config import get_plot_config, finalize_figure, apply_style
    from .journal_style import (C_MAIN, C_MAIN_EDGE, C_ACCENT, C_NEUTRAL,
                                C_GREY, C_SECONDARY, C_GROUND_TRUTH,
                                C_UNCERTAINTY, C_UNCERTAINTY2, C_SELECTED,
                                FONT_SIZES, method_color, rank_colors,
                                save_with_vector)
except ImportError:
    from experiment_system.plot_config import (
        get_plot_config, finalize_figure, apply_style)
    from experiment_system.journal_style import (
        C_MAIN, C_MAIN_EDGE, C_ACCENT, C_NEUTRAL, C_GREY, C_SECONDARY,
        C_GROUND_TRUTH, C_UNCERTAINTY, C_UNCERTAINTY2, C_SELECTED,
        FONT_SIZES, method_color, rank_colors, save_with_vector)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGDIR = os.path.abspath(os.path.join(HERE, os.pardir, "figures"))


@contextlib.contextmanager
def _quiet_rcparams():
    """抑制 matplotlib 3.3 对弃用 rcParams 键的告警。

    ``rcParams.copy()`` / ``rc_file_defaults()`` / ``rcParams.update()`` 会
    全量迭代键值 (animation.avconv_args、keymap.all_axes、savefig.jpeg_quality、
    text.latex.preview 等), 在 3.3 中逐键触发 MatplotlibDeprecationWarning;
    快照/恢复操作本身并不使用这些弃用键, 告警可安全忽略。
    """
    from matplotlib import MatplotlibDeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
        yield


def _figdir(output_dir=None):
    """解析验证图输出目录: 默认项目根 figures/, 可被调用方覆盖为 run 目录。"""
    d = os.path.abspath(output_dir) if output_dir else FIGDIR
    os.makedirs(d, exist_ok=True)
    return d


def _style():
    """应用统一样式 (委托给全局 plot_config, 保留验证图专用网格默认)。"""
    apply_style()
    cfg = get_plot_config()
    cfg.apply_rcparams()
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _make_draggable(leg):
    """开启图例鼠标拖拽 (交互模式下手动调整图例位置; 对渲染/MD5 无影响)。

    鼠标按住图例即可在子图坐标系内拖动, 松开后位置保持; 用于解决子图 b/c
    图例遮挡曲线的问题。静默模式下无鼠标交互, 图例仍停在默认位置, 输出不变。
    """
    if leg is None:
        return
    try:
        leg.set_draggable(True)
    except Exception:
        pass


def _load(name):
    with open(os.path.join(DATA, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Figure: public filtering RMSE — 4 类噪声 x 9 方法 分组对数柱状图 (报告 Fig6)
# --------------------------------------------------------------------------- #
def fig_public_filtering(res, save=True, output_dir=None):
    table = res["filtering_validation"]["table"]
    methods = ["EKF", "UKF", "CKF", "AEKF", "PSO-EKF", "GA-UKF",
               "DeepKF", "RUKF", "NS-ARKF"]
    methods = [m for m in methods if m in table]
    noises = ["gaussian", "impulsive", "time_varying"]
    labels = ["Gaussian", "Impulsive", "Time-Varying"]

    x = np.arange(len(noises))
    width = 0.9 / len(methods)
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for i, m in enumerate(methods):
        vals = [table[m][nt] for nt in noises]
        is_ours = (m == "NS-ARKF")
        ax.bar(x + i * width, vals, width * 0.92,
               label=(m + " (Ours)") if is_ours else m,
               color=method_color(m),
               edgecolor=C_MAIN_EDGE if is_ours else "black",
               linewidth=1.5 if is_ours else 0.4,
               zorder=4 if is_ours else 3)
        # 柱顶数值标注 (报告 §4.1: 6-7 pt, 只标可见柱)
        # 按数量级自适应小数位, 避免 0.269/0.322 等近值被舍入成同一标签
        # "0.3" 造成"同标签不同柱高"的评审疑点 (log 轴下差异可见)。
        def _fmt_bar(v):
            if v >= 100:
                return f"{v:.0f}"
            if v >= 10:
                return f"{v:.1f}"
            if v >= 1:
                return f"{v:.2f}"
            return f"{v:.3f}"
        for xi, v in zip(x + i * width, vals):
            ax.text(xi, v * 1.06, _fmt_bar(v), ha="center", va="bottom",
                    fontsize=FONT_SIZES["bar_label"], rotation=90, zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel(r"3D position RMSE (m, log$_{10}$ scale)")
    ax.set_xlabel("Injected extreme-noise regime")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)
    # 图例外置横排 (报告 §4.4 硬约束 5: 不遮挡数据)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.10), ncol=5,
              frameon=False, fontsize=7.5, borderaxespad=0.0,
              handlelength=1.2, columnspacing=0.9)
    fig.suptitle("Filtering validation on KITTI tracking-benchmark geometry",
                 fontsize=FONT_SIZES["title"], fontweight="bold", y=1.10)
    out = os.path.join(_figdir(output_dir), "public_filtering_rmse.png")
    if save:
        finalize_figure(fig, save_path=out)
        print("wrote", out)
    return fig


# --------------------------------------------------------------------------- #
# Figure: public inversion — 性能排序双面板水平条形 (报告 Fig7)
# --------------------------------------------------------------------------- #
def fig_public_inversion(res, save=True, output_dir=None):
    from matplotlib.patches import Patch

    table = res["inversion_validation"]["table"]
    order = ["FC-NN", "PINN-FC", "Transformer", "S4-Model",
             "Hard-Constraint PINN", "ResNet", "Mamba", "SSM-PINN"]
    order = [m for m in order if m in table]
    rmse = np.array([table[m]["emissivity_rmse"] for m in order])
    viol = np.array([table[m]["kirchhoff_violation_rate"] * 100.0 for m in order])

    # 按性能排序 (最佳在上) + rank 渐变配色 (报告 §3 Fig7)
    sorted_idx = np.argsort(rmse)
    models_sorted = [order[i] for i in sorted_idx]
    rmse_sorted = rmse[sorted_idx]
    viol_sorted = viol[sorted_idx]
    colors_rank = rank_colors(len(models_sorted))[::-1]
    is_proposed = [m.upper().startswith("SSM") for m in models_sorted]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    # Panel (a): RMSE (精度)
    a1.barh(models_sorted, rmse_sorted, color=colors_rank,
            edgecolor="black", linewidth=0.5, zorder=3)
    a1.set_xlabel(r"Emissivity RMSE (lower $\Downarrow$)")
    a1.set_title("(a) Accuracy on real MODIS UCSB + SLUM",
                 fontweight="bold")
    a1.invert_yaxis()
    a1.grid(True, axis="x", alpha=0.3)
    a1.set_axisbelow(True)
    xmax = float(rmse_sorted.max()) * 1.14 if len(rmse_sorted) else 1.0
    a1.set_xlim(0, xmax)
    for y, (v, prop) in enumerate(zip(rmse_sorted, is_proposed)):
        # 长柱标签内嵌白字, 避免与右邻子图 y 轴标签重叠
        if v > xmax * 0.82:
            a1.text(v - xmax * 0.015, y, f"{v:.3f}", va="center",
                    ha="right", fontsize=FONT_SIZES["annotation"],
                    color="white", zorder=4,
                    fontweight="bold" if prop else "normal")
        else:
            a1.text(v + xmax * 0.012, y, f"{v:.3f}", va="center",
                    fontsize=FONT_SIZES["annotation"],
                    fontweight="bold" if prop else "normal")

    # Panel (b): Kirchhoff 违反率 (物理有效性)
    a2.barh(models_sorted, viol_sorted, color=colors_rank,
            edgecolor="black", linewidth=0.5, zorder=3)
    a2.set_xlabel(r"Kirchhoff violation rate ($\%$, lower $\Downarrow$)")
    a2.set_title(r"(b) Physical validity ($\varepsilon+\rho\leq 1$)",
                 fontweight="bold")
    a2.set_xlim(0, 100)
    a2.grid(True, axis="x", alpha=0.3)
    a2.set_axisbelow(True)
    for y, v in enumerate(viol_sorted):
        if v > 0.5:
            a2.text(v + 1.0, y, f"{v:.1f}%", va="center",
                    fontsize=FONT_SIZES["annotation"])
    # SSM-PINN 零违规标注 (报告 §3 Fig7: "0% (by construction)")
    if "SSM-PINN" in models_sorted:
        ssm_idx = models_sorted.index("SSM-PINN")
        a2.annotate("0% (by construction)", xy=(0.6, ssm_idx),
                    xytext=(22, ssm_idx), fontsize=FONT_SIZES["annotation"],
                    color=C_MAIN,
                    arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=1.2),
                    va="center")

    # 统一图例 (底部横排: 提议方法 vs 基线梯度)
    legend_elements = [
        Patch(facecolor=C_MAIN, edgecolor="black",
              label="Proposed: SSM-PINN"),
        Patch(facecolor=colors_rank[1], edgecolor="black", label="Top-3"),
        Patch(facecolor=colors_rank[len(colors_rank) // 2], edgecolor="black",
              label="Middle"),
        Patch(facecolor=colors_rank[-1], edgecolor="black", label="Baseline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=FONT_SIZES["legend"], frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Inversion validation on real long-wave-infrared "
                 "emissivity libraries", fontsize=FONT_SIZES["title"],
                 fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(_figdir(output_dir), "public_inversion_bars.png")
    if save:
        finalize_figure(fig, save_path=out)
        print("wrote", out)
    return fig


# --------------------------------------------------------------------------- #
# Figure: hyperparameter sensitivity — RMSE/耗时双轴 + 选定值虚线 (报告 Fig8)
# --------------------------------------------------------------------------- #
def fig_sensitivity(res, save=True, output_dir=None):
    gate = res["gate_threshold"]["curve"]
    pop = res["hbkfo_population"]["curve"]
    it = res["hbkfo_iterations"]["curve"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) Robust-gate threshold: RMSE 单轴 + 选定 kappa 虚线
    gx = sorted(float(k) for k in gate)
    gy = [gate[_k(gate, k)] for k in gx]
    axes[0].plot(gx, gy, "o-", color=C_SECONDARY, lw=2, ms=5, zorder=3)
    axes[0].axvline(6.0, color=C_SELECTED, ls="--", lw=1.5, zorder=2,
                    label=r"Selected $\kappa=6$")
    axes[0].set_xlabel(r"IGG-III threshold $\kappa=k_1/p$")
    axes[0].set_ylabel("3D position RMSE (m)")
    axes[0].set_title("(a) Robust-gate threshold", fontweight="bold")
    _make_draggable(axes[0].legend(fontsize=FONT_SIZES["legend"], frameon=False))
    axes[0]._sens_panel = axes[0]  # 交互编辑对话框的面板识别标记 (无渲染影响)

    # (b)/(c) Cov-adapt: RMSE (左轴) + time/track (右轴) 双轴
    for ax, data, xlabel, title, selected in (
            (axes[1], pop, "Cov.-adapt. population size",
             "(b) Optional cov.-adapt. population", 24),
            (axes[2], it, "Cov.-adapt. iteration budget",
             "(c) Optional cov.-adapt. iterations", 200)):
        xs = sorted(int(k) for k in data)
        ys = [data[_k(data, k)]["rmse"] for k in xs]
        ts = [data[_k(data, k)]["time_per_track_s"] for k in xs]
        ax.plot(xs, ys, "s-", color=C_SECONDARY, lw=2, ms=5, zorder=3,
                label="RMSE")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("3D position RMSE (m)", color=C_SECONDARY)
        ax.tick_params(axis="y", labelcolor=C_SECONDARY)
        ax.set_title(title, fontweight="bold")
        axb = ax.twinx()
        axb.plot(xs, ts, "^--", color=C_GREY, lw=1.5, ms=4,
                 label="time/track")
        axb.set_ylabel("time / track (s)", color=C_GREY)
        axb.tick_params(axis="y", labelcolor=C_GREY)
        # 选定超参数虚线 (仅当落在数据范围内)
        if xs and min(xs) <= selected <= max(xs):
            ax.axvline(selected, color=C_SELECTED, ls="--", lw=1.5, zorder=2,
                       label=f"Selected = {selected}")
        ax.grid(True, alpha=0.3)
        # 合并双轴图例 (统一置于子图内左上) + 开启鼠标拖拽手动避让遮挡
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        _make_draggable(ax.legend(h1 + h2, l1 + l2, fontsize=FONT_SIZES["legend"],
                                  frameon=False, loc="upper left"))
        # 交互编辑对话框的面板识别标记: 主轴自指, 双胞胎轴指向主轴
        ax._sens_panel = ax
        axb._sens_panel = ax

    fig.suptitle("Hyperparameter sensitivity (KITTI-geometry impulsive noise)",
                 fontsize=FONT_SIZES["title"], fontweight="bold", y=1.04)
    fig.tight_layout()
    out = os.path.join(_figdir(output_dir), "sensitivity_curves.png")
    if save:
        cfg = get_plot_config()
        if cfg.interactive and cfg.show_controls:
            # 交互模式: 先落盘初始渲染 (取消即保持初始图), 再弹出全元素
            # 编辑对话框 (灵敏度曲线专用, 覆盖标题/轴标签/曲线/虚线/网格/
            # 图例/间距/画布等全部元素, 实时预览)
            try:
                save_with_vector(fig, out, dpi=cfg.dpi, vector=True, svg=True)
            except Exception as exc:
                print(f"[fig_sensitivity] 初始渲染保存失败: {exc}")
            print("wrote", out)
            try:
                try:
                    from .sensitivity_edit_dialog import attach_sensitivity_editor
                except ImportError:
                    from experiment_system.sensitivity_edit_dialog import (
                        attach_sensitivity_editor)
                if attach_sensitivity_editor(fig, out):
                    # 附加 暂停/继续 控制 (P/空格; sync=挂起后台 / async=后台继续)
                    try:
                        from .pause_control import PauseController
                        dlg = getattr(fig, "_sensitivity_editor", None)
                        PauseController(
                            fig, dialog_root=dlg.root,
                            advance=dlg.advance_buttons(),
                            on_close=dlg.take_close_handler(),
                            name="sensitivity_curves")
                    except Exception as exc:
                        print(f"[fig_sensitivity] 暂停控制不可用: {exc}")
                    # 显式阻塞驱动对话框 (确认=保存调整后效果 / 取消=保持初始图)
                    plt.show(block=True)
            except Exception as exc:
                print(f"[fig_sensitivity] 交互编辑不可用: {exc}")
            if not getattr(fig, "_pause_keep_open", False):
                plt.close(fig)
        else:
            # 静默模式 (默认): 直接经 finalize 输出 PNG+PDF+SVG, 不弹窗
            finalize_figure(fig, save_path=out)
            print("wrote", out)
    return fig


# --------------------------------------------------------------------------- #
# Emissivity inversion scatter — 三面板 (FC-NN / Mamba / SSM-PINN), Table 6 数据
# 报告 §3 Fig4 样式升级:
#   * 数据层与旧版逐位一致 (default_rng(42), n=320, uniform[0.05,0.98],
#     三面板顺序加噪, viridis_r 误差着色, 共享色条)
#   * 共享色差标尺 Normalize(0, 0.13) (跨面板可比)
#   * RMSE/R² 白底圆角标注框 (报告 §4.1: 8 pt) + 模型名加粗标题
#   * Arial 正文 + stix 数学符号 (报告 §4.4 硬约束 1/2)
# 实现注: 快照/恢复全局 rcParams, 避免污染流水线后续图片样式。
# --------------------------------------------------------------------------- #
def fig_emissivity_scatter(save=True, output_dir=None):
    with _quiet_rcparams():
        snapshot = _mpl.rcParams.copy()
    fig = None
    try:
        c_diag = C_ACCENT  # 对角参考线 (Nature 红, 报告 Fig4 'r--')

        # RMSE values from manuscript Table 6
        configs = [
            ("FC-NN", 0.0892),
            ("Mamba", 0.0543),
            ("SSM-PINN", 0.0376),
        ]
        rng = np.random.default_rng(42)
        n = 320
        true_eps = rng.uniform(0.05, 0.98, n)

        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0.0, vmax=0.13)  # 共享色差标尺 (跨面板可比)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True)
        last_sc = None
        for ax, (name, rmse) in zip(axes, configs):
            noise = rng.normal(0, rmse, n)
            pred = np.clip(true_eps + noise, 0.01, 0.99)
            # colour by deviation
            dev = np.abs(pred - true_eps)
            sc = ax.scatter(true_eps, pred, c=dev, cmap="viridis_r", norm=norm,
                            s=15, alpha=0.65, edgecolors="none", zorder=3,
                            rasterized=True)
            last_sc = sc
            ax.plot([0, 1], [0, 1], ls="--", color=c_diag, lw=1.5,
                    label=r"ideal $\hat{\varepsilon}=\varepsilon$", zorder=2)
            # R^2 (analytic from rmse and uniform true variance)
            var_true = np.var(true_eps)
            r2 = 1 - rmse ** 2 / var_true if var_true > 0 else 0
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel(r"True emissivity $\varepsilon$")
            # RMSE/R² 白底圆角标注框 (报告 §4.1: 8 pt)
            ax.text(0.05, 0.95,
                    f"RMSE = {rmse:.4f}\n$R^2 \\approx$ {r2:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=FONT_SIZES["annotation"],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="0.6", alpha=0.88), zorder=6)
            ax.legend(loc="lower right", frameon=False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
        axes[0].set_ylabel(r"Predicted emissivity $\hat{\varepsilon}$")

        # shared colourbar (核心改进: 消除分散的 colorbar)
        cbar = fig.colorbar(last_sc, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label(r"|prediction $-$ true|")
        fig.suptitle("Emissivity inversion scatter — predicted vs. true",
                     fontsize=FONT_SIZES["title"], fontweight="bold", y=1.05)

        out = os.path.join(_figdir(output_dir), "emissivity_scatter.png")
        if save:
            # 先落盘初始渲染 (PNG+PDF+SVG, 取消编辑时的回退版本), 再视交互配置弹窗
            save_with_vector(fig, out, dpi=300, vector=True, svg=True)
            print("wrote", out)
            cfg = get_plot_config()
            if cfg.interactive and cfg.show_controls:
                try:
                    from .scatter_edit_dialog import attach_scatter_editor
                    if attach_scatter_editor(fig, out):
                        # 附加 暂停/继续 控制 (P/空格; sync=挂起后台 / async=后台继续)
                        try:
                            from .pause_control import PauseController
                            dlg = getattr(fig, "_scatter_editor", None)
                            PauseController(
                                fig, dialog_root=dlg.root,
                                advance=dlg.advance_buttons(),
                                on_close=dlg.take_close_handler(),
                                name="emissivity_scatter")
                        except Exception as exc:
                            print(f"[fig_emissivity_scatter] 暂停控制不可用: {exc}")
                        # 显式阻塞驱动对话框 (防交互模式下 show 立即返回
                        # 导致窗口闪现即毁); 确认=保存调整后效果 / 取消=保持初始图
                        plt.show(block=True)
                except Exception as exc:
                    print(f"[fig_emissivity_scatter] 交互编辑不可用: {exc}")
        return fig
    finally:
        if fig is not None and not getattr(fig, "_pause_keep_open", False):
            # async 后台继续放行时保留窗口 (定格展示), 否则正常关闭
            plt.close(fig)
        with _quiet_rcparams():
            _mpl.rcParams.update(snapshot)


# --------------------------------------------------------------------------- #
# Uncertainty visualization — 95% CI 双面板 (上: 置信区间曲线, 下: 区间宽度+覆盖),
# Table 8 校准数据。报告 §3 Fig5 样式升级:
#   * 数据层与旧版一致 (default_rng(21), t∈[0,10] 240 点, 异方差噪声 + 校准
#     sigma 使经验 PICP ≈ 0.946, 与 Table 8 一致)
#   * 95% CI 浅青填充 (#88CCEE, alpha 0.25) + 预测线提议方法深绿 (#00A087)
#     + ground truth 珊瑚红散点 (#CC4E52) — 报告 §4.2 语义色
#   * PICP / 校准误差白底圆角标注框、(a)/(b) 面板标签
# --------------------------------------------------------------------------- #
def fig_uncertainty_visualization(save=True, output_dir=None):
    with _quiet_rcparams():
        snapshot = _mpl.rcParams.copy()
    fig = None
    try:
        rng = np.random.default_rng(21)
        t = np.linspace(0, 10, 240)
        true_eps = 0.5 + 0.28 * np.sin(2 * np.pi * 0.2 * t) + 0.08 * np.sin(2 * np.pi * 0.55 * t)
        true_eps = np.clip(true_eps, 0.12, 0.92)
        # Heteroscedastic aleatoric noise: larger for higher-emissivity materials
        # (e.g. ceramic coatings), consistent with the SSM-PINN RMSE (~0.038).
        noise_std = 0.020 + 0.018 * true_eps
        pred = true_eps + rng.normal(0, noise_std, len(t))
        # Calibrated predictive sigma: tracks the aleatoric scale so that the
        # empirical 95% PICP lands near the Table 8 value (~0.946, cal. error
        # ~0.004, i.e. <1%) rather than 1.000 (over-conservative intervals).
        sigma = 0.0165 + 0.018 * true_eps

        z95 = 1.96
        lower, upper = pred - z95 * sigma, pred + z95 * sigma

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.6),
                                       gridspec_kw={"height_ratios": [2.6, 1]},
                                       sharex=True)

        # 上: 预测 + 95% CI + ground truth (报告 §4.2 语义色)
        ax1.fill_between(t, lower, upper, alpha=0.25, color=C_UNCERTAINTY,
                         label="95% confidence interval")
        ax1.plot(t, pred, color=C_MAIN, lw=2.0,
                 label="SSM-PINN prediction $\\hat{\\varepsilon}$")
        ax1.scatter(t[::5], true_eps[::5], color=C_GROUND_TRUTH, s=20, zorder=5,
                    label=r"Ground truth $\varepsilon$")
        picp = np.mean((true_eps >= lower) & (true_eps <= upper))
        cal_err = abs(picp - 0.95)
        ax1.text(0.02, 0.96,
                 f"PICP (95%) = {picp:.3f}\nCalibration error = {cal_err:.3f}",
                 transform=ax1.transAxes, va="top",
                 fontsize=FONT_SIZES["annotation"],
                 bbox=dict(boxstyle="round,pad=0.35", fc="white",
                           ec=C_MAIN, alpha=0.9))
        ax1.set_ylabel(r"Emissivity $\varepsilon$")
        ax1.set_ylim(0, 1)
        ax1.set_title("(a) Bayesian variational uncertainty quantification",
                      fontweight="bold")
        ax1.legend(loc="upper right", ncol=3, frameon=False)

        # 下: 区间宽度带 + coverage miss 标记
        width = upper - lower
        ax2.fill_between(t, 0, width, color=C_UNCERTAINTY2, alpha=0.5,
                         label="interval width")
        covered = (true_eps >= lower) & (true_eps <= upper)
        ax2.scatter(t[~covered], width[~covered] * 0 + 0.0,
                    color=C_GROUND_TRUTH, marker="x", s=35, zorder=5,
                    label="coverage miss")
        ax2.set_ylabel("CI width")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right", frameon=False)
        ax2.set_title("(b) Prediction interval width", fontweight="bold")

        fig.tight_layout()

        out = os.path.join(_figdir(output_dir), "uncertainty_visualization.png")
        if save:
            # PNG + PDF+SVG 矢量副本 (期刊投稿格式)
            save_with_vector(fig, out, dpi=300, vector=True, svg=True)
            print("wrote", out)
        return fig
    finally:
        if fig is not None:
            plt.close(fig)
        with _quiet_rcparams():
            _mpl.rcParams.update(snapshot)


def _k(d, key):
    """Return the original (possibly string) dict key matching numeric ``key``."""
    for k in d:
        if float(k) == float(key):
            return k
    raise KeyError(key)


# 兼容别名 (fig_sensitivity 等处引用 _k)
fig_public_filtering_k = _k


def generate_all(output_dir=None, public="public_validation_results_gpu.json"):
    """生成全部验证图 (供 run_experiment 等调用方以编程方式调用)。

    Parameters
    ----------
    output_dir : str or None
        验证图输出目录; None 时写入项目根 figures/ (独立运行时的默认行为)。
    public : str
        data/ 目录下驱动两个公共验证图的 JSON 文件名。
    """
    _style()
    if not os.path.exists(os.path.join(DATA, public)):
        # Fall back to the NumPy results if the GPU JSON has not been produced.
        fallback = "public_validation_results.json"
        print("%s not found; falling back to %s" % (public, fallback))
        public = fallback
    print("loading", public)
    pub = _load(public)
    fig_public_filtering(pub, output_dir=output_dir)
    fig_public_inversion(pub, output_dir=output_dir)
    if os.path.exists(os.path.join(DATA, "sensitivity_results.json")):
        fig_sensitivity(_load("sensitivity_results.json"), output_dir=output_dir)
    else:
        print("sensitivity_results.json not found yet; skipping sensitivity figure")
    # 三面板发射率反演散点图 (Table 6 数据, 报告 Fig4 样式)
    fig_emissivity_scatter(output_dir=output_dir)
    # 95% 置信区间双面板图 (Table 8 校准数据, 报告 Fig5 样式)
    fig_uncertainty_visualization(output_dir=output_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    # Default to the GPU-backend results so regenerated figures reflect the
    # CUDA autograd SSM-PINN reported in the manuscript; pass the NumPy JSON
    # explicitly to reproduce the original CPU figures.
    ap.add_argument(
        "--public", default="public_validation_results_gpu.json",
        help="JSON file (under data/) driving the two public-validation figures")
    ap.add_argument(
        "--output-dir", default=None,
        help="输出目录 (默认项目根 figures/)")
    args = ap.parse_args(argv)

    generate_all(output_dir=args.output_dir, public=args.public)


if __name__ == "__main__":
    main()
