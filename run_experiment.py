"""M2 实验系统 —— 一键统一入口 (供 run_experiment.bat / 命令行调用)

本脚本是"打包为 Windows 可执行脚本、一键启动实验"的 Python 侧入口, 职责:

1. 解析命令行参数 (实验时长、图片交互开关、图例/字体默认值、输出目录等)。
2. 按两阶段流水线自动执行实验 (无需人工干预):
     Phase 1 (simulator): 用合成仿真数据集执行完整实验流程
         —— 采集/滤波 → 反演 → 分析 → 可视化 → 验证, 生成结果数据+图片
     Phase 2 (public):   用已下载的公开数据集 (MODIS UCSB + SLUM) 做算法
         标准化测试, 输出反演/滤波的标准化验证结果
     两阶段自动衔接, 各自使用独立工作目录, 避免历史数据累积导致归档卡顿。
3. (可选) 论文图表生成 (generate_paper_figures) / 公共数据集验证图 /
   额外一组可视化图表 (figure_generator: 消融柱状图/材料包络图/级联示意图/
   校准曲线/公开基准热力图)。
4. (可选) 启动 FigureStudio 图表编辑器 (figure_studio): 单窗口实时预览,
   支持图表细节编辑、图例位置/大小/样式调整、子图布局重排、独立保存
   (PNG/PDF/SVG+DPI) 与修改历史回溯, 取代逐图弹窗的交互方式。
5. 实验结束后, 每个阶段产生的结果已落在 run_<时间戳>/phaseX_*/ 下,
   无需再做整目录 copytree (这正是历史卡顿的根因)。

图片输出统一走 experiment_system.plot_config: 默认"仅保存 PNG, 不弹窗" (批量/
无人值守场景直接跑, 不会被编辑/保存对话框打断); 需要精修图表时加 --interactive
恢复"弹出可交互窗口 + 对话框式图片保存窗口"(figure_save_dialog, 可调整输出格式/
DPI/透明背景/紧裁剪/全局字号/图例位置与字号并独立保存), 交互模式下实验结束后可
自动启动 FigureStudio (亦可独立运行: python -m experiment_system.figure_studio)。

用法示例:
    python run_experiment.py                          # 两阶段自动执行 (仅保存)
    python run_experiment.py --duration 120           # 两阶段, 采集 120s
    python run_experiment.py --interactive            # 弹窗交互 + 编辑/保存对话框
    python run_experiment.py --data-source public     # 仅单阶段 (公开数据集)
    python run_experiment.py --data-source simulator  # 仅单阶段 (仿真合成)
    python run_experiment.py --with-paper-figures --with-validation-figures
    python run_experiment.py --with-extra-figures     # 额外生成一组可视化图表
    python run_experiment.py --figure-studio          # 结束后打开图表编辑器
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime

# 保证以脚本方式 (python run_experiment.py) 运行时能 import experiment_system 包
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 终端为 GBK 等非 Unicode 代码页时 (重定向/管道/部分沙箱终端), 进度条与报告中
# 的 █░✓✗ 等字符会触发 UnicodeEncodeError 直接中断实验; 统一降级为"替换"策略,
# 无法编码的字符输出为 '?', 保证任何终端环境下都能跑完 (不影响 UTF-8 终端显示)。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="M2 非合作目标测量实验系统 —— 一键运行入口 (两阶段自动执行)")
    # 实验主流程参数
    p.add_argument("--duration", type=int, default=60,
                   help="实验采集时长(秒), 默认 60")
    p.add_argument("--filter-type", type=str, default="ns_arkf",
                   choices=["ns_arkf", "ekf"], help="滤波器类型")
    p.add_argument("--material-id", type=int, default=1,
                   help="反演使用的材料 ID (来自数据库, 默认 1)")
    p.add_argument("--data-source", type=str, default="two-phase",
                   choices=["two-phase", "public", "database", "file", "simulator"],
                   help="数据来源: two-phase=两阶段自动执行(simulator→public, 默认) / "
                        "public=仅公开数据集 / database=查询历史实验 / "
                        "file=读本地文件 / simulator=仅实时仿真采集")
    p.add_argument("--load-experiment", type=int, default=None,
                   help="当 data-source=database/file 时, 要加载的历史实验 ID")
    p.add_argument("--no-simulator", action="store_true",
                   help="禁用仿真, 使用真实硬件 (需连接传感器)")

    # 输出与归档
    p.add_argument("--output-dir", type=str, default="./experiment_output",
                   help="结果归档根目录 (默认 ./experiment_output)")
    # NOTE: work-dir 不再默认指向 ./data (历史累积会拖慢归档);
    # 两阶段模式下每阶段自动使用独立 work_dir, 此参数仅用于单阶段模式。
    p.add_argument("--work-dir", type=str, default=None,
                   help="单阶段模式的工作数据目录 (默认在 run_dir 下自动创建)")

    # 可交互图片输出参数 (透传给 plot_config)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--interactive", action="store_true",
                   help="弹出可交互图窗与编辑/保存对话框 (默认不弹窗, 仅保存"
                        "图片; 需要精修图表时加此开关恢复弹窗交互)")
    g.add_argument("--no-interactive", action="store_true",
                   help="(兼容旧用法) 与默认行为一致: 不弹窗, 仅保存图片")
    p.add_argument("--legend-loc", type=str, default="best",
                   help="图例默认位置 (best/upper right/upper left/lower left/...)")
    p.add_argument("--font-size", type=float, default=11.0,
                   help="图表基准字体大小 (默认 11)")
    p.add_argument("--no-plot-controls", action="store_true",
                   help="弹窗内不显示字体/图例交互控件")
    p.add_argument("--pause-mode", type=str, default="sync",
                   choices=["sync", "async"],
                   help="图窗暂停行为: sync=暂停时后台同步挂起 (默认), "
                        "async=暂停时自动保存当前图并放行后台继续, "
                        "本窗口转为定格展示窗; 快捷键 P/空格 触发暂停/继续")
    p.add_argument("--style", type=str, default="eaai",
                   choices=["eaai", "legacy"],
                   help="图表主题: eaai=SCI 1区期刊风格 (白底/Arial/Nature 色盲"
                        "友好色板/字号层级/PNG+PDF+SVG 矢量副本, 默认), "
                        "legacy=旧版 seaborn 灰底风格")

    # 额外图表生成
    p.add_argument("--with-paper-figures", action="store_true",
                   help="额外生成论文架构/结果图 (generate_paper_figures)")
    p.add_argument("--with-validation-figures", action="store_true",
                   help="额外生成公共数据集验证图 (generate_validation_figures)")
    p.add_argument("--with-extra-figures", action="store_true",
                   help="额外生成一组可视化图表 (figure_generator: 消融柱状图/"
                        "材料包络图/级联示意图/校准曲线/公开基准热力图)")
    p.add_argument("--figure-studio", action="store_true",
                   help="实验结束后启动 FigureStudio 图表编辑器 (交互式调整"
                        "图例/子图布局并实时预览; 非交互模式(默认)下忽略)")
    p.add_argument("--figure-tool", action="store_true",
                   help="实验结束后启动 pubfig 交互式调图工具 (pubfig_interactive_"
                        "tool: 滑块实时调整布局/配色/字体/图例, 双矢量导出, "
                        "SCI 1区 规格预设; 亦可独立运行: python "
                        "pubfig_interactive_tool.py)")
    return p.parse_args(argv)


def _configure_plots(args, fig_output_dir):
    """按命令行参数配置全局可交互绘图。

    NOTE: fig_output_dir 须传"阶段根目录"而非其中的 figures/ 子目录:
    plot_config.resolve_save_path 会自动在其下再拼一层 figures/,
    传 phase_root 才能得到 phase_root/figures/xxx.png (与旧版布局一致),
    传 fig_dir 会产生 figures/figures/ 双层嵌套。
    """
    from experiment_system.plot_config import configure as configure_plots
    configure_plots(
        interactive=args.interactive,
        save=True,
        font_size=args.font_size,
        legend_loc=args.legend_loc,
        show_controls=not args.no_plot_controls,
        output_dir=fig_output_dir,
        pause_mode=args.pause_mode,
        journal_style=(args.style == "eaai"),
        export_vector=(args.style == "eaai"),
    )


def _run_phase(args, phase_label, data_source, phase_root, material_id=None):
    """运行单个实验阶段。

    每个阶段使用独立的工作目录 (phase_root/work_data), 避免与历史实验数据混在一起;
    图片输出到 phase_root/figures/。阶段结束后, 结果即落在 phase_root 下,
    无需额外 copytree 归档 (这正是历史卡顿的根因)。

    Parameters
    ----------
    phase_label : str   阶段标签 (用于日志)
    data_source : str   该阶段的数据源 (simulator / public / database / file)
    phase_root  : str   该阶段根目录 (如 run_dir/phase1_simulator/)
    material_id : int   反演使用的材料 ID
    """
    from experiment_system.config import DEFAULT_CONFIG, save_config
    from experiment_system.main import ExperimentSystem
    from experiment_system.progress import log_status

    work_dir = os.path.join(phase_root, "work_data")
    fig_dir = os.path.join(phase_root, "figures")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 配置该阶段的图片输出目录 (传阶段根目录, resolve_save_path 自动拼 figures/)
    _configure_plots(args, phase_root)

    config = DEFAULT_CONFIG
    config.experiment.duration_seconds = args.duration
    config.experiment.output_dir = work_dir
    config.processing.filter_type = args.filter_type
    # 两阶段均使用仿真模式 (无硬件依赖); simulator 阶段用传感器仿真,
    # public 阶段直接从公开数据集加载, use_simulator 标志不影响其数据加载。
    config.experiment.use_simulator = not args.no_simulator
    config.experiment.data_source = data_source
    config.experiment.load_experiment_id = args.load_experiment

    save_config(config, os.path.join(work_dir, "current_config.yaml"))

    log_status("=" * 60, "INFO")
    log_status(f"Phase: {phase_label}", "INFO")
    log_status(f"Data source    : {config.experiment.data_source}", "INFO")
    log_status(f"Duration (s)   : {config.experiment.duration_seconds}", "INFO")
    log_status(f"Work dir       : {work_dir}", "INFO")
    log_status(f"Figures dir    : {fig_dir}", "INFO")
    log_status("=" * 60, "INFO")

    system = ExperimentSystem(config)
    system.run_full_experiment(material_id=material_id)

    return {"phase": phase_label, "data_source": data_source,
            "work_dir": work_dir, "fig_dir": fig_dir}


def _run_paper_figures(fig_dir):
    """生成论文图表 (架构图/RMSE/散点/不确定性)。"""
    from experiment_system.generate_paper_figures import generate_all_figures
    generate_all_figures(output_dir=fig_dir)


def _run_validation_figures(fig_dir):
    """生成公共数据集验证图 (KITTI 滤波 / MODIS+SLUM 反演 / 灵敏度曲线) +
    RMSE 对比图 (rmse_comparison.png, 含交互编辑)。

    输出到 run 目录的 figures/ 下, 与旧版归档布局保持一致;
    结果 JSON (public_validation_results_gpu.json 等) 位于
    experiment_system/data/, 由 generate_all 自行加载。
    """
    try:
        from experiment_system.generate_validation_figures import generate_all
        generate_all(output_dir=fig_dir)
    except Exception as exc:
        print(f"[run_experiment] 验证图生成跳过/失败: {exc}")
    # RMSE 对比图 (原属 paper figures, 现并入验证图流, 交互模式下弹编辑对话框)
    try:
        from experiment_system.generate_paper_figures import generate_rmse_comparison
        generate_rmse_comparison(os.path.join(fig_dir, "rmse_comparison.png"))
    except Exception as exc:
        print(f"[run_experiment] RMSE 对比图生成跳过/失败: {exc}")


def _run_extra_figures(fig_dir):
    """额外生成一组可视化图表 (figure_generator, 数据取自实验 JSON/CSV)。"""
    try:
        from experiment_system.figure_generator import generate_extra_figures
        generate_extra_figures(output_dir=fig_dir, show=False)
    except Exception as exc:
        print(f"[run_experiment] 额外图表生成跳过/失败: {exc}")


def _launch_figure_studio(args):
    """启动 FigureStudio 图表编辑器 (单窗口实时预览/历史回溯, 取代逐图弹窗)。"""
    if not args.interactive:
        print("[run_experiment] 非交互模式 (默认) 下跳过 FigureStudio 启动; "
              "需要时加 --interactive 重跑")
        return
    try:
        from experiment_system.figure_studio import launch as launch_studio
        launch_studio()
    except Exception as exc:
        print(f"[run_experiment] FigureStudio 启动失败: {exc}")


def _launch_figure_tool():
    """启动 pubfig 交互式调图工具 (滑块实时预览, 布局/配色/字体/图例可调)。

    独立 GUI 应用, 交互/非交互模式下均可启动 (--figure-tool 显式请求)。
    """
    try:
        import pubfig_interactive_tool as tool
        tool.main()
    except Exception as exc:
        print(f"[run_experiment] 交互式调图工具启动失败: {exc}")


def _archive_aux_files(run_dir):
    """归档根目录下散落的辅助产物 (不复制 ./data, 避免历史累积卡顿)。

    仅为 calibration_data.json / current_config.yaml 等小文件做复制;
    阶段级数据已由 _run_phase 直接写入 run_dir/phaseX_/, 无需再 copytree。
    """
    os.makedirs(run_dir, exist_ok=True)
    copied = []
    for fname in ("calibration_data.json",):
        src = os.path.join(_ROOT, fname)
        if os.path.isfile(src):
            dst = os.path.join(run_dir, fname)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def main(argv=None):
    args = _parse_args(argv)

    started = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.output_dir, f"run_{timestamp}"))
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print(" M2 EXPERIMENT SYSTEM — ONE-CLICK RUN")
    print(f" Output (this run): {run_dir}")
    print(f" Mode            : {args.data_source}")
    print("=" * 70)

    exit_code = 0
    phases_run = []
    try:
        if args.data_source == "two-phase":
            # ---- Phase 1: simulator 合成数据集 (完整实验) ----
            p1_root = os.path.join(run_dir, "phase1_simulator")
            info1 = _run_phase(args, "Phase 1 (Simulator synthetic dataset)",
                              "simulator", p1_root, material_id=args.material_id)
            phases_run.append(info1)
            print("\n[run_experiment] Phase 1 (simulator) 完成。\n")

            # ---- Phase 2: public 公开数据集 (算法标准化测试) ----
            p2_root = os.path.join(run_dir, "phase2_public")
            info2 = _run_phase(args, "Phase 2 (Public datasets: MODIS UCSB + SLUM)",
                              "public", p2_root, material_id=args.material_id)
            phases_run.append(info2)
            print("\n[run_experiment] Phase 2 (public) 完成。\n")

            # 额外图表 (可选)
            if args.with_paper_figures:
                _run_paper_figures(os.path.join(run_dir, "paper_figures"))
            if args.with_validation_figures:
                _run_validation_figures(os.path.join(run_dir, "figures"))
            if args.with_extra_figures:
                _run_extra_figures(os.path.join(run_dir, "extra_figures"))
            if args.figure_studio:
                _launch_figure_studio(args)
            if args.figure_tool:
                _launch_figure_tool()
        else:
            # ---- 单阶段模式 (向后兼容: --data-source simulator/public/...) ----
            if args.work_dir:
                work_dir = args.work_dir
            else:
                work_dir = os.path.join(run_dir, "work_data")
            phase_root = os.path.dirname(work_dir) if args.work_dir else run_dir
            fig_dir = os.path.join(run_dir, "figures")
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(fig_dir, exist_ok=True)
            _configure_plots(args, run_dir)

            from experiment_system.config import DEFAULT_CONFIG, save_config
            from experiment_system.main import ExperimentSystem
            from experiment_system.progress import log_status

            config = DEFAULT_CONFIG
            config.experiment.duration_seconds = args.duration
            config.experiment.output_dir = work_dir
            config.processing.filter_type = args.filter_type
            config.experiment.use_simulator = not args.no_simulator
            config.experiment.data_source = args.data_source
            config.experiment.load_experiment_id = args.load_experiment
            save_config(config, os.path.join(work_dir, "current_config.yaml"))

            log_status(f"Data source    : {config.experiment.data_source}", "INFO")
            log_status(f"Duration (s)   : {config.experiment.duration_seconds}", "INFO")
            log_status(f"Work dir       : {work_dir}", "INFO")

            system = ExperimentSystem(config)
            system.run_full_experiment(material_id=args.material_id)
            phases_run.append({"phase": f"single ({args.data_source})",
                               "data_source": args.data_source,
                               "work_dir": work_dir, "fig_dir": fig_dir})

            # 额外图表 / 图表编辑器 (可选)
            if args.with_paper_figures:
                _run_paper_figures(os.path.join(run_dir, "paper_figures"))
            if args.with_validation_figures:
                _run_validation_figures(fig_dir)
            if args.with_extra_figures:
                _run_extra_figures(os.path.join(run_dir, "extra_figures"))
            if args.figure_studio:
                _launch_figure_studio(args)
            if args.figure_tool:
                _launch_figure_tool()

    except KeyboardInterrupt:
        print("\n[run_experiment] 被用户中断 (KeyboardInterrupt)")
        exit_code = 130
    except Exception as exc:
        import traceback
        print(f"\n[run_experiment] 实验运行出错: {exc}")
        traceback.print_exc()
        exit_code = 1
    finally:
        # 归档: 阶段数据已直接落在 run_dir/phaseX_/ 下, 仅复制少量根目录辅助文件。
        # 不再 copytree 整个 ./data (历史累积导致卡顿的根因)。
        print("\n" + "-" * 70)
        print("[run_experiment] 归档实验结果 ...")
        copied = _archive_aux_files(run_dir)
        for c in copied:
            print(f"  已保存: {c}")
        for info in phases_run:
            print(f"  阶段 [{info['phase']}] 数据: {info['work_dir']}")
            print(f"  阶段 [{info['phase']}] 图片: {info['fig_dir']}")
        elapsed = time.time() - started
        print(f"[run_experiment] 完成, 用时 {elapsed:.1f}s, 结果目录: {run_dir}")
        print("-" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
