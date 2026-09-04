"""全局绘图配置模块 (交互式窗口 + 图例/字体实时调节)

本模块是整个实验系统"可交互图片输出"的统一入口, 提供三类能力:

1. 交互式后端 (``enable_interactive_backend``)
   在支持 GUI 的环境下自动选择可交互的 matplotlib 后端 (TkAgg/Qt5Agg),
   使 ``plt.show()`` 弹出可缩放/拖动/保存的窗口; 无 GUI 时回退到 Agg 静默保存。

2. 全局默认样式 (``PlotConfig`` + ``apply_rcparams``)
   统一字体大小、图例位置、DPI、网格等默认值, 可被命令行参数或配置文件覆盖。

3. 图窗保存对话框 (``finalize_figure`` / ``figure_save_dialog``)
   交互模式下在每个图窗右侧弹出 Tkinter 对话框 (迁移自
   sample_size_sensitivity_plot.py 的控制面板思路), 支持输出格式
   (png/pdf/svg/eps/jpg/tiff)、DPI、透明背景、bbox 紧裁剪、留白、
   全局字号、图例位置/字号/显隐等丰富参数的实时调整与独立保存;
   无 Tk 环境时回退到窗口内嵌控件 (字体滑块 + 图例位置单选)。

统一约定 (所有绘图函数都应遵循):
    show=True   -> 弹出可交互窗口 (finalize_figure)
    save_path   -> 同时保存 PNG 到指定目录 (推荐: 保存+弹窗同时进行)

用户选择的模式: "同时保存 + 弹窗", 且 "图例/字体通过窗口内交互控件调节"。
"""

import os
import matplotlib

# 图例可选的 9 个标准位置 (matplotlib loc 关键字)
LEGEND_LOCATIONS = [
    'best', 'upper right', 'upper left', 'lower left', 'lower right',
    'right', 'center left', 'center right', 'lower center', 'upper center',
    'center',
]


class PlotConfig:
    """全局绘图配置 (可被命令行/配置文件覆盖)

    Attributes
    ----------
    interactive : bool
        是否弹出可交互窗口 (True: 同时保存并弹窗; False: 仅保存)。
    save : bool
        是否保存 PNG 到磁盘 (默认 True, 满足"实验结束自动保存"要求)。
    font_size : float
        全局基准字体大小 (标题为 +2, 刻度为 -1)。
    legend_loc : str
        图例默认位置, 取值见 ``LEGEND_LOCATIONS``。
    dpi : int
        保存图片的分辨率。
    show_controls : bool
        是否在窗口内注入字体/图例交互控件。
    output_dir : str
        图片保存的根目录 (由一键脚本传入指定目录)。
    """

    def __init__(self):
        self.interactive = True          # 默认: 保存 + 弹窗
        self.save = True                 # 默认: 自动保存
        self.font_size = 11.0
        self.legend_loc = 'best'
        self.dpi = 300
        self.show_controls = True        # 窗口内交互控件 (滑块/下拉框)
        self.output_dir = None
        self.pause_mode = 'sync'         # 暂停模式: sync=挂起后台 / async=后台继续
        self.journal_style = True        # 期刊主题 (白底/Arial/Nature 色板/字号层级)
        self.export_vector = True        # PNG 之外同步输出 PDF+SVG 矢量副本 (投稿)

    def apply_rcparams(self):
        """把当前配置写入 matplotlib rcParams (影响后续所有绘图)。

        期刊主题 (journal_style=True) 开启时跳过字号覆盖 —— 字号层级由
        journal_style.FONT_SIZES 统一控制 (刻度 8/图例 9/轴标签 10/子图标题 11),
        避免 cfg.font_size 默认值把期刊层级冲掉; 仅同步 DPI/bbox/图例位置。
        """
        import matplotlib.pyplot as plt
        if self.journal_style:
            try:
                from .journal_style import journal_rcparams
                journal = journal_rcparams()
            except Exception:
                journal = {}
            plt.rcParams.update({
                'figure.dpi': 150,
                'savefig.dpi': self.dpi,
                'savefig.bbox': 'tight',
                'legend.loc': self.legend_loc,
            })
            if journal:
                plt.rcParams.update({
                    'font.size': journal.get('font.size', 10),
                    'axes.titlesize': journal.get('axes.titlesize', 11),
                    'axes.labelsize': journal.get('axes.labelsize', 10),
                    'xtick.labelsize': journal.get('xtick.labelsize', 8),
                    'ytick.labelsize': journal.get('ytick.labelsize', 8),
                    'legend.fontsize': journal.get('legend.fontsize', 9),
                })
            return
        fs = self.font_size
        plt.rcParams.update({
            'font.size': fs,
            'axes.titlesize': fs + 2,
            'axes.labelsize': fs,
            'xtick.labelsize': fs - 1,
            'ytick.labelsize': fs - 1,
            'legend.fontsize': fs - 1,
            'figure.dpi': 150,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'legend.loc': self.legend_loc,
        })


# 全局单例: 所有绘图代码通过 get_plot_config() 读取当前配置
_CONFIG = PlotConfig()


def get_plot_config() -> PlotConfig:
    """获取全局绘图配置单例。"""
    return _CONFIG


def configure(interactive=None, save=None, font_size=None, legend_loc=None,
              dpi=None, show_controls=None, output_dir=None, pause_mode=None,
              journal_style=None, export_vector=None):
    """统一配置入口 (供 main / 一键脚本 / 命令行调用)。

    只更新显式传入的字段, 其余保持默认。调用后自动应用 rcParams 与后端。
    """
    cfg = _CONFIG
    if interactive is not None:
        cfg.interactive = interactive
    if save is not None:
        cfg.save = save
    if font_size is not None:
        cfg.font_size = float(font_size)
    if legend_loc is not None:
        if legend_loc not in LEGEND_LOCATIONS:
            raise ValueError(
                f"legend_loc 必须是 {LEGEND_LOCATIONS} 之一, 收到: {legend_loc}")
        cfg.legend_loc = legend_loc
    if dpi is not None:
        cfg.dpi = int(dpi)
    if show_controls is not None:
        cfg.show_controls = show_controls
    if output_dir is not None:
        cfg.output_dir = output_dir
    if pause_mode is not None:
        if pause_mode not in ('sync', 'async'):
            raise ValueError("pause_mode 必须是 'sync' 或 'async'")
        cfg.pause_mode = pause_mode
    if journal_style is not None:
        cfg.journal_style = bool(journal_style)
    if export_vector is not None:
        cfg.export_vector = bool(export_vector)

    enable_interactive_backend(cfg.interactive)
    cfg.apply_rcparams()
    try:
        apply_style()
    except Exception:
        pass
    return cfg


def apply_style():
    """应用统一绘图风格。

    期刊主题开启时委托 journal_style (白底/Arial/Nature 色板/字号层级);
    legacy 回退 seaborn 灰底, 并先复位到全新解释器基线, 避免主题键残留
    (spines/frameon 等) 造成两种样式互相污染。
    """
    import matplotlib
    import matplotlib.pyplot as plt
    if _CONFIG.journal_style:
        try:
            from .journal_style import apply_journal_style
            apply_journal_style()
            return
        except Exception:
            pass
    # legacy: 先复位到全新解释器基线, 避免主题键残留
    try:
        matplotlib.rc_file_defaults()
    except Exception:
        pass
    for style in ('seaborn-v0_8', 'seaborn'):
        try:
            plt.style.use(style)
            return
        except Exception:
            continue
    plt.style.use('default')


def enable_interactive_backend(interactive: bool):
    """按需切换 matplotlib 后端。

    interactive=True 时尝试选择支持 GUI 的后端 (TkAgg 优先, 其次 Qt5Agg),
    使 plt.show() 弹出可交互窗口; 若无可用 GUI 后端则回退到 Agg (静默保存)。
    interactive=False 时强制 Agg (无窗口, 仅保存)。
    """
    if not interactive:
        try:
            matplotlib.use('Agg', force=True)
        except Exception:
            pass
        return False

    # 若已是 TkAgg 则直接返回 (figure_save_dialog / sensitivity_edit_dialog
    # 等交互对话框依赖 tkinter, 非 Tk 后端会导致对话框附着失败而静默跳过)
    current = matplotlib.get_backend().lower()
    if "tkagg" in current:
        return True

    # 即使当前已是 Qt5Agg/QtAgg 等交互后端, 也优先尝试切到 TkAgg,
    # 因为项目的全部交互编辑对话框均基于 tkinter 构建
    for backend in ('TkAgg', 'Qt5Agg', 'QtAgg'):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue

    # 无可用 GUI 后端 (如纯命令行/CI): 回退 Agg, 仅保存不弹窗
    try:
        matplotlib.use('Agg', force=True)
    except Exception:
        pass
    _CONFIG.interactive = False
    return False


def resolve_save_path(save_path):
    """把相对文件名解析到配置的输出目录下。

    若 save_path 为绝对路径则原样返回; 否则拼接到 cfg.output_dir/figures。
    自动创建父目录。
    """
    if not save_path:
        return None
    cfg = _CONFIG
    if os.path.isabs(save_path):
        final = save_path
    elif cfg.output_dir:
        final = os.path.join(cfg.output_dir, 'figures', os.path.basename(save_path))
    else:
        final = save_path
    parent = os.path.dirname(final)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return final


def _iter_axes(fig):
    """遍历 figure 中的"数据坐标轴", 跳过控件坐标轴。"""
    for ax in fig.axes:
        if getattr(ax, '_is_control_axis', False):
            continue
        yield ax


def _apply_font_size(fig, size):
    """把字体大小实时应用到 figure 内所有文本元素。"""
    size = float(size)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(size + 3)
    for ax in _iter_axes(fig):
        ax.title.set_fontsize(size + 2)
        ax.xaxis.label.set_fontsize(size)
        ax.yaxis.label.set_fontsize(size)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(size - 1)
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(size - 1)
    fig.canvas.draw_idle()


def _apply_legend_loc(fig, loc):
    """把图例位置实时应用到 figure 内所有含图例的坐标轴。"""
    for ax in _iter_axes(fig):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc=loc)
    fig.canvas.draw_idle()


def attach_interactive_controls(fig):
    """在窗口底部注入字体大小滑块 + 图例位置下拉框。

    仅当交互后端可用且 cfg.show_controls=True 时生效; 保存 PNG 时不含控件
    (控件在 show 之前不占据数据区, 保存发生在 finalize 之前)。

    返回注入的控件对象 (需被 figure 持有引用, 否则会被 GC 回收失效)。
    """
    cfg = _CONFIG
    if not cfg.interactive or not cfg.show_controls:
        return None
    try:
        from matplotlib.widgets import Slider, RadioButtons
    except Exception:
        return None

    # 为控件预留底部空间
    try:
        fig.subplots_adjust(bottom=0.22)
    except Exception:
        pass

    # 字体大小滑块 (左下)
    ax_slider = fig.add_axes([0.12, 0.06, 0.45, 0.03])
    ax_slider._is_control_axis = True
    slider = Slider(ax_slider, 'Font size', 6.0, 24.0,
                    valinit=cfg.font_size, valstep=1.0)

    # 图例位置下拉 (用 RadioButtons 实现, 右下)
    ax_radio = fig.add_axes([0.66, 0.02, 0.30, 0.16])
    ax_radio._is_control_axis = True
    ax_radio.set_title('Legend position', fontsize=8)
    common_locs = ['best', 'upper right', 'upper left', 'lower left', 'lower right']
    init_idx = common_locs.index(cfg.legend_loc) if cfg.legend_loc in common_locs else 0
    radio = RadioButtons(ax_radio, common_locs, active=init_idx)
    for lbl in radio.labels:
        lbl.set_fontsize(8)

    def _on_font(val):
        _apply_font_size(fig, val)

    def _on_loc(label):
        _apply_legend_loc(fig, label)

    slider.on_changed(_on_font)
    radio.on_clicked(_on_loc)

    # 应用初始配置的图例位置
    _apply_legend_loc(fig, cfg.legend_loc)

    # 持有引用, 防止被回收
    fig._interactive_controls = (slider, radio)
    return fig._interactive_controls


def finalize_figure(fig, save_path=None, show=None):
    """统一收尾: 先(可选)保存 PNG, 再(可选)弹出带交互控件的窗口。

    这是所有绘图函数应调用的收尾函数, 取代原先散落的
    ``savefig`` / ``plt.show()`` / ``plt.close()`` 逻辑。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    save_path : str or None
        文件名或绝对路径; None 表示不保存。相对路径会解析到输出目录。
    show : bool or None
        None 时使用全局配置 cfg.interactive。
    """
    import matplotlib.pyplot as plt
    cfg = _CONFIG

    do_show = cfg.interactive if show is None else (show and cfg.interactive)

    # 1) 先保存 (保存的图不含交互控件, 保持论文/报告用图干净)
    #    期刊主题下经由 save_with_vector 输出 PNG+PDF+SVG (投稿格式, 报告 §4.3)
    final_path = None
    if save_path and cfg.save:
        final_path = resolve_save_path(save_path)
        if cfg.journal_style and cfg.export_vector:
            try:
                from .journal_style import save_with_vector
                save_with_vector(fig, final_path, dpi=cfg.dpi,
                                 vector=True, svg=True)
            except Exception as exc:
                print(f"[plot_config] 矢量副本输出失败: {exc}")
        else:
            try:
                fig.savefig(final_path, dpi=cfg.dpi, bbox_inches='tight')
            except Exception as exc:
                print(f"[plot_config] 保存图片失败 {final_path}: {exc}")

    # 2) 再弹窗 (优先附着"对话框式保存窗口" figure_save_dialog:
    #    格式/DPI/透明/紧裁剪/字号/图例位置等丰富参数可调并实时预览;
    #    无 Tk 环境时回退到窗口内嵌滑块+图例控件)
    if do_show:
        from .pause_control import pump_pending_events
        dialog = None
        if cfg.show_controls:
            try:
                from .figure_save_dialog import attach_save_dialog
                dialog = attach_save_dialog(fig, save_path=final_path)
            except Exception as exc:
                print(f"[plot_config] 保存对话框附着失败: {exc}")
        if dialog is None:
            attach_interactive_controls(fig)
        # 附加 暂停/继续 控制 (快捷键 P/空格; sync=挂起后台 / async=后台继续)
        controller = None
        try:
            from .pause_control import PauseController
            extra = []
            ic = getattr(fig, "_interactive_controls", None)
            if ic:
                extra += list(ic)
            controller = PauseController(
                fig,
                dialog_root=dialog.root if dialog is not None else None,
                freeze_extra=extra,
                advance=(dialog.advance_buttons() if dialog is not None else ()),
                on_close=(dialog.take_close_handler() if dialog is not None else None),
                name=os.path.basename(final_path) if final_path else "figure")
        except Exception as exc:
            print(f"[plot_config] 暂停控制不可用: {exc}")
        plt.show(block=True)
        if getattr(fig, "_pause_keep_open", False):
            # async 后台继续放行: 窗口保留为定格展示, 不在此销毁
            pass
        else:
            plt.close(fig)
        pump_pending_events()
    else:
        plt.close(fig)

    return final_path
