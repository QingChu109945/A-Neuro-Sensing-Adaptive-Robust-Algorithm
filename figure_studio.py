"""FigureStudio —— 可视化图表交互式编辑器

针对评审"弹窗交互不灵活、图例/子图无法自由调整、无历史回溯"的问题,
提供一个**单窗口、实时预览、可回溯**的图表编辑器, 取代逐图弹窗:

功能
----
1. 图表细节编辑   标题/坐标轴标签/字体大小/网格/对数轴/颜色, 逐子图编辑
2. 独立保存选项   Save As 对话框, 支持 PNG / PDF / SVG + DPI 自选,
                   与实验流程完全独立 (不依赖 run_experiment)
3. 图例灵活调整   位置 (11 个标准 loc)、字号、列数、边框、透明度、标题
4. 子图自由布局   行 x 列重排、子图间距 (wspace/hspace)、子图顺序互换
5. 实时预览       所有调整立即反映到右侧画布 (60ms 防抖)
6. 修改历史回溯   每次应用记录快照; 撤销/重做/双击历史项跳转;
                   保存图形时写 sidecar ``<名字>.history.json`` 持久化
7. 解决弹窗不灵活 单窗口管理全部图表 (下拉切换), 不再每张图弹一个窗口

数据来源: ``experiment_system.figure_generator.EXTRA_FIGURES`` 注册表
(消融图/材料包络图/级联示意图/校准图/热力图), 亦支持自定义数据目录。

用法:
    python -m experiment_system.figure_studio                 # 图形界面
    python -m experiment_system.figure_studio --list          # 列出可编辑图
    python -m experiment_system.figure_studio --only heatmap  # 启动后载入指定图
    python -m experiment_system.figure_studio --data-dir DIR  # 自定义数据目录

无显示环境 (远程/CI) 自动降级为 Agg 并提示, 不崩溃。
"""
import copy
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.text import Annotation
import numpy as np

try:
    from . import figure_generator as fg
    from .plot_config import get_plot_config
except ImportError:
    from experiment_system import figure_generator as fg
    from experiment_system.plot_config import get_plot_config

LEGEND_LOCS = ['best', 'upper right', 'upper left', 'lower left',
               'lower right', 'right', 'center left', 'center right',
               'lower center', 'upper center', 'center']
SAVE_FORMATS = {'PNG': '.png', 'PDF': '.pdf', 'SVG': '.svg', 'EPS': '.eps'}

# 历史文件命名
HIST_SUFFIX = '.history.json'


# --------------------------------------------------------------------------- #
# 通用子图内容提取/重建 (支持折线、柱状、散点、热力图、示意图)
# --------------------------------------------------------------------------- #
def _capture_axis(ax):
    """把一个数据轴的全部可重建内容提取为纯数据 dict。"""
    cap = {
        'title': ax.get_title(),
        'xlabel': ax.get_xlabel(),
        'ylabel': ax.get_ylabel(),
        'xscale': ax.get_xscale(),
        'yscale': ax.get_yscale(),
        'xlim': ax.get_xlim(),
        'ylim': ax.get_ylim(),
        'grid': bool(ax.xaxis.get_gridlines() and
                     ax.xaxis._major_tick_kw.get('gridOn', False)),
        'lines': [],
        'bars': [],
        'scatters': [],
        'fills': [],
        'images': [],
        'texts': [],
        'annotations': [],
        'legend': None,
    }
    for ln in ax.lines:
        cap['lines'].append({
            'x': ln.get_xdata(), 'y': ln.get_ydata(),
            'color': ln.get_color(), 'lw': ln.get_linewidth(),
            'ls': ln.get_linestyle(), 'marker': ln.get_marker(),
            'ms': ln.get_markersize(), 'label': ln.get_label(),
            'alpha': ln.get_alpha(),
        })
    for p in ax.patches:
        try:
            cap['bars'].append({
                'xy': list(p.get_xy()), 'w': float(p.get_width()),
                'h': float(p.get_height()),
                'fc': p.get_facecolor(), 'ec': p.get_edgecolor(),
                'lw': p.get_linewidth(), 'label': p.get_label(),
                'alpha': p.get_alpha(),
            })
        except Exception:
            continue
    for coll in ax.collections:
        paths = coll.get_paths()
        if not paths:
            continue
        try:
            pts = np.asarray(paths[0].vertices)
        except Exception:
            continue
        cap['fills'].append({
            'pts': pts, 'fc': coll.get_facecolor(),
            'alpha': coll.get_alpha(), 'label': coll.get_label(),
        })
    for coll in ax.collections:
        if not hasattr(coll, 'get_offsets') or not hasattr(coll, 'get_sizes'):
            continue
        offs = np.asarray(coll.get_offsets())
        if offs.ndim != 2 or len(offs) == 0:
            continue
        cap['scatters'].append({
            'xy': offs, 'fc': coll.get_facecolors(),
            'ec': coll.get_edgecolors(), 's': coll.get_sizes(),
            'label': coll.get_label(), 'alpha': coll.get_alpha(),
        })
    for im in ax.images:
        cap['images'].append({
            'data': im.get_array(), 'cmap': im.get_cmap().name,
            'interpolation': 'nearest', 'extent': im.get_extent(),
            'alpha': im.get_alpha(),
        })
    for t in ax.texts:
        if isinstance(t, Annotation):
            # annotate() 创建的 Annotation 也挂在 ax.texts 下, 单独捕获箭头
            try:
                cap['annotations'].append({
                    'xy': tuple(t.xy), 'xytext': tuple(t.xyann),
                    's': t.get_text(),
                    'color': t.arrow_patch.get_facecolor()
                    if t.arrow_patch is not None else 'black',
                })
            except Exception:
                pass
            continue
        cap['texts'].append({
            'x': t.get_position()[0], 'y': t.get_position()[1],
            's': t.get_text(), 'fontsize': t.get_fontsize(),
            'ha': t.get_ha(), 'va': t.get_va(), 'color': t.get_color(),
            'rotation': t.get_rotation(),
        })
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        cap['legend'] = {'labels': list(labels)}
    return cap


def _rebuild_axis(ax, cap):
    """按捕获的数据在空白轴上重建子图内容。"""
    for ln in cap['lines']:
        kw = dict(color=ln['color'], lw=ln['lw'], ls=ln['ls'],
                  marker=ln['marker'], ms=ln['ms'], label=ln['label'])
        if ln['alpha'] is not None:
            kw['alpha'] = ln['alpha']
        try:
            ax.plot(np.asarray(ln['x']), np.asarray(ln['y']), **kw)
        except Exception:
            continue
    for b in cap['bars']:
        kw = dict(width=b['w'], height=b['h'], facecolor=b['fc'],
                  edgecolor=b['ec'], lw=b['lw'], label=b['label'])
        if b['alpha'] is not None:
            kw['alpha'] = b['alpha']
        try:
            # Rectangle.get_xy() -> (x0, y0) 左下角
            ax.add_patch(plt.Rectangle((b['xy'][0], b['xy'][1]), **kw))
        except Exception:
            continue
    for f in cap['fills']:
        try:
            ax.fill(f['pts'][:, 0], f['pts'][:, 1],
                    color=f['fc'][0] if len(f['fc']) else COLOR_BLUE,
                    alpha=f['alpha'] if f['alpha'] is not None else 0.3,
                    label=f['label'] or None, zorder=1)
        except Exception:
            continue
    for s in cap['scatters']:
        try:
            ax.scatter(s['xy'][:, 0], s['xy'][:, 1], s=s['s'],
                       c=s['fc'], edgecolors=s['ec'] if len(s['ec']) else None,
                       label=s['label'] or None,
                       alpha=s['alpha'] if s['alpha'] is not None else 1.0)
        except Exception:
            continue
    for im in cap['images']:
        try:
            ax.imshow(im['data'], cmap=im['cmap'],
                      interpolation=im['interpolation'],
                      extent=im['extent'],
                      alpha=im['alpha'] if im['alpha'] is not None else 1.0)
        except Exception:
            continue
    for t in cap['texts']:
        ax.text(t['x'], t['y'], t['s'], fontsize=t['fontsize'], ha=t['ha'],
                va=t['va'], color=t['color'], rotation=t['rotation'])
    for a in cap['annotations']:
        ax.annotate(a.get('s', ''), xy=a['xy'], xytext=a['xytext'],
                    arrowprops=dict(arrowstyle='-|>', lw=1.6,
                                    color=a['color'] if isinstance(
                                        a['color'], str) else 'black'))
    ax.set_title(cap['title'])
    ax.set_xlabel(cap['xlabel'])
    ax.set_ylabel(cap['ylabel'])
    try:
        ax.set_xscale(cap['xscale'])
        ax.set_yscale(cap['yscale'])
    except Exception:
        pass
    ax.set_xlim(cap['xlim'])
    ax.set_ylim(cap['ylim'])
    if cap['grid']:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)
    return cap['legend']


COLOR_BLUE = '#1f77b4'


def _data_axes(fig):
    """figure 中可编辑的数据轴 (跳过控件轴与 colorbar 轴)。"""
    return [ax for ax in fig.axes
            if not getattr(ax, '_is_control_axis', False)
            and ax.get_label() != '<colorbar>']


def apply_layout(fig, rows, cols, wspace=0.28, hspace=0.38):
    """把 figure 中现有数据子图按 rows x cols 网格重排 (内容迁移重建)。"""
    data_axes = _data_axes(fig)
    caps = [_capture_axis(ax) for ax in data_axes]
    suptitle = fig._suptitle.get_text() if fig._suptitle else ''

    fig.clf()
    # 网格自动扩容, 保证全部子图都放得下 (不覆盖丢失)
    n = len(caps)
    eff_rows = max(rows, (n + cols - 1) // cols)
    gs = fig.add_gridspec(eff_rows, cols, wspace=wspace, hspace=hspace)
    legend_caps = []
    for i, cap in enumerate(caps):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        legend_caps.append(_rebuild_axis(ax, cap))
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.995)
    # 重建图例
    for ax, lcap in zip(fig.axes, legend_caps):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    try:
        fig.tight_layout()
    except Exception:
        pass  # twinx/colorbar 轴不兼容 tight_layout 时保持当前布局
    return fig


# --------------------------------------------------------------------------- #
# 默认可编辑图源 (额外图表注册表 + 演示空图)
# --------------------------------------------------------------------------- #
def _build_demo(data_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.linspace(0, 10, 100)
    axes[0].plot(x, np.sin(x), label='sin', color=COLOR_BLUE)
    axes[0].plot(x, np.cos(x), label='cos', color='#2ca02c')
    axes[0].set_title('Demo (a) curves')
    axes[0].set_xlabel('t (s)')
    axes[0].set_ylabel('value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(['A', 'B', 'C'], [3, 5, 2], color='#aab7c4',
                edgecolor='black', label='series 1')
    axes[1].bar(['A', 'B', 'C'], [1, 2, 3], color=COLOR_BLUE,
                edgecolor='black', label='series 2')
    axes[1].set_title('Demo (b) bars')
    axes[1].set_ylabel('count')
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return fig, None


# --------------------------------------------------------------------------- #
# 验证图适配器 (generate_validation_figures 的三张数据图, 可在 Studio 中编辑)
# --------------------------------------------------------------------------- #
def _load_validation_json(names, data_dir=None):
    for name in names:
        path = os.path.join(data_dir or fg.DATA_DIR, name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
    return None


def _build_val_filtering(data_dir=None):
    from .generate_validation_figures import fig_public_filtering
    res = _load_validation_json(['public_validation_results_gpu.json',
                                 'public_validation_results.json'], data_dir)
    if res is None:
        return None, 'public_validation_results*.json 不存在, 跳过'
    return fig_public_filtering(res, save=False), None


def _build_val_inversion(data_dir=None):
    from .generate_validation_figures import fig_public_inversion
    res = _load_validation_json(['public_validation_results_gpu.json',
                                 'public_validation_results.json'], data_dir)
    if res is None:
        return None, 'public_validation_results*.json 不存在, 跳过'
    return fig_public_inversion(res, save=False), None


def _build_val_sensitivity(data_dir=None):
    from .generate_validation_figures import fig_sensitivity
    res = _load_validation_json(['sensitivity_results.json'], data_dir)
    if res is None:
        return None, 'sensitivity_results.json 不存在, 跳过'
    return fig_sensitivity(res, save=False), None


def _available_sources(data_dir):
    sources = {'Demo (editable)': _build_demo}
    for key, info in fg.EXTRA_FIGURES.items():
        label = f"{key} — {info['description']}"
        sources[label] = info['builder']
    # 公共数据集验证图 / 敏感性图 (Studio 内可编辑副本)
    sources['val:filtering — 公开滤波验证 RMSE 柱状图'] = _build_val_filtering
    sources['val:inversion — 公开反演验证 RMSE/违反率图'] = _build_val_inversion
    sources['val:sensitivity — 超参数敏感性曲线'] = _build_val_sensitivity
    return sources


def _default_state(source_label):
    return {
        'source': source_label,
        'title': '',
        'title_size': 13.0,
        'font_size': 10.0,
        'grid': True,
        'rows': 1,
        'cols': 2,
        'wspace': 0.28,
        'hspace': 0.38,
        'legend_loc': 'best',
        'legend_size': 8.0,
        'legend_ncol': 2,
        'legend_frame': True,
        'legend_alpha': 0.9,
        'legend_title': '',
        'axes': [],          # [{title,xlabel,ylabel,yscale}]
        'selected_ax': 0,
    }


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class FigureStudio(tk.Tk):
    def __init__(self, data_dir=None, initial=None):
        super().__init__()
        self.title('M2 Figure Studio — 可视化图表编辑器')
        self.geometry('1480x880')

        self.data_dir = data_dir
        self.sources = _available_sources(data_dir)
        self.state = _default_state(initial or next(iter(self.sources)))
        self.history = [{'time': datetime.now().strftime('%H:%M:%S'),
                         'desc': '初始状态', 'state': copy.deepcopy(self.state)}]
        self.hist_index = 0
        self._rebuild_job = None
        self._current_caps = []      # 重建时的原始子图内容缓存

        self._build_ui()
        self.rebuild(record=False)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ----------------------------- UI 构建 -------------------------------- #
    def _build_ui(self):
        top = ttk.Frame(self, padding=4)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text='图表:').pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value=self.state['source'])
        self.source_cb = ttk.Combobox(top, textvariable=self.source_var,
                                      values=list(self.sources),
                                      width=46, state='readonly')
        self.source_cb.pack(side=tk.LEFT, padx=4)
        self.source_cb.bind('<<ComboboxSelected>>', self._on_source_change)
        ttk.Button(top, text='载入', width=6,
                   command=self._on_source_change).pack(side=tk.LEFT)
        ttk.Button(top, text='↩ 撤销', command=self.undo).pack(side=tk.LEFT,
                                                              padx=(16, 2))
        ttk.Button(top, text='↪ 重做', command=self.redo).pack(side=tk.LEFT,
                                                              padx=2)
        ttk.Label(top, text='格式:').pack(side=tk.LEFT, padx=(18, 2))
        self.fmt_var = tk.StringVar(value='PNG')
        ttk.Combobox(top, textvariable=self.fmt_var, values=list(SAVE_FORMATS),
                     width=5, state='readonly').pack(side=tk.LEFT)
        ttk.Label(top, text='DPI:').pack(side=tk.LEFT, padx=(8, 2))
        self.dpi_var = tk.IntVar(value=300)
        ttk.Spinbox(top, from_=72, to=600, increment=24,
                    textvariable=self.dpi_var, width=5).pack(side=tk.LEFT)
        ttk.Button(top, text='保存图形...', command=self.save_figure_as
                   ).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text='清空历史',
                   command=self.clear_history).pack(side=tk.LEFT)

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 左侧控制面板
        panel = ttk.Notebook(body)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        self._page_figure(panel)
        self._page_legend(panel)
        self._page_layout(panel)
        self._page_axes(panel)
        self._page_history(panel)

        # 右侧画布
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(10, 6), dpi=110)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=
                                            False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                         expand=True)

        self.status = ttk.Label(self, text='就绪', anchor='w',
                                relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _page_figure(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text=' Figure ')
        row = lambda: ttk.Frame(f); row().pack(fill=tk.X, pady=2)

        ttk.Label(row(), text='总标题').pack(side=tk.LEFT)
        self.e_title = ttk.Entry(row())
        self.e_title.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.e_title.bind('<KeyRelease>', lambda e: self._update(
            title=self.e_title.get()))

        r = row()
        ttk.Label(r, text='总标题字号').pack(side=tk.LEFT)
        self.s_title_size = ttk.Scale(r, from_=8, to=26, value=13,
                                      command=lambda v: self._update(
                                          title_size=float(v)))
        self.s_title_size.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        r = row()
        ttk.Label(r, text='全局字体').pack(side=tk.LEFT)
        self.s_font = ttk.Scale(r, from_=6, to=22, value=10,
                                command=lambda v: self._update(
                                    font_size=float(v)))
        self.s_font.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.v_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='显示网格', variable=self.v_grid,
                        command=lambda: self._update(grid=self.v_grid.get())
                        ).pack(anchor='w', pady=4)
        ttk.Label(f, foreground='#666', wraplength=230, justify=tk.LEFT,
                  text=('提示: 总标题/字体/网格即时生效; '
                        '选中子图页可单独编辑每个子图的标签与坐标轴。')).pack(
            anchor='w', pady=(12, 0))

    def _page_legend(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text=' Legend ')
        r0 = ttk.Frame(f); r0.pack(fill=tk.X, pady=2)
        ttk.Label(r0, text='位置').pack(side=tk.LEFT)
        self.v_loc = tk.StringVar(value='best')
        cb = ttk.Combobox(r0, textvariable=self.v_loc, values=LEGEND_LOCS,
                          state='readonly', width=14)
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind('<<ComboboxSelected>>',
                lambda e: self._update(legend_loc=self.v_loc.get()))

        r1 = ttk.Frame(f); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text='字号').pack(side=tk.LEFT)
        self.s_lsize = ttk.Scale(r1, from_=5, to=18, value=8,
                                 command=lambda v: self._update(
                                     legend_size=float(v)))
        self.s_lsize.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        r2 = ttk.Frame(f); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text='列数').pack(side=tk.LEFT)
        self.v_ncol = tk.IntVar(value=2)
        ttk.Spinbox(r2, from_=1, to=6, textvariable=self.v_ncol, width=4,
                    command=lambda: self._update(
                        legend_ncol=self.v_ncol.get())).pack(side=tk.LEFT)
        self.v_ncol.trace_add('write', lambda *a: self._update(
            legend_ncol=self.v_ncol.get()))

        self.v_frame = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='显示边框', variable=self.v_frame,
                        command=lambda: self._update(
                            legend_frame=self.v_frame.get())
                        ).pack(anchor='w', pady=2)

        r3 = ttk.Frame(f); r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text='底色不透明度').pack(side=tk.LEFT)
        self.s_alpha = ttk.Scale(r3, from_=0.1, to=1.0, value=0.9,
                                 command=lambda v: self._update(
                                     legend_alpha=float(v)))
        self.s_alpha.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        r4 = ttk.Frame(f); r4.pack(fill=tk.X, pady=2)
        ttk.Label(r4, text='图例标题').pack(side=tk.LEFT)
        self.e_ltitle = ttk.Entry(r4)
        self.e_ltitle.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.e_ltitle.bind('<KeyRelease>',
                           lambda e: self._update(
                               legend_title=self.e_ltitle.get()))

    def _page_layout(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text=' Layout ')
        r1 = ttk.Frame(f); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text='行数').pack(side=tk.LEFT)
        self.v_rows = tk.IntVar(value=1)
        ttk.Spinbox(r1, from_=1, to=5, textvariable=self.v_rows, width=4,
                    command=lambda: self._update(
                        rows=self.v_rows.get())).pack(side=tk.LEFT, padx=4)
        self.v_rows.trace_add('write', lambda *a: self._update(
            rows=self.v_rows.get()))
        ttk.Label(r1, text='列数').pack(side=tk.LEFT, padx=(12, 0))
        self.v_cols = tk.IntVar(value=2)
        ttk.Spinbox(r1, from_=1, to=5, textvariable=self.v_cols, width=4,
                    command=lambda: self._update(
                        cols=self.v_cols.get())).pack(side=tk.LEFT, padx=4)
        self.v_cols.trace_add('write', lambda *a: self._update(
            cols=self.v_cols.get()))

        r2 = ttk.Frame(f); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text='水平间距').pack(side=tk.LEFT)
        self.s_wspace = ttk.Scale(r2, from_=0.02, to=0.8, value=0.28,
                                  command=lambda v: self._update(
                                      wspace=float(v)))
        self.s_wspace.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        r3 = ttk.Frame(f); r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text='垂直间距').pack(side=tk.LEFT)
        self.s_hspace = ttk.Scale(r3, from_=0.02, to=0.9, value=0.38,
                                  command=lambda v: self._update(
                                      hspace=float(v)))
        self.s_hspace.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        r4 = ttk.Frame(f); r4.pack(fill=tk.X, pady=(10, 2))
        ttk.Label(r4, text='子图顺序互换').pack(side=tk.LEFT)
        self.v_swap_a = tk.StringVar(value='0')
        self.v_swap_b = tk.StringVar(value='1')
        ttk.Spinbox(r4, from_=0, to=8, textvariable=self.v_swap_a,
                    width=3).pack(side=tk.LEFT, padx=4)
        ttk.Spinbox(r4, from_=0, to=8, textvariable=self.v_swap_b,
                    width=3).pack(side=tk.LEFT, padx=4)
        ttk.Button(r4, text='互换', command=self._swap_axes).pack(
            side=tk.LEFT, padx=6)

        r5 = ttk.Frame(f); r5.pack(fill=tk.X, pady=2)
        ttk.Label(r5, text='选中子图').pack(side=tk.LEFT)
        self.v_sel = tk.StringVar(value='0')
        self.v_sel.trace_add('write', lambda *a: self._select_ax())
        ttk.Spinbox(r5, from_=0, to=11, textvariable=self.v_sel, width=3,
                    state='readonly').pack(side=tk.LEFT, padx=4)
        self.lbl_sel = ttk.Label(f, text='(编号从 0 开始, 行优先)',
                                 foreground='#666')
        self.lbl_sel.pack(anchor='w')

    def _page_axes(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text=' Axes ')
        for label, attr in (('子图标题', 'ax_title'), ('X 轴标签', 'ax_xlabel'),
                            ('Y 轴标签', 'ax_ylabel')):
            r = ttk.Frame(f); r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=10).pack(side=tk.LEFT)
            e = ttk.Entry(r)
            e.pack(side=tk.LEFT, fill=tk.X, expand=True)
            e.bind('<KeyRelease>',
                   lambda e_, a=attr: self._apply_ax_prop(a, e_.get()))
            setattr(self, attr, e)
        r = ttk.Frame(f); r.pack(fill=tk.X, pady=4)
        ttk.Label(r, text='Y 轴刻度').pack(side=tk.LEFT)
        self.v_yscale = tk.StringVar(value='linear')
        ttk.Combobox(r, textvariable=self.v_yscale,
                     values=['linear', 'log'], state='readonly', width=8
                     ).pack(side=tk.LEFT, padx=4)
        self.v_yscale.trace_add('write', lambda *a: self._apply_ax_prop(
            'ax_yscale', self.v_yscale.get()))
        ttk.Button(f, text='应用选中子图并记录历史',
                   command=self._commit_ax_edits).pack(anchor='w', pady=8)

    def _page_history(self, nb):
        f = ttk.Frame(nb, padding=8)
        nb.add(f, text=' History ')
        ttk.Label(f, text='修改历史 (双击回溯到该状态)').pack(anchor='w')
        self.hist_list = tk.Listbox(f, height=16, activestyle='dotbox')
        self.hist_list.pack(fill=tk.BOTH, expand=True, pady=4)
        self.hist_list.bind('<Double-Button-1>', self._goto_history)
        r = ttk.Frame(f); r.pack(fill=tk.X)
        ttk.Button(r, text='跳转到选中状态',
                   command=self._goto_history).pack(side=tk.LEFT, padx=2)
        ttk.Button(r, text='导出历史 JSON...',
                   command=self.export_history).pack(side=tk.LEFT, padx=4)

    # ----------------------------- 状态/重绘 ------------------------------- #
    def _update(self, **kw):
        self.state.update(kw)
        self.schedule_rebuild(record=False)   # 实时预览: 不刷历史
        self._dirty = True

    def _commit(self, desc):
        """把当前状态作为一条历史记录 (用于离散操作: 互换/应用/换图)。"""
        self.push_history(desc)

    def _on_source_change(self, event=None):
        label = self.source_var.get()
        if label not in self.sources:
            return
        self.state = _default_state(label)
        self.rebuild(record=True, desc=f'载入图表: {label}')

    def schedule_rebuild(self, record=False, desc=None):
        if self._rebuild_job:
            self.after_cancel(self._rebuild_job)
        self._rebuild_job = self.after(60, lambda: self.rebuild(
            record=record, desc=desc))

    def rebuild(self, record=False, desc=None):
        self._rebuild_job = None
        label = self.state['source']
        builder = self.sources.get(label)
        if builder is None:
            return
        try:
            fig, warn = builder(self.data_dir)
        except Exception as exc:
            messagebox.showerror('生成图表失败', str(exc))
            return
        if fig is None:
            self.status.config(text=f'未生成: {warn}')
            return
        old_fig = self.fig
        self.fig = fig
        # 布局重排: 仅当目标 rows x cols 与原图数据子图数不同时执行,
        # 避免破坏原生布局 (如 twinx 副轴、colorbar 位置)
        n_axes = len(_data_axes(self.fig))
        if self.state['rows'] * self.state['cols'] != n_axes:
            apply_layout(self.fig, self.state['rows'], self.state['cols'],
                         self.state['wspace'], self.state['hspace'])
        self._refresh_axes_panel()
        # 复用同一画布, 仅替换其上的 figure (避免旧画布残留/新画布不显示)
        self.canvas.figure = self.fig
        self._apply_all()
        self.canvas.draw()
        try:
            plt.close(old_fig)  # 回收 pyplot 注册的旧 figure, 防止内存累积
        except Exception:
            pass
        if record:
            self.push_history(desc or label)
        self.status.config(
            text=f'当前: {label} | 子图数 {len(_data_axes(self.fig))} | '
                 f'历史 {self.hist_index + 1}/{len(self.history)}'
                 + (f' | {warn}' if warn else ''))

    def _apply_all(self):
        st = self.state
        if st['title']:
            self.fig.suptitle(st['title'], fontsize=st['title_size'], y=0.995)
        elif self.fig._suptitle:
            self.fig.suptitle('')
        for ax in _data_axes(self.fig):
            if getattr(ax, '_is_control_axis', False):
                continue
            ax.grid(st['grid'], alpha=0.3) if st['grid'] else ax.grid(False)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                leg = ax.legend(
                    handles, labels, loc=st['legend_loc'],
                    fontsize=st['legend_size'], ncol=st['legend_ncol'],
                    frameon=st['legend_frame'],
                    framealpha=st['legend_alpha'],
                    title=st['legend_title'] or None)
                if st['legend_title']:
                    leg.get_title().set_fontsize(st['legend_size'])
        self.canvas.draw_idle()

    def _refresh_axes_panel(self):
        """按当前子图数量刷新 Axes 页与 Layout 页的取值范围。"""
        n = len(_data_axes(self.fig))
        self._ax_edits = [dict(title='', xlabel='', ylabel='', yscale=None)]
        self._load_ax_fields()

    def _load_ax_fields(self):
        idx = int(self.state.get('selected_ax', 0) or 0)
        axes = _data_axes(self.fig)
        if not axes:
            return
        idx = max(0, min(idx, len(axes) - 1))
        ax = axes[idx]
        self.ax_title.delete(0, tk.END)
        self.ax_title.insert(0, ax.get_title())
        self.ax_xlabel.delete(0, tk.END)
        self.ax_xlabel.insert(0, ax.get_xlabel())
        self.ax_ylabel.delete(0, tk.END)
        self.ax_ylabel.insert(0, ax.get_ylabel())
        self.v_yscale.set(ax.get_yscale())

    def _select_ax(self):
        try:
            self.state['selected_ax'] = int(self.v_sel.get())
        except ValueError:
            return
        try:
            self._load_ax_fields()
        except Exception:
            pass

    def _apply_ax_prop(self, attr, value):
        if not hasattr(self, '_ax_pending'):
            self._ax_pending = {}
        self._ax_pending[attr] = value

    def _commit_ax_edits(self):
        pending = getattr(self, '_ax_pending', {})
        if not pending:
            return
        idx = int(self.state.get('selected_ax', 0) or 0)
        axes = _data_axes(self.fig)
        idx = max(0, min(idx, len(axes) - 1))
        ax = axes[idx]
        if 'ax_title' in pending:
            ax.set_title(pending['ax_title'])
        if 'ax_xlabel' in pending:
            ax.set_xlabel(pending['ax_xlabel'])
        if 'ax_ylabel' in pending:
            ax.set_ylabel(pending['ax_ylabel'])
        if 'ax_yscale' in pending:
            try:
                ax.set_yscale(pending['ax_yscale'])
            except Exception:
                pass
        self._ax_pending = {}
        self.canvas.draw_idle()
        self.push_history(f'编辑子图 {idx} 标签/刻度')

    def _swap_axes(self):
        try:
            a = int(self.v_swap_a.get())
            b = int(self.v_swap_b.get())
        except ValueError:
            return
        axes = _data_axes(self.fig)
        if not (0 <= a < len(axes) and 0 <= b < len(axes)) or a == b:
            return
        ca, cb = _capture_axis(axes[a]), _capture_axis(axes[b])
        _rebuild_axis(axes[a], cb)
        _rebuild_axis(axes[b], ca)
        self._apply_all()
        self.push_history(f'互换子图 {a} <-> {b}')

    # ----------------------------- 历史机制 -------------------------------- #
    def push_history(self, desc):
        snapshot = copy.deepcopy(self.state)
        stamp = datetime.now().strftime('%H:%M:%S')
        entry = {'time': stamp, 'desc': desc, 'state': snapshot}
        self.history = self.history[:self.hist_index + 1]
        self.history.append(entry)
        self.hist_index = len(self.history) - 1
        self._refresh_history_list()
        self._sync_fields_from_state()

    def _refresh_history_list(self):
        self.hist_list.delete(0, tk.END)
        for i, e in enumerate(self.history):
            mark = ' *' if i == self.hist_index else ''
            self.hist_list.insert(tk.END,
                                  f"[{e['time']}] {e['desc']}{mark}")
        self.hist_list.selection_clear(0, tk.END)
        self.hist_list.selection_set(self.hist_index)

    def undo(self):
        if self.hist_index > 0:
            self.hist_index -= 1
            self._restore(self.history[self.hist_index])

    def redo(self):
        if self.hist_index < len(self.history) - 1:
            self.hist_index += 1
            self._restore(self.history[self.hist_index])

    def _goto_history(self, event=None):
        sel = self.hist_list.curselection()
        if sel:
            self.hist_index = sel[0]
            self._restore(self.history[self.hist_index])

    def _restore(self, entry):
        self.state = copy.deepcopy(entry['state'])
        self.source_var.set(self.state['source'])
        self.rebuild(record=False)
        self._sync_fields_from_state()
        self._refresh_history_list()
        self.status.config(text=f"已回溯: [{entry['time']}] {entry['desc']}")

    def clear_history(self):
        self.history = [copy.deepcopy(self.state)]
        self.hist_index = 0
        self._refresh_history_list()

    def _sync_fields_from_state(self):
        st = self.state
        self.e_title.delete(0, tk.END)
        self.e_title.insert(0, st['title'] or '')
        self.v_grid.set(st['grid'])
        self.v_loc.set(st['legend_loc'])
        self.v_frame.set(st['legend_frame'])
        self.v_rows.set(st['rows'])
        self.v_cols.set(st['cols'])

    # ----------------------------- 保存 ----------------------------------- #
    def save_figure_as(self):
        ext = SAVE_FORMATS.get(self.fmt_var.get(), '.png')
        fname = self.state['source'].split(' ')[0].replace('/', '_') or 'figure'
        default = f'{fname}_{datetime.now().strftime("%H%M%S")}{ext}'
        path = filedialog.asksaveasfilename(
            defaultextension=ext, initialfile=default,
            filetypes=[('Figure', f'*{ext}'), ('All files', '*.*')])
        if not path:
            return
        dpi = int(self.dpi_var.get() or 300)
        try:
            self.fig.savefig(path, dpi=dpi, bbox_inches='tight')
        except Exception as exc:
            messagebox.showerror('保存失败', str(exc))
            return
        hist_path = os.path.splitext(path)[0] + HIST_SUFFIX
        try:
            with open(hist_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'png': path, 'saved': datetime.now().isoformat(),
                    'dpi': dpi, 'history': self.history,
                }, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass
        self.status.config(text=f'已保存: {path} (历史: {hist_path})')

    def export_history(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.json', initialfile='figure_history.json')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.history, fh, ensure_ascii=False, indent=2)
        self.status.config(text=f'历史已导出: {path}')

    def _on_close(self):
        try:
            plt.close('all')
        finally:
            self.destroy()


# --------------------------------------------------------------------------- #
def launch(data_dir=None, initial=None):
    """启动 FigureStudio; 无显示环境时降级为提示 (不崩溃)。"""
    try:
        matplotlib.use('TkAgg', force=True)
        app = FigureStudio(data_dir=data_dir, initial=initial)
        app.mainloop()
        return 0
    except tk.TclError as exc:
        print(f'[figure_studio] 无可用显示环境 (Tk): {exc}\n'
              '  请在本地桌面环境运行, 或改用:\n'
              '    python -m experiment_system.figure_generator --no-interactive')
        return 1


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='可视化图表交互式编辑器')
    ap.add_argument('--data-dir', default=None,
                    help='JSON/CSV 数据目录 (默认 experiment_system/data)')
    ap.add_argument('--only', default=None,
                    help='启动后载入的图 (如 heatmap / ablation / materials)')
    ap.add_argument('--list', action='store_true', help='列出可编辑图并退出')
    args = ap.parse_args(argv)

    if args.list:
        print('Demo (editable)  —— 空白演示图 (可自由编辑布局)')
        for key, info in fg.EXTRA_FIGURES.items():
            print(f'{key:12s} — {info["description"]}')
        return 0

    initial = None
    if args.only:
        for label in _available_sources(args.data_dir):
            if label == args.only or label.startswith(args.only):
                initial = label
                break
        if initial is None:
            print(f'[figure_studio] 未找到图源: {args.only}')
            return 2
    return launch(data_dir=args.data_dir, initial=initial)


if __name__ == '__main__':
    raise SystemExit(_main())
