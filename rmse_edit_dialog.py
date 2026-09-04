"""RMSE 对比图 (rmse_comparison.png) 的交互编辑对话框。

``attach_rmse_editor(fig, save_path)`` 在图窗 (TkAgg) 右侧弹出编辑对话框,
覆盖图中**全部可调元素**的实时预览调整:

1. 图例:  显隐 / 位置 (11 预设) / 字号 / 鼠标拖拽 (按住图例即可在子图内移动)
2. 箭头标注:  鼠标拖拽文本端=改变箭头方向 / 鼠标拖拽箭头端=改变指向 +
              对话框微调方向 (dx/dy) / 颜色 / 线宽 / 显隐 / 文本内容
3. 柱顶数值标签:  显隐 / 字号 / 格式 (小数位)
4. 文本编辑:  标题 / X轴标签 / Y轴标签 / 标注文本 实时编辑
5. 全局:  字号缩放 / 画布尺寸
6. 保存:  PNG + PDF + SVG 矢量副本

返回 True 表示对话框已接管 (调用方随后用 plt.show() 驱动);
返回 False 表示环境不支持 (无 Tk / 非 TkAgg), 调用方自行保存。
"""
import tkinter as tk
from tkinter import ttk, colorchooser

import matplotlib as mpl
import matplotlib.text as mtext
import matplotlib.pyplot as plt

LEGEND_LOCS = ["best", "upper right", "upper left", "lower left",
               "lower right", "right", "center left", "center right",
               "lower center", "upper center", "center"]
_LOC_CODES = {0: "best", 1: "upper right", 2: "upper left", 3: "lower left",
              4: "lower right", 5: "right", 6: "center left",
              7: "center right", 8: "lower center", 9: "upper center",
              10: "center"}


def _loc_str(leg):
    try:
        loc = leg._loc
    except Exception:
        return "best"
    if isinstance(loc, str):
        return loc if loc in LEGEND_LOCS else "best"
    return _LOC_CODES.get(int(loc), "best")


def _enable_drag(leg):
    if leg is None:
        return
    try:
        leg.set_draggable(True)
    except Exception:
        pass


class _DragManager:
    """统一管理图窗中所有可拖拽元素 (图例 / 标注) 的鼠标事件。

    拖拽标注的文本端 (xytext) → 改变箭头方向;
    拖拽标注的箭头端 (xy)     → 改变指向目标。
    用 button_press_event + contains 判断, 不依赖 picker, 更可靠。
    """

    def __init__(self, fig, ax, annotations):
        self.fig = fig
        self.ax = ax
        self.canvas = fig.canvas
        self.annotations = annotations
        self._dragging = None      # None | "text" | "arrow"
        self._active_ann = None    # 当前拖拽的 Annotation
        self._start = None         # 鼠标起始 data 坐标
        self._pos0 = None          # 拖拽端点起始 data 坐标

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        if not self.annotations:
            return
        ex, ey = event.x, event.y  # display 坐标
        edx, edy = event.xdata, event.ydata  # data 坐标

        # 按从后到前的顺序 (顶层优先) 检查每个标注
        for ann in reversed(self.annotations):
            if not ann.get_visible():
                continue
            # 1) 检查是否点中了文本区域
            try:
                bbox = ann.get_window_extent(renderer=self.canvas.renderer)
                if bbox.contains(ex, ey):
                    self._dragging = "text"
                    self._active_ann = ann
                    self._start = (edx, edy)
                    self._pos0 = ann.get_position()
                    return
            except Exception:
                pass
            # 2) 检查是否点中了箭头端点 (xy) 附近
            try:
                xy = ann.xy  # data 坐标
                # 把 xy 转 display 坐标判断距离
                xy_display = self.ax.transData.transform(xy)
                dist = ((ex - xy_display[0]) ** 2 +
                        (ey - xy_display[1]) ** 2) ** 0.5
                if dist < 15:  # 15 像素容差
                    self._dragging = "arrow"
                    self._active_ann = ann
                    self._start = (edx, edy)
                    self._pos0 = xy
                    return
            except Exception:
                pass

    def _on_motion(self, event):
        if not self._dragging or event.inaxes is not self.ax:
            return
        if self._active_ann is None or self._start is None:
            return
        dx = event.xdata - self._start[0]
        dy = event.ydata - self._start[1]
        try:
            if self._dragging == "text":
                self._active_ann.set_position(
                    (self._pos0[0] + dx, self._pos0[1] + dy))
            elif self._dragging == "arrow":
                self._active_ann.xy = (
                    self._pos0[0] + dx, self._pos0[1] + dy)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_release(self, _event):
        self._dragging = None
        self._active_ann = None
        self._start = None
        self._pos0 = None


class RmseEditDialog:
    """rmse_comparison 全元素编辑对话框 (附着于图窗右侧的 Toplevel)。"""

    def __init__(self, fig, root_window, save_path):
        self.fig = fig
        self.save_path = save_path
        self._ready = False
        self._ax = fig.axes[0] if fig.axes else None
        self._annotations = []
        self._bar_containers = []
        self._bar_labels = []
        self._font0 = {}
        self._snapshot_fonts()

        self.root = tk.Toplevel(root_window)
        self.root.title("RMSE 对比图编辑 — 全部元素")
        self.root.withdraw()
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._capture()
        self._build()
        self._ready = True
        self._place_right_of(root_window)

    def _capture(self):
        if self._ax is None:
            return
        leg = self._ax.get_legend()
        if leg is not None:
            _enable_drag(leg)
        # 采集所有标注 (含箭头和纯文本)
        for ann in self._ax.texts:
            self._annotations.append(ann)
        # 统一挂载拖拽管理器
        if self._annotations:
            self._drag_mgr = _DragManager(self.fig, self._ax, self._annotations)
        # 柱状图容器
        for c in self._ax.containers:
            if hasattr(c, "__len__") and len(c) > 0:
                self._bar_containers.append(c)

    def _snapshot_fonts(self):
        for t in self.fig.findobj(mtext.Text):
            try:
                self._font0[id(t)] = float(t.get_fontsize())
            except Exception:
                pass

    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        nb.add(self._tab_text(nb), text=" 文本编辑 ")
        nb.add(self._tab_legend(nb), text=" 图例 ")
        nb.add(self._tab_arrows(nb), text=" 箭头标注 ")
        nb.add(self._tab_bars(nb), text=" 柱顶数值 ")
        nb.add(self._tab_global(nb), text=" 全局 ")

        btns = ttk.Frame(self.root)
        btns.pack(fill="x", padx=8, pady=(4, 2))
        self.btn_ok = ttk.Button(btns, text="✓ 确认并保存",
                                 command=self._on_ok)
        self.btn_ok.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_cancel = ttk.Button(btns, text="✕ 取消",
                                     command=self._on_cancel)
        self.btn_cancel.pack(side="left", expand=True, fill="x", padx=2)
        self.status = tk.StringVar(
            value="图例/箭头可直接在图中鼠标拖拽 (拖文本端=改箭头方向); 确认保存 / 取消保持初始图")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  relief="sunken", padding=(6, 2)).pack(
                      side="bottom", fill="x", pady=(2, 4))

    def advance_buttons(self):
        return [self.btn_ok, self.btn_cancel]

    def take_close_handler(self):
        return self._on_cancel

    # ---------------------------------------------------------------- 文本
    def _tab_text(self, nb):
        f = ttk.Frame(nb, padding=8)
        r = 0
        # 标题
        title_obj = self._ax.get_title() if self._ax else ""
        ttk.Label(f, text="标题:").grid(row=r, column=0, sticky="w", pady=2)
        self._title_var = tk.StringVar(value=title_obj)
        e = ttk.Entry(f, textvariable=self._title_var)
        e.grid(row=r, column=1, columnspan=3, sticky="ew", padx=4)
        e.bind("<KeyRelease>", self._on_title)
        r += 1
        # X轴标签
        xlabel_obj = self._ax.get_xlabel() if self._ax else ""
        ttk.Label(f, text="X 轴标签:").grid(row=r, column=0, sticky="w")
        self._xlabel_var = tk.StringVar(value=xlabel_obj)
        e = ttk.Entry(f, textvariable=self._xlabel_var)
        e.grid(row=r, column=1, columnspan=3, sticky="ew", padx=4)
        e.bind("<KeyRelease>", self._on_xlabel)
        r += 1
        # Y轴标签
        ylabel_obj = self._ax.get_ylabel() if self._ax else ""
        ttk.Label(f, text="Y 轴标签:").grid(row=r, column=0, sticky="w")
        self._ylabel_var = tk.StringVar(value=ylabel_obj)
        e = ttk.Entry(f, textvariable=self._ylabel_var)
        e.grid(row=r, column=1, columnspan=3, sticky="ew", padx=4)
        e.bind("<KeyRelease>", self._on_ylabel)
        r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=4,
                              sticky="ew", pady=4)
        r += 1
        # 标注文本 (逐个编辑)
        ttk.Label(f, text="标注文本 (箭头):",
                  font=("", 9, "bold")).grid(row=r, column=0,
                                             columnspan=4, sticky="w")
        r += 1
        self._ann_text_vars = []
        for i, ann in enumerate(self._annotations):
            txt = ann.get_text()
            short = (txt[:25] + "...") if len(txt) > 25 else txt
            ttk.Label(f, text=f"标注 {i+1} ({short}):",
                      font=("", 8)).grid(row=r, column=0, sticky="w")
            var = tk.StringVar(value=txt)
            self._ann_text_vars.append((ann, var))
            e = ttk.Entry(f, textvariable=var)
            e.grid(row=r, column=1, columnspan=3, sticky="ew", padx=4)
            e.bind("<KeyRelease>", lambda _e, i=i: self._on_ann_text(i))
            r += 1
        f.columnconfigure(1, weight=1)
        return f

    # ---------------------------------------------------------------- 图例
    def _tab_legend(self, nb):
        f = ttk.Frame(nb, padding=8)
        leg = self._ax.get_legend() if self._ax else None
        self._leg_show = tk.BooleanVar(
            value=bool(leg is not None and leg.get_visible()))
        self._leg_loc = tk.StringVar(
            value=_loc_str(leg) if leg is not None else "best")
        self._leg_fs = tk.IntVar(
            value=(int(leg.get_texts()[0].get_fontsize())
                   if leg is not None and leg.get_texts() else 9))
        r = 0
        ttk.Label(f, text="图例:").grid(row=r, column=0, sticky="w", pady=2)
        ttk.Checkbutton(f, text="显示", variable=self._leg_show,
                        command=self._on_legend).grid(
                            row=r, column=1, sticky="w")
        r += 1
        ttk.Label(f, text="位置:").grid(row=r, column=0, sticky="e")
        cb = ttk.Combobox(f, textvariable=self._leg_loc, values=LEGEND_LOCS,
                          width=11, state="readonly")
        cb.grid(row=r, column=1, columnspan=2, sticky="w", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_legend())
        r += 1
        ttk.Label(f, text="图例字号:").grid(row=r, column=0, sticky="w")
        sp = ttk.Spinbox(f, from_=5, to=20, width=4,
                         textvariable=self._leg_fs,
                         command=self._on_legend)
        sp.grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(f, text="(也可直接在图中鼠标拖拽图例)",
                  foreground="#888").grid(row=r, column=0, columnspan=3,
                                          sticky="w")
        f.columnconfigure(1, weight=1)
        return f

    # ---------------------------------------------------------------- 箭头
    def _tab_arrows(self, nb):
        f = ttk.Frame(nb, padding=8)
        if not self._annotations:
            ttk.Label(f, text="未检测到箭头标注").grid(row=0, column=0)
            return f
        r = 0
        ttk.Label(f, text="提示: 在图中鼠标按住标注文本即可拖动;",
                  foreground="#888").grid(row=r, column=0, columnspan=4,
                                          sticky="w")
        r += 1
        ttk.Label(f, text="拖文本端=改变箭头方向, 拖箭头端=改变指向;",
                  foreground="#888").grid(row=r, column=0, columnspan=4,
                                          sticky="w")
        r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=4,
                              sticky="ew", pady=4)
        r += 1
        self._arrow_vars = []
        for i, ann in enumerate(self._annotations):
            txt = ann.get_text() or f"箭头 {i+1}"
            short = (txt[:20] + "...") if len(txt) > 20 else txt
            v = {
                "show": tk.BooleanVar(value=bool(ann.get_visible())),
                "color": tk.StringVar(value=_ann_color(ann)),
                "lw": tk.DoubleVar(value=_ann_lw(ann)),
                "dx": tk.DoubleVar(value=0.0),
                "dy": tk.DoubleVar(value=0.0),
            }
            self._arrow_vars.append((ann, v))
            ttk.Label(f, text=short, font=("", 8, "bold")).grid(
                row=r, column=0, columnspan=4, sticky="w")
            r += 1
            ttk.Checkbutton(f, text="显示", variable=v["show"],
                            command=lambda i=i: self._on_arrow(i)).grid(
                                row=r, column=0, sticky="w")
            ttk.Button(f, text="颜色", width=4,
                       command=lambda i=i: self._pick_arrow_color(i)).grid(
                           row=r, column=1, padx=2)
            ttk.Label(f, text="线宽:").grid(row=r, column=2, sticky="e")
            sp = ttk.Spinbox(f, from_=0.5, to=6, increment=0.5, width=4,
                             textvariable=v["lw"],
                             command=lambda i=i: self._on_arrow(i))
            sp.grid(row=r, column=3, padx=2)
            r += 1
            ttk.Label(f, text="文本端微调:").grid(row=r, column=0, sticky="w")
            ttk.Label(f, text="dx").grid(row=r, column=1, sticky="e")
            spx = ttk.Spinbox(f, from_=-5, to=5, increment=0.1, width=4,
                              textvariable=v["dx"],
                              command=lambda i=i: self._on_arrow_move(i))
            spx.grid(row=r, column=2, padx=2)
            ttk.Label(f, text="dy").grid(row=r, column=3, sticky="e")
            r += 1
            spy = ttk.Spinbox(f, from_=-5, to=5, increment=0.1, width=4,
                              textvariable=v["dy"],
                              command=lambda i=i: self._on_arrow_move(i))
            spy.grid(row=r, column=3, padx=2)
            r += 1
            ttk.Separator(f).grid(row=r, column=0, columnspan=4,
                                  sticky="ew", pady=2)
            r += 1
        f.columnconfigure(1, weight=1)
        return f

    # -------------------------------------------------------- 柱顶数值
    def _tab_bars(self, nb):
        f = ttk.Frame(nb, padding=8)
        self._bar_show = tk.BooleanVar(value=True)
        self._bar_fs = tk.IntVar(value=8)
        self._bar_decimals = tk.IntVar(value=3)
        r = 0
        ttk.Label(f, text="柱顶数值标签:").grid(row=r, column=0, sticky="w")
        ttk.Checkbutton(f, text="显示", variable=self._bar_show,
                        command=self._on_bars).grid(
                            row=r, column=1, sticky="w")
        r += 1
        ttk.Label(f, text="字号:").grid(row=r, column=0, sticky="w")
        sp = ttk.Spinbox(f, from_=5, to=16, width=4,
                         textvariable=self._bar_fs, command=self._on_bars)
        sp.grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(f, text="小数位:").grid(row=r, column=0, sticky="w")
        spd = ttk.Spinbox(f, from_=0, to=4, width=4,
                          textvariable=self._bar_decimals,
                          command=self._on_bars)
        spd.grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(f, text="(数值来自柱状图数据, 实时刷新)",
                  foreground="#888").grid(row=r, column=0, columnspan=2,
                                          sticky="w")
        f.columnconfigure(1, weight=1)
        return f

    # ------------------------------------------------------------ 全局
    def _tab_global(self, nb):
        f = ttk.Frame(nb, padding=8)
        r = 0
        ttk.Label(f, text="全局字号缩放:").grid(row=r, column=0, sticky="w")
        self._font_scale = tk.DoubleVar(value=1.0)
        ttk.Scale(f, from_=0.5, to=2.0, variable=self._font_scale,
                  command=self._on_font_scale, length=150).grid(
                      row=r, column=1, columnspan=2, sticky="ew", padx=4)
        self._fs_lbl = ttk.Label(f, text="1.00")
        self._fs_lbl.grid(row=r, column=3)
        r += 1
        w0, h0 = self.fig.get_size_inches()
        ttk.Label(f, text="画布宽 (in):").grid(row=r, column=0, sticky="w")
        self._fig_w = tk.DoubleVar(value=float(w0))
        spw = ttk.Spinbox(f, from_=4, to=26, increment=0.5, width=5,
                          textvariable=self._fig_w, command=self._on_size)
        spw.grid(row=r, column=1, sticky="w", padx=4)
        r += 1
        ttk.Label(f, text="画布高 (in):").grid(row=r, column=0, sticky="w")
        self._fig_h = tk.DoubleVar(value=float(h0))
        sph = ttk.Spinbox(f, from_=2, to=18, increment=0.5, width=5,
                          textvariable=self._fig_h, command=self._on_size)
        sph.grid(row=r, column=1, sticky="w", padx=4)
        f.columnconfigure(1, weight=1)
        return f

    # --------------------------------------------------- 应用逻辑
    def _redraw(self):
        self.fig.canvas.draw_idle()

    def _on_title(self, _e=None):
        if not self._ready or self._ax is None:
            return
        self._ax.set_title(self._title_var.get())
        self._redraw()

    def _on_xlabel(self, _e=None):
        if not self._ready or self._ax is None:
            return
        self._ax.set_xlabel(self._xlabel_var.get())
        self._redraw()

    def _on_ylabel(self, _e=None):
        if not self._ready or self._ax is None:
            return
        self._ax.set_ylabel(self._ylabel_var.get())
        self._redraw()

    def _on_ann_text(self, i):
        if not self._ready or i >= len(self._ann_text_vars):
            return
        ann, var = self._ann_text_vars[i]
        ann.set_text(var.get())
        self._redraw()

    def _on_legend(self, _e=None):
        if not self._ready or self._ax is None:
            return
        leg = self._ax.get_legend()
        if self._leg_show.get():
            handles, labels = self._ax.get_legend_handles_labels()
            if handles:
                leg = self._ax.legend(handles, labels,
                                      fontsize=int(self._leg_fs.get()),
                                      frameon=False,
                                      loc=self._leg_loc.get())
                _enable_drag(leg)
        elif leg is not None:
            leg.set_visible(False)
        self._redraw()

    def _on_arrow(self, i):
        if not self._ready or i >= len(self._arrow_vars):
            return
        ann, v = self._arrow_vars[i]
        ann.set_visible(bool(v["show"].get()))
        _set_ann_style(ann, color=v["color"].get(), lw=float(v["lw"].get()))
        self._redraw()

    def _on_arrow_move(self, i):
        if not self._ready or i >= len(self._arrow_vars):
            return
        ann, v = self._arrow_vars[i]
        base = getattr(ann, "_move_base", None)
        if base is None:
            base = ann.get_position()
            ann._move_base = base
        new_pos = (base[0] + float(v["dx"].get()),
                   base[1] + float(v["dy"].get()))
        ann.set_position(new_pos)
        self._redraw()

    def _pick_arrow_color(self, i):
        if i >= len(self._arrow_vars):
            return
        ann, v = self._arrow_vars[i]
        try:
            _rgb, hexv = colorchooser.askcolor(color=v["color"].get(),
                                               parent=self.root)
        except Exception:
            return
        if hexv:
            v["color"].set(str(hexv))
            self._on_arrow(i)

    def _on_bars(self, _e=None):
        if not self._ready or self._ax is None:
            return
        # 清除对话框生成的柱顶数值标签
        for lbl in self._bar_labels:
            try:
                lbl.remove()
            except Exception:
                pass
        self._bar_labels = []
        # 也清除初始渲染时 generate_rmse_comparison 生成的标签 (zorder=5 的 text)
        for t in list(self._ax.texts):
            if t not in self._annotations and not t.get_text().startswith(
                    ("Figure", "Gaussian", "Mixture", "Impulsive",
                     "Time")):
                # 只清除数值标签 (纯数字)
                txt = t.get_text().strip()
                try:
                    float(txt)
                    t.remove()
                except ValueError:
                    pass
        if not self._bar_show.get():
            self._redraw()
            return
        decimals = int(self._bar_decimals.get())
        fmt = f"%.{decimals}f"
        fs = int(self._bar_fs.get())
        for c in self._bar_containers:
            try:
                for rect in c:
                    h = rect.get_height()
                    x = rect.get_x() + rect.get_width() / 2
                    lbl = self._ax.text(x, h, fmt % h, ha="center",
                                        va="bottom", fontsize=fs,
                                        fontweight="bold", zorder=5)
                    lbl._rmse_bar_label = True
                    self._bar_labels.append(lbl)
            except Exception:
                pass
        self._redraw()

    def _on_font_scale(self, _v=None):
        if not self._ready:
            return
        s = float(self._font_scale.get())
        self._fs_lbl.config(text="%.2f" % s)
        for t in self.fig.findobj(mtext.Text):
            size0 = self._font0.get(id(t))
            if size0 is not None:
                try:
                    t.set_fontsize(size0 * s)
                except Exception:
                    pass
        self._redraw()

    def _on_size(self):
        if not self._ready:
            return
        try:
            self.fig.set_size_inches(float(self._fig_w.get()),
                                     float(self._fig_h.get()), forward=True)
        except Exception:
            pass
        self._redraw()

    # --------------------------------------------------- 确认/取消
    def _save(self):
        try:
            written = None
            try:
                from .journal_style import save_with_vector
                written = save_with_vector(self.fig, self.save_path, dpi=300,
                                           vector=True, svg=True)
            except ImportError:
                from experiment_system.journal_style import save_with_vector
                written = save_with_vector(self.fig, self.save_path, dpi=300,
                                           vector=True, svg=True)
            if not written:
                raise RuntimeError("save_with_vector 未写盘")
            self.status.set("已保存: %s" % self.save_path)
        except Exception as exc:
            try:
                self.fig.savefig(self.save_path, dpi=300,
                                 bbox_inches="tight", pad_inches=0.08)
                self.status.set("已保存 (回退): %s" % self.save_path)
            except Exception as exc2:
                self.status.set("保存失败: %s / %s" % (exc, exc2))

    def _on_ok(self):
        self._save()
        self._close()

    def _on_cancel(self):
        self._close()

    def _close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            self.fig.canvas.manager.window.destroy()
        except Exception:
            pass

    def _place_right_of(self, root_window, _retries=None):
        try:
            self.root.update_idletasks()
            sw = root_window.winfo_screenwidth()
            sh = root_window.winfo_screenheight()
            dw = max(self.root.winfo_reqwidth(), 380)
            dh = max(self.root.winfo_reqheight(), 420)
            fx = root_window.winfo_x()
            fy = root_window.winfo_y()
            fw = max(root_window.winfo_width(), 1)
            x = fx + fw + 8
            if x + dw > sw - 4:
                x = fx - dw - 8
            if x < 4 or x + dw > sw - 4:
                x = max(4, sw - dw - 4)
            y = max(4, min(fy, sh - dh - 40))
            self.root.geometry("+%d+%d" % (int(x), int(y)))
            if not self.root.winfo_ismapped():
                self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.lift()
        except Exception:
            return
        if _retries is None:
            _retries = 4
        if _retries > 0:
            try:
                root_window.after(350,
                                  lambda: self._place_right_of(root_window,
                                                               _retries - 1))
            except Exception:
                pass


def _ann_color(ann):
    try:
        ap = ann.arrowprops
        if ap and "color" in ap:
            import matplotlib.colors as mcolors
            return mcolors.to_hex(ap["color"])
    except Exception:
        pass
    return "#1B5E20"


def _ann_lw(ann):
    try:
        ap = ann.arrowprops
        if ap and "lw" in ap:
            return float(ap["lw"])
        if ap and "linewidth" in ap:
            return float(ap["linewidth"])
    except Exception:
        pass
    return 1.5


def _set_ann_style(ann, color=None, lw=None):
    try:
        if color is not None:
            ann.arrow_patch.set_color(color)
        if lw is not None:
            ann.arrow_patch.set_linewidth(lw)
    except Exception:
        pass


def attach_rmse_editor(fig, save_path):
    """在图窗右侧弹出 RMSE 对比图全元素编辑对话框。

    Returns
    -------
    bool
        True  = 对话框已创建 (调用方以 plt.show() 驱动交互);
        False = 环境不支持 (无 Tk / 非 TkAgg), 调用方自行保存。
    """
    import matplotlib
    if "tk" not in matplotlib.get_backend().lower():
        return False
    try:
        tk_root = fig.canvas.manager.window
    except Exception:
        return False
    try:
        dialog = RmseEditDialog(fig, tk_root, save_path)
    except Exception as exc:
        print(f"[rmse_edit_dialog] 对话框创建失败, 回退直接保存: {exc}")
        return False
    fig._rmse_editor = dialog
    return True
