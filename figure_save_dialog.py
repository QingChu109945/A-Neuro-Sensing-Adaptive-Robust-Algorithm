"""科研绘图"对话框式图片保存窗口" (迁移自 sample_size_sensitivity_plot.py)

``attach_save_dialog(fig, save_path)`` 会在 matplotlib 交互窗口 (TkAgg) 的
右侧弹出一个 Tkinter 对话框, 提供比窗口内嵌滑块更丰富的保存/细节调整能力:

- 输出格式: png / pdf / svg / eps / jpg / tiff (矢量格式可直接投稿)
- 分辨率:   DPI 自定义 (72-1200), 附 150/300/600 快捷档
- 画布细节: 透明背景 / bbox 紧裁剪 / pad_inches 留白 / 底色 (auto/white)
- 字体:     全局字号实时调节 (标题/轴标签/刻度/图例)
- 图例:     11 个标准位置实时切换 + 图例字号 + 显隐
- 保存:     默认文件名 (自动带扩展名切换) / Save / Save As… / 打开输出目录 / 复位

无 Tk 环境 (纯命令行/Agg 后端) 时 ``attach_save_dialog`` 返回 None,
调用方应回退到 ``plot_config.attach_interactive_controls`` 的窗口内控件。
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

SAVE_FORMATS = ["png", "pdf", "svg", "eps", "jpg", "tiff"]
DPI_PRESETS = [150, 300, 600]


def _apply_font_size(fig, size):
    """全局字号 -> 全部文本元素 (与 plot_config 保持一致的规则)。"""
    from .plot_config import _apply_font_size as _apply
    _apply(fig, size)


def _apply_legend_loc(fig, loc):
    from .plot_config import _apply_legend_loc as _apply
    _apply(fig, loc)


class FigureSaveDialog:
    """附着在 matplotlib 图窗右侧的保存对话框 (Toplevel)。"""

    def __init__(self, fig, root_window, save_path=None):
        self.fig = fig
        self.root = tk.Toplevel(root_window)
        self.root.title("图片保存设置 — " + (os.path.basename(save_path) if save_path else "figure"))
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        cfg = self._defaults(save_path)

        # ---- 1. 输出格式 + 分辨率 --------------------------------------
        box1 = ttk.LabelFrame(self.root, text=" 输出格式与分辨率 ", padding=6)
        box1.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(box1, text="格式:").grid(row=0, column=0, sticky="w")
        self.fmt_var = tk.StringVar(value=cfg["format"])
        self.fmt_cb = ttk.Combobox(box1, textvariable=self.fmt_var,
                                   values=SAVE_FORMATS, width=8, state="readonly")
        self.fmt_cb.grid(row=0, column=1, padx=(2, 12))
        self.fmt_cb.bind("<<ComboboxSelected>>", lambda e: self._sync_ext())

        ttk.Label(box1, text="DPI:").grid(row=0, column=2, sticky="w")
        self.dpi_var = tk.IntVar(value=cfg["dpi"])
        dpi_spin = ttk.Spinbox(box1, from_=72, to=1200, increment=1,
                               textvariable=self.dpi_var, width=6)
        dpi_spin.grid(row=0, column=3, padx=(2, 8))
        for i, d in enumerate(DPI_PRESETS):
            ttk.Button(box1, text=str(d), width=4,
                       command=lambda v=d: self.dpi_var.set(v)).grid(
                           row=0, column=4 + i, padx=2)

        # ---- 2. 画布细节 ------------------------------------------------
        box2 = ttk.LabelFrame(self.root, text=" 画布细节 ", padding=6)
        box2.pack(fill="x", padx=8, pady=2)
        self.trans_var = tk.BooleanVar(value=cfg["transparent"])
        ttk.Checkbutton(box2, text="透明背景", variable=self.trans_var).grid(
            row=0, column=0, sticky="w")
        self.bbox_var = tk.BooleanVar(value=cfg["bbox_tight"])
        ttk.Checkbutton(box2, text="bbox 紧裁剪", variable=self.bbox_var).grid(
            row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(box2, text="留白 (in):").grid(row=0, column=2, sticky="w", padx=(12, 2))
        self.pad_var = tk.DoubleVar(value=cfg["pad_inches"])
        ttk.Spinbox(box2, from_=0.0, to=1.0, increment=0.05, width=5,
                    textvariable=self.pad_var).grid(row=0, column=3)
        ttk.Label(box2, text="底色:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.face_var = tk.StringVar(value=cfg["facecolor"])
        ttk.Combobox(box2, textvariable=self.face_var, values=["auto", "white"],
                     width=8, state="readonly").grid(row=1, column=1, sticky="w",
                                                     pady=(4, 0))

        # ---- 3. 字体与图例 ----------------------------------------------
        box3 = ttk.LabelFrame(self.root, text=" 字体与图例 (实时预览) ", padding=6)
        box3.pack(fill="x", padx=8, pady=2)
        ttk.Label(box3, text="全局字号:").grid(row=0, column=0, sticky="w")
        self.font_var = tk.DoubleVar(value=cfg["font_size"])
        ttk.Scale(box3, from_=6, to=24, variable=self.font_var,
                  command=self._on_font).grid(row=0, column=1, sticky="ew",
                                              padx=4, columnspan=2)
        self.font_lbl = ttk.Label(box3, text=str(int(cfg["font_size"])))
        self.font_lbl.grid(row=0, column=3)

        ttk.Label(box3, text="图例位置:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        from .plot_config import LEGEND_LOCATIONS
        self.loc_var = tk.StringVar(value=cfg["legend_loc"])
        loc_cb = ttk.Combobox(box3, textvariable=self.loc_var,
                              values=LEGEND_LOCATIONS, width=12, state="readonly")
        loc_cb.grid(row=1, column=1, sticky="w", pady=(4, 0))
        loc_cb.bind("<<ComboboxSelected>>", lambda e: self._on_legend())

        ttk.Label(box3, text="图例字号:").grid(row=1, column=2, sticky="e", pady=(4, 0))
        self.legfs_var = tk.IntVar(value=cfg["legend_fontsize"])
        ttk.Spinbox(box3, from_=5, to=20, width=4, textvariable=self.legfs_var,
                    command=self._on_legend).grid(row=1, column=3, pady=(4, 0))

        self.leg_vis = tk.BooleanVar(value=cfg["legend_visible"])
        ttk.Checkbutton(box3, text="显示图例", variable=self.leg_vis,
                        command=self._on_legend).grid(row=2, column=0, columnspan=2,
                                                      sticky="w", pady=(4, 0))

        box3.columnconfigure(1, weight=1)

        # ---- 4. 保存 -----------------------------------------------------
        box4 = ttk.LabelFrame(self.root, text=" 保存 ", padding=6)
        box4.pack(fill="x", padx=8, pady=2)
        self.path_var = tk.StringVar(value=cfg["save_path"])
        ttk.Entry(box4, textvariable=self.path_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(box4, text="浏览…", command=self._browse).pack(side="left", padx=2)

        btns = ttk.Frame(box4)
        btns.pack(fill="x", pady=(6, 0))
        self.btn_save = ttk.Button(btns, text="💾 保存", command=self._save)
        self.btn_save.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_save_as = ttk.Button(btns, text="↺ 另存为…", command=self._save_as)
        self.btn_save_as.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_open_dir = ttk.Button(btns, text="📂 打开目录", command=self._open_dir)
        self.btn_open_dir.pack(side="left", padx=2)
        self.btn_reset = ttk.Button(btns, text="⟲ 复位", command=self._reset)
        self.btn_reset.pack(side="left", padx=2)

        # ---- 5. 状态栏 ----------------------------------------------------
        self.status = tk.StringVar(value="调整参数后点击保存; 关闭图窗即同步关闭本对话框")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  relief="sunken", padding=(6, 2)).pack(side="bottom", fill="x")

        self._remember_defaults(cfg)

        # 图窗关闭时同步关闭对话框 (走 matplotlib close_event, 不覆盖 Tk protocol)
        try:
            import matplotlib.pyplot as plt
            fig.canvas.mpl_connect(
                "close_event", lambda _e: self.root.destroy())
        except Exception:
            pass

        self._place_right_of(root_window)

    # ------------------------------------------------------------------
    def _defaults(self, save_path):
        from .plot_config import get_plot_config
        cfg = get_plot_config()
        path = save_path or os.path.join(os.getcwd(), "figure.png")
        base, ext = os.path.splitext(path)
        fmt = ext.lstrip(".").lower()
        if fmt not in SAVE_FORMATS:
            fmt, path = "png", base + ".png"
        leg = self.fig.axes[0].get_legend() if self.fig.axes else None
        leg_fs = leg.get_texts()[0].get_fontsize() if leg and leg.get_texts() else cfg.font_size - 1
        return {
            "format": fmt, "dpi": cfg.dpi,
            "transparent": False, "bbox_tight": True, "pad_inches": 0.1,
            "facecolor": "auto",
            "font_size": cfg.font_size,
            "legend_loc": cfg.legend_loc,
            "legend_fontsize": int(leg_fs),
            "legend_visible": True,
            "save_path": path,
        }

    def _remember_defaults(self, cfg):
        self._init = dict(cfg)

    def _place_right_of(self, root_window, _retries=None):
        try:
            x = root_window.winfo_x() + root_window.winfo_width() + 8
            y = root_window.winfo_y()
            self.root.update_idletasks()
            self.root.geometry(f"+{x}+{y}")
        except Exception:
            return
        # 图窗完成首次渲染后再校正几次 (渲染前 winfo_width 可能返回 1)
        if _retries is None:
            _retries = 4
        if _retries > 0:
            try:
                root_window.after(350, lambda: self._place_right_of(root_window, _retries - 1))
            except Exception:
                pass

    # ---------------- 实时预览回调 ------------------------------------
    def _on_font(self, _=None):
        size = float(self.font_var.get())
        self.font_lbl.config(text=str(int(size)))
        _apply_font_size(self.fig, size)

    def _on_legend(self):
        loc = self.loc_var.get()
        visible = self.leg_vis.get()
        fs = int(self.legfs_var.get())
        for ax in self.fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.set_visible(visible)
                for txt in leg.get_texts():
                    txt.set_fontsize(fs)
        if visible:
            _apply_legend_loc(self.fig, loc)
        self.fig.canvas.draw_idle()

    def _sync_ext(self):
        base, _ = os.path.splitext(self.path_var.get())
        self.path_var.set(base + "." + self.fmt_var.get())

    # ---------------- 保存 --------------------------------------------
    def _do_save(self, path):
        try:
            fmt = os.path.splitext(path)[1].lstrip(".").lower() or self.fmt_var.get()
            kwargs = dict(
                dpi=int(self.dpi_var.get()),
                format=fmt if fmt in SAVE_FORMATS else None,
                transparent=bool(self.trans_var.get()),
                bbox_inches="tight" if self.bbox_var.get() else None,
                pad_inches=float(self.pad_var.get()),
            )
            if self.face_var.get() == "white":
                kwargs["facecolor"] = "white"
            self.fig.savefig(path, **kwargs)
            self.status.set(f"已保存: {path}")
            messagebox.showinfo("保存成功", f"图片已保存:\n{path}")
        except Exception as exc:
            self.status.set(f"保存失败: {exc}")
            messagebox.showerror("保存失败", str(exc))

    def _save(self):
        self._do_save(self.path_var.get())

    def _save_as(self):
        p = filedialog.asksaveasfilename(
            defaultextension="." + self.fmt_var.get(),
            initialfile=os.path.basename(self.path_var.get()),
            initialdir=os.path.dirname(self.path_var.get()) or os.getcwd(),
            filetypes=[(f.upper(), "*." + f) for f in SAVE_FORMATS] + [("All", "*.*")])
        if p:
            self.path_var.set(p)
            self._do_save(p)

    def _browse(self):
        p = filedialog.asksaveasfilename(
            defaultextension="." + self.fmt_var.get(),
            initialfile=os.path.basename(self.path_var.get()))
        if p:
            self.path_var.set(p)
            self._sync_ext()

    def _open_dir(self):
        d = os.path.dirname(self.path_var.get()) or os.getcwd()
        try:
            os.startfile(d)  # Windows
        except AttributeError:
            import subprocess
            subprocess.call(["xdg-open", d])

    def _reset(self):
        v = self._init
        self.fmt_var.set(v["format"]); self.dpi_var.set(v["dpi"])
        self.trans_var.set(v["transparent"]); self.bbox_var.set(v["bbox_tight"])
        self.pad_var.set(v["pad_inches"]); self.face_var.set(v["facecolor"])
        self.font_var.set(v["font_size"]); self._on_font()
        self.loc_var.set(v["legend_loc"]); self.legfs_var.set(v["legend_fontsize"])
        self.leg_vis.set(v["legend_visible"]); self._on_legend()
        self.path_var.set(v["save_path"])
        self.status.set("已复位为默认参数")

    def _on_close(self):
        self.root.destroy()

    def advance_buttons(self):
        """推进类按钮 (sync 暂停时由 PauseController 禁用)。"""
        return [self.btn_save, self.btn_save_as]

    def take_close_handler(self):
        """把 WM_DELETE 回调移交给 PauseController 接管, 返回原回调。"""
        return self._on_close


def attach_save_dialog(fig, save_path=None):
    """在图窗右侧创建保存对话框 (同步构建, matplotlib 主循环由 plt.show 驱动)。

    Returns
    -------
    FigureSaveDialog or None
        TkAgg 可用时返回对话框实例; 不可用时返回 None,
        调用方应回退到 ``plot_config.attach_interactive_controls``。
    """
    import matplotlib
    if "tk" not in matplotlib.get_backend().lower():
        return None
    try:
        tk_root = fig.canvas.manager.window  # TkAgg 的 tkinter.Tk root
    except Exception:
        return None
    try:
        dialog = FigureSaveDialog(fig, tk_root, save_path=save_path)
    except Exception as exc:
        print(f"[figure_save_dialog] 对话框创建失败, 回退窗口内控件: {exc}")
        return None
    fig._save_dialog = dialog  # 持引用防 GC
    return dialog
