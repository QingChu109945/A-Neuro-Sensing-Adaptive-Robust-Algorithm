"""灵敏度曲线图 (sensitivity_curves) 的全元素交互编辑对话框。

``attach_sensitivity_editor(fig, save_path)`` 在图窗 (TkAgg) 右侧弹出编辑
对话框, 通过 Notebook 四个标签页覆盖图中**全部元素**的实时预览调整:

1. 全局:  总标题文本/字号、全局字号缩放 (保持字号层级等比缩放)、
          子图间距 wspace/hspace、画布宽/高、背景色
2. 面板 a/b/c (各一页): 面板标题 / X 轴标签 / Y 轴(左)标签 /
          Y2 轴(右, 仅双轴面板)标签 文本; RMSE 曲线与耗时曲线 (双轴面板)
          的颜色/线宽/线型/标记/标记尺寸; 选中超参数虚线
          (显隐/颜色/线型/线宽); 网格显隐; 图例 (显隐/位置/字号)

约定 (与调用方 ``fig_sensitivity`` 配合):
  * 弹窗前调用方已把初始渲染写入 save_path —— 点"取消"或直接关闭
    对话框即放弃更改, 磁盘上保持初始图片;
  * 点"确认并保存"则把当前预览状态写入 save_path (PNG+PDF+SVG 矢量副本),
    最终图片准确反映用户调整后的参数。

面板识别: 调用方 ``fig_sensitivity`` 为主轴设置 ``_sens_panel`` 自指标记、
双胞胎轴 (twinx) 设置指向主轴的同一标记 (纯 Python 属性, 不影响渲染/MD5)。

返回 True 表示对话框已接管 (调用方随后用 plt.show() 驱动);
返回 False 表示环境不支持 (无 Tk / 非 TkAgg), 调用方自行保存即可。
"""
import tkinter as tk
from tkinter import ttk, colorchooser

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.text as mtext

LEGEND_LOCS = ["best", "upper right", "upper left", "lower left",
               "lower right", "right", "center left", "center right",
               "lower center", "upper center", "center"]
_LOC_CODES = {0: "best", 1: "upper right", 2: "upper left", 3: "lower left",
              4: "lower right", 5: "right", 6: "center left",
              7: "center right", 8: "lower center", 9: "upper center",
              10: "center"}
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "none"]
FACES = ["auto", "white", "#F5F5F5"]


def _enable_drag(leg):
    """重建图例后重新开启鼠标拖拽 (手动调整图例位置避让遮挡曲线)。"""
    if leg is None:
        return
    try:
        leg.set_draggable(True)
    except Exception:
        pass


def _loc_str(leg):
    """把 Legend 的 loc 属性规整为标准位置关键字。"""
    try:
        loc = leg._loc
    except Exception:
        return "best"
    if isinstance(loc, str):
        return loc if loc in LEGEND_LOCS else "best"
    return _LOC_CODES.get(int(loc), "best")


class SensitivityEditDialog:
    """sensitivity_curves 全元素编辑对话框 (附着于图窗右侧的 Toplevel)。"""

    def __init__(self, fig, root_window, save_path):
        self.fig = fig
        self.save_path = save_path
        self._ready = False
        self._font0 = {}       # id(Text) -> 初始字号 (全局字号缩放用)
        self._panels = []      # 各面板元素引用 (ax/twin/曲线/图例/文本)
        self._grid_vars = {}
        self._leg_vars = {}
        self._capture()
        self._snapshot_fonts()

        self.root = tk.Toplevel(root_window)
        self.root.title("灵敏度曲线编辑 — 全部元素")
        # 先隐藏, 待完成屏内定位后再显示, 避免闪现到默认位置/屏幕外
        self.root.withdraw()
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._build()
        self._ready = True
        self._place_right_of(root_window)

    # ------------------------------------------------------------ 元素采集
    def _capture(self):
        """按 _sens_panel 标记采集主轴/双胞胎轴/曲线/图例/文本对象引用。"""
        mains = []
        for ax in self.fig.axes:
            p = getattr(ax, "_sens_panel", None)
            if p is not None and not any(p is m for m in mains):
                mains.append(p)
        for ax in mains:
            twin = next((a for a in self.fig.axes
                         if getattr(a, "_sens_panel", None) is ax and a is not ax),
                        None)
            marked = [l for l in ax.lines
                      if l.get_marker() not in ("", "None", None, "none")]
            plain = [l for l in ax.lines if l not in marked]
            leg = ax.get_legend()
            handles = ([h for h in getattr(leg, "legendHandles", [])
                        if h is not None] if leg is not None else [])
            labels = ([t.get_text() for t in leg.get_texts()]
                      if leg is not None else [])
            self._panels.append({
                "ax": ax, "twin": twin,
                "title": ax.title,
                "xlabel": ax.xaxis.label, "ylabel": ax.yaxis.label,
                "y2label": twin.yaxis.label if twin is not None else None,
                "curve": marked[0] if marked else None,
                "time": (twin.lines[0]
                         if twin is not None and twin.lines else None),
                "vline": plain[-1] if plain else None,
                "leg_handles": handles, "leg_labels": labels,
            })

    def _snapshot_fonts(self):
        for t in self.fig.findobj(mtext.Text):
            try:
                self._font0[id(t)] = float(t.get_fontsize())
            except Exception:
                pass

    # ------------------------------------------------------------------ UI
    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        nb.add(self._tab_global(nb), text=" 全局 ")
        for i in range(len(self._panels)):
            nb.add(self._tab_panel(nb, i), text=" 面板 %s " % "abc"[i])

        btns = ttk.Frame(self.root)
        btns.pack(fill="x", padx=8, pady=(4, 2))
        self.btn_ok = ttk.Button(btns, text="✓ 确认并保存", command=self._on_ok)
        self.btn_ok.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_cancel = ttk.Button(btns, text="✕ 取消", command=self._on_cancel)
        self.btn_cancel.pack(side="left", expand=True, fill="x", padx=2)
        self.status = tk.StringVar(
            value="图例可直接在图中鼠标拖拽避让遮挡; 调整参数实时预览; 确认保存 / 取消保持初始图")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  relief="sunken", padding=(6, 2)).pack(
                      side="bottom", fill="x", pady=(2, 4))

    def advance_buttons(self):
        """推进类按钮 (sync 暂停时由 PauseController 禁用)。"""
        return [self.btn_ok, self.btn_cancel]

    def take_close_handler(self):
        """把 WM_DELETE 回调移交给 PauseController 接管, 返回原回调。"""
        return self._on_cancel

    def _tab_global(self, nb):
        f = ttk.Frame(nb, padding=8)
        sup = self.fig._suptitle
        r = 0
        # 1. 总标题 -------------------------------------------------------
        if sup is not None:
            ttk.Label(f, text="总标题:").grid(row=r, column=0, sticky="w")
            self.sup_txt = tk.StringVar(value=sup.get_text())
            e = ttk.Entry(f, textvariable=self.sup_txt)
            e.grid(row=r, column=1, columnspan=3, sticky="ew", padx=4)
            e.bind("<KeyRelease>", self._on_suptitle)
            r += 1
            ttk.Label(f, text="总标题字号:").grid(row=r, column=0, sticky="w")
            self.sup_fs = tk.DoubleVar(value=float(sup.get_fontsize()))
            sp = ttk.Spinbox(f, from_=8, to=30, increment=0.5, width=5,
                             textvariable=self.sup_fs, command=self._on_suptitle)
            sp.grid(row=r, column=1, sticky="w", padx=4)
            sp.bind("<KeyRelease>", lambda _e: self._on_suptitle())
            r += 1
        # 2. 全局字号缩放 ---------------------------------------------------
        ttk.Label(f, text="全局字号缩放:").grid(row=r, column=0, sticky="w")
        self.font_scale = tk.DoubleVar(value=1.0)
        ttk.Scale(f, from_=0.5, to=2.0, variable=self.font_scale,
                  command=self._on_font_scale, length=150).grid(
                      row=r, column=1, columnspan=2, sticky="ew", padx=4)
        self.fs_lbl = ttk.Label(f, text="1.00")
        self.fs_lbl.grid(row=r, column=3)
        r += 1
        # 3. 子图间距 -------------------------------------------------------
        ttk.Label(f, text="水平间距 wspace:").grid(row=r, column=0, sticky="w")
        self.wsp = tk.DoubleVar(
            value=float(mpl.rcParams["figure.subplot.wspace"]))
        ttk.Scale(f, from_=0.0, to=0.6, variable=self.wsp,
                  command=self._on_spacing, length=150).grid(
                      row=r, column=1, columnspan=2, sticky="ew", padx=4)
        r += 1
        ttk.Label(f, text="垂直间距 hspace:").grid(row=r, column=0, sticky="w")
        self.hsp = tk.DoubleVar(
            value=float(mpl.rcParams["figure.subplot.hspace"]))
        ttk.Scale(f, from_=0.0, to=0.6, variable=self.hsp,
                  command=self._on_spacing, length=150).grid(
                      row=r, column=1, columnspan=2, sticky="ew", padx=4)
        r += 1
        # 4. 画布尺寸 -------------------------------------------------------
        w0, h0 = self.fig.get_size_inches()
        ttk.Label(f, text="画布宽 (in):").grid(row=r, column=0, sticky="w")
        self.fig_w = tk.DoubleVar(value=float(w0))
        spw = ttk.Spinbox(f, from_=4, to=26, increment=0.5, width=5,
                          textvariable=self.fig_w, command=self._on_size)
        spw.grid(row=r, column=1, sticky="w", padx=4)
        spw.bind("<KeyRelease>", lambda _e: self._on_size())
        r += 1
        ttk.Label(f, text="画布高 (in):").grid(row=r, column=0, sticky="w")
        self.fig_h = tk.DoubleVar(value=float(h0))
        sph = ttk.Spinbox(f, from_=2, to=18, increment=0.5, width=5,
                          textvariable=self.fig_h, command=self._on_size)
        sph.grid(row=r, column=1, sticky="w", padx=4)
        sph.bind("<KeyRelease>", lambda _e: self._on_size())
        r += 1
        # 5. 背景色 ---------------------------------------------------------
        ttk.Label(f, text="背景色:").grid(row=r, column=0, sticky="w")
        self.face = tk.StringVar(value="auto")
        cb = ttk.Combobox(f, textvariable=self.face, values=FACES,
                          width=9, state="readonly")
        cb.grid(row=r, column=1, sticky="w", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_face())
        f.columnconfigure(1, weight=1)
        return f

    def _tab_panel(self, nb, i):
        p = self._panels[i]
        f = ttk.Frame(nb, padding=8)
        r = 0
        # 1. 文本元素 -------------------------------------------------------
        r = self._text_row(f, r, "面板标题:", p["title"])
        r = self._text_row(f, r, "X 轴标签:", p["xlabel"])
        r = self._text_row(f, r, "Y 轴标签:", p["ylabel"])
        if p["y2label"] is not None:
            r = self._text_row(f, r, "Y2 轴标签:", p["y2label"])
        ttk.Separator(f).grid(row=r, column=0, columnspan=6,
                              sticky="ew", pady=4)
        r += 1
        # 2. 数据曲线 -------------------------------------------------------
        if p["curve"] is not None:
            r = self._line_row(f, r, "RMSE 曲线:", p["curve"])
        if p["time"] is not None:
            r = self._line_row(f, r, "耗时曲线:", p["time"])
        # 3. 选定值虚线 -----------------------------------------------------
        if p["vline"] is not None:
            r = self._vline_row(f, r, "选定值虚线:", p["vline"])
        ttk.Separator(f).grid(row=r, column=0, columnspan=6,
                              sticky="ew", pady=4)
        r += 1
        # 4. 网格 -----------------------------------------------------------
        grid_vis = any(gl.get_visible() for gl in p["ax"].get_xgridlines())
        self._grid_vars[i] = tk.BooleanVar(value=bool(grid_vis))
        ttk.Checkbutton(f, text="显示网格", variable=self._grid_vars[i],
                        command=lambda: self._on_grid(i)).grid(
                            row=r, column=0, columnspan=2, sticky="w")
        r += 1
        # 5. 图例 -----------------------------------------------------------
        r = self._legend_rows(f, r, i, p)
        f.columnconfigure(1, weight=1)
        return f

    # ------------------------------------------------------------ 行构造器
    def _text_row(self, f, r, label, text_obj):
        ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=1)
        var = tk.StringVar(value=text_obj.get_text())
        e = ttk.Entry(f, textvariable=var)
        e.grid(row=r, column=1, columnspan=5, sticky="ew", padx=4)

        def apply(_e=None):
            if not self._ready:
                return
            text_obj.set_text(var.get())
            self._redraw()

        e.bind("<KeyRelease>", apply)
        return r + 1

    def _line_row(self, f, r, label, line):
        """数据曲线编辑行: 颜色/线宽/线型/标记/标记尺寸。"""
        ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
        col = tk.StringVar(value=mcolors.to_hex(line.get_color()))
        sw = tk.Label(f, bg=col.get(), width=2, relief="groove")
        sw.grid(row=r, column=1, padx=(4, 0))

        def apply(_e=None):
            if not self._ready:
                return
            try:
                line.set_color(col.get())
                line.set_linewidth(float(lw.get()))
                line.set_linestyle(ls.get())
                m = mk.get()
                line.set_marker(None if m == "none" else m)
                line.set_markersize(float(ms.get()))
            except Exception:
                pass
            try:
                sw.config(bg=col.get())
            except Exception:
                pass
            self._redraw()

        ttk.Button(f, text="颜色", width=4,
                   command=lambda: self._pick_color(col, apply)).grid(
                       row=r, column=2, padx=2)
        lw = tk.DoubleVar(value=float(line.get_linewidth()))
        sp_lw = ttk.Spinbox(f, from_=0.5, to=6, increment=0.5, width=4,
                            textvariable=lw, command=apply)
        sp_lw.grid(row=r, column=3, padx=2)
        sp_lw.bind("<KeyRelease>", apply)
        ls = tk.StringVar(value=line.get_linestyle())
        cb_ls = ttk.Combobox(f, textvariable=ls, values=LINESTYLES,
                             width=3, state="readonly")
        cb_ls.grid(row=r, column=4, padx=2)
        cb_ls.bind("<<ComboboxSelected>>", apply)
        mk = tk.StringVar(value=str(line.get_marker() or "none"))
        cb_mk = ttk.Combobox(f, textvariable=mk, values=MARKERS,
                             width=3, state="readonly")
        cb_mk.grid(row=r, column=5, padx=2)
        cb_mk.bind("<<ComboboxSelected>>", apply)
        ms = tk.DoubleVar(value=float(line.get_markersize()))
        sp_ms = ttk.Spinbox(f, from_=1, to=14, increment=0.5, width=4,
                            textvariable=ms, command=apply)
        sp_ms.grid(row=r, column=6, padx=2)
        sp_ms.bind("<KeyRelease>", apply)
        return r + 1

    def _vline_row(self, f, r, label, line):
        """选定超参数虚线编辑行: 显隐/颜色/线型/线宽。"""
        ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2)
        show = tk.BooleanVar(value=bool(line.get_visible()))
        col = tk.StringVar(value=mcolors.to_hex(line.get_color()))
        sw = tk.Label(f, bg=col.get(), width=2, relief="groove")
        sw.grid(row=r, column=1, padx=(4, 0))

        def apply(_e=None):
            if not self._ready:
                return
            try:
                line.set_visible(bool(show.get()))
                line.set_color(col.get())
                line.set_linewidth(float(lw.get()))
                line.set_linestyle(ls.get())
            except Exception:
                pass
            try:
                sw.config(bg=col.get())
            except Exception:
                pass
            self._redraw()

        ttk.Checkbutton(f, text="显示", variable=show, command=apply).grid(
            row=r, column=2, padx=2)
        ttk.Button(f, text="颜色", width=4,
                   command=lambda: self._pick_color(col, apply)).grid(
                       row=r, column=3, padx=2)
        lw = tk.DoubleVar(value=float(line.get_linewidth()))
        sp_lw = ttk.Spinbox(f, from_=0.5, to=6, increment=0.5, width=4,
                            textvariable=lw, command=apply)
        sp_lw.grid(row=r, column=4, padx=2)
        sp_lw.bind("<KeyRelease>", apply)
        ls = tk.StringVar(value=line.get_linestyle())
        cb_ls = ttk.Combobox(f, textvariable=ls, values=LINESTYLES,
                             width=3, state="readonly")
        cb_ls.grid(row=r, column=5, padx=2)
        cb_ls.bind("<<ComboboxSelected>>", apply)
        return r + 1

    def _legend_rows(self, f, r, i, p):
        ax = p["ax"]
        leg = ax.get_legend()
        v = {
            "show": tk.BooleanVar(value=bool(leg is not None
                                             and leg.get_visible())),
            "loc": tk.StringVar(
                value=_loc_str(leg) if leg is not None else "best"),
            "fs": tk.IntVar(value=(int(leg.get_texts()[0].get_fontsize())
                                   if leg is not None and leg.get_texts()
                                   else 9)),
        }
        self._leg_vars[i] = v
        ttk.Label(f, text="图例:").grid(row=r, column=0, sticky="w", pady=2)
        ttk.Checkbutton(f, text="显示", variable=v["show"],
                        command=lambda: self._on_legend(i)).grid(
                            row=r, column=1, sticky="w")
        ttk.Label(f, text="位置:").grid(row=r, column=2, sticky="e")
        cb = ttk.Combobox(f, textvariable=v["loc"], values=LEGEND_LOCS,
                          width=11, state="readonly")
        cb.grid(row=r, column=3, columnspan=2, sticky="w", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_legend(i))
        r += 1
        ttk.Label(f, text="图例字号:").grid(row=r, column=0, sticky="w")
        sp = ttk.Spinbox(f, from_=5, to=20, width=4, textvariable=v["fs"],
                         command=lambda: self._on_legend(i))
        sp.grid(row=r, column=1, sticky="w")
        sp.bind("<KeyRelease>", lambda _e: self._on_legend(i))
        return r + 1

    def _pick_color(self, var, apply):
        try:
            _rgb, hexv = colorchooser.askcolor(color=var.get(),
                                               parent=self.root)
        except Exception:
            return
        if hexv:
            var.set(str(hexv))
            apply()

    # ------------------------------------------------------------ 应用逻辑
    def _redraw(self):
        self.fig.canvas.draw_idle()

    def _on_suptitle(self, _e=None):
        if not self._ready or self.fig._suptitle is None:
            return
        self.fig._suptitle.set_text(self.sup_txt.get())
        try:
            self.fig._suptitle.set_fontsize(float(self.sup_fs.get()))
        except Exception:
            pass
        self._redraw()

    def _on_font_scale(self, _v=None):
        if not self._ready:
            return
        s = float(self.font_scale.get())
        self.fs_lbl.config(text="%.2f" % s)
        for t in self.fig.findobj(mtext.Text):
            size0 = self._font0.get(id(t))
            if size0 is not None:
                try:
                    t.set_fontsize(size0 * s)
                except Exception:
                    pass
        self._redraw()

    def _on_spacing(self, _v=None):
        if not self._ready:
            return
        self.fig.subplots_adjust(wspace=float(self.wsp.get()),
                                 hspace=float(self.hsp.get()))
        self._redraw()

    def _on_size(self):
        if not self._ready:
            return
        try:
            self.fig.set_size_inches(float(self.fig_w.get()),
                                     float(self.fig_h.get()), forward=True)
        except Exception:
            pass
        self._redraw()

    def _on_face(self):
        if not self._ready:
            return
        v = self.face.get()
        self.fig.set_facecolor("white" if v == "auto" else v)
        self._redraw()

    def _on_grid(self, i):
        if not self._ready:
            return
        self._panels[i]["ax"].grid(bool(self._grid_vars[i].get()))
        self._redraw()

    def _on_legend(self, i):
        if not self._ready:
            return
        p = self._panels[i]
        v = self._leg_vars[i]
        ax = p["ax"]
        if v["show"].get() and p["leg_handles"]:
            leg = ax.legend(p["leg_handles"], p["leg_labels"],
                            fontsize=int(v["fs"].get()), frameon=False,
                            loc=v["loc"].get())
            _enable_drag(leg)
        elif ax.get_legend() is not None:
            ax.get_legend().set_visible(False)
        self._redraw()

    # ------------------------------------------------------------ 确认/取消
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
        # 弹窗前初始渲染已在 save_path, 直接关窗即放弃更改
        self._close()

    def _close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        # 同步关闭图窗, 结束调用方的 plt.show() 阻塞
        try:
            self.fig.canvas.manager.window.destroy()
        except Exception:
            pass

    def _place_right_of(self, root_window, _retries=None):
        """定位到图窗右侧; 屏幕放不下时退到左侧, 再不行 clamp 进屏。"""
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


def attach_sensitivity_editor(fig, save_path):
    """在图窗右侧弹出灵敏度曲线全元素编辑对话框。

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
        tk_root = fig.canvas.manager.window  # TkAgg 的 tkinter.Tk root
    except Exception:
        return False
    try:
        dialog = SensitivityEditDialog(fig, tk_root, save_path)
    except Exception as exc:
        print(f"[sensitivity_edit_dialog] 对话框创建失败, 回退直接保存: {exc}")
        return False
    fig._sensitivity_editor = dialog  # 持引用防 GC
    return True
