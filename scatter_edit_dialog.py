"""三面板发射率反演散点图的交互编辑对话框。

``attach_scatter_editor(fig, save_path)`` 在图窗 (TkAgg) 右侧弹出编辑对话框,
支持以下参数的实时预览调整 (所有变更即时重绘画布):

1. 子图间距:      水平间距 wspace / 垂直间距 hspace 滑块
2. "ideal" 图例:  位置 (loc + X/Y 偏移)、字号 (大小)、
                  边框显隐 / 圆角 / 边框颜色 / 底纹透明度 (样式)
3. "prediction" 图例: 同上, 并带"显示"开关 (默认关闭; 勾选时才为
                  三个面板按需创建散点图例, 不影响默认渲染)

约定 (与调用方 ``fig_emissivity_scatter`` 配合):
  * 弹窗前调用方已把初始渲染写入 save_path —— 点"取消"或直接关闭
    对话框即放弃更改, 磁盘上保持初始图片;
  * 点"确认并保存"则把当前预览状态写入 save_path, 最终图片准确
    反映用户调整后的参数。

返回 True 表示对话框已接管 (调用方随后用 plt.show() 驱动);
返回 False 表示环境不支持 (无 Tk / 非 TkAgg), 调用方自行保存即可。
"""
import tkinter as tk
from tkinter import ttk

LEGEND_LOCS = ["best", "upper right", "upper left", "lower left",
               "lower right", "right", "center left", "center right",
               "lower center", "upper center", "center"]
EDGE_COLORS = ["default", "black", "grey", "#C62828", "#00897B"]

IDEAL_TAG = "ideal"        # 图例文本含此关键字 -> ideal 图例
PRED_TAG = "prediction"    # 同理 -> prediction 图例


def _find_legend(ax, tag):
    """在单个子图中按图例文本查找图例对象 (ideal / prediction)。"""
    candidates = list(getattr(ax, "artists", []))
    if ax.get_legend() is not None:
        candidates.append(ax.get_legend())
    for leg in candidates:
        try:
            texts = [t.get_text() for t in leg.get_texts()]
        except Exception:
            continue
        if any(tag in t for t in texts):
            return leg
    return None


class ScatterEditDialog:
    def __init__(self, fig, root_window, save_path):
        import matplotlib as mpl
        self.fig = fig
        self.save_path = save_path
        self.wspace0 = float(mpl.rcParams["figure.subplot.wspace"])
        self.hspace0 = float(mpl.rcParams["figure.subplot.hspace"])
        self.created = []  # 对话框期间创建的 prediction 图例 (便于取消时移除)
        # 只编辑三个散点面板 (fig.axes 还包含 colorbar axes, 其 QuadMesh
        # 不支持 legend, 且无图例可调)
        self._panels = [ax for ax in fig.axes
                        if _find_legend(ax, IDEAL_TAG) is not None]

        self.root = tk.Toplevel(root_window)
        self.root.title("散点图编辑 — 子图间距 / 图例")
        # 先隐藏, 待完成屏内定位后再显示, 避免闪现到默认位置/屏幕外
        self.root.withdraw()
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._ready = False  # _build 期间控件初始化会触发回调, 先屏蔽
        self._build()
        self._ready = True
        self._place_right_of(root_window)

    # ------------------------------------------------------------------ UI
    def _build(self):
        # 1. 子图间距 ------------------------------------------------------
        box1 = ttk.LabelFrame(self.root, text=" 子图间距 ", padding=6)
        box1.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(box1, text="水平间距 wspace:").grid(row=0, column=0, sticky="w")
        self.wsp = tk.DoubleVar(value=self.wspace0)
        self.wsp_s = ttk.Scale(box1, from_=0.0, to=0.6, variable=self.wsp,
                               command=self._on_spacing, length=150)
        self.wsp_s.grid(row=0, column=1, padx=4)
        self.wsp_l = ttk.Label(box1, text="%.2f" % self.wspace0, width=5)
        self.wsp_l.grid(row=0, column=2)

        ttk.Label(box1, text="垂直间距 hspace:").grid(row=1, column=0, sticky="w")
        self.hsp = tk.DoubleVar(value=self.hspace0)
        self.hsp_s = ttk.Scale(box1, from_=0.0, to=0.6, variable=self.hsp,
                               command=self._on_spacing, length=150)
        self.hsp_s.grid(row=1, column=1, padx=4)
        self.hsp_l = ttk.Label(box1, text="%.2f" % self.hspace0, width=5)
        self.hsp_l.grid(row=1, column=2)

        # 2. ideal 图例 ----------------------------------------------------
        self.box2 = self._legend_group(" ideal 图例 (全部面板) ", with_show=False)
        self.box2.pack(fill="x", padx=8, pady=2)

        # 3. prediction 图例 ------------------------------------------------
        self.box3 = self._legend_group(" prediction 图例 (全部面板) ", with_show=True)
        self.box3.pack(fill="x", padx=8, pady=2)
        self._set_group_state(self.box3, disabled=True)

        # 4. 按钮 + 状态栏 ---------------------------------------------------
        btns = ttk.Frame(self.root)
        btns.pack(fill="x", padx=8, pady=(4, 8))
        self.btn_ok = ttk.Button(btns, text="✓ 确认并保存", command=self._on_ok)
        self.btn_ok.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_cancel = ttk.Button(btns, text="✕ 取消", command=self._on_cancel)
        self.btn_cancel.pack(side="left", expand=True, fill="x", padx=2)
        self.status = tk.StringVar(value="调整参数实时预览; 确认保存 / 取消保持初始图")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  relief="sunken", padding=(6, 2)).pack(side="bottom", fill="x")

    def advance_buttons(self):
        """推进类按钮 (sync 暂停时由 PauseController 禁用)。"""
        return [self.btn_ok, self.btn_cancel]

    def take_close_handler(self):
        """把 WM_DELETE 回调移交给 PauseController 接管, 返回原回调。"""
        return self._on_cancel

    def _legend_group(self, title, with_show):
        box = ttk.LabelFrame(self.root, text=title, padding=6)
        r = 0
        if with_show:
            self.pred_show = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(box, text="显示 prediction 图例",
                                 variable=self.pred_show,
                                 command=self._on_pred_toggle)
            cb.grid(row=r, column=0, columnspan=6, sticky="w")
            r += 1
        ttk.Label(box, text="位置:").grid(row=r, column=0, sticky="w")
        self.loc = tk.StringVar(value="upper left")
        self.loc_cb = ttk.Combobox(box, textvariable=self.loc, values=LEGEND_LOCS,
                                   width=11, state="readonly")
        self.loc_cb.grid(row=r, column=1, padx=(2, 8))
        self.loc_cb.bind("<<ComboboxSelected>>", lambda e: self._on_legend_apply())
        ttk.Label(box, text="X偏移:").grid(row=r, column=2, sticky="w")
        self.ax0 = tk.DoubleVar(value=0.0)
        self.ax_s = ttk.Scale(box, from_=-0.5, to=1.5, variable=self.ax0,
                              command=self._on_anchor, length=110)
        self.ax_s.grid(row=r, column=3, padx=2)
        ttk.Label(box, text="Y偏移:").grid(row=r, column=4, sticky="w", padx=(8, 0))
        self.ay0 = tk.DoubleVar(value=1.0)
        self.ay_s = ttk.Scale(box, from_=-0.5, to=1.5, variable=self.ay0,
                              command=self._on_anchor, length=110)
        self.ay_s.grid(row=r, column=5, padx=2)
        r += 1
        ttk.Label(box, text="字号:").grid(row=r, column=0, sticky="w")
        self.fs = tk.IntVar(value=9)
        self.fs_sp = ttk.Spinbox(box, from_=5, to=20, width=4, textvariable=self.fs,
                                 command=self._on_legend_apply)
        self.fs_sp.grid(row=r, column=1, sticky="w", padx=2)
        self.frameon = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="边框", variable=self.frameon,
                        command=self._on_legend_apply).grid(row=r, column=2, sticky="w")
        self.fancy = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="圆角", variable=self.fancy,
                        command=self._on_legend_apply).grid(row=r, column=3, sticky="w")
        ttk.Label(box, text="边框色:").grid(row=r, column=4, sticky="w", padx=(8, 0))
        self.edge = tk.StringVar(value="default")
        self.edge_cb = ttk.Combobox(box, textvariable=self.edge, values=EDGE_COLORS,
                                    width=8, state="readonly")
        self.edge_cb.grid(row=r, column=5, padx=2)
        self.edge_cb.bind("<<ComboboxSelected>>", lambda e: self._on_legend_apply())
        r += 1
        ttk.Label(box, text="底纹透明度:").grid(row=r, column=0, sticky="w")
        self.fa = tk.DoubleVar(value=0.8)
        self.fa_s = ttk.Scale(box, from_=0.0, to=1.0, variable=self.fa,
                              command=self._on_legend_apply, length=150)
        self.fa_s.grid(row=r, column=1, columnspan=3, sticky="ew", padx=2)
        return box

    def _set_group_state(self, box, disabled):
        state = "disabled" if disabled else "normal"
        for w in box.winfo_children():
            if isinstance(w, ttk.Checkbutton) and w.cget("text").startswith("显示"):
                continue  # "显示" 开关本身保持可用
            try:
                w.state([state] if state == "disabled" else ["!disabled"])
            except Exception:
                pass

    def _place_right_of(self, root_window, _retries=None):
        """定位到图窗右侧; 屏幕放不下时退到左侧, 再不行 clamp 进屏。

        图窗完成首次渲染前 winfo_width 返回 1, 因此用 after 重复校正
        (上限 4 次, 与 figure_save_dialog 一致)。
        """
        try:
            self.root.update_idletasks()
            sw = root_window.winfo_screenwidth()
            sh = root_window.winfo_screenheight()
            dw = max(self.root.winfo_reqwidth(), 320)
            dh = max(self.root.winfo_reqheight(), 380)
            fx = root_window.winfo_x()
            fy = root_window.winfo_y()
            fw = max(root_window.winfo_width(), 1)
            # 首选图窗右侧; 放不下则左侧; 仍放不下则贴屏右缘
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
                root_window.after(350, lambda: self._place_right_of(root_window, _retries - 1))
            except Exception:
                pass

    # ---------------------------------------------------------- 应用逻辑
    def _redraw(self):
        self.fig.canvas.draw_idle()

    def _on_spacing(self, _=None):
        if not self._ready:
            return
        self.fig.subplots_adjust(wspace=float(self.wsp.get()),
                                 hspace=float(self.hsp.get()))
        self.wsp_l.config(text="%.2f" % self.wsp.get())
        self.hsp_l.config(text="%.2f" % self.hsp.get())
        self._redraw()

    def _on_anchor(self, _=None):
        """X/Y 偏移滑块: 轻量更新 bbox_to_anchor, 不重建图例。"""
        if not self._ready:
            return
        anchor = (float(self.ax0.get()), float(self.ay0.get()))
        for ax in self._panels:
            leg = _find_legend(ax, IDEAL_TAG)
            if leg is not None:
                leg.set_bbox_to_anchor(anchor)
            leg = _find_legend(ax, PRED_TAG)
            if leg is not None:
                leg.set_bbox_to_anchor(anchor)
        self._redraw()

    def _legend_kwargs(self):
        kw = dict(loc=self.loc.get(),
                  bbox_to_anchor=(float(self.ax0.get()), float(self.ay0.get())),
                  fontsize=int(self.fs.get()),
                  frameon=bool(self.frameon.get()),
                  framealpha=float(self.fa.get()))
        if self.fancy.get():
            kw["fancybox"] = True
        if self.edge.get() != "default":
            kw["edgecolor"] = self.edge.get()
        return kw

    @staticmethod
    def _legend_content(leg):
        """提取图例的 (handles, labels) — 兼容 matplotlib 3.3 (无
        get_handles_labels 方法, 用 legendHandles + texts)。"""
        handles = [h for h in getattr(leg, "legendHandles", []) if h is not None]
        labels = [t.get_text() for t in leg.get_texts()]
        return handles, labels[:len(handles)]

    def _on_legend_apply(self):
        """loc / 字号 / 样式变更: 重建全部面板的既有图例。"""
        if not self._ready:
            return
        kw = self._legend_kwargs()
        for ax in self._panels:
            for tag in (IDEAL_TAG, PRED_TAG):
                leg = _find_legend(ax, tag)
                if leg is None:
                    continue
                if tag == PRED_TAG and not self.pred_show.get():
                    continue
                handles, labels = self._legend_content(leg)
                if leg is ax.get_legend():
                    ax.legend(handles, labels, **kw)
                else:
                    ax.artists.remove(leg)
                    ax.add_artist(ax.legend(handles, labels, **kw))
        self._redraw()

    def _on_pred_toggle(self):
        if not self._ready:
            return
        if self.pred_show.get():
            self._set_group_state(self.box3, disabled=False)
            kw = self._legend_kwargs()
            for ax in self._panels:
                if _find_legend(ax, PRED_TAG) is not None:
                    continue
                if not ax.collections:
                    continue
                leg = ax.legend([ax.collections[0]], ["prediction"], **kw)
                self.created.append((ax, leg))
        else:
            for ax, leg in list(self.created):
                if leg is ax.get_legend():
                    ax.legend_ = None
                elif leg in ax.artists:
                    ax.artists.remove(leg)
            self.created.clear()
            self._set_group_state(self.box3, disabled=True)
        self._redraw()

    # ---------------------------------------------------------- 确认/取消
    def _save(self):
        try:
            self.fig.savefig(self.save_path, dpi=300,
                             bbox_inches="tight", pad_inches=0.08)
            self.status.set("已保存: %s" % self.save_path)
        except Exception as exc:
            self.status.set("保存失败: %s" % exc)

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


def attach_scatter_editor(fig, save_path):
    """在图窗右侧弹出散点图编辑对话框。

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
        dialog = ScatterEditDialog(fig, tk_root, save_path)
    except Exception as exc:
        print(f"[scatter_edit_dialog] 对话框创建失败, 回退直接保存: {exc}")
        return False
    fig._scatter_editor = dialog  # 持引用防 GC
    return True
