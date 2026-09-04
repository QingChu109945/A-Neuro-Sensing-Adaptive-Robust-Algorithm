"""可视化图片窗口的暂停/继续控制。

``PauseController`` 给一个 matplotlib 图窗 (可选附带对话框) 附加统一的
暂停/继续控制, 满足:

1. **暂停机制**  激活后窗口保持当前显示状态并定格 (交互控件被冻结,
   画面不会被改动), 后台流程按所选模式处理;
2. **控制接口**  图窗底部控制条按钮 + 快捷键 ``P`` / ``空格``
   (文本输入框获得焦点时按键不会误触发);
3. **数据完整性**  暂停只冻结 UI 交互与流程推进, 不触碰任何已写盘
   的数据/图片文件; 同步模式下保存/确认按钮被禁用, 不会产生半写文件;
4. **视觉指示**  状态标签变色 (绿 "● 运行中" / 橙 "⏸ 已暂停")、
   窗口标题追加 "⏸"、图内右下角 "⏸ PAUSED" 角标;
5. **恢复功能**  再次点击按钮或按快捷键即从暂停状态恢复正常更新。

后台处理两种模式 (``plot_config.pause_mode``, 由 run_experiment
``--pause-mode`` 选择):

* ``sync``  (默认) 同步暂停 —— 暂停期间推进类控件 (保存/确认/取消)
  被禁用、窗口关闭被拦截, 流水线在当前图处挂起, 直到继续;
* ``async`` 后台继续 —— 暂停时自动保存当前图并放行流水线继续生成
  后续图片, 本窗口转为定格展示窗 (通过事件泵保持暂停条/关闭可响应,
  观察完成后可手动关闭)。
"""
import tkinter as tk

from .plot_config import get_plot_config

ACCENT_RUN = "#2E7D32"     # 绿 — 运行中
ACCENT_PAUSE = "#E65100"   # 橙 — 已暂停
BTN_PAUSE_BG = "#FFF59D"   # 暂停按钮底色
TXT_RUN = "● 运行中 — 后台正常推进"
TXT_PAUSED_SYNC = "⏸ 已暂停 — 后台已挂起 (按 P/空格 或点击按钮继续)"
TXT_PAUSED_ASYNC = "⏸ 已暂停(后台继续) — 本窗口定格展示, 可继续观察后手动关闭"

_TEXT_LIKE = ("Entry", "Spinbox", "Text", "TCombobox", "TSpinbox")


def pump_pending_events():
    """处理所有仍打开的 matplotlib 图窗的待处理 Tk 事件。

    async (后台继续) 模式下, 流水线放行后继续在主线程运行, 被暂停的
    定格展示窗依赖在各绘图节点调用本函数来响应暂停条/关闭等事件。
    sync 模式下 Gcf 中通常已无窗口, 调用为无害空操作。
    """
    try:
        from matplotlib._pylab_helpers import Gcf
    except Exception:
        return
    for mgr in list(Gcf.get_all_fig_managers()):
        try:
            w = getattr(mgr, "window", None)
            if w is not None:
                w.update()
        except Exception:
            pass


def _iter_widgets(w):
    """深度优先遍历 Tk 控件树。"""
    for c in w.winfo_children():
        yield c
        yield from _iter_widgets(c)


class PauseController:
    """给图窗 (+可选对话框) 附加暂停/继续控制。"""

    def __init__(self, fig, dialog_root=None, freeze_extra=(),
                 advance=(), on_close=None, name="figure"):
        """Parameters
        ----------
        fig : matplotlib.figure.Figure
            目标图窗 (需 TkAgg 后端)。
        dialog_root : tk.Widget or None
            附着对话框的 Toplevel; 提供时其控件纳入冻结范围, 并接管其
            ``WM_DELETE_WINDOW`` (原回调经 ``on_close`` 传入)。
        freeze_extra : sequence
            额外纳入冻结的控件 (如图窗内嵌的实时预览滑块)。
        advance : sequence
            "推进类"控件 (保存/确认/取消按钮) —— sync 暂停时禁用,
            防止后台流程在挂起期间被推进。
        on_close : callable or None
            对话框原关闭回调 (async 放行后仍可正常关闭)。
        name : str
            日志显示用的图名。
        """
        self.fig = fig
        self.name = name
        self.dialog_root = dialog_root
        self.advance = [w for w in advance if w is not None]
        self._orig_on_close = on_close
        self.paused = False
        self._released = False      # async 已放行
        self._badge = None
        self._prev_title = None
        self._frozen = []           # 本次暂停被禁用的控件
        self._orig_states = {}

        try:
            self.win = fig.canvas.manager.window
        except Exception:
            self.win = None
        if self.win is None:
            raise RuntimeError("PauseController: 图窗无 Tk 窗口 (非 TkAgg?)")

        # ---- 图窗底部控制条: 暂停按钮 + 状态标签 --------------------
        canvas_w = fig.canvas.get_tk_widget()
        self.bar = tk.Frame(self.win, relief="groove", bd=1)
        self.bar.pack(side="bottom", fill="x", before=canvas_w)
        self.btn = tk.Button(self.bar, text="⏸ 暂停 (P/空格)",
                             command=self.toggle, bg=BTN_PAUSE_BG,
                             relief="raised", padx=8)
        self.btn.pack(side="left", padx=4, pady=2)
        self.status = tk.Label(self.bar, text=TXT_RUN, fg=ACCENT_RUN,
                               anchor="w", padx=8)
        self.status.pack(side="left", fill="x", expand=True)

        # 冻结范围: 对话框控件 + 图窗内嵌控件 (排除控制条自身)
        self.freeze_widgets = list(freeze_extra)
        if dialog_root is not None:
            self.freeze_widgets += list(_iter_widgets(dialog_root))
        self.freeze_widgets = [w for w in dict.fromkeys(self.freeze_widgets)
                               if w not in (self.bar, self.btn, self.status)
                               and not self._is_descendant_of_bar(w)]

        # ---- 快捷键 (文本框获焦时不触发) -----------------------------
        for target in filter(None, (self.win, dialog_root)):
            for seq in ("<Key-p>", "<Key-P>", "<space>"):
                try:
                    target.bind(seq, self._on_key)
                except Exception:
                    pass

        # ---- 窗口关闭拦截 (sync 暂停时拒绝推进) ----------------------
        try:
            self._orig_wm_close = self.win.protocol("WM_DELETE_WINDOW")
            self.win.protocol("WM_DELETE_WINDOW", self._on_win_close)
        except Exception:
            self._orig_wm_close = None
        if dialog_root is not None and on_close is not None:
            dialog_root.protocol("WM_DELETE_WINDOW", self._on_dialog_close)

    # ------------------------------------------------------------ 属性
    @property
    def mode(self):
        return get_plot_config().pause_mode

    # ------------------------------------------------------------ 内部
    def _is_descendant_of_bar(self, w):
        if w is self.bar or w in (self.btn, self.status):
            return True
        p = getattr(w, "master", None)
        while p is not None:
            if p is self.bar:
                return True
            p = getattr(p, "master", None)
        return False

    def _on_key(self, event=None):
        try:
            f = self.win.focus_get()
            if f is not None and f.winfo_class() in _TEXT_LIKE:
                return  # 文本输入中的 p/空格不触发
        except Exception:
            pass
        self.toggle()
        return "break"

    def _set_status(self, text, color):
        try:
            self.status.config(text=text, fg=color)
        except Exception:
            pass

    def flash(self, msg):
        self._set_status("⏸ " + msg, ACCENT_PAUSE)

    def _freeze_ui(self):
        self._orig_states.clear()
        self._frozen.clear()
        for w in self.freeze_widgets + self.advance:
            try:
                self._orig_states[w] = self._widget_state(w)
                w.state(["disabled"])
                self._frozen.append(w)
            except Exception:
                pass

    def _thaw_ui(self):
        for w in self._frozen:
            try:
                self._restore_state(w)
            except Exception:
                pass
        self._frozen.clear()
        self._orig_states.clear()

    @staticmethod
    def _widget_state(w):
        try:
            flags = w.state()
            return tuple(flags)
        except Exception:
            try:
                return ("normal" if str(w.cget("state")) == "normal" else "disabled",)
            except Exception:
                return ("normal",)

    @staticmethod
    def _restore_state(w):
        try:
            flags = PauseController._widget_state(w)
            want = [f for f in flags if f in ("disabled", "active", "pressed",
                                             "readonly", "focus")]
            w.state(want or ["!disabled"])
        except Exception:
            try:
                w.config(state="normal")
            except Exception:
                pass

    def _add_badge(self):
        if self._badge is not None:
            return
        try:
            self._badge = self.fig.text(
                0.995, 0.012, "|| PAUSED", ha="right", va="bottom",
                fontsize=14, color=ACCENT_PAUSE, fontweight="bold",
                alpha=0.85, zorder=1000)
            self.fig.canvas.draw_idle()
        except Exception:
            self._badge = None

    def _remove_badge(self):
        if self._badge is None:
            return
        try:
            self._badge.remove()
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        self._badge = None

    # ------------------------------------------------------------ 动作
    def toggle(self, _e=None):
        (self.resume if self.paused else self.pause)()

    def pause(self):
        if self.paused:
            return
        self.paused = True
        cfg = get_plot_config()
        self._set_status(TXT_PAUSED_SYNC if cfg.pause_mode == "sync"
                         else TXT_PAUSED_ASYNC, ACCENT_PAUSE)
        self.btn.config(text="▶ 继续 (P/空格)")
        try:
            if self._prev_title is None:
                self._prev_title = self.win.title()
            self.win.title(self._prev_title + "  ⏸ 已暂停")
        except Exception:
            pass
        self._add_badge()

        if cfg.pause_mode == "async":
            # 后台继续: 自动保存当前图 → 放行流水线, 窗口转为定格展示
            self._released = True
            print(f"[pause] {self.name}: 暂停(后台继续) — 当前图已保存, "
                  f"流程继续; 本窗口转为定格展示")
            self._release_soon()
        else:
            self._freeze_ui()
            print(f"[pause] {self.name}: 暂停(同步) — 后台已挂起, "
                  f"按 P/空格 或点击继续恢复")

    def resume(self):
        if not self.paused:
            return
        self.paused = False
        self._thaw_ui()
        self.btn.config(text="⏸ 暂停 (P/空格)")
        self._set_status(TXT_RUN, ACCENT_RUN)
        self._remove_badge()
        try:
            if self._prev_title is not None:
                self.win.title(self._prev_title)
                self._prev_title = None
        except Exception:
            pass
        print(f"[pause] {self.name}: 继续 — 恢复正常更新")

    def _release_soon(self):
        """async: 稍后退出 mainloop 放行流程 (不销毁窗口)。"""
        self.win.after(120, self._do_release)

    def _do_release(self):
        try:
            self.fig._pause_keep_open = True
        except Exception:
            pass
        try:
            self.win.quit()  # 退出 mainloop; 窗口保留为定格展示
        except Exception:
            pass

    # ------------------------------------------------------------ 关闭
    def _on_win_close(self):
        if self.paused and self.mode == "sync" and not self._released:
            self.flash("已暂停 — 请先按 P/空格 或点击继续后再关闭")
            return
        if self._orig_wm_close:
            try:
                self.win.tk.call(self._orig_wm_close)
            except Exception:
                try:
                    self.win.destroy()
                except Exception:
                    pass

    def _on_dialog_close(self):
        if self.paused and self.mode == "sync" and not self._released:
            self.flash("已暂停 — 后台已挂起, 请先继续后再操作")
            return
        if self._orig_on_close:
            self._orig_on_close()

    def pump(self):
        """async 放行后由流水线节点调用, 让定格展示窗保持可响应。"""
        if self._released:
            pump_pending_events()
