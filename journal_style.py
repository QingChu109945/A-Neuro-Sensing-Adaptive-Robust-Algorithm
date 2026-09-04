"""EAAI / SCI 1区 期刊图表主题 (基于 Nature 出版规范)。

针对 Elsevier/EAAI 与 SCI 1 区投稿要求对 matplotlib 输出做的主题级适配,
灵感与参数基线取自 pubfig (https://github.com/Galaxy-Dawn/pubfig, MIT) 的
Theme 体系与 ggsci/Nature 学术调色板; 同时以"软依赖"方式接入 pubfig: 当运行
环境 (Python >=3.10) 已安装 pubfig 时优先使用其主题与调色板, 否则回退到本
模块内置的等价实现 (兼容实验系统的 Python 3.8 GPU 环境)。

对齐的期刊要点 (figure_beautification_integrated.md §4):
1. 白底 (无 seaborn 灰底), 全文无衬线 Arial (兜底 Helvetica/DejaVu Sans),
   数学符号经 mathtext (stix, Times 风格斜体) 渲染;
2. 字体以 TrueType (fonttype 42) 嵌入矢量输出, 文本可编辑;
3. 轴线/刻度线宽 0.8 pt (Elsevier 建议 0.25-1.5 pt, 主线条 ~1 pt);
4. 去除顶部/右侧脊线, 刻度朝外, 图例无框;
5. Nature 色盲友好调色板 + 语义色映射 (提议方法=深绿 #00A087), 灰度可分;
6. 字号层级: 刻度 8 / 图例 9 / 轴标签 10 / 子图标题 11 / 主标题 13;
7. 同时输出 PNG (300 dpi 预览) + PDF/SVG 矢量副本 (投稿格式)。
"""
import os

import matplotlib

# ---------------------------------------------------------------- 调色板
# Nature 期刊 10 色板 (与 pubfig.colors.palettes.NATURE 一致, SCI 1 区标准)
EAAI_PALETTE = [
    "#E64B35",  # 0 红     — EKF / 基线
    "#4DBBD5",  # 1 青     — UKF / 基线
    "#00A087",  # 2 深绿   — 提议方法 (NS-ARKF / SSM-PINN)
    "#3C5488",  # 3 深蓝   — CKF / 次主色
    "#F39B7F",  # 4 浅橙   — AEKF / 强调
    "#8491B4",  # 5 蓝紫   — DeepKF / 选定值
    "#B09C85",  # 6 棕     — PSO-EKF
    "#7E6148",  # 7 深棕   — GA-UKF
    "#DFCFBE",  # 8 浅棕   — RUKF
    "#DC0000",  # 9 正红   — 保留给关键告警/覆盖失败标记
]

# 常用语义色 (论文级统一配色, 见 figure_beautification_integrated.md §4.2)
C_MAIN = "#00A087"       # 提议方法主线/主柱 (NS-ARKF / SSM-PINN, Nature 深绿)
C_MAIN_EDGE = "#1B5E20"  # 提议方法柱边框 (深绿加粗)
C_ACCENT = "#E64B35"     # 基线/警示 (Nature 红)
C_SECONDARY = "#3C5488"  # 次主色 (深蓝: RMSE 曲线等)
C_NEUTRAL = "#4D4D4D"    # 中性灰 (基线对比)
C_GREY = "#888888"       # 次级灰线 (time/track 等次要指标)
C_GROUND_TRUTH = "#CC4E52"  # 珊瑚红 (ground truth 散点)
C_UNCERTAINTY = "#88CCEE"   # 浅青 (95% CI 填充)
C_UNCERTAINTY2 = "#56B4E9"  # 中青 (CI 宽度带)
C_SELECTED = "#8491B4"   # 蓝紫 (选定超参数虚线)
C_BASELINE = "#9E9E9E"   # 中性灰 (消融/基线柱)
C_HIGHLIGHT = "#F39B7F"  # 浅橙 (协方差自适应等次级模块)

# 滤波方法 -> Nature 9 色映射 (报告 §3 Fig6: 9 色互异且灰度亮度可分)
METHOD_COLORS = {
    "EKF": "#E64B35",
    "UKF": "#4DBBD5",
    "CKF": "#3C5488",
    "AEKF": "#F39B7F",
    "PSO-EKF": "#B09C85",
    "GA-UKF": "#7E6148",
    "DeepKF": "#8491B4",
    "RUKF": "#DFCFBE",
    "NS-ARKF": "#00A087",   # 提议方法 (粗边框强调)
}

# SCI 1 区字号层级 (报告 §4.1): 刻度 8 / 图例 9 / 轴标签 10 / 子图标题 11 / 主标题 13
FONT_SIZES = {
    "base": 10,        # 全局基准 (坐标轴标签 9-10)
    "title": 13,       # 主图标题 (Arial Bold)
    "axes_title": 11,  # 子图标签/小标题 (10-11, Bold)
    "label": 10,       # 坐标轴标签 (含单位)
    "tick": 8,         # 刻度数字
    "legend": 9,       # 图例文字 (8-9)
    "annotation": 8,   # 面板内注释 (RMSE/R², 白底圆角框)
    "bar_label": 6.5,  # 数值标注 (柱顶, 6-7)
}

_PROPOSED_TOKENS = ("ours", "ssm-pinn", "ns-arkf", "proposed")


def method_color(name):
    """按方法名取 Nature 语义色: 提议方法 -> 深绿, 其余查 METHOD_COLORS,
    未知名回退中性灰 (保证新增方法不致配色冲突)。"""
    low = str(name).lower()
    if any(tok in low for tok in _PROPOSED_TOKENS):
        return C_MAIN
    for key, col in METHOD_COLORS.items():
        if key.lower() in low:
            return col
    return C_BASELINE


def rank_colors(n, cmap_list=("#009E73", "#56B4E9", "#CC79A7", "#D55E00")):
    """性能排序渐变色 (报告 §3 Fig7: 绿->蓝->紫红->橙, 灰度亮度递变)。"""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("rank", list(cmap_list))
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    return [cmap(norm(i)) for i in range(n)]


_FALLBACK_FONTS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def _available_fonts():
    """matplotlib 可用字体名集合。"""
    try:
        from matplotlib import font_manager
        return {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return set()


def pick_font_family():
    """按优先级选出首个可用无衬线字体 (Arial 优先, 兜底 DejaVu Sans)。"""
    avail = _available_fonts()
    for name in _FALLBACK_FONTS:
        if name in avail:
            return name
    return "DejaVu Sans"


def palette(n=None, with_pubfig=True):
    """返回前 n 个颜色; pubfig 可用时用其 NATURE 调色板 (内容一致)。"""
    colors = None
    if with_pubfig:
        try:
            from pubfig.colors.palettes import NATURE as colors  # noqa: E501
        except Exception:
            colors = None
    if colors is None:
        colors = list(EAAI_PALETTE)
    if n is not None:
        colors = (colors * ((n // len(colors)) + 1))[:n]
    return colors


def journal_rcparams():
    """期刊主题 rcParams (基于 pubfig Theme.rc_params 的等价子集 + 报告 §4.1 字号层级)。"""
    fam = pick_font_family()
    fs = FONT_SIZES
    return {
        # 排版 (Arial 全文 + stix 数学符号)
        "font.family": fam,
        "font.size": fs["base"],
        "mathtext.fontset": "stix",
        "text.color": "black",
        "axes.titlesize": fs["axes_title"],
        "axes.labelsize": fs["label"],
        "xtick.labelsize": fs["tick"],
        "ytick.labelsize": fs["tick"],
        "legend.fontsize": fs["legend"],
        "axes.labelpad": 2.5,
        "axes.titlepad": 4.0,
        # 图例 (出版风格: 无框, 外置不遮挡数据)
        "legend.frameon": False,
        "legend.fancybox": False,
        "legend.borderaxespad": 0.3,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        # 默认线宽 (Elsevier: 主线条 ~1 pt, 最小 0.25 pt)
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.6,
        # 轴 (去顶/右脊, 0.8 pt)
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.alpha": 0.8,
        # 刻度
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # 背景
        "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        # 矢量输出文本可编辑 (fonttype 42 = TrueType)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # 默认颜色循环 = Nature 调色板
        "axes.prop_cycle": matplotlib.cycler(color=palette()),
        "patch.facecolor": EAAI_PALETTE[2],
    }


def apply_journal_style():
    """把期刊主题写入全局 rcParams; 返回是否由 pubfig 提供 (信息用)。"""
    with_pubfig = _try_pubfig_style()
    if not with_pubfig:
        matplotlib.rcParams.update(journal_rcparams())
    return with_pubfig


def _try_pubfig_style():
    """pubfig 可用时直接套用其 default 主题 + NATURE 循环。

    pubfig 要求 Python>=3.10; 实验环境 (3.8 GPU) 下不可用, 返回 False,
    由本模块内置等价实现兜底 (两者参数同源, 视觉一致)。
    """
    try:
        import pubfig.themes as themes
        from pubfig.colors.palettes import NATURE
        import matplotlib.pyplot as plt
    except Exception:
        return False
    try:
        theme = themes.get_theme("default")
        rc = theme.rc_params()
        rc.setdefault("axes.prop_cycle",
                      matplotlib.cycler(color=list(NATURE)))
        plt.rcParams.update(rc)
        return True
    except Exception:
        return False


def vector_copy_paths(png_path):
    """由 PNG 路径得到同名矢量副本路径列表 (PDF 投稿 + SVG 可编辑, 报告 §4.3)。"""
    if not png_path:
        return []
    base, _ = os.path.splitext(os.path.abspath(png_path))
    return [base + ".pdf", base + ".svg"]


def vector_copy_path(png_path):
    """兼容旧接口: 返回首个 (PDF) 矢量副本路径。"""
    paths = vector_copy_paths(png_path)
    return paths[0] if paths else None


def save_with_vector(fig, png_path, dpi=300, vector=True, svg=True):
    """保存 PNG (300 dpi 预览) 并按需输出同名 PDF + SVG 矢量副本。

    返回实际写盘路径列表。矢量文件由 matplotlib 原生生成 (fonttype 42,
    文本保留可编辑), 满足期刊对 PDF/SVG 投稿格式的要求 (报告 §4.3: 主稿件
    只提交 PDF/SVG, PNG 仅审稿预览)。
    """
    written = []
    if not png_path:
        return written
    try:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        written.append(png_path)
    except Exception as exc:
        print(f"[journal_style] PNG 保存失败 {png_path}: {exc}")
        return written
    if vector:
        pdf = vector_copy_paths(png_path)[0]
        try:
            fig.savefig(pdf, bbox_inches="tight")  # 矢量, dpi 不影响
            written.append(pdf)
        except Exception as exc:
            print(f"[journal_style] PDF 矢量副本保存失败 {pdf}: {exc}")
    if svg:
        svg_path = vector_copy_paths(png_path)[1]
        try:
            fig.savefig(svg_path, bbox_inches="tight")
            written.append(svg_path)
        except Exception as exc:
            print(f"[journal_style] SVG 矢量副本保存失败 {svg_path}: {exc}")
    return written
