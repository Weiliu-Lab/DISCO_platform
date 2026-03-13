import matplotlib.pyplot as plt
plt.rcParams.update({
                    # 画布大小, 分辨率
                    'figure.figsize': (6, 5),         # 单位英寸
                    'figure.dpi': 300,

                    # 全局字体
                    'font.family': 'Arial',           # Arial, Times New Roman  #'font.serif': ['Times New Roman'],
                    # 'font.size': 14,                # 全局参数控制, 包括刻度, text, legend, xylabel, title等等
                    'font.weight': 'bold',            # 所有字体默认为粗体
                    'legend.fontsize': 14,            # 图例字体大小

                    # 轴(刻度数字似乎没有单独加粗的全局参数)
                    'xtick.labelsize': 16,            # 设置横轴刻度大小
                    'ytick.labelsize': 16,            # 设置纵轴刻度大小
                    'axes.labelsize': 20,             # 轴标签大小
                    'axes.labelweight': 'bold',       # 轴标签为粗体

                    # 线宽
                    'lines.linewidth': 2,             # 折现,散点连线的线宽
                    'axes.linewidth': 2,              # 坐标轴外框线宽度

                    # 刻度
                    'xtick.direction': 'out',
                    'ytick.direction': 'out',

                    'ytick.major.width': 2,
                    'ytick.minor.width': 2,
                    'xtick.major.width': 2,
                    'xtick.minor.width': 2,

                    # 刻度长度
                    'xtick.major.size': 4,
                    'xtick.minor.size': 2,
                    # 'xtick.minor.visible': True,      # 显示副刻度
                    'ytick.major.size': 4,
                    'ytick.minor.size': 2,
                    # 'ytick.minor.visible': True,      # 显示副刻度

                    # 间距
                    'axes.labelpad': 7,               # 与ORIGIN的0位置一致(只针对(5,4)的画布)
                    # 'ytick.major.pad': 7,
                    # 'xtick.major.pad': 7,
                    })

# 调整线框位置确保一致
#def create_fig_ax(pos=[0.12, 0.21, 0.75, 0.75], figsize=(5, 4), dpi=300):
#    fig = plt.figure(figsize=figsize, dpi=dpi)
#    ax = fig.add_axes(pos)
#    return fig, ax

# 导入python文件时, 顶层代码会被立即执行, 包括定义的变量, 命令, if__main. 函数会被加载, 但是里面的命令不会执行, 直到真正使用.
# 导入库时, 当前目录的优先级大于库文件, 避免重名
# 只有代码被直接执行的时, if __name__ == "__main__"才会被执行, 调用不会, 方便调试
# __all__ = ["module1"]  # 指定通过 from mypackage import * 时只导出 module1
# 包(package)与目录对应, module与file对应, .表示层级关系, mypackage.subpackage.module2 → ./mypackage/subpackage/module2.py 文件
# __init__.py 中可以编写代码(如变量, 函数, 类的定义), 这些代码会在包被导入时自动执行


from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp import types

import io
import json
import base64

import pandas as pd
import numpy as np

# matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# scipy 的 gaussian_kde
try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None


# MCP 名字
mcp = FastMCP("results-csv-density")

# CSV 文件路径（固定路径，不读取环境变量）
CSV_PATH = Path(r"C:\Users\hp\Desktop\results.csv")


def _load_results_df() -> pd.DataFrame:
    path = CSV_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"没有找到 CSV 文件: {path}\n"
            f"请确认该文件存在：{path}"
        )
    df = pd.read_csv(path, engine="python")
    return df


def _df_head_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    df_head = df.head(max_rows)
    cols = list(df_head.columns)

    def esc(s) -> str:
        return str(s).replace("|", "\\|")

    header_line = "| " + " | ".join(esc(c) for c in cols) + " |"
    sep_line = "| " + " | ".join("---" for _ in cols) + " |"

    data_lines = []
    for _, row in df_head.iterrows():
        data_lines.append("| " + " | ".join(esc(row[c]) for c in cols) + " |")

    return "\n".join([header_line, sep_line, *data_lines])


@mcp.tool()
def preview_results(max_rows: int = 20):
    """
    预览 results.csv 的前几行内容。
    """
    try:
        df = _load_results_df()
    except Exception as e:
        return [types.TextContent(type="text", text=str(e))]

    table_md = _df_head_markdown(df, max_rows=max_rows)
    meta = {
        "csv_path": str(CSV_PATH),
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "note": "上面是 CSV 的前几行（markdown 表）。"
    }

    return [
        types.TextContent(type="text", text=table_md),
        types.TextContent(
            type="text",
            text=json.dumps(meta, ensure_ascii=False, indent=2),
        ),
    ]


def density_scatter(x, y, xlabel, ylabel, title, logx=False):
    """
    按点密度上色的散点图（不 plt.show，返回 Figure）。
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # 去掉 NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return None

    # 计算二维密度
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # 按密度排序（低密度在上）
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=35, alpha=0.9)
    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Point density")
    fig.tight_layout()
    return fig


@mcp.tool()
def plot_results():
    """
    使用 results.csv，按你提供的密度散点代码画两张图：
      1. FE vs 面积归一产率（log X）
      2. FE vs 质量归一产率（log X）
    CSV 格式要求：
      - 有一列 performance，里面是 JSON 字符串
      - JSON 里含有：
         faradaic_efficiency_percent
         yield_area_mg_cm2_h
         yield_mass_mg_gcat_h
    """
    if plt is None:
        return [types.TextContent(type="text", text="matplotlib 未安装或不可用，无法绘图。")]
    if gaussian_kde is None:
        return [types.TextContent(type="text", text="scipy 未安装或不可用，无法计算密度。")]

    try:
        df = _load_results_df()
    except Exception as e:
        return [types.TextContent(type="text", text=str(e))]

    if "performance" not in df.columns:
        return [types.TextContent(
            type="text",
            text=f"CSV 缺少 performance 列。现有列: {list(df.columns)}"
        )]

    # 解析 performance JSON
    def parse_perf(s):
        try:
            return json.loads(s) if isinstance(s, str) else {}
        except Exception:
            return {}

    perf = df["performance"].apply(parse_perf)
    df["faradaic_eff"] = perf.apply(lambda d: d.get("faradaic_efficiency_percent"))
    df["yield_area"] = perf.apply(lambda d: d.get("yield_area_mg_cm2_h"))
    df["yield_mass"] = perf.apply(lambda d: d.get("yield_mass_mg_gcat_h"))

    # 转成数值
    for c in ["faradaic_eff", "yield_area", "yield_mass"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 图 1：FE vs 面积归一产率
    mask_area = df["faradaic_eff"].notna() & df["yield_area"].notna()
    fig1 = density_scatter(
        df.loc[mask_area, "yield_area"],
        df.loc[mask_area, "faradaic_eff"],
        xlabel="NH$_3$ yield (mg cm$^{-2}$ h$^{-1}$)",
        ylabel="Faradaic efficiency (%)",
        title="FE vs areal NH$_3$ yield",
        logx=True,
    )

    # 图 2：FE vs 质量归一产率
    mask_mass = df["faradaic_eff"].notna() & df["yield_mass"].notna()
    fig2 = density_scatter(
        df.loc[mask_mass, "yield_mass"],
        df.loc[mask_mass, "faradaic_eff"],
        xlabel="NH$_3$ yield (mg g$_{cat}^{-1}$ h$^{-1}$)",
        ylabel="Faradaic efficiency (%)",
        title="FE vs yield (mass-norm.)",
        logx=True,
    )

    contents: list = []

    for fig in [fig1, fig2]:
        if fig is None:
            contents.append(types.TextContent(type="text", text="没有足够数据生成图。"))
            continue

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png_bytes = buf.read()
        b64 = base64.b64encode(png_bytes).decode("ascii")

        contents.append(
            types.ImageContent(
                type="image",
                data=b64,
                mimeType="image/png",
            )
        )
        plt.close(fig)

    summary = {
        "csv_path": str(CSV_PATH),
        "rows_after_filter": int(df.shape[0]),
        "area_points": int(df.loc[mask_area].shape[0]),
        "mass_points": int(df.loc[mask_mass].shape[0]),
        "note": "上面的图片是密度着色的 FE vs 产率 散点图（横轴对数）。",
    }

    contents.append(
        types.TextContent(
            type="text",
            text=json.dumps(summary, ensure_ascii=False, indent=2),
        )
    )

    return contents


if __name__ == "__main__":
    mcp.run()
