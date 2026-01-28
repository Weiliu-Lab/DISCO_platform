import base64
import os
import io
import time
import re
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
import ase.io
import traceback
import sys
from pathlib import Path
import json

from flask import send_from_directory

import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS

# 路径设置
# ml_prediction 作为独立包运行时，不需要通过 sys.path 注入项目根目录。
MODULE_ROOT = Path(__file__).resolve().parent
# 兼容旧代码：部分逻辑将相对路径视为“项目根目录”。
# 在 ml_prediction 中，我们把它定义为包根目录。
PROJECT_ROOT = MODULE_ROOT

def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "/").strip()
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not prefix.endswith("/"):
        prefix += "/"
    if prefix == "//":
        return "/"
    return prefix

ML_URL_PREFIX = _normalize_prefix(os.getenv("ML_URL_PREFIX", "/"))

# 兼容直接运行：python ml_prediction/app.py
if __package__ in (None, ""):
    pkg_parent = str(MODULE_ROOT.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)

from ml_prediction.config.loader import (
    load_config,
    resolve_config_path,
    get_commands,
    get_executor,
    get_local_paths,
    get_remote_server,
    get_module_task,
)
from ml_prediction.config.ssh_manager import SSHManager
from ml_prediction.parsing import parse_atoms, parse_structure_for_viewer, parse_csv_content
from ml_prediction.sisso.config_manager import SissoConfigManager
from ml_prediction.sisso.data_builder import SissoTrainDataBuilder


TRADITIONAL_MODEL_OPTIONS = ['xgb', 'rf', 'svr', 'gpr', 'krr', 'mlp']
GNN_MODEL_OPTIONS = ['dimenet_pp']


def _download_text(ssh: SSHManager, remote_path: str) -> str:
    ok, content = ssh.read_remote_file(remote_path)
    if not ok:
        raise RuntimeError(f"读取远端文件失败: {remote_path} ({content})")
    return content

# =============================================================================
# 2. 核心配置与 DataBuilder
# =============================================================================

DEFAULT_CFG_PATH = resolve_config_path()
try:
    CONFIG = load_config(str(DEFAULT_CFG_PATH))
except Exception as e:
    print(f"[FATAL] 配置加载失败: {e}\n{traceback.format_exc()}")
    raise

MAX_BATCHES = 12
elements_csv = MODULE_ROOT / "elements_properties_all.csv"
if not elements_csv.exists():
    raise FileNotFoundError(f"缺少必需文件: {elements_csv}")
ELEMENTS_DF = pd.read_csv(elements_csv)
if ELEMENTS_DF is None or ELEMENTS_DF.empty:
    raise ValueError(f"元素属性表为空: {elements_csv}")

# =============================================================================
# 3. UI 初始化
# =============================================================================
_LOCAL_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_LOCAL_ASSETS_ROUTE = f"{ML_URL_PREFIX}_local_assets/<path:filename>"

app = dash.Dash(
    __name__,
    assets_folder=SETTINGS.ASSETS_PATH,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",
        f"{ML_URL_PREFIX}_local_assets/style.css?v=1",
        f"{ML_URL_PREFIX}_local_assets/modal_tabs_fix.css?v=1",
    ],
    requests_pathname_prefix=ML_URL_PREFIX,
    routes_pathname_prefix=ML_URL_PREFIX,
)
server = app.server


@server.route(_LOCAL_ASSETS_ROUTE)
def _serve_local_assets(filename):
    return send_from_directory(str(_LOCAL_ASSETS_DIR), filename)

# 工具函数（ASE only for parsing; pymatgen only at CTK boundary）
new_batch_uploader = dcc.Upload(
    id="new-batch-uploader",
    accept=".cif,.vasp,.poscar,.contcar,.POSCAR,.CONTCAR,.CIF,.VASP",
    children=html.Div([
        html.I(className="bi bi-cloud-upload", style={"fontSize": "2rem"}),
        html.Div("拖入文件 (.cif/.vasp)"),
        html.Div("生成新批次", className="text-muted small")
    ]),
    className="upload-container",
    multiple=True
)

task_control_card = dbc.Card([
    dbc.CardHeader("3. 任务控制", className="fw-bold text-primary py-2"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dbc.Button("预览文件 & 提交", id="btn-generate", color="primary", className="w-100"), width=6),
            dbc.Col(dbc.Button("拉取状态 & 后处理", id="btn-pull-status", outline=True, color="warning", className="w-100"), width=6),
        ], className="g-2 mb-1"),
        html.Pre(
            id="log-main",
            style={
                "flex": "1 1 0",
                "minHeight": 0,
                "overflowY": "auto",
                "backgroundColor": "#000",
                "color": "#fff",
                "fontSize": "0.7rem",
                "whiteSpace": "pre-wrap",
                "padding": "6px",
                "border": "1px solid #333",
                "margin": 0,
            },
        ),
    ], style={"display": "flex", "flexDirection": "column", "height": "100%", "minHeight": 0})
], style={"height": "100%", "minHeight": 0})

sisso_settings_card = dbc.Card([
    dbc.CardHeader("2. 全局参数 (Settings)", className="fw-bold text-primary py-2"),
    dbc.CardBody([
        dbc.Label("方法选择 (Method)", className="small fw-bold"),
        dbc.RadioItems(
            id="tool-selector",
            options=[
                {"label": "SISSO 描述符", "value": "sisso"},
                {"label": "传统机器学习 (Traditional ML)", "value": "trad_ml"},
                {"label": "图神经网络 (GNN)", "value": "gnn"},
            ],
            value="sisso",
            inline=True,
            className="mb-3 small"
        ),
        
        # --- Context: Traditional ML ---
        html.Div([
            dbc.Label("模型选择 (Model)", className="small fw-bold"),
            dcc.Dropdown(
                id="ml-model-selector",
                options=[{'label': m.upper(), 'value': m} for m in TRADITIONAL_MODEL_OPTIONS],
                value='xgb',
                clearable=False,
                className="mb-3"
            )
        ], id="container-trad-ml", style={"display": "none"}),

        # --- Context: GNN ---
        html.Div([
            dbc.Label("模型选择 (GNN Model)", className="small fw-bold"),
            dcc.Dropdown(
                id="gnn-model-selector",
                options=[{'label': 'DimeNet++', 'value': 'dimenet_pp'}],
                value='dimenet_pp',
                clearable=False,
                className="mb-3"
            ),
            dbc.Alert(
                "GNN 目前仅支持从结构生成数据（不支持直接上传）。",
                color="warning",
                className="py-1 small",
            ),
        ], id="container-gnn", style={"display": "none"}),

        # --- Context: SISSO ---
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("维度范围 (Min-Max)", className="small"),
                    dbc.InputGroup([
                        dbc.Input(id="inp-dim-min", type="number", value=1, min=1, max=5, size="sm"),
                        dbc.InputGroupText("-", style={"padding": "0 5px"}),
                        dbc.Input(id="inp-dim-max", type="number", value=3, min=1, max=5, size="sm"),
                    ], size="sm")
                ], width=6),
                dbc.Col([
                    dbc.Label("复杂度范围 (Min-Max)", className="small"),
                    dbc.InputGroup([
                        dbc.Input(id="inp-cplx-min", type="number", value=2, min=1, max=10, size="sm"),
                        dbc.InputGroupText("-", style={"padding": "0 5px"}),
                        dbc.Input(id="inp-cplx-max", type="number", value=4, min=1, max=10, size="sm"),
                    ], size="sm")
                ], width=6)
            ], className="mb-2"),
            dbc.Label("运算符", className="small"),
            dcc.Dropdown(id="inp-ops", options=[{'label': o, 'value': f'({o})'} for o in ['+', '-', '*', '/', 'exp', 'log', '^2', 'sqrt', 'sin', 'cos']], value=['(+)', '(-)', '(*)', '(/)'], multi=True, style={"fontSize": "0.8rem"}),
        ], id="container-sisso"),

        html.Div([
            html.Hr(className="my-2"),
            dbc.Label("特征属性 (Input Features)", className="small"),
            dcc.Dropdown(id="feature-columns", options=[{'label': c, 'value': c} for c in ELEMENTS_DF.columns if c not in ['symbol', 'name', 'description']], multi=True, placeholder="留空默认全选", style={"fontSize": "0.8rem"})
        ], id="feature-selection-container")
    ], style={"overflow": "auto", "minHeight": 0, "height": "100%"})
], style={"overflow": "visible", "zIndex": 100, "height": "100%", "minHeight": 0})

# 新增：直接上传组件
direct_train_uploader = dcc.Upload(
    id="direct-train-uploader",
    accept=".dat,.txt,.csv",
    children=html.Div([
        html.I(className="bi bi-file-earmark-code", style={"fontSize": "2rem"}),
        html.Div("拖入 train.dat"),
        html.Div("直接使用现有数据", className="text-muted small")
    ]),
    className="upload-container",
    multiple=False
)

file_editor_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("预览与编辑")),
    dbc.ModalBody([
        dbc.ButtonGroup(
            [
                dbc.Button(id="modal-file-btn-a", color="primary", outline=False, size="sm"),
                dbc.Button(id="modal-file-btn-b", color="secondary", outline=True, size="sm"),
            ],
            className="mb-2",
        ),
        dcc.Textarea(
            id="editor-sisso",
            style={
                "width": "100%",
                "height": "520px",
                "fontFamily": "monospace",
                "fontSize": "14px",
                "marginTop": "10px",
                "display": "block"
            }
        ),
        dcc.Textarea(
            id="editor-train",
            style={
                "width": "100%",
                "height": "520px",
                "fontFamily": "monospace",
                "fontSize": "14px",
                "whiteSpace": "pre",
                "overflowX": "scroll",
                "marginTop": "10px",
                "display": "none"
            }
        )
    ]),
    dbc.ModalFooter([
        dbc.Button("取消", id="btn-close-modal", className="me-2"),
        dcc.Loading(dbc.Button("提交任务", id="btn-submit-modal", color="primary"), type="circle")
    ])
], id="modal-file-editor", size="xl", backdrop=True, style={"zIndex": 10000}, is_open=False)


@app.callback(
    Output("modal-file-btn-a", "children"),
    Output("modal-file-btn-b", "children"),
    Input("tool-selector", "value"),
)
def update_modal_file_button_labels(tool_mode):
    if tool_mode == "trad_ml":
        return "model.py", "train_data.csv"
    if tool_mode == "gnn":
        return "run_gnn.py", "targets.csv"
    return "SISSO.in", "train.dat"


@app.callback(
    Output("editor-sisso", "style"),
    Output("editor-train", "style"),
    Output("modal-file-btn-a", "color"),
    Output("modal-file-btn-a", "outline"),
    Output("modal-file-btn-b", "color"),
    Output("modal-file-btn-b", "outline"),
    Input("modal-file-btn-a", "n_clicks"),
    Input("modal-file-btn-b", "n_clicks"),
    prevent_initial_call=True
)
def toggle_modal_file_view(btn_a_clicks, btn_b_clicks):
    trigger = ctx.triggered_id
    
    if trigger == "modal-file-btn-a":
        return (
            {"width": "100%", "height": "520px", "fontFamily": "monospace", "fontSize": "14px", "marginTop": "10px", "display": "block"},
            {"width": "100%", "height": "520px", "fontFamily": "monospace", "fontSize": "14px", "whiteSpace": "pre", "overflowX": "scroll", "marginTop": "10px", "display": "none"},
            "primary", False,
            "secondary", True
        )
    elif trigger == "modal-file-btn-b":
        return (
            {"width": "100%", "height": "520px", "fontFamily": "monospace", "fontSize": "14px", "marginTop": "10px", "display": "none"},
            {"width": "100%", "height": "520px", "fontFamily": "monospace", "fontSize": "14px", "whiteSpace": "pre", "overflowX": "scroll", "marginTop": "10px", "display": "block"},
            "secondary", True,
            "primary", False
        )
    
    raise PreventUpdate


@app.callback(
    Output("tabs-input-mode", "options"),
    Output("tabs-input-mode", "value"),
    Input("tool-selector", "value"),
    State("tabs-input-mode", "value"),
)
def sync_input_mode_options(tool_mode, current_value):
    base_options = [
        {"label": "从结构生成", "value": "tab-struct"},
        {"label": "直接上传", "value": "tab-direct"},
    ]

    # GNN 不支持直接上传
    if tool_mode == "gnn":
        return [{"label": "从结构生成", "value": "tab-struct"}], "tab-struct"

    # 其他模式保持两种输入方式
    if current_value not in {"tab-struct", "tab-direct"}:
        current_value = "tab-struct"
    return base_options, current_value

left_panel = html.Div(
    [
        html.Div(
            dbc.Card([
                dbc.CardHeader("1. 新建 (New Batch)", className="fw-bold text-primary py-2"),
                dbc.CardBody([
                    dbc.RadioItems(
                        id="tabs-input-mode",
                        className="btn-group w-100 mb-3",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "从结构生成", "value": "tab-struct"},
                            {"label": "直接上传", "value": "tab-direct"},
                        ],
                        value="tab-struct",
                    ),
                    html.Div(new_batch_uploader, id="content-tab-struct"),
                    html.Div([
                        direct_train_uploader,
                        html.Div(id="direct-upload-status", className="mt-2 text-success small")
                    ], id="content-tab-direct", style={"display": "none"})
                ], className="p-2", style={"overflow": "auto", "minHeight": 0, "height": "100%"})
            ], className="h-100", style={"height": "100%", "minHeight": 0}),
            # 占比：上 1/5
            style={"flex": "1 1 0", "minHeight": 0},
        ),
        html.Div(
            sisso_settings_card,
            # 占比：下两块均分剩余（各 2/5）
            style={"flex": "2 1 0", "minHeight": 0},
        ),
        html.Div(
            task_control_card,
            style={"flex": "2 1 0", "minHeight": 0},
        ),
    ],
    style={"display": "flex", "flexDirection": "column", "gap": "4px", "height": "100%", "minHeight": 0}
)
right_panel = [
    dbc.Card([
        dbc.CardHeader([
            "批次工作区 (Workspace)", 
            dbc.Button("🗑️ 清空所有", id="btn-reset-all", color="link", size="sm", className="float-end text-decoration-none text-danger py-0")
        ], className="fw-bold text-primary py-2"), 
              dbc.CardBody([html.Div(id="batches-container", className="row g-2"), html.Div("请在左侧拖入结构文件以开始...", id="empty-placeholder", className="text-center text-muted py-5")], className="p-2")], className="h-100")
]

result_panel = dbc.Card(
    [
        dbc.CardHeader("4. 计算结果", className="fw-bold text-primary py-2"),
        dbc.CardBody(html.Div(id='result-display')),
    ],
    className="h-100",
    style={"flex": "1 1 0", "minHeight": 0, "overflow": "auto"}
)

ctc.register_crystal_toolkit(app=app, layout=dbc.Container([
    file_editor_modal, 
    # 使用 sessionStorage，避免 localStorage 残留旧批次（例如误拖 train.dat 后一直带着）
    dcc.Store(id='store-batches-data', data=[], storage_type='session'), 
    dcc.Store(id='store-job-info', data={}, storage_type='session'), 
    dcc.Store(id='store-log-gen', data=[], storage_type='session'),
    dcc.Store(id='store-log-job', data=[], storage_type='session'),
    dcc.Interval(id='interval-job-monitor', interval=10000, n_intervals=0),
    # 主体区域：占满高度
    dbc.Row(
        [
            dbc.Col(left_panel, width=12, lg=3, style={"height": "100%", "minHeight": 0}),
            dbc.Col(
                [
                    html.Div(
                        right_panel,
                        id="container-batch-workspace",
                        style={"flex": "1 1 0", "minHeight": 0, "overflow": "auto"}
                    ),
                    result_panel,
                ],
                width=12,
                lg=9,
                style={"display": "flex", "flexDirection": "column", "gap": "12px", "height": "100%", "minHeight": 0}
            ),
        ],
        style={"flex": "1 1 0", "minHeight": 0}
    ),
    ], fluid=True, style={"height": "100vh", "backgroundColor": "#f8f9fa", "display": "flex", "flexDirection": "column"}))

# =============================================================================
# 4. 回调函数
# =============================================================================

@app.callback(
    Output("store-batches-data", "data"), Output("batches-container", "children"), Output("empty-placeholder", "style"), Output("new-batch-uploader", "contents"),
    Input("new-batch-uploader", "contents"), Input("btn-reset-all", "n_clicks"),
    State("new-batch-uploader", "filename"), State("store-batches-data", "data"), State("batches-container", "children"),
    State("tool-selector", "value"),
)
def create_new_batch(contents, n_reset, filenames, current_data, current_children, tool_mode):
    if ctx.triggered_id == "btn-reset-all": return [], [], {"display": "block"}, None
    if not contents: raise PreventUpdate
    if current_data is None: current_data = []
    if current_children is None: current_children = []
    
    # 仅接受结构文件，避免误拖 train.dat 等非结构文件
    allowed_ext = {".cif", ".vasp", ".poscar", ".contcar"}
    new_structures = []
    for c, f in zip(contents, filenames or []):
        fn = os.path.basename(str(f)).strip()
        ext = os.path.splitext(fn.lower())[1]
        if ext not in allowed_ext:
            continue
        new_structures.append({'filename': fn, 'content': c.split(",")[1]})

    if not new_structures:
        raise PreventUpdate
    
    batch_id = len(current_data)
    current_data.append({"id": batch_id, "structures": new_structures})
    
    init_struct = parse_structure_for_viewer(new_structures[0]['content'], new_structures[0].get('filename')) if new_structures else None
    
    hide_indices = (tool_mode == "gnn")
    card = dbc.Col(dbc.Card([
        dbc.CardHeader([dbc.Row([
            dbc.Col([html.Strong(f"#{batch_id+1}"), html.Span(f"{len(new_structures)}", className="badge bg-secondary ms-1")], width="auto"),
            dbc.Col([dcc.Dropdown(id={'type': 'batch-struct-select', 'index': batch_id}, options=[{'label': s['filename'], 'value': i} for i, s in enumerate(new_structures)], value=0, clearable=False)], width=3),
            dbc.Col(
                [dbc.Input(id={'type': 'batch-indices-input', 'index': batch_id}, placeholder="Index (e.g. 48 52)", size="sm")],
                width=4,
                style={"display": "none"} if hide_indices else None,
            ),
            dbc.Col([dcc.Upload(id={'type': 'batch-csv-upload', 'index': batch_id}, accept=".csv", children=html.Div([html.Div([html.I(className="bi bi-file-earmark-arrow-up"), " CSV"], id={'type': 'batch-csv-label', 'index': batch_id}), html.Div(id={'type': 'batch-csv-status', 'index': batch_id}, className="text-success small fw-bold ms-1")]), style={"border": "1px dashed #6c757d", "height": "31px", "cursor": "pointer", "backgroundColor": "#f8f9fa", "display": "flex", "alignItems": "center", "justifyContent": "center"})], width=3)
        ], className="g-1 align-items-center")]),
        dbc.CardBody([ctc.StructureMoleculeComponent(init_struct, id=f"viewer-batch-{batch_id}", color_scheme="VESTA").layout(size="550px")])
    ], className="shadow-sm border-0 mb-3"), width=12, lg=6, xl=6)
    
    current_children.append(card)
    return current_data, current_children, {"display": "none"}, None

@app.callback([Output(f"viewer-batch-{i}", "data") for i in range(MAX_BATCHES)], Input({'type': 'batch-struct-select', 'index': ALL}, 'value'), State("store-batches-data", "data"))
def update_dynamic_viewers(vals, data):
    outs = [no_update] * MAX_BATCHES
    if not data or not vals: return outs
    for i, idx in enumerate(vals):
        if i < len(data) and idx is not None:
            s = data[i]['structures'][idx]
            outs[i] = parse_structure_for_viewer(s.get('content'), s.get('filename'))
    return outs

@app.callback(Output({'type': 'batch-csv-status', 'index': MATCH}, 'children'), Output({'type': 'batch-csv-label', 'index': MATCH}, 'style'), Input({'type': 'batch-csv-upload', 'index': MATCH}, 'contents'), State({'type': 'batch-csv-upload', 'index': MATCH}, 'filename'))
def update_csv_status(c, f):
    if not c: return "", {"display": "block"}
    df, cnt = parse_csv_content(c)
    return f"✓ {f[:5]}..", {"display": "none"} if cnt > 0 else {"display": "block"}

# --- 新增：Tab 切换与直接上传状态 ---
@app.callback(
    Output("content-tab-struct", "style"), 
    Output("content-tab-direct", "style"),
    Output("feature-selection-container", "style"),
    Output("container-batch-workspace", "style"),
    Input("tabs-input-mode", "value"),
    Input("tool-selector", "value"),
)
def switch_tab_content(at, tool_mode):
    # GNN：没有特征属性选择
    feature_style = {"display": "none"} if tool_mode == "gnn" else {"display": "block"}

    if at == "tab-direct":
        return (
            {"display": "none"},
            {"display": "block"},
            {"display": "none"},
            {"display": "none"},
        )
    return (
        {"display": "block"},
        {"display": "none"},
        feature_style,
        {"display": "block", "flex": "1 1 0", "minHeight": 0, "overflow": "auto"},
    )

@app.callback(
    Output("container-sisso", "style"),
    Output("container-trad-ml", "style"),
    Output("container-gnn", "style"),
    Input("tool-selector", "value")
)
def toggle_method_settings(method):
    if method == "trad_ml":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}
    if method == "gnn":
        return {"display": "none"}, {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}

@app.callback(Output("direct-upload-status", "children"), Input("direct-train-uploader", "contents"), State("direct-train-uploader", "filename"))

def update_direct_status(c, f):
    if c: return f"已加载: {f}"
    return ""


@app.callback(
    Output("log-main", "children"),
    Input("store-log-gen", "data"),
    Input("store-log-job", "data"),
)
def render_combined_log(gen_logs, job_logs):
    gen_logs = gen_logs or []
    job_logs = job_logs or []

    merged = []
    for e in gen_logs:
        if isinstance(e, dict) and 'ts' in e and 'msg' in e:
            merged.append(e)
    for e in job_logs:
        if isinstance(e, dict) and 'ts' in e and 'msg' in e:
            merged.append(e)

    merged.sort(key=lambda x: x.get('ts', 0))

    lines = []
    for e in merged[-200:]:
        ts = e.get('ts', 0)
        src = e.get('src', 'LOG')
        try:
            tstr = time.strftime('%H:%M:%S', time.localtime(ts))
        except Exception:
            tstr = "--:--:--"
        msg = str(e.get('msg', ''))
        lines.append(f"[{tstr}][{src}]\n{msg}\n")

    return "\n".join(lines).strip()

# --- [核心合并逻辑] 列名标准化 ---
@app.callback(
    Output("store-log-gen", "data"),
    Output("editor-sisso", "value"),
    Output("editor-train", "value"),
    Input("btn-generate", "n_clicks"),
    State("store-batches-data", "data"),
    State({'type': 'batch-indices-input', 'index': ALL}, 'value'),
    State({'type': 'batch-csv-upload', 'index': ALL}, 'contents'),
    State("inp-dim-min", "value"),
    State("inp-dim-max", "value"),
    State("inp-cplx-min", "value"),
    State("inp-cplx-max", "value"),
    State("inp-ops", "value"),
    State("feature-columns", "value"),
    State("tabs-input-mode", "value"),
    State("direct-train-uploader", "contents"),
    State("tool-selector", "value"),
    State("ml-model-selector", "value"),
    State("gnn-model-selector", "value"),
    State("store-log-gen", "data"),
)
def generate_merge(n, batch_data_list, indices_list, csv_contents_list, dim_min, dim_max, cplx_min, cplx_max, ops, feat_cols, active_tab, direct_content, tool_mode, ml_model, gnn_model, log_store):
    if not n: raise PreventUpdate
    logs = ["开始处理...", f"[Debug] tab={active_tab}, batches={len(batch_data_list or [])}"]

    if log_store is None:
        log_store = []

    def _append_log(block: str):
        log_store.append({"ts": time.time(), "src": "GEN", "msg": block})
    
    # 占位符模板生成逻辑
    def get_sisso_template(nsample, nsf):
        try:
            cm = SissoConfigManager("sisso/templates/SISSO.in")
            # 使用占位符 {{dim}} 和 {{cplx}}
            return cm.update_template({
                "desc_dim": "{{dim}}",
                "nsample": nsample, 
                "nsf": nsf,
                "fcomplexity": "{{cplx}}",
                "ops": "".join(ops) if ops else ""
            })
        except Exception as e:
            return f"Template Error: {e}"

    def get_ml_config(model):
        return json.dumps({
            "model_name": model,
            "params": {} 
        }, indent=2)

    def get_ml_script(default_model: str):
        try:
            ml_script_path = MODULE_ROOT / "trad_ml" / "model.py"
            with open(ml_script_path, "r", encoding="utf-8") as f:
                tpl = f.read()
            return tpl.replace("__DEFAULT_MODEL__", (default_model or "xgb").strip().lower())
        except Exception as e:
            return f"# Failed to load ML model.py template: {e}\n"

    def get_gnn_script():
        try:
            gnn_script_path = MODULE_ROOT / "gnn" / "run_gnn.py"
            with open(gnn_script_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"# Failed to load GNN run_gnn.py template: {e}\n"

    # --- 分支 1: 直接上传模式 ---
    if active_tab == "tab-direct":
        if not direct_content:
            _append_log("\n".join(logs + ["错误: 请先上传数据文件"]))
            return log_store, "", ""
        
        try:
            # 解析
            content_type, content_string = direct_content.split(',')
            decoded = base64.b64decode(content_string)
            final_train_dat = decoded.decode('utf-8')

            if tool_mode == "trad_ml":
                # 尝试验证 CSV
                logs.append(f"【Trad ML】直接上传模式: {ml_model}")
                _append_log("\n".join(logs))
                return log_store, get_ml_script(ml_model), final_train_dat

            if tool_mode == "gnn":
                _append_log("\n".join(logs + ["错误: GNN 不支持直接上传，请切换到‘从结构生成’。"]))
                return log_store, "", ""
            
            # SISSO 逻辑
            lines = final_train_dat.strip().split('\n')
            lines = [l for l in lines if l.strip()]
            
            if len(lines) < 2:
                _append_log("\n".join(logs + ["错误: train.dat 内容过短"]))
                return log_store, "", ""
                
            header = lines[0].split()
            real_nsample = len(lines) - 1
            # nsf = 列数 - 2 (第一列通常是 Materials, 第二列是 Property)
            real_nsf = len(header) - 2
            
            logs.append(f"【直接模式】已解析 train.dat: nsample={real_nsample}, nsf={real_nsf}")
            logs.append(f"【参数范围】Dim: {dim_min}-{dim_max}, Cplx: {cplx_min}-{cplx_max}")
            
            sisso_in_content = get_sisso_template(real_nsample, real_nsf)
            _append_log("\n".join(logs))
            return log_store, sisso_in_content, final_train_dat
            
        except Exception as e:
            _append_log("\n".join(logs + [f"解析文件失败: {e}"]))
            return log_store, "", ""

    # --- 分支 2: 结构生成模式 ---
    if not batch_data_list:
        raise PreventUpdate

    # --- GNN: 结构生成 targets.csv（不依赖原子索引/元素特征） ---
    if tool_mode == "gnn":
        try:
            gnn_model = (gnn_model or "dimenet_pp").strip().lower()
            if gnn_model not in GNN_MODEL_OPTIONS:
                _append_log("\n".join(logs + [f"错误: 不支持的 GNN 模型: {gnn_model}"]))
                return log_store, "", ""

            targets_rows = []

            if len(batch_data_list) == len(csv_contents_list):
                valid_batches = zip(batch_data_list, csv_contents_list)
            else:
                min_len = min(len(batch_data_list), len(csv_contents_list))
                valid_batches = zip(batch_data_list[:min_len], csv_contents_list[:min_len])

            for i, (batch_data, csv_content) in enumerate(valid_batches):
                raw_structs = batch_data.get('structures') or []

                csv_df, _ = parse_csv_content(csv_content)

                targets_map = csv_df.iloc[:, 0].to_dict()

                for s in raw_structs:
                    fname = os.path.basename(str(s.get('filename', ''))).strip()
                    stem = os.path.splitext(fname)[0]
                    key_candidates = [fname, stem, fname.split('.')[0]]
                    matched_key = next((k for k in key_candidates if k in targets_map), None)
                    if matched_key is None:
                        continue
                    try:
                        y = float(targets_map[matched_key])
                    except Exception:
                        raise ValueError(f"Batch #{i+1} 目标值无法转为数值: key={matched_key!r} value={targets_map[matched_key]!r}")
                    gid = f"b{i+1}_{stem}"
                    targets_rows.append({"id": gid, "target": y})

            if not targets_rows:
                _append_log("\n".join(logs + ["错误: 未能从 CSV 匹配到任何结构目标值（请检查结构名与CSV第一列是否一致）。"]))
                return log_store, "", ""

            targets_df = pd.DataFrame(targets_rows)
            gnn_script = get_gnn_script()
            logs.append(f"【GNN 生成】targets={len(targets_df)} 行；模型: {gnn_model}")
            _append_log("\n".join(logs))
            return log_store, gnn_script, targets_df.to_csv(index=False)
        except Exception as e:
            _append_log("\n".join(logs + [f"GNN 生成失败: {e}"]))
            return log_store, "", ""

    if not feat_cols or len(feat_cols) <= 0:
        raise ValueError("未选择特征列（feature-columns 不能为空）")
    current_feat_list = feat_cols

    builder = SissoTrainDataBuilder(ELEMENTS_DF)

    all_dfs = []
    
    # [Fix] 使用 zip 安全遍历，避免 State 列表长度不一致导致的 index out of range
    # 过滤掉 None 的输入
    if len(batch_data_list) != len(indices_list) or len(batch_data_list) != len(csv_contents_list):
        raise ValueError(
            f"内部状态长度不一致：batches={len(batch_data_list)}, indices={len(indices_list)}, csv={len(csv_contents_list)}"
        )
    valid_batches = zip(batch_data_list, indices_list, csv_contents_list)

    for i, (batch_data, indices_val, csv_content) in enumerate(valid_batches):
        indices_str = str(indices_val).replace(",", " ").strip()
        if not indices_str:
            raise ValueError(f"Batch #{i+1} 未输入原子索引")
        indices = [int(x) for x in indices_str.split()]

        # --- Debug: 结构文件名样本 ---
        raw_structs = (batch_data or {}).get("structures") or []
        sample_names = [os.path.basename(str(s.get("filename", ""))).strip() for s in raw_structs[:10]]
        logs.append(f"Batch #{i+1} 结构数: {len(raw_structs)}; 文件名样本: {sample_names}")

        csv_df, _ = parse_csv_content(csv_content)

        # --- Debug: CSV 结构 ---
        logs.append(f"Batch #{i+1} CSV列: {list(csv_df.columns)}")
        logs.append(f"Batch #{i+1} CSV key样本: {[str(k).strip() for k in list(csv_df.index)[:10]]}")
        suspicious_cols = {"atomic_radius", "en_pauling", "atomic_mass"}
        if suspicious_cols.intersection(set(map(str, csv_df.columns))):
            raise ValueError(
                "你上传的CSV看起来像元素属性表（atomic_radius/en_pauling等），不是目标值CSV。"
                "目标CSV应：第一列=id，第二列=目标值，第三列起=附加特征(可选)。"
            )

        targets_map = csv_df.iloc[:, 0].to_dict()

        # --- Debug: 前几个结构解析+索引元素检查 ---
        for s in raw_structs[:3]:
            fn = os.path.basename(str(s.get("filename", ""))).strip()
            key_candidates = [fn, os.path.splitext(fn)[0], fn.split(".")[0]]
            matched = next((k for k in key_candidates if k in targets_map), None)
            atoms = parse_atoms(s.get("content"), fn)

            idx_elems = []
            for atom_idx in indices:
                if 1 <= atom_idx <= len(atoms):
                    idx_elems.append(f"{atom_idx}:{atoms[atom_idx-1].symbol}")
                else:
                    idx_elems.append(f"{atom_idx}:OOB")
            logs.append(f"  [结构] {fn} keys={key_candidates} match={matched or '-'} idx元素={idx_elems}")

        dat, valid_cnt, errs = builder.build_train_dat(raw_structs, targets_map, indices, current_feat_list, parse_atoms)

        if valid_cnt == 0:
            struct_keys = []
            for s in raw_structs[:10]:
                fn = os.path.basename(str(s.get("filename", ""))).strip()
                struct_keys.append(os.path.splitext(fn)[0])
            csv_keys = [str(k).strip() for k in list(targets_map.keys())[:10]]
            raise ValueError(
                "Batch #{i} 无匹配结构。".format(i=i + 1)
                + f"\n结构样本key(去后缀): {struct_keys}"
                + f"\nCSV样本key: {csv_keys}"
                + "\n提示: 结构名已取 basename；仍不匹配请检查大小写/空格/后缀。"
            )

        df_base = pd.read_csv(io.StringIO(dat), sep=r"\s+", engine="python")

        valid_ids = df_base["materials"].tolist()
        extra = csv_df.iloc[:, 1:]
        if not extra.empty:
            matched_extra = extra.loc[extra.index.intersection(valid_ids)]
            matched_extra = matched_extra.reindex(valid_ids).reset_index(drop=True)
            if not matched_extra.empty:
                df_base = pd.concat([df_base, matched_extra], axis=1)

        df_base["materials"] = df_base["materials"].apply(lambda x: f"b{i+1}_{x}")
        all_dfs.append(df_base)

    if not all_dfs:
        _append_log("\n".join(logs))
        return log_store, "", ""

    final_df = pd.concat(all_dfs, ignore_index=True)
    if final_df.isnull().values.any():
        null_cols = [c for c in final_df.columns if final_df[c].isnull().any()]
        raise ValueError(f"生成的数据存在缺失值(NaN)，请先修正输入数据。受影响列: {null_cols}")

    if tool_mode == "trad_ml":
        logs.append(f"【Trad ML 生成】Model: {ml_model}")
        
        # 1. Rename materials -> filename
        if 'materials' in final_df.columns:
            final_df.rename(columns={'materials': 'filename'}, inplace=True)
        # 2. 将 Property (Target) 移到最后
        if 'Property' in final_df.columns:
            prop = final_df.pop('Property')
            final_df['target'] = prop
            
        script_content = get_ml_script(ml_model)
        _append_log("\n".join(logs))
        return log_store, script_content, final_df.to_csv(index=False)

    else:
        # SISSO Mode
        real_nsample = len(final_df)
        real_nsf = final_df.shape[1] - 2
        
        logs.append(f"【SISSO 生成成功】nsample={real_nsample}, nsf={real_nsf}")
        logs.append(f"【参数范围】Dim: {dim_min}-{dim_max}, Cplx: {cplx_min}-{cplx_max}")

        sisso_in_content = get_sisso_template(real_nsample, real_nsf)
        _append_log("\n".join(logs))
        return log_store, sisso_in_content, final_df.to_string(index=False)

@app.callback(
    Output("modal-file-editor", "is_open"),
    Input("btn-generate", "n_clicks"),
    Input("btn-close-modal", "n_clicks"),
    Input("btn-submit-modal", "n_clicks"),
    State("modal-file-editor", "is_open")
)
def toggle_modal(n_generate, n_close, n_submit, is_open):
    if ctx.triggered_id == "btn-generate":
        return True
    if ctx.triggered_id in ["btn-close-modal", "btn-submit-modal"]:
        return False
    return is_open

# --- [作业管理] 提交 -> 监控 -> 提取 -> 显示 ---
@app.callback(
    Output("store-job-info", "data"), Output("store-log-job", "data"), Output("result-display", "children"),
    Input("btn-submit-modal", "n_clicks"), Input("interval-job-monitor", "n_intervals"), Input("btn-pull-status", "n_clicks"),
    State("editor-sisso", "value"), State("editor-train", "value"), State("store-job-info", "data"), State("store-log-job", "data"),
    State("inp-dim-min", "value"), State("inp-dim-max", "value"), State("inp-cplx-min", "value"), State("inp-cplx-max", "value"),
    State("tool-selector", "value"),
    State("ml-model-selector", "value"),
    State("gnn-model-selector", "value"),
    State("store-batches-data", "data"),
    prevent_initial_call=True
)
def manage_job(n_submit, n_interval, n_pull, sisso_template, train, job_info, current_log, dim_min, dim_max, cplx_min, cplx_max, tool_mode, ml_model, gnn_model, batches_data):
    trigger = ctx.triggered_id

    if current_log is None:
        current_log = []

    def _append_job_log(block: str):
        current_log.append({"ts": time.time(), "src": "JOB", "msg": block})
    
    # 提交作业
    if trigger == "btn-submit-modal":
        try:
            remote_cfg = get_remote_server(CONFIG)
            if not remote_cfg.get("hostname") or not remote_cfg.get("username") or not remote_cfg.get("password"):
                _append_job_log("配置缺失：请在 queue/config/default_config.json 里配置 active_cluster 对应的 clusters.<name>.remote_server")
                return {}, current_log, no_update

            ssh = SSHManager(**remote_cfg)
            ok, msg = ssh.connect()
            if not ok:
                _append_job_log(f"连接失败: {msg}")
                return {}, current_log, no_update

            try:
                ssh.open_sftp(raise_on_error=True)
            except Exception as e:
                try:
                    ssh.close()
                except Exception:
                    pass
                _append_job_log(f"SFTP 打开失败: {e}")
                return {}, current_log, no_update
            
            # 创建主目录
            main_rd = f"Job_{int(time.time())}"
            ssh.mkdir_remote(main_rd)
            
            submitted_jobs = []
            logs = [f"创建任务目录: {main_rd}"]

            executor = get_executor(CONFIG)

            def _resolve_task(task_name: str):
                try:
                    task_cfg = get_module_task(CONFIG, "ml_prediction", task_name)
                    cmd = str(task_cfg.get("command") or "").strip()
                    env_lines = task_cfg.get("env_lines") or []
                    if isinstance(env_lines, str):
                        env_lines = [env_lines]
                    elif not isinstance(env_lines, list):
                        env_lines = list(env_lines)
                    if not cmd:
                        raise ValueError(f"modules.ml_prediction.tasks.{task_name}.command 不能为空")
                    return cmd, env_lines
                except Exception as e:
                    legacy = get_commands(CONFIG)
                    cmd = str(legacy.get(task_name) or "").strip()
                    if not cmd:
                        raise e
                    return cmd, []

            # --- Branch: TRADITIONAL ML ---
            if tool_mode == "trad_ml":
                # sisso_template 存了 model.py, train 存了 train_data.csv
                if not sisso_template or not str(sisso_template).strip():
                    _append_job_log("Trad ML: model.py 内容为空（请先生成/预览，再提交）")
                    ssh.close()
                    return {}, current_log, no_update

                ssh.write_remote_file(f"{main_rd}/model.py", sisso_template)
                ssh.write_remote_file(f"{main_rd}/train_data.csv", train)

                try:
                    ml_cmd, ml_env = _resolve_task("trad_ml")
                except Exception as e:
                    _append_job_log(f"配置缺失: modules.ml_prediction.tasks.trad_ml ({e})")
                    ssh.close()
                    return {}, current_log, no_update
                ok_sub, jid = ssh.submit_job(
                    executor_cfg=executor,
                    remote_dir=main_rd,
                    command=ml_cmd,
                    script_name="slurm.sh",
                    env_lines=ml_env,
                )
                if ok_sub:
                    submitted_jobs.append(jid)
                    logs.append(f"  [ML提交] JobID {jid}")
                else:
                    logs.append(f"  [ML失败] {jid}")

            # --- Branch: GNN ---
            elif tool_mode == "gnn":
                gnn_model = (gnn_model or "dimenet_pp").strip().lower()
                if gnn_model not in GNN_MODEL_OPTIONS:
                    _append_job_log(f"GNN: 不支持的模型: {gnn_model}")
                    ssh.close()
                    return {}, current_log, no_update

                if not sisso_template or not str(sisso_template).strip():
                    _append_job_log("GNN: run_gnn.py 内容为空（请先生成/预览，再提交）")
                    ssh.close()
                    return {}, current_log, no_update

                if not train or not str(train).strip():
                    _append_job_log("GNN: targets.csv 内容为空（请先生成/预览，再提交）")
                    ssh.close()
                    return {}, current_log, no_update

                # 创建结构目录
                struct_rd = f"{main_rd}/structures"
                ssh.mkdir_remote(struct_rd)

                # 上传 targets.csv
                ssh.write_remote_file(f"{main_rd}/targets.csv", train)

                # 上传 run_gnn.py
                ssh.write_remote_file(f"{main_rd}/run_gnn.py", sisso_template)

                # 上传 utils.py（convert_to_graph + set_seed）
                try:
                    utils_path = MODULE_ROOT / "gnn" / "utils.py"
                    with open(utils_path, "r", encoding="utf-8") as f:
                        ssh.write_remote_file(f"{main_rd}/utils.py", f.read())
                except Exception as e:
                    logs.append(f"  [警告] utils.py 读取失败: {e}")

                # 上传结构文件，并按照 b{i}_{stem}.ext 重命名
                batches_data = batches_data or []
                uploaded = 0
                for i, batch in enumerate(batches_data):
                    raw_structs = (batch or {}).get("structures") or []
                    for s in raw_structs:
                        fname = os.path.basename(str(s.get('filename', ''))).strip()
                        if not fname:
                            continue
                        stem = os.path.splitext(fname)[0]
                        ext = os.path.splitext(fname)[1] or ".cif"
                        gid = f"b{i+1}_{stem}"
                        remote_name = f"{gid}{ext}"
                        try:
                            payload_b64 = s.get('content')
                            if not payload_b64:
                                continue
                            text = base64.b64decode(payload_b64).decode('utf-8', errors='replace')
                            ssh.write_remote_file(f"{struct_rd}/{remote_name}", text)
                            uploaded += 1
                        except Exception:
                            continue

                if uploaded == 0:
                    _append_job_log("GNN: 未上传任何结构文件（请先在左侧拖入结构文件，并使用‘从结构生成’生成 targets.csv）")
                    ssh.close()
                    return {}, current_log, no_update
                logs.append(f"  [GNN结构] 已上传 {uploaded} 个结构到 {struct_rd}")

                try:
                    gnn_cmd, gnn_env = _resolve_task("gnn")
                except Exception as e:
                    _append_job_log(f"配置缺失: modules.ml_prediction.tasks.gnn ({e})")
                    ssh.close()
                    return {}, current_log, no_update
                ok_sub, jid = ssh.submit_job(
                    executor_cfg=executor,
                    remote_dir=main_rd,
                    command=gnn_cmd,
                    script_name="slurm.sh",
                    env_lines=gnn_env,
                )
                if ok_sub:
                    submitted_jobs.append(jid)
                    logs.append(f"  [GNN提交] JobID {jid}")
                else:
                    logs.append(f"  [GNN失败] {jid}")

            # --- Branch: SISSO ---
            else:
                d_min = int(dim_min) if dim_min else 1
                d_max = int(dim_max) if dim_max else d_min
                c_min = int(cplx_min) if cplx_min else 1
                c_max = int(cplx_max) if cplx_max else c_min

                # SISSO 只关心 command；统一通过 Slurm 队列提交
                
                # 遍历所有组合
                try:
                    sisso_cmd, sisso_env = _resolve_task("sisso")
                except Exception as e:
                    ssh.close()
                    _append_job_log(f"配置缺失: modules.ml_prediction.tasks.sisso ({e})")
                    return {}, current_log, no_update

                for d in range(d_min, d_max + 1):
                    for c in range(c_min, c_max + 1):
                        sub_dir_name = f"d{d}_c{c}"
                        full_remote_path = f"{main_rd}/{sub_dir_name}"
                        
                        ssh.mkdir_remote(full_remote_path)
                        
                        current_sisso = sisso_template.replace("{{dim}}", str(d)).replace("{{cplx}}", str(c))
                        
                        ssh.write_remote_file(f"{full_remote_path}/SISSO.in", current_sisso)
                        ssh.write_remote_file(f"{full_remote_path}/train.dat", train)

                        ok_sub, jid = ssh.submit_job(
                            executor_cfg=executor,
                            remote_dir=full_remote_path,
                            command=sisso_cmd,
                            script_name="slurm.sh",
                            env_lines=sisso_env,
                        )
                        if ok_sub:
                            submitted_jobs.append(jid)
                            logs.append(f"  [SISSO提交] {sub_dir_name} -> JobID {jid}")
                        else:
                            logs.append(f"  [失败] {sub_dir_name} -> {jid}")
            
            ssh.close()
            
            # 【关键修复】即使只有部分成功，也必须返回列表，否则全盘皆输
            if submitted_jobs:
                last_jid = submitted_jobs[-1]
                _append_job_log("\n".join(logs))
                # 记录所有 job_ids，而不仅仅是最后一个
                return {
                    "remote_dir": main_rd, 
                    "job_id": last_jid, 
                    "job_ids": submitted_jobs,  # <--- CRITICAL FIX for multi-task
                    "status": "submitted", 
                    "mode": tool_mode, 
                    "executor": executor
                }, current_log, no_update
            else:
                _append_job_log("\n".join(logs) + "\n全部提交失败")
                return {}, current_log, no_update
                
        except Exception as e:
            _append_job_log(f"异常: {e}\n{traceback.format_exc()}")
            # 【容错增强】如果中途崩溃但已有任务提交，抢救已有的 ID
            if submitted_jobs:
                 return {
                    "remote_dir": main_rd, 
                    "job_id": submitted_jobs[-1], 
                    "job_ids": submitted_jobs, 
                    "status": "submitted", 
                    "mode": tool_mode, 
                    "executor": executor
                }, current_log, no_update
            return {}, current_log, no_update

    # 监控与提取
    elif trigger in ["interval-job-monitor", "btn-pull-status"]:
        if not job_info or job_info.get("status") != "submitted":
            if trigger == "btn-pull-status":
                _append_job_log("没有正在进行的任务或任务信息已丢失（请重新提交）")
                return no_update, current_log, no_update
            raise PreventUpdate
        
        ssh = SSHManager(**get_remote_server(CONFIG))
        ok, msg = ssh.connect()
        if not ok:
            _append_job_log(f"连接中断: {msg}")
            return no_update, current_log, no_update

        try:
            ssh.open_sftp(raise_on_error=True)
        except Exception as e:
            ssh.close()
            _append_job_log(f"SFTP 打开失败: {e}")
            return no_update, current_log, no_update
        
        # --- 多任务状态检查 ---
        all_jids = job_info.get("job_ids", [])
        if not all_jids and job_info.get("job_id"):
            all_jids = [job_info.get("job_id")]
            
        executor_cfg = (job_info.get("executor") or get_executor(CONFIG))
        
        any_running = False
        unknown_errors = []
        
        # 只要有一个 RUNNING，整体就是 RUNNING
        # 只要没有 RUNNING 且有 UNKNOWN，整体暂定为 UNKNOWN
        # 全部 COMPLETED 才算 COMPLETED (既非 Running 也非 Unknown)
        
        for jid in all_jids:
            s_code, s_det = ssh.check_job_status(job_id=str(jid), executor_cfg=executor_cfg)
            if s_code == "RUNNING":
                any_running = True
                break
            elif s_code == "UNKNOWN":
                unknown_errors.append(f"{jid}:{s_det}")
        
        if any_running:
            ssh.close()
            # 任务运行中，不更新 UI，仅维持轮询
            return no_update, current_log, no_update
            
        if unknown_errors:
            ssh.close()
            # 只有当没有任务在跑但有未知的，才报告异常 (防止网络波动导致误判完成)
            _append_job_log(f"状态查询异常(如有任务正在排队请忽略): {unknown_errors[0]}...")
            return no_update, current_log, no_update
            
        # 到此处说明所有任务都已结束 (squeue 查不到记录 -> 视为 COMPLETED)
        if True:  # block start logic for COMPLETED

            remote_dir = job_info['remote_dir']
            mode = job_info.get("mode", "sisso")
            
            try:
                display_children = []
                local_paths_cfg = get_local_paths(CONFIG)
                results_root = local_paths_cfg.get("results_root")
                if not results_root or not str(results_root).strip():
                    raise KeyError("配置缺失: clusters.<name>.local_paths.results_root")
                local_root = Path(str(results_root))
                out_dir = local_root / "discriptor"
                out_dir.mkdir(parents=True, exist_ok=True)

                if mode == "trad_ml":
                    # --- Traditional ML Extraction ---
                    csv_content = _download_text(ssh, f"{remote_dir}/results.csv")
                    metrics_content = _download_text(ssh, f"{remote_dir}/metrics.json")

                    if metrics_content:
                        try:
                            metrics = json.loads(metrics_content)
                            info_bits: list[str] = []
                            for k in [
                                "model",
                                "n_samples",
                                "n_features",
                                "cv_splits",
                                "oof_mae",
                                "oof_rmse",
                                "oof_r2",
                                "optuna_best_cv_mae",
                            ]:
                                if k in metrics and metrics[k] is not None:
                                    info_bits.append(f"{k}: {metrics[k]}")
                            if info_bits:
                                display_children.append(dbc.Alert(" | ".join(info_bits), color="info"))
                        except Exception:
                            display_children.append(dbc.Alert("metrics.json 解析失败（格式不正确）", color="warning"))

                    if csv_content:
                        df_res = pd.read_csv(io.StringIO(csv_content))
                        df_res.to_csv(out_dir / "results.csv", index=False)
                        csv_href = "data:text/csv;charset=utf-8," + base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
                        display_children.append(html.H5("📊 Traditional ML 结果"))
                        display_children.append(
                            html.A(
                                dbc.Button("📥 下载 results.csv", color="success", size="sm", className="mb-2"),
                                href=csv_href,
                                download="results.csv",
                                target="_blank",
                            )
                        )
                        display_children.append(dbc.Table.from_dataframe(df_res.head(30), striped=True, bordered=True, hover=True, size="sm"))
                    else:
                        display_children.append(dbc.Alert("未找到 results.csv（可能训练失败或未写出）", color="danger"))

                elif mode == "gnn":
                    # --- GNN Extraction ---
                    metrics_content = _download_text(ssh, f"{remote_dir}/gnn_metrics.json")
                    parity_svg = _download_text(ssh, f"{remote_dir}/train_test_parity.svg")
                    test_pred_content = _download_text(ssh, f"{remote_dir}/gnn_test_predictions.csv")
                    legacy_pred_content = _download_text(ssh, f"{remote_dir}/gnn_predictions.csv")
                    train_pred_content = _download_text(ssh, f"{remote_dir}/gnn_train_predictions.csv")

                    if metrics_content:
                        try:
                            metrics = json.loads(metrics_content)
                            info_bits: list[str] = []
                            optuna_best = metrics.get("optuna", {}).get("best_value")
                            if optuna_best is not None:
                                info_bits.append(f"optuna_best_val_mae: {float(optuna_best):.6f}")
                            train_mae = metrics.get("parity", {}).get("train_mae")
                            test_mae = metrics.get("parity", {}).get("test_mae")
                            if train_mae is not None:
                                info_bits.append(f"train_mae: {float(train_mae):.6f}")
                            if test_mae is not None:
                                info_bits.append(f"test_mae: {float(test_mae):.6f}")
                            if info_bits:
                                display_children.append(dbc.Alert(" | ".join(info_bits), color="info"))
                        except Exception:
                            display_children.append(dbc.Alert("gnn_metrics.json 解析失败（格式不正确）", color="warning"))
                    else:
                        display_children.append(dbc.Alert("未找到 gnn_metrics.json（可能训练失败或未写出）", color="danger"))

                    if parity_svg:
                        display_children.append(html.H6("📈 Train/Test Parity 图"))
                        display_children.append(
                            html.Iframe(
                                srcDoc=parity_svg,
                                style={"width": "100%", "height": "520px", "border": "1px solid #e5e7eb", "borderRadius": "8px"},
                            )
                        )
                        svg_href = "data:image/svg+xml;charset=utf-8," + base64.b64encode(parity_svg.encode("utf-8")).decode("utf-8")
                        display_children.append(
                            html.A(
                                dbc.Button("📥 下载 Parity SVG", color="secondary", size="sm", className="mb-2"),
                                href=svg_href,
                                download="train_test_parity.svg",
                                target="_blank",
                            )
                        )

                    def _add_csv_block(title: str, content: str, filename: str):
                        df_res = pd.read_csv(io.StringIO(content))
                        csv_href = "data:text/csv;charset=utf-8," + base64.b64encode(content.encode("utf-8")).decode("utf-8")
                        display_children.append(html.H6(title))
                        display_children.append(
                            html.A(
                                dbc.Button("📥 下载 CSV", color="success", size="sm", className="mb-2"),
                                href=csv_href,
                                download=filename,
                                target="_blank",
                            )
                        )
                        display_children.append(dbc.Table.from_dataframe(df_res.head(30), striped=True, bordered=True, hover=True, size="sm"))

                    if test_pred_content:
                        _add_csv_block("Test 预测 (gnn_test_predictions.csv)", test_pred_content, "gnn_test_predictions.csv")
                    elif legacy_pred_content:
                        _add_csv_block("Test 预测 (兼容输出 gnn_predictions.csv)", legacy_pred_content, "gnn_predictions.csv")
                    else:
                        display_children.append(dbc.Alert("未找到 test 预测 CSV（gnn_test_predictions.csv / gnn_predictions.csv），请检查远程日志。", color="danger"))

                    if train_pred_content:
                        _add_csv_block("Train 预测 (gnn_train_predictions.csv)", train_pred_content, "gnn_train_predictions.csv")

                else:
                    # --- SISSO Extraction (Existing Logic) ---
                    # 1. SISSO_extract (batch mode: generate */models.csv and all_models_rmse_complexity.csv)
                    extract_script_path = str(MODULE_ROOT / "sisso" / "SISSO_extract.py")

                    if not os.path.exists(extract_script_path):
                        raise FileNotFoundError(f"本地缺少后处理脚本: {extract_script_path}")

                    with open(extract_script_path, "r", encoding="utf-8") as f:
                        extract_content = f.read()
                    ssh.write_remote_file(f"{remote_dir}/SISSO_extract.py", extract_content)
                    code_ex, out_ex, err_ex = ssh.exec_command(f"cd ~/{remote_dir} && python SISSO_extract.py")
                    if code_ex != 0:
                        raise RuntimeError(f"SISSO_extract 执行失败: {err_ex or out_ex}")

                    # 2. draw.py
                    draw_script_path = str(MODULE_ROOT / "sisso" / "draw.py")

                    if not os.path.exists(draw_script_path):
                        raise FileNotFoundError(f"本地缺少后处理脚本: {draw_script_path}")

                    with open(draw_script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()
                    ssh.write_remote_file(f"{remote_dir}/draw.py", script_content)
                    code_draw, out_draw, err_draw = ssh.exec_command(f"cd ~/{remote_dir} && python draw.py")
                    if code_draw != 0:
                        raise RuntimeError(f"draw.py 执行失败: {err_draw or out_draw}")

                    csv_content = _download_text(ssh, f"{remote_dir}/all_models_rmse_complexity.csv")

                    ok_ls, files = ssh.list_remote_files(remote_dir)
                    if not ok_ls:
                        raise RuntimeError("列出远端输出文件失败")
                    img_filename = next((f for f in files if f.endswith(".png")), None)
                    if not img_filename:
                        raise FileNotFoundError("远端未生成任何 .png 图片输出")

                    if not ssh.sftp:
                        raise RuntimeError("SFTP 未连接")
                    with ssh.sftp.file(f"{remote_dir}/{img_filename}", "rb") as f:
                        img_bytes = f.read()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                    df_res = pd.read_csv(io.StringIO(csv_content))
                    df_res.to_csv(out_dir / "all_models_rmse_complexity.csv", index=False)
                    csv_href = "data:text/csv;charset=utf-8," + base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
                    display_children.append(html.H5("📊 描述符统计结果"))
                    display_children.append(
                        html.A(
                            dbc.Button("📥 下载 CSV 数据", color="success", size="sm", className="mb-2"),
                            href=csv_href,
                            download="all_models_rmse_complexity.csv",
                            target="_blank",
                        )
                    )
                    display_children.append(
                        dbc.Table.from_dataframe(
                            df_res,
                            striped=True,
                            bordered=True,
                            hover=True,
                            size="sm",
                            style={"maxHeight": "300px", "overflowY": "scroll"},
                        )
                    )
                    display_children.append(html.Hr())

                    img_src = f"data:image/png;base64,{img_base64}"
                    display_children.append(html.H5("📈 帕累托前沿图"))
                    display_children.append(
                        html.A(
                            dbc.Button("📥 下载图片", color="info", size="sm", className="mb-2"),
                            href=img_src,
                            download=img_filename,
                            target="_blank",
                        )
                    )
                    display_children.append(
                        html.Img(src=img_src, style={"maxWidth": "100%", "border": "1px solid #ddd", "padding": "5px"})
                    )

                ssh.close()
                job_info["status"] = "finished"

                if not display_children:
                    _append_job_log("作业完成，但未生成有效结果文件")
                    return job_info, current_log, no_update
                    
                _append_job_log("作业完成，结果已保存。")
                return job_info, current_log, display_children
                
            except Exception as e:
                ssh.close()
                _append_job_log(f"提取过程出错: {e}\n{traceback.format_exc()}")
                return job_info, current_log, no_update
        
        ssh.close()
    return no_update, no_update, no_update

if __name__ == "__main__":
    # Dash 入口，固定端口 8050，避免与其他模块冲突
    app.run(debug=True, port=8050)
