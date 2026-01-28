"""Dash chat UI for DISCO-Pilot multi-agent orchestration."""

from __future__ import annotations

import base64
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, ctx, no_update
from flask import abort, send_file

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow.orchestrator import (
    Orchestrator,
    format_plan,
    summarize_results,
)


def load_avatar(file_name: str) -> Optional[str]:
    path = BASE_DIR / file_name
    if not path.exists():
        return None

    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None
    return f"data:{mime};base64,{encoded}"


AVATARS: Dict[str, Optional[str]] = {
    "assistant": load_avatar("agent.png"),
    "user": load_avatar("user.png"),
}

DEFAULT_MESSAGES: List[Dict[str, str]] = [
    {
        "role": "assistant",
        "content": (
            "Hello! I am DISCO-Pilot, your multi-agent research assistant. "
            "Tell me the modeling or computation task you want to study.\n\n"
            "Example: Build an MgO(111) slab with O adsorption."
        ),
    }
]

ORCHESTRATOR = Orchestrator()
EVENT_LOCK = threading.Lock()
EVENT_QUEUE: List[Dict] = []
TASK_STATE = {"status": "idle"}


def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "/").strip()
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not prefix.endswith("/"):
        prefix += "/"
    if prefix == "//":
        return "/"
    return prefix


def _resolve_results_dir(config: Dict) -> Path:
    local_paths = config.get("local_paths") or {}
    results_dir = (
        local_paths.get("results_dir")
        or local_paths.get("results_root")
        or "./results"
    )
    path = Path(results_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


OUTPUT_ROOT = _resolve_results_dir(ORCHESTRATOR.config)
DISCOPILOT_URL_PREFIX = _normalize_prefix(os.getenv("DISCOPILOT_URL_PREFIX", "/"))
DOWNLOAD_ROUTE = f"{DISCOPILOT_URL_PREFIX}download/<path:subpath>"

MODEL_KEYWORDS = {
    "model",
    "modeling",
    "compute",
    "calculation",
    "dft",
    "vasp",
    "aims",
    "fhi",
    "slab",
    "surface",
    "adsorption",
    "adsorbate",
    "bulk",
    "structure",
    "supercell",
    "k-point",
    "kpoint",
    "relax",
    "optimization",
    "energy",
    "band",
    "dos",
    "\u5efa\u6a21",
    "\u8ba1\u7b97",
    "\u7ed3\u6784",
    "\u6676\u4f53",
    "\u8868\u9762",
    "\u5438\u9644",
    "\u6676\u9762",
    "\u5f1b\u8c6b",
    "\u4f18\u5316",
    "\u80fd\u91cf",
    "\u80fd\u5e26",
    "\u6001\u5bc6\u5ea6",
    "\u5438\u9644\u80fd",
}

APPROVE_TOKENS = {
    "approve",
    "approved",
    "ok",
    "okay",
    "yes",
    "start",
    "run",
    "execute",
    "proceed",
    "go ahead",
    "sounds good",
    "looks good",
    "sure",
    "\u540c\u610f",
    "\u786e\u8ba4",
    "\u6267\u884c",
    "\u5f00\u59cb",
    "\u53ef\u4ee5",
    "\u6ca1\u95ee\u9898",
    "\u7ee7\u7eed",
    "\u7ee7\u7eed\u5427",
    "\u597d\u7684",
    "\u597d\u5427",
    "\u884c\u5427",
}

REJECT_TOKENS = {
    "reject",
    "rejected",
    "no",
    "cancel",
    "stop",
    "skip",
    "dont",
    "do not",
    "abort",
    "no thanks",
    "\u4e0d\u540c\u610f",
    "\u62d2\u7edd",
    "\u53d6\u6d88",
    "\u4e0d\u8981",
    "\u5148\u4e0d",
    "\u6682\u505c",
    "\u505c\u6b62",
    "\u4e0d\u53ef\u4ee5",
    "\u4e0d\u884c",
    "\u7b97\u4e86",
    "\u5148\u522b",
    "\u522b",
    "\u4e0d\u7528",
}

APPROVE_EXACT_TOKENS = {
    "approve",
    "approved",
    "ok",
    "okay",
    "yes",
    "y",
    "sure",
    "\u540c\u610f",
    "\u786e\u8ba4",
    "\u6267\u884c",
    "\u5f00\u59cb",
    "\u53ef\u4ee5",
    "\u53ef\u4ee5\u4e86",
    "\u53ef\u4ee5\u7684",
    "\u6ca1\u95ee\u9898",
    "\u7ee7\u7eed",
    "\u7ee7\u7eed\u5427",
    "\u884c",
    "\u884c\u5427",
    "\u884c\u7684",
    "\u597d",
    "\u597d\u7684",
    "\u597d\u5427",
    "\u6069",
    "\u6069\u6069",
    "\u5bf9",
    "\u662f",
}

REJECT_EXACT_TOKENS = {
    "reject",
    "rejected",
    "no",
    "n",
    "dont",
    "abort",
    "cancel",
    "stop",
    "skip",
    "no thanks",
    "\u4e0d\u540c\u610f",
    "\u62d2\u7edd",
    "\u53d6\u6d88",
    "\u4e0d\u8981",
    "\u5148\u4e0d",
    "\u6682\u505c",
    "\u505c\u6b62",
    "\u4e0d\u53ef\u4ee5",
    "\u4e0d\u884c",
    "\u7b97\u4e86",
    "\u5148\u522b",
    "\u522b",
    "\u4e0d\u7528",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _decision(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    if compact in REJECT_EXACT_TOKENS:
        return "reject"
    if compact in APPROVE_EXACT_TOKENS:
        return "approve"
    for token in REJECT_TOKENS:
        if token in normalized:
            return "reject"
    for token in APPROVE_TOKENS:
        if token in normalized:
            return "approve"
    return None


def _is_modeling_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if any(token in normalized for token in MODEL_KEYWORDS):
        return True
    if re.search(r"\b[A-Z][a-z]?\w*\(\d{3}\)\b", text or ""):
        return True
    if re.search(r"\bmp-\d+\b", normalized):
        return True
    return False


def _plan_needs_details(plan: Optional[Dict]) -> bool:
    if not plan:
        return False
    questions = plan.get("questions") or []
    return bool(questions)


def _missing_details_lines(plan: Optional[Dict]) -> List[str]:
    if not plan:
        return []
    questions = plan.get("questions") or []
    if not questions:
        return []
    lines = ["Missing details:"]
    for item in questions:
        lines.append(f"- {item}")
    return lines


def _recent_history(messages: List[Dict[str, str]], limit: int = 8) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = msg.get("content") or ""
        cleaned.append({"role": role, "content": content})
    return cleaned[-limit:]


def _push_event(event: Dict) -> None:
    with EVENT_LOCK:
        EVENT_QUEUE.append(event)


def _pop_events(limit: int = 4) -> List[Dict]:
    with EVENT_LOCK:
        if not EVENT_QUEUE:
            return []
        events = EVENT_QUEUE[:limit]
        del EVENT_QUEUE[:limit]
        return events


def _task_running() -> bool:
    with EVENT_LOCK:
        return TASK_STATE["status"] in {"planning", "executing", "chatting"} or bool(EVENT_QUEUE)


def _set_task_status(status: str) -> None:
    with EVENT_LOCK:
        TASK_STATE["status"] = status


def _start_planning(user_text: str) -> bool:
    with EVENT_LOCK:
        if TASK_STATE["status"] in {"planning", "executing"}:
            return False
        TASK_STATE["status"] = "planning"
    thread = threading.Thread(target=_run_planning, args=(user_text,), daemon=True)
    thread.start()
    return True


def _run_planning(user_text: str) -> None:
    def _logger(message: str) -> None:
        _push_event({"kind": "log", "content": message})

    plan, _logs = ORCHESTRATOR.build_plan(user_text, logger=_logger)
    if not plan or plan.get("error"):
        error_text = plan.get("error") if isinstance(plan, dict) else "Planner failed."
        _push_event({"kind": "error", "scope": "plan", "content": error_text})
        _set_task_status("idle")
        return

    plan_text = format_plan(plan)
    _push_event({"kind": "plan_start", "plan": plan})
    _push_event({"kind": "plan_line", "line": ""})
    for line in plan_text.splitlines():
        _push_event({"kind": "plan_line", "line": line})
    _push_event({"kind": "plan_end"})
    _set_task_status("idle")


def _start_execution(plan: Dict) -> bool:
    with EVENT_LOCK:
        if TASK_STATE["status"] in {"planning", "executing"}:
            return False
        TASK_STATE["status"] = "executing"
    thread = threading.Thread(target=_run_execution, args=(plan,), daemon=True)
    thread.start()
    return True


def _run_execution(plan: Dict) -> None:
    def _logger(message: str) -> None:
        _push_event({"kind": "log", "content": message})

    try:
        results, _logs = ORCHESTRATOR.execute_plan(plan, logger=_logger)
    except Exception as exc:
        _push_event({"kind": "error", "scope": "execute", "content": str(exc)})
        _set_task_status("idle")
        return

    _push_event({"kind": "exec_done", "results": results})
    _set_task_status("idle")


def _start_chat(history: List[Dict[str, str]]) -> bool:
    with EVENT_LOCK:
        if TASK_STATE["status"] in {"planning", "executing", "chatting"}:
            return False
        TASK_STATE["status"] = "chatting"
    thread = threading.Thread(target=_run_chat, args=(history,), daemon=True)
    thread.start()
    return True


def _run_chat(history: List[Dict[str, str]]) -> None:
    try:
        assistant_text = ORCHESTRATOR.chat(history)
    except Exception:
        assistant_text = "Hi! How can I help today?"
    _push_event({"kind": "chat_done", "content": assistant_text})
    _set_task_status("idle")


def _remove_thinking(messages: List[Dict[str, str]], scope: str) -> None:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("kind") == "thinking" and msg.get("scope") == scope:
            messages.pop(idx)
            break


def _append_log(messages: List[Dict[str, str]], line: str) -> None:
    if messages and messages[-1].get("kind") == "log":
        current = messages[-1].get("content") or ""
        messages[-1]["content"] = f"{current}\n{line}" if current else line
        return
    messages.append(
        {
            "role": "assistant",
            "kind": "log",
            "content": f"Workflow log:\n{line}",
        }
    )


def _append_plan_line(messages: List[Dict[str, str]], line: str) -> None:
    if messages and messages[-1].get("kind") == "plan":
        current = messages[-1].get("content") or ""
        messages[-1]["content"] = f"{current}\n{line}" if current else line
        return
    messages.append({"role": "assistant", "kind": "plan", "content": line})


def render_chat(messages: List[Dict[str, str]]) -> List[html.Div]:
    items: List[html.Div] = []
    for msg in messages or []:
        role = msg.get("role") or "assistant"
        is_user = role == "user"
        avatar_src = AVATARS.get(role)
        kind = msg.get("kind")
        bubble_class = "msg-bubble user" if is_user else "msg-bubble assistant"
        content_block = dcc.Markdown(msg.get("content", ""), className="msg-text")

        if not is_user and kind == "thinking":
            bubble_class = "msg-bubble assistant thinking"
            content_block = html.Div(
                [
                    html.Span(msg.get("content") or "Thinking", className="thinking-text"),
                    html.Span(
                        [
                            html.Span(".", className="dot"),
                            html.Span(".", className="dot"),
                            html.Span(".", className="dot"),
                        ],
                        className="thinking-dots",
                    ),
                ],
                className="thinking-row",
            )
        elif not is_user and kind == "log":
            bubble_class = "msg-bubble assistant log"
            content_block = html.Pre(msg.get("content", ""), className="log-text")
        elif not is_user and kind == "download":
            bubble_class = "msg-bubble assistant download"
            download_items = []
            for item in msg.get("items") or []:
                label = item.get("label") or "inputs.zip"
                url = item.get("url") or ""
                site = item.get("site")
                title = f"{site}: {label}" if site else label
                download_items.append(
                    html.Div(
                        [
                            html.Div(title, className="download-title"),
                            html.A(
                                "下载",
                                href=url,
                                target="_blank",
                                download=label,
                                className="download-link",
                            ),
                            html.Span(url, className="download-url"),
                        ],
                        className="download-item",
                    )
                )
            content_block = html.Div(download_items, className="download-list")
        elif not is_user and kind == "plan":
            content_block = dcc.Markdown(msg.get("content", ""), className="msg-text")
        row_class = f"chat-row {'user' if is_user else 'assistant'}"
        sender_label = "You" if is_user else "DISCO-Pilot"

        items.append(
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src=avatar_src,
                            alt=f"{role} avatar",
                            className="avatar",
                        )
                        if avatar_src
                        else html.Div(className="avatar avatar-fallback"),
                        className="avatar-wrap",
                    ),
                    html.Div(
                        [
                            html.Div(sender_label, className="msg-meta-name"),
                            content_block,
                        ],
                        className=bubble_class,
                    ),
                ],
                className=row_class,
            )
        )
    return items or [html.Div("No messages yet", className="text-muted")]


app: Dash = dash.Dash(
    __name__,
    assets_folder=str(BASE_DIR / "assets"),
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",
    ],
    requests_pathname_prefix=DISCOPILOT_URL_PREFIX,
    routes_pathname_prefix=DISCOPILOT_URL_PREFIX,
)
server = app.server


@server.route(DOWNLOAD_ROUTE)
def download_result_file(subpath: str):
    base = OUTPUT_ROOT.resolve()
    target = (base / subpath).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        abort(404)
    if not target.is_file():
        abort(404)
    try:
        return send_file(target, as_attachment=True, download_name=target.name)
    except TypeError:
        return send_file(target, as_attachment=True, attachment_filename=target.name)


app.layout = html.Div(
    [
        dcc.Store(id="store-messages", data=DEFAULT_MESSAGES, storage_type="session"),
        dcc.Store(id="store-plan", data=None),
        dcc.Store(id="store-status", data="idle"),
        dcc.Store(id="store-results", data=[]),
        dcc.Interval(id="execution-ticker", interval=700, n_intervals=0, disabled=False),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("DISCOpilot", className="logo"),
                        html.Div(
                            "自动化执行表界面建模计算和后处理分析任务",
                            className="subtitle",
                        ),
                    ],
                    className="topbar",
                ),
                html.Div(id="chat-body", className="chat-body"),
                html.Div(
                    [
                        dcc.Textarea(
                            id="prompt-input",
                            placeholder="Ask anything",
                            rows=1,
                            className="composer-input",
                        ),
                        html.Button(
                            html.I(className="bi bi-arrow-up"),
                            id="btn-send",
                            className="send-btn",
                            **{"aria-label": "Send"},
                        ),
                    ],
                    className="composer",
                ),
            ],
            className="page",
        ),
    ],
    className="app-root",
)


@app.callback(Output("chat-body", "children"), Input("store-messages", "data"))
def render_chat_body(messages):
    return render_chat(messages)


@app.callback(
    Output("store-messages", "data"),
    Output("store-plan", "data"),
    Output("store-status", "data"),
    Output("store-results", "data"),
    Output("prompt-input", "value"),
    Output("execution-ticker", "disabled"),
    Input("btn-send", "n_clicks"),
    Input("execution-ticker", "n_intervals"),
    State("prompt-input", "value"),
    State("store-messages", "data"),
    State("store-plan", "data"),
    State("store-status", "data"),
    State("store-results", "data"),
    prevent_initial_call=True,
)
def handle_chat_flow(
    n_send,
    n_intervals,
    prompt,
    messages,
    plan,
    status,
    results,
):
    trigger = ctx.triggered_id
    messages = list(messages or [])
    results = list(results or [])
    plan = plan or None

    if trigger == "execution-ticker":
        events = _pop_events(limit=4)
        if not events:
            if not _task_running():
                return messages, plan, status, results, no_update, True
            raise dash.exceptions.PreventUpdate

        for event in events:
            kind = event.get("kind")
            if kind == "log":
                _append_log(messages, event.get("content", ""))
                continue
            if kind == "plan_start":
                _remove_thinking(messages, "plan")
                plan = event.get("plan") or plan
                messages.append(
                    {
                        "role": "assistant",
                        "kind": "plan",
                        "content": "I have drafted a high-level plan based on your request:",
                    }
                )
                continue
            if kind == "plan_line":
                _append_plan_line(messages, event.get("line", ""))
                continue
            if kind == "plan_end":
                needs_details = _plan_needs_details(plan)
                _append_plan_line(messages, "")
                if needs_details:
                    for line in _missing_details_lines(plan):
                        _append_plan_line(messages, line)
                    _append_plan_line(messages, "Please reply with the missing details.")
                    status = "awaiting_details"
                else:
                    _append_plan_line(
                        messages,
                        "请问是否同意？",
                    )
                    status = "awaiting_approval"
                continue
            if kind == "chat_done":
                _remove_thinking(messages, "chat")
                messages.append(
                    {
                        "role": "assistant",
                        "content": event.get("content", ""),
                    }
                )
                status = "idle"
                continue
            if kind == "exec_done":
                _remove_thinking(messages, "execute")
                exec_results = event.get("results") or []
                summary = summarize_results(exec_results)
                messages.append({"role": "assistant", "content": summary})
                downloads = []
                for item in exec_results:
                    url = item.get("download_url")
                    if not url:
                        continue
                    downloads.append(
                        {
                            "site": item.get("site"),
                            "label": "inputs.zip",
                            "url": url,
                        }
                    )
                if downloads:
                    messages.append(
                        {
                            "role": "assistant",
                            "kind": "download",
                            "items": downloads,
                        }
                    )
                status = "completed"
                results = exec_results
                continue
            if kind == "error":
                _remove_thinking(messages, event.get("scope") or "")
                messages.append({"role": "assistant", "content": event.get("content", "")})
                status = "idle"
                plan = None
                continue

        return messages, plan, status, results, no_update, not _task_running()

    if trigger == "btn-send":
        if not prompt or not prompt.strip():
            raise dash.exceptions.PreventUpdate
        user_text = prompt.strip()
        messages.append({"role": "user", "content": user_text})

        if status in {"planning", "executing", "chatting"}:
            if status == "planning":
                note = "Planning is in progress."
            elif status == "executing":
                note = "Execution is in progress."
            else:
                note = "Response is in progress."
            messages.append(
                {
                    "role": "assistant",
                    "content": f"{note} I'll update once it's finished.",
                }
            )
            return messages, plan, status, results, "", no_update

        if status == "awaiting_approval":
            decision = _decision(user_text)
            if decision == "approve":
                if not plan:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "No plan available. Please send a request first.",
                        }
                    )
                    status = "idle"
                    return messages, plan, status, [], "", no_update

                started = _start_execution(plan)
                if not started:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "Execution is already running. Please wait.",
                        }
                    )
                    return messages, plan, status, results, "", no_update

                messages.append(
                    {
                        "role": "assistant",
                        "content": "Executing the plan now... I'll post updates here.",
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "kind": "thinking",
                        "scope": "execute",
                        "content": "Running",
                    }
                )
                status = "executing"
                return messages, plan, status, results, "", False

            if decision == "reject":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Plan rejected. Please clarify or provide a new request.",
                    }
                )
                status = "idle"
                return messages, None, status, [], "", no_update

            base_request = ""
            if isinstance(plan, dict):
                base_request = str(plan.get("user_request") or "")
            merged_request = f"{base_request}\n\n{user_text}".strip()
            if not base_request and not _is_modeling_request(merged_request):
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Please describe a computational chemistry modeling task "
                            "(structure, surface, adsorption, engine, etc.)."
                        ),
                    }
                )
                status = "idle"
                return messages, None, status, [], "", no_update

            started = _start_planning(merged_request)
            if not started:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Planning is already running. Please wait.",
                    }
                )
                return messages, plan, status, results, "", no_update

            messages.append(
                {
                    "role": "assistant",
                    "kind": "thinking",
                    "scope": "plan",
                    "content": "Planning",
                }
            )
            status = "planning"
            return messages, plan, status, results, "", False

        if status == "awaiting_details":
            decision = _decision(user_text)
            if decision == "reject":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Okay. Send a new request if you want to continue.",
                    }
                )
                status = "idle"
                return messages, None, status, [], "", no_update
            if decision == "approve":
                lines = _missing_details_lines(plan)
                if lines:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "\n".join(lines),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "Please answer the questions so I can proceed.",
                        }
                    )
                return messages, plan, status, results, "", no_update

            base_request = ""
            if isinstance(plan, dict):
                base_request = str(plan.get("user_request") or "")
            merged_request = f"{base_request}\n\n{user_text}".strip()
            if not base_request and not _is_modeling_request(merged_request):
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Please describe a computational chemistry modeling task "
                            "(structure, surface, adsorption, engine, etc.)."
                        ),
                    }
                )
                status = "idle"
                return messages, None, status, [], "", no_update

            started = _start_planning(merged_request)
            if not started:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Planning is already running. Please wait.",
                    }
                )
                return messages, plan, status, results, "", no_update

            messages.append(
                {
                    "role": "assistant",
                    "kind": "thinking",
                    "scope": "plan",
                    "content": "Planning",
                }
            )
            status = "planning"
            return messages, plan, status, results, "", False

        if not _is_modeling_request(user_text):
            history = _recent_history(messages)
            started = _start_chat(history)
            if not started:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Response is already running. Please wait.",
                    }
                )
                return messages, None, status, results, "", no_update

            messages.append(
                {
                    "role": "assistant",
                    "kind": "thinking",
                    "scope": "chat",
                    "content": "Thinking",
                }
            )
            status = "chatting"
            return messages, None, status, results, "", False

        started = _start_planning(user_text)
        if not started:
            messages.append(
                {
                    "role": "assistant",
                    "content": "Planning is already running. Please wait.",
                }
            )
            return messages, plan, status, results, "", no_update

        messages.append(
            {
                "role": "assistant",
                "kind": "thinking",
                "scope": "plan",
                "content": "Planning",
            }
        )
        status = "planning"
        return messages, plan, status, results, "", False

    raise dash.exceptions.PreventUpdate


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8054"))
    hot_reload = os.getenv("DASH_HOT_RELOAD", "0") == "1"
    use_reloader = os.getenv("DASH_USE_RELOADER", "0") == "1"
    app.run_server(
        debug=True,
        port=port,
        dev_tools_hot_reload=hot_reload,
        use_reloader=use_reloader,
    )
