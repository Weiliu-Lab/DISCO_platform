import datetime
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.config.manager import ConfigManager

from agents.compute_agent import ComputeAgent
from agents.modeler import ModelingAgent
from agents.parameter_agent import ParameterAgent
from agents.planner import PlannerAgent
from llm.client import DeepSeekClient
from toolboxes.compute.aims import prepare_aims_inputs


DEFAULT_MP_KEY = "31HfDNN66lqSNhq4YH6zCxTQ2Re9t6cD"


def _safe_slug(value: str, fallback: str = "model") -> str:
    slug = re.sub(r"[^\w\-]+", "_", str(value), flags=re.ASCII).strip("_")
    return slug or fallback


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
        path = ROOT / path
    return path


DISCOPILOT_URL_PREFIX = _normalize_prefix(os.getenv("DISCOPILOT_URL_PREFIX", "/"))


class Orchestrator:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config

        mp_key = (
            os.environ.get("MP_API_KEY")
            or os.environ.get("MATERIALS_PROJECT_API_KEY")
            or DEFAULT_MP_KEY
        )
        llm_cfg = self.config.get("llm", {}) or {}
        self.llm_client = None
        try:
            api_key = llm_cfg.get("api_key") or ""
            if api_key:
                self.llm_client = DeepSeekClient(
                    api_key=api_key,
                    api_base=llm_cfg.get("api_base") or "https://api.deepseek.com",
                    model=llm_cfg.get("model") or "deepseek-chat",
                    temperature=llm_cfg.get("temperature", 0.1),
                )
        except Exception:
            self.llm_client = None

        self.planner = PlannerAgent(self.llm_client)
        self.parameter_agent = ParameterAgent()
        self.modeler = ModelingAgent(mp_key, self.llm_client)
        self.compute_agent = ComputeAgent()

    @staticmethod
    def _emit(
        logs: List[str],
        message: str,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        logs.append(message)
        print(message, flush=True)
        if logger:
            logger(message)

    def _prepare_inputs(
        self,
        plan: Dict,
        models: List[Dict],
        logs: List[str],
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Dict]:
        output_root = _resolve_results_dir(self.config)
        run_dir = (
            output_root
            / "disco_inputs"
            / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        self._emit(logs, f"[Prepare] output_dir={run_dir}", logger)

        engine = (plan.get("engine") or "aims").lower().strip()
        aims_params = plan.get("aims_params") or {}
        results: List[Dict] = []

        for idx, item in enumerate(models):
            site = item.get("site") or f"site_{idx + 1}"
            struct = item.get("structure")
            meta = item.get("meta") or {}
            tag = f"{plan.get('surface', 'surface')}_{plan.get('adsorbate', 'ads')}_{site}"
            safe_tag = _safe_slug(tag, f"site_{idx + 1}")
            model_dir = run_dir / safe_tag
            model_dir.mkdir(parents=True, exist_ok=True)

            if not struct:
                error = "Missing structure for input preparation."
                results.append(
                    {
                        "site": site,
                        "status": "error",
                        "error": error,
                        "dir": str(model_dir),
                    }
                )
                self._emit(logs, f"[Prepare] {site}: error: {error}", logger)
                continue

            output_files: List[str] = []
            if engine == "aims":
                ok, files = prepare_aims_inputs(
                    struct,
                    safe_tag,
                    aims_params=aims_params,
                )
                if not ok:
                    error = "prepare failed"
                    if isinstance(files, dict):
                        error = files.get("error") or error
                    results.append(
                        {
                            "site": site,
                            "status": "error",
                            "error": error,
                            "dir": str(model_dir),
                        }
                    )
                    self._emit(logs, f"[Prepare] {site}: error: {error}", logger)
                    continue

                for fname, content in files.items():
                    if fname == "params":
                        continue
                    out_name = "slurm.sh" if fname == "slurm" else fname
                    (model_dir / out_name).write_text(content, encoding="utf-8")
                    output_files.append(out_name)
            else:
                self._emit(
                    logs,
                    f"[Prepare] {site}: engine={engine}, inputs not generated",
                    logger,
                )

            (model_dir / "structure.json").write_text(
                json.dumps(struct, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            output_files.append("structure.json")
            (model_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "site": site,
                        "tag": tag,
                        "engine": engine,
                        "meta": meta,
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            output_files.append("meta.json")

            zip_name = "inputs.zip"
            zip_path = model_dir / zip_name
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                for fname in output_files:
                    zipf.write(model_dir / fname, arcname=fname)
            output_files.append(zip_name)

            download_url = None
            try:
                rel_path = zip_path.relative_to(output_root)
                download_url = f"{DISCOPILOT_URL_PREFIX}download/{rel_path.as_posix()}"
            except ValueError:
                download_url = None

            results.append(
                {
                    "site": site,
                    "status": "prepared",
                    "dir": str(model_dir),
                    "files": output_files,
                    "download_url": download_url,
                }
            )
            self._emit(logs, f"[Prepare] {site}: saved inputs", logger)
            if download_url:
                self._emit(logs, f"[Prepare] {site}: download={download_url}", logger)

        return results

    def chat(self, history: List[Dict]) -> str:
        if not self.llm_client:
            raise RuntimeError("Missing LLM client for chat.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DISCO-Pilot, a friendly computational chemistry assistant. "
                    "If the user is not asking for a modeling or computation task, "
                    "respond conversationally and briefly. Do not create plans."
                ),
            }
        ]
        messages.extend(history)
        return self.llm_client.chat(messages)

    def build_plan(
        self,
        user_request: str,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Dict, List[str]]:
        logs: List[str] = []
        try:
            plan = self.planner.plan(user_request)
        except Exception as exc:
            msg = f"Planner error: {exc}"
            self._emit(logs, msg, logger)
            return {"error": msg}, logs

        plan["user_request"] = user_request
        objective = plan.get("objective") or "Modeling and computation workflow"
        self._emit(logs, f"[Planner] objective: {objective}", logger)
        steps = plan.get("steps") or []
        if steps:
            self._emit(logs, f"[Planner] steps: {', '.join(steps)}", logger)
        questions = plan.get("questions") or []
        if questions:
            self._emit(logs, f"[Planner] questions: {', '.join(questions)}", logger)
        return plan, logs

    def execute_plan(
        self,
        plan: Dict,
        logger: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[Dict], List[str]]:
        logs: List[str] = []
        user_request = plan.get("user_request") or ""
        try:
            self._emit(logs, "[Modeling] planning", logger)
            if user_request:
                self._emit(logs, f"[Modeling] input: {user_request}", logger)
            model_plan, plan_logs = self.modeler.plan(user_request)
        except Exception as exc:
            msg = f"Modeling plan error: {exc}"
            self._emit(logs, msg, logger)
            return [], logs

        for line in plan_logs:
            self._emit(logs, f"[Modeling] {line}", logger)

        summary = (
            f"type={model_plan.get('structure_type')}, formula={model_plan.get('formula')}, "
            f"surface={model_plan.get('surface')}, adsorbate={model_plan.get('adsorbate')}, "
            f"sites={model_plan.get('sites')}"
        )
        self._emit(logs, f"[Modeling] plan: {summary}", logger)
        self._emit(logs, f"[Modeling] slab={model_plan.get('slab')}", logger)
        self._emit(logs, f"[Modeling] fixation={model_plan.get('fixation')}", logger)

        model_plan = self.parameter_agent.apply(model_plan)
        self._emit(logs, f"[Parameter] engine={model_plan.get('engine')}", logger)
        if model_plan.get("engine") == "aims":
            self._emit(logs, f"[Parameter] aims_params={model_plan.get('aims_params')}", logger)
        else:
            self._emit(logs, f"[Parameter] mlp={model_plan.get('mlp')}", logger)

        models, model_logs = self.modeler.build(model_plan)
        for line in model_logs:
            self._emit(logs, f"[Modeling] {line}", logger)
        if not models:
            return [], logs + ["No models generated. Execution stopped."]

        self._emit(logs, f"[Modeling] outputs: {len(models)} structures", logger)

        submit = bool((model_plan.get("execution") or {}).get("submit"))
        prepare_only = bool((model_plan.get("execution") or {}).get("prepare_only"))
        if submit:
            prepare_only = False

        if prepare_only:
            self._emit(logs, "[Prepare] generating input files", logger)
            results = self._prepare_inputs(model_plan, models, logs, logger)
            self._emit(logs, "[Prepare] finished", logger)
            return results, logs

        self._emit(logs, f"[Compute] engine={model_plan.get('engine')}", logger)
        self._emit(logs, f"[Compute] inputs: {len(models)} structures", logger)
        self._emit(logs, f"[Compute] submit={submit}", logger)
        results, compute_logs = self.compute_agent.run(model_plan, models, self.config)
        for line in compute_logs:
            self._emit(logs, f"[Compute] {line}", logger)
        self._emit(logs, "[Compute] finished", logger)
        return results, logs


def format_plan(plan: Dict) -> str:
    if not plan:
        return "- No plan available."
    if plan.get("error"):
        return f"- {plan.get('error')}"

    objective = str(plan.get("objective") or "Modeling and computation workflow").strip()
    lines = [f"- Objective: {objective}"]

    steps = plan.get("steps") or []
    if steps:
        lines.append("- Steps:")
        for step in steps:
            lines.append(f"  - {step}")

    questions = plan.get("questions") or []
    if questions:
        lines.append("- Questions:")
        for item in questions:
            lines.append(f"  - {item}")

    assumptions = plan.get("assumptions") or []
    if assumptions:
        lines.append("- Assumptions:")
        for item in assumptions:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def summarize_results(results: List[Dict]) -> str:
    if not results:
        return "No results produced."

    lines = ["Execution summary:"]
    for item in results:
        site = item.get("site", "site")
        status = item.get("status", "unknown")
        details: List[str] = []
        dir_path = item.get("dir")
        files = item.get("files") or []
        error = item.get("error")
        if dir_path:
            details.append(f"dir: {dir_path}")
        if files:
            details.append(f"files: {', '.join(files)}")
        if error:
            details.append(f"error: {error}")
        if details:
            lines.append(f"- {site}: {status} ({'; '.join(details)})")
        else:
            lines.append(f"- {site}: {status}")
    return "\n".join(lines)


def format_execution_logs(logs: List[str]) -> str:
    if not logs:
        return "Workflow log:\n- (empty)"
    lines = ["Workflow log:"]
    for line in logs:
        lines.append(f"- {line}")
    return "\n".join(lines)
