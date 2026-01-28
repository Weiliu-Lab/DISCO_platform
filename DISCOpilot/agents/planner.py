from __future__ import annotations

from typing import Dict

from llm.client import DeepSeekClient
from llm.parse import coerce_list, extract_json


DEFAULT_STEPS = [
    "Modeling agent interprets the request and builds candidate structures.",
    "Parameter agent sets calculation settings for the chosen engine.",
    "Compute agent runs the calculation and collects outputs.",
    "Report results and summarize key findings.",
]


class PlannerAgent:
    def __init__(self, llm_client: DeepSeekClient | None):
        self.llm_client = llm_client

    def plan(self, user_request: str) -> Dict:
        if not self.llm_client:
            raise RuntimeError("Missing LLM client for planner.")

        content = self.llm_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a planner agent for a multi-agent modeling system. "
                        "Provide a high-level plan only. Do not include detailed "
                        "modeling parameters or tool calls. Return JSON only. "
                        "Use the same language as the user request."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'The user request is: "{user_request}".\n\n'
                        "Return JSON with keys:\n"
                        "- objective: brief statement\n"
                        "- steps: list of 3-5 short steps\n"
                        "- assumptions: list (optional)\n"
                        "- questions: list only if critical details are missing\n"
                        "Reply in the same language as the user request.\n"
                    ),
                },
            ]
        )
        data = extract_json(content)

        objective = str(data.get("objective") or "Modeling and computation workflow").strip()
        steps = coerce_list(data.get("steps")) or DEFAULT_STEPS
        assumptions = coerce_list(data.get("assumptions"))
        questions = coerce_list(data.get("questions"))

        return {
            "objective": objective,
            "steps": steps,
            "assumptions": assumptions,
            "questions": questions,
        }
