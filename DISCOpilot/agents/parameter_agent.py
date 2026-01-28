from typing import Dict

from pathlib import Path

from toolboxes.params.aims import merge_aims_params


class ParameterAgent:
    def apply(self, plan: Dict) -> Dict:
        engine = (plan.get("engine") or "aims").lower().strip()
        plan["engine"] = "aims" if engine not in {"aims", "mlp"} else engine

        structure_type = str(plan.get("structure_type") or "").lower().strip()
        if structure_type != "bulk":
            plan.setdefault("slab", {})
            plan["slab"].setdefault("min_size", 10.0)
            plan["slab"].setdefault("vacuum", 15.0)

            plan.setdefault("fixation", {})
            plan["fixation"].setdefault("z_frac", 0.5)

        if plan["engine"] == "aims":
            plan["aims_params"] = merge_aims_params(plan.get("aims_params"))
        else:
            plan.setdefault("mlp", {})
            default_model = Path(__file__).resolve().parents[1] / "DPA-3.1-3M.pt"
            plan["mlp"].setdefault(
                "model_path",
                plan.get("mlp_model_path") or (str(default_model) if default_model.exists() else ""),
            )
            plan["mlp"].setdefault("fmax", 0.05)
            plan["mlp"].setdefault("steps", 50)

        plan.setdefault("execution", {})
        plan["execution"].setdefault("submit", False)
        plan["execution"].setdefault("prepare_only", True)

        return plan
