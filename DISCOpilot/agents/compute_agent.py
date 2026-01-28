from typing import Dict, List, Tuple

from toolboxes.compute.aims import run_aims_calculation
from toolboxes.compute.mlp import run_mlp_relaxation


class ComputeAgent:
    def run(self, plan: Dict, models: List[Dict], config: Dict) -> Tuple[List[Dict], List[str]]:
        logs: List[str] = []
        engine = plan.get("engine", "aims")
        results: List[Dict] = []

        if engine == "aims":
            aims_params = plan.get("aims_params") or {}
            submit = bool((plan.get("execution") or {}).get("submit"))
            for item in models:
                site = item.get("site", "site")
                struct = item.get("structure")
                tag = f"{plan.get('surface', 'surface')}_{plan.get('adsorbate', 'ads')}_{site}"
                res = run_aims_calculation(
                    structure_dict=struct,
                    structure_name=tag,
                    config_dict=config,
                    aims_params=aims_params,
                    submit=submit,
                )
                results.append({"site": site, "status": res.get("status"), "detail": res})
                logs.append(f"AIMS {site}: {res.get('status')}")
            return results, logs

        if engine == "mlp":
            mlp_cfg = plan.get("mlp") or {}
            model_path = mlp_cfg.get("model_path") or ""
            for item in models:
                site = item.get("site", "site")
                struct = item.get("structure")
                res = run_mlp_relaxation(
                    struct,
                    model_path=model_path,
                    fmax=float(mlp_cfg.get("fmax", 0.05)),
                    steps=int(mlp_cfg.get("steps", 50)),
                )
                results.append({"site": site, "status": res.get("status"), "detail": res})
                logs.append(f"MLP {site}: {res.get('status')}")
            return results, logs

        return [], ["Unknown engine"]
